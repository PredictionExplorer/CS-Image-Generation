"""Material, camera, and native hardware contracts for the painting renderer."""

import os
import unittest

import numpy as np

from tools.estuary.optics import Material, reflectance

from .surface import DEFAULTS, Surface, camera_basis, validate_config, validate_fields


def fields(width=128, height=96):
    """Uniform paint on a dimensioned plane; no synthetic scene assets."""
    return {
        "pigment": np.full((height, width, 3), 0.1, dtype="f4"),
        "height": np.zeros((height, width), dtype="f4"),
        "wetness": np.zeros((height, width), dtype="f4"),
        "direction": np.zeros((height, width, 2), dtype="f4"),
        "roughness": np.full((height, width), 0.5, dtype="f4"),
        "coverage": np.ones((height, width), dtype="f4"),
    }


class SurfaceContracts(unittest.TestCase):
    def test_config_is_strict_finite_and_defensively_copied(self):
        for bad in (
            {"unknown": 1},
            {"height_scale": True},
            {"grain_um": float("nan")},
            {"family": "oil"},
            {"tone_map": "filmic"},
            {"anisotropy": 2},
            {"scattering": [0, 1, 1]},
            {"pigments_srgb": [[1, 1, 1]]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(bad)
        original = DEFAULTS["pigments_srgb"][0][0]
        copy = validate_config({})
        copy["pigments_srgb"][0][0] = 0.99
        self.assertEqual(DEFAULTS["pigments_srgb"][0][0], original)

    def test_family_defaults_resolve_once_and_accept_explicit_overrides(self):
        default = validate_config({"family": "nocturne"})
        self.assertGreater(default["anisotropy"], validate_config({})["anisotropy"])
        self.assertEqual(validate_config(default), default)
        self.assertEqual(validate_config({"family": "nocturne", "anisotropy": 0})["anisotropy"], 0)

    def test_camera_is_orthonormal_fixed_up_and_frontal_identity(self):
        np.testing.assert_allclose(camera_basis(0, -30), np.eye(3), atol=1e-15)
        for tilt in (-30, 12, 22, 42):
            matrix = camera_basis(tilt, 37)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
            self.assertGreater(matrix[1, 1], 0)
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0)
        with self.assertRaises(ValueError):
            camera_basis(70, 0)

    def test_material_validation_rejects_wrong_units_nonfinite_and_invalid_axial_vectors(self):
        valid = fields()
        self.assertEqual(validate_fields(valid)["height"].shape, (96, 128))
        for name, value in (
            ("height", 4.0),
            ("wetness", -0.1),
            ("pigment", float("inf")),
            ("roughness", 1.2),
            ("direction", 0.8),
        ):
            invalid = {key: array.copy() for key, array in valid.items()}
            invalid[name].fill(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_fields(invalid)
        invalid = {**valid, "height": valid["height"].astype("f8")}
        with self.assertRaises(ValueError):
            validate_fields(invalid)


@unittest.skipUnless(
    os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires a hardware OpenGL 4.3 context"
)
class SurfaceHardwareTests(unittest.TestCase):
    def make_surface(self, **kwargs):
        surface = Surface(kwargs)
        self.addCleanup(surface.close)
        return surface

    def test_flat_ambient_surface_matches_cpu_pigment_optics(self):
        surface = self.make_surface(
            ambient=1, key_strength=0, fill_strength=0, grain_um=0, exposure=1, tone_map="none"
        )
        material = Material(
            **{
                k: surface.config[k]
                for k in ("pigments_srgb", "substrate_srgb", "scattering", "layer_scale")
            }
        )
        state = fields()
        state["pigment"][..., 0] = np.linspace(0, 1, 96, dtype="f4")[:, None]
        actual = surface.render(state, (80, 60))
        # The central guard-band crop samples this affine material field exactly.
        y = ((np.arange(60) + 0.5) / 60 - 0.5) / 1.6 + 0.5
        density = np.tile([0.0, 0.1, 0.1], (60, 80, 1))
        density[..., 0] = ((y * 96 - 0.5) / 95)[:, None]
        expected = reflectance(density, material)[::-1]
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=3e-6)

    def test_camera_parallax_tracks_physical_height(self):
        surface = self.make_surface(
            ambient=1,
            key_strength=0,
            fill_strength=0,
            grain_um=0,
            exposure=1,
            tone_map="none",
            substrate_srgb=[1, 1, 1],
        )
        state = fields(256, 192)
        x = (np.arange(256) + 0.5) / 256 - 0.5
        state["pigment"].fill(0)
        state["pigment"][..., 0] = np.exp(-((x[None, :] / 0.035) ** 2))
        flat = surface.render(state, (128, 96), tilt_degrees=30, azimuth_degrees=0)
        state["height"].fill(0.02)
        raised = surface.render(state, (128, 96), tilt_degrees=30, azimuth_degrees=0)

        def centroid(image):
            weight = (1 - image[..., 0]).sum(axis=0)
            return float(np.dot(weight, np.arange(128)) / weight.sum())

        # Orthographic lateral displacement = -height * sin(tilt) / canvas width.
        expected = -0.02 * np.sin(np.pi / 6) / 0.4 * 128
        self.assertAlmostEqual(centroid(raised) - centroid(flat), expected, delta=0.06)

    def test_nocturne_direction_changes_highlight_without_changing_paint(self):
        surface = self.make_surface(family="nocturne", grain_um=0)
        state = fields()
        state["roughness"].fill(0.3)
        state["direction"][..., 0] = 1
        one = surface.render(state, (64, 48), tilt_degrees=18, azimuth_degrees=30)
        state["direction"][..., 0] = -1  # Axial rotation of 90 degrees.
        other = surface.render(state, (64, 48), tilt_degrees=18, azimuth_degrees=30)
        self.assertGreater(float(np.mean(np.abs(one - other))), 0.0001)
        np.testing.assert_array_equal(state["pigment"], np.full_like(state["pigment"], 0.1))

    def test_unpainted_nocturne_ground_has_no_fallback_directional_brush(self):
        surface = self.make_surface(family="nocturne", grain_um=0)
        state = fields()
        state["pigment"].fill(0)
        state["coverage"].fill(0)
        state["direction"][..., 0] = 1
        a = surface.render(state, (64, 48), tilt_degrees=20, azimuth_degrees=35)
        state["direction"][..., 0] = -1
        b = surface.render(state, (64, 48), tilt_degrees=20, azimuth_degrees=35)
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)

    def test_interleaved_contexts_repeat_exactly_and_close_is_idempotent(self):
        first = self.make_surface(family="monotype")
        second = self.make_surface(family="nocturne")
        state = fields()
        before = first.render(state, (64, 48), tilt_degrees=16)
        other = second.render(state, (64, 48), tilt_degrees=16)
        after = first.render(state, (64, 48), tilt_degrees=16)
        np.testing.assert_array_equal(before, after)
        self.assertFalse(np.array_equal(before, other))
        first.close()
        first.close()
        with self.assertRaises(RuntimeError):
            first.render(state, (64, 48))

    def test_explicit_frozen_reuse_survives_context_switch_and_caller_mutation(self):
        first = self.make_surface(family="monotype")
        second = self.make_surface(family="fresco")
        with self.assertRaisesRegex(ValueError, "Upload material"):
            first.render(None, (64, 48))
        state = fields()
        state["height"].fill(0.001)
        first.render(state, (64, 48))
        expected = first.render(state, (128, 96), tilt_degrees=20, azimuth_degrees=35)
        second.render(fields(), (64, 48))
        state["pigment"].fill(0)
        state["pigment"][..., 0] = 0.8
        # Explicit reuse refers to uploaded values, never to a caller's array id.
        reused = first.render(None, (128, 96), tilt_degrees=20, azimuth_degrees=35)
        np.testing.assert_array_equal(reused, expected)
        changed = first.render(state, (128, 96), tilt_degrees=20, azimuth_degrees=35)
        self.assertFalse(np.array_equal(changed, reused))

    def test_relief_casts_shadow_and_rendering_preserves_every_input_field(self):
        lit = self.make_surface(
            ambient=0.15,
            key_strength=1,
            fill_strength=0,
            grain_um=0,
            key_elevation_degrees=18,
            key_azimuth_degrees=0,
            shadow_strength=0,
            occlusion_strength=0,
        )
        shadowed = self.make_surface(
            ambient=0.15,
            key_strength=1,
            fill_strength=0,
            grain_um=0,
            key_elevation_degrees=18,
            key_azimuth_degrees=0,
            shadow_strength=1,
            occlusion_strength=0,
        )
        state = fields(256, 192)
        x = (np.arange(256) + 0.5) / 256 - 0.5
        state["height"][:] = (0.008 * np.exp(-((x / 0.025) ** 2)))[None, :]
        originals = {name: value.copy() for name, value in state.items()}
        a = lit.render(state, (128, 96))
        b = shadowed.render(state, (128, 96))
        self.assertGreater(float((a - b).max()), 0.01)
        self.assertGreater(float((a - b).mean()), 0.0003)
        for name, value in state.items():
            np.testing.assert_array_equal(value, originals[name])

    def test_output_sizes_and_height_bounds_are_checked(self):
        surface = self.make_surface()
        state = fields()
        with self.assertRaises(ValueError):
            surface.render(state, (64, 64))
        with self.assertRaises(ValueError):
            surface.render(state, (64.0, 48))
        with self.assertRaises(ValueError):
            surface.render(state, (64, 48), tilt_degrees=80)
        narrow_guard = self.make_surface(domain_scale=1.25)
        with self.assertRaisesRegex(ValueError, "guarded painting"):
            narrow_guard.render(state, (64, 48), tilt_degrees=42, azimuth_degrees=0)
