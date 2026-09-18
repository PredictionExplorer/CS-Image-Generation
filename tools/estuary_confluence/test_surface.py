"""GPU optics/reference agreement and frozen physically scaled rendering."""

import os
import unittest

import numpy as np

from .optics import layered_reflectance, reflectance
from .surface import Surface, camera_basis, validate_config, validate_fields
from .test_optics import palette


def fields(width=32, height=24, count=4):
    rng = np.random.default_rng(17)
    phases = {
        name: rng.uniform(0, 0.06, (height, width, count)).astype("f4")
        for name in ("mobile", "deposit", "underpaint")
    }
    return {
        **phases,
        "pigment": phases["mobile"] + phases["deposit"] + phases["underpaint"],
        "height": np.zeros((height, width), "f4"),
        "wetness": np.zeros((height, width), "f4"),
        "mixing": rng.random((height, width)).astype("f4"),
        "direction": np.zeros((height, width, 2), "f4"),
        "roughness": np.full((height, width), 0.6, "f4"),
        "coverage": np.ones((height, width), "f4"),
    }


class SurfaceContracts(unittest.TestCase):
    def test_strict_controls_are_finite_and_idempotent(self):
        self.assertEqual(validate_config(validate_config()), validate_config())
        for bad in (
            {"family": "fresco"},
            {"mix_control": 1.1},
            {"height_scale": True},
            {"exposure": float("nan")},
            {"mode": "magic"},
        ):
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_phases_must_sum_to_total_and_match_palette_dimensions(self):
        state = fields()
        validate_fields(state, 4)
        with self.assertRaises(ValueError):
            validate_fields(state, 6)
        state["pigment"][0, 0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "sum"):
            validate_fields(state, 4)

    def test_invalid_units_shapes_and_direction_are_rejected(self):
        for key, value in (
            ("height", 1),
            ("mixing", 1.1),
            ("mobile", -1),
            ("direction", 0.9),
            ("wetness", float("nan")),
        ):
            state = fields()
            state[key].fill(value)
            with self.assertRaises(ValueError):
                validate_fields(state)
        state = fields()
        state["height"] = state["height"].astype("f8")
        with self.assertRaises(ValueError):
            validate_fields(state)

    def test_frontal_camera_and_fixed_up_are_orthonormal(self):
        np.testing.assert_allclose(camera_basis(0, 30), np.eye(3), atol=1e-15)
        for tilt in (-30, 20, 35):
            matrix = camera_basis(tilt, 22)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
            self.assertGreater(matrix[1, 1], 0)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class SurfaceHardwareTests(unittest.TestCase):
    def surface(self, count=4, **config):
        s = Surface(config, palette(count))
        self.addCleanup(s.close)
        return s

    def prepass(self, s, state):
        s.render(state, (32, 24))
        h, w = state["height"].shape
        with s.ctx:
            return (
                np.frombuffer(s._textures[0].read(alignment=1), dtype="f4")
                .reshape(h, w, 4)[..., :3]
                .copy()
            )

    def test_gpu_prepass_matches_reference_for_both_counts_modes_and_mixture_endpoints(self):
        for count in (4, 6):
            for mode in ("layered", "homogeneous"):
                for control in (0, 1):
                    s = self.surface(count, mode=mode, mix_control=control)
                    state = fields(count=count)
                    actual = self.prepass(s, state)
                    mixed = 1 if control == 0 else state["mixing"]
                    if mode == "layered":
                        expected = layered_reflectance(
                            state["underpaint"],
                            state["deposit"],
                            state["mobile"],
                            s.palette,
                            mixedness=mixed,
                        )
                    else:
                        expected = reflectance(state["pigment"], s.palette, mixedness=mixed)
                    np.testing.assert_allclose(actual, expected, rtol=7e-5, atol=3e-6)
                    s.close()

    def test_uniform_flat_ambient_render_has_correct_linear_color(self):
        s = self.surface(
            ambient=1, key_strength=0, fill_strength=0, grain_um=0, exposure=1, tone_map="none"
        )
        state = fields()
        for key in ("mobile", "deposit", "underpaint"):
            state[key].fill(0.04)
        state["pigment"][:] = state["mobile"] + state["deposit"] + state["underpaint"]
        state["mixing"].fill(0.3)
        actual = s.render(state, (64, 48))
        color = layered_reflectance(
            state["underpaint"][0, 0],
            state["deposit"][0, 0],
            state["mobile"][0, 0],
            s.palette,
            mixedness=0.3,
        )
        np.testing.assert_allclose(
            actual, np.broadcast_to(color, actual.shape), rtol=7e-5, atol=3e-6
        )

    def test_frozen_cache_survives_context_switch_caller_mutation_and_close_is_idempotent(self):
        s = self.surface()
        other = self.surface(6)
        with self.assertRaisesRegex(ValueError, "Upload material"):
            s.render(None, (32, 24))
        state = fields()
        originals = {k: v.copy() for k, v in state.items()}
        first = s.render(state, (64, 48), tilt_degrees=18)
        other.render(fields(count=6), (32, 24))
        for key in state:
            np.testing.assert_array_equal(state[key], originals[key])
        state["mobile"].fill(0)
        state["pigment"][:] = state["underpaint"] + state["deposit"]
        frozen = s.render(None, (64, 48), tilt_degrees=18)
        np.testing.assert_array_equal(first, frozen)
        changed = s.render(state, (64, 48), tilt_degrees=18)
        self.assertFalse(np.array_equal(first, changed))
        s.close()
        s.close()
        with self.assertRaises(RuntimeError):
            s.render(None, (32, 24))

    def test_geometry_parallax_is_in_metres_and_camera_stays_within_guard(self):
        s = self.surface(
            ambient=1,
            key_strength=0,
            fill_strength=0,
            grain_um=0,
            exposure=1,
            tone_map="none",
            mix_control=0,
        )
        state = fields(256, 192)
        for key in ("mobile", "deposit", "underpaint", "pigment"):
            state[key].fill(0)
        x = (np.arange(256) + 0.5) / 256 - 0.5
        state["mobile"][..., 0] = np.exp(-((x[None, :] / 0.035) ** 2))
        state["pigment"][:] = state["mobile"]
        flat = s.render(state, (128, 96), tilt_degrees=30, azimuth_degrees=0)
        state["height"].fill(0.02)
        raised = s.render(state, (128, 96), tilt_degrees=30, azimuth_degrees=0)

        def centroid(image):
            weight = (image[..., 0].max() - image[..., 0]).sum(axis=0)
            return float(np.dot(weight, np.arange(128)) / weight.sum())

        self.assertAlmostEqual(
            centroid(raised) - centroid(flat), -0.02 * 0.5 / 0.4 * 128, delta=0.07
        )
        with self.assertRaises(ValueError):
            s.render(None, (64, 64))
        with self.assertRaises(ValueError):
            s.render(None, (64, 48), tilt_degrees=70)
