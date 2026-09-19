"""GPU optics/reference agreement and frozen physically scaled rendering."""

import os
import unittest

import numpy as np

from tools.estuary.optics import srgb_to_linear

from .optics import glazed_density_scale, layered_reflectance, reflectance
from .surface import GLAZE_DEFAULTS, Surface, camera_basis, validate_config, validate_fields
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
    def test_glaze_defaults_are_opt_in_and_old_resolved_recipes_are_unchanged(self):
        for finish in ("fresco", "crisp"):
            resolved = validate_config({"finish": finish})
            self.assertFalse(set(GLAZE_DEFAULTS) & resolved.keys())
            self.assertEqual(validate_config(resolved), resolved)
        glazed = validate_config({"finish": "glazed"})
        self.assertEqual(validate_config(glazed), glazed)
        for key, value in GLAZE_DEFAULTS.items():
            self.assertEqual(glazed[key], value)
        for bad in (
            {"glaze_min_mass_ratio": 0},
            {"glaze_max_mass_ratio": 0.9},
            {"glaze_relief_strength": 1.1},
            {"glaze_relief_strength": True},
        ):
            with self.assertRaises(ValueError):
                validate_config({"finish": "glazed", **bad})

    def test_strict_controls_are_finite_and_idempotent(self):
        self.assertEqual(validate_config(validate_config()), validate_config())
        for bad in (
            {"family": "fresco"},
            {"mix_control": 1.1},
            {"height_scale": True},
            {"exposure": float("nan")},
            {"mode": "magic"},
            {"finish": "varnish"},
            {"paint_mass_threshold": 0},
            {"paint_mass_reference": float("inf")},
            {"ground_srgb": [1, True, 1]},
            {"ground_srgb": [1, 1]},
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


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class CrispSurfaceHardwareTests(unittest.TestCase):
    def make_surface(self, config=None, chosen_palette=None):
        config = {} if config is None else config
        surface = Surface({"finish": "crisp", **config}, chosen_palette or palette())
        self.addCleanup(surface.close)
        return surface

    @staticmethod
    def pure_fields(width=128, height=96, concentration=0.02):
        state = fields(width, height)
        for name in ("mobile", "deposit", "underpaint", "pigment"):
            state[name].fill(0)
        state["mobile"][..., 0] = concentration
        state["pigment"][:] = state["mobile"]
        return state

    @staticmethod
    def flat_config():
        return {
            "ambient": 1,
            "key_strength": 0,
            "fill_strength": 0,
            "grain_um": 0,
            "exposure": 1,
            "tone_map": "none",
            "mix_control": 0,
        }

    def test_empty_and_low_mass_have_exact_uniform_ground_for_every_camera(self):
        ground = [0.9, 0.9, 0.9]
        surface = self.make_surface({"ground_srgb": ground, "grain_um": 30})
        state = self.pure_fields(concentration=0)
        state["wetness"][:] = np.linspace(0, 1, 128, dtype="f4")[None, :]
        state["height"][:] = np.linspace(0, 0.02, 96, dtype="f4")[:, None]
        state["roughness"][:] = np.linspace(0, 1, 128, dtype="f4")[None, :]
        state["direction"][..., 0] = 1
        for mass in (0, 0.001):
            state["deposit"][..., 0] = mass * 0.6
            state["underpaint"][..., 1] = mass * 0.4
            state["pigment"][:] = state["deposit"] + state["underpaint"]
            originals = {name: value.copy() for name, value in state.items()}
            for tilt in (0, 12, 30):
                image = surface.render(state, (128, 96), tilt_degrees=tilt)
                expected = np.broadcast_to(srgb_to_linear(ground).astype("f4"), image.shape)
                np.testing.assert_array_equal(image, expected)
            for name in state:
                np.testing.assert_array_equal(state[name], originals[name])

    def test_crisp_controls_have_no_effect_on_default_fresco_pixels(self):
        first = self.make_surface({"finish": "fresco"})
        second = self.make_surface(
            {
                "finish": "fresco",
                "paint_mass_threshold": 0.9,
                "paint_mass_reference": 5,
                "ground_srgb": [0, 0.3, 0.8],
            }
        )
        state = fields()
        np.testing.assert_array_equal(first.render(state, (64, 48)), second.render(state, (64, 48)))

    def test_optical_normalization_preserves_phase_ratios_and_avoids_white_fringes(self):
        chosen = palette()
        for count in (4, 6):
            chosen = palette(count)
            state = fields(count=count)
            original = {name: value.copy() for name, value in state.items()}
            mass = state["pigment"].sum(axis=-1)
            scale = 0.15 / mass
            for mode in ("homogeneous", "layered"):
                surface = self.make_surface({"mode": mode}, chosen)
                surface.render(state, (32, 24))
                with surface.ctx:
                    rgba = np.frombuffer(
                        surface._textures[0].read(alignment=1), dtype="f4"
                    ).reshape(24, 32, 4)
                np.testing.assert_allclose(rgba[..., 3], mass, rtol=3e-7, atol=1e-8)
                if mode == "homogeneous":
                    expected = reflectance(
                        state["pigment"] * scale[..., None], chosen, mixedness=state["mixing"]
                    )
                else:
                    expected = layered_reflectance(
                        *(
                            state[name] * scale[..., None]
                            for name in ("underpaint", "deposit", "mobile")
                        ),
                        chosen,
                        mixedness=state["mixing"],
                    )
                np.testing.assert_allclose(
                    rgba[..., :3] / rgba[..., 3, None], expected, rtol=7e-5, atol=3e-6
                )
                surface.close()
            for name in state:
                np.testing.assert_array_equal(state[name], original[name])

    def test_mass_contour_is_palette_independent_and_stable_in_world_space(self):
        state = self.pure_fields(256, 192)
        u = (np.arange(256, dtype="f4") + 0.5) / 256
        state["mobile"][..., 0] = 0.01 + 0.015 * (u[None, :] - 0.56)
        state["pigment"][:] = state["mobile"]
        config = {**self.flat_config(), "paint_mass_threshold": 0.01}
        first = self.make_surface(config)
        other_palette = palette()
        other_palette["pigments_srgb"][0] = [0.7, 0.12, 0.3]
        second = self.make_surface(config, other_palette)
        for size in ((128, 96), (256, 192)):
            for tilt in (0, 25):
                masks = []
                for surface in (first, second):
                    image = surface.render(state, size, tilt_degrees=tilt, azimuth_degrees=0)
                    interior = image[size[1] // 2, -10, 0]
                    mask = (1 - image[..., 0]) / (1 - interior)
                    masks.append(mask)
                    edge = size[0] - float(mask.mean(axis=0).sum())
                    expected = size[0] * (0.5 + 0.06 * 1.6 * np.cos(np.radians(tilt)))
                    self.assertAlmostEqual(edge, expected, delta=0.02)
                    # A silhouette may antialias one pixel, never a broad wash.
                    partial = (mask[size[1] // 2] > 0.0001) & (mask[size[1] // 2] < 0.9999)
                    self.assertLessEqual(int(partial.sum()), 2)
                np.testing.assert_allclose(masks[0], masks[1], atol=2e-5, rtol=0)

    def test_outside_does_not_receive_shadows_from_neighboring_paint_relief(self):
        state = self.pure_fields(256, 192, concentration=0.0004)
        state["mobile"][:, 100:180, 0] = 0.04
        state["pigment"][:] = state["mobile"]
        state["height"][:, 100:180] = 0.01
        surface = self.make_surface({"shadow_strength": 1, "key_elevation_degrees": 12})
        image = surface.render(state, (128, 96))
        np.testing.assert_array_equal(image[:, :30], np.ones_like(image[:, :30]))
        np.testing.assert_array_equal(image[:, -17:], np.ones_like(image[:, -17:]))
        self.assertLess(float(image[:, 64:80].mean()), 0.9)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class GlazedSurfaceHardwareTests(unittest.TestCase):
    def surface(self, count=4, **config):
        surface = Surface({"finish": "glazed", **config}, palette(count))
        self.addCleanup(surface.close)
        return surface

    def test_gpu_optical_thickness_matches_cpu_for_both_models_counts_and_layer_orders(self):
        from . import optics, spectral

        for count in (4, 6):
            state = fields(count=count)
            state["deposit"].fill(0)
            for phase in ("underpaint", "mobile"):
                state[phase][..., -1] = 0
                state[phase] *= np.geomspace(0.02, 8, 32, dtype="f4")[None, :, None]
            state["pigment"][:] = state["underpaint"] + state["mobile"]
            mass = state["pigment"].sum(-1)
            scale = glazed_density_scale(mass)
            for model in ("rgb", "spectral"):
                for mode in ("layered", "homogeneous"):
                    surface = self.surface(count, optics_model=model, mode=mode)
                    surface.render(state, (32, 24))
                    with surface.ctx:
                        rgba = np.frombuffer(surface._textures[0].read(), dtype="f4").reshape(
                            24, 32, 4
                        )
                    reference = spectral if model == "spectral" else optics
                    material = surface.spectral if model == "spectral" else surface.palette
                    arguments = (
                        [
                            state[name] * scale[..., None]
                            for name in ("underpaint", "deposit", "mobile")
                        ]
                        if mode == "layered"
                        else [state["pigment"] * scale[..., None]]
                    )
                    evaluator = (
                        reference.layered_reflectance
                        if mode == "layered"
                        else reference.reflectance
                    )
                    expected = evaluator(*arguments, material, mixedness=state["mixing"])
                    np.testing.assert_allclose(rgba[..., 3], mass, rtol=4e-7, atol=1e-8)
                    np.testing.assert_allclose(
                        rgba[..., :3] / rgba[..., 3, None], expected, atol=3e-6, rtol=9e-5
                    )
                    surface.close()

    def test_actual_interior_amount_varies_but_empty_exterior_is_exact_ground(self):
        config = CrispSurfaceHardwareTests.flat_config()
        surface = self.surface(**config, ground_srgb=[0.06, 0.09, 0.13])
        state = CrispSurfaceHardwareTests.pure_fields(128, 96, concentration=0)
        state["mobile"][:, 48:64, 0] = 0.065
        state["mobile"][:, 64:80, 0] = 0.28
        state["pigment"][:] = state["mobile"]
        image = surface.render(state, (128, 96))
        ground = srgb_to_linear([0.06, 0.09, 0.13]).astype("f4")
        np.testing.assert_array_equal(image[:, :25], np.broadcast_to(ground, image[:, :25].shape))
        self.assertGreater(float(np.max(np.abs(image[48, 52] - image[48, 77]))), 0.03)
        # Arbitrary unpainted height/wetness cannot introduce exterior haze.
        state["height"][:, :32] = 0.005
        state["wetness"][:, :32] = 1
        image = surface.render(state, (128, 96), tilt_degrees=18)
        np.testing.assert_array_equal(image[:, :20], np.broadcast_to(ground, image[:, :20].shape))

    def test_legacy_finishes_ignore_all_glaze_controls(self):
        for finish in ("fresco", "crisp"):
            first = self.surface(finish=finish)
            second = self.surface(
                finish=finish,
                glaze_min_mass_ratio=0.9,
                glaze_max_mass_ratio=7,
                glaze_relief_strength=0,
            )
            state = fields()
            np.testing.assert_array_equal(
                first.render(state, (64, 48)), second.render(state, (64, 48))
            )

    def test_laminate_native_capture_matches_cpu_and_real_layer_swaps_change_color(self):
        from unittest.mock import patch

        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette
        from .run import field_digest

        for count in (3, 5):
            chosen = generate_palette("0xbc53af1cd380", count, mode="harmonic")
            chosen["layer_fractions"] = [0.3] * (count + 1)
            engine = Engine(
                SourceFixture(),
                {
                    "resolution": [96, 72],
                    "steps": 60,
                    "material_model": "laminate",
                    "initial_pattern": "scattered",
                    "deposition": 0,
                    "settling_scale": 0,
                    "underpaint_strength": 0,
                    "underpaint_release": 0,
                    "burial_rate": 0,
                    "substrate_um": 0,
                },
                chosen,
                [],
            )
            self.addCleanup(engine.close)
            engine.advance_to(20)
            for model in ("rgb", "spectral"):
                surface = Surface(
                    {"finish": "glazed", "optics_model": model, "layer_scale": 12, "grain_um": 0},
                    chosen,
                    gpu_frame=engine.gpu_frame(),
                )
                self.addCleanup(surface.close)
                state = engine.snapshot()
                before = field_digest(state)
                for tilt in (0, 18):
                    with patch.object(engine, "snapshot", side_effect=AssertionError("readback")):
                        actual = surface.render_gpu(
                            engine.gpu_frame(), (128, 96), tilt_degrees=tilt, supersampling=2
                        )
                    raster = surface.render(state, (256, 192), tilt_degrees=tilt)
                    expected = raster.reshape(96, 2, 128, 2, 3).mean((1, 3), dtype="f4")
                    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
                self.assertEqual(field_digest(engine.snapshot()), before)
                surface.close()
            # A controlled real two-layer stack isolates optical order from transport.
            with engine._gpu.ctx:
                for groups in (engine._gpu.blocks, engine._gpu.underpaints):
                    for group in groups:
                        group[0].write(np.zeros((72, 96, 4), dtype="f4"))
                lower = np.zeros((72, 96, 4), dtype="f4")
                upper = lower.copy()
                lower[..., 0] = 0.09
                upper[..., 1] = 0.08
                engine._gpu.underpaints[0][0].write(lower)
                engine._gpu.blocks[0][0].write(upper)
            surface = Surface(
                {
                    "finish": "glazed",
                    "optics_model": "spectral",
                    "layer_scale": 12,
                    **CrispSurfaceHardwareTests.flat_config(),
                },
                chosen,
                gpu_frame=engine.gpu_frame(),
            )
            self.addCleanup(surface.close)
            first_state = engine.snapshot()
            first = surface.render_gpu(engine.gpu_frame(), (128, 96))
            with engine._gpu.ctx:
                engine._gpu.underpaints[0][0].write(upper)
                engine._gpu.blocks[0][0].write(lower)
            second_state = engine.snapshot()
            second = surface.render_gpu(engine.gpu_frame(), (128, 96))
            np.testing.assert_array_equal(first_state["pigment"], second_state["pigment"])
            self.assertGreater(float(np.max(np.abs(first - second))), 0.01)
            np.testing.assert_allclose(
                second, surface.render(second_state, (128, 96)), atol=3e-6, rtol=3e-5
            )
            surface.close()
            engine.close()
