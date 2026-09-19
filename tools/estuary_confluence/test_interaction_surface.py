"""Material-history contracts and opt-in frozen surface-scattering behavior."""

from __future__ import annotations

import copy
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from . import test_gpu_frame
from .surface import Surface, interaction_enabled, validate_config, validate_fields
from .test_optics import palette
from .test_surface import fields


def interaction_fields(width=32, height=24, count=4):
    result = fields(width, height, count)
    for layer in ("upper", "lower"):
        result[f"origin_{layer}"] = np.zeros((height, width, 2), "f4")
        result[f"interaction_{layer}"] = np.zeros((height, width, 4), "f4")
    return result


class InteractionSurfaceContracts(unittest.TestCase):
    def test_grain_contrast_is_optional_bounded_and_does_not_enable_absent_effects(self):
        self.assertNotIn("grain_contrast", validate_config({"interaction": {}})["interaction"])
        for contrast in (1, 4, 8):
            c = validate_config(
                {
                    "interaction": {
                        "grain_contrast": contrast,
                        "silk_strength": 0,
                        "grain_strength": 0,
                    }
                }
            )
            self.assertFalse(interaction_enabled(c))
            self.assertEqual(validate_config(c), c)
        for contrast in (0.9, 8.1, True, float("nan")):
            with self.subTest(contrast=contrast), self.assertRaises(ValueError):
                validate_config({"interaction": {"grain_contrast": contrast}})

    def test_configuration_is_optional_strict_and_idempotent(self):
        self.assertNotIn("interaction", validate_config())
        for value in (None, {"silk_strength": 0, "grain_strength": 0}):
            config = validate_config({"interaction": value})
            self.assertFalse(interaction_enabled(config))
            self.assertEqual(validate_config(config), config)
        config = validate_config({"interaction": {}})
        self.assertTrue(interaction_enabled(config))
        self.assertEqual(validate_config(config), config)
        for bad in (
            True,
            [],
            {"noise": 1},
            {"silk_strength": True},
            {"silk_strength": -0.1},
            {"grain_strength": 1.1},
            {"grain_strength": float("nan")},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config({"interaction": bad})

    def test_material_extension_is_atomic_and_preserved_without_clipping(self):
        state = interaction_fields()
        state["interaction_upper"][..., 0] = 0.6
        state["interaction_upper"][..., 1:3] = (0.3, 0.4)
        state["interaction_upper"][..., 3] = 0.2
        original = copy.deepcopy(state)
        actual = validate_fields(state)
        for name in state:
            np.testing.assert_array_equal(actual[name], original[name])
        for name in ("origin_upper", "origin_lower", "interaction_upper", "interaction_lower"):
            partial = state.copy()
            del partial[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                validate_fields(partial)

    def test_invalid_material_history_is_rejected(self):
        for name, component, value in (
            ("origin_upper", 0, float("nan")),
            ("origin_lower", 1, 1e7),
            ("interaction_upper", 0, -0.1),
            ("interaction_upper", 0, 1.1),
            ("interaction_upper", 1, 0.5),
            ("interaction_upper", 1, 1e35),
            ("interaction_lower", 3, -0.1),
            ("interaction_lower", 3, 1.1),
        ):
            state = interaction_fields()
            state[name][..., component] = value
            with self.subTest(name=name, component=component), self.assertRaises(ValueError):
                validate_fields(state)
        for name in ("origin_lower", "interaction_upper"):
            for change in (lambda a: a.astype("f8"), lambda a: a[..., :1]):
                state = interaction_fields()
                state[name] = change(state[name])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_fields(state)

    def test_native_history_requires_atomic_laminate_float_textures(self):
        owner, frame = test_gpu_frame.ViewContracts.fixture()
        texture = SimpleNamespace(ctx=frame.context, size=frame.size, components=4, dtype="f4")
        extension = {
            name: texture
            for name in ("origin_upper", "origin_lower", "interaction_upper", "interaction_lower")
        }
        valid = replace(frame, material_model="laminate", **extension)
        self.assertIs(valid.validate(), valid)
        self.assertTrue(valid.has_interaction)
        self.assertFalse(frame.has_interaction)
        for changes in (
            {"origin_upper": None},
            {"interaction_lower": None},
            {"material_model": "legacy"},
            {
                "origin_lower": SimpleNamespace(
                    ctx=frame.context, size=frame.size, components=2, dtype="f4"
                )
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes).validate()
        self.assertIsNotNone(owner)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class InteractionSurfaceHardwareTests(unittest.TestCase):
    def surface(self, interaction=None, **overrides):
        config = {
            "finish": "glazed",
            "grain_um": 0,
            "height_scale": 0,
            "key_elevation_degrees": 65,
            "key_azimuth_degrees": 0,
            "ambient": 0.2,
            "key_strength": 2,
            "exposure": 1,
            "interaction": interaction,
            **overrides,
        }
        surface = Surface(config, palette(4))
        self.addCleanup(surface.close)
        return surface

    @staticmethod
    def uniform_state():
        state = interaction_fields()
        for name in ("mobile", "deposit", "underpaint"):
            state[name].fill(0.025)
        state["pigment"][:] = state["mobile"] + state["deposit"] + state["underpaint"]
        state["direction"][...] = (1, 0)
        state["mixing"].fill(0.5)
        return state

    def test_disabled_extension_is_pixel_identical_and_missing_history_fails(self):
        state = self.uniform_state()
        base = {
            name: value
            for name, value in state.items()
            if not name.startswith(("origin_", "interaction_"))
        }
        expected = self.surface().render(base, (64, 48), tilt_degrees=16)
        disabled = self.surface({"silk_strength": 0, "grain_strength": 0})
        np.testing.assert_array_equal(expected, disabled.render(state, (64, 48), tilt_degrees=16))
        with self.assertRaisesRegex(ValueError, "material history"):
            self.surface({}).render(base, (64, 48))

    def test_grain_contrast_preserves_identity_quiet_paint_and_raw_material(self):
        state = self.uniform_state()
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 3] = 0.4
        original = copy.deepcopy(state)
        controls = {"silk_strength": 0.4, "grain_strength": 0.8}
        linear = self.surface(controls)
        identity = self.surface({**controls, "grain_contrast": 1})
        contrasted = self.surface({**controls, "grain_contrast": 8})
        expected = linear.render(state, (64, 48))
        np.testing.assert_array_equal(expected, identity.render(state, (64, 48)))
        visible = contrasted.render(state, (64, 48))
        self.assertGreater(float(np.max(np.abs(expected - visible))), 1e-5)
        for name in state:
            np.testing.assert_array_equal(state[name], original[name])
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 3] = 0
        np.testing.assert_array_equal(
            linear.render(state, (64, 48)), contrasted.render(state, (64, 48))
        )

    def test_silk_reads_frozen_fabric_not_velocity_and_grain_changes_scattering(self):
        state = self.uniform_state()
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][...] = (1, 1, 0, 0)
        silk = self.surface({"silk_strength": 1, "grain_strength": 0})
        a = silk.render(state, (64, 48))
        state["direction"][...] = (-1, 0)
        np.testing.assert_array_equal(a, silk.render(state, (64, 48)))
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 1] = -1
        b = silk.render(state, (64, 48))
        self.assertGreater(float(np.max(np.abs(a - b))), 1e-5)
        grain = self.surface({"silk_strength": 0, "grain_strength": 1})
        smooth = grain.render(state, (64, 48))
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 3] = 1
        granular = grain.render(state, (64, 48))
        self.assertGreater(float(np.max(np.abs(smooth - granular))), 1e-5)
        self.assertTrue(np.isfinite(granular).all())
        self.assertGreaterEqual(float(granular.min()), 0)
        self.assertLessEqual(float(granular.max()), 1)

    def test_quiet_history_and_bare_ground_stay_unchanged(self):
        state = self.uniform_state()
        control = self.surface()
        textured = self.surface({})
        np.testing.assert_allclose(
            control.render(state, (64, 48)), textured.render(state, (64, 48)), atol=2e-7
        )
        for name in ("mobile", "underpaint", "deposit", "pigment"):
            state[name].fill(0)
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][...] = (1, 1, 0, 1)
        np.testing.assert_array_equal(
            control.render(state, (64, 48)), textured.render(state, (64, 48))
        )

    def test_cpu_upload_keeps_camera_frozen_after_caller_mutation(self):
        state = self.uniform_state()
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][...] = (1, 0, 1, 0.5)
        surface = self.surface({})
        expected = surface.render(state, (64, 48), tilt_degrees=20)
        state["interaction_upper"].fill(0)
        state["interaction_lower"].fill(0)
        np.testing.assert_array_equal(expected, surface.render(None, (64, 48), tilt_degrees=20))

    def native_engine(self):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette
        from .test_laminate import LAMINATE

        engine = Engine(
            SourceFixture(),
            {**LAMINATE, "interaction": {}, "resolution": [96, 72], "steps": 40},
            generate_palette("0xb7f327f9f722", 3, mode="composed"),
            [],
        )
        self.addCleanup(engine.close)
        return engine

    def test_native_capture_matches_cpu_and_is_frozen_after_engine_advance(self):
        engine = self.native_engine()
        engine.advance_to(20)
        surface = Surface(
            {"finish": "glazed", "grain_um": 0, "interaction": {}},
            engine.palette,
            gpu_frame=engine.gpu_frame(),
        )
        self.addCleanup(surface.close)
        state = engine.snapshot()
        gpu = surface.render_gpu(engine.gpu_frame(), (128, 96), tilt_degrees=18)
        cpu = surface.render(state, (128, 96), tilt_degrees=18)
        np.testing.assert_allclose(gpu, cpu, rtol=3e-5, atol=3e-6)
        # Refresh staging through the native path before changing live textures.
        frozen = surface.render_gpu(engine.gpu_frame(), (128, 96), tilt_degrees=18)
        engine.advance_to(40)
        np.testing.assert_array_equal(frozen, surface.render_gpu(None, (128, 96), tilt_degrees=18))

    def test_native_invalid_history_is_rejected_before_rendering(self):
        engine = self.native_engine()
        frame = engine.gpu_frame()
        surface = Surface(
            {"finish": "glazed", "grain_um": 0, "interaction": {}},
            engine.palette,
            gpu_frame=frame,
        )
        self.addCleanup(surface.close)
        with frame.context:
            corrupt = np.zeros((72, 96, 4), dtype="f4")
            corrupt[0, 0, 1] = 0.5  # Fabric cannot exceed zero accumulated contact.
            frame.interaction_upper.write(corrupt)
        with self.assertRaisesRegex(FloatingPointError, "Native GPU material"):
            surface.render_gpu(frame, (128, 96))
        with self.assertRaisesRegex(ValueError, "Upload material"):
            surface.render_gpu(None, (128, 96))


if __name__ == "__main__":
    unittest.main()
