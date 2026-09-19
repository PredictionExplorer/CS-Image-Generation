"""Conservative packing relief: native volume, topology, bounds and frozen capture."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from .packing import FILM_FRACTION, packing_displacement, packing_plan
from .surface import Surface, interaction_enabled, validate_config
from .test_interaction_surface import interaction_fields
from .test_optics import palette


def fixture(width=128, height=96):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    occupied = x * x + y * y < 0.85**2
    h = np.where(occupied, (0.8 + 0.2 * np.cos(x * 4) * np.cos(y * 3)) * 1e-3, 0)
    aggregate = np.clip(0.5 + 0.35 * np.sin(x * 17) * np.cos(y * 13), 0, 1)
    return h, aggregate, occupied


class PackingContracts(unittest.TestCase):
    def test_height_interpretation_has_explicit_bounded_range_and_unchanged_default(self):
        self.assertEqual(validate_config()["height_scale"], 1)
        self.assertEqual(validate_config({"height_scale": 100})["height_scale"], 100)
        for value in (-0.01, 100.01, True, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"height_scale": value})

    def test_controls_preserve_old_recipes_and_zero_does_not_enable_a_shader(self):
        self.assertNotIn("packing_strength", validate_config({"interaction": {}})["interaction"])
        c = validate_config(
            {"interaction": {"silk_strength": 0, "grain_strength": 0, "packing_strength": 0}}
        )
        self.assertFalse(interaction_enabled(c))
        c["interaction"]["packing_strength"] = 0.5
        self.assertTrue(interaction_enabled(c))
        self.assertEqual(validate_config(c), c)
        for controls in (
            {"packing_strength": True},
            {"packing_strength": -1},
            {"packing_length_um": 0},
            {"packing_length_um": float("inf")},
        ):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                validate_config({"interaction": controls})

    def test_plan_has_bounded_passes_and_world_unit_coupling(self):
        plan = packing_plan((4096, 3072), 0.64, 1200)
        self.assertEqual(len(plan), 28)
        self.assertEqual(max(p.stride for p in plan), 8)
        self.assertEqual(min(p.stride for p in plan), 1)
        self.assertTrue(all(0 < p.relaxation <= 1 for p in plan))
        with self.assertRaisesRegex(ValueError, "16-cell"):
            packing_plan((4096, 3072), 0.05, 2400)

    def test_conservation_positive_skeleton_and_capacity(self):
        h, g, occupied = fixture()
        original = h.copy()
        for strength in (0, 0.5, 1):
            delta = packing_displacement(
                h, g, occupied, canvas_width_m=0.0625, strength=strength, length_um=2400
            )
            self.assertLess(abs(delta.sum()) / h.sum(), 3e-16)
            self.assertTrue(np.all(np.abs(delta) <= FILM_FRACTION * strength * h + 1e-18))
            self.assertTrue(np.all((h + delta)[occupied] >= (1 - FILM_FRACTION) * h[occupied]))
            np.testing.assert_array_equal(delta[~occupied], 0)
            if strength > 0:
                self.assertGreater(np.max(np.abs(delta)), 1e-6)
        np.testing.assert_array_equal(h, original)

    def test_constant_aggregate_and_disabled_packing_are_exactly_quiet(self):
        h, g, occupied = fixture()
        for constant in (0, 0.37, 1):
            np.testing.assert_array_equal(
                packing_displacement(
                    h, np.full_like(g, constant), occupied, canvas_width_m=0.0625, length_um=2400
                ),
                0,
            )
        np.testing.assert_array_equal(
            packing_displacement(h, g, occupied, canvas_width_m=0.0625, strength=0), 0
        )

    def test_coarse_pairs_cannot_jump_a_one_cell_bare_gap(self):
        h = np.full((32, 64), 1e-3)
        occupied = np.ones_like(h, dtype=bool)
        occupied[:, 31] = False
        h[:, 31] = 0
        g = np.zeros_like(h)
        g[:, 32:] = 1
        # A forbidden coarse transfer would move film despite each connected
        # component having perfectly uniform aggregation and its own equilibrium.
        np.testing.assert_array_equal(
            packing_displacement(h, g, occupied, canvas_width_m=0.02, length_um=2400), 0
        )


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class PackingHardwareTests(unittest.TestCase):
    def state(self):
        h, g, occupied = fixture()
        fields = interaction_fields(128, 96)
        for name in ("mobile", "underpaint"):
            fields[name][...] = occupied[..., None] * 0.04
        fields["deposit"].fill(0)
        fields["pigment"][:] = fields["mobile"] + fields["underpaint"]
        fields["height"][:] = h
        for layer in ("upper", "lower"):
            fields[f"interaction_{layer}"][..., 3] = g
        return fields, g, occupied

    def surface(self, strength, **overrides):
        surface = Surface(
            {
                "finish": "glazed",
                "grain_um": 0,
                "canvas_width_m": 0.05,
                "domain_scale": 1.25,
                "height_scale": 2,
                "glaze_relief_strength": 0.2,
                "interaction": {
                    "silk_strength": 0,
                    "grain_strength": 0,
                    "packing_strength": strength,
                    "packing_length_um": 2400,
                },
                **overrides,
            },
            palette(4),
        )
        self.addCleanup(surface.close)
        return surface

    @staticmethod
    def transformed_height(state, config):
        mass = state["pigment"].astype("f8").sum(-1)
        t = np.clip((mass / config["paint_mass_reference"] - 0.6) / 1.6, 0, 1)
        bank = t * t * (3 - 2 * t)
        return (
            state["height"].astype("f8")
            * config["height_scale"]
            * (1 - config["glaze_relief_strength"] + config["glaze_relief_strength"] * bank)
        )

    def test_gpu_matches_reference_and_conserves_native_displayed_volume(self):
        state, _, occupied = self.state()
        surface = self.surface(1)
        surface.render(state, (128, 96))
        with surface.ctx:
            relative = np.frombuffer(
                surface._packing_relief.relative_height.read(), dtype="f4"
            ).reshape(96, 128)
            potential = np.frombuffer(surface._packing_relief.potential.read(), dtype="f4").reshape(
                96, 128, 2
            )
        native_height = potential[..., 0].astype("f8") / FILM_FRACTION
        actual = relative * native_height
        expected = packing_displacement(
            native_height, potential[..., 1], occupied, canvas_width_m=0.0625, length_um=2400
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-10)
        self.assertLess(abs(actual.sum()) / native_height.sum(), 2e-7)
        self.assertLessEqual(float(np.abs(relative).max()), FILM_FRACTION + 2e-7)
        self.assertTrue(np.all(1 + relative >= 1 - FILM_FRACTION - 2e-7))
        np.testing.assert_array_equal(relative[~occupied], 0)
        np.testing.assert_allclose(
            native_height, self.transformed_height(state, surface.config), rtol=3e-7, atol=1e-10
        )

    def test_quiet_height_is_pixel_identical_and_camera_does_not_recompute(self):
        state, _, _ = self.state()
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 3] = 0.37
        control, packed = self.surface(0), self.surface(1)
        a = control.render(state, (128, 96), tilt_degrees=18)
        b = packed.render(state, (128, 96), tilt_degrees=18)
        np.testing.assert_array_equal(a, b)
        with patch.object(
            packed._packing_relief, "prepare", side_effect=AssertionError("Recomputed frozen paint")
        ):
            np.testing.assert_array_equal(b, packed.render(None, (128, 96), tilt_degrees=18))

    def test_quiet_packing_near_height_limit_uses_actual_extent(self):
        state, _, occupied = self.state()
        # 14.9% of visible width is legal; hypothetical +12% expansion is not.
        state["height"][:] = occupied * (0.05 * 0.149 / 2)
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 3] = 0.37
        control, packed = self.surface(0), self.surface(1)
        expected = control.render(state, (128, 96))
        np.testing.assert_array_equal(expected, packed.render(state, (128, 96)))

    def test_large_height_interpretation_retains_actual_geometry_safeguard(self):
        state, _, _ = self.state()
        state["height"] *= 0.001
        surface = self.surface(1, height_scale=100)
        self.assertTrue(np.isfinite(surface.render(state, (128, 96), tilt_degrees=18)).all())
        state["height"] *= 100
        with self.assertRaisesRegex(ValueError, "15%"):
            surface.render(state, (128, 96))

    def test_native_capture_matches_cpu_and_retains_frozen_geometry(self):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette
        from .test_laminate import LAMINATE

        engine = Engine(
            SourceFixture(),
            {**LAMINATE, "interaction": {}, "resolution": [96, 72], "substrate_um": 0, "steps": 40},
            generate_palette("0xb7f327f9f722", 3, mode="composed"),
            [],
        )
        self.addCleanup(engine.close)
        engine.advance_to(20)
        with Surface(
            {"finish": "glazed", "grain_um": 0, "interaction": {"packing_strength": 1}},
            engine.palette,
            gpu_frame=engine.gpu_frame(),
        ) as surface:
            native = surface.render_gpu(engine.gpu_frame(), (128, 96), tilt_degrees=18)
            cpu = surface.render(engine.snapshot(), (128, 96), tilt_degrees=18)
            np.testing.assert_allclose(native, cpu, rtol=5e-5, atol=4e-6)
            frozen = surface.render_gpu(engine.gpu_frame(), (128, 96), tilt_degrees=18)
            engine.advance_to(40)
            with patch.object(
                surface._packing_relief,
                "prepare",
                side_effect=AssertionError("Recomputed frozen paint"),
            ):
                np.testing.assert_array_equal(
                    frozen, surface.render_gpu(None, (128, 96), tilt_degrees=18)
                )


if __name__ == "__main__":
    unittest.main()
