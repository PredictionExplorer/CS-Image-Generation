"""Directional contact relief: orientation, native volume, gaps and frozen views."""

from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

import numpy as np

from . import test_packing
from .packing import (
    DIRECTIONAL_VERSION,
    FILM_FRACTION,
    affinity_plan,
    directional_affinity,
    directional_packing_displacement,
    packing_displacement,
    validate_directional,
)
from .surface import Surface, validate_config, validate_fields
from .test_interaction_surface import interaction_fields
from .test_optics import palette
from .test_packing import fixture
from .test_surface import fields

STRONG = {"version": DIRECTIONAL_VERSION, "strength": 1.0, "anisotropy": 0.95}


def bump(size=129, angle=0):
    y, x = np.mgrid[:size, :size] - size // 2
    aggregate = np.exp(-(x * x + y * y) / (2 * 1.5**2))
    contact = np.ones((size, size))
    fabric = np.broadcast_to((np.cos(2 * angle), np.sin(2 * angle)), (size, size, 2)).copy()
    return np.full((size, size), 1e-3), aggregate, contact, fabric, np.ones((size, size), bool)


def covariance(values):
    weights = np.maximum(values, 0)
    y, x = np.indices(weights.shape)
    x = x - np.sum(x * weights) / weights.sum()
    y = y - np.sum(y * weights) / weights.sum()
    return (
        np.array(
            [
                [np.sum(weights * x * x), np.sum(weights * x * y)],
                [np.sum(weights * x * y), np.sum(weights * y * y)],
            ]
        )
        / weights.sum()
    )


class DirectionalReliefContracts(unittest.TestCase):
    def test_versioned_controls_are_optional_bounded_and_zero_is_exact_legacy(self):
        base = validate_config({"interaction": {"packing_strength": 1}})
        for value in (None, {"strength": 0}):
            self.assertEqual(
                validate_config(
                    {**base, "interaction": {**base["interaction"], "directional_relief": value}}
                ),
                base,
            )
        active = validate_config({"interaction": {"packing_strength": 1, "directional_relief": {}}})
        self.assertEqual(active, validate_config(active))
        self.assertEqual(
            active["interaction"]["directional_relief"]["version"], DIRECTIONAL_VERSION
        )
        for value in (
            False,
            [],
            {"version": "v2"},
            {"strength": True},
            {"strength": -1},
            {"strength": 1.01},
            {"anisotropy": 0.96},
            {"anisotropy": float("nan")},
            {"unknown": 1},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_directional(value)
        with self.assertRaisesRegex(ValueError, "positive packing"):
            validate_config({"interaction": {"directional_relief": {}}})
        for bias in (-0.04, 0.04):
            self.assertEqual(validate_config({"roughness_bias": bias})["roughness_bias"], bias)

    def test_fabric_rotation_turns_affinity_and_relief_along_the_fabric(self):
        for angle in (0, np.pi / 4, np.pi / 2):
            h, g, c, q, mask = bump(angle=angle)
            affinity = directional_affinity(
                g, c, q, mask, canvas_width_m=0.0645, directional=STRONG, length_um=2400
            )
            delta = directional_packing_displacement(
                h,
                g,
                mask,
                contact=c,
                fabric=q,
                canvas_width_m=0.0645,
                directional=STRONG,
                length_um=2400,
            )
            direction = np.array((np.cos(angle), np.sin(angle)))
            for value in (affinity, delta):
                eigenvalues, axes = np.linalg.eigh(covariance(value))
                self.assertGreater(abs(float(axes[:, -1] @ direction)), 0.99)
                self.assertGreater(eigenvalues[-1] / eigenvalues[0], 2)
            self.assertGreater(delta.max(), 1e-5)
            self.assertLess(abs(delta.sum()) / h.sum(), 1e-15)

    def test_inactive_contact_or_fabric_adds_no_affinity_pattern(self):
        h, g, c, q, mask = bump()
        expected = packing_displacement(h, g, mask, canvas_width_m=0.0645, length_um=2400)
        for contact, fabric, controls in (
            (c, q * 0, STRONG),
            (c * 0, q * 0, STRONG),
            (None, None, {"strength": 0}),
        ):
            actual = directional_packing_displacement(
                h,
                g,
                mask,
                contact=contact,
                fabric=fabric,
                directional=controls,
                canvas_width_m=0.0645,
                length_um=2400,
            )
            np.testing.assert_array_equal(actual, expected)
        constant = directional_packing_displacement(
            h,
            g * 0 + 0.37,
            mask,
            contact=c,
            fabric=q,
            directional=STRONG,
            canvas_width_m=0.0645,
            length_um=2400,
        )
        np.testing.assert_array_equal(constant, 0)

    def test_four_direction_paths_cannot_bridge_slits_or_corner_contacts(self):
        y, x = np.mgrid[:48, :48]
        for mask, aggregate in (
            (x != 24, (x > 24).astype(float)),
            (x != y, (x > y).astype(float)),
            ((x + y) % 2 == 0, (x / 48).astype(float)),
        ):
            q = np.zeros((*mask.shape, 2))
            q[..., 1] = 1
            actual = directional_packing_displacement(
                mask * 1e-3,
                aggregate,
                mask,
                contact=np.ones_like(aggregate),
                fabric=q,
                directional=STRONG,
                canvas_width_m=0.025,
                length_um=2400,
            )
            np.testing.assert_array_equal(actual, 0)

    def test_random_history_conserves_volume_capacity_and_caller_arrays(self):
        rng = np.random.default_rng(912)
        h, g, mask = fixture(72, 48)
        contact = rng.uniform(0, 1, h.shape)
        angle = rng.uniform(-np.pi, np.pi, h.shape)
        fabric = contact[..., None] * np.stack((np.cos(angle), np.sin(angle)), -1)
        original = [value.copy() for value in (h, g, mask, contact, fabric)]
        for strength in (0.25, 1):
            delta = directional_packing_displacement(
                h,
                g,
                mask,
                contact=contact,
                fabric=fabric,
                directional=STRONG,
                canvas_width_m=0.05,
                length_um=2400,
                strength=strength,
            )
            self.assertLess(abs(delta.sum()) / h.sum(), 1e-15)
            self.assertTrue(np.all(np.abs(delta) <= FILM_FRACTION * strength * h + 1e-18))
            self.assertTrue(np.all(h + delta >= 0))
            np.testing.assert_array_equal(delta[~mask], 0)
        for before, after in zip(original, (h, g, mask, contact, fabric), strict=True):
            np.testing.assert_array_equal(before, after)

    def test_zero_height_cells_also_stop_affinity_transport(self):
        h = np.full((32, 32), 1e-3)
        h[:, 16] = 0
        g = np.zeros_like(h)
        g[:, 17:] = 1
        q = np.zeros((*h.shape, 2))
        q[..., 0] = 1
        delta = directional_packing_displacement(
            h,
            g,
            np.ones_like(h, bool),
            contact=np.ones_like(h),
            fabric=q,
            directional=STRONG,
            canvas_width_m=0.025,
            length_um=2400,
        )
        np.testing.assert_array_equal(delta, 0)

    def test_physical_plan_and_subpixel_response_are_bounded(self):
        for width in (128, 256, 512):
            plan = affinity_plan((width, width), 0.4, 1200)
            self.assertEqual(len(plan), 28)
            self.assertEqual({step.axis for step in plan}, {0, 1, 2, 3})
            self.assertTrue(
                all(1 <= step.stride <= 16 and 0 < step.relaxation <= 1 for step in plan)
            )
        with self.assertRaisesRegex(ValueError, "16-cell"):
            affinity_plan((4096, 3072), 0.05, 2400)
        _, g, c, q, mask = bump(33)
        coarse = directional_affinity(
            g, c, q, mask, canvas_width_m=0.4, directional=STRONG, length_um=80
        )
        self.assertLess(float(np.max(np.abs(coarse - g))), 5e-5)


class ExtendedSurfaceFieldsTests(unittest.TestCase):
    def test_traits_and_structure_are_optional_atomic_and_preserved(self):
        for interaction in (False, True):
            state = interaction_fields() if interaction else fields()
            for layer in ("upper", "lower"):
                state[f"structure_{layer}"] = np.full((24, 32), 0.4, "f4")
                if interaction:
                    state[f"trait_{layer}"] = np.full((24, 32, 2), -0.2, "f4")
            original = copy.deepcopy(state)
            actual = validate_fields(state)
            for name in state:
                np.testing.assert_array_equal(actual[name], original[name])
            for name in ("structure_upper", "trait_upper"):
                if name in state:
                    bad = state.copy()
                    del bad[name]
                    with self.assertRaises(ValueError):
                        validate_fields(bad)

    def test_extra_field_shapes_bounds_and_unbound_traits_are_rejected(self):
        for prefix, shape, value in (("trait", (24, 32, 2), 1.01), ("structure", (24, 32), -0.01)):
            for bad in (
                np.full(shape, value, "f4"),
                np.zeros(shape, "f8"),
                np.zeros((24, 32, 1), "f4"),
                np.full(shape, np.nan, "f4"),
            ):
                state = interaction_fields()
                state[f"{prefix}_upper"] = bad
                state[f"{prefix}_lower"] = np.zeros(shape, "f4")
                with self.assertRaises(ValueError):
                    validate_fields(state)
        state = fields()
        state.update(
            trait_upper=np.zeros((24, 32, 2), "f4"), trait_lower=np.zeros((24, 32, 2), "f4")
        )
        with self.assertRaises(ValueError):
            validate_fields(state)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class DirectionalReliefHardwareTests(unittest.TestCase):
    def state(self, angle=0):
        h, g, occupied = fixture(128, 96)
        state = interaction_fields(128, 96)
        for name in ("mobile", "underpaint"):
            state[name][...] = occupied[..., None] * 0.04
        state["deposit"].fill(0)
        state["pigment"][:] = state["mobile"] + state["underpaint"]
        state["height"][:] = h
        for layer in ("upper", "lower"):
            history = state[f"interaction_{layer}"]
            history[..., 0] = 0.8
            history[..., 1:3] = np.array((np.cos(2 * angle), np.sin(2 * angle))) * 0.7
            history[..., 3] = g
        return state, g, occupied

    def surface(self, directional, **overrides):
        controls = {
            "silk_strength": 0,
            "grain_strength": 0,
            "packing_strength": 1,
            "packing_length_um": 2400,
        }
        if directional is not None:
            controls["directional_relief"] = directional
        surface = Surface(
            {
                "finish": "glazed",
                "grain_um": 0,
                "canvas_width_m": 0.05,
                "domain_scale": 1.25,
                "height_scale": 2,
                "glaze_relief_strength": 0.2,
                "interaction": controls,
                **overrides,
            },
            palette(4),
        )
        self.addCleanup(surface.close)
        return surface

    def test_gpu_affinity_and_conservative_height_match_cpu_for_four_directions(self):
        for angle in (0, np.pi / 4, np.pi / 2, -np.pi / 4):
            state, g, occupied = self.state(angle)
            surface = self.surface(STRONG)
            surface.render(state, (128, 96))
            with surface.ctx:
                potential = np.frombuffer(surface._packing_relief.potential.read(), "f4").reshape(
                    96, 128, 4
                )
                relative = np.frombuffer(
                    surface._packing_relief.relative_height.read(), "f4"
                ).reshape(96, 128)
            native = potential[..., 0].astype("f8") / FILM_FRACTION
            history = state["interaction_upper"].astype("f8")
            affinity = directional_affinity(
                g,
                history[..., 0],
                history[..., 1:3],
                occupied,
                canvas_width_m=0.0625,
                length_um=2400,
                directional=STRONG,
            )
            np.testing.assert_allclose(potential[..., 1], affinity, rtol=4e-6, atol=3e-7)
            expected = packing_displacement(
                native, affinity, occupied, canvas_width_m=0.0625, length_um=2400
            )
            actual = relative * native
            np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=4e-10)
            self.assertLess(abs(actual.sum()) / native.sum(), 2e-7)
            self.assertLessEqual(float(np.abs(relative).max()), FILM_FRACTION + 2e-7)
            self.assertTrue(np.all(native + actual >= 0))
            np.testing.assert_array_equal(relative[~occupied], 0)
            np.testing.assert_allclose(
                native,
                test_packing.PackingHardwareTests.transformed_height(state, surface.config),
                rtol=3e-7,
                atol=1e-10,
            )
            surface.close()

    def test_disabled_and_unaligned_modes_retain_legacy_pixels(self):
        state, _, _ = self.state()
        for layer in ("upper", "lower"):
            state[f"interaction_{layer}"][..., 1:3] = 0
        baseline = self.surface(None).render(state, (128, 96), tilt_degrees=18)
        for controls in ({"strength": 0}, STRONG):
            actual = self.surface(controls).render(state, (128, 96), tilt_degrees=18)
            np.testing.assert_array_equal(actual, baseline)

    def test_effect_is_visible_and_capture_or_camera_cannot_advance_it(self):
        state, _, _ = self.state(np.pi / 4)
        baseline = self.surface(None).render(state, (256, 192), tilt_degrees=18)
        surface = self.surface(STRONG)
        original = copy.deepcopy(state)
        rendered = surface.render(state, (256, 192), tilt_degrees=18)
        self.assertGreater(float(np.max(np.abs(rendered - baseline))), 0.003)
        self.assertGreater(float(np.mean(np.abs(rendered - baseline))), 1e-5)
        np.testing.assert_array_equal(rendered, surface.render(state, (256, 192), tilt_degrees=18))
        with patch.object(
            surface._packing_relief,
            "prepare",
            side_effect=AssertionError("Recomputed frozen relief"),
        ):
            surface.render(None, (256, 192), tilt_degrees=22, azimuth_degrees=37)
            np.testing.assert_array_equal(
                rendered, surface.render(None, (256, 192), tilt_degrees=18)
            )
        for name in state:
            np.testing.assert_array_equal(state[name], original[name])


if __name__ == "__main__":
    unittest.main()
