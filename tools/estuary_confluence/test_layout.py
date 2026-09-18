"""Physical placement contracts independent of rendering and palette choice."""

import json
import math
import unittest
from unittest import mock

import numpy as np

from .layout import MAX_LOAD_RADIUS, plan_layout, pool_profile


class LayoutTests(unittest.TestCase):
    def test_same_seed_is_exact_and_json_round_trips(self):
        seed = hex((1 << 255) | 0x123456789ABCDEF123456789ABCDEF)
        first = plan_layout(seed, 5, 4 / 3)
        self.assertEqual(first, plan_layout(int(seed, 16), 5, 4 / 3))
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_full_seed_entropy_changes_placement(self):
        low = 0xAB12
        high = low | (1 << 255)
        self.assertNotEqual(plan_layout(low, 5, 4 / 3), plan_layout(high, 5, 4 / 3))

    def test_three_and_five_share_exact_physical_pools(self):
        for seed in (0, 1, 2**256 - 1, 0xBC53AF1CD380):
            three = plan_layout(seed, 3, 4 / 3)
            five = plan_layout(seed, 5, 4 / 3)
            self.assertEqual(three["pools"], five["pools"][:3])
            self.assertEqual([p["pigment_index"] for p in five["pools"]], list(range(5)))

    def test_supported_aspects_and_radii_keep_all_pools_separate_and_inside(self):
        for seed in range(60):
            for aspect in (0.2, 0.5, 1, 4 / 3, 2, 5):
                for radius in (0.001, 0.28, MAX_LOAD_RADIUS):
                    pools = plan_layout(seed, 5, aspect, load_radius=radius)["pools"]
                    for index, pool in enumerate(pools):
                        x, y = pool["position"]
                        self.assertLess(abs(x) + pool["radius"], aspect)
                        self.assertLess(abs(y) + pool["radius"], 1)
                        for previous in pools[:index]:
                            distance = math.dist(pool["position"], previous["position"])
                            self.assertGreater(distance, pool["radius"] + previous["radius"])

    def test_added_colors_receive_substantial_independent_material(self):
        pools = plan_layout("0xbc53af1cd380", 5, 4 / 3)["pools"]
        masses = [p["load"] * math.pi * p["radius"] ** 2 for p in pools]
        self.assertGreater(min(masses), 0.4 * max(masses))
        self.assertEqual(len({tuple(p["position"]) for p in pools}), 5)

    def test_normal_sampling_uses_continuous_positions_not_fallback_strata(self):
        positions = []
        for seed in range(100):
            plan = plan_layout(seed, 5, 4 / 3)
            self.assertEqual(plan["placement_method"], "blue-noise-rejection")
            positions.append(plan["pools"][0]["position"])
        points = np.asarray(positions)
        self.assertGreater(float(np.ptp(points[:, 0])), 1.5)
        self.assertGreater(float(np.ptp(points[:, 1])), 1.2)
        self.assertTrue(np.any(np.abs(points[:, 0]) < 0.15))
        self.assertTrue(np.any(np.abs(points[:, 1]) < 0.15))

    def test_bounded_fallback_is_valid_and_preserves_the_count_prefix(self):
        with mock.patch("tools.estuary_confluence.layout.MAX_LAYOUT_ATTEMPTS", 0):
            for aspect in (0.2, 1, 5):
                five = plan_layout(94, 5, aspect, load_radius=MAX_LOAD_RADIUS)
                three = plan_layout(94, 3, aspect, load_radius=MAX_LOAD_RADIUS)
                self.assertEqual(five["placement_method"], "bounded-fallback")
                self.assertEqual(three["pools"], five["pools"][:3])

    def test_layout_does_not_accept_resolution_or_frame_rate_inputs(self):
        for key in ("resolution", "fps", "colors", "palette"):
            with self.assertRaises(TypeError):
                plan_layout(1, 3, 4 / 3, **{key: 100})

    def test_edge_is_finite_and_confined_to_requested_radial_width(self):
        profile = pool_profile(np.array([0, 0.5, 0.98, 0.99, 1, 1.1]), 1, 0.02)
        np.testing.assert_allclose(profile, [1, 1, 1, 0.5, 0, 0], atol=1e-13)
        self.assertTrue(np.isfinite(profile).all())

    def test_pixel_integration_approaches_the_same_physical_mass(self):
        pool = plan_layout("0x91", 3, 1)["pools"][0]
        radius, edge = pool["radius"], pool["edge_width"]
        exact = pool["load"] * math.pi * radius**2 * (1 - edge + 0.3 * edge**2)
        for width in (512, 1024):
            x = (np.arange(width) + 0.5) / width * 2 - 1
            distance = np.hypot(x[None, :] - pool["position"][0], x[:, None] - pool["position"][1])
            measured = (
                float(pool_profile(distance, radius, edge).sum()) * pool["load"] * (2 / width) ** 2
            )
            self.assertAlmostEqual(measured / exact, 1, delta=0.002)

    def test_invalid_values_are_rejected_without_clamping(self):
        for args in ((1, 4, 1), (True, 3, 1), (1, True, 1), (1, 3, float("nan"))):
            with self.assertRaises(ValueError):
                plan_layout(*args)
        for kwargs in ({"load_radius": 0.35}, {"edge_width": 0}, {"initial_load": -1}):
            with self.assertRaises(ValueError):
                plan_layout(1, 3, 1, **kwargs)


if __name__ == "__main__":
    unittest.main()
