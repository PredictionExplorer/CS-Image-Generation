"""Inactive pairs cannot select, displace, or relocate encounter water events."""

from __future__ import annotations

import copy
import itertools
import json
import unittest
from types import SimpleNamespace

import numpy as np

from tools.estuary.source import PAIRS

from .events import REFRACTORY_FRACTION, plan_events


class ProjectedSource:
    """Already-projected measurements with the same frozen source conditioning."""

    def __init__(self, pulses=((0.23, 0, 0.85), (0.51, 1, 0.85), (0.8, 2, 0.85))):
        self.projection = {"proximity_distance_scale_normalized_3d": 0.3, "scale": 1.0}
        self.pulses = pulses
        self.inactive_body = None
        self.calls = []

    def sample(self, fractions):
        fractions = np.asarray(fractions)
        self.calls.append(fractions.copy())
        distance = np.full((len(fractions), 3), 0.9)
        for center, pair, depth in self.pulses:
            distance[:, pair] -= depth * np.exp(-(((fractions - center) / 0.014) ** 2))
        positions = np.zeros((len(fractions), 3, 2))
        positions[..., 0] = fractions[:, None]
        positions[..., 1] = [-0.2, 0.1, 0.4]
        velocities = np.ones_like(positions)
        arc_lengths = np.repeat(fractions[:, None], 3, axis=1)
        proximity = np.full_like(arc_lengths, 0.2)
        if self.inactive_body is not None:
            body = self.inactive_body
            positions[:, body] = np.stack((50 + fractions**2, -30 - fractions), axis=-1)
            velocities[:, body] = [1000, -700]
            arc_lengths[:, body] = 500 * fractions
            proximity[:, body] = 0.99
            for pair, endpoints in enumerate(PAIRS):
                if body in endpoints:
                    distance[:, pair] = 0.9 - 0.895 * np.exp(-(((fractions - 0.237) / 0.009) ** 2))
        return SimpleNamespace(
            pair_distances=distance,
            positions=positions,
            velocities=velocities,
            arc_lengths=arc_lengths,
            proximity=proximity,
        )


class BodyInfluenceEventTests(unittest.TestCase):
    def test_all_three_and_omitted_have_identical_legacy_bytes_and_sampling(self):
        source = ProjectedSource()
        expected = json.dumps(plan_events(source), sort_keys=True).encode()
        expected_calls = source.calls
        for bodies in (None, *itertools.permutations((0, 1, 2))):
            for value in (bodies, list(bodies) if bodies is not None else None):
                with self.subTest(bodies=value):
                    source = ProjectedSource()
                    actual = plan_events(source, bodies=value)
                    self.assertEqual(json.dumps(actual, sort_keys=True).encode(), expected)
                    self.assertEqual(len(source.calls), len(expected_calls))
                    for actual_times, expected_times in zip(
                        source.calls, expected_calls, strict=True
                    ):
                        np.testing.assert_array_equal(actual_times, expected_times)

    def test_each_pair_uses_original_indices_and_both_active_endpoints(self):
        expected = plan_events(ProjectedSource())
        for index, pair in enumerate(PAIRS):
            with self.subTest(pair=pair.tolist()):
                bodies = list(reversed(pair.tolist()))
                actual = plan_events(ProjectedSource(), bodies=bodies)
                self.assertEqual(actual, [expected[index]])
                self.assertEqual(actual[0]["pair"], pair.tolist())

    def test_single_body_has_no_pair_events_and_does_not_sample_source(self):
        for body in range(3):
            source = ProjectedSource()
            self.assertEqual(plan_events(source, bodies=[body]), [])
            self.assertEqual(source.calls, [])

    def test_quiet_active_pair_does_not_borrow_inactive_encounters(self):
        source = ProjectedSource(((0.23, 1, 0.85), (0.8, 2, 0.85)))
        self.assertEqual(plan_events(source, bodies=[0, 1]), [])

    def test_stronger_inactive_pair_cannot_suppress_or_use_quota_of_active_pair(self):
        pulses = ((0.23, 0, 0.69), (0.25, 1, 0.85))
        self.assertLess(pulses[1][0] - pulses[0][0], REFRACTORY_FRACTION)
        for count in (1, 3):
            with self.subTest(count=count):
                original = plan_events(ProjectedSource(pulses), count)
                self.assertEqual([event["pair"] for event in original], [[1, 2]])
                active = plan_events(ProjectedSource(pulses), count, bodies=[0, 1])
                self.assertEqual([event["pair"] for event in active], [[0, 1]])
                self.assertAlmostEqual(active[0]["fraction"], 0.23, delta=8e-6)
                self.assertLess(active[0]["strength"], original[0]["strength"])

    def test_inactive_post_projection_measurements_do_not_change_active_events(self):
        for pair in PAIRS:
            bodies = pair.tolist()
            base, changed = ProjectedSource(), ProjectedSource()
            changed.inactive_body = next(body for body in range(3) if body not in bodies)
            projection = copy.deepcopy(changed.projection)
            expected = plan_events(base, bodies=bodies)
            actual = plan_events(changed, bodies=bodies)
            with self.subTest(bodies=bodies):
                self.assertTrue(expected)
                self.assertEqual(actual, expected)
                self.assertEqual(changed.projection, projection)
                self.assertEqual(len(base.calls), len(changed.calls))
                for a, b in zip(base.calls, changed.calls, strict=True):
                    np.testing.assert_array_equal(a, b)

    def test_body_subsets_are_validated_even_without_requested_events(self):
        invalid = ([], [0, 0], [True], [0, False], [-1], [3], [0.0], [0, 1, 2, 3], 0, {0}, "01")
        for bodies in invalid:
            for count in (0, 3):
                with self.subTest(bodies=bodies, count=count), self.assertRaises(ValueError):
                    plan_events(ProjectedSource(), count, bodies=bodies)
        source = ProjectedSource()
        self.assertEqual(plan_events(source, 0, bodies=[0, 1]), [])
        self.assertEqual(source.calls, [])


if __name__ == "__main__":
    unittest.main()
