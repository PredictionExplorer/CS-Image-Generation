"""Encounters originate in source geometry, independent of output cadence."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from .events import REFRACTORY_FRACTION, SAMPLE_COUNT, plan_events


class SyntheticSource:
    projection: ClassVar = {"proximity_distance_scale_normalized_3d": 0.3, "scale": 1.0}

    def __init__(self, pulses=((0.23, 0), (0.51, 1), (0.80, 2))):
        self.pulses = pulses
        self.calls = []

    def sample(self, fractions):
        self.calls.append(np.asarray(fractions).copy())
        distance = np.full((len(fractions), 3), 0.9)
        for center, pair in self.pulses:
            distance[:, pair] -= 0.85 * np.exp(-(((fractions - center) / 0.014) ** 2))
        positions = np.zeros((len(fractions), 3, 2))
        positions[..., 0] = np.asarray(fractions)[:, None]
        positions[..., 1] = [-0.2, 0.1, 0.4]
        return SimpleNamespace(pair_distances=distance, positions=positions)


class EncounterTests(unittest.TestCase):
    def test_recovers_real_pair_minima_and_midpoints(self):
        source = SyntheticSource()
        events = plan_events(source)
        self.assertEqual(len(events), 3)
        np.testing.assert_allclose(
            [event["fraction"] for event in events], [0.23, 0.51, 0.8], atol=8e-6
        )
        self.assertEqual([event["pair"] for event in events], [[0, 1], [1, 2], [2, 0]])
        for event, y in zip(events, [-0.05, 0.25, 0.1], strict=True):
            np.testing.assert_allclose(event["position"], [event["fraction"], y])
            self.assertLess(event["pair_distance"], 0.051)
            self.assertGreater(event["strength"], 0.99)
            self.assertTrue(0.003 <= event["duration"] <= 0.025)
        json.dumps(events, allow_nan=False)

    def test_fixed_lattice_and_call_count_ignore_frame_rate(self):
        a, b = SyntheticSource(), SyntheticSource()
        b.sample(np.linspace(0, 1, 901))  # A previous visualization has no effect.
        self.assertEqual(plan_events(a), plan_events(b))
        self.assertEqual(len(a.calls[0]), SAMPLE_COUNT)
        np.testing.assert_array_equal(a.calls[0], np.linspace(0, 1, SAMPLE_COUNT))
        self.assertEqual([len(call) for call in a.calls[1:]], [65, 65, 65])

    def test_competing_nearby_pairs_share_one_event(self):
        events = plan_events(SyntheticSource(((0.23, 0), (0.27, 1), (0.61, 2))))
        self.assertEqual(len(events), 2)
        self.assertTrue(events[1]["fraction"] - events[0]["fraction"] >= REFRACTORY_FRACTION)

    def test_constant_separation_and_projected_overlap_do_not_invent_blooms(self):
        source = SyntheticSource(())
        # All bodies can project onto one pixel while their actual separation
        # remains large; event selection never consults projected proximity.
        self.assertEqual(plan_events(source), [])
        self.assertEqual(plan_events(source, count=0), [])
        source.pulses = ((0.3, 0),)
        self.assertEqual(len(plan_events(source)), 1)

    def test_monotonic_approach_has_no_interior_encounter(self):
        source = SimpleNamespace(projection=SyntheticSource.projection)
        source.sample = lambda times: SimpleNamespace(
            pair_distances=np.repeat((1 - 0.98 * times)[:, None], 3, axis=1),
            positions=np.zeros((len(times), 3, 2)),
        )
        self.assertEqual(plan_events(source), [])

    def test_invalid_inputs_fail_before_a_schedule_is_used(self):
        for count in (True, -1, 13, 2.0):
            with self.assertRaises(ValueError):
                plan_events(SyntheticSource(), count)
        source = SimpleNamespace(projection=SyntheticSource.projection)
        source.sample = lambda times: SimpleNamespace(
            pair_distances=np.full((len(times), 3), np.nan),
            positions=np.zeros((len(times), 3, 2)),
        )
        with self.assertRaises(ValueError):
            plan_events(source)


if __name__ == "__main__":
    unittest.main()
