"""Source-aware placement, prefix identity, and honest bounded pilot contracts."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.estuary.flow_reference import velocity

from . import participation_layout as planner
from .layout import plan_layout


class Orbit:
    def __init__(self, seed="0xabcd", shift=0.0, aspect=4 / 3, moving=True):
        self.seed, self.shift, self.aspect, self.moving = seed, shift, aspect, moving
        self.sha256 = hashlib.sha256(f"{shift}/{aspect}/{moving}".encode()).hexdigest()
        self.projection = {"aspect": aspect, "shift": shift, "method": "test analytic orbit"}
        self.calls = 0

    def sample(self, fractions):
        self.calls += 1
        t = np.asarray(fractions)[:, None]
        angle = 2 * math.pi * (t + np.arange(3)[None] / 3)
        positions = np.stack([0.42 * np.cos(angle) + self.shift, 0.4 * np.sin(angle)], axis=-1)
        velocities = np.stack(
            [-0.84 * math.pi * np.sin(angle), 0.8 * math.pi * np.cos(angle)], axis=-1
        )
        if not self.moving:
            velocities.fill(0)
        z = 0.1 * np.sin(angle * 0.7)
        distances = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            distances.append(
                np.sqrt(
                    np.sum((positions[:, a] - positions[:, b]) ** 2, axis=1)
                    + (z[:, a] - z[:, b]) ** 2
                )
            )
        return SimpleNamespace(
            positions=positions, velocities=velocities, pair_distances=np.stack(distances, axis=-1)
        )

    def frame(self, fraction):
        frame = self.sample([fraction])
        return SimpleNamespace(
            positions=frame.positions[0],
            velocities=frame.velocities[0],
            pair_distances=frame.pair_distances[0],
        )


def config():
    return {
        "carrier_velocity": [0, 0],
        "flow_domain_scale": 1,
        "stir_radius": 0.22,
        "flow_strength": 1.1,
        "pair_swirl": 0.9,
        "load_radius": 0.28,
        "initial_load": 0.18,
        "initial_edge_width": 0.02,
    }


class ParticipationLayoutTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        # Geometry and archive contracts need no production-sized pilot. Actual
        # 1024-step pilots over the three recorded sources are checked on server.
        for name, value in (("PILOT_STEPS", 96), ("CANDIDATE_LAYOUTS", 6)):
            self.context.enter_context(patch.object(planner, name, value))

    def test_repeatability_full_seed_entropy_and_json_round_trip(self):
        source = Orbit()
        a = planner.plan_engaged_layout(source, 5, config())
        self.assertEqual(a, planner.plan_engaged_layout(source, 5, config()))
        self.assertEqual(a, json.loads(json.dumps(a, allow_nan=False)))
        self.assertEqual(a, planner.plan_engaged_layout(Orbit(seed="0X000ABCD"), 5, config()))
        other = Orbit(seed=hex((1 << 255) | 0xABCD))
        self.assertNotEqual(a["pools"], planner.plan_engaged_layout(other, 5, config())["pools"])

    def test_first_three_and_all_pilot_metadata_are_identical_across_counts(self):
        three = planner.plan_engaged_layout(Orbit(), 3, config())
        five = planner.plan_engaged_layout(Orbit(), 5, config())
        self.assertEqual(three["pools"], five["pools"][:3])
        self.assertEqual(three["pilot"], five["pilot"])
        self.assertEqual(len(five["pilot"]["selection"]["prefix_three"]["per_pool"]), 3)
        self.assertEqual(len(five["pilot"]["selection"]["all_five"]["per_pool"]), 5)

    def test_v2_keeps_the_original_placement_random_stream(self):
        seed = int("abcd", 16).to_bytes(32, "big")
        label = "placement/30/3/0/anchor"
        digest = hashlib.sha256(
            b"engaged-pigment-layout-v1\0" + seed + b"\0" + label.encode()
        ).digest()
        expected = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
        self.assertEqual(planner._unit(seed, label), expected)
        result = planner.plan_engaged_layout(Orbit(), 5, config())
        self.assertEqual(result["version"], "engaged-pigment-layout-v2")
        self.assertEqual(result["seed_namespace"], "engaged-pigment-layout-v1")

    def test_area_tracers_represent_the_interior_and_have_correct_covariance(self):
        offsets = planner._particle_offsets()
        radii = np.linalg.norm(offsets, axis=1)
        self.assertEqual(offsets.shape, (17, 2))
        self.assertEqual(np.count_nonzero(radii < 0.6), 9)
        np.testing.assert_allclose(offsets.mean(axis=0), [0, 0], atol=1e-16)
        np.testing.assert_allclose(offsets.T @ offsets / len(offsets), np.eye(2) * 0.25, atol=0.001)

    def test_one_touching_tracer_does_not_make_the_whole_pool_involved(self):
        points = np.empty((1, 5, 17, 2))
        points[:, 0] = [1, 1]
        points[:, 0, 0] = [0, 0]
        for index in range(1, 5):
            points[:, index] = [10 * (index - 1), 10 * (index - 1)]
        proximity = planner._tracer_proximity(points, np.ones((1, 5)))
        directed = proximity.max(axis=-1).mean(axis=-1)
        self.assertAlmostEqual(float(directed[0, 0]), 1 / 17, places=9)
        self.assertAlmostEqual(float(directed[0, 1]), 1)
        self.assertLess(float(directed[0, 0]), 0.06)

    def test_each_tracer_can_meet_a_different_neighbor(self):
        points = np.zeros((1, 5, 17, 2))
        points[:, 0, 8:] = [1, 1]
        points[:, 2] = [1, 1]
        points[:, 3] = [10, 10]
        points[:, 4] = [20, 20]
        proximity = planner._tracer_proximity(points, np.ones((1, 5)))
        directed = proximity[:, :3, :, :3].max(axis=-1).mean(axis=-1)
        self.assertAlmostEqual(float(directed[0, 0]), 1)

    def test_real_source_geometry_changes_positions_without_changing_seed(self):
        a = planner.plan_engaged_layout(Orbit(), 5, config())
        b = planner.plan_engaged_layout(Orbit(shift=0.20), 5, config())
        self.assertNotEqual(a["pools"], b["pools"])
        self.assertNotEqual(a["source_sha256"], b["source_sha256"])
        self.assertNotEqual(a["source_projection"], b["source_projection"])

    def test_pools_are_separate_visible_source_eligible_and_form_near_contact_prefixes(self):
        source = Orbit()
        plan = planner.plan_engaged_layout(source, 5, config())
        anchors = source.sample(np.linspace(0, 1, 1025)).positions.reshape(-1, 2)
        for index, pool in enumerate(plan["pools"]):
            x, y = pool["position"]
            self.assertLess(abs(x) + pool["radius"], source.aspect)
            self.assertLess(abs(y) + pool["radius"], 1)
            # Centers are drawn near actual body or pair activity. Ring orbits'
            # pair midpoints lie inside the body circle, covered by this bound.
            distance = float(np.linalg.norm(anchors - [x, y], axis=1).min())
            self.assertLess(distance, pool["radius"] + config()["stir_radius"] * 1.5)
            if index:
                gaps = [
                    math.dist(pool["position"], p["position"]) - pool["radius"] - p["radius"]
                    for p in plan["pools"][:index]
                ]
                self.assertGreaterEqual(min(gaps), 0.008 - 1e-12)
                self.assertLessEqual(min(gaps), 0.35 * config()["stir_radius"] + 1e-12)
        masses = [p["load"] * p["radius"] ** 2 for p in plan["pools"]]
        self.assertGreater(min(masses) / max(masses), 0.5)

    def test_pilot_samples_source_once_and_ignores_diffusion_paint_grid_and_video_choices(self):
        source = Orbit()
        first = planner.plan_engaged_layout(source, 5, config())
        self.assertEqual(source.calls, 1)
        settings = {
            **config(),
            "diffusion": 0.01,
            "resolution": [8192, 6144],
            "steps": 7200,
            "spectral_mode": "another",
            "fps": 60,
        }
        self.assertEqual(first, planner.plan_engaged_layout(Orbit(), 5, settings))

    def test_flow_helper_preserves_conditioned_physical_three_dimensional_distance(self):
        source = Orbit()
        frame = source.frame(0.4)
        tools, pairs = planner.conditioned_uniforms(frame, 0.22)
        self.assertLessEqual(float(np.linalg.norm(tools[:, 2:], axis=1).max()), 24)
        self.assertLessEqual(float(np.abs(pairs[:, 2]).max()), 20)
        altered = SimpleNamespace(**vars(frame))
        altered.pair_distances = frame.pair_distances * 5
        _, distant = planner.conditioned_uniforms(altered, 0.22)
        self.assertTrue(np.all(np.abs(distant[:, 2]) < np.abs(pairs[:, 2])))
        points = np.array([[0.0, 0.0], [0.1, -0.1], [4 / 3, 0.1]])
        expected = velocity(
            points,
            tools,
            pairs,
            aspect=source.aspect,
            stir_radius=0.22,
            flow_strength=1.1,
            pair_swirl=0.9,
            domain_scale=1,
            carrier_velocity=(0, 0),
        )
        np.testing.assert_array_equal(
            planner.flow_velocity(source, points, 0.4, config()), expected
        )
        np.testing.assert_allclose(expected[-1], [0, 0], atol=1e-15)

    def test_no_active_source_fails_clearly_and_failed_selection_is_not_disguised(self):
        with self.assertRaisesRegex(ValueError, "no active source stirring"):
            planner.plan_engaged_layout(Orbit(moving=False), 5, config())
        with (
            patch.object(planner, "MAX_PLACEMENT_ATTEMPTS", 0),
            self.assertRaisesRegex(ValueError, "bounded search"),
        ):
            planner.plan_engaged_layout(Orbit(), 5, config())
        with patch.object(
            planner, "velocity", side_effect=lambda points, *_args, **_kwargs: np.zeros_like(points)
        ):
            result = planner.plan_engaged_layout(Orbit(), 5, config())
        self.assertFalse(result["pilot"]["selection"]["eligible"])
        self.assertEqual(result["pilot"]["eligible_candidates"], 0)
        self.assertTrue(
            all(not p["eligible"] for p in result["pilot"]["selection"]["all_five"]["per_pool"])
        )
        for pool in result["pilot"]["selection"]["all_five"]["per_pool"]:
            self.assertAlmostEqual(pool["maximum_stretch"], 1, places=8)

    def test_invalid_inputs_fail_without_silent_clamping(self):
        for count in (True, 0, 4):
            with self.assertRaises(ValueError):
                planner.plan_engaged_layout(Orbit(), count, config())
        for key, value in (
            ("stir_radius", 0),
            ("load_radius", 0.35),
            ("initial_load", 0),
            ("carrier_velocity", [0, float("nan")]),
        ):
            with self.assertRaises(ValueError):
                planner.plan_engaged_layout(Orbit(), 5, {**config(), key: value})

    def test_legacy_layout_output_is_not_reconfigured_by_the_new_planner(self):
        before = plan_layout("0xabcd", 5, 4 / 3)
        planner.plan_engaged_layout(Orbit(), 5, config())
        self.assertEqual(before, plan_layout("0xabcd", 5, 4 / 3))


if __name__ == "__main__":
    unittest.main()
