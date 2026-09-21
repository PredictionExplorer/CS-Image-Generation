"""Geometry, actual-component participation and immutable choreography contracts."""

from __future__ import annotations

import copy
import json
import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import choreography as planner
from .palette import generate_palette


def config(setup="active-pools"):
    return {
        "initial_choreography": {
            "version": planner.VERSION,
            "setup": setup,
            "target_mass": [0.024, 0.012, 0.006],
            "reference_radii": [0.12, 0.12, 0.12],
        },
        "lower_transport_scale": 0.82,
    }


def source():
    return SimpleNamespace(
        seed="0xabcd",
        aspect=4 / 3,
        sha256="a" * 64,
        projection={"aspect": 4 / 3, "method": "analytic test"},
    )


def prepared():
    flow = {
        "aspect": 4 / 3,
        "stir_radius": 0.2,
        "flow_strength": 1.4,
        "pair_swirl": 0.12,
        "domain_scale": 1.0,
        "carrier_velocity": [0, 0],
    }
    return (
        flow,
        np.zeros((planner.PILOT_STEPS, 3, 4)),
        np.zeros((planner.PILOT_STEPS, 3, 3)),
        None,
        np.array([[-0.27, 0], [0, 0], [0.27, 0]]),
        [
            (1.0, np.array([x, y]), 0.0, math.pi / 2)
            for x, y in [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
        ],
        "b" * 64,
    )


def compress(points, *_args, **_kwargs):
    x, y = np.asarray(points)[..., 0], np.asarray(points)[..., 1]
    # Curl of -2*x*y*(1-x*x/a*a)^2*(1-y*y)^2: central compression
    # with a no-through-flow boundary, unlike an unbounded affine saddle.
    ex, ey = np.maximum(1 - (x / (4 / 3)) ** 2, 0), np.maximum(1 - y * y, 0)
    return np.stack(
        [-2 * x * ex**2 * ey * (1 - 5 * y * y), 2 * y * ey**2 * ex * (1 - 5 * (x / (4 / 3)) ** 2)],
        axis=-1,
    )


class ChoreographyContracts(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        for name, value in (
            ("PILOT_STEPS", 64),
            ("PARTICLES_PER_COMPONENT", 16),
            ("OBSERVATION_STRIDE", 8),
            ("CANDIDATE_LIMIT", 12),
        ):
            self.context.enter_context(patch.object(planner, name, value))
        planner._LAYOUTS.clear()
        planner._PREPARED.clear()
        self.addCleanup(planner._LAYOUTS.clear)
        self.addCleanup(planner._PREPARED.clear)
        self.palette = generate_palette("abcd", 3, mode="composed")
        self.context.enter_context(
            patch.object(planner, "_source_data", side_effect=lambda *_: prepared())
        )
        self.context.enter_context(patch.object(planner, "velocity", side_effect=compress))

    def plan(self, setup="active-pools", settings=None):
        return planner.plan_layout(source(), settings or config(setup), self.palette)

    def test_all_eight_geometry_families_qualify_every_actual_component(self):
        for setup in planner.SETUPS:
            with self.subTest(setup=setup):
                layout = self.plan(setup)
                self.assertEqual({p["pigment_index"] for p in layout["primitives"]}, {0, 1, 2})
                self.assertEqual(len(layout["primitives"]), 4 if setup == "split-lobes" else 3)
                self.assertTrue(
                    all(p["eligible"] for p in layout["pilot"]["selection"]["components"])
                )
                self.assertLessEqual(layout["pilot"]["evaluated_candidates"], 12)
                self.assertEqual(layout, json.loads(json.dumps(layout, allow_nan=False)))
                planner.validate_layout(layout)

    def test_repeatability_and_cache_cannot_be_mutated_by_callers(self):
        a = self.plan()
        original = copy.deepcopy(a)
        a["primitives"][0]["points"][0][0] = 100
        with patch.object(planner, "_pilot", side_effect=AssertionError("pilot repeated")):
            b = self.plan()
        self.assertEqual(b, original)

    def test_mobility_changes_only_declared_layer_allocation(self):
        baseline = self.plan("stretch-ovals")
        settings = config("stretch-ovals")
        settings["initial_choreography"]["mobility_bias"] = [-0.12, 0.12, 0]
        palette_before = copy.deepcopy(self.palette)
        with patch.object(planner, "_pilot", side_effect=AssertionError("bias reran pilot")):
            changed = self.plan(settings=settings)
        self.assertEqual(changed["primitives"], baseline["primitives"])
        self.assertEqual(changed["pilot"], baseline["pilot"])
        self.assertEqual(changed["baseline_layer_fractions"], baseline["baseline_layer_fractions"])
        self.assertNotEqual(
            changed["effective_layer_fractions"], baseline["effective_layer_fractions"]
        )
        self.assertEqual(self.palette, palette_before)
        np.testing.assert_array_equal(
            planner.rasterize(changed, [1024, 768], 1.6),
            planner.rasterize(baseline, [1024, 768], 1.6),
        )

    def test_native_mass_and_concentration_bound_for_every_family(self):
        for setup in planner.SETUPS:
            with self.subTest(setup=setup):
                layout = self.plan(setup)
                image = planner.rasterize(layout, [1024, 768], 1.6)
                self.assertEqual(image.dtype, np.float32)
                self.assertTrue(np.isfinite(image).all())
                self.assertGreaterEqual(float(image.min()), 0)
                mass = image.sum(axis=(0, 1), dtype="f8") * (3.2 / 768) ** 2
                np.testing.assert_allclose(mass, layout["target_mass"], rtol=5e-7, atol=1e-12)
                self.assertTrue(
                    np.all(image.max(axis=(0, 1)) <= layout["peak_concentration_limits"])
                )

    def test_split_lobes_realize_separate_seventy_thirty_budgets(self):
        layout = self.plan("split-lobes")
        image = planner.rasterize(layout, [1024, 768], 1.6)
        parts = [p for p in layout["primitives"] if p["pigment_index"] == 0]
        np.testing.assert_allclose([p["target_mass"] for p in parts], [0.024 * 0.7, 0.024 * 0.3])
        x = ((np.arange(1024) + 0.5) / 1024 * 2 - 1) * (4 / 3) * 1.6
        y = ((np.arange(768) + 0.5) / 768 * 2 - 1) * 1.6
        xx, yy = np.meshgrid(x, y)
        distances = [
            np.hypot(xx - p["points"][0][0], yy - p["points"][0][1]) - p["radii"][0] for p in parts
        ]
        attribution = np.argmin(distances, axis=0)
        for index, p in enumerate(parts):
            actual = image[..., 0][attribution == index].sum(dtype="f8") * (3.2 / 768) ** 2
            self.assertAlmostEqual(actual / p["target_mass"], 1, places=6)

    def test_finite_time_oval_direction_tracks_anisotropic_flow(self):
        directions = planner._starting_directions(np.array([[[0, 0]]]), prepared())
        np.testing.assert_allclose(np.abs(np.sin(directions)), 1, atol=1e-10)
        with patch.object(
            planner, "velocity", side_effect=lambda p, *_a, **_k: np.asarray(p) * [1.3, -1.3]
        ):
            rotated = planner._starting_directions(np.array([[[0, 0]]]), prepared())
        np.testing.assert_allclose(np.abs(np.cos(rotated)), 1, atol=1e-10)

    def test_rigid_oval_rotation_is_not_deformation(self):
        points, _ = planner._samples(planner._ellipse([0, 0], 0.2, 0, 2.5))
        weights = np.full(len(points), 1 / len(points))
        centered = points - points.mean(axis=0)
        covariance = np.einsum("ni,nj->ij", centered, centered) / len(points)
        rotated = np.stack([-points[:, 1], points[:, 0]], axis=1) + np.array([0.1, -0.1])
        stretch, _ = planner._stretch(rotated, weights, covariance, points)
        self.assertAlmostEqual(float(stretch), 1, places=12)
        deformed = points * [2, 0.5]
        stretch, _ = planner._stretch(deformed, weights, covariance, points)
        self.assertAlmostEqual(float(stretch), 2, places=12)

    def test_stationary_and_rigid_translation_fail_honestly(self):
        for field in (
            lambda p, *_a, **_k: np.zeros_like(p),
            lambda p, *_a, **_k: np.broadcast_to([0.2, 0], p.shape),
        ):
            with self.subTest(field=field), patch.object(planner, "velocity", side_effect=field):
                planner._LAYOUTS.clear()
                with self.assertRaisesRegex(ValueError, "No qualified choreography"):
                    self.plan()

    def test_remote_split_lobe_cannot_hide_behind_active_same_color_lobe(self):
        candidates, _ = planner._candidates(
            config("split-lobes")["initial_choreography"], prepared()
        )
        candidate = candidates[0]
        remote = next(p for p in candidate if p["pigment_index"] == 0 and p["component"] == 1)
        remote["points"] = [[1.1, 0.65]]

        def active_center(points, *_a, **_k):
            return np.where((np.abs(points[..., :1]) < 0.9), compress(points), 0)

        with patch.object(planner, "velocity", side_effect=active_center):
            reports = planner._pilot([candidate], prepared(), np.array([0.7, 0.3, 0.5]), 0.82)
        self.assertFalse(reports[0]["eligible"])
        row = next(
            r for r in reports[0]["components"] if r["pigment_index"] == 0 and r["component"] == 1
        )
        self.assertFalse(row["eligible"])
        self.assertEqual(row["moved_mass_fraction"], 0)

    def test_portable_layout_rejects_forged_missing_or_boolean_gate_metrics(self):
        layout = self.plan()
        for name in (
            "maximum_rms_displacement_in_radii",
            "peak_stretch",
            "final_stretch",
            "moved_mass_fraction",
            "neighbor_exposure",
            "contacted_mass_fraction",
        ):
            for bad in (None, 0, True, float("nan")):
                with self.subTest(name=name, bad=bad):
                    changed = copy.deepcopy(layout)
                    row = changed["pilot"]["selection"]["components"][0]
                    if bad is None:
                        row.pop(name)
                    else:
                        row[name] = bad
                    with self.assertRaises(ValueError):
                        planner.validate_layout(changed)
        for mutate in (
            lambda d: d["pilot"].update(steps=31),
            lambda d: d["pilot"].update(source_settings_sha256="bad"),
            lambda d: d["pilot"]["selection"].update(shape_tiebreak=0),
            lambda d: d["pilot"]["selection"].update(minimum_participation_score=True),
            lambda d: d["primitives"][0].update(target_mass=0.1),
            lambda d: d.update(extra="unsupported"),
        ):
            changed = copy.deepcopy(layout)
            mutate(changed)
            with self.assertRaises(ValueError):
                planner.validate_layout(changed)


class ChoreographyConfigTests(unittest.TestCase):
    def test_config_rejects_unknown_nonfinite_boolean_and_bad_counts(self):
        good = config()["initial_choreography"]
        self.assertIsNone(planner.validate_config(None))
        self.assertEqual(planner.validate_config(good), good)
        self.assertEqual(planner.validate_config({**good, "mobility_bias": [0, 0, 0]}), good)
        for bad in (
            {},
            {**good, "extra": 1},
            {**good, "setup": []},
            {**good, "target_mass": [True, 1, 1]},
            {**good, "reference_radii": [0.1, float("nan"), 0.1]},
            {**good, "mobility_bias": [0, 0]},
            {**good, "mobility_bias": [0.16, 0, 0]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                planner.validate_config(bad)

    def test_bias_range_is_checked_without_touching_chalk_or_palette(self):
        p = {"layer_fractions": [0.7, 0.22, 0.55, 0.05]}
        before = copy.deepcopy(p)
        cfg = config()["initial_choreography"]
        effective = planner.effective_layer_fractions(p, {**cfg, "mobility_bias": [-0.12, 0.12, 0]})
        np.testing.assert_allclose(effective, [0.58, 0.34, 0.55, 0.05], atol=1e-7)
        self.assertEqual(p, before)
        with self.assertRaisesRegex(ValueError, "upper layer fractions"):
            planner.effective_layer_fractions(p, {**cfg, "mobility_bias": [0, -0.12, 0]})


if __name__ == "__main__":
    unittest.main()
