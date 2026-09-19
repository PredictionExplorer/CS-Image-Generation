"""Initial shape geometry, source isolation and equal native pigment budgets."""

from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np

from .initial_composition import SETUPS, VERSION, plan_layout, rasterize, validate_config
from .mass_budget import pigment_mass

CONFIG = {"target_mass": [0.043, 0.018, 0.008], "reference_radii": [0.18, 0.19, 0.2]}


class InitialSource:
    aspect = 4 / 3

    def __init__(self, positions=None):
        self.positions = np.array(
            [[0.3, -0.2], [-0.4, 0.3], [0.2, 0.5]] if positions is None else positions, dtype="f8"
        )
        self.velocities = np.array([[1, 0], [0, -2], [0, 0]], dtype="f8")
        self.calls = []

    def frame(self, t):
        self.calls.append(t)
        if t != 0:
            raise AssertionError("Composition inspected trajectory after its initial instant")
        return SimpleNamespace(positions=self.positions, velocities=self.velocities)


def layout(setup="random-circles", seed="0x808861c25b6c", source=None):
    return plan_layout(seed, 3, 4 / 3, {**CONFIG, "setup": setup}, source)


class CompositionContracts(unittest.TestCase):
    def test_config_is_explicit_three_color_and_seeded(self):
        config = {**CONFIG, "setup": "random-circles"}
        resolved = validate_config(config)
        self.assertEqual(resolved["version"], VERSION)
        self.assertEqual(validate_config(resolved), resolved)
        self.assertIsNone(validate_config(None))
        for invalid in (
            {},
            {**config, "initial_load": 1},
            {**config, "target_mass": [1, 2, 3, 4, 5]},
            {**config, "target_mass": [1, True, 1]},
            {**config, "reference_radii": [0, 1, 1]},
            {**config, "setup": "unknown"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(invalid)
        for count in (1, 2, 4, 5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                plan_layout("0x1", count, 4 / 3, config)

    def test_full_seed_determinism_shared_quantiles_and_only_actual_colors(self):
        quantiles = None
        for setup in SETUPS:
            source = InitialSource() if setup == "body-wedges" else None
            a = layout(setup, "0x0001", source)
            b = layout(setup, 1, source)
            c = layout(setup, (1 << 255) + 1, source)
            self.assertEqual(a, b)
            self.assertNotEqual(a["primitives"], c["primitives"])
            self.assertEqual(json.loads(json.dumps(a)), a)
            self.assertEqual(a["count"], 3)
            self.assertEqual({p["pigment_index"] for p in a["primitives"]}, {0, 1, 2})
            self.assertEqual(len(a["placement_quantiles"]), 3)
            if setup.startswith("random-") or setup == "scattered-commas":
                first = [q[0] for q in a["placement_quantiles"]]
                if quantiles is None:
                    quantiles = first
                self.assertEqual(first, quantiles)

    def test_first_five_never_inspect_source_and_do_not_require_separation(self):
        class UnavailableSource:
            def __getattribute__(self, _):
                raise AssertionError("Non-body setup inspected the trajectory")

        for setup in SETUPS[:-1]:
            self.assertEqual(layout(setup), layout(setup, source=UnavailableSource()))
        overlapping = False
        config = {**CONFIG, "setup": "random-circles", "reference_radii": [0.35] * 3}
        for seed in range(32):
            pieces = plan_layout(seed, 3, 4 / 3, config)["primitives"]
            for a in range(3):
                for b in range(a + 1, 3):
                    distance = np.linalg.norm(np.array(pieces[a]["center"]) - pieces[b]["center"])
                    overlapping |= distance < pieces[a]["radii"][0] + pieces[b]["radii"][0]
        self.assertTrue(overlapping)

    def test_wedges_use_exact_initial_anchors_velocity_and_uniform_fitting(self):
        source = InitialSource([[1.28, 0], [0, 0.94], [-0.9, -0.4]])
        source_before = source.positions.copy(), source.velocities.copy()
        planned = layout("body-wedges", source=source)
        origin_source = InitialSource(np.zeros((3, 2)))
        origin = layout("body-wedges", source=origin_source)
        self.assertEqual(source.calls, [0.0])
        for i, p in enumerate(planned["primitives"]):
            np.testing.assert_array_equal(p["center"], source.positions[i])
            self.assertEqual(p["body_index"], i)
            self.assertEqual(p["source_fraction"], 0)
            relative = np.asarray(p["points"]) - p["center"]
            reference = np.asarray(origin["primitives"][i]["points"])
            np.testing.assert_allclose(relative, reference * p["fit_scale"], atol=2e-16)
            if i < 2:
                expected = np.arctan2(source.velocities[i, 1], source.velocities[i, 0])
                self.assertEqual(p["heading_radians"], expected)
        self.assertLess(planned["primitives"][0]["fit_scale"], 1)
        np.testing.assert_array_equal(source.positions, source_before[0])
        np.testing.assert_array_equal(source.velocities, source_before[1])
        with self.assertRaises(ValueError):
            layout("body-wedges", source=InitialSource([[4 / 3, 0], [0, 0], [0, 0]]))

    def test_shapes_have_distinct_geometry_and_three_unequal_commas_per_color(self):
        planned = {
            s: layout(s, source=InitialSource() if s == "body-wedges" else None) for s in SETUPS
        }
        counts = {s: len(p["primitives"]) for s, p in planned.items()}
        self.assertEqual(list(counts.values()), [3, 3, 3, 5, 9, 3])
        self.assertTrue(all(len(p["points"]) == 1 for p in planned["random-circles"]["primitives"]))
        self.assertTrue(all(p["kind"] == "polygon" for p in planned["body-wedges"]["primitives"]))
        for i in range(3):
            pieces = [
                p for p in planned["scattered-commas"]["primitives"] if p["pigment_index"] == i
            ]
            widths = [max(p["radii"]) for p in pieces]
            self.assertGreater(widths[0], widths[1])
            self.assertGreater(widths[1], widths[2])
        for p in planned["random-crescents"]["primitives"]:
            points = np.asarray(p["points"])
            self.assertGreater(np.linalg.norm(points[0] - points[-1]), 2 * max(p["radii"]))


class CompositionRasterTests(unittest.TestCase):
    def test_every_setup_matches_target_mass_on_multiple_native_grids(self):
        for setup in SETUPS:
            planned = layout(setup, source=InitialSource() if setup == "body-wedges" else None)
            original = copy.deepcopy(planned)
            for size in ((512, 384), (768, 576)):
                field = rasterize(planned, size, 1.6)
                self.assertEqual(field.shape, (size[1], size[0], 3))
                self.assertEqual(field.dtype, np.float32)
                self.assertTrue(np.isfinite(field).all())
                self.assertGreaterEqual(field.min(), 0)
                np.testing.assert_allclose(
                    pigment_mass(field, 1.6), CONFIG["target_mass"], rtol=2e-7
                )
                x = ((np.arange(size[0]) + 0.5) / size[0] * 2 - 1) * (4 / 3) * 1.6
                y = ((np.arange(size[1]) + 0.5) / size[1] * 2 - 1) * 1.6
                outside = (np.abs(y[:, None]) > 1) | (np.abs(x[None, :]) > 4 / 3)
                np.testing.assert_array_equal(field[outside], 0)
            self.assertEqual(planned, original)

    def test_repeated_same_color_components_accumulate_before_normalization(self):
        planned = layout()
        primitive = copy.deepcopy(planned["primitives"][0])
        primitive.update(points=[[-0.4, 0]], center=[-0.4, 0], radii=[0.18])
        second = copy.deepcopy(primitive)
        third = copy.deepcopy(primitive)
        third.update(points=[[0.4, 0]], center=[0.4, 0])
        planned["primitives"] = [primitive, second, third, *planned["primitives"][1:]]
        field = rasterize(planned, (640, 480), 1.6)
        self.assertEqual(field[240, 260, 0], 2 * field[240, 380, 0])
        np.testing.assert_allclose(pigment_mass(field, 1.6), CONFIG["target_mass"], rtol=2e-7)

    def test_invalid_layouts_unresolved_tips_and_mismatched_grid_fail(self):
        planned = layout("random-ribbons")
        with self.assertRaisesRegex(ValueError, "finer"):
            rasterize(planned, (64, 48), 1.6)
        with self.assertRaisesRegex(ValueError, "aspect"):
            rasterize(planned, (512, 512), 1.6)
        for mutate in (
            lambda p: p.update(count=5),
            lambda p: p["primitives"][0].update(pigment_index=4),
            lambda p: p["primitives"][0].update(points=[[1e308, 0]], radii=[0.1]),
            lambda p: p["primitives"][0].update(points=[[0, 0], [0, 0]], radii=[0.1, 0.1]),
            lambda p: p["primitives"][0].update(radii=[float("nan")] * 65),
        ):
            bad = copy.deepcopy(planned)
            mutate(bad)
            with self.assertRaises(ValueError):
                rasterize(bad, (512, 384), 1.6)
        polygon = layout("body-wedges", source=InitialSource())
        polygon["primitives"][0]["points"][-1] = polygon["primitives"][0]["points"][0]
        with self.assertRaisesRegex(ValueError, "closing"):
            rasterize(polygon, (512, 384), 1.6)


if __name__ == "__main__":
    unittest.main()
