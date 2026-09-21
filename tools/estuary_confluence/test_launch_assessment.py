"""Launch proxies distinguish sampled movement from positional coincidence."""

import copy
import json
import unittest

import numpy as np

from .launch_assessment import DEFAULTS, SOURCE_FRACTIONS, LaunchAssessment, validate_config


class LaunchAssessmentTests(unittest.TestCase):
    def patch(self):
        field = np.zeros((64, 64, 2), dtype="f4")
        field[20:36, 12:28, 0] = 1
        return field

    def collector(self, field, count=1, **kwargs):
        return LaunchAssessment(field, count, 1, config={"footprint_margin": 0}, **kwargs)

    def test_static_paint_has_full_retention_and_preserves_inputs_and_source_clock(self):
        initial = self.patch()
        original = initial.copy()
        source = {"seed": "0x1", "sha256": "a" * 64}
        collector = self.collector(initial, source_metadata=source)
        for fraction in SOURCE_FRACTIONS[1:]:
            collector.sample(initial, fraction)
        source["seed"] = "changed"
        report = collector.report()
        pigment = report["pigments"][0]
        self.assertEqual(report["source"]["seed"], "0x1")
        self.assertEqual(report["source_clock"]["sample_fractions"], list(SOURCE_FRACTIONS))
        self.assertEqual(pigment["retained_initial_footprint_fraction"], 1)
        self.assertEqual(pigment["retained_low_change_footprint_fraction"], 1)
        self.assertEqual(pigment["initial_mass"], pigment["final_mass"])
        self.assertEqual(pigment["visible_centroid_displacement_world"], 0)
        self.assertEqual(pigment["visible_rms_radius_ratio"], 1)
        np.testing.assert_array_equal(initial, original)
        json.dumps(report, allow_nan=False)
        report["samples"][0]["source_fraction"] = 0.5
        self.assertEqual(collector.report()["samples"][0]["source_fraction"], 0)

    def test_translation_deformation_and_uniform_overlap_are_distinct_proxies(self):
        initial = self.patch()
        collector = self.collector(initial)
        for fraction, shift in zip(SOURCE_FRACTIONS[1:], (4, 8, 12, 16), strict=True):
            collector.sample(np.roll(initial, shift, axis=1), fraction)
        moved = collector.report()["pigments"][0]
        self.assertAlmostEqual(moved["visible_centroid_displacement_world"], 0.5)
        self.assertEqual(moved["visible_rms_radius_ratio"], 1)
        self.assertEqual(moved["retained_initial_footprint_fraction"], 0)
        # A uniform patch can move while overlapping concentrations stay constant.
        small_move = self.collector(initial)
        small_move.sample(np.roll(initial, 2, axis=1), 1)
        overlap = small_move.report()["pigments"][0]
        self.assertGreater(overlap["retained_low_change_footprint_fraction"], 0.8)
        self.assertGreater(overlap["visible_centroid_displacement_world"], 0)
        self.assertIn("Uniform flow", small_move.report()["limitation"])
        stretched = np.zeros_like(initial)
        stretched[24:32, 4:36, 0] = 1
        deformation = self.collector(initial)
        deformation.sample(stretched, 1)
        changed = deformation.report()["pigments"][0]
        self.assertEqual(changed["visible_centroid_displacement_world"], 0)
        self.assertGreater(changed["visible_rms_radius_ratio"], 1.4)
        self.assertAlmostEqual(
            changed["final_components"]["meaningful_components"][0]["covariance_aspect"], 4
        )

    def test_leave_and_return_is_not_mistaken_for_low_change_if_sampled(self):
        initial = self.patch()
        collector = self.collector(initial)
        collector.sample(np.roll(initial, 20, axis=1), 0.3)
        collector.sample(initial, 1)
        report = collector.report()
        pigment = report["pigments"][0]
        self.assertEqual(report["source_clock"]["sample_fractions"], [0, 0.3, 1])
        self.assertEqual(pigment["retained_initial_footprint_fraction"], 1)
        self.assertEqual(pigment["retained_low_change_footprint_fraction"], 0)
        self.assertEqual(pigment["visible_centroid_displacement_world"], 0)
        self.assertEqual(pigment["sampled_cumulative_change_per_initial_visible_mass"], 4)

    def test_contact_and_meaningful_component_moments_use_native_support_not_perimeter(self):
        initial = self.patch()
        initial[50, 50, 0] = 1  # Sub-percent detached speck is recorded, not a main shape.
        collector = self.collector(initial)
        collector.sample(initial, 1)
        coarse = collector.report()
        native = np.repeat(np.repeat(initial, 2, axis=0), 2, axis=1)
        fine = collector.report(final_native=native)
        self.assertEqual(fine["sample_grid"], [64, 64])
        self.assertEqual(fine["final_grid"], [128, 128])
        a, b = coarse["final_silhouette"], fine["final_silhouette"]
        self.assertEqual(a["component_count"], 2)
        self.assertEqual(len(a["meaningful_components"]), 1)
        self.assertEqual(a, b)
        mixed = np.zeros((64, 64, 4), dtype="f4")
        mixed[20:40, 20:40, :3] = 0.3
        together = self.collector(mixed, 3)
        together.sample(mixed, 1)
        self.assertTrue(
            all(row["final_contact_mass_fraction"] == 1 for row in together.report()["pigments"])
        )

    def test_guard_mass_and_empty_channels_are_not_invented_visible_components(self):
        field = np.zeros((64, 64, 4), dtype="f4")
        field[:8, :, 0] = 1
        field[25:35, 25:35, 1] = 1
        collector = LaunchAssessment(field, 3, 2)
        collector.sample(field, 1)
        rows = collector.report()["pigments"]
        self.assertGreater(rows[0]["initial_mass"], 0)
        self.assertEqual(rows[0]["final_visible_mass_fraction"], 0)
        self.assertEqual(rows[0]["final_components"]["component_count"], 0)
        self.assertEqual(rows[2]["final_mass"], 0)
        self.assertIsNone(rows[2]["visible_centroid_displacement_world"])
        self.assertIsNone(rows[2]["sampled_cumulative_change_per_initial_visible_mass"])

    def test_validation_rejects_bad_config_grid_clock_and_incomplete_reports(self):
        self.assertIsNone(validate_config(None))
        self.assertEqual(validate_config({}), DEFAULTS)
        for value in (True, [], {"version": "other"}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                validate_config(value)
        for key in DEFAULTS.keys() - {"version"}:
            for value in (True, -1, float("nan"), float("inf"), 10**400):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config({key: value})
        initial = self.patch()
        collector = self.collector(initial)
        with self.assertRaises(ValueError):
            collector.report()
        for fraction in (True, -1, 0, 2, float("nan"), float("inf"), 10**400):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                collector.sample(initial, fraction)
        with self.assertRaises(ValueError):
            collector.sample(initial[::2, ::2], 0.5)
        invalid = copy.deepcopy(initial)
        invalid[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            collector.sample(invalid, 0.5)
        collector.sample(initial, 1)
        with self.assertRaises(ValueError):
            collector.report(np.zeros((64, 32, 2)))
        with self.assertRaises(ValueError):
            collector.sample(initial, 1)


if __name__ == "__main__":
    unittest.main()
