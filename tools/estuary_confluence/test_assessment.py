"""Participation measurements distinguish contact from mere field activity."""

import unittest

import numpy as np

from .assessment import assess, image_balance


class AssessmentTests(unittest.TestCase):
    def test_pure_separate_colors_have_no_contact(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        for i in range(3):
            p[:, i * 4 : i * 4 + 4, i] = 1
        result = assess(p, 3, 1)
        self.assertEqual(result["shared_painted_area_fraction"], 0)
        self.assertEqual(result["minimum_contact_mass_fraction"], 0)
        self.assertTrue(all(x["visible_mass_fraction"] == 1 for x in result["pigments"]))

    def test_uniform_three_color_mix_has_complete_contact(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        p[..., :3] = 1 / 3
        result = assess(p, 3, 1)
        self.assertEqual(result["shared_painted_area_fraction"], 1)
        self.assertEqual(result["minimum_contact_mass_fraction"], 1)
        self.assertEqual(result["paint_mass_moments"]["normalized_centroid"], [0, 0])
        for pigment in result["pigments"]:
            self.assertAlmostEqual(pigment["dominant_area_fraction"], 1 / 3)

    def test_tied_dominance_is_invariant_to_pigment_order(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        p[..., :3] = (0.4, 0.4, 0.2)
        order = [2, 0, 1, 3]
        first = assess(p, 3, 1)["pigments"]
        second = assess(p[..., order], 3, 1)["pigments"]
        self.assertEqual([r["dominant_area_fraction"] for r in first], [0.5, 0.5, 0])
        self.assertEqual(
            [r["dominant_area_fraction"] for r in second],
            [first[i]["dominant_area_fraction"] for i in order[:3]],
        )

    def test_tiny_traces_do_not_inflate_significant_contact(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        p[..., 0], p[..., 1], p[..., 2] = 0.98, 0.01, 0.01
        result = assess(p, 3, 1)
        self.assertEqual(result["shared_painted_area_fraction"], 0)
        self.assertTrue(all(x["contact_mass_fraction"] == 0 for x in result["pigments"]))

    def test_guard_mass_is_reported_without_being_counted_as_visible_contact(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        p[..., 0] = 1
        p[:3, :, 1] = 1
        result = assess(p, 3, 2)
        self.assertEqual(result["visible_resolution"], [6, 6])
        self.assertEqual(result["pigments"][0]["visible_mass_fraction"], 0.25)
        self.assertEqual(result["pigments"][1]["visible_mass_fraction"], 0)
        self.assertEqual(result["shared_painted_area_fraction"], 0)

    def test_centroid_detects_right_bias_and_mirroring_reverses_it(self):
        p = np.zeros((16, 16, 4), dtype="f4")
        p[4:12, 10:14, 0] = 1
        a = assess(p, 3, 1)["paint_mass_moments"]["normalized_centroid"]
        b = assess(p[:, ::-1], 3, 1)["paint_mass_moments"]["normalized_centroid"]
        self.assertGreater(a[0], 0.4)
        self.assertEqual(a[0], -b[0])
        self.assertEqual(a[1], 0)

    def test_no_input_mutation_and_repeated_spatial_sampling_preserves_mass(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        p[3:9, 3:9, :3] = (0.2, 0.3, 0.5)
        original = p.copy()
        a = assess(p, 3, 1)
        b = assess(np.repeat(np.repeat(p, 2, axis=0), 2, axis=1), 3, 1)
        np.testing.assert_array_equal(p, original)
        self.assertEqual([x["mass"] for x in a["pigments"]], [x["mass"] for x in b["pigments"]])
        self.assertEqual(a["minimum_contact_mass_fraction"], b["minimum_contact_mass_fraction"])

    def test_blank_image_and_empty_paint_have_no_invented_center(self):
        result = assess(np.zeros((12, 12, 4), dtype="f4"), 3, 1)
        self.assertIsNone(result["paint_mass_moments"])
        self.assertEqual(result["painted_area_fraction"], 0)
        self.assertIsNone(image_balance(np.ones((12, 12, 3), dtype="f4")))

    def test_image_balance_uses_actual_paint_contrast(self):
        image = np.ones((12, 12, 3), dtype="f4")
        image[3:9, 8:12] = 0
        self.assertGreater(image_balance(image)[0], 0.5)
        self.assertEqual(image_balance(image)[1], 0)

    def test_invalid_states_and_thresholds_are_rejected(self):
        p = np.zeros((12, 12, 4), dtype="f4")
        for count, domain, threshold in ((3.0, 1, 0.008), (3, 0, 0.008), (3, 1, float("nan"))):
            with self.subTest(count=count, domain=domain), self.assertRaises(ValueError):
                assess(p, count, domain, mass_threshold=threshold)
        p[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            assess(p, 3, 1)


if __name__ == "__main__":
    unittest.main()
