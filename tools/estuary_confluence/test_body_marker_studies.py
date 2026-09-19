"""The body-position experiment changes presentation, never RC1 material."""

import unittest

from tools.estuary_studio.common import read

from .body_marker_studies import RELEASE, make_plan, released_recipe


class BodyMarkerStudyTests(unittest.TestCase):
    def test_all_ten_recipes_reconstruct_rc1_exactly(self):
        release = read(RELEASE)
        self.assertEqual(len(release["cases"]), 10)
        for accepted in release["cases"]:
            with self.subTest(seed=accepted["seed"]):
                actual, recipe = released_recipe(accepted["seed"])
                self.assertEqual(actual, accepted)
                self.assertNotIn("body_markers", recipe["render"])

    def test_markers_preserve_every_material_and_camera_control(self):
        plan = make_plan()
        self.assertEqual(len(plan["cases"]), 10)
        for case in plan["cases"]:
            with self.subTest(seed=case["seed"]):
                accepted, before = released_recipe(case["seed"])
                after = case["recipe"]
                for key in before.keys() - {"name", "looks", "render"}:
                    self.assertEqual(before[key], after[key])
                self.assertEqual(
                    before["render"],
                    {key: value for key, value in after["render"].items() if key != "body_markers"},
                )
                self.assertEqual(
                    case["accepted_physical_state_sha256"], accepted["physical_state_sha256"]
                )
                self.assertEqual(case["source_sha256"], accepted["source_sha256"])

    def test_unaccepted_or_duplicate_seeds_are_rejected(self):
        for seeds in ([], ["0x1"], ["0x808861c25b6c"] * 2):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                make_plan(seeds)


if __name__ == "__main__":
    unittest.main()
