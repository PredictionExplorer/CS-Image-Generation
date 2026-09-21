"""Pattern/palette isolation and stable full-trajectory production controls."""

import copy
import unittest

from tools.estuary.initial_patterns import PATTERNS
from tools.estuary_depth import pattern_studies as studies
from tools.estuary_depth.pattern_palette import make_palette
from tools.estuary_depth.render import motion_angles


class PatternStudyTests(unittest.TestCase):
    def test_exactly_ten_distinct_motifs_share_the_same_seed_palette_and_transport(self):
        self.assertEqual(set(studies.OPTIONS), set(PATTERNS))
        self.assertEqual(len(studies.OPTIONS), 10)
        recipes = [studies.make_formation_recipe(123, name) for name in studies.OPTIONS]
        common = []
        for recipe in recipes:
            self.assertEqual(recipe["optics"], make_palette(123)["optics"])
            self.assertEqual(recipe["simulation"]["resolution"], [6144, 4608])
            self.assertEqual(recipe["simulation"]["steps"], 7200)
            self.assertEqual(recipe["simulation"]["pigment_weights"], [1, 1, 1])
            self.assertEqual(recipe["render"]["frames"], 721)
            self.assertEqual(recipe["render"]["fps"], 24)
            normalized = copy.deepcopy(recipe)
            normalized["simulation"]["initial_design"].pop("pattern")
            common.append(normalized)
        self.assertTrue(all(recipe == common[0] for recipe in common))
        self.assertEqual(
            {r["simulation"]["initial_design"]["pattern"] for r in recipes}, set(PATTERNS)
        )

    def test_seed_aliases_are_equivalent_and_high_bits_change_both_art_controls(self):
        first = studies.make_formation_recipe(17, "folded-sash")
        self.assertEqual(first, studies.make_formation_recipe("0x0011", "folded-sash"))
        changed = studies.make_formation_recipe(17 + 2**240, "folded-sash")
        self.assertNotEqual(first["optics"], changed["optics"])
        self.assertNotEqual(
            first["simulation"]["initial_design"], changed["simulation"]["initial_design"]
        )

    def test_every_movie_finishes_at_its_paired_photograph_pose(self):
        for option in studies.OPTIONS:
            photo = studies.make_photo_recipe(7, option)
            motion = studies.make_motion_recipe(7, option)
            self.assertEqual(
                motion_angles(motion["camera"], 95, 96), motion_angles(photo["camera"], 0, 1)
            )
            for key in photo.keys() - {"camera", "render"}:
                self.assertEqual(photo[key], motion[key])

    def test_returned_controls_and_public_catalogue_cannot_mutate_other_studies(self):
        reference = studies.make_formation_recipe(1, "split-fan")
        changed = copy.deepcopy(reference)
        changed["optics"]["pigments_srgb"][0][0] = 1
        self.assertEqual(studies.make_formation_recipe(1, "split-fan"), reference)
        with self.assertRaises(TypeError):
            PATTERNS["unimplemented"] = "Cannot add a renderer by changing a label"
        with self.assertRaises(ValueError):
            studies.make_formation_recipe(1, "unimplemented")


if __name__ == "__main__":
    unittest.main()
