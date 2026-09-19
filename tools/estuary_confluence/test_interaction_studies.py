"""Texture recipes preserve the exact accepted artistic inputs."""

import unittest

from .interaction_studies import LOOKS, PROFILES, make_plan, released_case, texture_recipe


class InteractionStudyTests(unittest.TestCase):
    def test_every_accepted_count_and_seed_reconstructs_exactly(self):
        from tools.estuary_studio.common import read

        from .run import ROOT

        for case in read(ROOT / "releases/color-and-form-v1.json")["cases"]:
            actual, recipe = released_case(case["seed"], case["chromatic_count"])
            self.assertEqual(actual, case)
            self.assertEqual(recipe["simulation"]["resolution"], [2048, 1536])

    def test_profiles_change_only_history_and_appearance_not_pigment_controls(self):
        for count in (1, 2, 3):
            _, base = released_case("0xb7f327f9f722", count)
            for profile in PROFILES:
                recipe = texture_recipe("0xb7f327f9f722", count, profile)
                self.assertEqual(
                    {k: v for k, v in recipe["simulation"].items() if k != "interaction"},
                    base["simulation"],
                )
                self.assertEqual(recipe["projection"], base["projection"])
                self.assertEqual(recipe["surface"]["grain_um"], 0)
                self.assertEqual(recipe["surface"]["ground_srgb"], base["surface"]["ground_srgb"])
                self.assertEqual(recipe["looks"], LOOKS)
                self.assertEqual(recipe["render"]["formation_frames"], 721)

    def test_native_acceptance_hash_is_not_claimed_for_different_resolution(self):
        pairs = [("0xb7f327f9f722", 3)]
        native = make_plan(pairs, ["gentle"])
        coarse = make_plan(pairs, ["gentle"], width=1024)
        self.assertIsNotNone(native["cases"][0]["accepted_base_material_sha256"])
        self.assertIsNone(coarse["cases"][0]["accepted_base_material_sha256"])
        self.assertEqual(native, make_plan(pairs, ["gentle"]))
