"""Experiment plans isolate artistic factors and retain full-source timing."""

import copy
import tempfile
import unittest
from pathlib import Path

from .run import frame_plan
from .studies import execute_plan, make_plan, study_recipe


class StudyPlanTests(unittest.TestCase):
    def test_count_comparison_keeps_common_loads_flow_and_clock(self):
        recipes = [study_recipe(count, "original") for count in (1, 2, 3, 5)]
        for recipe in recipes:
            count = recipe["chromatic_count"]
            self.assertEqual(
                recipe["simulation"]["initial_pigment_weights"], [2.2, 0.9, 0.5, 0.6, 0.4][:count]
            )
            for key in (
                "steps",
                "pair_swirl",
                "pair_strain",
                "stir_radius",
                "initial_load",
                "load_radius",
            ):
                self.assertEqual(recipe["simulation"][key], recipes[-1]["simulation"][key])
            frames = frame_plan(recipe)
            self.assertEqual(frames[0]["source_fraction"], 0)
            self.assertEqual(frames[-1]["source_fraction"], 1)
            self.assertEqual(frames[-1]["step"], 7200)
            self.assertEqual(len(frames), 937)

    def test_flow_changes_do_not_change_palette_or_mass_controls(self):
        original = study_recipe(3, "original")
        folded = study_recipe(3, "fold")
        self.assertEqual(folded["simulation"]["pair_strain"], 0.7)
        self.assertEqual(folded["simulation"]["pair_swirl"], 0.15)
        for key in ("initial_pigment_weights", "initial_load", "load_radius", "steps"):
            self.assertEqual(original["simulation"][key], folded["simulation"][key])
        self.assertEqual(original["surface"], folded["surface"])
        self.assertEqual(original["palette_mode"], folded["palette_mode"])
        self.assertEqual(original["projection"], folded["projection"])
        folded["simulation"]["initial_pigment_weights"][0] = 7
        self.assertEqual(original["simulation"]["initial_pigment_weights"][0], 2.2)

    def test_thinner_appearances_never_change_the_physical_recipe(self):
        original = study_recipe(1, "fold")
        for appearance in ("thin-glaze", "ink"):
            changed = study_recipe(1, "fold", appearance=appearance)
            self.assertEqual(changed["simulation"], original["simulation"])
            self.assertEqual(changed["projection"], original["projection"])
            self.assertEqual(changed["render"], original["render"])
            self.assertEqual(changed["looks"], ["layered"])
            self.assertLess(changed["surface"]["layer_scale"], original["surface"]["layer_scale"])
            self.assertIn(" · ", changed["name"])

    def test_plan_is_deterministic_and_source_pinned(self):
        seeds = ["0xb7f327f9f722", "0x808861c25b6c"]
        a = make_plan(seeds, [(1, "original"), (3, "fold")])
        self.assertEqual(a, make_plan(seeds, [(1, "original"), (3, "fold")]))
        self.assertEqual(len(a["cases"]), 4)
        self.assertEqual(len({c["id"] for c in a["cases"]}), 4)
        self.assertEqual(a["cases"][0]["source_sha256"], a["cases"][2]["source_sha256"])
        self.assertEqual(a["cases"][0]["mode"], "still")
        with tempfile.TemporaryDirectory() as temporary:
            changed = copy.deepcopy(a)
            changed["cases"][0]["recipe"]["simulation"]["steps"] = 100
            with self.assertRaisesRegex(ValueError, "identity"):
                execute_plan(Path(temporary) / "uncreated", changed)
            self.assertFalse((Path(temporary) / "uncreated").exists())
        for bad in (["0x1"], [seeds[0], seeds[0]]):
            with self.assertRaises(ValueError):
                make_plan(bad, [(3, "original")])
