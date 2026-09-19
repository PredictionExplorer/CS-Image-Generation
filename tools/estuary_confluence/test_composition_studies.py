"""Six composition variants share their RC1 material amounts and artistic controls."""

import unittest

from .body_marker_studies import released_recipe
from .composition_studies import SETUPS, composition_recipe, make_plan, references
from .studies import verify_reference_mass


class CompositionStudyTests(unittest.TestCase):
    def test_all_sixty_recipes_change_only_the_explicit_initial_condition(self):
        for reference in references()["cases"]:
            seed = reference["seed"]
            _, original = released_recipe(seed)
            for setup in SETUPS:
                with self.subTest(seed=seed, setup=setup):
                    actual_reference, recipe = composition_recipe(seed, setup)
                    self.assertEqual(actual_reference, reference)
                    for key in original.keys() - {"simulation", "looks", "name"}:
                        self.assertEqual(recipe[key], original[key])
                    self.assertEqual(
                        {
                            k: v
                            for k, v in recipe["simulation"].items()
                            if k
                            not in (
                                "initial_pattern",
                                "initial_pigment_weights",
                                "initial_composition",
                            )
                        },
                        {
                            k: v
                            for k, v in original["simulation"].items()
                            if k not in ("initial_pattern", "initial_pigment_weights")
                        },
                    )
                    self.assertIsNone(recipe["simulation"]["initial_pigment_weights"])
                    self.assertEqual(
                        recipe["simulation"]["initial_composition"]["target_mass"],
                        reference["target_mass"],
                    )

    def test_plan_contains_every_seed_setup_once_and_full_films(self):
        plan = make_plan()
        self.assertEqual(len(plan["cases"]), 60)
        self.assertEqual(len({c["id"] for c in plan["cases"]}), 60)
        for case in plan["cases"]:
            self.assertEqual(case["mode"], "film")
            self.assertNotIn("accepted_physical_state_sha256", case)
            self.assertNotIn("body_markers", case["recipe"]["render"])
            self.assertEqual(case["recipe"]["render"]["formation_frames"], 721)
            self.assertEqual(case["reference_initial_mass"][-1], 0)

    def test_reference_budget_accepts_rounding_but_rejects_extra_paint_or_chalk(self):
        expected = [0.03, 0.01, 0.006, 0.0]
        verify_reference_mass({"initial_mass": [0.030000001, 0.01, 0.006, 0]}, expected)
        for actual in ([0.031, 0.01, 0.006, 0], [0.03, 0.01, 0.006, 1e-14], [0.03]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                verify_reference_mass({"initial_mass": actual}, expected)


if __name__ == "__main__":
    unittest.main()
