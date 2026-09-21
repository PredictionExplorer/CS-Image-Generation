"""Initial-paint studies preserve RC1 controls and remain reproducible."""

import copy
import unittest

from tools.estuary_studio.common import encoded

from .body_marker_studies import released_recipe
from .choreography_studies import (
    VARIANTS,
    choreography_recipe,
    identify_recipe,
    make_plan,
    references,
)
from .palette import generate_palette


class ChoreographyStudyTests(unittest.TestCase):
    def test_all_recipes_preserve_rc1_except_declared_initial_conditions(self):
        for reference in references()["cases"]:
            seed = reference["seed"]
            _, original = released_recipe(seed)
            palette = generate_palette(seed, 3, mode="composed")
            for variant, spec in VARIANTS.items():
                with self.subTest(seed=seed, variant=variant):
                    ref, actual = choreography_recipe(seed, variant)
                    self.assertEqual(ref, reference)
                    for key in original.keys() - {
                        "name",
                        "looks",
                        "simulation",
                        "launch_assessment",
                    }:
                        self.assertEqual(actual[key], original[key])
                    ignored = {"initial_pattern", "initial_choreography", "initial_pigment_weights"}
                    self.assertEqual(
                        {k: v for k, v in actual["simulation"].items() if k not in ignored},
                        {k: v for k, v in original["simulation"].items() if k not in ignored},
                    )
                    initial = actual["simulation"]["initial_choreography"]
                    self.assertEqual(initial["target_mass"], reference["target_mass"])
                    self.assertEqual(initial["reference_radii"], reference["reference_radii"])
                    self.assertEqual(initial["setup"], spec.setup)
                    if spec.mobility_bias:
                        effective = [
                            a + b
                            for a, b in zip(
                                palette["layer_fractions"][:3], spec.mobility_bias, strict=True
                            )
                        ]
                        self.assertTrue(all(0.15 <= f <= 0.85 for f in effective))
                    self.assertNotIn("rheology", actual["simulation"])
                    self.assertNotIn("body_influence", actual["simulation"])

    def test_plan_covers_complete_source_and_pins_pigments_without_pinning_geometry(self):
        plan = make_plan(source_root="/tmp/immutable-orbits")
        self.assertEqual(len(plan["cases"]), 30)
        self.assertEqual(len({c["id"] for c in plan["cases"]}), 30)
        self.assertEqual(encoded(plan), encoded(make_plan(source_root="/tmp/immutable-orbits")))
        for case in plan["cases"]:
            self.assertEqual(case["mode"], "still")
            self.assertEqual(case["reference_initial_mass"][3], 0)
            self.assertEqual(case["recipe"]["simulation"]["resolution"], [2048, 1536])
            self.assertEqual(case["recipe"]["render"]["formation_frames"], 721)
            self.assertNotIn("reference_layout_artifact", case)
            self.assertNotIn("accepted_physical_state_sha256", case)
        seed = references()["cases"][0]["seed"]
        self.assertEqual(
            make_plan([seed], ["stretch-ovals"], film=True)["cases"][0]["mode"], "film"
        )

    def test_identification_uses_all_controls_not_labels(self):
        from .backgrounds import generate_background

        seed = references()["cases"][0]["seed"]
        _, recipe = choreography_recipe(seed, "stretch-ovals")
        palette = generate_palette(seed, 3, mode=recipe["palette_mode"])
        recipe["surface"]["ground_srgb"] = generate_background(recipe["background"], palette)[
            "ground_srgb"
        ]
        self.assertEqual(identify_recipe(seed, recipe), "stretch-ovals")
        altered = copy.deepcopy(recipe)
        altered["simulation"]["flow_strength"] *= 1.001
        with self.assertRaisesRegex(ValueError, "exact choreography"):
            identify_recipe(seed, altered)


if __name__ == "__main__":
    unittest.main()
