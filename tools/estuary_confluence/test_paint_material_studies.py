"""A complete factorial study changes only declared RC1 material/display controls."""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.estuary_studio.common import encoded

from . import paint_material_studies as studies
from .body_marker_studies import released_recipe
from .run import frame_plan


class PaintMaterialStudiesTests(unittest.TestCase):
    def test_factorial_covers_each_combination_once_and_preserves_exact_reference(self):
        plan = studies.make_plan(source_root="/tmp/paint-material-source")
        self.assertEqual(len(plan["cases"]), 48)
        self.assertEqual(len({case["id"] for case in plan["cases"]}), 48)
        self.assertEqual(len(studies.FACTORIAL), 16)
        self.assertEqual(len({spec.features for spec in studies.FACTORIAL.values()}), 16)
        for seed in studies.DEFAULT_SEEDS:
            accepted, original = released_recipe(seed)
            rows = [case for case in plan["cases"] if case["seed"] == seed]
            self.assertEqual({case["study"]["variant"] for case in rows}, set(studies.FACTORIAL))
            baseline = next(case for case in rows if case["study"]["variant"] == "rc1")
            self.assertEqual(baseline["recipe"], original)
            self.assertEqual(
                hashlib.sha256(encoded(baseline["recipe"])).hexdigest(), accepted["recipe_sha256"]
            )
            for case in rows:
                recipe = case["recipe"]
                self.assertEqual(case["mode"], "still")
                self.assertEqual(recipe["simulation"]["resolution"], [2048, 1536])
                self.assertEqual(recipe["render"], original["render"])
                self.assertEqual(len(frame_plan(recipe)), 937)
                self.assertEqual(frame_plan(recipe), frame_plan(original))
                self.assertEqual(recipe["simulation"]["steps"], 7200)
                self.assertNotIn("body_influence", recipe["simulation"])
                self.assertNotIn("body_markers", recipe["render"])
                self.assertEqual(
                    Path(case["source"]).parent, Path("/tmp/paint-material-source").resolve()
                )

    def test_every_variant_changes_only_declared_controls(self):
        seed = studies.DEFAULT_SEEDS[0]
        _, original = released_recipe(seed)
        for variant, spec in studies.VARIANTS.items():
            with self.subTest(variant=variant):
                _, _, recipe = studies.material_recipe(seed, variant)
                expected = copy.deepcopy(original)
                if variant != "rc1":
                    expected["name"] = spec.label
                    expected["looks"] = ["silk-grain"]
                if spec.height_scale is not None:
                    expected["surface"]["height_scale"] = spec.height_scale
                if spec.roughness_bias is not None:
                    expected["surface"]["roughness_bias"] = spec.roughness_bias
                normalized = copy.deepcopy(recipe)
                normalized["surface"]["interaction"].pop("directional_relief", None)
                normalized["simulation"]["interaction"].pop("material_variation", None)
                normalized["simulation"].pop("rheology", None)
                self.assertEqual(normalized, expected)
                if spec.relief_strength is not None:
                    self.assertEqual(
                        recipe["surface"]["interaction"]["directional_relief"],
                        {
                            "version": "directional-contact-relief-v1",
                            "strength": spec.relief_strength,
                            "anisotropy": 0.8,
                        },
                    )
                if spec.trait_amplitude is not None:
                    self.assertEqual(
                        recipe["simulation"]["interaction"]["material_variation"],
                        {
                            "version": "paint-material-variation-v1",
                            "amplitude": spec.trait_amplitude,
                            "coarse_scale": 0.16,
                            "fine_scale": 0.035,
                            "fine_fraction": 0.25,
                        },
                    )
                if spec.resistance_strength is not None:
                    self.assertEqual(
                        recipe["simulation"]["rheology"]["strength"], spec.resistance_strength
                    )

    def test_material_identity_requirements_follow_physical_scope(self):
        seed = studies.DEFAULT_SEEDS[0]
        accepted, _ = released_recipe(seed)
        reference = next(row for row in studies.references()["cases"] if row["seed"] == seed)
        plan = studies.make_plan([seed], suite="all")
        self.assertEqual(len(plan["cases"]), 27)
        for case in plan["cases"]:
            spec = studies.VARIANTS[case["study"]["variant"]]
            self.assertEqual(case["reference_initial_mass"], [*reference["target_mass"], 0])
            self.assertEqual(case["reference_layout_artifact"], reference["layout_artifact"])
            self.assertEqual(
                case["reference_palette_identity_sha256"], reference["palette_identity_sha256"]
            )
            self.assertEqual(case["source_sha256"], reference["source_sha256"])
            self.assertEqual(
                case["study"]["features"],
                {feature: feature in spec.features for feature in studies.FEATURES},
            )
            if spec.resistance_strength is not None:
                self.assertNotIn("accepted_base_material_sha256", case)
                self.assertNotIn("accepted_physical_state_sha256", case)
            else:
                self.assertEqual(
                    case["accepted_base_material_sha256"], accepted["base_material_sha256"]
                )
                if spec.trait_amplitude is None:
                    self.assertEqual(
                        case["accepted_physical_state_sha256"], accepted["physical_state_sha256"]
                    )
                else:
                    self.assertNotIn("accepted_physical_state_sha256", case)

    def test_strength_sweep_and_later_full_films_keep_the_same_complete_timeline(self):
        strengths = studies.make_plan(suite="strengths")
        self.assertEqual(len(strengths["cases"]), 21)
        self.assertEqual(
            {case["study"]["variant"] for case in strengths["cases"]}, set(studies.STRENGTHS)
        )
        all_cases = studies.make_plan(suite="all")["cases"]
        self.assertEqual(len(all_cases), 81)
        films = studies.make_plan(
            [studies.DEFAULT_SEEDS[1]], ["rc1", "traits-resistance"], film=True
        )
        self.assertTrue(all(case["mode"] == "film" for case in films["cases"]))
        self.assertTrue(all(len(frame_plan(case["recipe"])) == 937 for case in films["cases"]))

    def test_refinements_keep_history_cutoff_separate_from_paint_amount_response(self):
        plan = studies.make_plan(suite="refinements")
        self.assertEqual(len(plan["cases"]), 12)
        for case in plan["cases"]:
            name = case["study"]["variant"]
            config = case["recipe"]["simulation"]["rheology"]
            self.assertEqual(config["minimum_concentration"], 1e-5)
            self.assertEqual(case["study"]["suite"], "refinements")
            if name == "resistance-wide":
                self.assertEqual(config["response_length"], 0.08)
                self.assertNotIn("occupancy_mass_reference", config)
            else:
                self.assertEqual(
                    config["occupancy_mass_reference"],
                    0.02 if name == "resistance-mass-02" else 0.01,
                )
                self.assertEqual(config["response_length"], 0.04)
        for name in studies.FACTORIAL:
            _, _, recipe = studies.material_recipe(studies.DEFAULT_SEEDS[0], name)
            self.assertNotIn("occupancy_mass_reference", recipe["simulation"].get("rheology", {}))

    def test_exact_catalog_rejects_control_palette_camera_or_label_drift(self):
        seed = studies.DEFAULT_SEEDS[0]
        catalog = studies.recipe_catalog(seed)
        for variant, recipe in catalog.items():
            self.assertEqual(studies.identify_recipe(seed, recipe), variant)
        original = catalog["traits"]
        for name in ("height", "ground", "name", "camera", "flow", "seed"):
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                if name == "height":
                    changed["surface"]["height_scale"] += 0.1
                elif name == "ground":
                    changed["surface"]["ground_srgb"][0] += 0.001
                elif name == "name":
                    changed["name"] = "RC1 reference"
                elif name == "camera":
                    changed["render"]["azimuth_end"] += 0.01
                elif name == "flow":
                    changed["simulation"]["flow_strength"] += 0.01
                with self.assertRaisesRegex(ValueError, "exact paint-material study"):
                    studies.identify_recipe(
                        studies.DEFAULT_SEEDS[1] if name == "seed" else seed, changed
                    )
        # Catalog callers receive independent dictionaries rather than shared presets.
        catalog["traits"]["surface"]["ground_srgb"][0] = 0
        self.assertNotEqual(studies.recipe_catalog(seed)["traits"], catalog["traits"])

    def test_background_resolution_matches_runner_and_does_not_normalize_supplied_tampering(self):
        seed = studies.DEFAULT_SEEDS[0]
        accepted, reference, original = studies._reference_recipe(seed)
        original["surface"]["ground_srgb"] = [0.8, 0.7, 0.6]
        with patch.object(
            studies, "_reference_recipe", return_value=(accepted, reference, original)
        ):
            catalog = studies.recipe_catalog(seed)
            self.assertNotEqual(
                catalog["rc1"]["surface"]["ground_srgb"], original["surface"]["ground_srgb"]
            )
            self.assertEqual(studies.identify_recipe(seed, catalog["rc1"]), "rc1")
            with self.assertRaises(ValueError):
                studies.identify_recipe(seed, original)

    def test_plan_is_deterministic_and_rejects_ambiguous_requests(self):
        seed = studies.DEFAULT_SEEDS[0]
        with patch.object(studies, "runtime_identity", return_value={"frozen": "source"}):
            left = studies.make_plan([seed], ["rc1", "fuller-traits"], source_root="/tmp/orbits")
            right = studies.make_plan([seed], ["rc1", "fuller-traits"], source_root="/tmp/orbits")
            self.assertEqual(encoded(left), encoded(right))
            self.assertEqual(
                left["identity_sha256"],
                hashlib.sha256(
                    encoded({key: value for key, value in left.items() if key != "identity_sha256"})
                ).hexdigest(),
            )
        for kwargs in (
            {"seeds": []},
            {"seeds": [seed, seed]},
            {"seeds": [1]},
            {"seeds": ["0x1"]},
            {"variants": []},
            {"variants": ["traits", "traits"]},
            {"variants": ["other"]},
            {"suite": "other"},
            {"suite": "all", "variants": ["rc1"]},
            {"film": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                studies.make_plan(**kwargs)


if __name__ == "__main__":
    unittest.main()
