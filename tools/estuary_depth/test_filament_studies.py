"""Isolation, reproducibility and bounded color variation for the fold studies."""

from __future__ import annotations

import copy
import json
import unittest

from tools.estuary.recipe import read_recipe, validate_recipe
from tools.estuary_depth import filament_studies as studies
from tools.estuary_depth.render import recipe as validate_depth_recipe


class FilamentStudyTests(unittest.TestCase):
    def test_reference_control_retains_original_simulation_optics_and_studio(self):
        original = read_recipe(studies.PAINT_RECIPE)
        control = studies.make_paint_recipe(studies.REFERENCE_SEED, "control")
        for key in ("simulation", "projection", "optics"):
            self.assertEqual(control[key], original[key])
        self.assertEqual(control["simulation"]["resolution"], [6144, 4608])
        self.assertEqual(control["simulation"]["steps"], 7200)
        self.assertEqual(control["render"]["resolution"], [2048, 1536])
        self.assertTrue(control["render"]["initial_image"])
        self.assertEqual(control["render"]["frames"], 901)
        depth = studies.make_depth_recipe(studies.REFERENCE_SEED, "The folded tide", proof=False)
        self.assertEqual(depth, validate_depth_recipe(json.loads(studies.DEPTH_RECIPE.read_text())))

    def test_each_variant_changes_only_its_declared_simulation_controls(self):
        changes = {
            "control": {},
            "plain-bands": {"strata_profile": {"fine_width_scale": 0.0}},
            "fine-bands": {"strata_profile": {"fine_width_scale": 0.5}},
            "narrow-main": {"strata_profile": {"main_width_scale": 0.65}},
            "stronger-folds": {"pair_swirl": 1.25},
            "calmer-current": {"carrier_velocity": [0.6, 0.06]},
            "three-pools": {"initial_pattern": "pools"},
            "three-broad-pools": {"initial_pattern": "pools", "load_radius": 0.5},
            "three-pools-folds": {
                "initial_pattern": "pools",
                "load_radius": 0.5,
                "pair_swirl": 1.25,
            },
            "coarse-control": {"resolution": [2048, 1536]},
            "coarse-three-pools": {
                "initial_pattern": "pools",
                "load_radius": 0.5,
                "resolution": [2048, 1536],
            },
        }
        self.assertEqual(set(changes), set(studies.VARIANTS))
        control = studies.make_paint_recipe(studies.REFERENCE_SEED, "control")
        for variant, overrides in changes.items():
            with self.subTest(variant=variant):
                expected = copy.deepcopy(control)
                expected["simulation"].update(overrides)
                self.assertEqual(
                    studies.make_paint_recipe(studies.REFERENCE_SEED, variant),
                    validate_recipe(expected),
                )

    def test_full_seed_changes_colors_but_not_geometry_and_canonical_aliases_match(self):
        low, high = 17, 17 + (1 << 240)
        first = studies.make_paint_recipe(low, "control")
        second = studies.make_paint_recipe(high, "control")
        self.assertNotEqual(first["optics"]["pigments_srgb"], second["optics"]["pigments_srgb"])
        self.assertEqual(first["simulation"], second["simulation"])
        self.assertEqual(first, studies.make_paint_recipe("0X0011", "control"))
        self.assertEqual(second, studies.make_paint_recipe(hex(high), "control"))
        first_depth = studies.make_depth_recipe(low, "Study")
        second_depth = studies.make_depth_recipe(high, "Study")
        self.assertNotEqual(first_depth["render"]["seed"], second_depth["render"]["seed"])
        self.assertEqual(first_depth, studies.make_depth_recipe("0011", "Study"))

    def test_palette_roles_stay_in_declared_bounds_for_the_cohort(self):
        reference = read_recipe(studies.PAINT_RECIPE)["optics"]["pigments_srgb"]
        seeds = [studies.REFERENCE_SEED, 0, (1 << 256) - 1, *range(64)]
        for seed in seeds:
            result = studies.make_paint_recipe(seed, "control")
            self.assertEqual(result, validate_recipe(result))
            palette = result["optics"]["pigments_srgb"]
            for color, original, offsets in zip(
                palette, reference, studies.PALETTE_OFFSETS, strict=True
            ):
                for value, center, bound in zip(color, original, offsets, strict=True):
                    self.assertLessEqual(abs(value - center), bound)
                    self.assertTrue(0 <= value <= 1)
            navy, ivory, oxide = palette
            self.assertGreater(navy[2], max(navy[:2]))
            self.assertGreater(min(ivory), 0.8)
            self.assertGreater(oxide[0], max(oxide[1:]))

    def test_proof_changes_only_output_quality_and_results_do_not_share_mutable_state(self):
        proof = studies.make_depth_recipe(12, "Study")
        final = studies.make_depth_recipe(12, "Study", proof=False)
        expected = copy.deepcopy(proof)
        expected["render"].update(resolution=[3840, 2880], samples=256)
        self.assertEqual(final, expected)
        self.assertEqual(proof, validate_depth_recipe(proof))
        self.assertEqual(proof["render"]["resolution"], [2048, 1536])
        self.assertEqual(proof["render"]["samples"], 128)
        final["material"]["coat"] = 0.2
        final["camera"]["target"][0] = 0.1
        self.assertEqual(studies.make_depth_recipe(12, "Study"), proof)
        first = studies.make_paint_recipe(12, "control")
        altered = studies.make_paint_recipe(12, "control")
        altered["optics"]["pigments_srgb"][0][0] = 0.5
        self.assertEqual(studies.make_paint_recipe(12, "control"), first)

    def test_invalid_seeds_registry_ids_and_quality_flags_are_rejected(self):
        for seed in (True, -1, 1 << 256, 1.5, "0x", "not-hex", None, "f" * 65):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                studies.make_paint_recipe(seed, "control")
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                studies.make_depth_recipe(seed, "Study")
        for variant in ("Control", "unknown", "../control", None, [], True):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                studies.make_paint_recipe(0, variant)
        for flag in (0, 1, None, "false"):
            with self.subTest(proof=flag), self.assertRaises(ValueError):
                studies.make_depth_recipe(0, "Study", proof=flag)
        with self.assertRaises(TypeError):
            studies.VARIANTS["other"] = studies.VARIANTS["control"]


if __name__ == "__main__":
    unittest.main()
