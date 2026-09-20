"""Matched body-selection experiments preserve the accepted artistic controls."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from .body_influence_studies import (
    REDUCED_VARIANTS,
    VARIANTS,
    influence_recipe,
    make_plan,
    references,
)
from .body_marker_studies import released_recipe
from .run import frame_plan
from .studies import verify_reference_design


class BodyInfluenceStudies(unittest.TestCase):
    def test_every_selection_changes_only_the_paint_influence_control(self):
        for row in references()["cases"]:
            seed = row["seed"]
            _, original = released_recipe(seed)
            for variant, bodies in VARIANTS.items():
                with self.subTest(seed=seed, variant=variant):
                    _, _, recipe = influence_recipe(seed, variant)
                    comparable = copy.deepcopy(recipe)
                    control = comparable["simulation"].pop("body_influence", None)
                    if len(bodies) == 3:
                        self.assertIsNone(control)
                    else:
                        self.assertEqual(
                            control, {"version": "body-influence-v1", "bodies": list(bodies)}
                        )
                    comparable["name"], comparable["looks"] = original["name"], original["looks"]
                    self.assertEqual(comparable, original)
                    self.assertEqual(frame_plan(recipe), frame_plan(original))

    def test_default_plan_contains_all_six_reduced_sets_and_complete_films(self):
        source_root = Path("/tmp/body-influence-source-test").resolve()
        plan = make_plan(source_root=source_root)
        self.assertEqual(len(plan["cases"]), 60)
        self.assertEqual(len({c["id"] for c in plan["cases"]}), 60)
        for reference in references()["cases"]:
            rows = [c for c in plan["cases"] if c["seed"] == reference["seed"]]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {tuple(c["recipe"]["simulation"]["body_influence"]["bodies"]) for c in rows},
                {VARIANTS[v] for v in REDUCED_VARIANTS},
            )
            for case in rows:
                self.assertEqual(case["source_sha256"], reference["source_sha256"])
                self.assertEqual(case["mode"], "film")
                self.assertEqual(case["reference_initial_mass"], [*reference["target_mass"], 0])
                self.assertEqual(case["reference_layout_artifact"], reference["layout_artifact"])
                self.assertEqual(
                    case["reference_palette_identity_sha256"], reference["palette_identity_sha256"]
                )
                self.assertEqual(Path(case["source"]).parent, source_root)
                self.assertEqual(len(frame_plan(case["recipe"])), 937)

    def test_full_control_requires_exact_rc1_state_and_invalid_requests_fail(self):
        seed = references()["cases"][0]["seed"]
        accepted, _ = released_recipe(seed)
        case = make_plan([seed], ["all-bodies"], film=False)["cases"][0]
        self.assertNotIn("body_influence", case["recipe"]["simulation"])
        self.assertEqual(case["mode"], "still")
        self.assertEqual(case["accepted_physical_state_sha256"], accepted["physical_state_sha256"])
        for kwargs in (
            {"seeds": []},
            {"seeds": [seed, seed]},
            {"seeds": ["0x1"]},
            {"variants": []},
            {"variants": ["body-1", "body-1"]},
            {"variants": ["body-4"]},
            {"film": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(**kwargs)

    def test_reference_binding_rejects_valid_but_different_starting_design(self):
        reference = references()["cases"][0]
        case = {
            "reference_layout_artifact": reference["layout_artifact"],
            "reference_palette_identity_sha256": reference["palette_identity_sha256"],
        }
        receipt = {"artifacts": {"layout.json": copy.deepcopy(reference["layout_artifact"])}}
        request = {"palette": {"identity_sha256": reference["palette_identity_sha256"]}}
        verify_reference_design(request, receipt, case)
        changed = copy.deepcopy(receipt)
        changed["artifacts"]["layout.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "starting layout"):
            verify_reference_design(request, changed, case)
        changed = {"palette": {"identity_sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "palette"):
            verify_reference_design(changed, receipt, case)
        verify_reference_design({}, {}, {})  # Older studies have no added design contract.


if __name__ == "__main__":
    unittest.main()
