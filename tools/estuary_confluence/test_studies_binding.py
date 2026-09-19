"""Verified batch results must still match the frozen experiment's intent."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.estuary_studio.common import read

from . import studies
from .backgrounds import generate_background
from .palette import generate_palette


class StudyBindingTests(unittest.TestCase):
    def execute(self, output, *, alteration=None):
        code = {"test-runtime": {"fixture.py": "a" * 64}}
        with patch.object(studies, "runtime_identity", return_value=code):
            plan = studies.make_plan(["0xb7f327f9f722"], [(2, "original")])
            case = plan["cases"][0]
            recipe = copy.deepcopy(case["recipe"])
            palette = generate_palette(case["seed"], 2, mode=recipe["palette_mode"])
            recipe["surface"]["ground_srgb"] = generate_background(recipe["background"], palette)[
                "ground_srgb"
            ]
            request = {
                "code": copy.deepcopy(code),
                "recipe": recipe,
                "mode": case["mode"],
                "source": {"sha256": case["source_sha256"]},
            }
            receipt = {
                "source_fraction": 1,
                "complete": True,
                "identity_sha256": "b" * 64,
                "physical_state_sha256": "c" * 64,
            }
            if alteration is not None:
                alteration(request)
            with (
                patch.object(studies, "digest", return_value=case["source_sha256"]),
                patch.object(studies, "verify_run", return_value=(request, receipt)),
                patch.object(studies.subprocess, "run") as process,
                patch("builtins.print"),
            ):
                result = studies.execute_plan(output, plan, workers=1)
                self.assertEqual(process.call_args.kwargs["cwd"], studies.ROOT.parents[1])
                return result

    def test_worker_uses_frozen_checkout_and_accepts_expected_seed_ground_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            result = self.execute(output)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["cases"]), 1)
            self.assertEqual(read(output / "results.json"), result)

    def test_self_consistent_archive_with_wrong_code_recipe_or_mode_is_rejected(self):
        mutations = {
            "code": lambda request: request["code"]["test-runtime"].update(
                {"fixture.py": "d" * 64}
            ),
            "recipe": lambda request: request["recipe"]["simulation"].update({"steps": 3600}),
            "output mode": lambda request: request.update({"mode": "film"}),
        }
        for label, alteration in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "experiment"
                with self.assertRaisesRegex(ValueError, "Some studies failed"):
                    self.execute(output, alteration=alteration)
                report = read(output / "results.json")
                self.assertFalse(report["complete"])
                self.assertEqual(report["cases"], [])
                self.assertIn(label, report["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
