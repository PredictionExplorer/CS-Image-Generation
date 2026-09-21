"""Fresh-seed derivation, portable cohort contracts and exact export evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.estuary_studio.common import artifact, read, write

from . import filament_cohort as cohort


class SeedDerivationTests(unittest.TestCase):
    def test_full_digest_vectors_and_leading_zero_bytes_are_preserved(self):
        result = cohort.derive_seeds([])
        self.assertEqual(len(result["seeds"]), 10)
        self.assertEqual(
            result["seeds"][0], "0xa709b7f63f68f9f92b92af3a3e8d4e578d3486296e165c7cba7d5b85dce8cb81"
        )
        self.assertEqual(
            result["seeds"][4], "0x00dbadbd782a9f2a7353d942595566f2045376f4d30a05298304b9646e2d1bed"
        )
        self.assertTrue(all(len(bytes.fromhex(seed[2:])) == 32 for seed in result["seeds"]))
        self.assertEqual([row["counter"] for row in result["attempts"]], list(range(10)))
        self.assertNotEqual(
            result["seeds"], cohort.derive_seeds([], domain=cohort.DOMAIN + "/other")["seeds"]
        )

    def test_numeric_alias_collisions_are_rejected_without_discarding_other_candidates(self):
        original = cohort.derive_seeds([])
        first = original["seeds"][0]
        blocked = cohort.derive_seeds([int(first, 16), first.upper(), first[2:]], count=2)
        self.assertEqual(blocked["seeds"], original["seeds"][1:3])
        self.assertEqual(blocked["attempts"][0]["rejection"], "historical-collision")
        self.assertEqual(blocked["historical_seeds"], [hex(int(first, 16))])
        self.assertEqual(cohort.derive_seeds([17, "0x0011"]), cohort.derive_seeds(["0x11"]))

    def test_invalid_or_unbounded_derivation_inputs_fail(self):
        for seed in (True, -1, 1 << 256, "0x", "no-seed", 1.5, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                cohort.derive_seeds([seed])
        for count in (0, 101, True, 2.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                cohort.derive_seeds([], count=count)


class CohortArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "exporter"
        self.binary.write_bytes(b"pinned test exporter")
        self.pin = patch.object(cohort, "EXPORTER_SHA256", artifact(self.binary)["sha256"])
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.history = self.root / "history.json"
        write(
            self.history,
            {"version": "historical-art-seed-inventory-v1", "seeds": ["0xbc53af1cd380", "0x1"]},
        )
        self.output = self.root / "cohort"
        self.plan = cohort.prepare_cohort(self.output, self.binary, self.history)
        self.rows = []
        for index, seed in enumerate(self.plan["seeds"]):
            path = self.output / f"orbits/{seed}.orbit"
            path.write_bytes(b"bound orbit fixture: " + seed.encode())
            log = self.output / f"logs/{seed}.log"
            log.write_text("Recorded complete source fixture\n")
            metadata = {
                "seed": seed,
                "sha256": artifact(path)["sha256"],
                "samples": 1000000,
                "samples_sha256": hashlib.sha256(f"samples-{index}".encode()).hexdigest(),
                "dt": 0.001,
                "source_first_step": 0,
                "source_last_step": 999999,
                "projection": {**cohort.PROJECTION, "scale": 1.0},
                "provenance": {
                    **cohort.PHYSICS,
                    "selection": {
                        "source": "new-production-selection",
                        "candidate_count": 30000,
                        "candidate_index": index,
                        "retry_count": 0,
                        "chaos_weight": 1.0,
                        "equil_weight": 2.0,
                        "selection_resolution": [1920, 1242],
                        "escape_threshold": -0.3,
                        "sampling_bounds": {
                            "min_mass": 100.0,
                            "max_mass": 300.0,
                            "location": 300.0,
                            "velocity": 1.0,
                        },
                    },
                },
            }
            config_path = self.output / f"configs/{seed}.json"
            self.rows.append(
                {
                    "seed": seed,
                    "path": f"orbits/{seed}.orbit",
                    **artifact(path),
                    "samples": 1000000,
                    "samples_sha256": metadata["samples_sha256"],
                    "dt": 0.001,
                    "source_metadata": metadata,
                    "config": read(config_path),
                    "returncode": 0,
                    "config_artifact": {"path": f"configs/{seed}.json", **artifact(config_path)},
                    "log_artifact": {"path": f"logs/{seed}.log", **artifact(log)},
                }
            )
        self.manifest = {
            "version": cohort.VERSION,
            "complete": True,
            "plan": self.plan,
            "seeds": self.plan["seeds"],
            "sources": self.rows,
            "failures": [],
        }
        self.seal(self.manifest)

    @staticmethod
    def seal(value):
        value["plan"]["identity_sha256"] = cohort._identity(value["plan"])
        value["identity_sha256"] = cohort._identity(value)

    def source(self, path, **_kwargs):
        row = next(row for row in self.rows if row["seed"] == Path(path).stem)
        return SimpleNamespace(metadata=copy.deepcopy(row["source_metadata"]))

    def test_offline_manifest_verification_requires_no_original_files(self):
        self.binary.unlink()
        self.assertEqual(list(cohort.verify_cohort(self.manifest)), self.plan["seeds"])
        self.assertEqual(read(self.output / "seeds.json"), self.plan["seeds"])
        self.plan["settings"]["sims"] = 1
        self.assertEqual(cohort.SETTINGS["sims"], 30000)

    def test_actual_files_are_bound_and_modified_orbits_are_rejected(self):
        with patch("tools.estuary.source.Source.read", side_effect=self.source) as reader:
            self.assertEqual(len(cohort.verify_cohort(self.manifest, source_root=self.output)), 10)
            self.assertEqual(reader.call_count, 10)
            (self.output / self.rows[0]["path"]).write_bytes(b"changed raw trajectory")
            with self.assertRaisesRegex(ValueError, "changed|missing|differs"):
                cohort.verify_cohort(self.manifest, source_root=self.output)

    def test_projection_roundoff_is_allowed_but_changed_projection_is_not(self):
        def with_delta(delta):
            def source(path, **kwargs):
                actual = self.source(path, **kwargs)
                actual.metadata["projection"]["scale"] += delta
                return actual

            return source

        with patch("tools.estuary.source.Source.read", side_effect=with_delta(1e-13)):
            cohort.verify_cohort(self.manifest, source_root=self.output)
        with (
            patch("tools.estuary.source.Source.read", side_effect=with_delta(1e-4)),
            self.assertRaisesRegex(ValueError, "metadata"),
        ):
            cohort.verify_cohort(self.manifest, source_root=self.output)

    def test_partial_rerolled_or_reordered_sources_are_rejected_even_when_rehashed(self):
        for change in ("partial", "dropped", "reordered", "rerolled"):
            altered = copy.deepcopy(self.manifest)
            if change == "partial":
                altered["complete"] = False
            elif change == "dropped":
                altered["sources"].pop()
            elif change == "reordered":
                altered["sources"].reverse()
            else:
                altered["plan"]["seeds"][0] = "0x" + "f" * 64
            self.seal(altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                cohort.verify_cohort(altered)

    def test_short_recording_different_physics_budget_or_path_is_rejected(self):
        for change in ("samples", "physics", "budget", "path", "boolean-step", "config"):
            altered = copy.deepcopy(self.manifest)
            row = altered["sources"][0]
            if change == "samples":
                row["samples"] = row["source_metadata"]["samples"] = 999999
            elif change == "physics":
                row["source_metadata"]["provenance"]["sample_stride"] = 2
            elif change == "budget":
                row["source_metadata"]["provenance"]["selection"]["candidate_count"] = 1024
            elif change == "path":
                row["path"] = "../outside.orbit"
            elif change == "boolean-step":
                row["source_metadata"]["source_first_step"] = False
            else:
                row["config"]["steps"] = 999999
            self.seal(altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                cohort.verify_cohort(altered)

    def test_export_failures_preserve_all_seed_choices_without_substitution(self):
        for row in self.rows:
            (self.output / row["path"]).unlink()
        failure = subprocess.CalledProcessError(1, ["fixture exporter"])
        with (
            patch.object(cohort.subprocess, "run", side_effect=failure) as exporter,
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "exports failed"),
        ):
            cohort.generate_cohort(self.output)
        binary = str((self.output / self.plan["generator"]["path"]).resolve())
        exports = [
            call for call in exporter.call_args_list if call.args and call.args[0][0] == binary
        ]
        self.assertEqual(len(exports), 10)
        failed = read(self.output / "cohort.json")
        self.assertFalse(failed["complete"])
        self.assertEqual(failed["seeds"], self.plan["seeds"])
        self.assertEqual({r["seed"] for r in failed["failures"]}, set(self.plan["seeds"]))


if __name__ == "__main__":
    unittest.main()
