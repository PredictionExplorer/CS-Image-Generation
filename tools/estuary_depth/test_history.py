"""History publication requires unchanged runtime and a byte-identical final state."""

from __future__ import annotations

import hashlib
import platform
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.estuary import run as paint_run
from tools.estuary.recipe import validate_recipe
from tools.estuary.run import artifact, digest, encoded, read_json, write_json
from tools.estuary_depth import history


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.run, self.output = self.folder / "run", self.folder / "history"
        (self.run / "inputs").mkdir(parents=True)
        path = self.run / "inputs/source.orbit"
        path.write_bytes(b"fixture source")
        self.source = SimpleNamespace(path=path, sha256=digest(path))
        self.recipe = {"simulation": {"steps": 360, "resolution": [128, 96]}, "projection": {}}
        self.request = {
            "code": {"engine.py": "original"},
            "recipe": self.recipe,
            "source": {"sha256": self.source.sha256},
            "backend": "egl",
            "hardware": {"renderer": "fixture"},
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "glcontext": version("glcontext"),
            },
        }
        self.identity = hashlib.sha256(encoded(self.request)).hexdigest()
        self.expected = np.ones((3, 4, 4), dtype=np.float32)
        np.save(self.run / "final-state.npy", self.expected)
        self.records = {"final-state.npy": artifact(self.run / "final-state.npy", self.run)}
        write_json(self.run / "request.json", self.request)
        write_json(self.run / "receipt.json", {"identity_sha256": self.identity})
        expected = self.expected
        instances = self.instances = []

        class Engine:
            def __init__(self, source, recipe, backend):
                self.step, self.internal_steps, self.maximum_courant = 0, 0, 0.0
                self.metadata, self.closed, self.targets = {"renderer": "fixture"}, False, []
                instances.append(self)

            def advance_to(self, step):
                self.targets.append(step)
                self.step = self.internal_steps = step

            def read_state(self):
                return expected * np.float32(self.step / 360)

            def close(self):
                self.closed = True

        self.engine_type = Engine
        patches = [
            patch.object(
                history,
                "verified_run",
                return_value=(
                    self.request,
                    self.identity,
                    self.recipe,
                    self.expected,
                    self.records,
                ),
            ),
            patch.object(history, "code_identity", return_value=self.request["code"]),
            patch.object(history.Source, "read", return_value=self.source),
            patch.object(history, "Engine", Engine),
            patch("builtins.print"),
        ]
        for mock in patches:
            mock.start()
            self.addCleanup(mock.stop)

    def test_publishes_exact_final_proof_and_real_canonical_intermediates(self):
        result = history.replay(self.run, self.output)
        self.assertTrue(result["complete"])
        self.assertEqual(result["final_state"]["sha256"], self.records["final-state.npy"]["sha256"])
        self.assertEqual([item["step"] for item in result["checkpoints"]], [126, 234])
        np.testing.assert_array_equal(
            np.load(self.output / "state-000126.npy"), self.expected * 0.35
        )
        self.assertTrue(self.instances[0].closed)
        self.assertEqual(self.instances[0].targets[-1], 360)
        count = len(self.instances)
        self.assertEqual(history.replay(self.run, self.output), result)
        self.assertEqual(len(self.instances), count)  # Reuse needs no GPU context.

    def _composition_archive(self):
        recipe = validate_recipe(
            {
                "simulation": {
                    "steps": 360,
                    "resolution": [128, 96],
                    "initial_pattern": "composition",
                    "initial_design": {
                        "version": "starting-patterns-v1",
                        "pattern": "folded-sash",
                        "seed": "0x" + "0" * 63 + "1",
                    },
                },
                "render": {"resolution": [128, 96], "frames": 31},
            }
        )
        request = {**self.request, "recipe": recipe, "code": paint_run.code_identity(recipe)}
        identity = hashlib.sha256(encoded(request)).hexdigest()
        write_json(self.run / "request.json", request)
        write_json(self.run / "receipt.json", {"identity_sha256": identity})
        return request, identity, recipe, self.expected, self.records

    def test_composition_replay_binds_the_optional_initializer_runtime(self):
        archive = self._composition_archive()
        self.assertIn("initial_patterns.py", archive[0]["code"])
        self.assertNotIn("initial_patterns.py", paint_run.code_identity())
        with (
            patch.object(history, "verified_run", return_value=archive),
            patch.object(history, "code_identity", side_effect=paint_run.code_identity),
        ):
            result = history.replay(self.run, self.output)
            self.assertTrue(result["complete"])
            self.assertEqual(result["code"], archive[0]["code"])
            self.assertEqual(
                result["final_state"]["sha256"], self.records["final-state.npy"]["sha256"]
            )
            self.assertEqual(history.replay(self.run, self.output), result)

    def test_changed_optional_initializer_during_replay_prevents_publication(self):
        archive = self._composition_archive()
        original_digest = paint_run.digest

        def changing_initializer(path):
            if Path(path).name == "initial_patterns.py" and self.instances:
                return "0" * 64
            return original_digest(path)

        with (
            patch.object(history, "verified_run", return_value=archive),
            patch.object(history, "code_identity", side_effect=paint_run.code_identity),
            patch.object(paint_run, "digest", side_effect=changing_initializer),
            self.assertRaisesRegex(ValueError, "runtime code changed during replay"),
        ):
            history.replay(self.run, self.output)
        self.assertFalse((self.output / "history.json").exists())
        self.assertFalse(read_json(self.output / "status.json")["complete"])
        self.assertTrue(self.instances[0].closed)

    def test_corrupted_history_cannot_be_reused(self):
        history.replay(self.run, self.output)
        np.save(self.output / "state-000126.npy", self.expected)
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            history.replay(self.run, self.output)

    def test_changed_runtime_is_rejected_before_rendering(self):
        with (
            patch.object(history, "code_identity", return_value={}),
            self.assertRaisesRegex(ValueError, "runtime differs"),
        ):
            history.replay(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_failed_final_equality_preserves_states_and_publishes_no_ledger(self):
        with (
            patch.object(self.engine_type, "read_state", return_value=self.expected * 0.9),
            self.assertRaisesRegex(ValueError, "does not exactly match"),
        ):
            history.replay(self.run, self.output)
        self.assertTrue((self.output / "final-state.npy").is_file())
        self.assertTrue((self.output / "state-000126.npy").is_file())
        self.assertFalse((self.output / "history.json").exists())
        self.assertFalse(read_json(self.output / "status.json")["complete"])
        self.assertTrue(self.instances[0].closed)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            history.replay(self.run, self.output)

    def test_interruption_closes_gpu_and_preserves_failed_status(self):
        with (
            patch.object(self.engine_type, "advance_to", side_effect=KeyboardInterrupt("stop")),
            self.assertRaises(KeyboardInterrupt),
        ):
            history.replay(self.run, self.output)
        self.assertTrue(self.instances[0].closed)
        self.assertFalse(read_json(self.output / "status.json")["complete"])
        self.assertFalse((self.output / "history.json").exists())

    def test_fraction_schedule_must_be_increasing_exact_source_steps(self):
        self.assertEqual(history.canonical_steps([0.35, 0.65], 7200), [2520, 4680])
        for values in [
            [],
            [0],
            [1],
            [0.65, 0.35],
            [0.35, 0.35],
            [True],
            [float("nan")],
            [0.123456],
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                history.canonical_steps(values, 7200)


if __name__ == "__main__":
    unittest.main()
