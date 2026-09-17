"""Finite collection scheduling, provenance and process failure safeguards."""

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "render_collection", Path(__file__).with_name("render_collection.py")
)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class CollectionTests(unittest.TestCase):
    def controller(self, folder, seeds=("0xaa", "0xbb", "0xcc")):
        result = batch.Collection.__new__(batch.Collection)
        result.output = folder
        result.args = SimpleNamespace(seeds=list(seeds))
        result.cancel = threading.Event()
        result.mutex = threading.Lock()
        result.state = {"complete": False, "seeds": {seed: {} for seed in seeds}}
        return result

    def test_cpu_normal_artifact_reuse_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertFalse(batch.normal_complete(folder))
            artifacts = []
            for name in batch.NORMAL_FILES:
                file = folder / name
                file.write_bytes(name.encode())
                artifacts.append(
                    {"path": name, "bytes": file.stat().st_size, "sha256": batch.digest(file)}
                )
            batch.write(folder / "normal.json", {"complete": True, "artifacts": artifacts})
            self.assertTrue(batch.normal_complete(folder))
            (folder / "normal-hq.mp4").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                batch.normal_complete(folder)
            batch.write(folder / "normal.json", {"complete": False})
            archived = batch.quarantine(folder)
            self.assertFalse(folder.exists())
            self.assertEqual((archived / "normal-hq.mp4").read_bytes(), b"corrupted")
            # TemporaryDirectory can no longer remove the deliberately renamed folder.
            import shutil

            shutil.rmtree(archived)

    def test_archive_identity_and_frozen_copy_are_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, frozen = folder / "source.py", folder / "frozen.py"
            source.write_bytes(b"original script")
            batch.preserve(frozen, source.read_bytes())
            controller = self.controller(folder)
            controller.inputs = {"source": {"path": str(source), "sha256": batch.digest(source)}}
            controller.archived = {"frozen": {"path": str(frozen), "sha256": batch.digest(frozen)}}
            controller.check_inputs()
            with self.assertRaisesRegex(ValueError, "Archived input changed"):
                batch.preserve(frozen, b"changed request")
            frozen.write_bytes(b"changed script")
            with self.assertRaisesRegex(ValueError, "input changed: frozen"):
                controller.check_inputs()

    def test_all_stills_precede_films_and_failed_seed_does_not_stop_other_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(Path(directory))
            events = []

            def still(seed):
                events.append(("still", seed))
                if seed == "0xbb":
                    raise RuntimeError("invalid mesh")

            controller.still = still
            controller.film = lambda seed: events.append(("film", seed))
            controller.lane(True)
            self.assertEqual(
                events,
                [
                    ("still", "0xaa"),
                    ("still", "0xbb"),
                    ("still", "0xcc"),
                    ("film", "0xaa"),
                    ("film", "0xcc"),
                ],
            )
            state = batch.read(Path(directory) / "status.json")["seeds"]
            self.assertEqual(state["0xbb"]["still"]["error"], "invalid mesh")
            self.assertEqual(state["0xcc"]["film"]["status"], "complete")

    def test_ready_seed_publishes_without_waiting_for_other_seed_and_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(Path(directory))
            for phase in ("normal", "still", "film"):
                controller.status("0xaa", phase, status="complete")
            controller.status("0xbb", "film", status="running")
            seen = []

            def publish(seed):
                seen.append(seed)
                raise RuntimeError("full decode failed")

            controller.publish = publish
            controller.publish_ready()
            controller.publish_ready()
            self.assertEqual(seen, ["0xaa"])
            self.assertEqual(controller.state["seeds"]["0xaa"]["publish"]["status"], "failed")
            self.assertNotIn("publish", controller.state["seeds"]["0xbb"])
            self.assertFalse(controller.state["complete"])

    def test_child_error_is_logged_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                batch.run_process(
                    [sys.executable, "-c", "print('broken frame'); raise SystemExit(7)"],
                    log,
                    threading.Event(),
                )
            self.assertIn("broken frame", log.read_text())
            self.assertIsInstance(json.loads(log.read_text().splitlines()[1][9:]), list)

    def test_cancel_terminates_child_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            cancel = threading.Event()
            timer = threading.Timer(0.1, cancel.set)
            timer.start()
            started = time.monotonic()
            with self.assertRaises(InterruptedError):
                batch.run_process(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    Path(directory) / "child.log",
                    cancel,
                )
            timer.join()
            self.assertLess(time.monotonic() - started, 3)

    def test_resume_does_not_accept_unowned_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "unknown-result").write_text("valuable")
            controller = self.controller(folder)
            with self.assertRaisesRegex(ValueError, "no archived request"):
                controller.prepare()
            self.assertEqual((folder / "unknown-result").read_text(), "valuable")

    def test_failed_lanes_make_collection_fail_even_when_children_return_success(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(Path(directory), ("0xaa",))
            controller.prepare = lambda: None
            controller.lane = lambda _gpu: None
            with self.assertRaisesRegex(RuntimeError, "failed seeds"):
                controller.run()
            self.assertFalse(batch.read(Path(directory) / "status.json")["complete"])


if __name__ == "__main__":
    unittest.main()
