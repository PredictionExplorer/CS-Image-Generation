"""Bounded launch, independent failures, immutable resume and owned cancellation."""

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.estuary import batch


class BatchTests(unittest.TestCase):
    def args(self, folder, seeds=("0xaa", "0xbb", "0xcc"), workers=2):
        return argparse.Namespace(
            sources=folder / "sources",
            recipe=folder / "recipe.json",
            output=folder / "output",
            seeds=list(seeds),
            workers=workers,
            ffmpeg=None,
            ffprobe=None,
        )

    def test_resume_is_planned_only_for_existing_seed_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            self.assertNotIn("--resume", batch.child_command(args, "0xaa"))
            (args.output / "0xaa").mkdir(parents=True)
            command = batch.child_command(args, "0xaa")
            self.assertEqual(command[:3], [sys.executable, "-m", "tools.estuary.run"])
            self.assertEqual(command[-1], "--resume")
            self.assertNotIn("--resume", batch.child_command(args, "0xbb"))

    def test_worker_cap_and_failure_do_not_prevent_other_verified_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            (args.output / "logs").mkdir(parents=True)
            actual_popen = subprocess.Popen
            launched, maxima, galleries = [], [], []

            def spawn(command, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(kwargs["cwd"], batch.REPO)
                self.assertNotIn("shell", kwargs)
                child = actual_popen(command, **kwargs)
                launched.append(child)
                maxima.append(sum(item.poll() is None for item in launched))
                return child

            def command(_args, seed):
                status = 7 if seed == "0xbb" else 0
                return [
                    sys.executable,
                    "-c",
                    f"import time; time.sleep(.06); raise SystemExit({status})",
                ]

            with (
                patch.object(batch, "request_for", return_value={}),
                patch.object(batch, "child_command", side_effect=command),
                patch.object(batch.subprocess, "Popen", side_effect=spawn),
                patch.object(batch, "verify_seed") as verify,
                patch.object(
                    batch,
                    "build_gallery",
                    side_effect=lambda renders, _out: galleries.append(
                        [path.name for path in renders]
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "Some films failed"),
            ):
                batch.execute(args, {})
            self.assertLessEqual(max(maxima), 2)
            self.assertEqual(len(launched), 3)
            self.assertEqual([call.args[1] for call in verify.call_args_list], ["0xaa", "0xcc"])
            self.assertEqual(galleries, [["0xaa"], ["0xaa", "0xcc"]])
            status = batch.read_json(args.output / "status.json")
            self.assertFalse(status["complete"])
            self.assertEqual(status["seeds"]["0xbb"]["status"], "failed")
            self.assertEqual(status["seeds"]["0xcc"]["status"], "complete")

    def test_only_a_verified_receipt_can_reach_the_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), ("0xaa",), 1)
            (args.output / "logs").mkdir(parents=True)
            with (
                patch.object(batch, "request_for", return_value={}),
                patch.object(batch, "child_command", return_value=[sys.executable, "-c", "pass"]),
                patch.object(batch, "verify_seed", side_effect=ValueError("incomplete receipt")),
                patch.object(batch, "build_gallery") as publish,
                self.assertRaises(RuntimeError),
            ):
                batch.execute(args, {})
            publish.assert_not_called()
            self.assertIn(
                "incomplete receipt",
                batch.read_json(args.output / "status.json")["seeds"]["0xaa"]["error"],
            )

    def test_changed_batch_inputs_are_rejected_before_launch_and_workers_can_change(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), ("0xaa",))
            args.sources.mkdir()
            (args.sources / "0xaa.orbit").write_bytes(b"frozen source")
            args.recipe.write_text("{}")
            with patch.object(batch, "execute") as execute:
                batch.run(args)
                args.workers = 3
                batch.run(args)
                self.assertEqual(execute.call_count, 2)
                (args.sources / "0xaa.orbit").write_bytes(b"different source")
                with self.assertRaisesRegex(ValueError, "different inputs"):
                    batch.run(args)
                self.assertEqual(execute.call_count, 2)

    def test_invalid_seed_and_worker_inputs_never_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), ("../wrong",))
            with self.assertRaises(ValueError):
                batch.run(args)
            self.assertFalse(args.output.exists())
            args.seeds, args.workers = ["0xaa"], 4
            with self.assertRaises(ValueError):
                batch.run(args)
            self.assertFalse(args.output.exists())

    def test_cancellation_stops_runner_and_its_detached_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            marker = folder / "encoder.pid"
            program = (
                "import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
                "start_new_session=True); "
                "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
            )
            stream = (folder / "child.log").open("wb")
            child = subprocess.Popen(
                [sys.executable, "-c", program, str(marker)],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            encoder = None
            try:
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                encoder = int(marker.read_text())
                batch.stop_children({"0xaa": (child, stream)})
                self.assertIsNotNone(child.poll())
                self.assertTrue(stream.closed)
                deadline = time.monotonic() + 3
                state = ""
                while time.monotonic() < deadline:
                    info = subprocess.run(
                        ["ps", "-p", str(encoder), "-o", "stat="],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    state = info.stdout.strip()
                    if not state or state.startswith("Z"):
                        break
                    time.sleep(0.01)
                self.assertTrue(not state or state.startswith("Z"), state)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
                if encoder:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(encoder, signal.SIGKILL)
                stream.close()

    def test_sigterm_handler_cleans_up_without_changing_other_signal_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)
        with (
            patch.object(batch, "parser", return_value=Mock(parse_args=lambda: object())),
            patch.object(batch, "run", side_effect=lambda _args: batch.interrupted(None, None)),
        ):
            self.assertEqual(batch.main(), 130)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


if __name__ == "__main__":
    unittest.main()
