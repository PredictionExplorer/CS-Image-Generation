"""Lossless image, canonical timing and immutable archive recovery contracts."""

import argparse
import copy
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tools.estuary import run as renderer
from tools.estuary.optics import linear_to_srgb


def decode_png(path):
    """Independently decode the writer's lossless, unfiltered truecolor rows."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG signature")
    chunks, payload, offset = {}, b"", 8
    while offset < len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        contents = data[offset + 8 : offset + 8 + size]
        crc = struct.unpack(">I", data[offset + 8 + size : offset + 12 + size])[0]
        if crc != zlib.crc32(kind + contents):
            raise ValueError("PNG CRC")
        chunks[kind] = contents
        if kind == b"IDAT":
            payload += contents
        offset += size + 12
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    if (color, compression, filtering, interlace) != (2, 0, 0, 0):
        raise ValueError("PNG encoding")
    rows = np.frombuffer(zlib.decompress(payload), dtype="u1").reshape(
        height, width * 3 * (depth // 8) + 1
    )
    if np.any(rows[:, 0]):
        raise ValueError("Unexpected row filter")
    decoded = np.frombuffer(rows[:, 1:].tobytes(), dtype=">u2" if depth == 16 else "u1")
    return decoded.reshape(height, width, 3), chunks


class RunnerTests(unittest.TestCase):
    def test_render_identity_tracks_runtime_and_shaders_but_ignores_gallery_and_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in renderer.RUNTIME_FILES:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            for name in ("gallery.py", "flow_reference.py", "test_run.py"):
                (folder / name).write_text("not imported by the renderer")
            with patch.object(renderer, "HERE", folder):
                original = renderer.code_identity()
                self.assertEqual(set(original), set(renderer.RUNTIME_FILES))
                self.assertIn("requirements.txt", original)
                for name in ("gallery.py", "flow_reference.py", "test_run.py"):
                    (folder / name).write_text("editorial changes")
                self.assertEqual(renderer.code_identity(), original)
                for name in ("engine.py", "shaders/flow.glsl"):
                    (folder / name).write_text("changed rendering behavior")
                    self.assertNotEqual(renderer.code_identity()[name], original[name])
                (folder / "shaders/additional.glsl").write_text("new shader")
                self.assertIn("shaders/additional.glsl", renderer.code_identity())
                (folder / "requirements.txt").unlink()
                with self.assertRaisesRegex(
                    ValueError, "runtime file is missing: requirements.txt"
                ):
                    renderer.code_identity()

    def test_rgb16_roundtrip_retains_precision_color_tags_and_row_order(self):
        colors = np.array(
            [[[0, 0.0031308, 0.5], [1, 0.124578, 0.25]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poster.png"
            renderer.write_png(path, colors)
            actual, chunks = decode_png(path)
            expected = np.rint(linear_to_srgb(colors) * 65535).astype(np.uint16)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(chunks[b"sRGB"], b"\0")
            self.assertEqual(struct.unpack(">I", chunks[b"gAMA"])[0], 45455)
            self.assertTrue(np.any(actual % 257), "The master must not be an enlarged RGB8 image")
            renderer.write_png(path, colors)
            with self.assertRaisesRegex(ValueError, "both files preserved"):
                renderer.write_png(path, np.zeros_like(colors))
            np.testing.assert_array_equal(decode_png(path)[0], expected)

    def test_invalid_pixel_data_never_creates_a_master(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            for values in (np.zeros((2, 3)), np.full((2, 2, 3), np.nan), np.full((2, 2, 3), 1.01)):
                with self.assertRaises(ValueError):
                    renderer.write_png(path, values)
                self.assertFalse(path.exists())

    def test_canonical_frame_timing_includes_both_endpoints_without_rounding(self):
        plan = renderer.frame_plan(3600, 301)
        self.assertEqual(len(plan), 301)
        self.assertEqual((plan[0], plan[-1]), (0, 3600))
        self.assertEqual(set(np.diff(plan)), {12})
        for steps, frames in ((3601, 301), (2, 4), (30, 1), (True, 2)):
            with self.assertRaises(ValueError):
                renderer.frame_plan(steps, frames)

    def test_trailing_exposures_preserve_exact_endpoints_without_overlap_or_overshoot(self):
        endpoints = renderer.frame_plan(7200, 901)
        single = renderer.exposure_plan(endpoints, 1)
        self.assertEqual(single, [[point] for point in endpoints])
        exposures = renderer.exposure_plan(endpoints, 8)
        self.assertEqual(exposures[0], [0])
        self.assertEqual(exposures[1], list(range(1, 9)))
        self.assertEqual(exposures[-1], list(range(7193, 7201)))
        for previous, current, endpoint in zip(
            exposures[:-1], exposures[1:], endpoints[1:], strict=True
        ):
            self.assertLess(previous[-1], current[0])
            self.assertEqual(current[-1], endpoint)
        for points, samples in (
            ([0, 4, 8], 5),
            ([0, 8, 16], 0),
            ([0, 8], 9),
            ([0, 8], True),
            ([1, 8], 2),
            ([0, 8, 7], 1),
        ):
            with self.assertRaises(ValueError):
                renderer.exposure_plan(points, samples)

    def test_temporal_average_is_linear_light_and_leaves_sharp_endpoint_state(self):
        class FakeEngine:
            step = 0

            def advance_to(self, step):
                if step < self.step:
                    raise ValueError("Cannot rewind")
                self.step = step

            def render(self, width, height):
                value = 0.0 if self.step == 1 else 1.0
                return np.full((height, width, 3), value, dtype=np.float32)

        engine = FakeEngine()
        averaged = renderer.render_exposure(engine, [1, 2], 2, 1)
        np.testing.assert_array_equal(averaged, np.full((1, 2, 3), 0.5, dtype=np.float32))
        self.assertEqual(averaged.dtype, np.float32)
        self.assertEqual(engine.step, 2)
        self.assertGreater(float(linear_to_srgb(averaged)[0, 0, 0]), 0.73)
        np.testing.assert_array_equal(engine.render(2, 1), np.ones((1, 2, 3), dtype=np.float32))

    def test_completed_archive_requires_exact_identity_and_every_required_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = []
            for name in (
                "poster.png",
                "linear.npy",
                "final-state.npy",
                "recipe.json",
                "inputs/source.orbit",
            ):
                path = folder / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(name.encode())
                records.append(renderer.artifact(path, folder))
            renderer.write_json(
                folder / "receipt.json",
                {"identity_sha256": "a", "complete": True, "mode": "still", "artifacts": records},
            )
            self.assertTrue(renderer.completed(folder, "a"))
            with self.assertRaisesRegex(ValueError, "different render identity"):
                renderer.completed(folder, "b")
            (folder / "linear.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                renderer.completed(folder, "a")
            with self.assertRaisesRegex(ValueError, "inside the archive"):
                renderer.checked_artifact(folder, {"path": "../private"})

    def test_checkpoint_restores_exact_state_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "frames").mkdir()
            renderer.write_png(folder / "frames/000000.png", np.ones((2, 2, 3)), 8)
            state = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
            engine = SimpleNamespace(
                read_state=lambda: state, restore=Mock(), internal_steps=7, maximum_courant=1.2
            )
            records = [renderer.artifact(folder / "frames/000000.png", folder)]
            renderer.checkpoint(folder, "identity", engine, 0, 0, records)
            first, restored = renderer.restore_checkpoint(folder, "identity", engine, [0, 12])
            self.assertEqual((first, restored), (1, records))
            np.testing.assert_array_equal(engine.restore.call_args.args[0], state)
            self.assertEqual(engine.restore.call_args.args[1], 0)
            self.assertEqual(
                engine.restore.call_args.kwargs, {"internal_steps": 7, "maximum_courant": 1.2}
            )
            checkpoint = renderer.read_json(folder / "checkpoint.json")
            del checkpoint["internal_steps"]
            renderer.write_json(folder / "checkpoint.json", checkpoint)
            with self.assertRaisesRegex(ValueError, "cumulative transport diagnostics"):
                renderer.restore_checkpoint(folder, "identity", engine, [0, 12])
            checkpoint["internal_steps"] = 7
            renderer.write_json(folder / "checkpoint.json", checkpoint)
            (folder / "frames/000000.png").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                renderer.restore_checkpoint(folder, "identity", engine, [0, 12])

    def test_checkpoint_retention_preserves_current_state_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "frames").mkdir()
            engine = SimpleNamespace(
                read_state=lambda: np.ones((2, 2, 4), dtype=np.float32),
                restore=Mock(),
                internal_steps=3,
                maximum_courant=0.5,
            )
            records = []
            for index in range(4):
                frame = folder / "frames" / f"{index:06d}.png"
                renderer.write_png(frame, np.ones((2, 2, 3)), 8)
                records.append(renderer.artifact(frame, folder))
                renderer.checkpoint(folder, "identity", engine, index, index, records, retention=2)
            self.assertEqual(
                sorted(path.name for path in (folder / "checkpoints").iterdir()),
                ["state-000002.npy", "state-000003.npy"],
            )
            first, restored = renderer.restore_checkpoint(
                folder, "identity", engine, list(range(5))
            )
            self.assertEqual((first, restored), (4, records))
            unrelated = folder / "checkpoints/state-not-owned.npy"
            unrelated.write_bytes(b"keep")
            future = folder / "checkpoints/state-999999.npy"
            future.write_bytes(b"keep")
            renderer.checkpoint(folder, "identity", engine, 3, 3, records, retention=2)
            self.assertEqual(unrelated.read_bytes(), b"keep")
            self.assertEqual(future.read_bytes(), b"keep")

    def test_failed_checkpoint_publication_does_not_retire_recovery_states(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            engine = SimpleNamespace(
                read_state=lambda: np.ones((2, 2, 4), dtype=np.float32),
                internal_steps=1,
                maximum_courant=0.5,
            )
            for index in range(3):
                renderer.checkpoint(folder, "identity", engine, index, index, [])
            previous = (folder / "checkpoint.json").read_bytes()
            with (
                patch.object(renderer, "write_json", side_effect=OSError("full disk")),
                self.assertRaises(OSError),
            ):
                renderer.checkpoint(folder, "identity", engine, 3, 3, [], retention=2)
            self.assertEqual((folder / "checkpoint.json").read_bytes(), previous)
            self.assertTrue((folder / "checkpoints/state-000000.npy").exists())
            self.assertTrue((folder / "checkpoints/state-000002.npy").exists())

    def test_checkpoint_retention_rejects_unsafe_counts_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for value in (0, 1, -1, True, 2.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    renderer.checkpoint(folder, "identity", Mock(), 0, 0, [], retention=value)
            self.assertFalse((folder / "checkpoints").exists())

    def test_incomplete_archive_without_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "valuable-partial").write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "preserved"):
                renderer.restore_checkpoint(folder, "identity", Mock(), [0, 1])
            self.assertEqual((folder / "valuable-partial").read_bytes(), b"keep")

    def test_still_archive_reuses_only_unchanged_source_recipe_hardware_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            orbit = folder / "source.orbit"
            orbit.write_bytes(b"source bytes")
            source = SimpleNamespace(
                path=orbit,
                sha256=renderer.digest(orbit),
                seed="0xaa",
                metadata={"seed": "0xaa", "samples": 5, "projection": "fixed"},
            )
            recipe = {
                "simulation": {"resolution": [4, 4], "steps": 12},
                "render": {"resolution": [4, 4], "frames": 3, "fps": 2},
                "projection": {},
            }
            fake = ModuleType("tools.estuary.engine")
            engine = SimpleNamespace(
                metadata={"gpu": "test"},
                advance_to=Mock(),
                render=lambda w, h: np.full((h, w, 3), 0.25, dtype=np.float32),
                read_state=lambda: np.zeros((4, 4, 4), dtype=np.float32),
                internal_steps=24,
                maximum_courant=1.4,
                close=Mock(),
            )
            fake.Engine = Mock(return_value=engine)
            args = argparse.Namespace(
                orbit=orbit,
                recipe=folder / "recipe.json",
                output=folder / "render",
                still_only=True,
                resume=False,
                backend="test",
                ffmpeg=None,
                ffprobe=None,
            )
            with (
                patch.dict("sys.modules", {"tools.estuary.engine": fake}),
                patch("tools.estuary.source.Source.read", return_value=source),
                patch("tools.estuary.recipe.read_recipe", return_value=recipe),
                patch.object(renderer, "code_identity", return_value={}),
            ):
                renderer.run(args)
                engine.close.assert_called_once_with()
                receipt = renderer.read_json(args.output / "receipt.json")
                self.assertTrue(receipt["complete"])
                self.assertEqual(receipt["final_step"], 12)
                self.assertEqual(
                    receipt["diagnostics"], {"internal_steps": 24, "maximum_courant": 1.4}
                )
                engine.advance_to.assert_called_once_with(12)
                args.resume = True
                renderer.run(args)
                self.assertEqual(engine.advance_to.call_count, 1)
                self.assertEqual(engine.close.call_count, 2)
                changed = copy.deepcopy(recipe)
                changed["simulation"]["steps"] = 24
                with (
                    patch("tools.estuary.recipe.read_recipe", return_value=changed),
                    self.assertRaisesRegex(ValueError, "identity differs"),
                ):
                    renderer.run(args)
                self.assertEqual(engine.close.call_count, 3)
                before = renderer.digest(args.output / "poster.png")
                receipt["complete"] = False
                renderer.write_json(args.output / "receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, "Incomplete still preserved"):
                    renderer.run(args)
                self.assertEqual(renderer.digest(args.output / "poster.png"), before)
                self.assertEqual(engine.close.call_count, 4)
                with (
                    patch.object(renderer, "_render_archive", side_effect=KeyboardInterrupt),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    renderer.run(args)
                self.assertEqual(engine.close.call_count, 5)

    def test_interrupted_encoder_is_terminated_and_reaped(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        communicate = child.communicate
        first = True

        def interrupt_then_communicate(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt("stop encoder")
            return communicate(*args, **kwargs)

        try:
            with (
                patch.object(renderer.subprocess, "Popen", return_value=child) as spawn,
                patch.object(child, "communicate", side_effect=interrupt_then_communicate),
                self.assertRaises(KeyboardInterrupt),
            ):
                renderer.run_child([sys.executable, "owned-child"])
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertIsNotNone(child.poll())
            self.assertLess(child.returncode, 0)
        finally:
            if child.poll() is None:
                child.kill()
            communicate()

    def test_cli_sigterm_uses_normal_interrupt_cleanup_and_restores_handler(self):
        previous = renderer.signal.getsignal(renderer.signal.SIGTERM)
        with (
            patch.object(renderer, "parser") as parse,
            patch.object(
                renderer, "run", side_effect=lambda _args: renderer.interrupted(None, None)
            ),
        ):
            parse.return_value.parse_args.return_value = object()
            self.assertEqual(renderer.main(), 130)
        self.assertEqual(renderer.signal.getsignal(renderer.signal.SIGTERM), previous)

    def test_movie_verification_failure_never_publishes_completed_movie(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            recipe = {"render": {"frames": 3, "fps": 2, "resolution": [128, 96]}}
            probe = SimpleNamespace(
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "width": 128,
                                "height": 96,
                                "nb_read_frames": "2",
                                "avg_frame_rate": "2/1",
                            }
                        ]
                    }
                )
            )

            def fake_encode(*_args):
                (folder / "film.partial.mp4").write_bytes(b"incomplete film")

            with (
                patch.object(renderer, "command", side_effect=fake_encode),
                patch.object(renderer, "run_child", return_value=probe),
                self.assertRaisesRegex(ValueError, "frame count"),
            ):
                renderer.encode_movie(folder, recipe, "ffmpeg", "ffprobe")
            self.assertFalse((folder / "film.mp4").exists())
            self.assertFalse((folder / "movie.json").exists())
            self.assertTrue((folder / "film.partial.mp4").exists())


if __name__ == "__main__":
    unittest.main()
