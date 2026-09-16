"""Film orchestration contracts; child builders, renderers and encoders are mocked."""

import copy
import hashlib
import importlib.util
import math
import subprocess
import tempfile
import threading
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location("remaining_film", Path(__file__).with_name("film.py"))
film = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(film)
adapter = film.load_adapter(Path(__file__).with_name("render.py"))


class FilmTests(unittest.TestCase):
    def test_parallel_completion_keeps_published_frames_in_timeline_order(self):
        release_first = threading.Event()
        cancel = threading.Event()
        completed = {}
        completion_order = []
        published = []

        def job(index):
            if index == 0 and not release_first.wait(timeout=2):
                raise RuntimeError("Test controller did not release the first frame")
            return {"index": index}

        def accept(record):
            completion_order.append(record["index"])
            completed[record["index"]] = record
            published.append([completed[i]["index"] for i in sorted(completed)])
            if record["index"] == 1:
                release_first.set()

        film.run_phase([0, 1], 2, job, accept, cancel)
        self.assertEqual(completion_order, [1, 0])
        self.assertEqual(published[-1], [0, 1])
        self.assertFalse(cancel.is_set())

    def test_parallel_failure_cancels_owned_sibling_work(self):
        cancel = threading.Event()

        def job(index):
            if index == 1:
                raise RuntimeError("synthetic frame failure")
            if not cancel.wait(timeout=2):
                raise RuntimeError("Sibling was not cancelled")
            return {"index": index}

        with self.assertRaisesRegex(RuntimeError, "synthetic frame failure"):
            film.run_phase([0, 1], 2, job, lambda _: None, cancel)
        self.assertTrue(cancel.is_set())

    def test_subprocess_cooperatively_terminates_after_controller_cancellation(self):
        cancel = threading.Event()
        child = mock.Mock()
        child.poll.return_value = None

        def wait(timeout=None):
            if timeout == 0.5:
                cancel.set()
                raise subprocess.TimeoutExpired("mock child", timeout)
            return 0

        child.wait.side_effect = wait
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(film.subprocess, "Popen", return_value=child),
            self.assertRaises(InterruptedError),
        ):
            film.run_process(["mock child"], Path(temporary) / "child.log", cancel)
        child.terminate.assert_called_once()

    def test_source_endpoints_and_orbit_use_one_final_mesh(self):
        camera = {"position": [3.0, -4.0, 2.0], "target": [0.5, 0.25, -0.5]}
        frames = film.frame_plan(72, 18, 35.0, camera)
        self.assertEqual(frames[0]["source_fraction"], 0.0)
        self.assertEqual(frames[71]["source_fraction"], 1.0)
        self.assertTrue(
            all(a["source_fraction"] < b["source_fraction"] for a, b in pairwise(frames[:72]))
        )
        self.assertTrue(
            all(frame["camera_position"] == camera["position"] for frame in frames[:72])
        )
        self.assertTrue(
            all(
                frame["source_fraction"] == 1.0 and frame["geometry_index"] == 71
                for frame in frames[72:]
            )
        )
        first_radius = math.hypot(camera["position"][0] - 0.5, camera["position"][1] - 0.25)
        for frame in frames[72:]:
            position = frame["camera_position"]
            self.assertAlmostEqual(math.hypot(position[0] - 0.5, position[1] - 0.25), first_radius)
            self.assertEqual(position[2], 2.0)

    def test_holds_are_explicit_duplicate_slots_with_exact_encoded_duration(self):
        frames = film.frame_plan(24, 6, 35.0, {"position": [3, -4, 2], "target": [0, 0, 0]})
        slots = film.encoding_timeline(frames, 24, 1.0, 2.0)
        self.assertEqual(len(slots), 24 + 30 + 48)
        self.assertTrue(
            all(s["kind"] == "start_hold" and s["render_frame"] == 0 for s in slots[:24])
        )
        self.assertTrue(
            all(s["kind"] == "final_hold" and s["render_frame"] == 29 for s in slots[-48:])
        )
        self.assertEqual([s["render_frame"] for s in slots[24:54]], list(range(30)))
        self.assertAlmostEqual(slots[-1]["time_seconds"] + 1 / 24, len(slots) / 24)

    def test_studio_requires_fixed_ground_even_dimensions_and_four_workers(self):
        supplied = copy.deepcopy(adapter.DEFAULTS)
        with self.assertRaises(ValueError):
            film.fixed_studio(adapter, supplied)
        supplied["ground"]["height_units"] = -2.0
        supplied["render"]["threads"] = 12
        resolved = film.fixed_studio(adapter, supplied)
        self.assertEqual(resolved["render"]["threads"], 4)
        self.assertEqual(resolved["ground"]["height_units"], -2.0)
        self.assertEqual(supplied["render"]["threads"], 12)
        supplied["render"]["resolution"][0] = 321
        with self.assertRaises(ValueError):
            film.fixed_studio(adapter, supplied)

    def test_invalid_frame_counts_or_hold_times_fail(self):
        camera = {"position": [3, -4, 2], "target": [0, 0, 0]}
        for count, turns in [(23, 0), (722, 0), (24, -1), (24, 241)]:
            with self.assertRaises(ValueError):
                film.frame_plan(count, turns, 35.0, camera)
        with self.assertRaises(ValueError):
            film.encoding_timeline(film.frame_plan(24, 0, 0, camera), 24, float("nan"), 2)

    def test_mesh_receipt_must_match_frozen_source_time_and_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            mesh = folder / "mesh.ply"
            mesh.write_bytes(b"synthetic mesh fixture")
            recipe = {"source_fraction": 0.5, "resolution": 144, "field": {}}
            request = {"inputs": {"orbit": {"sha256": "orbit"}, "builder": {"sha256": "builder"}}}
            record = {
                "identity": {
                    "recipe": recipe,
                    "orbit_sha256": "orbit",
                    "executable_sha256": "builder",
                },
                "mesh_sha256": film.digest(mesh),
            }
            film.write_json(folder / "build.json", record)
            film.write_json(folder / "recipe.json", recipe)
            film.verify_build(folder, recipe, request)
            with self.assertRaises(ValueError):
                film.verify_build(folder, {**recipe, "source_fraction": 0.6}, request)
            mesh.write_bytes(b"tampered mesh fixture")
            with self.assertRaises(ValueError):
                film.verify_build(folder, recipe, request)

    def test_scene_verification_delegates_complete_artifact_hash_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            studio = {"camera": "fixture"}
            scene = {
                "config": studio,
                "mesh_sha256": "mesh",
                "view": "front",
                "script_sha256": "script",
                "runtime": {"binary_sha256": "blender"},
            }
            request = {
                "inputs": {"render_script": {"sha256": "script"}, "blender": {"sha256": "blender"}}
            }
            film.write_json(folder / "request.json", scene)
            artifacts = {}
            for name in ("render.png", "render.exr", "scene.blend", "recipe.json"):
                (folder / name).write_bytes(name.encode())
                artifacts[name] = {"sha256": film.digest(folder / name)}
            identity = hashlib.sha256(adapter.encoded(scene)).hexdigest()
            film.write_json(
                folder / "receipt.json",
                {"complete": True, "identity_sha256": identity, "artifacts": artifacts},
            )
            film.verify_scene(adapter, folder, studio, "mesh", request)
            (folder / "render.exr").write_bytes(b"changed linear master")
            with self.assertRaises(ValueError):
                film.verify_scene(adapter, folder, studio, "mesh", request)

    def test_encoding_counts_decoded_frames_and_reuses_only_valid_movie_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "logs").mkdir()
            args = SimpleNamespace(ffmpeg=Path("/mock/ffmpeg"), fps=24)
            frames = film.frame_plan(24, 0, 0, {"position": [3, -4, 2], "target": [0, 0, 0]})
            request = {"encoding_timeline": film.encoding_timeline(frames, 24, 0.125, 0.25)}
            manifest = {"request_sha256": "request", "frames": []}
            for i in range(24):
                path = output / f"image_{i}.png"
                path.write_bytes(f"synthetic image {i}".encode())
                manifest["frames"].append({"image": path.name, "png_sha256": film.digest(path)})
            commands = []

            def process(command, log):
                commands.append(command)
                if "-c:v" in command:
                    self.assertEqual(command[command.index("-crf") + 1], "18")
                    self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
                    Path(command[-1]).write_bytes(b"synthetic encoded movie")
                    log.write_text("mock encoder\n")
                else:
                    log.write_text(f"frame={len(request['encoding_timeline'])}\nprogress=end\n")
                return 0.125

            with mock.patch.object(film, "run_process", side_effect=process):
                receipt = film.encode_film(args, output, request, manifest)
            self.assertEqual(receipt["frames"], 33)
            self.assertTrue(receipt["full_decode_verified"])
            self.assertEqual(len(commands), 2)
            with mock.patch.object(film, "run_process") as runner:
                self.assertEqual(film.encode_film(args, output, request, manifest), receipt)
                runner.assert_not_called()
            receipt["full_decode_verified"] = False
            film.write_json(output / "movie.json", receipt)
            with self.assertRaises(ValueError):
                film.encode_film(args, output, request, manifest)


if __name__ == "__main__":
    unittest.main()
