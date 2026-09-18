"""Finite experiment reuse, process limits and media verification contracts."""

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.estuary_depth import experiment as exp


def render_fixture(folder, motion=0):
    folder.mkdir(parents=True, exist_ok=True)
    request = {"recipe": "a fixed source and studio"}
    exp.write(folder / "request.json", request)
    names = ["render.png", "render.exr", "scene.blend"]
    for index in range(motion):
        name = f"frames/{index:06d}.png"
        names.append(name)
        path = folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 128, 96))
    for name in names[:3]:
        (folder / name).write_bytes(name.encode())
    receipt = {
        "complete": True,
        "identity_sha256": hashlib.sha256(exp.encoded(request)).hexdigest(),
        "artifacts": {
            name: {"sha256": exp.digest(folder / name), "bytes": (folder / name).stat().st_size}
            for name in names
        },
    }
    exp.write(folder / "receipt.json", receipt)
    return receipt


def bound_fixture(root, cases, *, seed="0xaa", render_identity="paint-a", motion=0):
    """Create self-consistent archives bound to named experiment cases."""
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    (inputs / "recipes").mkdir(parents=True)
    (inputs / "bundle").mkdir()
    for name in ("render.py", "materials.py", "bundle/bundle.npz"):
        (inputs / name).write_text(name)
    source = {"seed": seed, "sha256": "s" * 64}
    preparation = {"inputs": {"render_identity": render_identity}}
    exp.write(
        inputs / "bundle/manifest.json",
        {
            "complete": True,
            "source": source,
            "request": preparation,
            "identity_sha256": hashlib.sha256(exp.encoded(preparation)).hexdigest(),
            "bundle": {
                "path": "bundle.npz",
                "sha256": exp.digest(inputs / "bundle/bundle.npz"),
                "bytes": (inputs / "bundle/bundle.npz").stat().st_size,
            },
        },
    )
    for name, recipe in cases.items():
        exp.write(inputs / f"recipes/{name}.json", recipe)
    files = {
        str(path.relative_to(inputs)): {"sha256": exp.digest(path)}
        for path in inputs.rglob("*")
        if path.is_file()
    }
    experiment = {"files": files, "motion_frames": motion or 1, "fps": 24}
    identity = hashlib.sha256(exp.encoded(experiment)).hexdigest()
    exp.write(root / "experiment-request.json", experiment)
    for name, recipe in cases.items():
        folder = root / name
        receipt = render_fixture(folder, motion)
        resolved = {"family": "relief", "name": name, "render": {"resolution": [128, 96]}, **recipe}
        request = {
            "recipe": resolved,
            "bundle_manifest_sha256": files["bundle/manifest.json"]["sha256"],
            "bundle_sha256": files["bundle/bundle.npz"]["sha256"],
            "renderer": {name: files[name]["sha256"] for name in ("render.py", "materials.py")},
            "motion": {
                "frames": motion or 1,
                "fps": 24,
                "source_fraction": 1.0,
                "semantics": "frozen completed painting; camera only",
            },
        }
        exp.write(folder / "request.json", request)
        receipt.update(
            identity_sha256=hashlib.sha256(exp.encoded(request)).hexdigest(),
            source=source,
            source_fraction=1.0,
            motion_frames=motion or 1,
        )
        exp.write(folder / "receipt.json", receipt)
        movie = None
        if motion:
            (folder / "film.mp4").write_bytes(b"certified movie bytes")
            movie = {
                "sha256": exp.digest(folder / "film.mp4"),
                "bytes": (folder / "film.mp4").stat().st_size,
                "frames": motion,
                "fps": 24,
                "resolution": [128, 96],
                "full_decode_verified": True,
            }
        exp.write(
            folder / "experiment-result.json",
            {
                "complete": True,
                "identity_sha256": identity,
                "render_receipt_sha256": exp.digest(folder / "receipt.json"),
                "movie": movie,
            },
        )
    return identity


class ExperimentTests(unittest.TestCase):
    def args(self, folder):
        return argparse.Namespace(
            blender=Path(sys.executable),
            render_script=folder / "render.py",
            bundle=folder / "bundle",
            recipes=folder / "recipes",
            output=folder / "output",
            workers=2,
            motion_frames=1,
            fps=24,
            baseline=None,
            ffmpeg=None,
            ffprobe=None,
        )

    def inputs(self, folder):
        args = self.args(folder)
        args.render_script.write_text("renderer version one")
        (folder / "materials.py").write_text("immutable optics")
        args.bundle.mkdir()
        (args.bundle / "bundle.npz").write_bytes(b"verified bundle bytes")
        request = {"source": "frozen"}
        exp.write(
            args.bundle / "manifest.json",
            {
                "complete": True,
                "request": request,
                "identity_sha256": hashlib.sha256(exp.encoded(request)).hexdigest(),
                "bundle": {
                    "path": "bundle.npz",
                    "bytes": (args.bundle / "bundle.npz").stat().st_size,
                    "sha256": exp.digest(args.bundle / "bundle.npz"),
                },
            },
        )
        args.recipes.mkdir()
        exp.write(args.recipes / "relief.json", {"name": "Porcelain relief", "mode": "relief"})
        return args

    def test_archive_freezes_renderer_materials_bundle_and_recipes_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.inputs(Path(directory))
            with patch.object(exp, "execute") as execute:
                exp.run(args)
                self.assertEqual(
                    (args.output / "inputs/materials.py").read_text(), "immutable optics"
                )
                self.assertEqual(
                    (args.output / "inputs/render.py").read_text(), "renderer version one"
                )
                args.workers = 3
                exp.run(args)
                self.assertEqual(execute.call_count, 2)
                args.render_script.write_text("changed renderer")
                with self.assertRaisesRegex(ValueError, "different inputs"):
                    exp.run(args)
                self.assertEqual(execute.call_count, 2)

    def test_completed_case_reuse_requires_exact_job_identity_and_unchanged_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = bound_fixture(root, {"a": {"name": "Study a"}})
            folder = root / "a"
            self.assertTrue(exp.finished(folder, identity))
            with self.assertRaisesRegex(ValueError, "different experiment inputs"):
                exp.finished(folder, "job-b")
            (folder / "scene.blend").write_bytes(b"changed scene")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                exp.finished(folder, identity)

    def test_self_consistent_render_cannot_be_swapped_between_case_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = bound_fixture(
                root,
                {
                    "a": {"name": "Study a", "camera": {"tilt": 10}},
                    "b": {"name": "Study b", "camera": {"tilt": 20}},
                },
            )
            self.assertTrue(exp.finished(root / "a", identity))
            self.assertTrue(exp.finished(root / "b", identity))
            shutil.rmtree(root / "b")
            shutil.copytree(root / "a", root / "b")
            with self.assertRaisesRegex(ValueError, "own archived input"):
                exp.finished(root / "b", identity)
            request = exp.read(root / "a/request.json")
            request["bundle_sha256"] = "unrelated bundle"
            exp.write(root / "a/request.json", request)
            with self.assertRaisesRegex(ValueError, "different bundle or renderer"):
                exp.verify_case_inputs(root / "a", identity)

    def test_blender_command_is_explicit_bounded_and_adds_motion_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            recipe = Path("a.json")
            command = exp.blender_command(args, recipe, args.output / "a")
            self.assertEqual(command[command.index("--threads") + 1], "4")
            self.assertIn("--factory-startup", command)
            self.assertNotIn("--motion-frames", command)
            args.motion_frames = 48
            command = exp.blender_command(args, recipe, args.output / "a")
            self.assertEqual(command[-4:], ["--motion-frames", "48", "--fps", "24"])

    def test_one_failed_blender_case_does_not_stop_other_cases_or_exceed_worker_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            (args.output / "logs").mkdir(parents=True)
            recipes = [Path(f"{name}.json") for name in ("a", "b", "c")]
            popen = subprocess.Popen
            processes, maxima, published = [], [], []

            def spawn(command, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                self.assertNotIn("shell", kwargs)
                child = popen(command, **kwargs)
                processes.append(child)
                maxima.append(sum(item.poll() is None for item in processes))
                return child

            def command(_args, recipe, folder):
                program = (
                    "import sys,time;from pathlib import Path;"
                    "from tools.estuary_depth.test_experiment import render_fixture;"
                    "time.sleep(.06);render_fixture(Path(sys.argv[1]));"
                    f"raise SystemExit({7 if recipe.stem == 'b' else 0})"
                )
                return [sys.executable, "-c", program, str(folder)]

            with (
                patch.object(exp, "blender_command", side_effect=command),
                patch.object(exp, "verify_case_inputs"),
                patch.object(exp.subprocess, "Popen", side_effect=spawn),
                patch(
                    "tools.estuary_depth.gallery.build_gallery",
                    side_effect=lambda _out, ready, _baseline: published.append(
                        [path.stem for path in ready]
                    ),
                ),
                self.assertRaises(RuntimeError),
            ):
                exp.execute(args, {"files": {}}, recipes)
            self.assertLessEqual(max(maxima), 2)
            self.assertEqual(len(processes), 3)
            self.assertEqual(published, [["a"], ["a", "c"]])
            status = exp.read(args.output / "status.json")
            self.assertFalse(status["complete"])
            self.assertEqual(status["cases"]["b"]["status"], "failed")
            self.assertEqual(status["cases"]["c"]["status"], "complete")

    def test_incomplete_attempt_is_preserved_before_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            (args.output / "logs").mkdir(parents=True)
            folder = args.output / "a"
            folder.mkdir()
            (folder / "valuable-partial.blend").write_bytes(b"keep this")
            with (
                patch.object(
                    exp,
                    "blender_command",
                    return_value=[sys.executable, "-c", "raise SystemExit(1)"],
                ),
                self.assertRaises(RuntimeError),
            ):
                exp.execute(args, {"files": {}}, [Path("a.json")])
            preserved = list(args.output.glob("a.incomplete-*"))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / "valuable-partial.blend").read_bytes(), b"keep this")

    def test_encoding_requires_exact_full_decode_before_publishing_movie(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            render_fixture(folder, motion=2)
            (folder / "film.partial.mp4").write_bytes(b"keep incomplete media")
            probe = json.dumps(
                {
                    "streams": [
                        {
                            "width": 128,
                            "height": 96,
                            "nb_read_frames": "2",
                            "avg_frame_rate": "24/1",
                        }
                    ]
                }
            )
            with (
                patch.object(exp, "capture", side_effect=["", probe, "frame=1\nprogress=end\n"]),
                self.assertRaisesRegex(ValueError, "full decode"),
            ):
                exp.encode_movie(folder, 2, 24, "ffmpeg", "ffprobe")
            self.assertFalse((folder / "film.mp4").exists())
            self.assertTrue((folder / "film.partial.mp4").exists())

    def test_interrupted_child_is_terminated_and_reaped(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        communicate, first = child.communicate, True

        def interrupt(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt
            return communicate(*args, **kwargs)

        try:
            with (
                patch.object(exp.subprocess, "Popen", return_value=child),
                patch.object(child, "communicate", side_effect=interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                exp.capture([sys.executable, "owned process"])
            self.assertIsNotNone(child.poll())
            self.assertLess(child.returncode, 0)
        finally:
            if child.poll() is None:
                child.kill()
            communicate()


if __name__ == "__main__":
    unittest.main()
