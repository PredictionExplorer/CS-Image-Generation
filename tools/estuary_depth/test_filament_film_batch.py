"""Film queue recovery and real archive validation with small CPU fixtures.

Only canonical paint dimensions are reduced. Source parsing, map preparation,
all receipt identities and cross-stage verification use the production code.
Encoded image/movie bytes are fixtures; the codec suites verify real encoding.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.recipe import validate_recipe
from tools.estuary.run import artifact as paint_artifact
from tools.estuary.run import code_identity, exposure_plan, frame_plan, write_png
from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_depth import experiment, film
from tools.estuary_depth import filament_film_batch as batch
from tools.estuary_depth.filament_motion import FPS, MOTION_FRAMES, make_formation_recipe
from tools.estuary_depth.filament_studies import REFERENCE_SEED
from tools.estuary_depth.render import camera_pose, motion_angles
from tools.estuary_studio.common import artifact, read, write


def small_formation(seed, option):
    result = make_formation_recipe(seed, option)
    result["simulation"]["resolution"] = [128, 96]
    result["render"]["resolution"] = [128, 96]
    return validate_recipe(result)


class FilmBatchTests(unittest.TestCase):
    def setUp(self):
        self.actual_root = batch.ROOT
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for target, name, value in (
            (batch, "ROOT", self.root),
            (batch, "make_formation_recipe", small_formation),
            (batch, "BUNDLE_RESOLUTION", (64, 48)),
            (batch, "MESH_RESOLUTION", (32, 24)),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        source = self.root / f"{REFERENCE_SEED}.orbit"
        write_orbit(source, orbit_points(), seed=REFERENCE_SEED)
        self.cohort = {"sources": [{"seed": REFERENCE_SEED, **artifact(source)}]}
        path = self.root / "tools/estuary_confluence/recipes/ten-seeds.json"
        path.parent.mkdir(parents=True)
        write(path, self.cohort)
        self.tools = {}
        for name in ("blender", "ffmpeg", "ffprobe"):
            path = self.root / name
            path.write_bytes(f"Pinned {name}".encode())
            self.tools[name] = path
        self.plan = self.make_plan()
        # Simulate the coordinator's immutable deployment while other tests or
        # development tasks may edit unrelated files in the live workspace.
        frozen = patch.object(
            batch, "runtime_identity", return_value=copy.deepcopy(self.plan["runtime"])
        )
        frozen.start()
        self.addCleanup(frozen.stop)
        self.output = self.root / "batch"
        self.output.mkdir()
        write(self.output / "plan.json", self.plan)
        for material_id in self.plan["materials"]:
            self.material_id = material_id
            self.material = self.output / "materials" / material_id
            self.build_material()
        self.folders = []
        for case in self.plan["cases"]:
            self.material_id = case["material_id"]
            self.material = self.output / "materials" / self.material_id
            folder = self.output / "cases" / case["id"]
            self.folders.append(folder)
            self.build_photo(case, folder, motion=False)
            self.build_photo(case, folder, motion=True)
            self.build_edit(case, folder)
            record, _ = batch._inspect_case(folder, self.plan, case)
            write(folder / "study.json", record)
        self.folder = self.folders[0]
        self.material_id = next(iter(self.plan["materials"]))
        self.material = self.output / "materials" / self.material_id

    def make_plan(self, **kwargs):
        defaults = {"options": ["control", "light-flat"]}
        if getattr(self, "study_family", None) is not None:
            defaults = {
                "study_family": self.study_family,
                "options": ["lacuna-banks", "folded-sash"],
            }
        return batch.make_plan(
            self.cohort,
            cohort_kind="legacy",
            source_root=self.root,
            **self.tools,
            **(kwargs if kwargs else defaults),
        )

    def build_material(self):
        material = self.plan["materials"][self.material_id]
        paint = self.material / "paint"
        (paint / "inputs").mkdir(parents=True)
        shutil.copyfile(material["source"], paint / "inputs/source.orbit")
        recipe = material["recipe"]
        source = Source.read(paint / "inputs/source.orbit", aspect=4 / 3, **recipe["projection"])
        steps = frame_plan(recipe["simulation"]["steps"], recipe["render"]["frames"])
        code = batch._runtime_contract(self.plan)["paint"]
        request = {
            "mode": "film",
            "recipe": recipe,
            "source": source.metadata,
            "code": code,
            "tools": {
                name: batch._tool(self.plan["tools"][name]) for name in ("ffmpeg", "ffprobe")
            },
            "frame_steps": steps,
            "exposure_steps": exposure_plan(steps, 4),
        }
        write(paint / "request.json", request)
        write(paint / "recipe.json", recipe)
        np.save(paint / "final-state.npy", np.full((96, 128, 4), 0.1, np.float32))
        np.save(paint / "linear.npy", np.full((96, 128, 3), 0.5, np.float32))
        for name in ("initial.png", "poster.png"):
            write_png(paint / name, np.full((96, 128, 3), 0.5, np.float32))
        (paint / "film.mp4").write_bytes(b"Certified formation fixture")
        write(
            paint / "movie.json",
            {
                "frames": 721,
                "fps": FPS,
                "resolution": [128, 96],
                "full_decode_verified": True,
                "artifact": paint_artifact(paint / "film.mp4", paint),
            },
        )
        names = [
            "initial.png",
            "poster.png",
            "final-state.npy",
            "linear.npy",
            "recipe.json",
            "inputs/source.orbit",
            "film.mp4",
            "movie.json",
        ]
        for name in code:
            target = paint / "inputs/code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.actual_root / "tools/estuary" / name, target)
            names.append(f"inputs/code/{name}")
        write(
            paint / "receipt.json",
            {
                "complete": True,
                "mode": "film",
                "identity_sha256": batch._sha(request),
                "source": source.metadata,
                "final_step": 7200,
                "source_fraction": 1.0,
                "artifacts": [paint_artifact(paint / name, paint) for name in names],
            },
        )
        batch.build_bundle(
            paint, self.material / "bundle", resolution=(64, 48), mesh_resolution=(32, 24)
        )

    def build_photo(self, case, folder, *, motion):
        stage, frames = ("motion", MOTION_FRAMES) if motion else ("photographs", 1)
        recipe = case["motion_recipe" if motion else "photo_recipe"]
        recipes = folder / f"{stage}-recipes"
        recipes.mkdir(parents=True)
        write(recipes / "00-painting.json", recipe)
        args = argparse.Namespace(
            recipes=recipes,
            bundle=self.material / "bundle",
            blender=self.tools["blender"],
            render_script=self.actual_root / "tools/estuary_depth/render.py",
            baseline=None,
            motion_frames=frames,
            fps=FPS,
            ffmpeg=self.tools["ffmpeg"],
            ffprobe=self.tools["ffprobe"],
        )
        exp, files, _ = experiment.make_request(args)
        target = folder / stage / "00-painting"
        target.mkdir(parents=True)
        write(target.parent / "experiment-request.json", exp)
        for name, source in files.items():
            output = target.parent / "inputs" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        bundle = read(self.material / "bundle/manifest.json")
        request = {
            "recipe": recipe,
            "bundle_manifest_sha256": artifact(self.material / "bundle/manifest.json")["sha256"],
            "bundle_sha256": bundle["bundle"]["sha256"],
            "renderer": batch._runtime_contract(self.plan)["photo"],
            "motion": {"frames": frames, "fps": FPS, "source_fraction": 1.0},
        }
        write(target / "request.json", request)
        camera = []
        for index in range(frames):
            tilt, azimuth = motion_angles(recipe["camera"], index, frames)
            camera.append(
                {
                    "frame": index,
                    "tilt_degrees": tilt,
                    "azimuth_degrees": azimuth,
                    "matrix_world": camera_pose(
                        tilt, azimuth, [*recipe["camera"]["target"], 0.001]
                    ).tolist(),
                }
            )
        write(target / "camera.json", camera)
        for name in ("render.png", "render.exr", "scene.blend"):
            (target / name).write_bytes(name.encode())
        write(
            target / "receipt.json",
            {
                "complete": True,
                "identity_sha256": batch._sha(request),
                "source": bundle["source"],
                "source_fraction": 1.0,
                "history_fractions": [1.0],
                "motion_frames": frames,
                "artifacts": {
                    name: artifact(target / name)
                    for name in ("render.png", "render.exr", "scene.blend", "camera.json")
                },
            },
        )
        movie = None
        if motion:
            (target / "film.mp4").write_bytes(b"Certified orbit fixture")
            movie = {
                **artifact(target / "film.mp4"),
                "frames": frames,
                "fps": FPS,
                "resolution": recipe["render"]["resolution"],
                "full_decode_verified": True,
            }
        write(
            target / "experiment-result.json",
            {
                "complete": True,
                "case": "00-painting",
                "identity_sha256": batch._sha(exp),
                "render_receipt_sha256": artifact(target / "receipt.json")["sha256"],
                "movie": movie,
            },
        )

    def build_edit(self, case, folder):
        paths = batch._paths(self.output, case)
        source, inputs, timeline = film.validate_inputs(paths["paint"], paths["motion"])
        edit = paths["film"]
        edit.mkdir()
        (edit / "inputs").mkdir()
        for name, value in inputs.items():
            shutil.copyfile(value["path"], edit / "inputs" / f"{name}.mp4")
        request = {
            "source": source,
            "inputs": inputs,
            "timeline": timeline,
            "editor_sha256": self.plan["runtime"]["estuary_depth"]["film.py"],
            "tools": {
                name: batch._tool(self.plan["tools"][name]) for name in ("ffmpeg", "ffprobe")
            },
        }
        write(edit / "request.json", request)
        write(edit / "command.json", ["fixture"])
        frames = timeline["output_frames"]
        write(
            edit / "probe.json",
            {
                "nb_read_frames": str(frames),
                "avg_frame_rate": "24/1",
                "duration_ts": frames,
                "time_base": "1/24",
                "width": 1920,
                "height": 1440,
                "color_primaries": "bt709",
                "color_transfer": "iec61966-2-1",
                "color_space": "bt709",
                "color_range": "tv",
            },
        )
        (edit / "decode.txt").write_text(f"frame={frames}\nprogress=end\n")
        (edit / "film.mp4").write_bytes(b"Certified complete edit fixture")
        names = [
            "request.json",
            "command.json",
            "probe.json",
            "decode.txt",
            "film.mp4",
            "inputs/formation.mp4",
            "inputs/orbit.mp4",
        ]
        write(
            edit / "receipt.json",
            {
                "complete": True,
                "identity_sha256": batch._sha(request),
                "source": source,
                "movie": {
                    "path": "film.mp4",
                    **artifact(edit / "film.mp4"),
                    "frames": frames,
                    "fps": FPS,
                    "resolution": [1920, 1440],
                    "full_decode_verified": True,
                },
                "artifacts": {name: artifact(edit / name) for name in names},
            },
        )

    def test_complete_pair_binds_all_stages_and_lighting_reuses_one_material(self):
        self.assertEqual(len(self.plan["cases"]), 2)
        self.assertEqual(len(self.plan["materials"]), 1)
        states = []
        for folder in self.folders:
            record, paths = batch.verify_case(folder)
            self.assertTrue(record["complete"])
            self.assertEqual(record["movie"]["frames"], 817)
            self.assertEqual(record["timeline"]["crossfade_start_frame"], 721)
            self.assertEqual(Path(paths["material"]), self.material)
            states.append(record["final_state_sha256"])
        self.assertEqual(len(set(states)), 1)

    def test_backfill_selection_is_exact_and_prior_paint_state_is_enforced(self):
        case_id = self.plan["cases"][0]["id"]
        plan = self.make_plan(
            references={case_id: {"final_state_sha256": "a" * 64, "master": False}}
        )
        self.assertEqual([case["id"] for case in plan["cases"]], [case_id])
        write(self.output / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "Backfill final paint differs"):
            batch.verify_case(self.folder)

    def test_master_quality_changes_only_static_photo_quality_and_preserves_default_identity(self):
        for settings in (
            {"options": ["control", "light-flat"]},
            {"study_family": "pattern-studies-v1", "options": ["lacuna-banks", "folded-sash"]},
            {"study_family": "pattern-studies-v2", "options": ["lacuna-banks", "folded-sash"]},
        ):
            with self.subTest(family=settings.get("study_family")):
                original = self.make_plan(**settings)
                self.assertEqual(original, self.make_plan(**settings, master_quality=False))
                self.assertNotIn("master_quality", original)
                expected = copy.deepcopy(original)
                for case in expected["cases"]:
                    case["master"] = True
                    case["photo_recipe"]["render"].update(resolution=[3840, 2880], samples=256)
                expected["identity_sha256"] = batch._sha(
                    {key: value for key, value in expected.items() if key != "identity_sha256"}
                )
                self.assertEqual(self.make_plan(**settings, master_quality=True), expected)

    def test_master_quality_rejects_conflicting_references_and_non_boolean_values(self):
        case_id = self.plan["cases"][0]["id"]
        reference = {case_id: {"final_state_sha256": "a" * 64, "master": False}}
        with self.assertRaisesRegex(ValueError, "conflicts with the prior reference"):
            self.make_plan(references=reference, master_quality=True)
        reference[case_id]["master"] = True
        for quality in (False, True):
            result = self.make_plan(references=reference, master_quality=quality)
            self.assertIs(result["cases"][0]["master"], True)
            self.assertEqual(result["cases"][0]["reference"], reference[case_id])
        for quality in (None, 0, 1, "true"):
            with (
                self.subTest(master_quality=quality),
                self.assertRaisesRegex(ValueError, "boolean"),
            ):
                self.make_plan(options=["control"], master_quality=quality)

    def test_plan_tampering_and_unused_or_duplicate_selections_fail(self):
        plan = copy.deepcopy(self.plan)
        plan["cases"][0]["motion_recipe"]["render"]["samples"] = 1
        with self.assertRaisesRegex(ValueError, "plan identity"):
            batch.validate_plan(plan)
        plan["identity_sha256"] = batch._sha(
            {k: v for k, v in plan.items() if k != "identity_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "canonical controls"):
            batch.validate_plan(plan)
        for options in (["control", "control"], ["unknown"], []):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_plan(options=options)

    def test_pattern_family_dispatch_is_opt_in_and_binds_its_initializer(self):
        self.assertNotIn("study_family", self.plan)
        self.assertNotIn("paint_runtime_extensions", self.plan)
        self.assertIs(batch.study_catalog().formation_recipe, batch.make_formation_recipe)
        for family in ("pattern-studies-v1", "pattern-studies-v2"):
            with self.subTest(family=family):
                plan = self.make_plan(study_family=family, options=["lacuna-banks", "folded-sash"])
                self.assertEqual(plan["study_family"], family)
                self.assertEqual(plan["paint_runtime_extensions"], ["initial_patterns.py"])
                self.assertEqual(len(plan["materials"]), 2)
                catalog = batch.resolve_catalog(plan)
                self.assertEqual(catalog.reference_option, "lacuna-banks")
                self.assertEqual(catalog.default_option, "folded-sash")
                for material in plan["materials"].values():
                    recipe = material["recipe"]
                    self.assertEqual(recipe["simulation"]["initial_pattern"], "composition")
                    self.assertEqual(code_identity(recipe), batch._runtime_contract(plan)["paint"])
        for family, extensions in (
            (None, []),
            ("unknown-v1", ["initial_patterns.py"]),
            ("pattern-studies-v1", []),
            ("pattern-studies-v1", ["initial_patterns.py"]),
        ):
            changed = copy.deepcopy(plan)
            changed.update(study_family=family, paint_runtime_extensions=extensions)
            changed["identity_sha256"] = batch._sha(
                {k: v for k, v in changed.items() if k != "identity_sha256"}
            )
            with self.subTest(family=family, extensions=extensions), self.assertRaises(ValueError):
                batch.validate_plan(changed)

    def test_a_still_without_a_complete_edit_cannot_be_published(self):
        (self.folder / "film/film.mp4").unlink()
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            batch.verify_case(self.folder)

    def test_certified_motion_camera_cannot_end_at_a_different_pose(self):
        target = self.folder / "motion/00-painting"
        camera = read(target / "camera.json")
        camera[-1]["tilt_degrees"] += 1
        write(target / "camera.json", camera)
        receipt = read(target / "receipt.json")
        receipt["artifacts"]["camera.json"] = artifact(target / "camera.json")
        write(target / "receipt.json", receipt)
        result = read(target / "experiment-result.json")
        result["render_receipt_sha256"] = artifact(target / "receipt.json")["sha256"]
        write(target / "experiment-result.json", result)
        self.assertTrue(
            experiment.finished(target, batch._sha(read(target.parent / "experiment-request.json")))
        )
        with self.assertRaisesRegex(ValueError, "paired photograph"):
            batch.verify_case(self.folder)

    def test_all_motion_poses_are_checked_after_receipt_hashes_are_rebound(self):
        target = self.folder / "motion/00-painting"
        original = read(target / "camera.json")
        bundle = read(self.material / "bundle/manifest.json")
        for field in ("angle", "frame", "matrix", "target"):
            camera = copy.deepcopy(original)
            pose = camera[len(camera) // 2]
            if field == "angle":
                pose["tilt_degrees"] += 1
            elif field == "frame":
                pose["frame"] += 1
            elif field == "matrix":
                pose["matrix_world"][0][0] += 0.01
            else:
                pose["matrix_world"][2][3] += 0.001
            write(target / "camera.json", camera)
            receipt = read(target / "receipt.json")
            receipt["artifacts"]["camera.json"] = artifact(target / "camera.json")
            write(target / "receipt.json", receipt)
            result = read(target / "experiment-result.json")
            result["render_receipt_sha256"] = artifact(target / "receipt.json")["sha256"]
            write(target / "experiment-result.json", result)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Camera"):
                batch._verify_photograph(
                    self.plan, self.plan["cases"][0], target, bundle, motion=True
                )

    def test_fixed_motion_target_must_match_the_paired_still(self):
        target = self.folder / "motion/00-painting"
        camera = read(target / "camera.json")
        for pose in camera:
            pose["matrix_world"][2][3] += 0.01
        # This is a valid fixed-height camera path by itself. The independently
        # certified still is what exposes its incorrect target height.
        batch.validate_camera_ledger(
            camera, self.plan["cases"][0]["motion_recipe"]["camera"], MOTION_FRAMES
        )
        write(target / "camera.json", camera)
        receipt = read(target / "receipt.json")
        receipt["artifacts"]["camera.json"] = artifact(target / "camera.json")
        write(target / "receipt.json", receipt)
        result = read(target / "experiment-result.json")
        result["render_receipt_sha256"] = artifact(target / "receipt.json")["sha256"]
        write(target / "experiment-result.json", result)
        with self.assertRaisesRegex(ValueError, "paired photograph matrix"):
            batch.verify_case(self.folder)

    def test_pure_camera_verifier_rejects_nonfinite_and_allows_only_roundoff(self):
        recipe = self.plan["cases"][0]["photo_recipe"]["camera"]
        original = read(self.folder / "photographs/00-painting/camera.json")
        rounded = copy.deepcopy(original)
        rounded[0]["tilt_degrees"] += 1e-12
        rounded[0]["matrix_world"][0][0] += 1e-12
        batch.validate_camera_ledger(rounded, recipe, 1)
        for invalid in (float("nan"), float("inf"), True, "0"):
            altered = copy.deepcopy(original)
            altered[0]["matrix_world"][0][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "matrix"):
                batch.validate_camera_ledger(altered, recipe, 1)

    def test_changed_tool_fails_before_creating_a_batch(self):
        self.tools["blender"].write_bytes(b"Changed executable")
        target = self.root / "unstarted"
        with self.assertRaisesRegex(ValueError, "executable changed"):
            batch.execute_plan(target, self.plan)
        self.assertFalse(target.exists())

    def test_completed_queue_resumes_and_callbacks_are_serial_and_require_verified_pairs(self):
        commands, callbacks = [], []
        owner = threading.get_ident()

        def completed_process(_self, command, _log):
            commands.append(command)
            self.assertIn("--resume", command)
            self.assertEqual(command[command.index("--checkpoint-retention") + 1], "2")

        def published(folder):
            self.assertEqual(threading.get_ident(), owner)
            batch.verify_case(folder)
            callbacks.append(folder.name)

        before = [artifact(folder / "study.json") for folder in self.folders]
        with patch.object(batch._Processes, "run", completed_process):
            first = batch.execute_plan(self.output, self.plan, on_case_complete=published)
            second = batch.execute_plan(self.output, self.plan, on_case_complete=published)
        self.assertTrue(first["complete"] and second["complete"])
        self.assertEqual(len(commands), 2)  # One shared material per execution.
        self.assertEqual(len(callbacks), 4)
        self.assertEqual(before, [artifact(folder / "study.json") for folder in self.folders])

    def test_failed_case_does_not_hide_an_independent_completed_option(self):
        (self.folder / "study.json").unlink()

        def fail_photo(_self, command, _log):
            if "tools.estuary_depth.experiment" in command:
                raise RuntimeError("Interrupted photographic stage")

        callbacks = []
        with (
            patch.object(batch._Processes, "run", fail_photo),
            self.assertRaisesRegex(ValueError, "Some film stages failed"),
        ):
            batch.execute_plan(self.output, self.plan, on_case_complete=callbacks.append)
        status = read(self.output / "results.json")
        self.assertEqual(status["cases"][self.folder.name]["stage"], "failed")
        self.assertEqual(status["cases"][self.folders[1].name]["stage"], "verified")
        self.assertEqual(callbacks, [self.folders[1]])
        self.assertTrue((self.folder / "photographs/00-painting/render.png").exists())

    def test_primary_films_precede_lighting_in_plan_order_with_two_bounded_workers(self):
        plan = copy.deepcopy(self.plan)
        # Deliberately oppose object-key order and explicit case priority.
        plan["materials"] = {key: {} for key in ("material-c", "material-b", "material-a")}
        plan["cases"] = [
            {"id": f"{key}-control", "material_id": key}
            for key in ("material-a", "material-b", "material-c")
        ] + [
            {"id": f"{key}-{option}", "material_id": key}
            for key, option in (
                ("material-a", "flat"),
                ("material-a", "deep"),
                ("material-b", "flat"),
            )
        ]
        state_lock, simultaneous = threading.Lock(), threading.Barrier(2)
        material_starts, completed_primary = [], []
        active, maximum = 0, 0

        def enter():
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)

        def leave():
            nonlocal active
            with state_lock:
                active -= 1

        def material(_root, _plan, key, _processes, _progress):
            enter()
            with state_lock:
                material_starts.append(key)
            if key in ("material-a", "material-b"):
                simultaneous.wait(timeout=3)
            leave()

        def render(_root, _plan, case, _processes, _progress):
            enter()
            with state_lock:
                if case["id"].endswith("-control"):
                    completed_primary.append(case["id"])
                else:
                    self.assertEqual(len(completed_primary), 3)
            leave()

        target = self.root / "scheduler"
        target.mkdir()
        with (
            patch.object(batch, "_ensure_material", material),
            patch.object(batch, "_ensure_case", render),
        ):
            result = batch._execute(target, plan, 2, None)
        self.assertTrue(result["complete"])
        self.assertEqual(set(material_starts[:2]), {"material-a", "material-b"})
        self.assertEqual(material_starts[2:], ["material-c"])
        self.assertEqual(maximum, 2)
        self.assertEqual(active, 0)

    def test_incomplete_paint_without_checkpoint_is_preserved_before_restart(self):
        paint = self.material / "paint"
        receipt = read(paint / "receipt.json")
        write(paint / "receipt.json", {**receipt, "complete": False})
        commands = []

        def stop_before_work(command, _log):
            commands.append(command)
            raise RuntimeError("Stop before GPU work")

        processes = batch._Processes()
        with (
            patch.object(processes, "run", stop_before_work),
            self.assertRaisesRegex(RuntimeError, "Stop before GPU"),
        ):
            batch._ensure_material(
                self.output, self.plan, self.material_id, processes, lambda *_: None
            )
        self.assertNotIn("--resume", commands[0])
        old = list(self.material.glob("paint.incomplete-*"))
        self.assertEqual(len(old), 1)
        self.assertTrue((old[0] / "final-state.npy").exists())


class OwnedProcessTests(unittest.TestCase):
    def test_cancellation_allows_nested_detached_child_cleanup_to_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / "nested.pid"
            nested = (
                "import os,signal,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
            )
            wrapper = (
                "import signal,sys; from tools.estuary_depth.experiment import capture; "
                "signal.signal(signal.SIGTERM,lambda *_: "
                "(_ for _ in ()).throw(KeyboardInterrupt())); "
                f"capture([sys.executable,'-c',{nested!r}])"
            )
            processes, errors = batch._Processes(), []

            def run():
                try:
                    processes.run([sys.executable, "-c", wrapper], root / "stage.log")
                except Exception as error:
                    errors.append(error)

            worker = threading.Thread(target=run)
            worker.start()
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pidfile.exists(), "Nested process did not start")
                nested_pid = int(pidfile.read_text())
                # Uses the real 8-second encoder shutdown path. Equal wrapper
                # and child deadlines orphan this detached process.
                processes.cancel()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ValueError)
                with self.assertRaises(ProcessLookupError):
                    os.kill(nested_pid, 0)
                self.assertEqual(processes.children, set())
            finally:
                processes.cancel()
                worker.join(timeout=2)
                if pidfile.exists():
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(int(pidfile.read_text()), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
