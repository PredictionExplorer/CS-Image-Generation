"""Cross-archive provenance checks using real small paint and material archives.

Only the study's canvas dimensions are reduced. Source parsing, artifact hashes,
bundle preparation, plan validation and photograph archive verification are real;
the photograph bytes are fixtures, so no GPU or Blender process is required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.recipe import validate_recipe
from tools.estuary.run import HERE as PAINT_ROOT
from tools.estuary.run import artifact as paint_artifact
from tools.estuary.run import code_identity, write_png
from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_depth import experiment
from tools.estuary_depth import filament_batch as batch
from tools.estuary_depth.filament_studies import REFERENCE_SEED, make_paint_recipe
from tools.estuary_studio.common import artifact, encoded, read, write


def identify(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def small_recipe(seed, variant):
    value = make_paint_recipe(seed, variant)
    value["simulation"]["resolution"] = [128, 96]
    value["render"]["resolution"] = [128, 96]
    return validate_recipe(value)


class FilamentBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(batch, "ROOT", self.root).start()
        patch.object(batch, "make_paint_recipe", small_recipe).start()
        self.source = self.root / f"{REFERENCE_SEED}.orbit"
        write_orbit(self.source, orbit_points(), seed=REFERENCE_SEED)
        cohort = self.root / "tools/estuary_confluence/recipes/ten-seeds.json"
        cohort.parent.mkdir(parents=True)
        write(cohort, {"sources": [{"seed": REFERENCE_SEED, **artifact(self.source)}]})
        self.blender = self.root / "blender"
        self.blender.write_bytes(b"pinned Blender fixture")
        self.plan = batch.make_plan(
            [REFERENCE_SEED], ["control"], source_root=self.root, blender=self.blender
        )
        self.plan.update(bundle_resolution=[64, 48], mesh_resolution=[32, 24])
        self.plan["identity_sha256"] = identify(
            {key: value for key, value in self.plan.items() if key != "identity_sha256"}
        )
        self.folder = self.root / "batch/cases" / self.plan["cases"][0]["id"]
        self.folder.mkdir(parents=True)
        write(self.folder.parent.parent / "plan.json", self.plan)
        self.build_paint()
        self.build_photograph()
        self.record = {
            "version": batch.VERSION,
            "complete": True,
            "seed": REFERENCE_SEED,
            "variant": "control",
            "proof": True,
            "source_sha256": artifact(self.source)["sha256"],
            "paint_identity_sha256": identify(read(self.folder / "paint/request.json")),
            "photo_identity_sha256": read(self.photo / "receipt.json")["identity_sha256"],
            "plan_identity_sha256": self.plan["identity_sha256"],
        }
        write(self.folder / "study.json", self.record)

    def build_paint(self):
        paint = self.folder / "paint"
        (paint / "inputs").mkdir(parents=True)
        shutil.copyfile(self.source, paint / "inputs/source.orbit")
        recipe = self.plan["cases"][0]["paint_recipe"]
        source = Source.read(self.source, aspect=4 / 3, **recipe["projection"])
        code = code_identity()
        request = {"recipe": recipe, "source": source.metadata, "code": code}
        for name in code:
            target = paint / "inputs/code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PAINT_ROOT / name, target)
        write(paint / "request.json", request)
        write(paint / "recipe.json", recipe)
        np.save(paint / "final-state.npy", np.full((96, 128, 4), 0.1, np.float32))
        np.save(paint / "linear.npy", np.full((96, 128, 3), 0.5, np.float32))
        for name in ("initial.png", "poster.png"):
            write_png(paint / name, np.full((96, 128, 3), 0.5, np.float32))
        write(
            paint / "receipt.json",
            {
                "complete": True,
                "identity_sha256": identify(request),
                "source": source.metadata,
                "final_step": recipe["simulation"]["steps"],
                "source_fraction": 1.0,
                "artifacts": [
                    paint_artifact(paint / name, paint)
                    for name in (
                        "initial.png",
                        "poster.png",
                        "final-state.npy",
                        "linear.npy",
                        "recipe.json",
                        "inputs/source.orbit",
                        *(f"inputs/code/{name}" for name in code),
                    )
                ],
            },
        )
        batch.build_bundle(
            paint, self.folder / "bundle", resolution=(64, 48), mesh_resolution=(32, 24)
        )

    def build_photograph(self):
        recipes = self.folder / "depth-recipes"
        recipes.mkdir()
        depth_recipe = self.plan["cases"][0]["depth_recipe"]
        write(recipes / "00-painting.json", depth_recipe)
        args = argparse.Namespace(
            recipes=recipes,
            bundle=self.folder / "bundle",
            blender=self.blender.resolve(),
            render_script=Path(batch.__file__).with_name("render.py"),
            baseline=None,
            motion_frames=1,
            fps=24,
            ffmpeg=None,
            ffprobe=None,
        )
        experiment_request, files, _ = experiment.make_request(args)
        self.photo = self.folder / "photographs/00-painting"
        self.photo.mkdir(parents=True)
        write(self.photo.parent / "experiment-request.json", experiment_request)
        for name, path in files.items():
            target = self.photo.parent / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        request = {
            "recipe": depth_recipe,
            "bundle_manifest_sha256": artifact(self.folder / "bundle/manifest.json")["sha256"],
            "bundle_sha256": artifact(self.folder / "bundle/bundle.npz")["sha256"],
            "renderer": {
                name: artifact(files[name])["sha256"] for name in ("render.py", "materials.py")
            },
            "motion": {"frames": 1, "fps": 24, "source_fraction": 1.0},
        }
        write(self.photo / "request.json", request)
        for name in ("render.png", "render.exr", "scene.blend"):
            (self.photo / name).write_bytes(name.encode())
        write(
            self.photo / "receipt.json",
            {
                "complete": True,
                "identity_sha256": identify(request),
                "artifacts": {
                    name: artifact(self.photo / name)
                    for name in ("render.png", "render.exr", "scene.blend")
                },
            },
        )
        self.bind_experiment_result()

    def bind_experiment_result(self):
        write(
            self.photo / "experiment-result.json",
            {
                "complete": True,
                "case": "00-painting",
                "identity_sha256": identify(read(self.photo.parent / "experiment-request.json")),
                "render_receipt_sha256": artifact(self.photo / "receipt.json")["sha256"],
                "movie": None,
            },
        )

    def test_legacy_record_shape_verifies_without_live_source_or_blender(self):
        self.source.unlink()
        self.blender.unlink()
        record, request, _ = batch.verify_case(self.folder)
        self.assertEqual(record, self.record)
        self.assertEqual(request["source"]["sha256"], record["source_sha256"])

    def test_record_source_plan_identity_and_controls_are_bound(self):
        for key, value in (
            ("source_sha256", "f" * 64),
            ("plan_identity_sha256", "f" * 64),
            ("proof", False),
            ("variant", "fine-bands"),
        ):
            write(self.folder / "study.json", {**self.record, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "archived plan"):
                batch.verify_case(self.folder)

    def test_recertified_different_orbit_with_same_seed_is_rejected(self):
        paint = self.folder / "paint"
        orbit = paint / "inputs/source.orbit"
        write_orbit(orbit, orbit_points() * 1.01, seed=REFERENCE_SEED)
        request = read(paint / "request.json")
        request["source"] = Source.read(
            orbit, aspect=4 / 3, **request["recipe"]["projection"]
        ).metadata
        write(paint / "request.json", request)
        receipt = read(paint / "receipt.json")
        receipt.update(source=request["source"], identity_sha256=identify(request))
        receipt["artifacts"] = [
            paint_artifact(paint / row["path"], paint) for row in receipt["artifacts"]
        ]
        write(paint / "receipt.json", receipt)
        write(
            self.folder / "study.json", {**self.record, "paint_identity_sha256": identify(request)}
        )
        batch.verified_run(paint)
        with self.assertRaisesRegex(ValueError, "Study source recording differs"):
            batch.verify_case(self.folder)

    def test_rehashed_plan_cannot_relabel_the_cohort_or_use_an_escaping_case_id(self):
        for key, value in (("source_sha256", "f" * 64), ("id", "../../escape")):
            plan = copy.deepcopy(self.plan)
            plan["cases"][0][key] = value
            plan["identity_sha256"] = identify(
                {k: v for k, v in plan.items() if k != "identity_sha256"}
            )
            write(self.folder.parent.parent / "plan.json", plan)
            with self.subTest(key=key), self.assertRaises(ValueError):
                batch.verify_case(self.folder)

    def test_changed_blender_or_source_fails_before_creating_batch_output(self):
        for path, expected in ((self.blender, "Blender binary"), (self.source, "Source changed")):
            before = path.read_bytes()
            path.write_bytes(before + b"changed")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, expected):
                batch.execute_plan(self.root / "new-batch", self.plan)
            self.assertFalse((self.root / "new-batch").exists())
            path.write_bytes(before)

    def test_self_consistent_photograph_from_a_different_blender_is_rejected(self):
        path = self.photo.parent / "experiment-request.json"
        experiment_request = read(path)
        experiment_request["blender"]["sha256"] = "a" * 64
        write(path, experiment_request)
        self.bind_experiment_result()
        self.assertTrue(experiment.finished(self.photo, identify(experiment_request)))
        with self.assertRaisesRegex(ValueError, "different Blender"):
            batch.verify_case(self.folder)

    def test_incomplete_experiment_or_changed_archived_input_is_rejected(self):
        result = read(self.photo / "experiment-result.json")
        write(self.photo / "experiment-result.json", {**result, "complete": False})
        with self.assertRaisesRegex(ValueError, "experiment is incomplete"):
            batch.verify_case(self.folder)
        write(self.photo / "experiment-result.json", result)
        (self.photo.parent / "inputs/materials.py").write_text("changed material")
        with self.assertRaisesRegex(ValueError, "experiment input changed"):
            batch.verify_case(self.folder)

    def test_changed_plan_or_mislabeled_material_manifest_fails(self):
        plan_path = self.folder.parent.parent / "plan.json"
        write(plan_path, {**self.plan, "proof": False})
        with self.assertRaisesRegex(ValueError, "Plan identity"):
            batch.verify_case(self.folder)
        write(plan_path, self.plan)
        manifest_path = self.folder / "bundle/manifest.json"
        manifest = read(manifest_path)
        manifest["request"]["parameters"]["specific_volumes"] = [1, 1, 1]
        write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "incorrectly identified material bundle"):
            batch.verify_case(self.folder)

    def rebuild_photograph(self):
        shutil.rmtree(self.folder / "photographs")
        shutil.rmtree(self.folder / "depth-recipes")
        self.build_photograph()
        self.record["photo_identity_sha256"] = read(self.photo / "receipt.json")["identity_sha256"]
        write(self.folder / "study.json", self.record)

    def test_complete_recaptured_material_cannot_change_undeclared_specific_volumes(self):
        shutil.rmtree(self.folder / "bundle")
        batch.build_bundle(
            self.folder / "paint",
            self.folder / "bundle",
            resolution=(64, 48),
            mesh_resolution=(32, 24),
            specific_volumes=(1, 1, 1),
        )
        self.rebuild_photograph()
        self.assertTrue(
            experiment.finished(
                self.photo, identify(read(self.photo.parent / "experiment-request.json"))
            )
        )
        with self.assertRaisesRegex(ValueError, "preparation parameters"):
            batch.verify_case(self.folder)

    def test_self_consistent_alternate_paint_runtime_is_rejected(self):
        paint = self.folder / "paint"
        code_file = paint / "inputs/code/engine.py"
        code_file.write_text("different numerical engine")
        request = read(paint / "request.json")
        request["code"]["engine.py"] = artifact(code_file)["sha256"]
        write(paint / "request.json", request)
        receipt = read(paint / "receipt.json")
        receipt["identity_sha256"] = identify(request)
        receipt["artifacts"] = [
            paint_artifact(paint / row["path"], paint) for row in receipt["artifacts"]
        ]
        write(paint / "receipt.json", receipt)
        self.record["paint_identity_sha256"] = identify(request)
        shutil.rmtree(self.folder / "bundle")
        batch.build_bundle(
            paint, self.folder / "bundle", resolution=(64, 48), mesh_resolution=(32, 24)
        )
        self.rebuild_photograph()
        with self.assertRaisesRegex(ValueError, "Paint runtime"):
            batch.verify_case(self.folder)

    def test_preparation_and_photo_hashes_bind_to_archived_plan_not_live_runtime(self):
        with patch.object(batch, "runtime_identity", return_value={"different": "today"}):
            batch.verify_case(self.folder)
        request = read(self.folder / "paint/request.json")
        artifacts = {
            row["path"]: row for row in read(self.folder / "paint/receipt.json")["artifacts"]
        }
        bundle = read(self.folder / "bundle/manifest.json")
        for key in ("prepare_sha256", "optics_sha256"):
            changed = copy.deepcopy(bundle)
            changed["request"][key] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preparation runtime"):
                batch._verify_paint_preparation(self.plan, request, artifacts, changed)
        shot = read(self.photo / "request.json")
        shot["renderer"]["materials.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Photograph runtime"):
            batch._verify_photo_runtime(self.plan, shot)


if __name__ == "__main__":
    unittest.main()
