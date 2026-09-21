"""Portable fine-fold media, canonical controls and cross-stage provenance."""

from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.estuary.run import HERE as PAINT_ROOT
from tools.estuary.run import code_identity
from tools.estuary_studio.common import artifact, encoded, read, write

from . import filament_gallery as gallery
from .filament_batch import SPECIFIC_VOLUMES, runtime_identity
from .filament_studies import REFERENCE_SEED, VARIANTS, make_depth_recipe, make_paint_recipe


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def fake_artifact(name, token):
    return {"path": name, "sha256": hashlib.sha256(token.encode()).hexdigest(), "bytes": 128}


class FilamentGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.batch = self.root / "batch"
        self.output = self.root / "publication"
        cohort = read(
            Path(gallery.__file__).parents[1] / "estuary_confluence/recipes/ten-seeds.json"
        )
        self.source_hash = next(
            row["sha256"] for row in cohort["sources"] if row["seed"] == REFERENCE_SEED
        )
        self.source = {
            "seed": REFERENCE_SEED,
            "sha256": self.source_hash,
            "samples": 1000000,
            "source_first_step": 0,
            "source_last_step": 999999,
            "projection": {"aspect": 4 / 3, "fill": 0.78},
            "dt": 0.001,
        }
        self.blender = {"path": "/original/bin/blender", "sha256": "b" * 64, "bytes": 100}
        self.runtime = runtime_identity()
        self.cases = [
            self.batch / "cases" / (REFERENCE_SEED + "-" + variant)
            for variant in ("control", "fine-bands")
        ]
        plan = {
            "version": gallery.STUDY_VERSION,
            "proof": True,
            "blender": self.blender,
            "runtime": self.runtime,
            "bundle_resolution": [4096, 3072],
            "mesh_resolution": [1536, 1152],
            "cases": [
                {
                    "id": folder.name,
                    "seed": REFERENCE_SEED,
                    "variant": variant,
                    "source": "/original/source.orbit",
                    "source_sha256": self.source_hash,
                    "paint_recipe": make_paint_recipe(REFERENCE_SEED, variant),
                    "depth_recipe": make_depth_recipe(REFERENCE_SEED, VARIANTS[variant].label),
                }
                for folder, variant in zip(self.cases, ("control", "fine-bands"), strict=True)
            ],
        }
        plan["identity_sha256"] = sha(plan)
        self.batch.mkdir(parents=True)
        write(self.batch / "plan.json", plan)
        for folder, variant in zip(self.cases, ("control", "fine-bands"), strict=True):
            self.make_case(folder, variant, plan)
        # Renderer/source execution is qualified elsewhere. These fixtures use
        # real canonical recipes and valid small PNG files; portable checks run
        # unmocked, including the plan and pinned cohort source identity.
        stub = patch.object(
            gallery,
            "verify_case",
            side_effect=lambda p: (
                read(p / "study.json"),
                read(p / "paint/request.json"),
                read(p / "photographs/00-painting/receipt.json"),
            ),
        )
        stub.start()
        self.addCleanup(stub.stop)

    def photo(self, folder, bundle_path, expected, source):
        folder.mkdir(parents=True)
        for name, value in (("recipe.json", expected), ("camera.json", [{"frame": 0}])):
            write(folder / name, value)
        Image.new("RGB", tuple(expected["render"]["resolution"]), (12, 26, 70)).save(
            folder / "render.png"
        )
        manifest = read(bundle_path)
        request = {
            "schema_version": 1,
            "recipe": expected,
            "bundle_sha256": manifest["bundle"]["sha256"],
            "bundle_manifest_sha256": artifact(bundle_path)["sha256"],
            "renderer": {
                name: self.runtime["estuary_depth"][name] for name in ("render.py", "materials.py")
            },
            "motion": {"frames": 1, "fps": 24, "source_fraction": 1.0},
        }
        write(folder / "request.json", request)
        receipt = {
            "schema_version": 1,
            "complete": True,
            "identity_sha256": sha(request),
            "source": source,
            "history_fractions": [1.0],
            "motion_frames": 1,
            "artifacts": {
                name: artifact(folder / name)
                for name in ("render.png", "recipe.json", "camera.json")
            },
        }
        receipt["artifacts"].update(
            {
                name: {k: v for k, v in fake_artifact(name, name).items() if k != "path"}
                for name in ("scene.blend", "render.exr")
            }
        )
        write(folder / "receipt.json", receipt)
        return request, receipt

    def experiment(self, root, folder, request):
        recipes = root / "inputs/recipes"
        recipes.mkdir(parents=True, exist_ok=True)
        write(recipes / (folder.name + ".json"), request["recipe"])
        experiment = {
            "schema_version": 1,
            "blender": {k: self.blender[k] for k in ("path", "sha256")},
            "motion_frames": 1,
            "fps": 24,
            "files": {
                "bundle/bundle.npz": {"sha256": request["bundle_sha256"]},
                "bundle/manifest.json": {"sha256": request["bundle_manifest_sha256"]},
                **{name: {"sha256": value} for name, value in request["renderer"].items()},
                f"recipes/{folder.name}.json": artifact(recipes / (folder.name + ".json")),
            },
        }
        write(root / "experiment-request.json", experiment)
        write(
            folder / "experiment-result.json",
            {
                "case": folder.name,
                "complete": True,
                "identity_sha256": sha(experiment),
                "movie": None,
                "render_receipt_sha256": artifact(folder / "receipt.json")["sha256"],
            },
        )

    def make_case(self, folder, variant, plan):
        paint = folder / "paint"
        paint.mkdir(parents=True)
        recipe = make_paint_recipe(REFERENCE_SEED, variant)
        write(paint / "recipe.json", recipe)
        Image.new("RGB", tuple(recipe["render"]["resolution"]), (20, 30, 80)).save(
            paint / "initial.png"
        )
        request = {
            "schema_version": 1,
            "source": self.source,
            "recipe": recipe,
            "code": code_identity(),
        }
        write(paint / "request.json", request)
        artifacts = {
            name: {"path": name, **artifact(paint / name)}
            for name in ("recipe.json", "initial.png")
        }
        artifacts.update(
            {
                name: fake_artifact(name, variant + name)
                for name in ("final-state.npy", "linear.npy", "inputs/source.orbit")
            }
        )
        artifacts["inputs/source.orbit"]["sha256"] = self.source_hash
        artifacts.update(
            {
                f"inputs/code/{name}": {
                    "path": f"inputs/code/{name}",
                    **artifact(PAINT_ROOT / name),
                }
                for name in request["code"]
            }
        )
        receipt = {
            "complete": True,
            "identity_sha256": sha(request),
            "source": self.source,
            "final_step": recipe["simulation"]["steps"],
            "source_fraction": 1.0,
            "artifacts": list(artifacts.values()),
        }
        write(paint / "receipt.json", receipt)
        inputs = {
            "render_identity": sha(request),
            "request_sha256": artifact(paint / "request.json")["sha256"],
            "receipt_sha256": artifact(paint / "receipt.json")["sha256"],
            "history": [],
            "artifacts": {
                name: artifacts[name]
                for name in ("final-state.npy", "linear.npy", "recipe.json", "inputs/source.orbit")
            },
        }
        geometry = {
            "domain_scale": recipe["simulation"]["domain_scale"],
            "view_aspect": 4 / 3,
            "coordinates": {"row_order": "bottom-to-top"},
        }
        bundle_request = {
            "inputs": inputs,
            "parameters": {
                "history_fractions": [],
                "resolution": plan["bundle_resolution"],
                "mesh_resolution": plan["mesh_resolution"],
                "specific_volumes": list(SPECIFIC_VOLUMES),
            },
            "geometry": geometry,
            "source": self.source,
            "optics_sha256": self.runtime["estuary"]["optics.py"],
            "prepare_sha256": self.runtime["estuary_depth"]["prepare.py"],
        }
        bundle = {
            "schema_version": 1,
            "complete": True,
            "request": bundle_request,
            "identity_sha256": sha(bundle_request),
            "source": self.source,
            "bundle": fake_artifact("bundle.npz", variant + "bundle"),
            **geometry,
            "history": {"source_fractions": [1.0]},
        }
        (folder / "bundle").mkdir()
        write(folder / "bundle/manifest.json", bundle)
        photo = folder / "photographs/00-painting"
        shot, photo_receipt = self.photo(
            photo,
            folder / "bundle/manifest.json",
            make_depth_recipe(REFERENCE_SEED, VARIANTS[variant].label),
            self.source,
        )
        self.experiment(photo.parent, photo, shot)
        write(
            folder / "study.json",
            {
                "version": gallery.STUDY_VERSION,
                "complete": True,
                "seed": REFERENCE_SEED,
                "variant": variant,
                "proof": True,
                "source_sha256": self.source_hash,
                "paint_identity_sha256": sha(request),
                "photo_identity_sha256": photo_receipt["identity_sha256"],
                "plan_identity_sha256": plan["identity_sha256"],
            },
        )

    def build(self, **kwargs):
        return gallery.build_review(self.output, self.cases, **kwargs)

    def rehash(self):
        publication = read(self.output / "publication.json")
        publication["artifacts"] = {
            name: artifact(self.output / name) for name in publication["artifacts"]
        }
        publication["comparison_sha256"] = sha(read(self.output / "comparison.json"))
        write(self.output / "publication.json", publication)

    def test_portable_publication_after_original_cases_are_deleted(self):
        picks = [{"seed": REFERENCE_SEED, "variant": "fine-bands", "note": "Clearer small folds."}]
        data = self.build(picks=picks)
        self.assertEqual(data["variants"], ["control", "fine-bands"])
        self.assertTrue(all(row["film"] is None and row["features"] == {} for row in data["rows"]))
        self.assertEqual(data["picks"], picks)
        self.assertIn('"reference_variant": "control"', (self.output / "index.html").read_text())
        self.assertIn('"film_only_selection": true', (self.output / "index.html").read_text())
        shutil.rmtree(self.batch)
        self.assertEqual(gallery.verify_review(self.output), data)

    def test_broad_pools_default_preserves_verification_of_original_published_page(self):
        data = self.build()
        self.assertIn(
            '"default_variant": "three-broad-pools"', (self.output / "index.html").read_text()
        )
        (self.output / "index.html").write_text(
            gallery.document(data["title"], presentation=gallery._LEGACY_PRESENTATION)
        )
        self.rehash()
        self.assertEqual(gallery.verify_review(self.output), data)

    def rebind_photo(self, entry):
        paths = {key: self.output / value for key, value in entry["records"].items()}
        shot = read(paths["photo_request"])
        receipt = read(paths["photo_receipt"])
        receipt["identity_sha256"] = sha(shot)
        write(paths["photo_receipt"], receipt)
        study = read(paths["study"])
        study["photo_identity_sha256"] = sha(shot)
        write(paths["study"], study)
        experiment = read(paths["photo_experiment"])
        experiment["files"]["bundle/manifest.json"]["sha256"] = shot["bundle_manifest_sha256"]
        for name, value in shot["renderer"].items():
            experiment["files"][name]["sha256"] = value
        write(paths["photo_experiment"], experiment)
        result = read(paths["photo_result"])
        result["identity_sha256"] = sha(experiment)
        result["render_receipt_sha256"] = artifact(paths["photo_receipt"])["sha256"]
        write(paths["photo_result"], result)
        self.rehash()

    def test_rebound_paint_runtime_plan_cannot_certify_a_different_engine(self):
        self.build()
        entry = read(self.output / "publication.json")["entries"][0]
        path = self.output / entry["records"]["batch_plan"]
        plan = read(path)
        plan["runtime"]["estuary"]["engine.py"] = "0" * 64
        plan["identity_sha256"] = sha({k: v for k, v in plan.items() if k != "identity_sha256"})
        write(path, plan)
        study_path = self.output / entry["records"]["study"]
        study = read(study_path)
        study["plan_identity_sha256"] = plan["identity_sha256"]
        write(study_path, study)
        self.rehash()
        with self.assertRaisesRegex(ValueError, "Paint runtime"):
            gallery.verify_review(self.output)

    def test_rebound_photograph_renderer_cannot_escape_the_frozen_plan(self):
        self.build()
        entry = read(self.output / "publication.json")["entries"][0]
        path = self.output / entry["records"]["photo_request"]
        request = read(path)
        request["renderer"]["render.py"] = "0" * 64
        write(path, request)
        self.rebind_photo(entry)
        with self.assertRaisesRegex(ValueError, "Photograph runtime"):
            gallery.verify_review(self.output)

    def test_rebound_bundle_cannot_change_specific_volumes_or_preparation_runtime(self):
        self.build()
        entry = read(self.output / "publication.json")["entries"][0]
        path = self.output / entry["records"]["bundle"]
        baseline = read(path)
        for key in ("specific_volumes", "prepare_sha256", "optics_sha256"):
            bundle = copy.deepcopy(baseline)
            if key == "specific_volumes":
                bundle["request"]["parameters"][key] = [1.0, 1.0, 1.0]
            else:
                bundle["request"][key] = "0" * 64
            bundle["identity_sha256"] = sha(bundle["request"])
            write(path, bundle)
            shot_path = self.output / entry["records"]["photo_request"]
            shot = read(shot_path)
            shot["bundle_manifest_sha256"] = artifact(path)["sha256"]
            write(shot_path, shot)
            self.rebind_photo(entry)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preparation"):
                gallery.verify_review(self.output)

    def test_requires_control_and_refuses_overwrite_or_archive_destination(self):
        with self.assertRaisesRegex(ValueError, "Control"):
            gallery.build_review(self.root / "missing-control", self.cases[1:])
        with self.assertRaisesRegex(ValueError, "outside"):
            gallery.build_review(self.batch / "gallery", self.cases)
        self.build()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.build()

    def test_rehashed_label_or_settings_tamper_fails_regeneration(self):
        self.build()
        original = read(self.output / "comparison.json")
        for key in ("label", "settings", "film"):
            changed = copy.deepcopy(original)
            changed["rows"][1][key] = {"forged": True} if key == "settings" else "forged"
            write(self.output / "comparison.json", changed)
            self.rehash()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Comparison"):
                gallery.verify_review(self.output)

    def test_rehashed_noncanonical_paint_and_photo_recipes_are_rejected(self):
        self.build()
        entry = read(self.output / "publication.json")["entries"][1]
        for key in ("paint_request", "photo_request"):
            path = self.output / entry["records"][key]
            original = read(path)
            changed = copy.deepcopy(original)
            if key == "paint_request":
                changed["recipe"]["simulation"]["flow_strength"] += 0.01
            else:
                changed["recipe"]["relief_mm"] += 1
            write(path, changed)
            self.rehash()
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_review(self.output)
            write(path, original)

    def test_rehashed_plan_blender_and_full_source_interval_tampering(self):
        self.build()
        entry = read(self.output / "publication.json")["entries"][0]
        for key, change in (
            ("batch_plan", lambda d: d.update(identity_sha256="0" * 64)),
            ("photo_experiment", lambda d: d["blender"].update(sha256="0" * 64)),
            ("paint_receipt", lambda d: d.update(source_fraction=0.9)),
            (
                "bundle",
                lambda d: d["request"]["inputs"]["artifacts"]["final-state.npy"].update(
                    sha256="0" * 64
                ),
            ),
        ):
            path = self.output / entry["records"][key]
            original = read(path)
            changed = copy.deepcopy(original)
            change(changed)
            write(path, changed)
            self.rehash()
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_review(self.output)
            write(path, original)

    def test_media_tamper_and_escaping_record_path_are_rejected(self):
        self.build()
        publication = read(self.output / "publication.json")
        image = self.output / publication["entries"][0]["image"]
        original = image.read_bytes()
        image.write_bytes(b"not png")
        with self.assertRaises(ValueError):
            gallery.verify_review(self.output)
        image.write_bytes(original)
        publication["entries"][0]["records"]["study"] = "../study.json"
        write(self.output / "publication.json", publication)
        with self.assertRaisesRegex(ValueError, "escapes"):
            gallery.verify_review(self.output)

    def test_unknown_or_duplicate_visual_picks_are_rejected(self):
        pick = {"seed": REFERENCE_SEED, "variant": "fine-bands", "note": "Useful folds."}
        for index, picks in enumerate(([{**pick, "variant": "unknown"}], [pick, pick])):
            with self.subTest(picks=picks), self.assertRaises(ValueError):
                gallery.build_review(self.root / f"bad-picks-{index}", self.cases, picks=picks)

    def lighting(self):
        root = self.root / "lighting"
        (root / "inputs/bundle").mkdir(parents=True)
        shutil.copyfile(
            self.cases[0] / "bundle/manifest.json", root / "inputs/bundle/manifest.json"
        )
        cases = []
        for case_id, label, parameter, value in gallery.LIGHTING.values():
            recipe = make_depth_recipe(REFERENCE_SEED, label)
            if parameter == "relief_mm":
                recipe[parameter] = value
            else:
                recipe["lighting"][parameter] = value
            folder = root / case_id
            request, _ = self.photo(
                folder, root / "inputs/bundle/manifest.json", recipe, self.source
            )
            self.experiment(root, folder, request)
            cases.append(folder)
        # All four photographs share one experiment request, as in the real runner.
        experiment = read(root / "experiment-request.json")
        for folder in cases:
            experiment["files"][f"recipes/{folder.name}.json"] = artifact(
                root / "inputs/recipes" / (folder.name + ".json")
            )
        write(root / "experiment-request.json", experiment)
        for folder in cases:
            result = read(folder / "experiment-result.json")
            result["identity_sha256"] = sha(experiment)
            write(folder / "experiment-result.json", result)
        return root

    def test_optional_lighting_requires_the_identical_control_material(self):
        lighting = self.lighting()
        with (
            patch.object(gallery, "finished", return_value=True),
            patch.object(gallery, "verify_render"),
        ):
            data = self.build(lighting_root=lighting)
        self.assertEqual(len(data["rows"]), 6)
        self.assertEqual(
            {r["variant"] for r in data["rows"]}, {"control", "fine-bands", *gallery.LIGHTING}
        )
        control = next(r for r in data["rows"] if r["variant"] == "control")
        self.assertTrue(
            all(
                r["initial"] == control["initial"]
                for r in data["rows"]
                if r["variant"].startswith("light-")
            )
        )
        shutil.rmtree(self.batch)
        shutil.rmtree(lighting)
        self.assertEqual(gallery.verify_review(self.output), data)
        entry = next(
            e for e in read(self.output / "publication.json")["entries"] if e["kind"] == "lighting"
        )
        path = self.output / entry["records"]["bundle"]
        bundle = read(path)
        bundle["bundle"]["sha256"] = "0" * 64
        write(path, bundle)
        self.rehash()
        with self.assertRaisesRegex(ValueError, "differs from Control"):
            gallery.verify_review(self.output)


if __name__ == "__main__":
    unittest.main()
