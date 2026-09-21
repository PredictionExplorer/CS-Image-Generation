"""Atomic progressive reviews, complete film provenance and long seed controls."""

from __future__ import annotations

import copy
import shutil
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.estuary.recipe import validate_recipe
from tools.estuary.run import artifact as paint_artifact
from tools.estuary_confluence import test_review_page as page_tests
from tools.estuary_studio.common import artifact, read, write

from . import filament_film_gallery as gallery
from . import film
from . import test_filament_film_batch as batch_tests
from .render import camera_pose


class FilmGalleryTests(unittest.TestCase):
    def setUp(self):
        if getattr(self, "study_family", None) is not None:
            if self.study_family == "pattern-studies-v1":
                from . import pattern_studies
            elif self.study_family == "pattern-studies-v2":
                from . import pattern_studies_v2 as pattern_studies
            else:
                raise ValueError("Unknown pattern fixture family")

            original = pattern_studies.make_formation_recipe

            def small_pattern(seed, option):
                result = original(seed, option)
                result["simulation"]["resolution"] = [128, 96]
                result["render"]["resolution"] = [128, 96]
                return validate_recipe(result)

            factory = patch.object(pattern_studies, "make_formation_recipe", small_pattern)
            factory.start()
            self.addCleanup(factory.stop)
        fixture_type = batch_tests.FilmBatchTests
        original_material = fixture_type.build_material
        original_photo = fixture_type.build_photo
        original_edit = fixture_type.build_edit

        def material(fixture):
            original_material(fixture)
            paint = fixture.material / "paint"
            movie = read(paint / "movie.json")
            movie["duration_seconds"] = movie["frames"] / movie["fps"]
            write(paint / "movie.json", movie)
            receipt = read(paint / "receipt.json")
            receipt["artifacts"] = [
                paint_artifact(paint / r["path"], paint) for r in receipt["artifacts"]
            ]
            write(paint / "receipt.json", receipt)
            bundle_path = fixture.material / "bundle/manifest.json"
            bundle = read(bundle_path)
            bundle["request"]["inputs"]["receipt_sha256"] = artifact(paint / "receipt.json")[
                "sha256"
            ]
            bundle["identity_sha256"] = gallery._sha(bundle["request"])
            write(bundle_path, bundle)

        def photo(fixture, case, folder, *, motion):
            original_photo(fixture, case, folder, motion=motion)
            target = folder / ("motion" if motion else "photographs") / "00-painting"
            request = read(target / "request.json")
            request["motion"]["semantics"] = "frozen completed painting; camera only"
            write(target / "request.json", request)
            recipe = request["recipe"]
            write(target / "recipe.json", recipe)
            Image.new("RGB", tuple(recipe["render"]["resolution"]), (20, 30, 70)).save(
                target / "render.png"
            )
            camera = read(target / "camera.json")
            with np.load(fixture.material / "bundle/bundle.npz") as maps:
                target_z = float(np.median(maps["history_height"][-1])) * recipe["relief_mm"] / 1000
            for pose in camera:
                pose["matrix_world"] = camera_pose(
                    pose["tilt_degrees"],
                    pose["azimuth_degrees"],
                    [*recipe["camera"]["target"], target_z],
                ).tolist()
            write(target / "camera.json", camera)
            receipt = read(target / "receipt.json")
            receipt["identity_sha256"] = gallery._sha(request)
            for name in ("render.png", "recipe.json", "camera.json"):
                receipt["artifacts"][name] = artifact(target / name)
            if motion:
                (target / "frames").mkdir()
                for i in range(gallery.MOTION_FRAMES):
                    path = target / f"frames/{i:06d}.png"
                    shutil.copyfile(target / "render.png", path)
                    receipt["artifacts"][str(path.relative_to(target))] = artifact(path)
            write(target / "receipt.json", receipt)
            result = read(target / "experiment-result.json")
            result["render_receipt_sha256"] = artifact(target / "receipt.json")["sha256"]
            if motion:
                result["movie"]["duration_seconds"] = gallery.MOTION_FRAMES / gallery.FPS
            write(target / "experiment-result.json", result)

        def edit(fixture, case, folder):
            original_edit(fixture, case, folder)
            target = folder / "film"
            request = read(target / "request.json")
            write(
                target / "command.json",
                film.command(fixture.plan["tools"]["ffmpeg"]["path"], target, request["timeline"]),
            )
            receipt = read(target / "receipt.json")
            receipt["artifacts"]["command.json"] = artifact(target / "command.json")
            receipt["movie"]["duration_seconds"] = float(Fraction(request["timeline"]["duration"]))
            write(target / "receipt.json", receipt)

        self.fixture = fixture_type("runTest")
        self.fixture.study_family = getattr(self, "study_family", None)
        with (
            patch.object(fixture_type, "build_material", material),
            patch.object(fixture_type, "build_photo", photo),
            patch.object(fixture_type, "build_edit", edit),
        ):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / "review"

    def publish(self, **kwargs):
        return gallery.publish_review(self.output, [self.fixture.output], **kwargs)

    def test_complete_portable_pairs_survive_removal_of_original_batch(self):
        data = self.publish()
        self.assertEqual(data["progress"]["ready_pairs"], 2)
        self.assertTrue(data["progress"]["complete"])
        self.assertEqual([r["variant"] for r in data["rows"]], ["control", "light-flat"])
        self.assertTrue(
            all(row["film_frames"] == 817 and row["film_fps"] == 24 for row in data["rows"])
        )
        shutil.rmtree(self.fixture.output)
        self.assertEqual(gallery.verify_review(self.output), data)

    def test_new_previews_are_deterministic_small_and_leave_full_images_unchanged(self):
        original = artifact(self.fixture.folders[0] / "photographs/00-painting/render.png")
        data = self.publish()
        entry = read(self.output / data["provenance"]["entries"][0]["path"])
        row = data["rows"][0]
        self.assertEqual(gallery._fingerprint(entry["media"]["image"]), original)
        self.assertEqual(row["image"], entry["media"]["image"]["path"])
        self.assertNotEqual(row["preview"], row["image"])
        self.assertEqual(row["preview_resolution"], [640, 480])
        self.assertEqual(row["resolution"], [2048, 1536])
        self.assertEqual(
            entry["preview"], gallery._make_preview(self.output, entry["media"]["image"])
        )
        self.assertLess(entry["preview"]["artifact"]["bytes"], original["bytes"])
        with Image.open(self.output / row["preview"]) as image:
            self.assertEqual((image.size, image.mode), ((640, 480), "RGB"))
        with patch.object(
            gallery, "_preview_pixels", side_effect=AssertionError("preview regenerated")
        ):
            self.assertEqual(self.publish(), data)

    def test_legacy_entries_and_rows_remain_byte_identical_on_refresh(self):
        data = self.publish()
        legacy = []
        for descriptor in data["provenance"]["entries"]:
            entry = read(self.output / descriptor["path"])
            entry.pop("preview")
            path = self.output / "entries" / (gallery._sha(entry) + ".json")
            write(path, entry)
            legacy.append({"path": str(path.relative_to(self.output)), **artifact(path)})
        data["provenance"]["entries"] = legacy
        for row in data["rows"]:
            row["preview"] = row["image"]
            row.pop("preview_resolution")
        data["identity_sha256"] = gallery._sha(
            {k: v for k, v in data.items() if k != "identity_sha256"}
        )
        write(self.output / "comparison.json", data)
        before = (self.output / "comparison.json").read_bytes()
        self.assertEqual(gallery.verify_review(self.output), data)
        with patch.object(
            gallery, "_make_preview", side_effect=AssertionError("old entry changed")
        ):
            self.assertEqual(self.publish(), data)
        self.assertEqual((self.output / "comparison.json").read_bytes(), before)

    def test_preview_byte_tampering_is_rejected(self):
        data = self.publish()
        path = self.output / data["rows"][0]["preview"]
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaises(ValueError):
            gallery.verify_review(self.output)

    def test_preview_source_version_and_dimensions_are_bound(self):
        original = self.publish()
        for field, value in (
            ("source_image_sha256", "0" * 64),
            ("version", "unknown"),
            ("resolution", [320, 240]),
        ):
            write(self.output / "comparison.json", original)
            self.replace_entry(
                lambda entry, field=field, value=value: entry["preview"].update({field: value})
            )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Preview version"):
                gallery.verify_review(self.output)

    def test_rehashed_foreign_preview_pixels_are_rejected(self):
        self.publish()
        foreign = self.root / "foreign.png"
        Image.new("RGB", (640, 480), (230, 40, 15)).save(foreign)
        descriptor = gallery._copy(self.output, foreign, folder="assets")
        self.replace_entry(lambda entry: entry["preview"].update(artifact=descriptor))
        with self.assertRaisesRegex(ValueError, "Preview pixels differ"):
            gallery.verify_review(self.output)

    def test_incremental_empty_partial_complete_reuses_certified_cases(self):
        saved = []
        for folder in self.fixture.folders:
            saved.append(read(folder / "study.json"))
            (folder / "study.json").unlink()
        empty = self.publish()
        self.assertEqual(empty["rows"], [])
        self.assertFalse(empty["progress"]["complete"])
        self.assertEqual(
            empty["progress"]["pending_case_ids"], [c["id"] for c in self.fixture.plan["cases"]]
        )
        # A completed treatment cannot pretend that its missing baseline exists.
        write(self.fixture.folders[1] / "study.json", saved[1])
        waiting = self.publish()
        self.assertEqual(waiting["progress"]["ready_pairs"], 1)
        self.assertEqual(waiting["progress"]["visible_pairs"], 0)
        write(self.fixture.folders[0] / "study.json", saved[0])
        original_verify = gallery.verify_case
        with patch.object(gallery, "verify_case", wraps=original_verify) as verify:
            complete = self.publish()
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(len(complete["rows"]), 2)
        with patch.object(gallery, "verify_case", side_effect=AssertionError("raw case replayed")):
            self.assertEqual(self.publish(), complete)

    def test_interrupted_first_copy_resumes_and_later_failure_keeps_manifest(self):
        original = gallery._copy
        count = 0

        def fail_once(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 4:
                raise OSError("interrupted copy")
            return original(*args, **kwargs)

        with patch.object(gallery, "_copy", side_effect=fail_once), self.assertRaises(OSError):
            self.publish()
        self.assertFalse((self.output / "comparison.json").exists())
        self.publish()
        before = (self.output / "comparison.json").read_bytes()
        with self.assertRaises(ValueError):
            self.publish(picks=[{"seed": "bad", "variant": "bad", "note": "bad"}])
        self.assertEqual((self.output / "comparison.json").read_bytes(), before)

    def test_unrelated_or_archive_outputs_are_refused(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("untouched")
        with self.assertRaisesRegex(ValueError, "unrelated"):
            self.publish()
        with self.assertRaisesRegex(ValueError, "outside"):
            gallery.publish_review(self.fixture.output / "review", [self.fixture.output])
        self.assertEqual((self.output / "keep.txt").read_text(), "untouched")

    def test_rehashed_progress_or_labels_cannot_invent_completion(self):
        original = self.publish()
        for change in (
            lambda d: d["progress"].update(ready_pairs=150),
            lambda d: d["rows"][0].update(label="invented"),
            lambda d: d["rows"][0].update(film_frames=1),
        ):
            data = copy.deepcopy(original)
            change(data)
            data["identity_sha256"] = gallery._sha(
                {k: v for k, v in data.items() if k != "identity_sha256"}
            )
            write(self.output / "comparison.json", data)
            with self.assertRaises(ValueError):
                gallery.verify_review(self.output)

    def replace_entry(self, callback):
        data = read(self.output / "comparison.json")
        old = data["provenance"]["entries"][0]
        entry = read(self.output / old["path"])
        callback(entry)
        target = self.output / "entries" / (gallery._sha(entry) + ".json")
        write(target, entry)
        data["provenance"]["entries"][0] = {
            "path": str(target.relative_to(self.output)),
            **artifact(target),
        }
        data["identity_sha256"] = gallery._sha(
            {k: v for k, v in data.items() if k != "identity_sha256"}
        )
        write(self.output / "comparison.json", data)

    def new_record(self, value, *, text=False):
        source = self.root / ("attack.txt" if text else "attack.json")
        if text:
            source.write_text(value)
        else:
            write(source, value)
        return gallery._copy(self.output, source)

    def test_rehashed_short_decode_proof_is_rejected(self):
        self.publish()

        def attack(entry):
            entry["records"]["film_decode"] = self.new_record(
                "frame=816\nprogress=end\n", text=True
            )
            receipt = read(self.output / entry["records"]["film_receipt"]["path"])
            receipt["artifacts"]["decode.txt"] = gallery._fingerprint(
                entry["records"]["film_decode"]
            )
            entry["records"]["film_receipt"] = self.new_record(receipt)

        self.replace_entry(attack)
        with self.assertRaisesRegex(ValueError, "decode proof"):
            gallery.verify_review(self.output)

    def test_rehashed_different_final_state_cannot_pair_still_and_film(self):
        self.publish()

        def attack(entry):
            request = read(self.output / entry["records"]["film_request"]["path"])
            request["source"]["final_state_sha256"] = "0" * 64
            entry["records"]["film_request"] = self.new_record(request)
            receipt = read(self.output / entry["records"]["film_receipt"]["path"])
            receipt["identity_sha256"] = gallery._sha(request)
            receipt["source"] = request["source"]
            receipt["artifacts"]["request.json"] = gallery._fingerprint(
                entry["records"]["film_request"]
            )
            entry["records"]["film_receipt"] = self.new_record(receipt)

        self.replace_entry(attack)
        with self.assertRaisesRegex(ValueError, "Edited film source"):
            gallery.verify_review(self.output)

    def test_rehashed_intermediate_camera_pose_is_rejected(self):
        self.publish()

        def attack(entry):
            camera = read(self.output / entry["records"]["motion_camera"]["path"])
            camera[gallery.MOTION_FRAMES // 2]["tilt_degrees"] += 1
            entry["records"]["motion_camera"] = self.new_record(camera)
            receipt = read(self.output / entry["records"]["motion_receipt"]["path"])
            receipt["artifacts"]["camera.json"] = gallery._fingerprint(
                entry["records"]["motion_camera"]
            )
            entry["records"]["motion_receipt"] = self.new_record(receipt)
            result = read(self.output / entry["records"]["motion_result"]["path"])
            result["render_receipt_sha256"] = entry["records"]["motion_receipt"]["sha256"]
            entry["records"]["motion_result"] = self.new_record(result)

        self.replace_entry(attack)
        with self.assertRaisesRegex(ValueError, "Camera pose"):
            gallery.verify_review(self.output)

    def test_final_verification_hashes_video_even_when_refresh_reuses_it(self):
        data = self.publish()
        path = self.output / data["rows"][0]["film"]
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with patch.object(
            gallery, "verify_case", side_effect=AssertionError("unexpected raw check")
        ):
            self.publish()
        with self.assertRaises(ValueError):
            gallery.verify_review(self.output)

    def test_pending_cohort_counts_future_pairs_without_inventing_entries(self):
        seeds = ["0x" + f"{i:064x}" for i in range(10)]
        generation = {"identity_sha256": "a" * 64, "seeds": seeds}
        source = self.root / "pending-cohort.json"
        write(source, generation)
        with patch.object(gallery, "validate_cohort_plan", side_effect=lambda d: d):
            data = self.publish(pending_cohorts=[source])
            self.assertEqual(data["progress"]["expected_pairs"], 152)
            self.assertEqual(data["progress"]["ready_pairs"], 2)
            self.assertEqual(data["progress"]["new_seed_count"], 10)
            self.assertFalse(data["progress"]["complete"])
            self.assertEqual(data["progress"]["preparing_source_count"], 10)
            source.unlink()
            self.assertEqual(self.publish(), data)
            with self.assertRaisesRegex(ValueError, "cannot change or disappear"):
                self.publish(pending_cohorts=[])


@unittest.skipUnless(shutil.which("node"), "Node.js is required for viewer interaction checks")
class FilmReviewPageTests(unittest.TestCase):
    def run_page(self, assertions, data):
        fixture = page_tests.ReviewPageTests("runTest")
        with patch.object(
            page_tests, "document", side_effect=lambda title, **_kwargs: gallery.document(title)
        ):
            fixture.run_page(assertions, enabled=True, data=data)

    def test_full_width_seed_values_survive_short_labels_and_film_selection(self):
        fixture = page_tests.ReviewPageTests("runTest")
        data = fixture.data(reference_variant="control")
        mapping = {"A": "0x00" + "ab" * 31, "B": "0x" + "de" * 32, "C": "0x" + "f0" * 32}
        data["version"] = gallery.VERSION
        data["seeds"] = [mapping[s] for s in data["seeds"]]
        for row in data["rows"]:
            row["seed"] = mapping[row["seed"]]
            row["settings"]["seed"] = row["seed"]
        for pick in data["picks"]:
            pick["seed"] = mapping[pick["seed"]]
        data["progress"] = {
            "expected_pairs": 171,
            "ready_pairs": 3,
            "visible_pairs": 3,
            "new_seed_count": 10,
            "reference_seed_count": 3,
            "preparing_source_count": 10,
            "awaiting_control_case_ids": [],
            "complete": False,
        }
        self.run_page(
            r"""
assert.equal(get('seed').value,DATA.seeds[0]);
assert.equal(get('seed').firstChild.title,DATA.seeds[0]);
assert.ok(get('seed').firstChild.textContent.includes('…'));
assert.ok(get('seed').firstChild.textContent.length<30);
assert.equal(get('left').value,'control');
assert.match(get('settings').textContent,new RegExp(DATA.seeds[0]));
assert.match(get('pair-progress').textContent,/3 of 171/);
assert.equal(get('refresh-pairs').hidden,false);
mode('film');assert.equal(get('right').value,'compact');
assert.ok(videos().every(video=>video.controls&&video.plays===0));
assert.equal(get('seed').firstChild.value,DATA.seeds[0]);
assert.equal(get('seed').firstChild.title,DATA.seeds[0]);
""",
            data,
        )

    def test_zero_ready_page_does_not_render_placeholder_films(self):
        data = {
            "version": gallery.VERSION,
            "rows": [],
            "seeds": [],
            "picks": [],
            "progress": {
                "expected_pairs": 171,
                "ready_pairs": 0,
                "visible_pairs": 0,
                "new_seed_count": 10,
                "reference_seed_count": 3,
                "preparing_source_count": 10,
                "awaiting_control_case_ids": [],
                "complete": False,
            },
        }
        self.run_page(
            r"""
assert.match(get('status').textContent,/No verified/);
assert.match(get('pair-progress').textContent,/0 of 171/);
assert.equal(videos().length,0);assert.equal(get('grid').children.length,0);
""",
            data,
        )


class PatternFilmGalleryTests(unittest.TestCase):
    study_family = "pattern-studies-v1"
    setUp = FilmGalleryTests.setUp
    publish = FilmGalleryTests.publish

    def test_pattern_pairs_are_portable_and_keep_their_versioned_family(self):
        data = self.publish(title="Starting paint studies")
        self.assertEqual(data["study_family"], self.study_family)
        self.assertEqual(data["provenance"]["study_family"], self.study_family)
        self.assertEqual([row["variant"] for row in data["rows"]], ["lacuna-banks", "folded-sash"])
        self.assertTrue(all(row["study_family"] == self.study_family for row in data["rows"]))
        self.assertTrue(data["progress"]["complete"])
        page = (self.output / "index.html").read_text()
        self.assertIn("same ten trajectory seeds", page)
        self.assertIn("shared trajectory seeds", page)
        self.assertNotIn('" new seeds + "', page)
        for descriptor in data["provenance"]["entries"]:
            entry = read(self.output / descriptor["path"])
            self.assertEqual(entry["study_family"], self.study_family)
        shutil.rmtree(self.fixture.output)
        self.assertEqual(gallery.verify_review(self.output), data)

    def test_seed_visibility_waits_for_its_reference_pattern_instead_of_control(self):
        certificate = self.fixture.folders[0] / "study.json"
        record = read(certificate)
        certificate.unlink()
        partial = self.publish()
        self.assertEqual(partial["progress"]["ready_pairs"], 1)
        self.assertEqual(partial["progress"]["visible_pairs"], 0)
        self.assertEqual(
            partial["progress"]["awaiting_control_case_ids"], [self.fixture.folders[1].name]
        )
        write(certificate, record)
        completed = self.publish()
        self.assertEqual(completed["progress"]["visible_pairs"], 2)
        self.assertEqual(gallery.verify_review(self.output), completed)

    def test_family_cannot_be_removed_from_a_portable_publication(self):
        data = self.publish()
        altered = copy.deepcopy(data)
        altered["provenance"].pop("study_family")
        altered["identity_sha256"] = gallery._sha(
            {k: v for k, v in altered.items() if k != "identity_sha256"}
        )
        write(self.output / "comparison.json", altered)
        with self.assertRaisesRegex(ValueError, "Mixed study families"):
            gallery.verify_review(self.output)

    def test_pattern_and_legacy_families_require_separate_publications(self):
        other = self.root / "legacy"
        other.mkdir()
        write(other / "plan.json", self.fixture.make_plan(options=["control"]))
        with self.assertRaisesRegex(ValueError, "Mixed study families"):
            gallery.publish_review(self.output, [self.fixture.output, other])
        self.assertFalse(self.output.exists())


class PatternFilmGalleryV2Tests(PatternFilmGalleryTests):
    study_family = "pattern-studies-v2"

    def test_pattern_versions_require_separate_publications(self):
        other = self.root / "version-one"
        other.mkdir()
        write(
            other / "plan.json",
            self.fixture.make_plan(
                study_family="pattern-studies-v1", options=["lacuna-banks", "folded-sash"]
            ),
        )
        with self.assertRaisesRegex(ValueError, "Mixed study families"):
            gallery.publish_review(self.output, [self.fixture.output, other])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
