"""Portable views, exact pigment swatches, and honest material comparisons."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.run import write_png
from tools.estuary_studio.common import artifact, encoded, read, write

from . import gallery
from .palette import generate_palette, normalize_seed


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "gallery"
        self.verifier = patch.object(gallery, "verify_run", side_effect=self.verified_case)
        self.verifier.start()
        self.addCleanup(self.verifier.stop)

    @staticmethod
    def verified_case(case):
        request, receipt = read(case / "request.json"), read(case / "receipt.json")
        if not receipt["complete"]:
            raise ValueError("Incomplete source case")
        return request, receipt

    def case(
        self,
        seed="0xbc53af1cd380",
        count=3,
        looks=None,
        *,
        film=True,
        physical=None,
        source_hash=None,
        decode=True,
        orbit_frames=3,
    ):
        looks = ["layered", "homogeneous"] if looks is None else looks
        path = self.root / f"case-{len(list(self.root.glob('case-*')))}"
        path.mkdir()
        palette = generate_palette(seed, count)
        source_hash = source_hash or hashlib.sha256(normalize_seed(seed).encode()).hexdigest()
        physical = (
            physical or hashlib.sha256(f"{normalize_seed(seed)}/{count}".encode()).hexdigest()
        )
        request = {
            "source": {"seed": seed, "sha256": source_hash},
            "palette": palette,
            "events": [{"fraction": 0.3, "position": [0.1, -0.1], "strength": 0.7}],
            "recipe": {
                "name": "Fixture",
                "chromatic_count": count,
                "looks": looks,
                "simulation": {"steps": 10},
                "render": {
                    "still_resolution": [16, 12],
                    "resolution": [16, 12],
                    "formation_frames": 6,
                    "hold_frames": 2,
                    "orbit_frames": orbit_frames,
                    "fps": 24,
                },
            },
            "mode": "film" if film else "still",
        }
        identity = hashlib.sha256(encoded(request)).hexdigest()
        files, views = {}, {}
        for index, look in enumerate(looks):
            (path / look).mkdir()
            pixels = np.full((12, 16, 3), 0.2 + 0.05 * index + 0.02 * count, dtype="f4")
            write_png(path / look / "poster.png", pixels)
            files[f"{look}/poster.png"] = artifact(path / look / "poster.png")
            if film:
                (path / look / "film.mp4").write_bytes(f"{identity}-{look}".encode())
                files[f"{look}/film.mp4"] = artifact(path / look / "film.mp4")
            views[look] = {
                "physical_state_sha256": physical,
                "movie": {
                    "full_decode_verified": decode,
                    "frames": 6 + 2 + orbit_frames - 1,
                    "fps": 24,
                    "resolution": [16, 12],
                }
                if film
                else None,
            }
        receipt = {
            "complete": True,
            "identity_sha256": identity,
            "source": request["source"],
            "source_fraction": 1.0,
            "final_step": 10,
            "physical_state_sha256": physical,
            "looks": views,
            "artifacts": files,
        }
        for name, value in (
            ("request", request),
            ("receipt", receipt),
            ("palette", palette),
            ("events", request["events"]),
        ):
            write(path / f"{name}.json", value)
        return path

    def rehash_collection(self, collection):
        write(self.output / "collection.json", collection)
        curation = read(self.output / "curation.json")
        curation["artifacts"]["collection.json"] = artifact(self.output / "collection.json")
        write(self.output / "curation.json", curation)

    def test_six_cases_publish_nine_views_in_seed_count_look_order(self):
        seeds = ["0xbc53af1cd380", "0x808861c25b6c", "0xb7f327f9f722"]
        cases = []
        for seed in seeds:
            cases.extend([self.case(seed, 3), self.case(seed, 5, ["layered"])])
        result = gallery.build_gallery(self.output, cases)
        self.assertEqual(result, (self.output / "index.html").resolve())
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(len(curation["sources"]), 6)
        self.assertEqual(len(collection["studies"]), 9)
        expected = [
            (seed, count, look)
            for seed in seeds
            for count, look in ((3, "layered"), (5, "layered"), (3, "homogeneous"))
        ]
        self.assertEqual(
            [(s["seed"], s["chromatic_count"], s["group"]) for s in collection["studies"]], expected
        )
        for study in collection["studies"]:
            self.assertTrue((self.output / study["image"]).is_file())
            self.assertTrue((self.output / study["film"]).is_file())
            self.assertEqual(study["film_caption"], gallery.FILM_CAPTION)
            self.assertEqual(len(study["swatches"]), study["chromatic_count"] + 1)
            self.assertEqual(study["swatches"][-1]["role"], "chalk")
        for case in cases:
            case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_comparisons_distinguish_same_physics_from_added_pigments(self):
        gallery.build_gallery(self.output, [self.case(), self.case(count=5, looks=["layered"])])
        collection, _ = gallery.verify_gallery(self.output)
        layered, five, blended = collection["studies"]
        self.assertEqual(layered["comparison_id"], blended["id"])
        self.assertEqual(layered["comparison_caption"], "Blended · same material history")
        self.assertEqual(blended["comparison_id"], layered["id"])
        self.assertEqual(blended["comparison_caption"], "Layered · same material history")
        self.assertEqual(five["comparison_id"], layered["id"])
        self.assertEqual(five["comparison_caption"], "Three colors · layered")
        self.assertNotEqual(five["physical_state_sha256"], layered["physical_state_sha256"])
        self.assertEqual(layered["palette_record"], blended["palette_record"])
        self.assertEqual(layered["swatches"], blended["swatches"])

    def test_palette_swatches_are_exact_generated_display_values(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        collection, _ = gallery.verify_gallery(self.output)
        palette = read(case / "palette.json")
        study = collection["studies"][0]
        self.assertEqual(
            [item["rgba"] for item in study["swatches"]],
            [[*color, 1.0] for color in palette["pigments_srgb"]],
        )
        self.assertEqual([item["name"] for item in study["swatches"]], palette["pigment_names"])
        self.assertEqual(study["name"], "Mineral Tide · 3 colors")
        self.assertEqual(read(self.output / study["palette_record"]), palette)

    def test_no_comparison_is_invented_when_an_optical_pair_is_absent(self):
        gallery.build_gallery(self.output, [self.case(looks=["layered"])])
        collection, _ = gallery.verify_gallery(self.output)
        self.assertIsNone(collection["studies"][0]["comparison_id"])
        self.assertIsNone(collection["studies"][0]["baseline"])

    def test_different_physical_histories_cannot_be_presented_as_paired_looks(self):
        cases = [
            self.case(looks=["layered"], physical="a" * 64),
            self.case(looks=["homogeneous"], physical="b" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "different physical states"):
            gallery.build_gallery(self.output, cases)
        self.assertFalse((self.output / "index.html").exists())

    def test_same_seed_different_source_recordings_cannot_be_compared(self):
        cases = [self.case(), self.case(count=5, looks=["layered"], source_hash="f" * 64)]
        with self.assertRaisesRegex(ValueError, "different source trajectories"):
            gallery.build_gallery(self.output, cases)

    def test_wrong_seed_comparison_rejected_even_with_updated_collection_hash(self):
        gallery.build_gallery(self.output, [self.case(), self.case(seed="0x808861c25b6c")])
        collection = read(self.output / "collection.json")
        first, other = collection["studies"][0], collection["studies"][2]
        first["comparison_id"], first["baseline"] = other["id"], other["image"]
        self.rehash_collection(collection)
        with self.assertRaisesRegex(ValueError, "Comparison"):
            gallery.verify_gallery(self.output)

    def test_caption_palette_and_physical_tampering_cannot_be_rehashed_away(self):
        gallery.build_gallery(self.output, [self.case()])
        original = read(self.output / "collection.json")
        for key, value in (
            ("chromatic_count", 5),
            ("physical_state_sha256", "0" * 64),
            ("seed", "0xffff"),
            ("name", "Different painting"),
        ):
            collection = read(self.output / "collection.json")
            collection["studies"][0][key] = value
            self.rehash_collection(collection)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)
        altered = read(self.output / "collection.json")
        altered["studies"][0]["swatches"][0]["rgba"][0] += 0.01
        self.rehash_collection(altered)
        with self.assertRaisesRegex(ValueError, "palette association"):
            gallery.verify_gallery(self.output)

    def test_proofs_require_explicit_still_publication(self):
        case = self.case(film=False)
        with self.assertRaisesRegex(ValueError, "needs its film"):
            gallery.build_gallery(self.output, [case])
        self.assertFalse(self.output.exists())
        gallery.build_gallery(self.output, [case], allow_stills=True)
        collection, _ = gallery.verify_gallery(self.output)
        self.assertTrue(all(study["film"] is None for study in collection["studies"]))

    def test_no_film_without_full_decode_evidence_can_be_published(self):
        with self.assertRaisesRegex(ValueError, "decode evidence"):
            gallery.build_gallery(self.output, [self.case(decode=False)])
        self.assertFalse((self.output / "index.html").exists())

    def test_all_cases_are_verified_before_publishing(self):
        good, bad = self.case(), self.case(count=5, looks=["layered"])
        receipt = read(bad / "receipt.json")
        receipt["complete"] = False
        write(bad / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            gallery.build_gallery(self.output, [good, bad])
        self.assertFalse(self.output.exists())

    def test_changed_input_during_copy_cannot_be_certified(self):
        case = self.case()

        def change_after_verification(path):
            request, receipt = self.verified_case(path)
            changed = {**request, "changed": True}
            write(path / "request.json", changed)
            return request, receipt

        with (
            patch.object(gallery, "verify_run", side_effect=change_after_verification),
            self.assertRaisesRegex(ValueError, "Source changed while copying"),
        ):
            gallery.build_gallery(self.output, [case])
        self.assertFalse((self.output / "curation.json").exists())

    def test_republication_is_stable_and_media_tampering_is_never_overwritten(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        original = (self.output / "curation.json").read_bytes()
        gallery.build_gallery(self.output, [case])
        self.assertEqual((self.output / "curation.json").read_bytes(), original)
        collection, _ = gallery.verify_gallery(self.output)
        image = self.output / collection["studies"][0]["image"]
        image.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            gallery.verify_gallery(self.output)
        with self.assertRaisesRegex(ValueError, "Published media changed"):
            gallery.build_gallery(self.output, [case])
        self.assertEqual(image.read_bytes(), b"tampered")

    def test_duplicate_case_and_publication_inside_archive_are_rejected(self):
        case = self.case()
        with self.assertRaisesRegex(ValueError, "more than once"):
            gallery.build_gallery(self.output, [case, case])
        with self.assertRaisesRegex(ValueError, "outside"):
            gallery.build_gallery(case / "gallery", [case])

    def test_interface_uses_native_movies_safe_labels_and_dynamic_comparison_caption(self):
        document = gallery.document('<script>alert("title")</script>')
        self.assertIn("&lt;script&gt;", document)
        self.assertNotIn('<script>alert("title")</script>', document)
        self.assertIn("<video", document)
        self.assertIn(">Play film</button>", document)
        self.assertIn("comparisonCaption", document)
        self.assertIn("study.comparison_caption", document)
        self.assertIn("study.swatches", document)
        self.assertIn("study.film_caption", document)
        self.assertIn("colors + shared chalk", document)
        self.assertIn("[hidden]{display:none!important}", document)
        self.assertIn("http://127.0.0.1:8786/", document)
        self.assertNotIn("Compare original", document)
        self.assertNotIn("Earlier Estuary", document)

    def test_template_is_independent_of_previous_gallery_prose_and_archived_as_runtime(self):
        from .run import runtime_identity

        expected = gallery.document("Confluence Fresco")
        with patch("tools.estuary_studio.gallery.DEPTH_DOCUMENT", "Different old interface"):
            self.assertEqual(gallery.document("Confluence Fresco"), expected)
        self.assertIn("gallery.html", runtime_identity()["estuary_confluence"])
        self.assertNotIn(gallery.TITLE_TOKEN, expected)

    def test_formation_only_movie_never_advertises_a_camera_orbit(self):
        gallery.build_gallery(self.output, [self.case(orbit_frames=1)])
        collection, _ = gallery.verify_gallery(self.output)
        for study in collection["studies"]:
            self.assertEqual(study["film_caption"], "Complete formation of the painting")
            self.assertEqual(study["film_frames"], 8)
            self.assertEqual(study["film_fps"], 24)
            self.assertEqual(study["film_resolution"], [16, 12])
            self.assertEqual(study["film_seconds"], 8 / 24)

    def test_rehashed_movie_caption_and_dimensions_still_match_original_recipe(self):
        gallery.build_gallery(self.output, [self.case(orbit_frames=1)])
        original = read(self.output / "collection.json")
        for key, value in (
            ("film_caption", gallery.FILM_CAPTION),
            ("film_resolution", [3840, 2880]),
            ("film_frames", 10),
            ("film_fps", 60),
            ("film_seconds", 100),
        ):
            collection = read(self.output / "collection.json")
            collection["studies"][0][key] = value
            self.rehash_collection(collection)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "formation caption"):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)


if __name__ == "__main__":
    unittest.main()
