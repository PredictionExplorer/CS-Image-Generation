"""Curated media provenance and truthful gallery presentation contracts."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.run import write_png

from . import gallery
from .common import artifact, encoded, read, write


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "gallery"
        self.verifier = patch.object(gallery, "verified_run", side_effect=self.verified_case)
        self.verifier.start()
        self.addCleanup(self.verifier.stop)

    @staticmethod
    def verified_case(case):
        request, receipt = read(case / "request.json"), read(case / "receipt.json")
        if not receipt["complete"]:
            raise ValueError("Incomplete source painting")
        return request, receipt

    def case(self, family="fresco", *, seed="0xaaaa", film=True, name=None, dynamics=None):
        path = self.root / f"case-{len(list(self.root.glob('case-*')))}"
        path.mkdir()
        request = {
            "recipe": {
                "name": name or family.title(),
                "family": family,
                "dynamics": dynamics or ("fresco" if family == "fresco" else "monotype"),
                "simulation": {"steps": 10},
                "render": {"still_resolution": [16, 12]},
            },
            "source": {"seed": seed, "sha256": hashlib.sha256(seed.encode()).hexdigest()},
            "mode": "film" if film else "still",
        }
        identity = hashlib.sha256(encoded(request)).hexdigest()
        rgb = np.full((12, 16, 3), 0.2 + len(list(self.root.glob("case-*"))) * 0.05, dtype="f4")
        write_png(path / "poster.png", rgb)
        files = {"poster.png": artifact(path / "poster.png")}
        if film:
            (path / "film.mp4").write_bytes(f"film-{identity}".encode())
            files["film.mp4"] = artifact(path / "film.mp4")
        receipt = {
            "complete": True,
            "identity_sha256": identity,
            "source": request["source"],
            "source_fraction": 1.0,
            "final_step": 10,
            "artifacts": files,
            "movie": {"full_decode_verified": True} if film else None,
        }
        write(path / "request.json", request)
        write(path / "receipt.json", receipt)
        return path

    def test_three_families_publish_stills_films_and_portable_records(self):
        cases = [self.case(family) for family in ("fresco", "monotype", "nocturne")]
        result = gallery.build_gallery(self.output, cases)
        self.assertEqual(result, (self.output / "index.html").resolve())
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(
            [s["group"] for s in collection["studies"]], ["fresco", "monotype", "nocturne"]
        )
        self.assertEqual(len(curation["sources"]), 3)
        for study in collection["studies"]:
            self.assertIn("assets/", study["image"])
            self.assertTrue((self.output / study["film"]).is_file())
            self.assertEqual(study["film_caption"], gallery.FILM_CAPTION)
        # The published gallery remains verifiable after originals move away.
        for case in cases:
            case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_baselines_never_cross_seed_boundaries(self):
        a, b = self.case(seed="0xaaaa"), self.case(seed="0xbbbb")
        reference = self.root / "reference.png"
        write_png(reference, np.full((12, 16, 3), 0.5, dtype="f4"))
        gallery.build_gallery(self.output, [a, b], baselines=[f"0xaaaa={reference}"])
        collection, _ = gallery.verify_gallery(self.output)
        self.assertIsNotNone(collection["studies"][0]["baseline"])
        self.assertIsNone(collection["studies"][1]["baseline"])
        with self.assertRaisesRegex(ValueError, "multiple seeds"):
            gallery.baseline_arguments([str(reference)], {"0xaaaa", "0xbbbb"})
        self.assertEqual(
            gallery.baseline_arguments([str(reference)], {"0xaaaa"}), {"0xaaaa": reference}
        )

    def test_duplicate_or_unselected_baselines_are_rejected(self):
        for values in (
            ["0xffff=/somewhere.png"],
            ["0xaaaa=/first.png", "0xaaaa=/second.png"],
            ["not-a-seed=/first.png"],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                gallery.baseline_arguments(values, {"0xaaaa"})

    def test_proofs_need_explicit_still_only_publication(self):
        case = self.case(film=False)
        with self.assertRaisesRegex(ValueError, "needs its film"):
            gallery.build_gallery(self.output, [case])
        self.assertFalse(self.output.exists())
        gallery.build_gallery(self.output, [case], allow_stills=True)
        collection, _ = gallery.verify_gallery(self.output)
        self.assertIsNone(collection["studies"][0]["film"])

    def test_all_cases_are_verified_before_publishing_any_gallery(self):
        valid, invalid = self.case(), self.case(name="Incomplete")
        receipt = read(invalid / "receipt.json")
        receipt["complete"] = False
        write(invalid / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            gallery.build_gallery(self.output, [valid, invalid])
        self.assertFalse(self.output.exists())

    def test_changed_request_during_copy_cannot_be_published(self):
        case = self.case()

        def changed_after_verification(path):
            request, receipt = self.verified_case(path)
            altered = {**request, "changed_after_verification": True}
            write(path / "request.json", altered)
            return request, receipt

        with (
            patch.object(gallery, "verified_run", side_effect=changed_after_verification),
            self.assertRaisesRegex(ValueError, "Source changed while copying"),
        ):
            gallery.build_gallery(self.output, [case])
        self.assertFalse((self.output / "index.html").exists())
        self.assertFalse((self.output / "curation.json").exists())

    def test_republication_preserves_all_media_and_curation_bytes(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        before = (self.output / "curation.json").read_bytes()
        gallery.build_gallery(self.output, [case])
        self.assertEqual((self.output / "curation.json").read_bytes(), before)

    def test_duplicate_selection_and_source_archive_publication_are_rejected(self):
        case = self.case()
        with self.assertRaisesRegex(ValueError, "more than once"):
            gallery.build_gallery(self.output, [case, case])
        with self.assertRaisesRegex(ValueError, "outside"):
            gallery.build_gallery(case / "gallery", [case])

    def test_media_tampering_is_rejected_and_never_silently_replaced(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        collection, _ = gallery.verify_gallery(self.output)
        media = self.output / collection["studies"][0]["image"]
        media.write_bytes(b"changed published media")
        with self.assertRaises(ValueError):
            gallery.verify_gallery(self.output)
        with self.assertRaisesRegex(ValueError, "Published media changed"):
            gallery.build_gallery(self.output, [case])
        self.assertEqual(media.read_bytes(), b"changed published media")

    def test_caption_seed_cannot_change_even_with_updated_collection_hash(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        collection = read(self.output / "collection.json")
        collection["studies"][0]["seed"] = "0xeeee"
        write(self.output / "collection.json", collection)
        curation = read(self.output / "curation.json")
        curation["artifacts"]["collection.json"] = artifact(self.output / "collection.json")
        write(self.output / "curation.json", curation)
        with self.assertRaisesRegex(ValueError, "association"):
            gallery.verify_gallery(self.output)

    def test_nocturne_remains_named_nocturne_when_its_dynamics_are_fresco(self):
        case = self.case("nocturne", dynamics="fresco")
        gallery.build_gallery(self.output, [case])
        collection, _ = gallery.verify_gallery(self.output)
        study = collection["studies"][0]
        self.assertEqual(study["group"], "nocturne")
        self.assertEqual(study["dynamics"], "fresco")

    def test_gallery_document_has_truthful_movie_labels_and_escaped_title(self):
        result = gallery.document('<script>alert("title")</script>')
        self.assertIn("&lt;script&gt;", result)
        self.assertNotIn('<script>alert("title")</script>', result)
        self.assertIn(">Play film</button>", result)
        self.assertIn(gallery.FILM_CAPTION, result)
        self.assertNotIn("Camera orbit of the completed painting", result)
        self.assertNotIn("$('formation')", result)
        self.assertIn("[hidden]{display:none!important}", result)
        self.assertIn("Compare original", result)
        self.assertIn("Download the film", result)
        self.assertIn("nocturne:'Nocturne'", result)


if __name__ == "__main__":
    unittest.main()
