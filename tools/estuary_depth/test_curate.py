"""Curated media pairing, source identity and per-study reference contracts."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from tools.estuary_depth import curate
from tools.estuary_depth import experiment as exp
from tools.estuary_depth.gallery import DOCUMENT
from tools.estuary_depth.test_experiment import bound_fixture


class CuratorTests(unittest.TestCase):
    def catalog(self, root, studies, baselines=None):
        path = root / "catalog.json"
        exp.write(path, {"studies": studies, "baselines": baselines or {}})
        return path

    def test_different_experiments_and_seeds_keep_catalog_order_and_correct_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound_fixture(
                root / "still-a", {"hero": {"name": "A", "render": {"resolution": [3840, 2880]}}}
            )
            bound_fixture(root / "motion-a", {"orbit": {"name": "Orbit"}}, motion=2)
            bound_fixture(
                root / "still-b", {"proof": {"name": "B"}}, seed="0xbb", render_identity="paint-b"
            )
            a, b = root / "a.png", root / "b.png"
            a.write_bytes(b"flat painting a")
            b.write_bytes(b"flat painting b")
            name = '</script><img src=x onerror="alert(1)">'
            catalog = self.catalog(
                root,
                [
                    {
                        "still": str(root / "still-a/hero"),
                        "motion": str(root / "motion-a/orbit"),
                        "name": name,
                    },
                    {"still": str(root / "still-b/proof")},
                ],
                {"0xaa": str(a), "0xbb": str(b)},
            )
            output = root / "gallery"
            curate.curate(catalog, output)
            collection = exp.read(output / "collection.json")
            first, second = collection["studies"]
            self.assertEqual((first["id"], second["id"]), ("study-001", "study-002"))
            self.assertEqual(first["name"], name)
            self.assertEqual(first["resolution"], [3840, 2880])
            self.assertIsNotNone(first["film"])
            self.assertIsNone(second["film"])
            self.assertEqual((output / first["baseline"]).read_bytes(), a.read_bytes())
            self.assertEqual((output / second["baseline"]).read_bytes(), b.read_bytes())
            self.assertNotIn(name, (output / "index.html").read_text())
            self.assertNotIn("innerHTML", DOCUMENT)
            self.assertNotIn("autoplay", DOCUMENT)
            self.assertIn("Camera orbit of the completed painting", DOCUMENT)
            self.assertFalse(
                any(path.suffix in (".blend", ".npz", ".exr") for path in output.rglob("*"))
            )
            curate.curate(catalog, output)
            self.assertEqual(exp.read(output / "collection.json"), collection)

    def test_motion_from_a_different_seed_or_original_painting_is_rejected(self):
        for seed, painting in (("0xbb", "paint-a"), ("0xaa", "paint-b")):
            with (
                self.subTest(seed=seed, painting=painting),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                bound_fixture(root / "still", {"hero": {"name": "Hero"}})
                bound_fixture(
                    root / "motion",
                    {"orbit": {"name": "Orbit"}},
                    seed=seed,
                    render_identity=painting,
                    motion=2,
                )
                catalog = self.catalog(
                    root,
                    [{"still": str(root / "still/hero"), "motion": str(root / "motion/orbit")}],
                )
                with self.assertRaisesRegex(ValueError, "same seed and original"):
                    curate.curate(catalog, root / "gallery")
                self.assertFalse((root / "gallery").exists())

    def test_undecoded_movie_and_corrupted_movie_are_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound_fixture(root / "still", {"hero": {"name": "Hero"}})
            bound_fixture(root / "motion", {"orbit": {"name": "Orbit"}}, motion=2)
            movie = root / "motion/orbit"
            catalog = self.catalog(
                root, [{"still": str(root / "still/hero"), "motion": str(movie)}]
            )
            receipt = exp.read(movie / "experiment-result.json")
            receipt["movie"]["full_decode_verified"] = False
            exp.write(movie / "experiment-result.json", receipt)
            with self.assertRaisesRegex(ValueError, "fully decoded"):
                curate.curate(catalog, root / "gallery")
            receipt["movie"]["full_decode_verified"] = True
            exp.write(movie / "experiment-result.json", receipt)
            (movie / "film.mp4").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                curate.curate(catalog, root / "gallery")

    def test_swapped_case_cannot_acquire_the_other_study_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound_fixture(root / "experiment", {"a": {"name": "A"}, "b": {"name": "B"}})
            shutil.rmtree(root / "experiment/b")
            shutil.copytree(root / "experiment/a", root / "experiment/b")
            catalog = self.catalog(root, [{"still": str(root / "experiment/b")}])
            with self.assertRaisesRegex(ValueError, "own archived input"):
                curate.curate(catalog, root / "gallery")

    def test_catalog_updates_keep_previous_hash_named_assets_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound_fixture(root / "a", {"hero": {"name": "A"}})
            bound_fixture(
                root / "b", {"hero": {"name": "B"}}, seed="0xbb", render_identity="paint-b"
            )
            catalog = self.catalog(root, [{"still": str(root / "a/hero")}])
            output = root / "gallery"
            curate.curate(catalog, output)
            old = exp.read(output / "collection.json")["studies"][0]["image"]
            old_hash = exp.digest(output / old)
            self.catalog(root, [{"still": str(root / "b/hero")}, {"still": str(root / "a/hero")}])
            curate.curate(catalog, output)
            self.assertEqual(exp.digest(output / old), old_hash)
            self.assertEqual(exp.read(output / "collection.json")["studies"][0]["seed"], "0xbb")

    def test_optional_formation_receipt_is_verified_and_bound_to_the_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound_fixture(root / "still", {"hero": {"name": "Hero"}})
            formation = root / "formation"
            formation.mkdir()
            (formation / "film.mp4").write_bytes(b"verified combined edit")
            receipt = {"source": {"seed": "0xaa", "render_identity": "paint-a", "sha256": "s" * 64}}
            receipt["movie"] = {
                "sha256": exp.digest(formation / "film.mp4"),
                "bytes": (formation / "film.mp4").stat().st_size,
            }
            exp.write(formation / "receipt.json", receipt)
            verifier = ModuleType("tools.estuary_depth.film")
            verifier.verify_complete = Mock(return_value=receipt)
            catalog = self.catalog(
                root, [{"still": str(root / "still/hero"), "formation": str(formation)}]
            )
            with patch.dict("sys.modules", {"tools.estuary_depth.film": verifier}):
                curate.curate(catalog, root / "gallery")
                verifier.verify_complete.assert_called_once_with(formation.resolve())
                study = exp.read(root / "gallery/collection.json")["studies"][0]
                self.assertEqual(
                    (root / "gallery" / study["formation"]).read_bytes(), b"verified combined edit"
                )
                receipt["source"]["render_identity"] = "other painting"
                with self.assertRaisesRegex(ValueError, "same completed painting"):
                    curate.curate(catalog, root / "other-gallery")

    def test_same_source_formation_cannot_be_paired_with_a_different_orbit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound_fixture(root / "still", {"hero": {"name": "Hero"}})
            bound_fixture(root / "motion", {"orbit": {"name": "Orbit"}}, motion=2)
            orbit = exp.read(root / "motion/orbit/experiment-result.json")["movie"]
            expected = {key: orbit[key] for key in ("sha256", "bytes")}
            formation = root / "formation"
            formation.mkdir()
            (formation / "film.mp4").write_bytes(b"verified combined edit")
            receipt = {
                "source": {"seed": "0xaa", "render_identity": "paint-a", "sha256": "s" * 64},
                "movie": {
                    "sha256": exp.digest(formation / "film.mp4"),
                    "bytes": (formation / "film.mp4").stat().st_size,
                },
                "artifacts": {"inputs/orbit.mp4": dict(expected)},
            }
            exp.write(formation / "receipt.json", receipt)
            exp.write(formation / "request.json", {"inputs": {"orbit": dict(expected)}})
            verifier = ModuleType("tools.estuary_depth.film")
            verifier.verify_complete = Mock(return_value=receipt)
            catalog = self.catalog(
                root,
                [
                    {
                        "still": str(root / "still/hero"),
                        "motion": str(root / "motion/orbit"),
                        "formation": str(formation),
                    }
                ],
            )
            with patch.dict("sys.modules", {"tools.estuary_depth.film": verifier}):
                curate.curate(catalog, root / "correct-gallery")
                different = {"sha256": "different-orbit-from-the-same-painting", "bytes": 123}
                receipt["artifacts"]["inputs/orbit.mp4"] = different
                exp.write(formation / "request.json", {"inputs": {"orbit": different}})
                with self.assertRaisesRegex(ValueError, "exact paired camera orbit"):
                    curate.curate(catalog, root / "different-orbit-gallery")
                self.assertFalse((root / "different-orbit-gallery").exists())
                exp.write(formation / "request.json", {"inputs": {"orbit": expected}})
                with self.assertRaisesRegex(ValueError, "exact paired camera orbit"):
                    curate.curate(catalog, root / "inconsistent-input-gallery")

    def test_relative_paths_and_unknown_catalog_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self.catalog(root, [{"still": "relative/case"}])
            with self.assertRaisesRegex(ValueError, "absolute"):
                curate.curate(catalog, root / "gallery")
            catalog.write_text(json.dumps({"studies": [], "unexpected": 1}))
            with self.assertRaisesRegex(ValueError, "Catalog must contain"):
                curate.curate(catalog, root / "gallery")


if __name__ == "__main__":
    unittest.main()
