"""Portable appearance media and exact presentation-to-film association."""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import srgb_to_linear
from tools.estuary.run import write_png
from tools.estuary_studio.common import artifact, encoded, read, write

from . import appearance_gallery as gallery
from .appearance import PRESETS, VERSION, presentation
from .palette import generate_palette
from .run import frame_plan, surface_configs, validate_recipe


class AppearanceGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "gallery"
        for name in ("verify_study", "verify_run"):
            context = patch.object(gallery, name, side_effect=self.verify_fixture)
            context.start()
            self.addCleanup(context.stop)

    @staticmethod
    def verify_fixture(folder):
        request, receipt = read(folder / "request.json"), read(folder / "receipt.json")
        if not receipt["complete"]:
            raise ValueError("Incomplete source fixture")
        return request, receipt

    def study(self, seed="0xbc53af1cd380", count=5, names=("white", "palette-night", "relief")):
        folder = self.root / f"study-{len(list(self.root.glob('study-*')))}"
        folder.mkdir()
        recipe = validate_recipe(
            {
                "chromatic_count": count,
                "palette_mode": "harmonic",
                "looks": ["layered"],
                "simulation": {"resolution": [128, 96], "steps": 10},
                "surface": {"finish": "crisp"},
                "render": {
                    "resolution": [128, 96],
                    "still_resolution": [128, 96],
                    "formation_frames": 6,
                    "orbit_frames": 3,
                    "hold_frames": 2,
                },
            }
        )
        parent = {
            "source": {"seed": seed, "sha256": hashlib.sha256(seed.encode()).hexdigest()},
            "palette": generate_palette(seed, count, mode="harmonic"),
            "events": [],
            "recipe": recipe,
            "surface_configs": surface_configs(recipe),
            "spectral": None,
        }
        physical = {
            "complete": True,
            "identity_sha256": hashlib.sha256(encoded(parent)).hexdigest(),
            "physical_state_sha256": hashlib.sha256(f"{seed}/{count}".encode()).hexdigest(),
            "source_fraction": 1.0,
            "final_step": 10,
        }
        request = {
            "version": VERSION,
            "parent_identity_sha256": physical["identity_sha256"],
            "physical_state_sha256": physical["physical_state_sha256"],
            "seed": seed,
            "source_sha256": parent["source"]["sha256"],
            "chromatic_count": count,
            "resolution": [128, 96],
            "presentations": list(names),
            "looks": [presentation(parent, name) for name in names],
        }
        files, looks = {}, {}
        for filename, value in (("parent-request.json", parent), ("parent-receipt.json", physical)):
            write(folder / filename, value)
            files[filename] = artifact(folder / filename)
        for look in request["looks"]:
            name = look["id"]
            target = folder / name
            target.mkdir()
            image = np.empty((96, 128, 3), dtype="f4")
            image[:] = look["background"]["ground_linear"]
            image[24:72, 32:96] = srgb_to_linear(parent["palette"]["pigments_srgb"][0])
            write_png(target / "poster.png", image)
            write_png(target / "preview.png", image[::4, ::4], depth=8)
            np.save(target / "poster-linear.npy", image)
            write(target / "background.json", look["background"])
            for filename in ("poster.png", "preview.png", "poster-linear.npy", "background.json"):
                files[f"{name}/{filename}"] = artifact(target / filename)
            looks[name] = {"image_balance": [0.0, 0.0]}
        receipt = {
            "complete": True,
            "identity_sha256": hashlib.sha256(encoded(request)).hexdigest(),
            "artifacts": files,
            "looks": looks,
        }
        write(folder / "request.json", request)
        write(folder / "receipt.json", receipt)
        return folder

    def film(
        self,
        study,
        look_id="white",
        *,
        state=None,
        surface_change=None,
        camera_change=None,
        random_palette=False,
        source_change=False,
        decode=True,
    ):
        parent, physical = read(study / "parent-request.json"), read(study / "parent-receipt.json")
        look = next(x for x in read(study / "request.json")["looks"] if x["id"] == look_id)
        request = copy.deepcopy(parent)
        request["recipe"]["surface"] = copy.deepcopy(look["surface"])
        request["recipe"]["surface"].update(surface_change or {})
        request["surface_configs"] = {"layered": copy.deepcopy(request["recipe"]["surface"])}
        request["recipe"]["render"].update(
            still_tilt_degrees=look["camera"]["tilt_degrees"],
            azimuth_end=look["camera"]["azimuth_degrees"],
        )
        if camera_change is not None:
            request["recipe"]["render"]["still_tilt_degrees"] = camera_change
        if random_palette:
            request["recipe"]["palette_mode"] = "random"
            request["palette"] = generate_palette(
                parent["source"]["seed"], parent["recipe"]["chromatic_count"], mode="random"
            )
        if source_change:
            request["source"]["sha256"] = "f" * 64
        request["mode"] = "film"
        request["frames"] = frame_plan(request["recipe"])
        identity = hashlib.sha256(encoded(request)).hexdigest()
        folder = self.root / f"film-{len(list(self.root.glob('film-*')))}"
        folder.mkdir()
        (folder / "layered").mkdir()
        (folder / "layered/film.mp4").write_bytes(identity.encode())
        info = artifact(folder / "layered/film.mp4")
        movie = {
            "full_decode_verified": decode,
            "frames": len(request["frames"]),
            "fps": request["recipe"]["render"]["fps"],
            "resolution": [128, 96],
        }
        material = state or physical["physical_state_sha256"]
        receipt = {
            "complete": True,
            "identity_sha256": identity,
            "source_fraction": 1.0,
            "final_step": 10,
            "physical_state_sha256": material,
            "artifacts": {"layered/film.mp4": info},
            "looks": {"layered": {"physical_state_sha256": material, "movie": movie}},
        }
        write(folder / "request.json", request)
        write(folder / "receipt.json", receipt)
        return folder

    def rehash_collection(self, collection):
        write(self.output / "collection.json", collection)
        curation = read(self.output / "curation.json")
        curation["artifacts"]["collection.json"] = artifact(self.output / "collection.json")
        write(self.output / "curation.json", curation)

    def test_ten_seeds_and_three_extra_counts_publish_portable_previews_and_full_images(self):
        studies = [self.study(hex(seed + 100), 5) for seed in range(10)]
        studies += [self.study(hex(seed + 100), 3) for seed in range(3)]
        self.assertEqual(gallery.build_gallery(self.output, studies), self.output / "index.html")
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(len(collection["parents"]), 13)
        self.assertEqual(len({entry["seed"] for entry in collection["parents"]}), 10)
        self.assertEqual(len(curation["sources"]), 13)
        self.assertEqual(len(list(self.output.rglob("*.npy"))), 0)
        for entry in collection["parents"]:
            self.assertEqual(
                {view["id"] for view in entry["looks"]}, {"white", "palette-night", "relief"}
            )
            self.assertTrue(all(view["film"] is None for view in entry["looks"]))
            for view in entry["looks"]:
                self.assertTrue((self.output / view["image"]).is_file())
                self.assertTrue((self.output / view["preview"]).is_file())
        for folder in studies:
            folder.rename(folder.with_name(folder.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_all_eight_presentations_remain_named_and_seed_bound(self):
        study = self.study(names=tuple(PRESETS))
        gallery.build_gallery(self.output, [study])
        collection, _ = gallery.verify_gallery(self.output)
        views = collection["parents"][0]["looks"]
        self.assertEqual([look["id"] for look in views], list(PRESETS))
        self.assertEqual(len(views), 8)
        self.assertEqual(views[-1]["name"], "Satin reflection")

    def test_exact_matched_film_attaches_only_to_its_presentation(self):
        study = self.study()
        film = self.film(study, "palette-night")
        gallery.build_gallery(self.output, [study], films=[film])
        collection, _ = gallery.verify_gallery(self.output)
        views = {look["id"]: look for look in collection["parents"][0]["looks"]}
        self.assertIsNone(views["white"]["film"])
        self.assertIsNone(views["relief"]["film"])
        self.assertTrue((self.output / views["palette-night"]["film"]["src"]).is_file())
        self.assertIn("Complete formation", views["palette-night"]["film"]["caption"])
        film.rename(film.with_name(film.name + "-moved"))
        study.rename(study.with_name(study.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_changed_state_surface_camera_palette_or_source_never_attaches_a_film(self):
        study = self.study()
        cases = [
            self.film(study, state="0" * 64),
            self.film(study, surface_change={"exposure": 0.9}),
            self.film(study, camera_change=19),
            self.film(study, random_palette=True),
            self.film(study, source_change=True),
        ]
        gallery.build_gallery(self.output, [study], films=cases)
        collection, curation = gallery.verify_gallery(self.output)
        self.assertTrue(all(view["film"] is None for view in collection["parents"][0]["looks"]))
        self.assertEqual(len(curation["excluded_films"]), 5)
        self.assertEqual(curation["films"], [])

    def test_verified_matching_films_can_be_added_later_without_changing_posters(self):
        study = self.study()
        gallery.build_gallery(self.output, [study])
        before, _ = gallery.verify_gallery(self.output)
        films = [self.film(study)]
        gallery.build_gallery(self.output, [study], films=films)
        after, _ = gallery.verify_gallery(self.output)
        self.assertEqual(
            [v["image"] for v in before["parents"][0]["looks"]],
            [v["image"] for v in after["parents"][0]["looks"]],
        )
        self.assertIsNotNone(after["parents"][0]["looks"][0]["film"])

    def test_duplicate_parent_or_ambiguous_film_selection_is_rejected(self):
        study = self.study()
        with self.assertRaisesRegex(ValueError, "one appearance"):
            gallery.build_gallery(self.output, [study, study])
        a = self.film(study)
        # A second valid movie uses a different frame count but the same painting.
        b = self.film(study)
        request = read(b / "request.json")
        request["recipe"]["render"]["formation_frames"] = 11
        request["frames"] = frame_plan(request["recipe"])
        write(b / "request.json", request)
        receipt = read(b / "receipt.json")
        receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
        receipt["looks"]["layered"]["movie"]["frames"] = len(request["frames"])
        write(b / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Several films"):
            gallery.build_gallery(self.output, [study], films=[a, b])
        with self.assertRaisesRegex(ValueError, "outside"):
            gallery.build_gallery(study / "published", [study])

    def test_all_sources_must_be_complete_before_any_publication(self):
        good, bad = self.study(), self.study(seed="0x1234")
        receipt = read(bad / "receipt.json")
        receipt["complete"] = False
        write(bad / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            gallery.build_gallery(self.output, [good, bad])
        self.assertFalse(self.output.exists())

    def test_matched_film_requires_full_decode_evidence(self):
        study = self.study()
        with self.assertRaisesRegex(ValueError, "decode evidence"):
            gallery.build_gallery(self.output, [study], films=[self.film(study, decode=False)])
        self.assertFalse(self.output.exists())

    def test_published_media_changes_are_detected_and_never_overwritten(self):
        study = self.study()
        gallery.build_gallery(self.output, [study])
        collection, _ = gallery.verify_gallery(self.output)
        path = self.output / collection["parents"][0]["looks"][0]["preview"]
        path.write_bytes(b"changed preview")
        with self.assertRaises(ValueError):
            gallery.verify_gallery(self.output)
        with self.assertRaisesRegex(ValueError, "Published media changed"):
            gallery.build_gallery(self.output, [study])
        self.assertEqual(path.read_bytes(), b"changed preview")

    def test_white_comparison_cannot_be_rebound_to_another_physical_parent(self):
        gallery.build_gallery(self.output, [self.study(), self.study(seed="0x1234")])
        collection, _ = gallery.verify_gallery(self.output)
        collection["parents"][0]["looks"][0]["preview"] = collection["parents"][1]["looks"][0][
            "preview"
        ]
        self.rehash_collection(collection)
        with self.assertRaisesRegex(ValueError, "media differs"):
            gallery.verify_gallery(self.output)

    def test_film_cannot_be_moved_to_an_unmatched_look_after_rehash(self):
        study = self.study()
        gallery.build_gallery(self.output, [study], films=[self.film(study)])
        collection, _ = gallery.verify_gallery(self.output)
        views = collection["parents"][0]["looks"]
        views[1]["film"] = views[0]["film"]
        views[0]["film"] = None
        self.rehash_collection(collection)
        with self.assertRaisesRegex(ValueError, "does not match"):
            gallery.verify_gallery(self.output)

    def test_changed_source_record_during_copy_is_rejected(self):
        study = self.study()

        def changed(path):
            request, receipt = self.verify_fixture(path)
            write(path / "request.json", {**request, "altered": True})
            return request, receipt

        with (
            patch.object(gallery, "verify_study", side_effect=changed),
            self.assertRaisesRegex(ValueError, "Source changed while copying"),
        ):
            gallery.build_gallery(self.output, [study])
        self.assertFalse((self.output / "curation.json").exists())

    def test_template_uses_safe_labels_native_video_and_same_parent_white_lookup(self):
        text = gallery.document('<script>alert("title")</script>')
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn('<script>alert("title")</script>', text)
        self.assertIn("<video", text)
        self.assertIn("Choose a color count", text)
        self.assertIn("p.looks.find(v=>v.id==='white')", text)
        self.assertIn("parentsBySeed", text)
        self.assertIn("look.film", text)
        self.assertIn("[hidden]{display:none!important}", text)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for JavaScript syntax validation")
    def test_browser_script_syntax(self):
        script = re.findall(r"<script>([\s\S]*?)</script>", gallery.document("Appearance"))
        self.assertEqual(len(script), 1)
        path = self.root / "script.js"
        path.write_text(script[0])
        result = subprocess.run(
            [shutil.which("node"), "--check", str(path)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
