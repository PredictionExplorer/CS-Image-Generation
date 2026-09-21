"""Portable review integrity above the independently tested renderer archives."""

import copy
import hashlib
import shutil
import unittest
from unittest.mock import patch

from tools.estuary_studio.common import artifact, encoded, read, write

from . import choreography
from . import choreography_gallery as gallery
from . import test_composition_gallery as fixtures
from .laminate import layer_fractions


class ChoreographyGalleryTests(unittest.TestCase):
    def setUp(self):
        original_case = fixtures.gallery_fixtures.GalleryTests.case

        def case_with_projection(fixture, *args, **kwargs):
            kwargs["palette_mode"] = "composed"
            path = original_case(fixture, *args, **kwargs)
            request, receipt = fixture.verified_case(path)
            request["source"]["projection"] = {"aspect": 4 / 3, "method": "publisher fixture"}
            receipt["source"] = copy.deepcopy(request["source"])
            receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
            write(path / "request.json", request)
            write(path / "receipt.json", receipt)
            return path

        context = patch.object(fixtures.gallery_fixtures.GalleryTests, "case", case_with_projection)
        context.start()
        self.addCleanup(context.stop)
        self.fixture = fixtures.CompositionGalleryTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / "choreography-review"
        self.cases = self.fixture.cases[::6]
        # Small media use the existing receipt-bound fixtures; exact recipe
        # identification and native numerical verification have separate tests.
        for case in self.cases:
            request, receipt = self.fixture.fixture.verified_case(case)
            config = request["recipe"]["simulation"]
            config["initial_choreography"] = config.pop("initial_composition")
            config["initial_choreography"]["setup"] = "stretch-ovals"
            config["initial_choreography"]["version"] = "choreographed-paint-v1"
            config["initial_pattern"] = "choreographed"
            request["layout"] = {
                "config": copy.deepcopy(config["initial_choreography"]),
                "seed": request["source"]["seed"],
                "source_sha256": request["source"]["sha256"],
                "source_projection": request["source"]["projection"],
                "aspect": 4 / 3,
                "baseline_layer_fractions": layer_fractions(request["palette"], 4).tolist(),
                "effective_layer_fractions": layer_fractions(request["palette"], 4).tolist(),
            }
            self.fixture.refresh(case, request, receipt)
        for name, value in (
            ("REFERENCES", self.fixture.inputs_file),
            ("RELEASE", self.fixture.release_file),
        ):
            context = patch.object(gallery, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(gallery, "identify_recipe", return_value="stretch-ovals")
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(choreography, "validate_layout", side_effect=lambda value: value)
        context.start()
        self.addCleanup(context.stop)

    def build(self, **kwargs):
        return gallery.build_review(self.output, self.cases, self.fixture.reference, **kwargs)

    def test_reference_and_experiments_remain_portable_after_sources_are_removed(self):
        data = self.build()
        self.assertEqual(len(data["rows"]), 4)
        self.assertEqual(data["variants"], ["rc1", "stretch-ovals"])
        for row in data["rows"]:
            for key in ("image", "initial", "film", "request"):
                self.assertTrue((self.output / row[key]).is_file())
        for folder in (*self.cases, self.fixture.reference):
            shutil.rmtree(folder)
        self.assertEqual(gallery.verify_review(self.output), data)

    def test_shared_controls_and_pigment_amounts_cannot_drift(self):
        case = self.cases[0]
        original_request, original_receipt = self.fixture.fixture.verified_case(case)
        budget = read(case / "mass-budget.json")
        for attack in ("flow", "events", "mass", "layout"):
            with self.subTest(attack=attack):
                request, receipt = copy.deepcopy(original_request), copy.deepcopy(original_receipt)
                changed_budget = copy.deepcopy(budget)
                if attack == "flow":
                    request["recipe"]["simulation"]["flow_strength"] *= 1.01
                elif attack == "events":
                    request["events"][0]["fraction"] += 0.01
                elif attack == "mass":
                    changed_budget["initial_mass"][0] *= 1.01
                else:
                    request["layout"]["config"]["setup"] = "active-pools"
                write(case / "mass-budget.json", changed_budget)
                self.fixture.refresh(case, request, receipt)
                with self.assertRaises(ValueError):
                    self.build()
                shutil.rmtree(self.output)

    def test_rehashed_claims_and_page_cannot_replace_verified_records(self):
        self.build()
        data = read(self.output / "comparison.json")
        data["rows"][-1]["settings"]["caution"] = "Guaranteed beautiful"
        write(self.output / "comparison.json", data)
        proof = read(self.output / "publication.json")
        proof["artifacts"]["comparison.json"] = artifact(self.output / "comparison.json")
        proof["comparison_sha256"] = hashlib.sha256(encoded(data)).hexdigest()
        write(self.output / "publication.json", proof)
        with self.assertRaisesRegex(ValueError, "source records"):
            gallery.verify_review(self.output)

    def test_no_publication_inside_immutable_archives_and_no_unknown_visual_picks(self):
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            gallery.build_review(self.cases[0] / "review", self.cases, self.fixture.reference)
        self.assertFalse((self.cases[0] / "review").exists())
        with self.assertRaisesRegex(ValueError, "Unknown or repeated"):
            self.build(picks=[{"seed": "0x0", "variant": "rc1", "note": "Unknown"}])


if __name__ == "__main__":
    unittest.main()
