"""Same-state optical experiments preserve provenance and resolve exact controls."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary_studio.common import artifact, encoded, read, write

from . import appearance, blend_study
from .backgrounds import generate_background
from .palette import generate_palette
from .run import field_digest, surface_configs, validate_recipe


def parent_record():
    palette = generate_palette("0xceddf97909f39cc2", 3, mode="composed")
    background = generate_background("palette-night", palette)
    recipe = validate_recipe(
        {
            "chromatic_count": 3,
            "palette_mode": "composed",
            "looks": ["layered"],
            "simulation": {"resolution": [128, 96], "steps": 10},
            "surface": {
                "finish": "glazed",
                "optics_model": "spectral",
                "ground_srgb": background["ground_srgb"],
                "layer_scale": 24,
            },
            "render": {
                "resolution": [128, 96],
                "capture_resolution": [128, 96],
                "still_resolution": [128, 96],
                "formation_frames": 6,
            },
        }
    )
    return {
        "source": {"seed": palette["seed"], "sha256": "a" * 64},
        "palette": palette,
        "background": background,
        "spectral": None,
        "recipe": recipe,
        "surface_configs": surface_configs(recipe),
    }


class BlendPresentationTests(unittest.TestCase):
    def test_operators_are_distinct_and_keep_parent_camera_ground_and_palette(self):
        parent = parent_record()
        original = copy.deepcopy(parent)
        looks = {name: blend_study.presentation(parent, name) for name in blend_study.PRESETS}
        self.assertEqual(looks["ordered"]["surface"], parent["surface_configs"]["layered"])
        self.assertEqual(looks["intimate"]["surface"]["mode"], "homogeneous")
        self.assertEqual(looks["intimate"]["surface"]["mix_control"], 0)
        self.assertLess(
            looks["thin-glaze"]["surface"]["glaze_min_mass_ratio"],
            looks["ordered"]["surface"]["glaze_min_mass_ratio"],
        )
        for look in looks.values():
            self.assertEqual(look["background"], parent["background"])
            self.assertEqual(look["camera"], looks["ordered"]["camera"])
        self.assertEqual(parent, original)

    def test_film_recipe_selects_matching_operator_without_changing_transport_or_timing(self):
        parent = parent_record()
        for name in blend_study.PRESETS:
            with self.subTest(name=name):
                recipe = blend_study.film_recipe(parent, name)
                look = blend_study.presentation(parent, name)
                self.assertEqual(recipe["simulation"], parent["recipe"]["simulation"])
                self.assertEqual(recipe["render"], parent["recipe"]["render"])
                self.assertEqual(surface_configs(recipe)[recipe["looks"][0]], look["surface"])

    def test_ink_uses_actual_optical_mass_without_changing_silhouette_or_backing(self):
        parent = parent_record()
        original = parent["surface_configs"]["layered"]
        ink = blend_study.presentation(parent, "ink")["surface"]
        changed = {key for key in ink if ink[key] != original[key]}
        self.assertEqual(changed, {"glaze_min_mass_ratio", "layer_scale"})
        self.assertEqual(ink["glaze_min_mass_ratio"], 0.02)
        self.assertEqual(ink["layer_scale"], 4.0)
        self.assertEqual(ink["paint_mass_threshold"], original["paint_mass_threshold"])
        self.assertEqual(ink["finish"], "glazed")

    def test_older_named_ground_is_preserved_but_custom_or_changed_ground_is_rejected(self):
        parent = parent_record()
        original = blend_study.presentation(parent, "ordered")
        del parent["background"]
        self.assertEqual(blend_study.presentation(parent, "ordered"), original)
        parent["surface_configs"]["layered"]["ground_srgb"] = [0.11, 0.23, 0.34]
        with self.assertRaisesRegex(ValueError, "named ground"):
            blend_study.presentation(parent, "ordered")
        parent["background"] = original["background"]
        with self.assertRaisesRegex(ValueError, "ground differs"):
            blend_study.presentation(parent, "ordered")

    def test_unsupported_parent_and_unknown_look_are_explicit_errors(self):
        parent = parent_record()
        for name in (None, "noise", "", True):
            with self.assertRaisesRegex(ValueError, "Unknown blend"):
                blend_study.presentation(parent, name)
        parent["surface_configs"]["layered"]["optics_model"] = "rgb"
        with self.assertRaisesRegex(ValueError, "spectral glazed"):
            blend_study.presentation(parent, "ordered")


class BlendArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.case, self.output = self.root / "case", self.root / "study"
        self.case.mkdir()
        self.parent = parent_record()
        self.fields = {"pigment": np.zeros((96, 128, 4), dtype="f4")}
        self.fields["pigment"][24:72, 32:96, 0] = 0.18
        self.physical = {
            "complete": True,
            "identity_sha256": hashlib.sha256(encoded(self.parent)).hexdigest(),
            "physical_state_sha256": field_digest(self.fields),
            "source_fraction": 1.0,
            "final_step": 10,
        }
        write(self.case / "request.json", self.parent)
        write(self.case / "receipt.json", self.physical)
        np.savez(self.case / "final.npz", **self.fields)
        context = patch.object(appearance, "verify_run", return_value=(self.parent, self.physical))
        context.start()
        self.addCleanup(context.stop)

        class TestSurface:
            def __init__(self, config, palette, *, spectral):
                self.config = config
                self.metadata = {"fixture": "CPU archive-contract test"}

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                pass

            def render(self, fields, *, size, **camera):
                if any(value.flags.writeable for value in fields.values()):
                    raise AssertionError("Study exposed writable source material")
                width, height = size
                return np.full((height, width, 3), 0.25, dtype="f4")

        context = patch.object(appearance, "Surface", TestSurface)
        context.start()
        self.addCleanup(context.stop)

    def render(self):
        return blend_study.render_study(self.case, self.output, resolution=[128, 96])

    def rebind(self, request):
        write(self.output / "request.json", request)
        receipt = read(self.output / "receipt.json")
        receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
        write(self.output / "receipt.json", receipt)

    def test_portable_verified_archive_reuses_source_state_and_is_idempotent(self):
        before = (self.case / "final.npz").read_bytes()
        self.render()
        request, receipt = blend_study.verify_study(self.output)
        self.assertEqual(request["physical_state_sha256"], self.physical["physical_state_sha256"])
        self.assertEqual(request["presentations"], list(blend_study.PRESETS))
        self.assertTrue(receipt["complete"])
        self.assertEqual((self.case / "final.npz").read_bytes(), before)
        self.assertEqual(self.render(), self.output)
        self.case.rename(self.root / "moved-case")
        blend_study.verify_study(self.output)

    def test_wrong_material_identity_rejected_even_after_request_is_rehashed(self):
        self.render()
        request = read(self.output / "request.json")
        request["physical_state_sha256"] = "0" * 64
        self.rebind(request)
        with self.assertRaisesRegex(ValueError, "physical archive"):
            blend_study.verify_study(self.output)

    def test_changed_controls_rejected_even_after_request_is_rehashed(self):
        self.render()
        request = read(self.output / "request.json")
        request["looks"][1]["surface"]["mode"] = "layered"
        self.rebind(request)
        with self.assertRaisesRegex(ValueError, "controls differ"):
            blend_study.verify_study(self.output)

    def test_media_and_incomplete_archives_cannot_be_reused(self):
        self.render()
        path = self.output / "ordered/poster.png"
        path.write_bytes(path.read_bytes()[:-1] + b"x")
        with self.assertRaises(ValueError):
            blend_study.verify_study(self.output)
        receipt = read(self.output / "receipt.json")
        receipt["artifacts"]["ordered/poster.png"] = artifact(path)
        receipt["complete"] = False
        write(self.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.render()

    def test_source_changed_after_verification_is_rejected(self):
        self.fields["pigment"][0, 0, 0] = 0.2
        np.savez(self.case / "final.npz", **self.fields)
        with self.assertRaisesRegex(ValueError, "changed while loading"):
            self.render()

    def test_renderer_cannot_modify_material_silently(self):
        def mutate(fields, *, size, **camera):
            fields["pigment"].setflags(write=True)
            fields["pigment"][0, 0, 0] = 0.5
            return np.full((size[1], size[0], 3), 0.25, dtype="f4")

        with (
            patch.object(appearance.Surface, "render", side_effect=mutate),
            self.assertRaisesRegex(ValueError, "modified physical state"),
        ):
            self.render()
        self.assertFalse(read(self.output / "receipt.json")["complete"])


if __name__ == "__main__":
    unittest.main()
