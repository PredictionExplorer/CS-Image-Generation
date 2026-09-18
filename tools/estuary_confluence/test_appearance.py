"""Appearance experiments retain their physical parent and exact visual controls."""

import copy
import hashlib
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import srgb_to_linear
from tools.estuary_studio.common import artifact, encoded, read, write

from . import appearance
from .palette import generate_palette
from .run import PACKAGES, field_digest, surface_configs, validate_recipe
from .test_run import material_fields


class AppearanceSurface:
    fail = False

    def __init__(self, config, palette, *, spectral=None):
        self.config = config
        self.metadata = {"renderer": "Appearance pipeline fixture"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def render(self, fields, *, size, **camera):
        if self.fail:
            raise RuntimeError("Deliberate appearance failure")
        if any(array.flags.writeable for array in fields.values()):
            raise AssertionError("Appearance fields must be immutable")
        pixels = np.empty((size[1], size[0], 3), dtype="f4")
        pixels[...] = srgb_to_linear(self.config["ground_srgb"])
        pixels[size[1] // 4 : size[1] * 3 // 4, size[0] // 4 : size[0] * 3 // 4] = [0.2, 0.3, 0.5]
        return pixels


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.case, self.output = self.root / "physical", self.root / "appearance"
        self.case.mkdir()
        recipe = validate_recipe(
            {
                "chromatic_count": 5,
                "palette_mode": "harmonic",
                "looks": ["layered"],
                "simulation": {"resolution": [128, 96], "steps": 10},
                "surface": {"finish": "crisp"},
                "render": {
                    "resolution": [128, 96],
                    "still_resolution": [128, 96],
                    "formation_frames": 6,
                },
            }
        )
        self.parent = {
            "source": {"seed": "0xbc53af1cd380", "sha256": "1" * 64},
            "palette": generate_palette("0xbc53af1cd380", 5, mode="harmonic"),
            "recipe": recipe,
            "surface_configs": surface_configs(recipe),
        }
        fields = material_fields(128, 96, 6, 1)
        np.savez(self.case / "final.npz", **fields)
        self.physical = {
            "complete": True,
            "identity_sha256": hashlib.sha256(encoded(self.parent)).hexdigest(),
            "physical_state_sha256": field_digest(fields),
            "source_fraction": 1.0,
            "final_step": 10,
        }
        write(self.case / "request.json", self.parent)
        write(self.case / "receipt.json", self.physical)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(appearance, "verify_run", return_value=(self.parent, self.physical))
        )
        self.stack.enter_context(
            patch.object(
                appearance, "runtime_identity", return_value={name: {} for name in PACKAGES}
            )
        )
        self.stack.enter_context(patch.object(appearance, "Surface", AppearanceSurface))
        self.stack.enter_context(redirect_stdout(StringIO()))
        AppearanceSurface.fail = False

    def test_grounds_do_not_change_material_or_palette(self):
        before = (self.case / "final.npz").read_bytes()
        parent = copy.deepcopy(self.parent)
        appearance.render_study(self.case, self.output, names=["white", "palette-night", "relief"])
        request, receipt = appearance.verify_study(self.output)
        self.assertEqual(self.parent, parent)
        self.assertEqual((self.case / "final.npz").read_bytes(), before)
        self.assertEqual(request["physical_state_sha256"], self.physical["physical_state_sha256"])
        self.assertNotEqual(
            receipt["artifacts"]["white/poster.png"],
            receipt["artifacts"]["palette-night/poster.png"],
        )
        self.assertEqual(
            appearance.render_study(
                self.case, self.output, names=["white", "palette-night", "relief"]
            ),
            self.output,
        )

    def test_film_recipe_changes_only_appearance_and_its_camera(self):
        recipe = appearance.film_recipe(self.parent, "relief")
        self.assertEqual(recipe["simulation"], self.parent["recipe"]["simulation"])
        self.assertEqual(recipe["projection"], self.parent["recipe"]["projection"])
        self.assertEqual(recipe["render"]["still_tilt_degrees"], 20)
        self.assertEqual(recipe["surface"]["height_scale"], 5)
        validate_recipe(recipe)

    def test_preview_filters_linear_samples(self):
        p = np.zeros((768, 1024, 3), dtype="f4")
        p[::2] = 1
        result = appearance._preview(p)
        self.assertEqual(result.shape, (384, 512, 3))
        np.testing.assert_array_equal(result, 0.5)

    def test_immutable_inputs_invalid_choices_and_aspects_are_rejected(self):
        for output, names, size in (
            (self.case / "child", ["white"], None),
            (self.output, ["white", "white"], None),
            (self.output, ["invented"], None),
            (self.output, ["white"], [128, 128]),
        ):
            with self.subTest(names=names, size=size), self.assertRaises(ValueError):
                appearance.render_study(self.case, output, names=names, resolution=size)

    def test_failed_study_remains_incomplete_and_cannot_be_reused(self):
        AppearanceSurface.fail = True
        with self.assertRaises(RuntimeError):
            appearance.render_study(self.case, self.output, names=["white"])
        self.assertFalse(read(self.output / "receipt.json")["complete"])
        AppearanceSurface.fail = False
        with self.assertRaises(ValueError):
            appearance.render_study(self.case, self.output, names=["white"])

    def test_runtime_change_during_render_cannot_certify_a_complete_study(self):
        original = {name: {} for name in PACKAGES}
        changed = copy.deepcopy(original)
        changed["estuary_confluence"]["surface.py"] = "0" * 64
        with (
            patch.object(appearance, "runtime_identity", side_effect=[original, changed]),
            self.assertRaisesRegex(ValueError, "renderer changed"),
        ):
            appearance.render_study(self.case, self.output, names=["white"])
        self.assertFalse(read(self.output / "receipt.json")["complete"])

    def test_rehashed_background_record_cannot_change_its_certified_color(self):
        appearance.render_study(self.case, self.output, names=["charcoal"])
        path = self.output / "charcoal/background.json"
        background = read(path)
        background["ground_srgb"] = [1, 1, 1]
        write(path, background)
        receipt = read(self.output / "receipt.json")
        receipt["artifacts"]["charcoal/background.json"] = artifact(path)
        write(self.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Background record differs"):
            appearance.verify_study(self.output)

    def test_rehashed_view_controls_are_regenerated_from_the_parent(self):
        appearance.render_study(self.case, self.output, names=["charcoal"])
        request = read(self.output / "request.json")
        request["looks"][0]["camera"]["tilt_degrees"] = 25
        write(self.output / "request.json", request)
        receipt = read(self.output / "receipt.json")
        receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
        write(self.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Appearance controls"):
            appearance.verify_study(self.output)


if __name__ == "__main__":
    unittest.main()
