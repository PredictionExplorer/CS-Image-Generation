"""Matched optical studies must not silently change their material or ground."""

import copy
import unittest

from .backgrounds import generate_background
from .interaction_appearance import PRESETS, presentation
from .interaction_studies import texture_recipe
from .palette import generate_palette


class InteractionAppearanceTests(unittest.TestCase):
    def setUp(self):
        seed = "0x808861c25b6c"
        recipe = texture_recipe(seed, 3, "woven")
        palette = generate_palette(seed, 3, mode=recipe["palette_mode"])
        self.parent = {
            "recipe": recipe,
            "palette": palette,
            "background": generate_background(recipe["background"], palette),
        }

    def test_optical_comparisons_preserve_the_parent_and_ground(self):
        original = copy.deepcopy(self.parent)
        for name in PRESETS:
            with self.subTest(name=name):
                view = presentation(self.parent, name)
                self.assertEqual(view["background"], self.parent["background"])
                self.assertEqual(view["surface"]["grain_um"], 0)
                self.assertEqual(
                    view["surface"]["ground_srgb"],
                    self.parent["background"]["ground_srgb"],
                )
                self.assertEqual(self.parent, original)

    def test_raking_controls_share_camera_and_light(self):
        for prefix in ("", "raking-"):
            views = [
                presentation(self.parent, prefix + name)
                for name in ("control", "silk", "silk-grain")
            ]
            self.assertTrue(all(view["camera"] == views[0]["camera"] for view in views))
            surfaces = [
                {key: value for key, value in view["surface"].items() if key != "interaction"}
                for view in views
            ]
            self.assertTrue(all(surface == surfaces[0] for surface in surfaces))
            self.assertEqual(
                views[0]["surface"]["interaction"], {"silk_strength": 0.0, "grain_strength": 0.0}
            )

    def test_unrecorded_history_and_decorative_support_grain_are_rejected(self):
        for change in ("history", "ground", "grain"):
            parent = copy.deepcopy(self.parent)
            if change == "history":
                parent["recipe"]["simulation"].pop("interaction")
            elif change == "ground":
                parent.pop("background")
            else:
                parent["recipe"]["surface"]["grain_um"] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                presentation(parent, "silk-grain")

    def test_packing_and_control_use_matched_geometry_and_lighting_inputs(self):
        control = presentation(self.parent, "sculpted-control")
        textured = presentation(self.parent, "sculpted-packing")
        self.assertEqual(control["camera"], textured["camera"])
        self.assertEqual(
            {k: v for k, v in control["surface"].items() if k != "interaction"},
            {k: v for k, v in textured["surface"].items() if k != "interaction"},
        )
        self.assertEqual(textured["surface"]["interaction"]["packing_strength"], 1)
        parent = copy.deepcopy(self.parent)
        parent["recipe"]["simulation"]["substrate_um"] = 10
        with self.assertRaisesRegex(ValueError, "flat support"):
            presentation(parent, "packing")

    def test_film_presentation_changes_no_simulation_or_source_projection(self):
        base = texture_recipe("0x808861c25b6c", 3, "encounter", film=True)
        textured = texture_recipe(
            "0x808861c25b6c", 3, "encounter", film=True, appearance="sculpted-packing"
        )
        self.assertEqual(base["simulation"], textured["simulation"])
        self.assertEqual(base["projection"], textured["projection"])
        self.assertEqual(base["render"]["formation_frames"], textured["render"]["formation_frames"])
        self.assertEqual(textured["render"]["still_tilt_degrees"], 20)
        self.assertEqual(textured["render"]["orbit_tilt_degrees"], 20)


if __name__ == "__main__":
    unittest.main()
