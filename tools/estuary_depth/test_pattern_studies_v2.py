"""Version-two optics must not reinterpret existing version-one paintings."""

import copy
import hashlib
import unittest

from tools.estuary.recipe import canonical_bytes, validate_recipe
from tools.estuary_depth import pattern_studies as v1
from tools.estuary_depth import pattern_studies_v2 as v2
from tools.estuary_depth.pattern_palette_v2 import make_palette


class PatternStudyV2Tests(unittest.TestCase):
    def test_version_one_canonical_recipes_retain_their_frozen_identities(self):
        vectors = {
            0: "09a2aee63f446a663bdbc0ca65dba785f2fb7238154ed630f2fe5db776ad6c5e",
            17: "488c39389b13ed50a076a8ec3185aa485a80587f29029dd7b8eaec90c1aee94a",
            17 + (1 << 240): "0a8fbb7443829d4f0ccd68274580d33ec2948ce8328425c7d00fab0cb408946f",
        }
        for seed, expected in vectors.items():
            digest = hashlib.sha256()
            for option in v1.OPTIONS:
                digest.update(option.encode() + b"\0")
                digest.update(canonical_bytes(v1.make_formation_recipe(seed, option)))
            with self.subTest(seed=seed):
                self.assertEqual(digest.hexdigest(), expected)

    def test_version_two_changes_only_optics_for_every_pattern_and_complete_seed(self):
        for seed in (0, 17, 17 + (1 << 240)):
            for option in v1.OPTIONS:
                with self.subTest(seed=seed, option=option):
                    original = v1.make_formation_recipe(seed, option)
                    updated = v2.make_formation_recipe(seed, option)
                    self.assertEqual(updated, validate_recipe(updated))
                    self.assertEqual(updated["optics"], make_palette(seed)["optics"])
                    self.assertNotEqual(updated["optics"], original["optics"])
                    expected = copy.deepcopy(original)
                    expected["optics"] = updated["optics"]
                    self.assertEqual(updated, expected)
                    self.assertEqual(v1.make_formation_recipe(seed, option), original)

    def test_catalog_and_camera_factories_are_explicit_unchanged_reexports(self):
        self.assertEqual(v2.VERSION, "pattern-studies-v2")
        for name in (
            "OPTIONS",
            "PAINT_RUNTIME_EXTENSIONS",
            "REFERENCE_OPTION",
            "DEFAULT_OPTION",
            "make_photo_recipe",
            "make_motion_recipe",
        ):
            self.assertIs(getattr(v2, name), getattr(v1, name))
        first = v2.make_formation_recipe(17, "folded-sash")
        self.assertEqual(first, v2.make_formation_recipe("0x0011", "folded-sash"))
        self.assertNotEqual(
            first["optics"], v2.make_formation_recipe(17 + (1 << 240), "folded-sash")["optics"]
        )
        with self.assertRaises(ValueError):
            v2.make_formation_recipe(17, "unknown-pattern")


if __name__ == "__main__":
    unittest.main()
