"""Recipe identity, strict parsing, resource bounds and source-clock contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .recipe import DEFAULTS, canonical_bytes, read_recipe, recipe_sha256, validate_recipe


class RecipeTests(unittest.TestCase):
    def test_resolved_defaults_are_independent_and_round_trip(self):
        a, b = validate_recipe({}), validate_recipe({})
        self.assertEqual(a, b)
        self.assertEqual(validate_recipe(a), a)
        self.assertEqual(json.loads(canonical_bytes(a)), a)
        a["simulation"]["pigment_weights"][0] = 9
        self.assertEqual(b["simulation"]["pigment_weights"][0], 1)
        self.assertEqual(DEFAULTS["simulation"]["pigment_weights"][0], 1)

    def test_partial_recipe_has_same_identity_as_explicit_recipe(self):
        short = {"simulation": {"flow_strength": 0.2}, "optics": {"grain": 0}}
        resolved = validate_recipe(short)
        self.assertEqual(recipe_sha256(short), recipe_sha256(resolved))
        reordered = dict(reversed(list(resolved.items())))
        self.assertEqual(canonical_bytes(reordered), canonical_bytes(resolved))
        self.assertNotEqual(recipe_sha256(short), recipe_sha256({}))

    def test_unknown_fields_are_never_silently_ignored(self):
        for value in [
            {"simluation": {}},
            {"simulation": {"step": 3600}},
            {"optics": {"color": [1, 1, 1]}},
            {"render": {"frame": 30}},
        ]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unknown"):
                validate_recipe(value)

    def test_rejects_nonfinite_boolean_and_string_controls(self):
        for value in [True, "1", float("nan"), float("inf"), 10**400]:
            for group, key in [
                ("simulation", "flow_strength"),
                ("optics", "grain"),
                ("projection", "fill"),
            ]:
                with self.subTest(value=value, key=key), self.assertRaises(ValueError):
                    validate_recipe({group: {key: value}})
        for value in [True, 1.0, "1", 2]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recipe({"schema_version": value})

    def test_divisible_source_clock_and_matching_aspect_are_required(self):
        for value in [
            {"simulation": {"steps": 3601}},
            {"render": {"frames": 302}},
            {"render": {"resolution": [1600, 900]}},
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recipe(value)
        accepted = validate_recipe(
            {
                "simulation": {"resolution": [2048, 1536]},
                "render": {"resolution": [3840, 2880], "frames": 601},
            }
        )
        self.assertEqual(accepted["simulation"]["steps"] // (accepted["render"]["frames"] - 1), 6)

    def test_resource_bounds_and_even_video_dimensions(self):
        for value in [
            {"simulation": {"steps": 40_001}},
            {"simulation": {"resolution": [8192, 6144]}},
            {"render": {"resolution": [8192, 6144]}},
            {"render": {"resolution": [1601, 1200]}},
            {"render": {"frames": 1802}},
            {"render": {"fps": 61}},
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recipe(value)

    def test_optics_and_motion_ranges(self):
        for value in [
            {"simulation": {"stir_radius": 0.019}},
            {"simulation": {"domain_scale": 0.99}},
            {"simulation": {"domain_scale": 2.51}},
            {"simulation": {"brush_radius": 0.001}},
            {"simulation": {"pigment_weights": [1, 2]}},
            {"optics": {"pigments_srgb": [[1, 1, 1]] * 2}},
            {"optics": {"pigments_srgb": [[1, 1, 1], [1, 1, 1], [1, -1, 1]]}},
            {"optics": {"grain_frequency": [0, 1400]}},
            {"optics": {"scattering": [1, 1, True]}},
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recipe(value)

    def test_guard_domain_is_explicit_and_does_not_change_source_projection(self):
        original = validate_recipe({})
        guarded = validate_recipe({"simulation": {"domain_scale": 1.6}})
        self.assertEqual(original["simulation"]["domain_scale"], 1.0)
        self.assertEqual(guarded["simulation"]["domain_scale"], 1.6)
        self.assertEqual(original["projection"], guarded["projection"])
        self.assertEqual(original["render"], guarded["render"])
        self.assertNotEqual(recipe_sha256(original), recipe_sha256(guarded))

    def test_authored_initial_material_pattern_is_explicit(self):
        original = validate_recipe({})
        layered = validate_recipe({"simulation": {"initial_pattern": "strata"}})
        self.assertEqual(original["simulation"]["initial_pattern"], "pools")
        self.assertEqual(layered["simulation"]["initial_pattern"], "strata")
        self.assertEqual(layered["simulation"]["initial_load"], 0)
        self.assertNotEqual(recipe_sha256(original), recipe_sha256(layered))
        for invalid in ["swirls", "", True, None, 1, ["strata"]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_recipe({"simulation": {"initial_pattern": invalid}})

    def test_authored_carrier_is_bounded_and_defaults_to_still_water(self):
        self.assertEqual(validate_recipe({})["simulation"]["carrier_velocity"], [0, 0])
        current = validate_recipe({"simulation": {"carrier_velocity": [-8, 8]}})
        self.assertEqual(current["simulation"]["carrier_velocity"], [-8, 8])
        self.assertNotEqual(recipe_sha256(current), recipe_sha256({}))
        for invalid in [[-8.01, 0], [0, 8.01], [0], [0, 0, 0], [True, 0], [float("nan"), 0]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_recipe({"simulation": {"carrier_velocity": invalid}})

    def test_guarded_native_master_has_separate_simulation_pixel_budget(self):
        master = validate_recipe(
            {
                "simulation": {"resolution": [6144, 4608], "domain_scale": 1.6},
                "render": {"resolution": [3840, 2880]},
            }
        )
        self.assertEqual(master["simulation"]["resolution"], [6144, 4608])
        for invalid in [
            {"simulation": {"resolution": [8192, 6144]}},
            {"simulation": {"resolution": [8194, 96]}},
            {"render": {"resolution": [6144, 4608]}},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_recipe(invalid)

    def test_temporal_exposure_uses_only_available_canonical_states(self):
        self.assertEqual(validate_recipe({})["render"]["temporal_samples"], 1)
        production = validate_recipe(
            {
                "simulation": {"steps": 7200},
                "render": {"frames": 901, "temporal_samples": 4},
            }
        )
        self.assertEqual(production["render"]["temporal_samples"], 4)
        for invalid in [0, 9, True, 1.5, "4"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_recipe({"render": {"temporal_samples": invalid}})
        with self.assertRaisesRegex(ValueError, "canonical steps between frames"):
            validate_recipe(
                {"simulation": {"steps": 3600}, "render": {"frames": 1801, "temporal_samples": 3}}
            )
        self.assertNotEqual(recipe_sha256({}), recipe_sha256({"render": {"temporal_samples": 2}}))

    def test_file_parsing_rejects_duplicates_nonfinite_and_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            for text in [
                '{"schema_version":1,"schema_version":1}',
                '{"simulation":{"fade":NaN}}',
                " " * 65537,
                "[]",
            ]:
                path.write_text(text)
                with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                    read_recipe(path)
            path.write_text('{"projection":{"rotation_degrees":30}}')
            self.assertEqual(read_recipe(path)["projection"]["rotation_degrees"], 30)

    def test_input_dict_is_not_mutated(self):
        value = {"optics": {"substrate_srgb": [0.2, 0.3, 0.4]}}
        before = copy.deepcopy(value)
        validate_recipe(value)
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
