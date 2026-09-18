"""Seed identity, curated variation, and actual mixture-quality contracts."""

from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import Material, reflectance, srgb_to_linear

from . import procedural_palette as procedural
from .palette import (
    VERSION,
    _linear_to_oklab,
    _oklab_to_linear,
    generate_palette,
    mixture_reflectance,
    normalize_seed,
    palette_quality,
)


class PaletteTests(unittest.TestCase):
    def test_canonical_numeric_seed_aliases(self):
        aliases = ["0xBc53AF1cD380", "0X0000bc53af1cd380", "bc53af1cd380", int("bc53af1cd380", 16)]
        expected = generate_palette(aliases[0])
        for seed in aliases[1:]:
            self.assertEqual(generate_palette(seed), expected)
        self.assertEqual(normalize_seed("0x0000"), "0x0")
        self.assertEqual(normalize_seed((1 << 256) - 1), "0x" + "f" * 64)

    def test_seed_rejects_invalid_or_oversized_values(self):
        for seed in (True, -1, 1 << 256, 1.0, None, "0x", "0x-1", "hello", " 0x1", "0x" + "f" * 65):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                normalize_seed(seed)
        for count in (True, 4, 0, 6, "3", 3.0):
            with self.assertRaises(ValueError):
                generate_palette("0x1", count)

    def test_high_seed_bits_affect_actual_colors(self):
        first = generate_palette(19)
        second = generate_palette(19 + (1 << 255))
        self.assertNotEqual(first["pigments_srgb"], second["pigments_srgb"])
        self.assertNotEqual(first["substrate_seed"], second["substrate_seed"])
        self.assertNotEqual(first["identity_sha256"], second["identity_sha256"])

    def test_substrate_entropy_survives_json_as_canonical_full_width_hex(self):
        palette = generate_palette("0xbc53af1cd380")
        metadata = json.loads(json.dumps(palette))
        token = metadata["substrate_seed"]
        self.assertIsInstance(token, str)
        self.assertRegex(token, r"^0x[0-9a-f]{64}$")
        master = hashlib.sha256(
            VERSION.encode() + b"\0" + int(palette["seed"], 16).to_bytes(32, "big")
        ).digest()
        original_entropy = hashlib.sha256(master + b"\0substrate").digest()
        self.assertEqual(int(token, 16), int.from_bytes(original_entropy, "big"))
        self.assertGreater(int(token, 16), 2**53)

    def test_count_preserves_three_pigments_chalk_and_shared_choices(self):
        for seed in (0, 15, "0xbc53af1cd380", "0x808861c25b6c", "0xb7f327f9f722"):
            a, b = generate_palette(seed, 3), generate_palette(seed, 5)
            for key in (
                "pigments_srgb",
                "scattering",
                "settling",
                "release",
                "specific_volumes",
                "granulation",
                "pigment_names",
                "pigment_roles",
                "pigment_ids",
            ):
                self.assertEqual(a[key][:3], b[key][:3])
                self.assertEqual(a[key][-1], b[key][-1])
            for key in (
                "version",
                "seed",
                "family",
                "substrate_srgb",
                "substrate_seed",
                "generator_attempt",
                "quality",
                "body_weights",
                "underpaint_index",
            ):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["chalk_index"], 3)
            self.assertEqual(b["chalk_index"], 5)
            for row_a, row_b in zip(a["body_mixtures"], b["body_mixtures"], strict=True):
                self.assertEqual(row_a[-1], row_b[-1])
                self.assertGreaterEqual(sum(row_b[3:5]), 0.10)
                self.assertLessEqual(sum(row_b[3:5]), 0.145)

    def test_complete_json_identity_and_no_mutable_shared_state(self):
        first = generate_palette("0x123")
        second = json.loads(json.dumps(first, allow_nan=False))
        digest = second.pop("identity_sha256")
        encoded = json.dumps(
            second, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())
        self.assertEqual(first["version"], VERSION)
        first["pigments_srgb"][0][0] = -1
        self.assertEqual(generate_palette("0x123")["pigments_srgb"], second["pigments_srgb"])

    def test_three_existing_seeds_naturally_have_distinct_families(self):
        seeds = ["0xbc53af1cd380", "0x808861c25b6c", "0xb7f327f9f722"]
        self.assertEqual(
            [generate_palette(seed)["family"] for seed in seeds],
            ["mineral-tide", "red-earth", "violet-estuary"],
        )

    def test_collection_has_valid_mixtures_no_duplicate_colors_and_normalized_body_loads(self):
        signatures, families = set(), set()
        for seed in range(320):
            palette = generate_palette(seed, 3 if seed % 2 else 5)
            signatures.add(tuple(tuple(row) for row in palette["pigments_srgb"]))
            families.add(palette["family"])
            self.assertTrue(palette_quality(palette)["passes"])
            rows = np.asarray(palette["body_mixtures"])
            self.assertTrue(np.all(rows >= 0))
            np.testing.assert_allclose(rows.sum(axis=1), 1, atol=2e-16)
            for body in range(3):
                self.assertEqual(int(np.argmax(rows[body])), body)
        self.assertEqual(len(signatures), 320)
        self.assertEqual(len(families), 6)

    def test_oklab_roundtrip_and_achromatic_axes(self):
        colors = np.random.default_rng(19).uniform(0, 1, (100, 3))
        np.testing.assert_allclose(_oklab_to_linear(_linear_to_oklab(colors)), colors, atol=3e-7)
        np.testing.assert_allclose(
            _linear_to_oklab(np.array([1.0, 1.0, 1.0])), [1, 0, 0], atol=4e-8
        )

    def test_generalized_optics_matches_original_and_empty_substrate(self):
        material = Material()
        density = np.random.default_rng(18).uniform(0, 4, (12, 3))
        observed = mixture_reflectance(
            density,
            material.pigments_srgb,
            material.scattering,
            material.substrate_srgb,
            material.layer_scale,
        )
        np.testing.assert_allclose(observed, reflectance(density, material), atol=1e-15)
        palette = generate_palette(20, 5)
        blank = mixture_reflectance(
            np.zeros(6), palette["pigments_srgb"], palette["scattering"], palette["substrate_srgb"]
        )
        np.testing.assert_array_equal(blank, srgb_to_linear(palette["substrate_srgb"]))

    def test_mixture_evaluates_white_limit_without_invalid_operations(self):
        with np.errstate(all="raise"):
            color = mixture_reflectance([0.5], [[1, 1, 1]], [18], [0.2, 0.3, 0.4])
        self.assertTrue(np.all(color > 0.8))
        for density in ([-1], [float("nan")], [[1, 2]]):
            with self.assertRaises(ValueError):
                mixture_reflectance(density, [[1, 1, 1]], [18], [0.2, 0.3, 0.4])


class ProceduralPaletteTests(unittest.TestCase):
    def test_released_curated_defaults_remain_byte_identical(self):
        # Identity hashes cover every resolved field, including the old quality
        # values and physical arrays. Explicit curated mode adds no new fields.
        fixtures = {
            ("0x0", 3): "af7f173e92859c9096be99fc40a4913493e3122db4b7c21f45ba1f5a4e638791",
            ("0x0", 5): "d853ad2ea9cdc4b43b89c79d32a3424ec1bbd46fd53b7540b46044b01a38add8",
            (
                "0xbc53af1cd380",
                3,
            ): "a45566f5f3c86d04a9405e5ed5a1fe2f1b6bad56aef1c88361968580d197919b",
            (
                "0xbc53af1cd380",
                5,
            ): "0113b3f7020db5b809396588308322333f7d56729d485e93f8003a62c79bd50a",
            (
                "0x808861c25b6c",
                3,
            ): "7373b90009207fa66870aa5502188af704c8376f33882ac454fcad3885f448f9",
            (
                "0x808861c25b6c",
                5,
            ): "8a1a12ccbf917fed4ec29fc66c5df25553e7495ec9992a1adcfd83e9f5e6e078",
            (
                "0xb7f327f9f722",
                3,
            ): "dbea0e71c97730cf01a141a1e11cfc66a2b0811684ee128cce2dfda74bf9f7f0",
            (
                "0xb7f327f9f722",
                5,
            ): "86b0d6de06c190d5b8c34abe3b828ca8cbf36e4f09dccccbc59f0bd53d73d9a6",
            (
                "0x" + "f" * 64,
                3,
            ): "9e66e04cdf6bb4570b52147bda46d7de5cb4af768b05b5ff22e1430717132eb0",
            (
                "0x" + "f" * 64,
                5,
            ): "17d75dc18f4bf5346c00785bc8e36a544983ad38947ff1bb592ca52544f4745f",
        }
        for (seed, count), expected in fixtures.items():
            implicit = generate_palette(seed, count)
            explicit = generate_palette(seed, count, mode="curated")
            self.assertEqual(json.dumps(implicit), json.dumps(explicit))
            self.assertEqual(implicit["identity_sha256"], expected)

    def test_procedural_modes_never_consult_curated_color_anchors(self):
        expected = {mode: generate_palette(231, 5, mode=mode) for mode in ("harmonic", "random")}
        with patch("tools.estuary_confluence.palette.FAMILIES", ()):
            for mode in expected:
                self.assertEqual(generate_palette(231, 5, mode=mode), expected[mode])

    def test_invalid_modes_and_counts_are_rejected(self):
        for mode in (None, True, "rainbow", [], "Harmonic"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                generate_palette(0, mode=mode)
        for mode in ("harmonic", "random"):
            for count in (True, 0, 4, 3.0):
                with self.assertRaises(ValueError):
                    generate_palette(0, count, mode=mode)

    def test_new_modes_archive_version_and_full_seed_entropy(self):
        for mode in ("harmonic", "random"):
            expected = generate_palette(17, 5, mode=mode)
            self.assertEqual(expected, generate_palette("0X00011", 5, mode=mode))
            self.assertEqual(expected["version"], procedural.VERSION)
            self.assertNotEqual(expected["version"], VERSION)
            self.assertEqual(expected["physical_version"], VERSION)
            self.assertEqual(expected["mode"], mode)
            high = generate_palette(17 + (1 << 255), 5, mode=mode)
            self.assertNotEqual(high["pigments_srgb"], expected["pigments_srgb"])
            saved = json.loads(json.dumps(expected, allow_nan=False))
            identity = saved.pop("identity_sha256")
            self.assertEqual(
                identity,
                hashlib.sha256(
                    json.dumps(
                        saved, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest(),
            )
            self.assertRegex(saved["substrate_seed"], r"^0x[0-9a-f]{64}$")

    def test_optical_modes_share_identical_material_behavior_and_white_ground(self):
        for seed in (0, 81, "0xbc53af1cd380", "0x808861c25b6c", "0xb7f327f9f722"):
            for count in (3, 5):
                palettes = [
                    generate_palette(seed, count, mode=mode)
                    for mode in ("curated", "harmonic", "random")
                ]
                for key in (
                    "scattering",
                    "settling",
                    "release",
                    "specific_volumes",
                    "granulation",
                    "body_mixtures",
                    "body_weights",
                    "substrate_seed",
                ):
                    self.assertEqual(palettes[0][key], palettes[1][key])
                    self.assertEqual(palettes[1][key], palettes[2][key])
                for palette in palettes[1:]:
                    self.assertEqual(palette["substrate_srgb"], [1.0, 1.0, 1.0])
                    self.assertEqual(palette["pigments_srgb"][-1], [0.975, 0.975, 0.975])
                    self.assertEqual(palette["pigment_roles"][:-1], ["chromatic"] * count)

    def test_three_and_five_share_exact_first_colors_and_selection_decision(self):
        for seed in range(64):
            for mode in ("harmonic", "random"):
                a, b = (generate_palette(seed, count, mode=mode) for count in (3, 5))
                for key in (
                    "pigments_srgb",
                    "pigment_names",
                    "pigment_roles",
                    "pigment_ids",
                    "scattering",
                    "settling",
                    "release",
                    "specific_volumes",
                    "granulation",
                ):
                    self.assertEqual(a[key][:3], b[key][:3])
                    self.assertEqual(a[key][-1], b[key][-1])
                for key in (
                    "family",
                    "version",
                    "mode",
                    "quality",
                    "color_generation",
                    "generator_attempt",
                    "body_weights",
                    "substrate_seed",
                ):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(a["quality"]["evaluated_chromatic_count"], 5)

    def test_full_hue_coverage_distinct_paint_colors_and_neighboring_seed_variation(self):
        for mode in ("harmonic", "random"):
            signatures, hue_sectors, attempts = set(), set(), []
            for seed in range(320):
                palette = generate_palette(seed, 5, mode=mode)
                signatures.add(tuple(tuple(color) for color in palette["pigments_srgb"]))
                hue_sectors.add(int(palette["color_generation"]["base_hue_degrees"] // 30))
                attempts.append(palette["generator_attempt"])
                self.assertTrue(procedural.quality(palette)["passes"])
                self.assertEqual(len(set(tuple(color) for color in palette["pigments_srgb"])), 6)
            self.assertEqual(len(signatures), 320)
            self.assertEqual(hue_sectors, set(range(12)))
            self.assertLess(max(attempts), procedural.MAX_ATTEMPTS)

    def test_harmonic_angles_follow_relationships_around_the_seed_selected_base(self):
        relationships = dict(procedural.HARMONIES)
        for seed in range(50):
            palette = generate_palette(seed, 5, mode="harmonic")
            generation = palette["color_generation"]
            expected_offsets = relationships[generation["relationship"]]
            base = generation["base_hue_degrees"]
            for index, hue in enumerate(generation["candidate_hues_degrees"]):
                error = (hue - base - expected_offsets[index] + 180) % 360 - 180
                self.assertLessEqual(abs(error), 7.00000002)

    def test_failed_quality_search_is_bounded_and_never_silently_bypassed(self):
        with (
            patch.object(procedural, "MAX_ATTEMPTS", 3),
            patch.object(procedural, "quality", return_value={"passes": False}) as check,
            self.assertRaisesRegex(ValueError, "bounded material-quality"),
        ):
            generate_palette(19, mode="harmonic")
        self.assertEqual(check.call_count, 3)


if __name__ == "__main__":
    unittest.main()
