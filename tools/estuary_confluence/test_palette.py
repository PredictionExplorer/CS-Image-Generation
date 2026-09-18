"""Seed identity, curated variation, and actual mixture-quality contracts."""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from tools.estuary.optics import Material, reflectance, srgb_to_linear

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


if __name__ == "__main__":
    unittest.main()
