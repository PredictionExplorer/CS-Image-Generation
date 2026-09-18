"""Reproducible restrained grounds and background-aware image balance."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

import numpy as np

from tools.estuary.optics import srgb_to_linear

from .assessment import image_balance
from .backgrounds import NAMES, VERSION, generate_background, validate_background
from .palette import _linear_to_oklab, generate_palette


class BackgroundTests(unittest.TestCase):
    @staticmethod
    def rehash(record):
        payload = {key: value for key, value in record.items() if key != "identity_sha256"}
        record["identity_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def test_named_grounds_are_fixed_with_exact_linear_encoding(self):
        a, b = generate_palette(1), generate_palette(2, 5, mode="harmonic")
        for name in NAMES[:-1]:
            first, second = generate_background(name, a), generate_background(name, b)
            self.assertEqual(first["ground_srgb"], second["ground_srgb"])
            self.assertEqual(first["ground_linear"], srgb_to_linear(first["ground_srgb"]).tolist())
            self.assertEqual(first["version"], VERSION)
        white = generate_background("white", a)
        self.assertEqual(white["ground_srgb"], [1, 1, 1])
        self.assertEqual(white["ground_linear"], [1, 1, 1])

    def test_night_is_repeatable_and_records_bind_the_actual_palette(self):
        palette = generate_palette("0xbc53af1cd380", 5, mode="harmonic")
        original = copy.deepcopy(palette)
        first = generate_background("palette-night", palette)
        self.assertEqual(first, generate_background("palette-night", palette))
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))
        payload = {key: value for key, value in first.items() if key != "identity_sha256"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        self.assertEqual(first["identity_sha256"], expected)
        self.assertEqual(palette, original)
        palette["pigments_srgb"][0] = [0.1, 0.65, 0.25]
        changed = generate_background("palette-night", palette)
        self.assertNotEqual(first["ground_srgb"], changed["ground_srgb"])
        self.assertNotEqual(first["palette_colors_sha256"], changed["palette_colors_sha256"])

    def test_full_seed_bits_influence_night_even_when_pigment_colors_match(self):
        palette = generate_palette(17, mode="harmonic")
        aliases = copy.deepcopy(palette)
        aliases["seed"] = "0X00011"
        self.assertEqual(
            generate_background("palette-night", palette),
            generate_background("palette-night", aliases),
        )
        high = copy.deepcopy(palette)
        high["seed"] = hex(17 | (1 << 255))
        self.assertNotEqual(
            generate_background("palette-night", palette)["ground_srgb"],
            generate_background("palette-night", high)["ground_srgb"],
        )

    def test_night_uses_same_principal_colors_across_three_and_five(self):
        for seed in range(20):
            a = generate_background("palette-night", generate_palette(seed, 3, mode="harmonic"))
            b = generate_background("palette-night", generate_palette(seed, 5, mode="harmonic"))
            self.assertEqual(a["ground_srgb"], b["ground_srgb"])
            self.assertEqual(a["method"], b["method"])

    def test_nights_generalize_across_seeds_with_dark_restrained_gamut_safe_colors(self):
        unique = set()
        for seed in range(320):
            palette = generate_palette(seed, 5, mode="harmonic" if seed % 2 else "random")
            background = generate_background("palette-night", palette)
            color = np.asarray(background["ground_srgb"])
            self.assertTrue(np.all((color >= 0) & (color <= 1)))
            lab = _linear_to_oklab(background["ground_linear"])
            self.assertTrue(0.15499 <= lab[0] <= 0.21501)
            self.assertLessEqual(float(np.linalg.norm(lab[1:])), 0.02501)
            self.assertLess(
                float(np.dot(background["ground_linear"], [0.2126, 0.7152, 0.0722])), 0.014
            )
            unique.add(tuple(color))
        self.assertEqual(len(unique), 320)

    def test_achromatic_principal_palette_has_a_neutral_night(self):
        palette = generate_palette(19)
        palette["pigments_srgb"][:3] = [[0.3, 0.3, 0.3]] * 3
        background = generate_background("palette-night", palette)
        self.assertTrue(background["method"]["neutral_fallback"])
        np.testing.assert_allclose(
            background["ground_srgb"], background["ground_srgb"][0], atol=1e-9
        )

    def test_invalid_names_seeds_and_palette_values_are_rejected(self):
        palette = generate_palette(19)
        for name in (None, True, "black", "WHITE"):
            with self.assertRaises(ValueError):
                generate_background(name, palette)
        for key, value in (
            ("seed", True),
            ("chromatic_count", 4),
            ("pigments_srgb", [[1, 1, float("nan")]] * 4),
        ):
            with self.assertRaises(ValueError):
                generate_background("palette-night", {**palette, key: value})

    def test_validation_preserves_archived_roundoff_values_after_exact_self_hash_check(self):
        palette = generate_palette(481, mode="harmonic")
        record = generate_background("palette-night", palette)
        record["method"]["anchor_hue_degrees"] += 5e-13
        record["ground_linear"][0] += 5e-13
        self.rehash(record)
        result = validate_background(record, palette)
        self.assertEqual(result, record)
        self.assertIsNot(result, record)

    def test_background_self_hash_tampering_and_rebound_derivation_edits_are_rejected(self):
        palette = generate_palette(18)
        record = generate_background("palette-night", palette)
        record["ground_srgb"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_background(record, palette)
        self.rehash(record)
        with self.assertRaisesRegex(ValueError, "derivation"):
            validate_background(record, palette)
        record = generate_background("palette-night", palette)
        record["method"]["lightness"] += 0.001
        self.rehash(record)
        with self.assertRaisesRegex(ValueError, "derivation"):
            validate_background(record, palette)

    def test_background_validation_binds_palette_seed_and_record_schema(self):
        palette = generate_palette(18)
        for name in NAMES:
            record = generate_background(name, palette)
            self.assertEqual(validate_background(record, palette), record)
            with self.assertRaisesRegex(ValueError, "derivation"):
                validate_background(record, generate_palette(19))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_background({**record, "extra": 1}, palette)


class GroundBalanceTests(unittest.TestCase):
    @staticmethod
    def legacy_balance(rgb):
        h, w = rgb.shape[:2]
        weight = 1 - np.einsum("...i,i->...", rgb, [0.2126, 0.7152, 0.0722])
        mass = weight.sum(dtype="f8")
        if mass <= h * w * 1e-10:
            return None
        x = (np.arange(w) + 0.5) / w * 2 - 1
        y = 1 - (np.arange(h) + 0.5) / h * 2
        return [
            round(float(weight.sum(axis=0, dtype="f8") @ x / mass), 10),
            round(float(weight.sum(axis=1, dtype="f8") @ y / mass), 10),
        ]

    def test_default_and_explicit_white_are_identical_to_original_calculation(self):
        rng = np.random.default_rng(48)
        for dtype in ("f4", "f8"):
            for _ in range(12):
                image = rng.random((24, 32, 3)).astype(dtype)
                self.assertEqual(image_balance(image), self.legacy_balance(image))
                self.assertEqual(image_balance(image, [1, 1, 1]), self.legacy_balance(image))

    def test_light_paint_is_centered_by_its_contrast_on_dark_ground(self):
        ground = generate_background("palette-night", generate_palette(41))["ground_linear"]
        image = np.empty((12, 12, 3), dtype="f4")
        image[:] = ground
        image[3:9, 8:12] = [0.8, 0.5, 0.3]
        center = image_balance(image, ground)
        self.assertAlmostEqual(center[0], 2 / 3, places=8)
        self.assertEqual(center[1], 0)
        reflected = image_balance(image[:, ::-1], ground)
        self.assertEqual(reflected[0], -center[0])

    def test_blank_dark_grounds_do_not_invent_a_center_due_to_float32_rounding(self):
        for seed in range(30):
            for name in NAMES:
                ground = generate_background(name, generate_palette(seed))["ground_linear"]
                for dtype in ("f4", "f8"):
                    image = np.empty((12, 16, 3), dtype=dtype)
                    image[:] = ground
                    self.assertIsNone(image_balance(image, ground))

    def test_dark_and_light_marks_both_contribute_on_an_intermediate_ground(self):
        image = np.full((12, 12, 3), 0.5)
        image[4:8, 1:3] = 0.1
        image[4:8, 9:11] = 0.9
        self.assertEqual(image_balance(image, [0.5, 0.5, 0.5]), [0, 0])

    def test_invalid_explicit_ground_is_rejected(self):
        image = np.ones((4, 4, 3))
        for ground in ([0, 0], [-1, 0, 0], [0, float("nan"), 0], [[0, 0, 0]]):
            with self.assertRaises(ValueError):
                image_balance(image, ground)


if __name__ == "__main__":
    unittest.main()
