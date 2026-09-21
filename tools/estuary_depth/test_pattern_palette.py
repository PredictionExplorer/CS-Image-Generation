"""Perceptual, deterministic and finite-layer contracts for seeded pigments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import Material, linear_to_srgb, reflectance
from tools.estuary.recipe import validate_recipe
from tools.estuary_depth import pattern_palette as palette


class PaletteTests(unittest.TestCase):
    def test_published_primary_coordinates_and_rgb_round_trip(self):
        # Independent published OKLab/sRGB primary reference values, not values
        # computed by the generator under test.
        references = (
            ((1, 0, 0), (0.62795536, 0.22486306, 0.12584630)),
            ((0, 1, 0), (0.86643961, -0.23388757, 0.17949848)),
            ((0, 0, 1), (0.45201372, -0.03245698, -0.31152815)),
            ((1, 1, 1), (1, 0, 0)),
            ((0, 0, 0), (0, 0, 0)),
        )
        for rgb, expected in references:
            lab = palette.srgb_to_oklab(rgb)
            np.testing.assert_allclose(lab, expected, rtol=0, atol=4e-8)
            np.testing.assert_allclose(palette.oklab_to_linear_srgb(lab), rgb, rtol=0, atol=3e-7)

    def test_gamut_mapping_reduces_chroma_without_rotating_or_changing_value(self):
        for L in (0.34, 0.60, 0.86):
            for hue in range(0, 360, 7):
                color = palette.gamut_map_oklch(L, 0.4, hue)
                self.assertTrue(all(0 <= x <= 1 for x in color))
                actual_L, a, b = palette.srgb_to_oklab(color)
                self.assertAlmostEqual(actual_L, L, delta=1e-7)
                actual_hue = math.degrees(math.atan2(b, a))
                self.assertLess(abs((actual_hue - hue + 180) % 360 - 180), 0.0001)
                self.assertLess(math.hypot(a, b), 0.4)
        for value in (True, float("nan"), float("inf"), 10**1000):
            with self.subTest(value=repr(value)[:20]), self.assertRaises(ValueError):
                palette.gamut_map_oklch(value, 0.1, 30)

    def test_all_256_seed_bits_and_numeric_aliases_are_reproducible(self):
        record = palette.make_palette(0xABC)
        for seed in ("abc", "0xABC", "0X0aBc", "0x" + "0" * 61 + "abc"):
            self.assertEqual(palette.make_palette(seed), record)
        self.assertEqual(len(record["seed"]), 66)
        baseline = palette.make_palette(0)
        identities = {palette.make_palette(1 << bit)["identity_sha256"] for bit in range(256)}
        self.assertEqual(len(identities), 256)
        self.assertNotIn(baseline["identity_sha256"], identities)
        for seed in (True, -1, 1 << 256, 1.0, "", "0x", "0x-1", "0x" + "0" * 65, " 123"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                palette.make_palette(seed)

    def test_no_ambient_rng_or_call_order_and_no_shared_mutable_record(self):
        expected = palette.make_palette("0x123")
        with (
            patch("random.random", side_effect=AssertionError("ambient RNG")),
            patch("numpy.random.default_rng", side_effect=AssertionError("ambient RNG")),
        ):
            for seed in (101, 0, 1 << 255, 512):
                palette.make_palette(seed)
            self.assertEqual(palette.make_palette("0x123"), expected)
        altered = palette.make_palette("0x123")
        altered["optics"]["pigments_srgb"][0][0] = 0
        altered["metadata"]["roles"].clear()
        self.assertEqual(palette.make_palette("0x123"), expected)

    def test_optics_use_existing_recipe_and_material_contracts(self):
        for seed in (0, 1, 511, 1 << 255):
            record = palette.make_palette(seed)
            recipe = validate_recipe({"optics": record["optics"]})
            self.assertEqual(recipe["optics"], record["optics"])
            material = Material(**record["optics"])
            np.testing.assert_allclose(
                reflectance([[0, 0, 0]], material),
                reflectance([[0, 0, 0]], Material(**recipe["optics"])),
                rtol=0,
                atol=0,
            )
            S = record["optics"]["scattering"]
            self.assertTrue(S[0] < S[2] < S[1])
            self.assertLess(max(S) / min(S), 2)

    def test_portable_record_rejects_rehashed_optical_and_metadata_changes(self):
        record = palette.make_palette(9881)
        self.assertEqual(palette.verify_palette(json.loads(json.dumps(record))), record)
        for area in ("optics", "metadata", "seed"):
            altered = copy.deepcopy(record)
            if area == "optics":
                altered["optics"]["scattering"][1] *= 1.01
            elif area == "metadata":
                altered["metadata"]["pairwise_hue_separation_degrees"][0] += 1
            else:
                altered["seed"] = "0x" + "0" * 64
            altered.pop("identity_sha256")
            altered["identity_sha256"] = hashlib.sha256(
                json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            with self.subTest(area=area), self.assertRaisesRegex(ValueError, "seed contract"):
                palette.verify_palette(altered)

    def test_large_cohort_separation_variety_and_actual_finite_layer_appearance(self):
        bins = [set() for _ in range(3)]
        backgrounds, harmonies, permutations = [], set(), set()
        # Every color is checked after the existing KM finite-layer optics,
        # including the actual substrate and S coefficients, at initial load .6.
        amounts = np.concatenate(
            [np.eye(3) * 0.6, np.array([[0.3, 0.3, 0], [0.3, 0, 0.3], [0, 0.3, 0.3]])]
        )
        for seed in range(4096):
            record = palette.make_palette(seed)
            optics, meta = record["optics"], record["metadata"]
            labs = np.array([palette.srgb_to_oklab(color) for color in optics["pigments_srgb"]])
            self.assertTrue(labs[0, 0] < labs[2, 0] < labs[1, 0])
            self.assertGreaterEqual(min(meta["pairwise_oklab_distance"]), palette.MIN_DISTANCE)
            self.assertGreaterEqual(min(meta["pairwise_hue_separation_degrees"]), 75)
            for role, (_L, C, hue) in enumerate(meta["oklch"]):
                self.assertGreaterEqual(C, palette.MIN_CHROMA[role])
                bins[role].add(int(hue // 30))
            actual_rgb = linear_to_srgb(reflectance(amounts, Material(**optics)))
            actual = np.array([palette.srgb_to_oklab(color.tolist()) for color in actual_rgb])
            for a, b in ((0, 1), (0, 2), (1, 2)):
                self.assertGreater(np.linalg.norm(actual[a] - actual[b]), 0.22)
            self.assertGreater(np.ptp(actual[3:, 0]), 0.09)
            self.assertGreater(np.linalg.norm(actual[3:, 1:], axis=1).max(), 0.065)
            harmonies.add(meta["harmony"])
            permutations.add(tuple(meta["role_permutation"]))
            backgrounds.append(meta["background"]["kind"])
        self.assertEqual(bins, [set(range(12))] * 3)
        self.assertEqual(len(harmonies), 2)
        self.assertEqual(len(permutations), 6)
        dark_fraction = backgrounds.count("dark") / len(backgrounds)
        self.assertGreater(dark_fraction, 0.85)
        self.assertLess(dark_fraction, 0.90)


if __name__ == "__main__":
    unittest.main()
