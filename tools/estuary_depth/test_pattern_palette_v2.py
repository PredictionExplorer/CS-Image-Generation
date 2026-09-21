"""Bounded absorption must work on contaminated real paint, not only pure swatches."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest

import numpy as np

from tools.estuary.optics import (
    Material,
    absorption_over_scattering,
    linear_to_srgb,
    reflectance,
    srgb_to_linear,
)
from tools.estuary.recipe import validate_recipe
from tools.estuary_depth import pattern_palette as v1
from tools.estuary_depth import pattern_palette_v2 as v2

A709 = "0xa709b7f63f68f9f92b92af3a3e8d4e578d3486296e165c7cba7d5b85dce8cb81"
# Actual float32 concentrations from the completed A709 folded-sash pilot.
# final-state.npy SHA-256:
# 63a4902966bd8188ad995e2c3809c517139a221d7208a1d83134fe0d5adc2d89
# Native bottom-up pixels: (4003,2578), (4280,1324), (2987,2840), (3835,2679).
ACTUAL_MIXTURES = np.array(
    [
        [0.02018798515200615, 0.2049727588891983, 0.4571019411087036],
        [0.010413129813969135, 0.4711613655090332, 0.19162152707576752],
        [0.4223424196243286, 0.12999096512794495, 0.09740090370178223],
        [0.0030447845347225666, 0.012013208121061325, 0.5858986377716064],
    ],
    dtype=np.float32,
)


def rendered_lab(amounts, record):
    display = linear_to_srgb(reflectance(amounts, Material(**record["optics"])))
    return display, np.array([v1.srgb_to_oklab(color.tolist()) for color in display])


class BoundedPaletteTests(unittest.TestCase):
    def test_v1_identity_and_seed_choices_remain_unchanged(self):
        pins = {
            0: "61219dfed4a85c4870cf575936b87c8c8580402db6d48081c7c502b3e2ab92f7",
            A709: "4fa71318c3545ed0c1a0a8f6ddbd9340c3b07b0f8a947d3d7238df802bac90d8",
        }
        for seed, fingerprint in pins.items():
            parent = v1.make_palette(seed)
            self.assertEqual(parent["identity_sha256"], fingerprint)
            result = v2.make_palette(seed)
            self.assertEqual(v1.make_palette(seed), parent)
            provenance = result["metadata"]["seed_stream_provenance"]
            self.assertEqual(provenance["palette_identity_sha256"], fingerprint)
            self.assertEqual(provenance["version"], v1.VERSION)
            self.assertEqual(result["seed"], parent["seed"])
            for key in ("roles", "harmony", "base_hue_degrees", "role_permutation", "background"):
                self.assertEqual(result["metadata"][key], parent["metadata"][key])
            for key in ("substrate_srgb", "grain", "grain_frequency", "layer_scale"):
                self.assertEqual(result["optics"][key], parent["optics"][key])
            self.assertEqual(
                validate_recipe({"optics": result["optics"]})["optics"], result["optics"]
            )

    def test_gamut_inset_preserves_value_and_hue_instead_of_clipping_rgb(self):
        before, after = v1.make_palette(A709), v2.make_palette(A709)
        original = before["optics"]["pigments_srgb"][0]
        changed = after["optics"]["pigments_srgb"][0]
        self.assertLess(srgb_to_linear(original).min(), 1e-9)
        self.assertGreaterEqual(srgb_to_linear(changed).min(), v2.LINEAR_REFLECTANCE_FLOOR)
        self.assertNotEqual(original[1:], changed[1:])
        for old, new in zip(before["metadata"]["oklch"], after["metadata"]["oklch"], strict=True):
            self.assertAlmostEqual(old[0], new[0], delta=1e-7)
            self.assertLess(abs((old[2] - new[2] + 180) % 360 - 180), 0.0001)
            self.assertLessEqual(new[1], old[1] + 1e-7)

    def test_actual_contaminated_magenta_and_light_folds_retain_their_color(self):
        old_rgb, old_lab = rendered_lab(ACTUAL_MIXTURES, v1.make_palette(A709))
        new_rgb, new_lab = rendered_lab(ACTUAL_MIXTURES, v2.make_palette(A709))
        # The original near-zero red reflectance erases red even in a pixel
        # containing 97.49% magenta and only .51% dark contamination.
        self.assertLess(old_rgb[3, 0], 0.01)
        self.assertGreater(new_rgb[3, 0], 0.7)
        self.assertGreater(new_lab[3, 1], 0.15)  # Red/magenta opponent coordinate.
        # An actual medium-dominant mixed strand is now distinct from its dark
        # neighborhood, and the light-dominant mixture remains visibly lighter.
        self.assertGreater(np.linalg.norm(new_lab[0] - new_lab[2]), 0.15)
        self.assertGreater(new_lab[1, 0] - new_lab[0, 0], 0.07)
        self.assertGreater(new_lab[1, 0] - old_lab[1, 0], 0.15)
        # This is not a global brightening operation: a real 65%-dark mixture
        # retains its blue-dark identity and remains below the brighter strands.
        self.assertLess(new_rgb[2, 0], new_rgb[2, 2])
        self.assertLess(new_lab[2, 0], 0.52)

    def test_tiny_contamination_has_a_bounded_absorption_response(self):
        record = v2.make_palette(A709)
        ratios = absorption_over_scattering(srgb_to_linear(record["optics"]["pigments_srgb"]))
        self.assertLessEqual(ratios.max(), (1 - 0.003) ** 2 / (2 * 0.003))
        amounts = np.array([[0, 0, 0.6], [0.0006, 0, 0.5994], [0.006, 0, 0.594]])
        display, lab = rendered_lab(amounts, record)
        self.assertTrue(np.isfinite(display).all())
        self.assertLess(np.linalg.norm(lab[1] - lab[0]), 0.015)
        self.assertLess(np.linalg.norm(lab[2] - lab[0]), 0.08)
        self.assertGreater(display[2, 0], 0.7)

    def test_large_cohort_floor_perceptual_contracts_and_finite_optics(self):
        peak_ratio = (1 - v2.LINEAR_REFLECTANCE_FLOOR) ** 2 / (2 * v2.LINEAR_REFLECTANCE_FLOOR)
        amounts = np.concatenate(
            [
                np.eye(3) * 0.6,
                np.array([[0.006, 0.594, 0], [0.024, 0, 0.576], [0.39, 0.12, 0.09]]),
                np.array([[0, 0, 0], [1e-9, 0.02, 0.001], [5, 5, 5]]),
            ]
        )
        for seed in range(4096):
            record = v2.make_palette(seed)
            colors = record["optics"]["pigments_srgb"]
            linear = srgb_to_linear(colors)
            self.assertGreaterEqual(linear.min(), v2.LINEAR_REFLECTANCE_FLOOR)
            self.assertLessEqual(linear.max(), 1)
            self.assertLessEqual(absorption_over_scattering(linear).max(), peak_ratio)
            lch = record["metadata"]["oklch"]
            for role in range(3):
                self.assertGreaterEqual(lch[role][1], v1.MIN_CHROMA[role])
            self.assertLess(lch[0][0], lch[2][0])
            self.assertLess(lch[2][0], lch[1][0])
            self.assertGreaterEqual(min(record["metadata"]["pairwise_oklab_distance"]), 0.20)
            self.assertGreaterEqual(min(record["metadata"]["pairwise_hue_separation_degrees"]), 75)
            rgb, actual = rendered_lab(amounts, record)
            self.assertTrue(np.isfinite(rgb).all())
            self.assertTrue((rgb >= 0).all() and (rgb <= 1).all())
            for a, b in ((0, 1), (0, 2), (1, 2)):
                self.assertGreater(math.dist(actual[a], actual[b]), 0.22)

    def test_full_seed_determinism_aliases_and_rehashed_provenance_tampering(self):
        record = v2.make_palette(0xA709)
        self.assertEqual(v2.make_palette("0x" + "0" * 60 + "a709"), record)
        self.assertEqual(
            len({v2.make_palette(1 << bit)["identity_sha256"] for bit in range(256)}), 256
        )
        self.assertEqual(v2.verify_palette(json.loads(json.dumps(record))), record)
        for field in ("floor", "parent", "scattering"):
            altered = copy.deepcopy(record)
            if field == "floor":
                altered["metadata"]["gamut_mapping"]["linear_reflectance_floor"] = 0.001
            elif field == "parent":
                altered["metadata"]["seed_stream_provenance"]["palette_identity_sha256"] = "0" * 64
            else:
                altered["optics"]["scattering"][0] = 0.3
            altered.pop("identity_sha256")
            altered["identity_sha256"] = hashlib.sha256(
                json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "seed contract"):
                v2.verify_palette(altered)


if __name__ == "__main__":
    unittest.main()
