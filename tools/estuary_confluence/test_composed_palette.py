"""Composed color, finite optical identity, and deterministic layer allocation."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import srgb_to_linear

from . import composed_palette as composed
from .backgrounds import generate_background
from .palette import _linear_to_oklab, generate_palette, validate_palette
from .spectral import build_spectral_material, layer_spectra, reflectance

SEEDS = (
    "0xb7f327f9f722",
    "0xbc53af1cd380",
    "0x808861c25b6c",
    "0x5b26dc8faef2",
    "0xb08c219a243f",
    "0x6210d183445b5dd5",
    "0xa0c78ebadfb75018",
    "0x5584badfb5d82790",
    "0x2d01093da35729b3",
    "0xceddf97909f39cc2",
)


class ComposedPaletteTests(unittest.TestCase):
    def test_full_seed_determinism_canonical_aliases_and_no_shared_mutation(self):
        original = generate_palette(17, 5, mode="composed")
        self.assertEqual(original, generate_palette("0X00011", 5, mode="composed"))
        self.assertEqual(original["version"], composed.VERSION)
        self.assertEqual(original["physical_version"], composed.PHYSICAL_VERSION)
        high = generate_palette(17 + (1 << 255), 5, mode="composed")
        for key in ("pigments_srgb", "layer_fractions", "scattering", "substrate_seed"):
            self.assertNotEqual(original[key], high[key])
        original["pigments_srgb"][0][0] = -1
        original["layer_fractions"][0] = -1
        pristine = generate_palette(17, 5, mode="composed")
        self.assertGreaterEqual(pristine["pigments_srgb"][0][0], 0)
        self.assertGreater(pristine["layer_fractions"][0], 0)
        for count in (True, 0, 4, 3.0):
            with self.assertRaises(ValueError):
                generate_palette(0, count, mode="composed")

    def test_three_five_prefix_includes_material_roles_layers_and_background(self):
        for seed in SEEDS:
            a, b = (generate_palette(seed, count, mode="composed") for count in (3, 5))
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
                "layer_fractions",
            ):
                self.assertEqual(a[key], [*b[key][:3], b[key][-1]])
            for key in (
                "family",
                "version",
                "physical_version",
                "quality",
                "color_generation",
                "layer_allocation",
                "generator_attempt",
                "substrate_seed",
                "body_weights",
            ):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["quality"]["evaluated_chromatic_count"], 5)
            self.assertEqual(
                generate_background("palette-night", a)["ground_srgb"],
                generate_background("palette-night", b)["ground_srgb"],
            )

    def test_color_roles_have_meaningful_value_chroma_and_scattering_hierarchy(self):
        for seed in SEEDS:
            palette = generate_palette(seed, 5, mode="composed")
            self.assertEqual(palette["pigment_roles"], [*composed.ROLES, "chalk"])
            lab = _linear_to_oklab(srgb_to_linear(palette["pigments_srgb"][:5]))
            chroma = np.linalg.norm(lab[:, 1:], axis=1)
            self.assertGreater(lab[1, 0] - lab[3, 0], 0.39)
            self.assertLess(chroma[4], 0.046)
            self.assertLess(chroma[4], chroma[0])
            self.assertLess(chroma[4], chroma[2])
            self.assertGreater(palette["scattering"][1], 2 * palette["scattering"][0])
            self.assertGreater(palette["scattering"][4], 4 * palette["scattering"][3])
            self.assertGreater(palette["layer_fractions"][0], 0.6)
            self.assertLess(palette["layer_fractions"][1], 0.31)
            self.assertLess(palette["layer_fractions"][-1], 0.09)
            self.assertTrue(np.all(np.asarray(palette["layer_fractions"]) > 0))
            self.assertTrue(np.all(np.asarray(palette["layer_fractions"]) < 1))
            np.testing.assert_allclose(np.sum(palette["body_mixtures"], axis=1), 1, atol=2e-16)

    def test_finite_spectral_paint_has_contrast_on_night_and_thickness_variation(self):
        for seed in SEEDS:
            palette = generate_palette(seed, 5, mode="composed")
            material = build_spectral_material(palette)
            pure = np.eye(6)[:5]
            samples = reflectance(pure * 0.18, material, layer_scale=12)
            thin = reflectance(pure * 0.045, material, layer_scale=12)
            thick = reflectance(pure * 0.7, material, layer_scale=12)
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue(np.all((samples >= 0) & (samples <= 1)))
            pigment_linear = srgb_to_linear(palette["pigments_srgb"][:5])
            self.assertGreater(float(pigment_linear.min()), 1e-5)
            opaque = reflectance(pure * 100, material, layer_scale=12)
            np.testing.assert_allclose(opaque, pigment_linear, atol=1e-10)
            lab = _linear_to_oklab(samples)
            night = _linear_to_oklab(
                np.asarray(generate_background("palette-night", palette)["ground_linear"])
            )
            self.assertGreater(lab[1, 0] - night[0], 0.53)
            self.assertGreater(np.ptp(lab[:, 0]), 0.27)
            self.assertTrue(np.all(_linear_to_oklab(thin)[:, 0] > lab[:, 0]))
            self.assertTrue(np.all(lab[:, 0] > _linear_to_oklab(thick)[:, 0]))
            self.assertGreater(np.max(np.abs(thin - thick)), 0.1)

    def test_mixture_ramps_are_passive_continuous_and_layer_order_remains_visible(self):
        palette = generate_palette(SEEDS[2], 5, mode="composed")
        material = build_spectral_material(palette)
        weights = np.linspace(0, 1, 101)
        for a, b in itertools.combinations(range(5), 2):
            density = np.zeros((len(weights), 6))
            density[:, a] = weights * 0.18
            density[:, b] = (1 - weights) * 0.18
            samples = reflectance(density, material, layer_scale=12)
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue(np.all((samples >= 0) & (samples <= 1)))
            coarse_step = np.max(np.linalg.norm(np.diff(samples[::2], axis=0), axis=1))
            fine_step = np.max(np.linalg.norm(np.diff(samples, axis=0), axis=1))
            # Dark paint can legitimately change a bright mixture quickly.
            # Refinement must resolve that slope, not preserve a numerical jump.
            self.assertLess(fine_step, coarse_step * 0.85)
        upper, lower = np.zeros(6), np.zeros(6)
        upper[0], lower[1] = 0.11, 0.12
        ordered = layer_spectra([lower, upper], material, layer_scale=12)
        reversed_layers = layer_spectra([upper, lower], material, layer_scale=12)
        self.assertGreater(np.max(np.abs(ordered - reversed_layers)), 0.1)

    def test_diverse_cohort_and_full_hue_circle_without_equal_chroma_requirement(self):
        signatures, relationships, sectors = set(), set(), set()
        for seed in range(320):
            palette = generate_palette(seed, 5, mode="composed")
            self.assertTrue(composed.quality(palette)["passes"])
            signatures.add(tuple(tuple(color) for color in palette["pigments_srgb"]))
            generation = palette["color_generation"]
            relationships.add(generation["relationship"])
            sectors.add(int(generation["base_hue_degrees"] // 30))
            offsets = dict(composed.RELATIONSHIPS)[generation["relationship"]]
            for hue, offset in zip(generation["candidate_hues_degrees"], offsets, strict=True):
                delta = (hue - generation["base_hue_degrees"] - offset + 180) % 360 - 180
                self.assertLessEqual(abs(delta), 6.00000002)
        self.assertEqual(len(signatures), 320)
        self.assertEqual(relationships, set(dict(composed.RELATIONSHIPS)))
        self.assertEqual(sectors, set(range(12)))

    def test_bounded_quality_search_is_never_silently_bypassed(self):
        with (
            patch.object(composed, "MAX_ATTEMPTS", 3),
            patch.object(composed, "quality", return_value={"passes": False}) as check,
            self.assertRaisesRegex(ValueError, "bounded material-quality"),
        ):
            generate_palette(19, mode="composed")
        self.assertEqual(check.call_count, 3)
        with self.assertRaisesRegex(ValueError, "all five"):
            composed.quality(generate_palette(19, 3, mode="composed"))

    def test_strict_regeneration_rejects_tampering_even_with_fresh_self_hash(self):
        for mode in ("curated", "harmonic", "random", "composed"):
            record = generate_palette(91, 5, mode=mode)
            self.assertEqual(validate_palette(record), record)
        original = generate_palette(91, 5, mode="composed")
        for key, replacement in (
            ("version", "composed-palette-v2"),
            ("physical_version", "composed-material-v2"),
            ("pigment_roles", ["dominant"] * 6),
            ("layer_fractions", [0.5] * 6),
            ("scattering", [0.5] * 6),
            ("quality", {"passes": True}),
        ):
            modified = copy.deepcopy(original)
            modified[key] = replacement
            modified.pop("identity_sha256")
            modified["identity_sha256"] = hashlib.sha256(
                json.dumps(modified, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "not derived"):
                validate_palette(modified)
        copy_record = validate_palette(original)
        copy_record["layer_fractions"][0] = 0
        self.assertGreater(original["layer_fractions"][0], 0)
        # Python considers True equal to 1.0; canonical JSON must not permit
        # boolean coefficients under the numeric record's unchanged identity.
        modified = copy.deepcopy(original)
        modified["substrate_srgb"][0] = True
        with self.assertRaisesRegex(ValueError, "not derived"):
            validate_palette(modified)


if __name__ == "__main__":
    unittest.main()
