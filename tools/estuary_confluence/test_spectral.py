"""Passive spectral transport, seeded-palette fidelity and native GPU parity."""

from __future__ import annotations

import copy
import hashlib
import itertools
import os
import unittest
from pathlib import Path

import numpy as np

from tools.estuary.optics import srgb_to_linear

from . import spectral
from .palette import generate_palette
from .surface import Surface
from .test_optics import palette
from .test_surface import fields


class SpectralContracts(unittest.TestCase):
    def test_material_records_pin_license_data_and_exact_upstream_revision(self):
        material = spectral.build_spectral_material(palette())
        self.assertEqual(
            material["provenance"]["upstream_commit"], "bb2b05c9d1e65ae824d47e3b1cc17ea32c8ee68f"
        )
        self.assertEqual(material["provenance"]["license"], "MIT")
        license_path = Path(spectral.__file__).parent / material["provenance"]["license_file"]
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes()).hexdigest(),
            material["provenance"]["license_sha256"],
        )
        self.assertIn("Copyright (c) 2025 Ronald van Wijnen", license_path.read_text())
        self.assertIn("not measured", material["provenance"]["reflectance"])
        self.assertEqual(material["wavelengths_nm"], list(range(380, 751, 10)))
        self.assertEqual(spectral.validate_spectral_material(material, palette()), material)
        material["wavelengths_nm"][0] = 0
        self.assertEqual(spectral.build_spectral_material(palette())["wavelengths_nm"][0], 380)

    def test_pure_color_reconstruction_is_passive_and_colorimetrically_faithful(self):
        primaries = np.asarray(
            [*itertools.product([0.0, 1.0], repeat=3), [0.5, 0.5, 0.5], [0.08, 0.63, 0.27]]
        )
        curves = spectral.reconstruction(primaries)
        self.assertEqual(curves.shape, (10, 38))
        self.assertTrue(np.all(curves >= 1e-6) and np.all(curves <= 1))
        np.testing.assert_allclose(
            spectral.to_linear_rgb(curves), srgb_to_linear(primaries), atol=1.01e-6, rtol=0
        )
        np.testing.assert_array_equal(spectral.to_linear_rgb(np.ones(38)), np.ones(3))

    def test_zero_pigment_and_zero_optical_thickness_return_the_ground(self):
        p = palette()
        m = spectral.build_spectral_material(p)
        ground = srgb_to_linear(p["substrate_srgb"])
        for mixed in (0, 0.4, 1):
            np.testing.assert_allclose(
                spectral.reflectance(np.zeros(4), m, mixedness=mixed), ground, atol=2e-14
            )
            np.testing.assert_allclose(
                spectral.reflectance(np.ones(4), m, layer_scale=0, mixedness=mixed),
                ground,
                atol=2e-14,
            )
        p["pigments_srgb"] = [[1, 1, 1]] * 4
        p["substrate_srgb"] = [1, 1, 1]
        m = spectral.build_spectral_material(p)
        np.testing.assert_allclose(
            spectral.reflectance([0.1, 0.2, 0.3, 0.4], m), np.ones(3), atol=2e-14
        )

    def test_every_spectral_layer_is_passive_across_extreme_concentrations(self):
        rng = np.random.default_rng(419)
        for count in (4, 6):
            m = spectral.build_spectral_material(palette(count))
            density = 10.0 ** rng.uniform(-10, 5, (83, count))
            mixed = rng.random(83)
            r, t = spectral.layer_rt(density, m)
            self.assertTrue(np.all(r >= 0) and np.all(t >= 0))
            self.assertTrue(np.all(r + t <= 1 + 1e-12))
            spectra = spectral.layer_spectra([density, density * 0.3], m, mixedness=mixed)
            self.assertTrue(
                np.isfinite(spectra).all() and np.all(spectra >= 0) and np.all(spectra <= 1)
            )
            rgb = spectral.to_linear_rgb(spectra, m)
            self.assertTrue(np.isfinite(rgb).all() and np.all(rgb >= 0) and np.all(rgb <= 1))

    def test_independent_endpoint_composes_each_pure_column_before_averaging(self):
        m = spectral.build_spectral_material(palette())
        density = np.array([0.012, 0.007, 0.0025, 0.002])
        mass = density.sum()
        fractions = density / mass
        bottom = np.asarray(m["substrate_reflectance"])
        expected = np.zeros(38)
        old_r, old_t = np.zeros(38), np.zeros(38)
        for index, fraction in enumerate(fractions):
            pure = np.zeros(4)
            pure[index] = mass
            r, t = spectral.layer_rt(pure, m)
            expected += fraction * spectral.add_layer(bottom, r, t)
            old_r += fraction * r
            old_t += fraction * t
        actual = spectral.independent_layer_reflectance(density, bottom, m, mixedness=0)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        # This finite layer must not silently return the old averaged-R/T slab.
        self.assertGreater(
            float(np.max(np.abs(actual - spectral.add_layer(bottom, old_r, old_t)))), 0.001
        )
        intimate = spectral.independent_layer_reflectance(density, bottom, m, mixedness=1)
        partial = spectral.independent_layer_reflectance(density, bottom, m, mixedness=0.4)
        np.testing.assert_allclose(partial, 0.4 * intimate + 0.6 * actual, atol=1e-14)

    def test_partial_mixing_does_not_change_single_pigments_or_empty_layers(self):
        m = spectral.build_spectral_material(palette())
        bottom = np.linspace(0.1, 0.9, 38)
        for index in range(4):
            density = np.zeros(4)
            density[index] = 0.03
            expected = spectral.independent_layer_reflectance(density, bottom, m, mixedness=1)
            for mixedness in (0, 0.4, 1):
                np.testing.assert_allclose(
                    spectral.independent_layer_reflectance(density, bottom, m, mixedness=mixedness),
                    expected,
                    atol=1e-14,
                )
                np.testing.assert_array_equal(
                    spectral.independent_layer_reflectance(
                        np.zeros(4), bottom, m, mixedness=mixedness
                    ),
                    bottom,
                )

    def test_v2_record_identifies_the_corrected_optical_mixing_model(self):
        material = spectral.build_spectral_material(palette())
        self.assertEqual(material["version"], "confluence-spectral-v2")
        self.assertIn("before area averaging", material["provenance"]["partial_mixing"])

    def test_channel_permutation_and_intimate_layer_partition_invariance(self):
        rng = np.random.default_rng(38)
        for count in (4, 6):
            p = palette(count)
            m = spectral.build_spectral_material(p)
            density = rng.uniform(0, 0.2, (11, count))
            order = rng.permutation(count)
            reordered = copy.deepcopy(p)
            for key in ("pigments_srgb", "scattering"):
                reordered[key] = np.asarray(p[key])[order].tolist()
            other = spectral.build_spectral_material(reordered)
            for mixed in (0, 0.35, 1):
                np.testing.assert_allclose(
                    spectral.reflectance(density, m, mixedness=mixed),
                    spectral.reflectance(density[..., order], other, mixedness=mixed),
                    atol=2e-14,
                )
            np.testing.assert_allclose(
                spectral.layered_reflectance(density * 0.2, density * 0.3, density * 0.5, m),
                spectral.reflectance(density, m),
                atol=2e-14,
            )

    def test_phase_order_changes_color_and_crisp_normalization_preserves_fractions(self):
        m = spectral.build_spectral_material(palette())
        red = np.array([0, 0, 0.12, 0])
        chalk = np.array([0, 0, 0, 0.07])
        zero = np.zeros(4)
        a = spectral.layered_reflectance(zero, red, chalk, m)
        b = spectral.layered_reflectance(zero, chalk, red, m)
        self.assertGreater(float(np.max(np.abs(a - b))), 0.05)
        for scale in (0.1, 1, 10):
            normalized = spectral.layered_reflectance(
                zero, red * scale, chalk * scale, m, normalized_mass=0.18
            )
            expected = spectral.layered_reflectance(zero, red, chalk, m, normalized_mass=0.18)
            np.testing.assert_allclose(normalized, expected, atol=2e-14)

    def test_defined_blue_yellow_materials_make_a_green_mixture(self):
        p = palette()
        p["pigments_srgb"][0] = [0, 33 / 255, 133 / 255]
        p["pigments_srgb"][1] = [252 / 255, 210 / 255, 0]
        p["scattering"] = [0.14, 0.7, 0.6, 6]
        m = spectral.build_spectral_material(p)
        color = spectral.reflectance([0.2, 0.8, 0, 0], m, layer_scale=100)
        self.assertGreater(color[1], color[0] + 0.1)
        self.assertGreater(color[1], color[2] + 0.1)
        from .optics import reflectance as rgb_reflectance

        rgb = rgb_reflectance([0.2, 0.8, 0, 0], p, layer_scale=100)
        self.assertGreater(float(np.max(np.abs(color - rgb))), 0.03)

    def test_seeded_palette_and_mixture_audit_across_both_modes_and_counts(self):
        for seed in range(24):
            for mode in ("harmonic", "random"):
                for count in (3, 5):
                    p = generate_palette(hex(seed), count, mode=mode)
                    material = spectral.build_spectral_material(p)
                    curves = np.asarray(material["pigment_reflectance"])
                    np.testing.assert_allclose(
                        spectral.to_linear_rgb(curves, material),
                        srgb_to_linear(p["pigments_srgb"]),
                        atol=2e-6,
                        rtol=0,
                    )
                    pairs = list(itertools.combinations(range(count), 2))
                    density = np.zeros((len(pairs), count + 1))
                    for index, pair in enumerate(pairs):
                        density[index, list(pair)] = 0.5
                    mixed = spectral.reflectance(
                        density, material, normalized_mass=0.18, layer_scale=24
                    )
                    self.assertTrue(
                        np.isfinite(mixed).all() and np.all(mixed >= 0) and np.all(mixed <= 1)
                    )

    def test_material_hash_palette_binding_and_invalid_density_rejected(self):
        p = palette()
        material = spectral.build_spectral_material(p)
        altered = copy.deepcopy(material)
        altered["pigment_reflectance"][0][0] *= 0.5
        with self.assertRaisesRegex(ValueError, "identity"):
            spectral.validate_spectral_material(altered, p)
        other = copy.deepcopy(p)
        other["pigments_srgb"][0][0] += 0.01
        with self.assertRaisesRegex(ValueError, "palette"):
            spectral.validate_spectral_material(material, other)
        for density in ([1, 2, 3], [-1, 0, 0, 0], [np.nan, 0, 0, 0]):
            with self.assertRaises(ValueError):
                spectral.reflectance(density, material)
        for value in (0, True, float("nan"), 20):
            with self.assertRaises(ValueError):
                spectral.reflectance(np.ones(4), material, normalized_mass=value)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL4.3")
class SpectralHardwareTests(unittest.TestCase):
    def make_surface(self, count=4, **config):
        p = palette(count)
        surface = Surface(
            {"optics_model": "spectral", **config}, p, spectral=spectral.build_spectral_material(p)
        )
        self.addCleanup(surface.close)
        return surface

    def test_gpu_matches_cpu_for_both_counts_phase_modes_and_optical_finishes(self):
        for count in (4, 6):
            for mode in ("homogeneous", "layered"):
                for finish in ("fresco", "crisp"):
                    surface = self.make_surface(count, mode=mode, finish=finish)
                    state = fields(count=count)
                    surface.render(state, (32, 24))
                    with surface.ctx:
                        rgba = (
                            np.frombuffer(surface._textures[0].read(), dtype="f4")
                            .reshape(24, 32, 4)
                            .copy()
                        )
                    actual = (
                        rgba[..., :3] / rgba[..., 3, None] if finish == "crisp" else rgba[..., :3]
                    )
                    options = {
                        "mixedness": state["mixing"],
                        "normalized_mass": 0.15 if finish == "crisp" else None,
                    }
                    if mode == "layered":
                        expected = spectral.layered_reflectance(
                            state["underpaint"],
                            state["deposit"],
                            state["mobile"],
                            surface.spectral,
                            **options,
                        )
                    else:
                        expected = spectral.reflectance(
                            state["pigment"], surface.spectral, **options
                        )
                    np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=8e-5)
                    surface.close()

    def test_empty_pixels_low_mass_ground_and_frozen_camera_cache(self):
        surface = self.make_surface(6, finish="crisp", grain_um=0)
        state = fields(count=6)
        for key in ("mobile", "deposit", "underpaint", "pigment"):
            state[key].fill(0)
        for tilt in (0, 25):
            np.testing.assert_array_equal(
                surface.render(state, (64, 48), tilt_degrees=tilt), np.ones((48, 64, 3), dtype="f4")
            )
        state["mobile"][..., 0] = 0.0004
        state["pigment"][:] = state["mobile"]
        np.testing.assert_array_equal(
            surface.render(state, (64, 48)), np.ones((48, 64, 3), dtype="f4")
        )
        state = fields(count=6)
        original = {k: v.copy() for k, v in state.items()}
        first = surface.render(state, (64, 48), tilt_degrees=12)
        other = self.make_surface()
        other.render(fields(), (32, 24))
        np.testing.assert_array_equal(surface.render(None, (64, 48), tilt_degrees=12), first)
        for key in state:
            np.testing.assert_array_equal(state[key], original[key])

    def test_rgb_and_spectral_use_the_same_mass_silhouette(self):
        state = fields(256, 192)
        for key in ("mobile", "deposit", "underpaint", "pigment"):
            state[key].fill(0)
        u = (np.arange(256, dtype="f4") + 0.5) / 256
        state["mobile"][..., 0] = 0.01 + 0.015 * (u[None, :] - 0.56)
        state["pigment"][:] = state["mobile"]
        config = {
            "finish": "crisp",
            "paint_mass_threshold": 0.01,
            "ambient": 1,
            "key_strength": 0,
            "fill_strength": 0,
            "grain_um": 0,
            "exposure": 1,
            "tone_map": "none",
            "mix_control": 0,
        }
        a = Surface(config, palette())
        self.addCleanup(a.close)
        b = self.make_surface(**config)
        for size in ((128, 96), (256, 192)):
            masks = []
            for surface in (a, b):
                image = surface.render(state, size, tilt_degrees=25, azimuth_degrees=0)
                masks.append((1 - image[..., 0]) / (1 - image[size[1] // 2, -10, 0]))
            np.testing.assert_allclose(masks[0], masks[1], atol=2e-5, rtol=0)

    def test_spectral_record_cannot_be_bound_to_another_palette_or_rgb_shader(self):
        p = palette()
        record = spectral.build_spectral_material(p)
        with self.assertRaisesRegex(ValueError, "RGB optics"):
            Surface({}, p, spectral=record)
        p["pigments_srgb"][0][0] += 0.1
        with self.assertRaisesRegex(ValueError, "palette"):
            Surface({"optics_model": "spectral"}, p, spectral=record)
