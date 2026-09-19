"""Numerical contracts for real phase order and bounded incomplete mixing."""

import copy
import unittest

import numpy as np

from tools.estuary.optics import srgb_to_linear

from .optics import (
    finite_layer_rt,
    glazed_density_scale,
    layer_rt,
    layered_reflectance,
    reflectance,
)


def palette(count=4):
    colors = [[0.035, 0.15, 0.42], [0.10, 0.48, 0.38], [0.66, 0.17, 0.09]]
    if count == 6:
        colors += [[0.44, 0.18, 0.54], [0.85, 0.56, 0.13]]
    return {
        "pigments_srgb": [*colors, [0.94, 0.91, 0.83]],
        "scattering": [0.4] * (count - 1) + [8.0],
        "substrate_srgb": [0.87, 0.84, 0.77],
        "chalk_index": count - 1,
    }


class OpticalContracts(unittest.TestCase):
    def test_glazed_mass_preserves_interior_amount_and_bounds_only_extremes(self):
        mass = np.array([0, 0.001, 0.06, 0.15, 0.3, 100.0])
        scale = glazed_density_scale(mass)
        np.testing.assert_allclose(mass * scale, [0, 0.0525, 0.06, 0.15, 0.3, 0.375])
        self.assertTrue(np.isfinite(scale).all())
        self.assertEqual(scale[0], 0)
        for bad in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                glazed_density_scale(bad)
        for config in ({"mass_reference": True}, {"min_mass_ratio": 0}, {"max_mass_ratio": 9}):
            with self.assertRaises(ValueError):
                glazed_density_scale(0.1, **config)

    def test_glazed_common_scale_preserves_real_layers_and_visible_order(self):
        lower = np.array([0.09, 0, 0, 0])
        upper = np.array([0, 0.04, 0.012, 0])
        scale = glazed_density_scale((lower + upper).sum())
        p, empty = palette(), np.zeros(4)
        ordered = layered_reflectance(lower * scale, empty, upper * scale, p, layer_scale=12)
        swapped = layered_reflectance(upper * scale, empty, lower * scale, p, layer_scale=12)
        self.assertGreater(float(np.max(np.abs(ordered - swapped))), 0.015)
        np.testing.assert_allclose(
            lower * scale / (lower + upper).sum(), lower / (lower + upper).sum()
        )

    def test_empty_stack_and_zero_optical_scale_return_substrate(self):
        p = palette()
        empty = np.zeros((5, 4))
        expected = np.broadcast_to(srgb_to_linear(p["substrate_srgb"]), (5, 3))
        for mixed in (0, 0.4, 1):
            np.testing.assert_array_equal(reflectance(empty, p, mixedness=mixed), expected)
            np.testing.assert_array_equal(
                layered_reflectance(empty, empty, empty, p, mixedness=mixed), expected
            )
            np.testing.assert_allclose(
                reflectance(np.ones_like(empty), p, layer_scale=0, mixedness=mixed),
                expected,
                atol=1e-15,
            )

    def test_coefficients_are_passive_and_stack_is_energy_bounded(self):
        rng = np.random.default_rng(52)
        for count in (4, 6):
            p = palette(count)
            phases = [10.0 ** rng.uniform(-9, 4, (91, count)) for _ in range(3)]
            mixing = rng.random(91)
            r, t = layer_rt(phases[0], p, mixedness=mixing)
            self.assertTrue(np.all(r >= 0) and np.all(t >= 0))
            self.assertTrue(np.all(r + t <= 1 + 1e-12))
            image = layered_reflectance(*phases, p, mixedness=mixing)
            self.assertTrue(np.isfinite(image).all())
            self.assertTrue(np.all(image >= 0) and np.all(image <= 1))

    def test_channel_permutation_changes_neither_mixture_nor_layered_image(self):
        rng = np.random.default_rng(39)
        for count in (4, 6):
            p = palette(count)
            order = rng.permutation(count)
            permuted = copy.deepcopy(p)
            permuted["pigments_srgb"] = np.asarray(p["pigments_srgb"])[order].tolist()
            permuted["scattering"] = np.asarray(p["scattering"])[order].tolist()
            phases = [rng.random((23, count)) * 0.13 for _ in range(3)]
            for mixed in (0, 0.35, 1):
                expected = layered_reflectance(*phases, p, mixedness=mixed)
                actual = layered_reflectance(
                    *(x[..., order] for x in phases), permuted, mixedness=mixed
                )
                np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)

    def test_layer_order_is_visible_and_homogeneous_total_is_order_invariant(self):
        p = palette()
        red = np.array([0, 0, 0.12, 0])
        chalk = np.array([0, 0, 0, 0.07])
        zero = np.zeros(4)
        a = layered_reflectance(zero, red, chalk, p)
        b = layered_reflectance(zero, chalk, red, p)
        self.assertGreater(float(np.max(np.abs(a - b))), 0.08)
        np.testing.assert_array_equal(reflectance(red + chalk, p), reflectance(chalk + red, p))

    def test_identical_intimate_layers_equal_their_single_combined_layer(self):
        p = palette()
        mixture = np.array([0.08, 0.04, 0.02, 0.005])
        stacked = layered_reflectance(mixture * 0.2, mixture * 0.3, mixture * 0.5, p)
        np.testing.assert_allclose(stacked, reflectance(mixture, p), atol=2e-14)

    def test_areal_endpoint_uses_mass_preserving_single_pigment_columns(self):
        p = palette()
        density = np.array([0.1, 0.2, 0.04, 0.01])
        r, t = layer_rt(density, p, mixedness=0)
        mass = density.sum()
        expected_r = np.zeros(3)
        expected_t = np.zeros(3)
        for index, amount in enumerate(density):
            pure = np.zeros(4)
            pure[index] = mass
            pr, pt = layer_rt(pure, p, mixedness=1)
            expected_r += amount / mass * pr
            expected_t += amount / mass * pt
        np.testing.assert_allclose(r, expected_r, atol=1e-14)
        np.testing.assert_allclose(t, expected_t, atol=1e-14)
        ir, it = layer_rt(density, p, mixedness=1)
        mr, mt = layer_rt(density, p, mixedness=0.3)
        np.testing.assert_allclose(mr, 0.3 * ir + 0.7 * r, atol=1e-14)
        np.testing.assert_allclose(mt, 0.3 * it + 0.7 * t, atol=1e-14)

    def test_single_pigment_and_thick_limits(self):
        p = palette()
        for i in range(4):
            density = np.zeros(4)
            density[i] = 1e6
            expected = srgb_to_linear(p["pigments_srgb"][i])
            np.testing.assert_allclose(reflectance(density, p), expected, atol=2e-14)
            np.testing.assert_allclose(reflectance(density, p, mixedness=0), expected, atol=2e-14)
        r, t = finite_layer_rt(0, np.array([0, 1e-10, 1, 1e9]))
        np.testing.assert_allclose(r + t, 1, atol=1e-15)
        self.assertEqual(float(r[0]), 0)
        self.assertEqual(float(t[0]), 1)

    def test_rejects_invalid_shapes_nonfinite_negative_and_unbounded_inputs(self):
        p = palette()
        for density in ([1, 2, 3], [0, 0, -0.1, 0], [0, 0, float("nan"), 0], [0, 0, 1e9, 0]):
            with self.assertRaises(ValueError):
                reflectance(density, p)
        for mixed in (-0.1, 1.1, float("nan"), [0.1, 0.2]):
            with self.assertRaises(ValueError):
                reflectance(np.ones(4), p, mixedness=mixed)
        for scale in (True, -1, float("inf")):
            with self.assertRaises(ValueError):
                reflectance(np.ones(4), p, layer_scale=scale)
        with self.assertRaises(ValueError):
            layered_reflectance(np.ones((2, 4)), np.ones(4), np.ones(4), p)
