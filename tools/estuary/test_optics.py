"""Numerical material invariants and optional real GPU/reference agreement.

Run ``ESTUARY_TEST_GPU=1 ... -m unittest tools.estuary.test_optics`` on the EGL
render host to exercise the exact production shader as well as the CPU model.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

from tools.estuary.optics import (
    REFLECTANCE_FLOOR,
    Material,
    absorption_over_scattering,
    linear_to_srgb,
    reflectance,
    shader_source,
    srgb_to_linear,
)


class ColorSpaceTests(unittest.TestCase):
    def test_color_round_trip_including_transfer_break(self) -> None:
        encoded = np.array([0, 1e-8, 0.04045, 0.1, 0.5, 1.0])
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(encoded)), encoded, atol=3e-8)

    def test_color_helpers_preserve_array_shape(self) -> None:
        self.assertEqual(srgb_to_linear(np.zeros((2, 4, 3))).shape, (2, 4, 3))
        self.assertEqual(linear_to_srgb(0.0).shape, ())

    def test_invalid_colors_rejected(self) -> None:
        for value in (-1e-5, 1.00001, np.nan, np.inf):
            for function in (srgb_to_linear, linear_to_srgb, absorption_over_scattering):
                with (
                    self.subTest(value=value, function=function.__name__),
                    self.assertRaises(ValueError),
                ):
                    function(value)

    def test_absorption_white_and_ideal_black_are_finite(self) -> None:
        result = absorption_over_scattering([1.0, 0.0, 0.5])
        self.assertEqual(result[0], 0.0)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(result[2], 0.25)


class MaterialTests(unittest.TestCase):
    def test_empty_layer_is_substrate_exactly(self) -> None:
        material = Material()
        np.testing.assert_array_equal(
            reflectance([0, 0, 0], material), srgb_to_linear(material.substrate_srgb)
        )

    def test_zero_optical_scale_is_substrate(self) -> None:
        material = Material(layer_scale=0)
        expected = np.broadcast_to(srgb_to_linear(material.substrate_srgb), (8, 3))
        np.testing.assert_array_equal(reflectance(np.ones((8, 3)), material), expected)

    def test_thick_single_pigment_converges_to_authored_color(self) -> None:
        material = Material()
        pure = np.eye(3) * 1e4
        np.testing.assert_allclose(
            reflectance(pure, material), srgb_to_linear(material.pigments_srgb), atol=1e-12
        )

    def test_perfect_white_scattering_limit_and_black_limit(self) -> None:
        material = Material(
            pigments_srgb=((1, 1, 1), (0, 0, 0), (0.5, 0.5, 0.5)),
            substrate_srgb=(0, 0, 0),
            scattering=(1, 1, 1),
            layer_scale=1,
        )
        np.testing.assert_allclose(reflectance([3, 0, 0], material), [0.75] * 3, atol=1e-15)
        np.testing.assert_allclose(
            reflectance([0, 1e4, 0], material), [REFLECTANCE_FLOOR] * 3, atol=1e-15
        )

    def test_white_layer_on_white_substrate_remains_white(self) -> None:
        material = Material(pigments_srgb=((1, 1, 1),) * 3, substrate_srgb=(1, 1, 1))
        concentrations = np.zeros((80, 3))
        concentrations[:, 0] = np.geomspace(1e-12, 1e6, 80)
        np.testing.assert_allclose(
            reflectance(concentrations, material), np.ones((80, 3)), atol=2e-12
        )

    def test_single_pigment_monotonically_obscures_substrate(self) -> None:
        material = Material(substrate_srgb=(0.98, 0.98, 0.98))
        concentrations = np.zeros((120, 3))
        concentrations[:, 0] = np.geomspace(1e-12, 1e3, 120)
        result = reflectance(concentrations, material)
        self.assertTrue(np.all(np.diff(result, axis=0) <= 1e-14))

    def test_opaque_mixture_is_bounded_by_constituent_reflectances(self) -> None:
        generator = np.random.default_rng(183)
        material = Material()
        pigments = srgb_to_linear(material.pigments_srgb)
        result = reflectance(generator.uniform(0.1, 20, (400, 3)), material)
        self.assertTrue(np.all(result >= pigments.min(axis=0) - 1e-12))
        self.assertTrue(np.all(result <= pigments.max(axis=0) + 1e-12))

    def test_pigment_reordering_does_not_change_optics(self) -> None:
        material = Material()
        permutation = [2, 0, 1]
        reordered = replace(
            material,
            pigments_srgb=tuple(material.pigments_srgb[i] for i in permutation),
            scattering=tuple(material.scattering[i] for i in permutation),
        )
        density = np.array([[0.25, 0.8, 0.03], [2, 0.05, 0.7]])
        np.testing.assert_allclose(
            reflectance(density, material),
            reflectance(density[:, permutation], reordered),
            atol=1e-15,
        )

    def test_identical_pigments_depend_only_on_total_scattering(self) -> None:
        material = Material(pigments_srgb=((0.3, 0.6, 0.8),) * 3, scattering=(1, 1, 1))
        np.testing.assert_allclose(
            reflectance([0.2, 0.4, 0.7], material), reflectance([1.3, 0, 0], material), atol=1e-15
        )

    def test_extreme_concentrations_remain_finite_and_bounded(self) -> None:
        generator = np.random.default_rng(144)
        density = 10 ** generator.uniform(-300, 300, size=(500, 3))
        result = reflectance(density)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_saturated_concentrations_have_explicit_identical_cap(self) -> None:
        np.testing.assert_array_equal(reflectance([1e300, 2, 3]), reflectance([1e6, 2, 3]))

    def test_invalid_concentrations_are_rejected(self) -> None:
        for density in (1, [], [1, 2], [1, 2, 3, 4], [-1, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.subTest(density=density), self.assertRaises(ValueError):
                reflectance(density)

    def test_invalid_material_settings_are_rejected(self) -> None:
        invalid = (
            {"pigments_srgb": ((0, 0, 0),)},
            {"substrate_srgb": (0, 0, 1.1)},
            {"scattering": (0, 1, 1)},
            {"layer_scale": -1},
            {"layer_scale": np.inf},
            {"grain": 0.081},
            {"grain_frequency": (1000,)},
            {"grain_frequency": (1000, np.nan)},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Material(**changes)

    def test_uniforms_decode_colors_and_respect_texture_binding(self) -> None:
        material = Material()
        uniforms = material.uniforms(texture_unit=3)
        self.assertEqual(uniforms["u_paint"], 3)
        np.testing.assert_array_equal(
            uniforms["u_pigment_0"], srgb_to_linear(material.pigments_srgb[0])
        )
        np.testing.assert_array_equal(
            uniforms["u_substrate"], srgb_to_linear(material.substrate_srgb)
        )
        for value in (-1, 1.2, True):
            with self.subTest(texture_unit=value), self.assertRaises(ValueError):
                material.uniforms(value)


@unittest.skipUnless(
    os.environ.get("ESTUARY_TEST_GPU") == "1", "set ESTUARY_TEST_GPU=1 on the EGL host"
)
class GPUAgreementTests(unittest.TestCase):
    def test_production_shader_agrees_with_reference(self) -> None:
        import moderngl

        context = moderngl.create_standalone_context(require=330, backend="egl")
        vertex = """#version 330 core
        out vec2 v_uv;
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            v_uv = p;
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        }
        """
        generator = np.random.default_rng(196)
        density = np.zeros((16, 16, 4), dtype="f4")
        density[..., :3] = 10 ** generator.uniform(-8, 3, size=(16, 16, 3))
        density[0, :4, :3] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        texture = context.texture((16, 16), 4, density.tobytes(), dtype="f4")
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        target = context.texture((16, 16), 4, dtype="f4")
        framebuffer = context.framebuffer(color_attachments=[target])
        program = context.program(vertex_shader=vertex, fragment_shader=shader_source())
        vertex_array = context.vertex_array(program, [])
        try:
            for material in (
                Material(grain=0),
                Material(grain=0, pigments_srgb=((1, 1, 1), (0, 0, 0), (0.4, 0.3, 0.1))),
                Material(grain=0, layer_scale=0),
            ):
                for name, value in material.uniforms().items():
                    program[name].value = value
                texture.use(0)
                framebuffer.use()
                vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
                result = np.frombuffer(
                    framebuffer.read(components=4, dtype="f4"), dtype=np.float32
                ).reshape(16, 16, 4)
                np.testing.assert_allclose(
                    result[..., :3], reflectance(density[..., :3], material), atol=8e-6, rtol=2e-5
                )
                np.testing.assert_array_equal(result[..., 3], np.ones((16, 16)))
        finally:
            vertex_array.release()
            program.release()
            framebuffer.release()
            target.release()
            texture.release()
            context.release()


if __name__ == "__main__":
    unittest.main()
