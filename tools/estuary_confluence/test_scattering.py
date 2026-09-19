"""Energy, reciprocity, boundary limits and GPU parity of white EON scattering."""

from __future__ import annotations

import os
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from .scattering import aggregate_response, white_eon


def direction(cosine, azimuth=0.0):
    cosine, azimuth = np.broadcast_arrays(cosine, azimuth)
    sine = np.sqrt(np.maximum(0, 1 - cosine * cosine))
    return np.stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine), axis=-1)


class RoughDiffuseTests(unittest.TestCase):
    def test_aggregate_response_is_bounded_monotone_symmetric_and_exact_at_identity(self):
        g = np.linspace(0, 1, 4097)
        np.testing.assert_array_equal(aggregate_response(g), g)
        for contrast in (1, 1.5, 4, 8):
            response = aggregate_response(g, contrast)
            self.assertTrue(np.isfinite(response).all())
            self.assertTrue(np.all(np.diff(response) >= 0))
            np.testing.assert_array_equal(response[[0, 2048, -1]], [0, 0.5, 1])
            np.testing.assert_allclose(response + response[::-1], 1, atol=2e-16)
            slope = (
                aggregate_response(0.5 + 1e-7, contrast) - aggregate_response(0.5 - 1e-7, contrast)
            ) / 2e-7
            self.assertAlmostEqual(slope, contrast, places=6)
        for g, contrast in (
            (-0.1, 4),
            (1.1, 4),
            (float("nan"), 4),
            (0.5, 0.9),
            (0.5, 8.1),
            (0.5, float("inf")),
        ):
            with self.subTest(g=g, contrast=contrast), self.assertRaises(ValueError):
                aggregate_response(g, contrast)

    def test_reciprocity_nonnegative_and_finite_through_grazing(self):
        rng = np.random.default_rng(394)
        a = direction(rng.uniform(0, 1, 3000), rng.uniform(-np.pi, np.pi, 3000))
        b = direction(rng.uniform(0, 1, 3000), rng.uniform(-np.pi, np.pi, 3000))
        r = rng.uniform(0, 1, 3000)
        actual = white_eon(a, b, r)
        np.testing.assert_array_equal(actual, white_eon(b, a, r))
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0)
        for cosine in (0.0, 1e-12, 1e-8, 1e-4, 0.2, 1.0):
            for other in (0.0, 1e-12, 1e-4, 0.5, 1.0):
                value = white_eon(direction(cosine), direction(other, 0.9), 1)
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, 0)

    def test_smooth_limit_is_exact_and_converges_without_an_epsilon_seam(self):
        a, b = direction(0.1, 1.7), direction(0.35, -2.1)
        self.assertEqual(white_eon(a, b, 0), 1.0)
        for cosine in (0, 1e-12, 0.1, 1):
            self.assertEqual(white_eon(direction(cosine), direction(cosine), 0), 1.0)
        errors = [abs(white_eon(a, b, r) - 1) for r in (1e-2, 1e-4, 1e-6, 1e-8)]
        self.assertTrue(all(a > b for a, b in pairwise(errors)))
        self.assertLess(errors[-1], 2e-8)

    @staticmethod
    def energy_error(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        cosine = ((nodes + 1) / 2)[:, None]
        azimuth = (np.arange(order * 2) + 0.5) * np.pi / order
        incoming = direction(cosine, azimuth[None, :])
        worst = 0.0
        for outgoing_cosine in (1, 0.9, 0.5, 0.1, 0.01, 0.001):
            for roughness in (0, 0.01, 0.1, 0.5, 1):
                kernel = white_eon(incoming, direction(outgoing_cosine), roughness)
                # pi cancels the BRDF denominator and projected hemisphere measure.
                energy = np.sum(kernel.mean(-1) * cosine[:, 0] * weights)
                worst = max(worst, abs(energy - 1))
        return worst

    def test_projected_hemisphere_energy_is_one_and_quadrature_converges(self):
        coarse = self.energy_error(32)
        fine = self.energy_error(256)
        self.assertLess(fine, 2e-5)
        self.assertLess(fine, coarse / 20)

    def test_response_is_angular_and_preserves_rotational_symmetry(self):
        light = direction(np.sin(np.deg2rad(23)), 0)
        forward, away = (
            direction(np.cos(np.deg2rad(24)), 0),
            direction(np.cos(np.deg2rad(24)), np.pi),
        )
        self.assertGreater(white_eon(light, forward, 1), 1.2)
        self.assertLess(white_eon(light, away, 1), 0.7)
        rotated_light = direction(np.sin(np.deg2rad(23)), 1.7)
        rotated_view = direction(np.cos(np.deg2rad(24)), 1.7)
        self.assertAlmostEqual(
            white_eon(light, forward, 1), white_eon(rotated_light, rotated_view, 1), places=14
        )

    def test_invalid_inputs_fail_instead_of_clipping_material_controls(self):
        for bad in (-1, 1.01, float("nan"), float("inf")):
            with self.subTest(roughness=bad), self.assertRaises(ValueError):
                white_eon(direction(1), direction(1), bad)
        for bad in ((0, 0, -1), (0, 0, 2), (0, 0), (1e200, 0, 1), (0, float("nan"), 1)):
            with self.subTest(direction=bad), self.assertRaises(ValueError):
                white_eon(bad, direction(1), 0.5)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class RoughDiffuseHardwareTests(unittest.TestCase):
    def test_gpu_matches_reference_for_angles_strengths_and_zero_limit(self):
        import moderngl

        rng = np.random.default_rng(714)
        size = 4096
        cosines = np.concatenate(
            (np.geomspace(1e-8, 1e-2, size // 2), rng.uniform(0.01, 1, size // 2))
        )
        incoming = direction(cosines, rng.uniform(-np.pi, np.pi, size)).astype("f4")
        outgoing = direction(cosines[::-1], rng.uniform(-np.pi, np.pi, size)).astype("f4")
        roughness = rng.uniform(0, 1, size).astype("f4")
        roughness[::7] = 0
        roughness[1:3] = (1, 0.5)
        contrast = rng.uniform(1, 8, size).astype("f4")
        contrast[::11] = 1
        packed = np.zeros((size * 2, 4), "f4")
        packed[0::2, :3], packed[1::2, :3] = incoming, outgoing
        packed[0::2, 3] = roughness
        packed[1::2, 3] = contrast
        expected = white_eon(incoming, outgoing, roughness)
        ctx = moderngl.create_standalone_context(require=430, backend="egl")
        resources = []
        try:
            with ctx:
                self.assertFalse(
                    any(
                        name in ctx.info["GL_RENDERER"].lower()
                        for name in ("llvmpipe", "softpipe", "software")
                    )
                )
                shader = (Path(__file__).parent / "shaders/scattering.glsl").read_text()
                program = ctx.compute_shader(
                    "#version 430 core\n"
                    + shader
                    + """
                    layout(local_size_x=64) in;
                    layout(std430,binding=0) readonly buffer Input {vec4 samples[];};
                    layout(std430,binding=1) writeonly buffer Output {vec2 values[];};
                    void main(){
                        uint i=gl_GlobalInvocationID.x;
                        if(i>=values.length())return;
                        vec4 a=samples[2*i],b=samples[2*i+1];
                        values[i]=vec2(white_eon(a.z,b.z,dot(a.xy,b.xy),a.w),
                                       aggregate_response(a.w,b.w));
                    }
                    """
                )
                resources.append(program)
                inputs, outputs = ctx.buffer(packed.tobytes()), ctx.buffer(reserve=size * 8)
                resources.extend((inputs, outputs))
                inputs.bind_to_storage_buffer(0)
                outputs.bind_to_storage_buffer(1)
                program.run((size + 63) // 64)
                ctx.memory_barrier()
                actual = np.frombuffer(outputs.read(), dtype="f4").reshape(size, 2)
                np.testing.assert_allclose(actual[:, 0], expected, atol=5e-7, rtol=3e-6)
                np.testing.assert_array_equal(actual[::7, 0], np.ones_like(actual[::7, 0]))
                np.testing.assert_allclose(
                    actual[:, 1], aggregate_response(roughness, contrast), atol=2e-7, rtol=3e-6
                )
                np.testing.assert_array_equal(actual[::11, 1], roughness[::11])
        finally:
            with ctx:
                for resource in reversed(resources):
                    resource.release()
            ctx.release()


if __name__ == "__main__":
    unittest.main()
