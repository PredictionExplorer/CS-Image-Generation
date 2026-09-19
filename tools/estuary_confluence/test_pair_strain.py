"""Independent curl, source-response and GPU parity checks for pair strain."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.estuary.flow_reference import streamfunction, velocity

from . import participation_layout as planner
from .pair_strain import pair_strain_uniforms
from .test_participation_layout import Orbit, config


def frame():
    return SimpleNamespace(
        positions=np.array([[-0.3, 0.1], [0.3, 0.1], [0.0, -0.4]]),
        velocities=np.array([[-2.0, 0.4], [2.0, 0.4], [0.0, -0.8]]),
        pair_distances=np.array([0.6, 0.7, 0.7]),
    )


class SourceStrainTests(unittest.TestCase):
    def test_extension_direction_and_rate_are_from_the_recording(self):
        source = frame()
        result = pair_strain_uniforms(source, 0.22)
        np.testing.assert_array_equal(result[0, :2], [1, 0])
        conditioned = source.velocities / (
            1 + np.linalg.norm(source.velocities, axis=-1, keepdims=True) / 24
        )
        radial = 0.6 * (conditioned[1, 0] - conditioned[0, 0]) / (0.6**2 + 0.22**2)
        self.assertAlmostEqual(float(result[0, 2]), 20 * np.tanh(radial / 20), places=6)
        self.assertEqual(result.dtype, np.float32)
        source.velocities *= -1
        np.testing.assert_array_equal(pair_strain_uniforms(source, 0.22)[:, :2], result[:, :2])
        np.testing.assert_array_equal(pair_strain_uniforms(source, 0.22)[:, 2], -result[:, 2])

    def test_batch_matches_individual_frames_exactly(self):
        source = Orbit()
        fractions = [0, 0.13, 0.65, 1]
        actual = pair_strain_uniforms(source.sample(fractions), 0.22)
        expected = np.stack([pair_strain_uniforms(source.frame(t), 0.22) for t in fractions])
        np.testing.assert_array_equal(actual, expected)

    def test_stationary_source_and_projected_coincidence_do_not_invent_strain(self):
        source = frame()
        source.velocities.fill(0)
        np.testing.assert_array_equal(pair_strain_uniforms(source, 0.22)[:, 2], 0)
        source = frame()
        source.positions[1] = source.positions[0]
        np.testing.assert_array_equal(pair_strain_uniforms(source, 0.22)[0], 0)

    def test_real_distance_suppresses_projected_crossing_and_rate_is_bounded(self):
        source = frame()
        near = pair_strain_uniforms(source, 0.02)
        source.pair_distances *= 20
        far = pair_strain_uniforms(source, 0.02)
        self.assertLess(abs(far[0, 2]), abs(near[0, 2]) / 100)
        source.velocities *= 1e12
        source.pair_distances.fill(0)
        self.assertTrue(np.all(np.abs(pair_strain_uniforms(source, 0.02)[:, 2]) <= 20))

    def test_invalid_measurements_are_rejected(self):
        for radius in (True, 0, -1, 3, float("nan")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                pair_strain_uniforms(frame(), radius)
        for key, value in (
            ("positions", np.zeros((2, 2))),
            ("velocities", np.full((3, 2), np.nan)),
            ("pair_distances", [-1, 1, 1]),
        ):
            source = frame()
            setattr(source, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                pair_strain_uniforms(source, 0.22)

    def test_engine_rejects_unbounded_or_non_numeric_pair_strain(self):
        from .engine import validate_config

        self.assertEqual(validate_config({})["pair_strain"], 0)
        for value in (None, True, -0.01, 4.01, float("nan"), float("inf"), 10**1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"pair_strain": value})


class StrainFieldTests(unittest.TestCase):
    def setUp(self):
        self.tools, self.pairs = planner.conditioned_uniforms(frame(), 0.22)
        self.strains = pair_strain_uniforms(frame(), 0.22)
        self.settings = dict(
            aspect=4 / 3,
            stir_radius=0.22,
            flow_strength=1.1,
            pair_swirl=0.2,
            pair_strain=0.7,
            strains=self.strains,
        )

    def test_complete_analytic_field_is_curl_and_divergence_free(self):
        points = np.random.default_rng(1931).uniform([-1.332, -0.999], [1.332, 0.999], (512, 2))
        h = 2e-6
        derivatives, divergence = [], np.zeros(len(points))
        for index, offset in enumerate((np.array([h, 0]), np.array([0, h]))):
            derivatives.append(
                (
                    streamfunction(points + offset, self.tools, self.pairs, **self.settings)
                    - streamfunction(points - offset, self.tools, self.pairs, **self.settings)
                )
                / (2 * h)
            )
            divergence += (
                velocity(points + offset, self.tools, self.pairs, **self.settings)[..., index]
                - velocity(points - offset, self.tools, self.pairs, **self.settings)[..., index]
            ) / (2 * h)
        np.testing.assert_allclose(
            velocity(points, self.tools, self.pairs, **self.settings),
            np.stack((derivatives[1], -derivatives[0]), axis=-1),
            atol=6e-10,
            rtol=2e-7,
        )
        self.assertLess(np.max(np.abs(divergence)), 2e-8)

    def test_zero_gain_is_exact_legacy_field_and_boundary_stays_closed(self):
        points = np.random.default_rng(19).uniform(-1, 1, (100, 2))
        baseline = {k: v for k, v in self.settings.items() if k not in ("strains", "pair_strain")}
        np.testing.assert_array_equal(
            velocity(points, self.tools, self.pairs, **baseline),
            velocity(points, self.tools, self.pairs, **{**self.settings, "pair_strain": 0}),
        )
        boundary = np.array([[4 / 3, 0.2], [-4 / 3, -0.7], [0.3, 1], [-0.8, -1], [2, 2]])
        np.testing.assert_array_equal(
            velocity(boundary, self.tools, self.pairs, **self.settings), 0
        )

    def test_positive_rate_stretches_along_axis_and_contracts_across_it(self):
        settings = {
            **self.settings,
            "flow_strength": 0,
            "pair_swirl": 0,
            "strains": [[1, 0, 2], [0, 0, 0], [0, 0, 0]],
        }
        points = np.array([[0.02, 0], [-0.02, 0], [0, 0.02], [0, -0.02]])
        result = velocity(points, np.zeros((3, 4)), np.zeros((3, 3)), **settings)
        self.assertGreater(result[0, 0], 0)
        self.assertLess(result[1, 0], 0)
        self.assertLess(result[2, 1], 0)
        self.assertGreater(result[3, 1], 0)
        np.testing.assert_array_equal(result[:2, 1], 0)
        np.testing.assert_array_equal(result[2:, 0], 0)

    def test_invalid_active_descriptors_are_rejected(self):
        for changes in (
            {"pair_strain": -1},
            {"pair_strain": 4.1},
            {"pair_strain": float("nan")},
            {"strains": None},
            {"strains": [[2, 0, 1]] * 3},
            {"strains": [[0, 0, 1]] * 3},
            {"strains": [[1, 0, 21]] * 3},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                velocity([0, 0], self.tools, self.pairs, **{**self.settings, **changes})

    def test_layout_uses_the_same_strain_and_zero_preserves_archive(self):
        with (
            patch.object(planner, "PILOT_STEPS", 32),
            patch.object(planner, "CANDIDATE_LAYOUTS", 2),
        ):
            base = planner.plan_engaged_layout(Orbit(), 3, config())
            zero = planner.plan_engaged_layout(Orbit(), 3, {**config(), "pair_strain": 0})
            self.assertEqual(base, zero)
            active = planner.plan_engaged_layout(
                Orbit(), 3, {**config(), "flow_strength": 0, "pair_swirl": 0, "pair_strain": 0.7}
            )
            self.assertEqual(active["flow"]["pair_strain"], 0.7)
            self.assertTrue(all(np.isfinite(p["position"]).all() for p in active["pools"]))
        source, point, fraction = Orbit(), [[0.1, 0.2]], 0.31
        sampled = source.frame(fraction)
        tools, pairs = planner.conditioned_uniforms(sampled, 0.22)
        expected = velocity(
            point,
            tools,
            pairs,
            **{**self.settings, "strains": pair_strain_uniforms(sampled, 0.22)},
        )
        actual = planner.flow_velocity(
            source, point, fraction, {**config(), "pair_swirl": 0.2, "pair_strain": 0.7}
        )
        np.testing.assert_array_equal(actual, expected)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class GPUStrainTests(unittest.TestCase):
    def engine(self, **overrides):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette

        engine = Engine(
            SourceFixture(),
            {
                "resolution": [96, 72],
                "steps": 40,
                "initial_pattern": "scattered",
                "initial_pigment_weights": [2.2, 0.9, 0.5],
                "material_model": "laminate",
                "lower_transport_scale": 0.82,
                "interlayer_exchange_rate": 0.55,
                "carrier_velocity": [0, 0],
                "flow_domain_scale": 1,
                "pair_swirl": 0.15,
                "deposition": 0,
                "settling_scale": 0,
                "underpaint_strength": 0,
                "underpaint_release": 0,
                "burial_rate": 0,
                "substrate_um": 0,
                "mass_budget_interval_steps": 7,
                "diffusion_coefficient": 0.00002,
                **overrides,
            },
            generate_palette("0x808861c25b6c", 3, mode="composed"),
            [],
        )
        self.addCleanup(engine.close)
        return engine

    def test_integrated_strain_changes_material_and_explicit_zero_keeps_exact_default(self):
        baseline = self.engine()
        baseline.advance_to(40)
        expected = baseline.snapshot()
        self.assertNotIn("pair_strain", baseline.metadata)
        baseline.close()
        zero = self.engine(pair_strain=0)
        zero.advance_to(40)
        for key, values in zero.snapshot().items():
            np.testing.assert_array_equal(values, expected[key])
        zero.close()
        active = self.engine(pair_strain=0.9)
        active.advance_to(40)
        actual = active.snapshot()
        self.assertEqual(active.metadata["pair_strain"]["gain"], 0.9)
        self.assertGreater(float(np.max(np.abs(actual["pigment"] - expected["pigment"]))), 1e-4)
        self.assertTrue(all(np.isfinite(value).all() for value in actual.values()))
        self.assertLessEqual(active._gpu.maximum_courant, 1.500001)

    def test_integrated_strain_is_independent_of_capture_boundaries(self):
        uninterrupted = self.engine(pair_strain=0.9)
        uninterrupted.advance_to(40)
        expected, diagnostics = uninterrupted.snapshot(), uninterrupted.diagnostics
        uninterrupted.close()
        captured = self.engine(pair_strain=0.9)
        for step in (7, 11, 23, 39, 40):
            captured.advance_to(step)
            captured.snapshot((24, 18))
        for key, values in captured.snapshot().items():
            np.testing.assert_array_equal(values, expected[key])
        self.assertEqual(captured.diagnostics, diagnostics)

    def test_shader_matches_reference_and_zero_matches_original_shader(self):
        import moderngl

        ctx = moderngl.create_standalone_context(require=430, backend="egl")
        self.addCleanup(ctx.release)
        width, height, domain = 64, 48, 1.6
        velocity_texture = ctx.texture((width, height), 2, dtype="f4")
        self.addCleanup(velocity_texture.release)
        maximums = ctx.buffer(reserve=4 * 4 * 3)
        self.addCleanup(maximums.release)
        tools, pairs = planner.conditioned_uniforms(frame(), 0.22)
        strains = pair_strain_uniforms(frame(), 0.22)

        def render(path, gain, flow_domain):
            shader = ctx.compute_shader(path.read_text())
            try:
                for key, value in {
                    "u_size": (width, height),
                    "u_aspect": width / height,
                    "u_domain": domain,
                    "u_radius": 0.22,
                    "u_strength": 1.1,
                    "u_pair_gain": 0.2,
                    "u_carrier": (0, 0),
                    "u_pair_strain": gain,
                    "u_flow_domain": flow_domain,
                }.items():
                    if key in shader:
                        shader[key].value = value
                shader["u_tools"].write(tools.tobytes())
                shader["u_pairs"].write(pairs.tobytes())
                if "u_strains" in shader:
                    shader["u_strains"].write(strains.tobytes())
                velocity_texture.bind_to_image(0, read=False, write=True)
                maximums.bind_to_storage_buffer(1)
                shader.run(4, 3)
                ctx.memory_barrier()
                return np.frombuffer(velocity_texture.read(), "f4").reshape(height, width, 2)
            finally:
                shader.release()

        shader_root = Path(__file__).parent
        current = shader_root / "shaders/flow.glsl"
        original = shader_root.parent / "estuary/shaders/flow.glsl"
        np.testing.assert_array_equal(render(current, 0, domain), render(original, 0, domain))
        x = ((np.arange(width) + 0.5) / width * 2 - 1) * width / height * domain
        y = ((np.arange(height) + 0.5) / height * 2 - 1) * domain
        points = np.stack(np.meshgrid(x, y), axis=-1)
        for flow_domain in (1, domain):
            expected = velocity(
                points,
                tools,
                pairs,
                aspect=width / height,
                stir_radius=0.22,
                flow_strength=1.1,
                pair_swirl=0.2,
                pair_strain=0.7,
                strains=strains,
                domain_scale=flow_domain,
            )
            np.testing.assert_allclose(render(current, 0.7, flow_domain), expected, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
