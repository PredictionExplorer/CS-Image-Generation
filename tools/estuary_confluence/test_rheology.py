"""Structural recovery, a bounded response operator, and material feedback."""

from __future__ import annotations

import copy
import os
import unittest

import numpy as np

from tools.estuary.test_engine import SourceFixture

from .engine import Engine
from .engine import validate_config as simulation_config
from .mass_budget import pigment_mass
from .palette import generate_palette
from .rheology import (
    DEFAULTS,
    FIELD_NAMES,
    ITERATIONS,
    contact_resistance_factor,
    discrete_curl,
    evolve_structure,
    initialization_config,
    occupancy_factor,
    response_plan,
    screened_response,
    validate_config,
)


def settings(**overrides):
    return {
        "resolution": [96, 72],
        "steps": 20,
        "initial_pattern": "scattered",
        "material_model": "laminate",
        "deposition": 0,
        "settling_scale": 0,
        "underpaint_strength": 0,
        "underpaint_release": 0,
        "burial_rate": 0,
        "carrier_velocity": [0, 0],
        "flow_domain_scale": 1.0,
        "flow_strength": 0.7,
        "pair_swirl": 0.2,
        "pair_strain": 0.4,
        "mass_budget_interval_steps": 7,
        **overrides,
    }


class RheologyTests(unittest.TestCase):
    def test_occupancy_reference_is_optional_and_independent_of_history_support(self):
        self.assertNotIn("occupancy_mass_reference", DEFAULTS)
        self.assertNotIn("occupancy_mass_reference", validate_config({}))
        self.assertEqual(validate_config(DEFAULTS), DEFAULTS)
        for value in (1e-5, 0.01, 0.02, 1):
            resolved = validate_config({"occupancy_mass_reference": value})
            self.assertEqual(resolved.pop("occupancy_mass_reference"), float(value))
            self.assertEqual(resolved, DEFAULTS)
        for value in (None, False, 0, 0.000001, 1.01, float("nan"), float("inf"), 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"occupancy_mass_reference": value})

    def test_occupancy_gate_is_bounded_monotone_and_half_strength_at_reference(self):
        masses = np.array([0, 1e-7, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10])
        for reference in (1e-5, 0.01, 0.02, 1):
            gate = occupancy_factor(masses, reference)
            self.assertTrue(np.all((gate >= 0) & (gate < 1)))
            self.assertTrue(np.all(np.diff(gate) > 0))
            self.assertEqual(float(occupancy_factor(reference, reference)), 0.5)
            self.assertEqual(float(gate[0]), 0)
        self.assertTrue(np.all(occupancy_factor(masses, 0.02) <= occupancy_factor(masses, 0.01)))
        self.assertAlmostEqual(float(occupancy_factor(0.008, 0.01)), 4 / 9)

    def test_contact_gated_traits_are_bounded_neutral_and_mass_weighted(self):
        mass = np.array([1, 3, 0])
        affinity, contact = np.array([1, -1, 1]), np.array([0.5, 0.25, 1])
        self.assertEqual(
            contact_resistance_factor(mass, affinity, contact, amplitude=0.25), 0.984375
        )
        for amplitude in (0, 0.08, 0.25):
            self.assertEqual(
                contact_resistance_factor(mass, affinity, np.zeros(3), amplitude=amplitude), 1
            )
            self.assertEqual(
                contact_resistance_factor(np.zeros(3), affinity, contact, amplitude=amplitude), 1
            )
            for sign in (-1, 1):
                self.assertEqual(
                    contact_resistance_factor(
                        mass, np.full(3, sign), np.ones(3), amplitude=amplitude
                    ),
                    1 + sign * amplitude,
                )
        rng = np.random.default_rng(905)
        for _ in range(20):
            factor = contact_resistance_factor(
                rng.random((2, 7, 9)),
                rng.uniform(-1, 1, (2, 7, 9)),
                rng.random((2, 7, 9)),
                amplitude=0.25,
            )
            self.assertGreaterEqual(factor, 0.75)
            self.assertLessEqual(factor, 1.25)

    def test_contact_resistance_reference_rejects_invalid_material_inputs(self):
        for mass, affinity, contact, amplitude in (
            ([-1], [0], [1], 0.1),
            ([1], [1.1], [1], 0.1),
            ([1], [0], [1.1], 0.1),
            ([1], [0], [1], 0.26),
            ([1], [0], [1], True),
            ([1], [float("nan")], [1], 0.1),
            ([1, 2], [0], [1], 0.1),
        ):
            with self.subTest(amplitude=amplitude), self.assertRaises(ValueError):
                contact_resistance_factor(mass, affinity, contact, amplitude=amplitude)

    def test_optional_config_is_strict_and_zero_retains_exact_legacy_dictionary(self):
        self.assertEqual(validate_config({}), DEFAULTS)
        baseline = simulation_config(settings())
        for value in (None, {"strength": 0}):
            self.assertEqual(simulation_config(settings(rheology=value)), baseline)
        for value in (
            False,
            [],
            {"version": "future"},
            {"bogus": 1},
            {"strength": True},
            {"strength": 21},
            {"rebuild_rate": float("nan")},
            {"shear_scale": 0},
            {"response_length": 10**400},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config(value)
        with self.assertRaisesRegex(ValueError, "laminate"):
            simulation_config({"rheology": {}})

    def test_initial_layout_controls_are_unchanged_and_not_mutated(self):
        base = settings()
        selected = settings(rheology={"strength": 6})
        saved = copy.deepcopy(selected)
        self.assertEqual(initialization_config(selected), base)
        self.assertEqual(selected, saved)
        self.assertIs(initialization_config(base), base)

    def test_rest_rebuild_drying_and_shear_breakdown_are_bounded(self):
        config = {"rebuild_rate": 1, "dry_rebuild_rate": 2, "breakdown_rate": 8}
        wet_rest = evolve_structure(0.2, 1, 0, 1, config)
        dry_rest = evolve_structure(0.2, 0, 0, 1, config)
        sheared = evolve_structure(0.9, 1, 1000, 1, config)
        self.assertGreater(float(wet_rest), 0.2)
        self.assertGreater(float(dry_rest), float(wet_rest))
        self.assertLess(float(sheared), 0.2)
        state = np.linspace(0, 1, 101)
        for dt in (0, 1e-8, 1, 1000):
            result = evolve_structure(state, np.linspace(0, 1, 101), 1e9, dt, config)
            self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_exact_frozen_rate_reaction_has_timestep_semigroup(self):
        state = np.linspace(0, 1, 37)
        direct = evolve_structure(state, 0.6, 13, 0.8, {})
        for _ in range(80):
            state = evolve_structure(state, 0.6, 13, 0.01, {})
        np.testing.assert_allclose(state, direct, rtol=0, atol=2e-15)

    def test_response_plan_proves_fixed_contraction_across_shapes_and_lengths(self):
        for size in ([32, 32], [96, 72], [2048, 1536], [1001, 777], [512, 1024]):
            for length in (0.005, 0.04, 0.25):
                plan = response_plan(size, 1.6, {"response_length": length})
                self.assertLessEqual(max(plan["resolution"]), 128)
                self.assertLessEqual(max(plan["coefficients"]), 1 + 1e-14)
                self.assertLessEqual(plan["maximum_contraction"], 0.8 + 1e-14)
                self.assertLess(plan["exact_arithmetic_error_factor"], 6.3e-7)
                self.assertEqual(plan["iterations"], ITERATIONS)

    def test_uniform_structure_damps_a_known_discrete_streamfunction_mode(self):
        n = 33
        yy, xx = np.mgrid[:n, :n]
        psi = np.sin(np.pi * xx / (n - 1)) * np.sin(2 * np.pi * yy / (n - 1))
        psi[[0, -1]] = 0
        psi[:, [0, -1]] = 0
        coefficients = (0.8, 0.6)
        eigenvalue = (
            4 * coefficients[0] * np.sin(np.pi / (2 * (n - 1))) ** 2
            + 4 * coefficients[1] * np.sin(np.pi / (n - 1)) ** 2
        )
        for resistance in (0, 1, 6):
            delta = screened_response(psi, np.full_like(psi, resistance), coefficients)
            expected = psi * resistance / (1 + resistance + eigenvalue)
            np.testing.assert_allclose(delta, expected, rtol=0, atol=2e-12)
            self.assertLessEqual(np.linalg.norm(psi - delta), np.linalg.norm(psi))

    def test_variable_response_converges_with_bound_and_adds_no_discrete_divergence(self):
        rng = np.random.default_rng(451)
        psi, amount = rng.normal(size=(37, 41)), rng.uniform(0, 6, (37, 41))
        delta = screened_response(psi, amount, (1, 1))
        converged = screened_response(psi, amount, (1, 1), iterations=512)
        bound = np.max(np.abs(psi * amount)) * 0.8**ITERATIONS
        self.assertLessEqual(np.max(np.abs(delta - converged)), bound)
        self.assertFalse(delta[[0, -1]].any())
        self.assertFalse(delta[:, [0, -1]].any())
        curl = discrete_curl(delta, 0.03)
        divergence = (
            (curl[1:-1, 2:, 0] - curl[1:-1, :-2, 0]) + (curl[2:, 1:-1, 1] - curl[:-2, 1:-1, 1])
        ) / 0.06
        np.testing.assert_allclose(divergence, 0, rtol=0, atol=1e-12)
        self.assertFalse(curl[:, [0, -1], 0].any())
        self.assertFalse(curl[[0, -1], :, 1].any())

    def test_the_same_forcing_retains_prior_recovery_or_shear_history(self):
        config = {"rebuild_rate": 0.1, "dry_rebuild_rate": 3, "breakdown_rate": 8}
        rested = evolve_structure(0.5, 0, 0, 0.8, config)
        worked = evolve_structure(0.5, 1, 300, 0.8, config)
        self.assertGreater(float(rested), float(worked) + 0.5)
        psi = np.ones((17, 19))
        stiff = screened_response(psi, np.full_like(psi, 3 * rested**2), (1, 1))
        soft = screened_response(psi, np.full_like(psi, 3 * worked**2), (1, 1))
        self.assertGreater(float(stiff[8, 9]), float(soft[8, 9]) + 0.5)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "GPU qualification is opt-in")
class RheologyGPUTests(unittest.TestCase):
    def engine(self, **overrides):
        engine = Engine(
            SourceFixture(),
            settings(**overrides),
            generate_palette("0xbc53af1cd380", 3, mode="composed"),
            [],
        )
        self.addCleanup(engine.close)
        return engine

    def test_disabled_has_exact_legacy_fields_physics_and_diagnostics(self):
        first = self.engine()
        first.advance_to(first.steps)
        fields, diagnostics = first.snapshot(), first.diagnostics
        first.close()
        second = self.engine(rheology={"strength": 0})
        second.advance_to(second.steps)
        self.assertEqual(second.diagnostics, diagnostics)
        self.assertEqual(set(fields), set(second.snapshot()))
        for name, field in fields.items():
            np.testing.assert_array_equal(second.snapshot()[name], field)

    def test_explicit_legacy_occupancy_reference_preserves_all_omitted_state_bytes(self):
        controls = {"interaction": {"material_variation": {"amplitude": 0.25}}}
        baseline = self.engine(rheology={}, **controls)
        baseline.advance_to(baseline.steps)
        expected, diagnostics = baseline.snapshot(), baseline.diagnostics
        baseline.close()
        explicit = self.engine(rheology={"occupancy_mass_reference": 1e-5}, **controls)
        explicit.advance_to(explicit.steps)
        self.assertEqual(explicit.diagnostics, diagnostics)
        self.assertEqual(set(explicit.snapshot()), set(expected))
        for name, field in expected.items():
            self.assertEqual(explicit.snapshot()[name].tobytes(), field.tobytes())

    def test_amount_reference_sets_expected_coefficient_with_and_without_traits(self):
        for coupled in (False, True):
            interaction = {"material_variation": {"amplitude": 0.25}} if coupled else None
            engine = self.engine(
                rheology={"strength": 3, "occupancy_mass_reference": 0.01},
                interaction=interaction,
                resolution=[128, 96],
            )
            gpu = engine._gpu
            self.assertEqual(gpu.rheology.config["minimum_concentration"], 1e-5)
            state = np.zeros((gpu.height, gpu.width, 4), dtype="f4")
            state[..., 0] = 0.75
            with gpu.ctx:
                for texture in gpu.rheology.states:
                    texture.write(state.tobytes())
                if coupled:
                    contact = np.zeros_like(state)
                    contact[..., 0] = 1
                    for texture in (*gpu.interaction.traits, *gpu.interaction.states):
                        texture.write(contact.tobytes())
                measured = []
                for mass in (0, 0.0001, 0.001, 0.01, 0.1):
                    paint = np.zeros_like(state)
                    paint[..., 0] = mass / 2
                    for block in (*gpu.blocks, *gpu.underpaints):
                        block[0].write(paint.tobytes())
                    gpu._flow(0.37)
                    coefficient = np.frombuffer(gpu.rheology.guide.read(), "f4").reshape(-1, 4)[
                        :, 1
                    ]
                    expected = 3 * 0.75**2 * occupancy_factor(mass, 0.01) * (1.25 if coupled else 1)
                    np.testing.assert_allclose(coefficient, expected, rtol=3e-6, atol=2e-7)
                    measured.append(float(coefficient.mean()))
                self.assertTrue(np.all(np.diff(measured) > 0))
            engine.close()

    def test_amount_reference_changes_feedback_retaining_budget_and_capture_cadence(self):
        controls = {"interaction": {"material_variation": {"amplitude": 0.25}}}
        baseline = self.engine(rheology={"strength": 3}, **controls)
        initial_mass = pigment_mass(baseline.snapshot()["pigment"], 1.6)
        baseline.advance_to(baseline.steps)
        legacy = baseline.snapshot()["pigment"]
        baseline.close()
        new_controls = {**controls, "rheology": {"strength": 3, "occupancy_mass_reference": 0.01}}
        engine = self.engine(**new_controls)
        engine.advance_to(engine.steps)
        final, diagnostics = engine.snapshot(), engine.diagnostics
        self.assertGreater(float(np.abs(final["pigment"] - legacy).max()), 1e-5)
        np.testing.assert_allclose(
            pigment_mass(final["pigment"], 1.6), initial_mass, rtol=5e-6, atol=1e-12
        )
        engine.close()
        split = self.engine(**new_controls)
        for step in (0, 3, 7, 12, 20):
            split.advance_to(step)
            split.snapshot([48, 36])
        self.assertEqual(split.diagnostics, diagnostics)
        for name, field in final.items():
            np.testing.assert_array_equal(split.snapshot()[name], field)

    def test_combined_zero_contact_is_exact_and_signed_traits_scale_only_resistance(self):
        baseline = self.engine(rheology={"strength": 3}, interaction={})
        gpu = baseline._gpu
        with gpu.ctx:
            gpu._flow(0.37)
            plain_velocity = np.frombuffer(gpu.velocity.read(), "f4").copy()
            plain_guide = np.frombuffer(gpu.rheology.guide.read(), "f4").copy()
        baseline.close()
        combined = self.engine(
            rheology={"strength": 3}, interaction={"material_variation": {"amplitude": 0.25}}
        )
        gpu = combined._gpu
        with gpu.ctx:
            gpu._flow(0.37)
            np.testing.assert_array_equal(np.frombuffer(gpu.velocity.read(), "f4"), plain_velocity)
            np.testing.assert_array_equal(
                np.frombuffer(gpu.rheology.guide.read(), "f4"), plain_guide
            )
            shape = (gpu.height, gpu.width, 4)
            state = np.zeros(shape, dtype="f4")
            state[..., 0] = 1
            for texture in gpu.interaction.states:
                texture.write(state.tobytes())
            coefficients = []
            for sign in (-1, 0, 1):
                traits = np.zeros(shape, dtype="f4")
                traits[..., 0] = sign
                for texture in gpu.interaction.traits:
                    texture.write(traits.tobytes())
                gpu._flow(0.37)
                coefficients.append(
                    np.frombuffer(gpu.rheology.guide.read(), "f4").reshape(-1, 4)[:, 1].copy()
                )
        expected = plain_guide.reshape(-1, 4)[:, 1]
        self.assertGreater(float(expected.max()), 0)
        np.testing.assert_array_equal(coefficients[1], expected)
        for actual, factor in zip((coefficients[0], coefficients[2]), (0.75, 1.25), strict=True):
            np.testing.assert_allclose(actual, expected * factor, atol=2e-7, rtol=2e-6)
        self.assertTrue(np.all(coefficients[0] <= coefficients[1]))
        self.assertTrue(np.all(coefficients[1] <= coefficients[2]))

    def test_zero_trait_amplitude_preserves_pure_rheology_bytes_and_fields(self):
        baseline = self.engine(rheology={}, interaction={})
        baseline.advance_to(baseline.steps)
        expected, diagnostics = baseline.snapshot(), baseline.diagnostics
        baseline.close()
        zero = self.engine(rheology={}, interaction={"material_variation": {"amplitude": 0}})
        self.assertIsNone(zero._gpu.rheology.trait_amplitude)
        zero.advance_to(zero.steps)
        self.assertEqual(zero.diagnostics, diagnostics)
        self.assertEqual(set(zero.snapshot()), set(expected))
        for name, field in expected.items():
            np.testing.assert_array_equal(zero.snapshot()[name], field)

    def test_natural_contact_coupling_changes_paint_transport_and_retains_cadence(self):
        interaction = {"contact_rate": 20, "origin_distance": 0.002}
        baseline = self.engine(rheology={"strength": 3}, interaction=interaction)
        initial_mass = pigment_mass(baseline.snapshot()["pigment"], 1.6)
        baseline.advance_to(baseline.steps)
        plain = baseline.snapshot()["pigment"]
        baseline.close()
        settings = {
            "rheology": {"strength": 3},
            "interaction": {**interaction, "material_variation": {"amplitude": 0.25}},
        }
        combined = self.engine(**settings)
        combined.advance_to(combined.steps)
        final, diagnostics = combined.snapshot(), combined.diagnostics
        self.assertGreater(float(final["interaction_upper"][..., 0].max()), 1e-3)
        self.assertGreater(float(np.abs(final["pigment"] - plain).max()), 1e-7)
        np.testing.assert_allclose(
            pigment_mass(final["pigment"], 1.6), initial_mass, rtol=5e-6, atol=1e-12
        )
        combined.close()
        split = self.engine(**settings)
        for step in (0, 3, 7, 12, 20):
            split.advance_to(step)
            split.snapshot([48, 36])
        self.assertEqual(split.diagnostics, diagnostics)
        for name, field in final.items():
            np.testing.assert_array_equal(split.snapshot()[name], field)

    def test_feedback_changes_transport_preserves_mass_and_initial_paint(self):
        base = self.engine()
        initial = base.snapshot()
        base.advance_to(base.steps)
        expected = base.snapshot()["pigment"]
        base.close()
        engine = self.engine(rheology={"strength": 6})
        for name, field in initial.items():
            np.testing.assert_array_equal(engine.snapshot()[name], field)
        self.assertTrue(set(FIELD_NAMES).issubset(engine.snapshot()))
        engine.advance_to(engine.steps)
        actual = engine.snapshot()
        self.assertGreater(float(np.abs(actual["pigment"] - expected).max()), 1e-4)
        np.testing.assert_allclose(
            pigment_mass(actual["pigment"], 1.6),
            pigment_mass(initial["pigment"], 1.6),
            rtol=5e-6,
            atol=1e-12,
        )
        for name in FIELD_NAMES:
            self.assertTrue(np.isfinite(actual[name]).all())
            self.assertTrue(np.all((actual[name] >= 0) & (actual[name] <= 1)))
        self.assertLessEqual(engine.diagnostics["maximum_courant"], 1.500001)

    def test_flow_retries_and_capture_cadence_do_not_advance_structure(self):
        engine = self.engine(rheology={})
        initial = engine.snapshot()
        gpu = engine._gpu
        with gpu.ctx:
            for fraction in (0.1, 0.5, 0.1):
                gpu._flow(fraction)
        for name, field in initial.items():
            np.testing.assert_array_equal(engine.snapshot()[name], field)
        engine.advance_to(engine.steps)
        final, diagnostics = engine.snapshot(), engine.diagnostics
        engine.close()
        other = self.engine(rheology={})
        for step in (0, 3, 7, 12, 20):
            other.advance_to(step)
            other.snapshot([48, 36])
            other.gpu_frame()
        self.assertEqual(other.diagnostics, diagnostics)
        for name, field in final.items():
            np.testing.assert_array_equal(other.snapshot()[name], field)

    def test_gpu_solver_matches_float64_and_curl_correction_is_solenoidal(self):
        engine = self.engine(rheology={"strength": 6})
        gpu = engine._gpu
        with gpu.ctx:
            gpu._flow(0.37)
            rheology = gpu.rheology
            w, h = rheology.coarse_size
            guide = np.frombuffer(rheology.guide.read(), "f4").reshape(h, w, 4)
            delta = np.frombuffer(rheology.delta[0].read(), "f4").reshape(h, w, 4)[..., 0]
            native = np.frombuffer(gpu.velocity.read(), "f4").reshape(gpu.height, gpu.width, 2)
            source = np.frombuffer(rheology.source_velocity.read(), "f4").reshape(native.shape)
            potential = np.frombuffer(rheology.potential.read(), "f4").reshape(
                gpu.height, gpu.width, 4
            )[..., 0]
        reference = screened_response(guide[..., 0], guide[..., 1], rheology.plan["coefficients"])
        np.testing.assert_allclose(delta, reference, atol=2e-7, rtol=2e-6)
        correction = source - native
        np.testing.assert_allclose(
            correction, discrete_curl(potential, 2 * gpu.domain / gpu.height), atol=2e-6, rtol=2e-5
        )
        div = (
            (correction[1:-1, 2:, 0] - correction[1:-1, :-2, 0])
            + (correction[2:, 1:-1, 1] - correction[:-2, 1:-1, 1])
        ) / (4 * gpu.domain / gpu.height)
        self.assertLess(float(np.max(np.abs(div))), 1e-4)
        self.assertFalse(correction[:, [0, -1], 0].any())
        self.assertFalse(correction[[0, -1], :, 1].any())
        x = ((np.arange(gpu.width) + 0.5) / gpu.width * 2 - 1) * gpu.aspect * gpu.domain
        y = ((np.arange(gpu.height) + 0.5) / gpu.height * 2 - 1) * gpu.domain
        guard = (np.abs(x)[None, :] >= gpu.aspect) | (np.abs(y)[:, None] >= 1)
        self.assertFalse(correction[guard].any())

    def test_uniform_material_resists_source_work_and_zero_structure_is_exact(self):
        engine = self.engine(rheology={"strength": 6})
        gpu = engine._gpu
        rheology = gpu.rheology
        # Uniform occupied material isolates the constitutive sign from pool
        # placement and composition. Preserve genuine forcing descriptors.
        paint = np.zeros((gpu.height, gpu.width, 4), dtype="f4")
        paint[..., 0] = 0.1
        with gpu.ctx:
            for block in (*gpu.blocks, *gpu.underpaints):
                block[0].write(paint.tobytes())
            works = []
            for structure in (0.0, 0.4, 0.9):
                state = np.zeros_like(paint)
                state[..., 0] = structure
                for texture in rheology.states:
                    texture.write(state.tobytes())
                gpu._flow(0.37)
                current = np.frombuffer(gpu.velocity.read(), "f4").reshape(gpu.height, gpu.width, 2)
                source = np.frombuffer(rheology.source_velocity.read(), "f4").reshape(current.shape)
                if structure == 0:
                    np.testing.assert_array_equal(current, source)
                works.append(float(np.sum(current.astype("f8") * source)))
        self.assertGreater(works[0], works[1])
        self.assertGreater(works[1], works[2])
        self.assertGreater(works[2], 0)

    def test_gpu_reaction_matches_analytic_shear_recovery_and_clears_vacuum(self):
        engine = self.engine(rheology={"breakdown_rate": 8})
        gpu = engine._gpu
        shape = (gpu.height, gpu.width, 4)
        paint, state, water = (np.zeros(shape, dtype="f4") for _ in range(3))
        paint[..., 0], state[..., 0], water[..., 0] = 0.1, 0.7, 0.8
        paint[:, :8] = 0
        velocity = np.zeros((*shape[:2], 2), dtype="f4")
        y = ((np.arange(gpu.height) + 0.5) / gpu.height * 2 - 1) * gpu.domain
        velocity[..., 0] = 25 * y[:, None]
        with gpu.ctx:
            for block in (*gpu.blocks, *gpu.underpaints):
                block[0].write(paint.tobytes())
            for texture in gpu.rheology.states:
                texture.write(state.tobytes())
            gpu.carrier[0].write(water.tobytes())
            gpu.velocity.write(velocity.tobytes())
            gpu.rheology.update(
                tuple(block[0] for block in gpu.blocks),
                tuple(block[0] for block in gpu.underpaints),
                gpu.carrier[0],
                gpu.velocity,
                0.3,
            )
        result = engine.snapshot()
        for name, scale in zip(
            FIELD_NAMES, (1, engine.config["lower_transport_scale"]), strict=True
        ):
            expected = evolve_structure(0.7, 0.8, 25 * scale, 0.3, engine.config["rheology"])
            np.testing.assert_allclose(result[name][1:-1, 8:], expected, rtol=0, atol=3e-7)
            self.assertFalse(result[name][:, :8].any())


if __name__ == "__main__":
    unittest.main()
