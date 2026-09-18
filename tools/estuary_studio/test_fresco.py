"""Material invariants and native-GPU process checks for Tidal Fresco."""

import os
import unittest

import numpy as np

from .fresco import DEFAULTS, Fresco, phase_exchange, substrate_field, validate_config


class ConfigTests(unittest.TestCase):
    def test_resolved_defaults_are_independent(self):
        a, b = validate_config({}), validate_config({})
        a["settling"][0] = 19
        self.assertEqual(b["settling"][0], DEFAULTS["settling"][0])

    def test_invalid_controls_fail_before_gpu_allocation(self):
        for value in (
            {"dryng": 2},
            {"drying": float("nan")},
            {"steps": True},
            {"resolution": [32, 9000]},
            {"settling": [1, -1, 2]},
            {"seed": 0.5},
            {"initial_pattern": "random"},
            {"height_scale_mm": -1},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config(value)


class PhaseTests(unittest.TestCase):
    def test_exchange_preserves_each_pigment_and_is_positive_at_large_dt(self):
        mobile = np.array([[0.2, 0.7, 0.01], [0, 2, 1]])
        deposited = np.array([[0.8, 0.1, 0.04], [1, 0, 0]])
        for dt in (0, 1e-6, 1, 1000):
            a, b = phase_exchange(mobile, deposited, [0.2, 3, 10], [1, 0.5, 0], dt)
            np.testing.assert_allclose(a + b, mobile + deposited, atol=1e-14)
            self.assertGreaterEqual(float(a.min()), -1e-14)
            self.assertGreaterEqual(float(b.min()), -1e-14)

    def test_constant_rate_exchange_is_independent_of_partition(self):
        a, b = np.array([0.4, 0.1, 0.8]), np.array([0.7, 0.2, 0.1])
        expected = phase_exchange(a, b, [0.4, 2, 3], [1, 0.3, 0], 1)
        for _ in range(100):
            a, b = phase_exchange(a, b, [0.4, 2, 3], [1, 0.3, 0], 0.01)
        np.testing.assert_allclose(a, expected[0], atol=2e-14)
        np.testing.assert_allclose(b, expected[1], atol=2e-14)

    def test_zero_rates_are_identity_and_rewetting_lifts_deposit(self):
        m, d = phase_exchange([0, 0, 0], [1, 2, 3], 0, 0, 100)
        np.testing.assert_array_equal(m, [0, 0, 0])
        np.testing.assert_array_equal(d, [1, 2, 3])
        m, d = phase_exchange(m, d, 0, 2, 1)
        self.assertTrue(np.all(m > 0))
        np.testing.assert_allclose(d, np.array([1, 2, 3]) * np.exp(-2))

    def test_negative_or_nonfinite_values_rejected(self):
        for args in (
            (-1, 1, 1, 1, 1),
            (1, 1, float("nan"), 1, 1),
            (1, 1, 1, 1, -1),
            (1, 1, 1, 1, float("inf")),
        ):
            with self.assertRaises(ValueError):
                phase_exchange(*args)

    def test_substrate_is_fixed_bounded_and_varies_with_authored_seed(self):
        a = substrate_field(96, 72, 1.6, 17)
        np.testing.assert_array_equal(a, substrate_field(96, 72, 1.6, 17))
        self.assertTrue(np.all((a >= 0) & (a <= 1)))
        self.assertFalse(np.array_equal(a, substrate_field(96, 72, 1.6, 23)))


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class GPUFrescoTests(unittest.TestCase):
    def engine(self, **overrides):
        from tools.estuary.test_engine import SourceFixture

        engine = Fresco(
            SourceFixture(),
            {
                "resolution": [96, 72],
                "steps": 60,
                "flow_strength": 0,
                "pair_swirl": 0,
                "carrier_velocity": [0, 0],
                "deposition": 0,
                "wetting": 0,
                **overrides,
            },
        )
        self.addCleanup(engine.close)
        return engine

    def test_phase_exchange_conserves_local_pigment_without_transport(self):
        engine = self.engine()
        before = engine.snapshot()
        engine.advance_to(60)
        after = engine.snapshot()
        np.testing.assert_allclose(after["pigment"], before["pigment"], atol=1.5e-6)
        self.assertGreater(float(after["deposited"].sum()), 0)
        self.assertLess(float(after["wetness"].mean()), float(before["wetness"].mean()))
        self.assertTrue(np.all(after["height"] >= 0))
        self.assertTrue(np.all(after["roughness"] <= 1))
        self.assertTrue(np.all(np.linalg.norm(after["direction"], axis=-1) <= 1.000001))
        for value in after.values():
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(np.isfinite(value).all())

    def test_snapshot_cadence_does_not_change_the_process(self):
        a = self.engine(flow_strength=0.3, carrier_velocity=[0.2, 0.1])
        a.advance_to(30)
        a.snapshot()
        a.advance_to(60)
        expected = a.snapshot()
        a.close()
        b = self.engine(flow_strength=0.3, carrier_velocity=[0.2, 0.1])
        b.advance_to(60)
        for key, value in b.snapshot().items():
            np.testing.assert_array_equal(value, expected[key])

    def test_disabled_phase_exchange_preserves_the_original_transport(self):
        from tools.estuary.engine import Engine
        from tools.estuary.test_engine import SourceFixture

        fresco = self.engine(
            settling=[0, 0, 0],
            remobilization=[0, 0, 0],
            flow_strength=0.3,
            carrier_velocity=[0.2, 0.1],
        )
        recipe = fresco._gpu.recipe
        fresco.advance_to(60)
        actual = fresco.snapshot()
        fresco.close()
        original = Engine(SourceFixture(), recipe)
        self.addCleanup(original.close)
        original.advance_to(60)
        np.testing.assert_array_equal(actual["pigment"], original.read_state()[..., :3])
        np.testing.assert_array_equal(actual["deposited"], 0)

    def test_progress_validation_and_closed_context(self):
        engine = self.engine()
        engine.advance_to(1)
        for step in (0, 61, 1.5, True):
            with self.assertRaises(ValueError):
                engine.advance_to(step)
        engine.close()
        with self.assertRaises(RuntimeError):
            engine.snapshot()


if __name__ == "__main__":
    unittest.main()
