"""Actual-engine history integration, baseline isolation, and inspection invariance."""

import os
import unittest

import numpy as np

from .engine import Engine, validate_config
from .interaction import BASE_FIELDS, FIELD_NAMES
from .palette import generate_palette
from .test_laminate import LAMINATE


class InteractionConfigTests(unittest.TestCase):
    def test_disabled_history_preserves_old_resolved_configuration(self):
        self.assertEqual(
            validate_config(LAMINATE), validate_config({**LAMINATE, "interaction": None})
        )
        self.assertNotIn("interaction", validate_config(LAMINATE))
        with self.assertRaisesRegex(ValueError, "laminate"):
            validate_config({"interaction": {}})
        for value in (False, [], {"contact_rate": -1}, {"fabric_rate": float("nan")}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({**LAMINATE, "interaction": value})


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class InteractionEngineTests(unittest.TestCase):
    def engine(self, count=3, interaction=None, **changes):
        from tools.estuary.test_engine import SourceFixture

        config = {
            **LAMINATE,
            "resolution": [128, 96],
            "steps": 80,
            "carrier_velocity": [0.5, 0],
            "flow_strength": 0.4,
            "pair_swirl": 0.2,
            "lower_transport_scale": 0.82,
            "initial_pigment_weights": [2.2, 0.9, 0.5][:count],
            "interlayer_exchange_rate": 0.55,
            "mass_budget_interval_steps": 12,
            "diffusion_coefficient": 0.00002,
            "substrate_um": 0,
            **changes,
        }
        if interaction is not None:
            config["interaction"] = interaction
        engine = Engine(
            SourceFixture(), config, generate_palette("0xb7f327f9f722", count, mode="composed"), []
        )
        self.addCleanup(engine.close)
        return engine

    def test_history_never_changes_original_pigment_geometry_or_budget(self):
        for count in (1, 2, 3):
            baseline, textured = self.engine(count), self.engine(count, {})
            initial = textured.snapshot()
            self.assertEqual(set(initial), set(BASE_FIELDS) | set(FIELD_NAMES))
            for name in ("interaction_upper", "interaction_lower"):
                np.testing.assert_array_equal(initial[name], 0)
            baseline.advance_to(80)
            textured.advance_to(80)
            original, updated = baseline.snapshot(), textured.snapshot()
            for name in BASE_FIELDS:
                np.testing.assert_array_equal(original[name], updated[name], err_msg=name)
            self.assertEqual(baseline.mass_budget_report, textured.mass_budget_report)
            self.assertGreater(float(updated["interaction_upper"][..., 0].max()), 0)
            self.assertGreater(float(updated["interaction_upper"][..., 3].max()), 0)
            frame = textured.gpu_frame()
            self.assertTrue(frame.has_interaction)
            self.assertEqual(frame.origin_upper.components, 4)
            baseline.close()
            textured.close()

    def test_capture_cadence_and_readonly_snapshots_cannot_change_history(self):
        direct, inspected = self.engine(3, {}), self.engine(3, {})
        direct.advance_to(80)
        for step in (4, 17, 48, 63, 80):
            inspected.advance_to(step)
            inspected.snapshot((32, 24))
            inspected.gpu_frame().validate()
        a, b = direct.snapshot(), inspected.snapshot()
        for name in a:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        self.assertEqual(direct.diagnostics, inspected.diagnostics)

    def test_stationary_single_pigment_does_not_create_texture(self):
        engine = self.engine(1, {}, carrier_velocity=[0, 0], flow_strength=0, pair_swirl=0)
        engine.advance_to(80)
        fields = engine.snapshot()
        for name in ("interaction_upper", "interaction_lower"):
            np.testing.assert_array_equal(fields[name], 0)

    def test_history_empty_space_and_aggregate_partition_are_valid(self):
        engine = self.engine(3, {})
        engine.advance_to(80)
        fields = engine.snapshot()
        for phase, history in (
            ("mobile", "interaction_upper"),
            ("underpaint", "interaction_lower"),
        ):
            amount = fields[phase].sum(axis=-1)
            state = fields[history]
            # The global pigment-budget correction follows the reaction pass.
            # It can move positive paint just below the activation threshold;
            # that remains real paint, whereas exact vacuum has no history.
            np.testing.assert_array_equal(state[amount == 0], 0)
            self.assertTrue(np.isfinite(state).all())
            self.assertTrue(np.all((state[..., [0, 3]] >= 0) & (state[..., [0, 3]] <= 1)))
            self.assertTrue(
                np.all(np.linalg.norm(state[..., 1:3], axis=-1) <= state[..., 0] + 1e-5)
            )
            aggregate = fields[phase] * state[..., 3, None]
            dispersed = fields[phase] - aggregate
            self.assertGreaterEqual(float(dispersed.min()), 0)
            np.testing.assert_allclose(aggregate + dispersed, fields[phase], atol=2e-8, rtol=1e-7)
