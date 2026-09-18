"""Conservation, fixed-clock, event-dose and packed-channel contracts."""

import os
import unittest
from itertools import pairwise

import numpy as np

from .engine import (
    Engine,
    event_doses,
    phase_exchange,
    reduction_factor,
    substrate_field,
    substrate_seed,
    validate_config,
    validate_events,
)
from .palette import generate_palette


class ConfigTests(unittest.TestCase):
    def test_hex_entropy_preserves_existing_substrate_pixels(self):
        entropy = int("a19c61041e1a6798cfbdb53285160596f6775dfce438135d671ee250ea16bab0", 16)
        old = substrate_field(96, 72, 1.6, entropy % (2**63 - 1))
        new = substrate_field(96, 72, 1.6, substrate_seed(hex(entropy)))
        np.testing.assert_array_equal(new, old)
        self.assertEqual(substrate_seed(entropy), substrate_seed(hex(entropy)))
        for invalid in (True, -1, 2**256, 0.5, "not-a-seed"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                substrate_seed(invalid)

    def test_configs_are_independent(self):
        a, b = validate_config({}), validate_config({})
        a["carrier_velocity"][0] = 9
        self.assertEqual(b["carrier_velocity"], [2.0, 0.2])

    def test_invalid_controls_rejected(self):
        for config in (
            {"unknown": 1},
            {"steps": True},
            {"steps": 0},
            {"resolution": [32, 80000]},
            {"drying": float("nan")},
            {"drying": 10**1000},
            {"carrier_velocity": [0, float("nan")]},
            {"initial_pattern": "noise"},
            {"bloom_strength": -1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_reduction_requires_matching_integer_ratios(self):
        self.assertEqual(reduction_factor((4096, 3072), (1024, 768)), 4)
        self.assertEqual(reduction_factor((4096, 3072), None), 1)
        for size in ((1000, 768), (2048, 768), (8192, 6144), (0, 0), (True, 1)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                reduction_factor((4096, 3072), size)


class PhaseAndEventsTests(unittest.TestCase):
    def test_all_six_pigments_conserved_by_exchange(self):
        m = np.array([0.2, 0.4, 0.1, 0.8, 0.08, 0.7])
        d = np.array([0.4, 0.1, 0, 0.4, 0.9, 0.8])
        for dt in (0, 1e-8, 0.02, 1000):
            a, b = phase_exchange(m, d, [0, 0.3, 1, 4, 6, 2], [0, 0.5, 2, 0, 0.1, 0.2], dt)
            np.testing.assert_allclose(a + b, m + d, atol=3e-15)
            self.assertGreaterEqual(a.min(), -1e-15)
            self.assertGreaterEqual(b.min(), -1e-15)

    def test_integrated_blooms_are_bounded_and_partition_independent(self):
        events = [
            {
                "fraction": center,
                "position": [0, 0],
                "radius": 0.1,
                "duration": 0.01,
                "strength": 0.8,
            }
            for center in (0, 0.513, 1)
        ]
        total = event_doses(events, 0, 1)
        np.testing.assert_allclose(total[:, 3], [0.8, 0.8, 0.8])
        fractions = np.unique(np.r_[np.linspace(0, 1, 53), 0.5123, 0.5124])
        summed = sum(event_doses(events, a, b)[:, 3] for a, b in pairwise(fractions))
        np.testing.assert_allclose(summed, total[:, 3], atol=1e-7)
        self.assertTrue(np.all(event_doses(events, 0.2, 0.3)[:, 3] >= 0))

    def test_invalid_events_rejected(self):
        good = {
            "fraction": 0.5,
            "position": [0, 0],
            "radius": 0.1,
            "duration": 0.01,
            "strength": 0.8,
        }
        for invalid in (
            {**good, "duration": 0},
            {**good, "fraction": 2},
            {**good, "position": [float("nan"), 0]},
            {**good, "strength": -1},
        ):
            with self.assertRaises(ValueError):
                validate_events([invalid])
        with self.assertRaises(ValueError):
            validate_events([good] * 4)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class GPUConfluenceTests(unittest.TestCase):
    def engine(self, count=3, events=None, **overrides):
        from tools.estuary.test_engine import SourceFixture

        palette = generate_palette("0xbc53af1cd380", chromatic_count=count)
        engine = Engine(
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
            palette,
            [] if events is None else events,
        )
        self.addCleanup(engine.close)
        return engine

    def test_six_channel_local_phase_conservation_and_underpaint_release(self):
        engine = self.engine(count=5, underpaint_release=10, drying=0)
        before = engine.snapshot()
        engine.advance_to(60)
        after = engine.snapshot()
        np.testing.assert_allclose(after["pigment"], before["pigment"], atol=2e-6, rtol=1e-5)
        self.assertGreater(after["deposit"].sum(), 0)
        self.assertLess(after["underpaint"].sum(), before["underpaint"].sum())
        for phase in ("mobile", "deposit", "underpaint"):
            self.assertEqual(after[phase].shape, (72, 96, 6))
            self.assertGreaterEqual(after[phase].min(), 0)

    def test_gpu_reduction_preserves_material_means_and_does_not_write_state(self):
        engine = self.engine(count=5)
        engine.advance_to(20)
        before = engine.snapshot()
        small = engine.snapshot((24, 18))
        after = engine.snapshot()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        for key in ("mobile", "deposit", "underpaint", "pigment"):
            expected = before[key].reshape(18, 4, 24, 4, 6).mean(axis=(1, 3))
            np.testing.assert_allclose(small[key], expected, rtol=2e-6, atol=1e-8)

    def test_video_capture_cadence_is_not_physics(self):
        event = {
            "fraction": 0.5,
            "position": [0, 0],
            "radius": 0.8,
            "duration": 0.02,
            "strength": 1,
        }
        a = self.engine(events=[event], flow_strength=0.3, carrier_velocity=[0.2, 0.1])
        a.advance_to(17)
        a.snapshot((24, 18))
        a.advance_to(43)
        a.snapshot((48, 36))
        a.advance_to(60)
        expected = a.snapshot()
        a.close()
        b = self.engine(events=[event], flow_strength=0.3, carrier_velocity=[0.2, 0.1])
        b.advance_to(60)
        self.assertEqual(b.step, b.steps)
        for key, value in b.snapshot().items():
            np.testing.assert_array_equal(value, expected[key])

    def test_bloom_adds_water_without_creating_pigment(self):
        event = {"fraction": 0.8, "position": [0, 0], "radius": 1, "duration": 0.02, "strength": 1}
        a = self.engine(events=[event], drying=4)
        before = a.snapshot()
        a.advance_to(60)
        with_bloom = a.snapshot()
        np.testing.assert_allclose(with_bloom["pigment"], before["pigment"], atol=2e-6)
        a.close()
        b = self.engine(drying=4)
        b.advance_to(60)
        self.assertGreater(with_bloom["wetness"].mean(), b.snapshot()["wetness"].mean())

    def test_buried_history_forms_from_paint_without_an_initial_underlayer(self):
        engine = self.engine(underpaint_strength=0, burial_rate=8, drying=8)
        before = engine.snapshot()
        np.testing.assert_array_equal(before["underpaint"], 0)
        engine.advance_to(60)
        after = engine.snapshot()
        self.assertGreater(after["underpaint"].sum(), 0)
        np.testing.assert_allclose(after["pigment"], before["pigment"], atol=2e-6)

    def test_pooled_underpaint_stays_inside_the_initial_paint_footprint(self):
        engine = self.engine(initial_pattern="pools")
        initial = engine.snapshot()
        blank = initial["mobile"].sum(axis=-1) == 0
        self.assertTrue(blank.any())
        np.testing.assert_array_equal(initial["underpaint"][blank], 0)

    def test_fields_are_finite_and_bounded_and_invalid_time_fails(self):
        engine = self.engine(flow_strength=0.3, carrier_velocity=[0.2, 0.1])
        engine.advance_to(60)
        snapshot = engine.snapshot()
        for key, value in snapshot.items():
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(np.isfinite(value).all(), key)
        for key in ("wetness", "mixing", "roughness", "coverage"):
            self.assertTrue(np.all((snapshot[key] >= 0) & (snapshot[key] <= 1)), key)
        self.assertGreaterEqual(snapshot["height"].min(), 0)
        self.assertLessEqual(engine._gpu.maximum_courant, 1.500001)
        for step in (0, 61, 1.1, True):
            with self.assertRaises(ValueError):
                engine.advance_to(step)
        engine.close()
        with self.assertRaises(RuntimeError):
            engine.snapshot()


if __name__ == "__main__":
    unittest.main()
