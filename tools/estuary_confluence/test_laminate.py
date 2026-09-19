"""Laminate conservation, wet contact, differential motion and fixed-clock tests."""

import copy
import os
import unittest

import numpy as np

from .engine import Engine, validate_config
from .laminate import exchange_layers, layer_fractions
from .mass_budget import pigment_mass, validate_report
from .palette import generate_palette

LAMINATE = {
    "material_model": "laminate",
    "initial_pattern": "scattered",
    "deposition": 0,
    "settling_scale": 0,
    "underpaint_strength": 0,
    "underpaint_release": 0,
    "burial_rate": 0,
}


class LaminateTests(unittest.TestCase):
    def test_configuration_is_opt_in_and_rejects_legacy_sources(self):
        self.assertEqual(validate_config({})["material_model"], "legacy")
        self.assertEqual(validate_config(LAMINATE)["material_model"], "laminate")
        for key, value in (
            ("material_model", "unknown"),
            ("lower_transport_scale", 1.1),
            ("lower_transport_scale", True),
            ("interlayer_exchange_rate", float("nan")),
            ("interlayer_min_concentration", 0),
            ("deposition", 0.01),
            ("underpaint_strength", 0.1),
            ("underpaint_release", 0.1),
            ("burial_rate", 0.1),
            ("settling_scale", 0.1),
            ("initial_pattern", "pools"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**LAMINATE, key: value})

    def test_initial_fractions_are_explicit_bounded_and_include_every_species(self):
        np.testing.assert_array_equal(
            layer_fractions({"layer_fractions": [0.1, 0, 1, 0.7]}, 4),
            np.array([0.1, 0, 1, 0.7], dtype="f4"),
        )
        for values in (
            None,
            [0, 1],
            [True, 0, 1, 1],
            [0.2, -0.1, 0.8, 0.5],
            [0, 1, 1, float("nan")],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                layer_fractions({"layer_fractions": values}, 4)

    def test_exchange_preserves_every_species_and_each_local_layer_amount(self):
        rng = np.random.default_rng(732)
        upper, lower = rng.uniform(0, 0.4, (2, 7, 9, 6))
        wet = rng.uniform(0, 1, (7, 9))
        for duration in (0, 1e-9, 0.4, 1000):
            a, b = exchange_layers(upper, lower, wet, rate=0.8, dt=duration)
            self.assertGreaterEqual(min(a.min(), b.min()), 0)
            np.testing.assert_allclose(a + b, upper + lower, atol=2e-16)
            np.testing.assert_allclose(a.sum(-1), upper.sum(-1), atol=1e-15)
            np.testing.assert_allclose(b.sum(-1), lower.sum(-1), atol=1e-15)

    def test_constant_wetness_exchange_is_partition_independent(self):
        upper = np.array([[[2.0, 0.0]]])
        lower = np.array([[[0.0, 1.0]]])
        wet = np.array([[0.6]])
        whole = exchange_layers(upper, lower, wet, rate=2, dt=1)
        split = upper, lower
        for duration in (0.01, 0.1, 0.47, 0.02, 0.4):
            split = exchange_layers(*split, wet, rate=2, dt=duration)
        for a, b in zip(whole, split, strict=True):
            np.testing.assert_allclose(a, b, atol=2e-16)
        equilibrium = exchange_layers(upper, lower, wet, rate=10, dt=1000)
        np.testing.assert_allclose(equilibrium[0], [[[4 / 3, 2 / 3]]])
        np.testing.assert_allclose(equilibrium[1], [[[2 / 3, 1 / 3]]])

    def test_dry_contact_and_air_do_not_exchange(self):
        upper = np.array([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
        lower = np.array([[[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]])
        a, b = exchange_layers(upper, lower, [[0, 1, 1]], rate=10, dt=10)
        np.testing.assert_array_equal(a, upper)
        np.testing.assert_array_equal(b, lower)

    def test_invalid_fields_and_exposure_are_rejected(self):
        field = np.ones((2, 3, 4))
        wet = np.ones((2, 3))
        for kwargs in (
            {"rate": -1},
            {"dt": True},
            {"dt": float("inf")},
            {"minimum_concentration": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                exchange_layers(field, field, wet, **{"rate": 1, "dt": 1, **kwargs})
        for bad in (-field, field * float("nan"), field[..., :3]):
            with self.assertRaises(ValueError):
                exchange_layers(field, bad, wet, rate=1, dt=1)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class LaminateGPUTests(unittest.TestCase):
    def engine(self, *, chromatic_count=5, **overrides):
        from tools.estuary.test_engine import SourceFixture

        palette = generate_palette("0xbc53af1cd380", chromatic_count=chromatic_count)
        palette["layer_fractions"] = [*[0.8, 0.2, 0.55, 0.9, 0.15][:chromatic_count], 0.05]
        engine = Engine(
            SourceFixture(),
            {
                **LAMINATE,
                "resolution": [96, 72],
                "steps": 40,
                "carrier_velocity": [0, 0],
                "flow_strength": 0.8,
                "drying": 0,
                "wetting": 0,
                "mass_budget_interval_steps": 7,
                "diffusion_coefficient": 0.00002,
                **overrides,
            },
            palette,
            [],
        )
        self.addCleanup(engine.close)
        return engine

    def test_initial_split_changes_layers_not_total_starting_pigment(self):
        laminate = self.engine()
        legacy = self.engine(material_model="legacy")
        new, old = laminate.snapshot(), legacy.snapshot()
        np.testing.assert_allclose(new["pigment"], old["pigment"], rtol=1e-7, atol=1e-9)
        shares = np.array(laminate.palette["layer_fractions"], dtype="f4")
        np.testing.assert_allclose(new["mobile"], old["pigment"] * shares, atol=1e-9)
        np.testing.assert_array_equal(new["deposit"], 0)
        self.assertGreater(new["underpaint"].sum(), 0)
        self.assertEqual(laminate.gpu_frame().material_model, "laminate")
        self.assertEqual(legacy.gpu_frame().material_model, "legacy")

    def test_interlayer_gpu_matches_reference_including_packed_channels(self):
        engine = self.engine(flow_strength=0, pair_swirl=0, diffusion_coefficient=0)
        rng = np.random.default_rng(392)
        upper, lower = rng.uniform(0, 0.2, (2, 72, 96, 6)).astype("f4")
        upper[0] = 0
        lower[:, 0] = 0
        with engine._gpu.ctx:
            for layer, values in ((engine._gpu.blocks, upper), (engine._gpu.underpaints, lower)):
                for index, block in enumerate(layer):
                    packed = np.zeros((72, 96, 4), dtype="f4")
                    n = min(4, 6 - index * 4)
                    packed[..., :n] = values[..., index * 4 : index * 4 + n]
                    block[0].write(packed.tobytes())
            engine._gpu._exchange_layers(0.37)
        fields = engine.snapshot()
        expected = exchange_layers(upper, lower, np.ones((72, 96)), rate=0.18, dt=0.37)
        for name, values in zip(("mobile", "underpaint"), expected, strict=True):
            np.testing.assert_allclose(fields[name], values, atol=4e-8, rtol=3e-7)

    def test_three_colors_exchange_matches_reference_without_a_second_channel_pack(self):
        engine = self.engine(
            chromatic_count=3, flow_strength=0, pair_swirl=0, diffusion_coefficient=0
        )
        self.assertEqual(len(engine._gpu.blocks), 1)
        rng = np.random.default_rng(573)
        upper, lower = rng.uniform(0, 0.2, (2, 72, 96, 4)).astype("f4")
        upper[0] = 0
        lower[:, 0] = 0
        wet = rng.uniform(0, 1, (72, 96)).astype("f4")
        wet[1] = 0
        wet[2] = 1
        carrier = np.zeros((72, 96, 4), dtype="f4")
        carrier[..., 0] = wet
        carrier[..., 2] = 1
        # Include the small-exposure polynomial used by production time steps,
        # ordinary wet exchange, and relaxation close to equilibrium.
        for duration in (1e-5, 0.37, 20.0):
            with self.subTest(duration=duration):
                with engine._gpu.ctx:
                    engine._gpu.blocks[0][0].write(upper.tobytes())
                    engine._gpu.underpaints[0][0].write(lower.tobytes())
                    engine._gpu.carrier[0].write(carrier.tobytes())
                    engine._gpu._exchange_layers(duration)
                fields = engine.snapshot()
                expected = exchange_layers(upper, lower, wet, rate=0.18, dt=duration)
                for name, values in zip(("mobile", "underpaint"), expected, strict=True):
                    np.testing.assert_allclose(fields[name], values, atol=4e-8, rtol=3e-7)
                    self.assertGreaterEqual(fields[name].min(), 0)
                np.testing.assert_allclose(
                    fields["mobile"] + fields["underpaint"], upper + lower, atol=6e-8, rtol=3e-7
                )
                np.testing.assert_array_equal(fields["mobile"][1], upper[1])
                np.testing.assert_array_equal(fields["underpaint"][1], lower[1])

    def test_lower_layer_moves_differently_and_all_species_budgets_include_it(self):
        engine = self.engine(lower_transport_scale=0.5, interlayer_exchange_rate=0)
        initial = engine.snapshot()
        engine.advance_to(engine.steps)
        final = engine.snapshot()
        for name in ("mobile", "underpaint"):
            self.assertFalse(np.array_equal(initial[name], final[name]))
            self.assertGreaterEqual(final[name].min(), 0)
        # Channel 0 starts with identical normalized shapes. Different velocities
        # must separate their normalized images, rather than merely recolor one.
        upper = final["mobile"][..., 0] / final["mobile"][..., 0].sum()
        lower = final["underpaint"][..., 0] / final["underpaint"][..., 0].sum()
        self.assertGreater(float(np.abs(upper - lower).sum()), 0.01)
        target = pigment_mass(initial["pigment"], engine.config["domain_scale"])
        np.testing.assert_allclose(
            pigment_mass(final["pigment"], engine.config["domain_scale"]), target, rtol=5e-6
        )
        validate_report(
            engine.mass_budget_report,
            {"chromatic_count": 5, "simulation": engine.config},
            final,
            layout=engine.layout,
        )

    def test_snapshots_and_capture_cadence_do_not_change_layer_physics(self):
        direct = self.engine()
        direct.advance_to(direct.steps)
        sampled = self.engine()
        for step in (1, 9, 18, 27, 40):
            sampled.advance_to(step)
            sampled.snapshot((24, 18))
            sampled.gpu_frame()
        a, b = direct.snapshot(), sampled.snapshot()
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(direct.mass_budget_report, sampled.mass_budget_report)
        # Neither inspection nor cached film frames may mutate the ledger.
        before = copy.deepcopy(sampled.mass_budget_report)
        sampled.snapshot()
        self.assertEqual(before, sampled.mass_budget_report)


if __name__ == "__main__":
    unittest.main()
