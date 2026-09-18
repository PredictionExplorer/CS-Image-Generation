"""Conservation, fixed-clock, event-dose and packed-channel contracts."""

import math
import os
import unittest
from itertools import pairwise

import numpy as np

from .engine import (
    DIFFUSION_CFL,
    Engine,
    diffusion_substeps,
    event_doses,
    interdiffusion_step,
    mass_budget_factors,
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
            {"settling_scale": -1},
            {"initial_pattern": "scattered"},
            {"initial_pattern": "scattered", "underpaint_strength": 0, "load_radius": 0.5},
            {"initial_edge_width": 0},
            {"flow_domain_scale": 0.9},
            {"flow_domain_scale": 2},
            {"flow_domain_scale": float("nan")},
            {"diffusion_coefficient": -1},
            {"diffusion_min_concentration": 0},
            {"mass_budget_interval_steps": True},
            {"mass_budget_interval_steps": -1},
            {"mass_budget_interval_steps": 12},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_default_flow_boundary_resolves_to_the_simulation_guard(self):
        self.assertEqual(validate_config({})["flow_domain_scale"], 1.6)
        self.assertEqual(validate_config({"domain_scale": 2})["flow_domain_scale"], 2)
        self.assertEqual(validate_config({"flow_domain_scale": 1})["flow_domain_scale"], 1)

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


class InterdiffusionTests(unittest.TestCase):
    def test_contact_exchange_preserves_local_thickness_and_global_pigments(self):
        paint = np.array([[[1.0, 0.0], [0.0, 2.0]]])
        result = interdiffusion_step(paint, np.ones((1, 2)), coefficient=0.2, dt=1, pixel_size=1)
        np.testing.assert_allclose(result, [[[0.8, 0.2], [0.2, 1.8]]], atol=1e-15)
        np.testing.assert_allclose(result.sum(axis=-1), paint.sum(axis=-1), atol=1e-15)
        np.testing.assert_allclose(result.sum(axis=(0, 1)), paint.sum(axis=(0, 1)), atol=1e-15)

    def test_random_six_pigment_film_stays_positive_and_conservative(self):
        rng = np.random.default_rng(279)
        paint = rng.uniform(0, 0.3, (8, 9, 6))
        paint[0] = 0
        initial = paint.copy()
        wetness = rng.uniform(0, 1, (8, 9))
        for _ in range(100):
            paint = interdiffusion_step(paint, wetness, coefficient=0.002, dt=1, pixel_size=0.1)
        self.assertGreaterEqual(float(paint.min()), 0)
        np.testing.assert_allclose(paint.sum(axis=-1), initial.sum(axis=-1), atol=2e-14)
        np.testing.assert_allclose(paint.sum(axis=(0, 1)), initial.sum(axis=(0, 1)), atol=2e-13)
        np.testing.assert_array_equal(paint[0], 0)

    def test_air_and_dry_cells_block_flux(self):
        paint = np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
        result = interdiffusion_step(paint, np.ones((1, 3)), coefficient=0.2, dt=1, pixel_size=1)
        np.testing.assert_array_equal(result, paint)
        paint[0, 1] = [0.3, 0.3]
        result = interdiffusion_step(
            paint, np.array([[1.0, 0.0, 1.0]]), coefficient=0.2, dt=1, pixel_size=1
        )
        np.testing.assert_array_equal(result, paint)

    def test_coefficient_has_same_physical_decay_on_finer_grids(self):
        errors = []
        for width in (32, 64):
            x = (np.arange(width) + 0.5) / width
            first = 0.5 + 0.25 * np.cos(math.pi * x)
            paint = np.stack([first, 1 - first], axis=-1)[None, ...]
            pieces = diffusion_substeps(0.001, 1, 1 / width)
            for _ in range(pieces):
                paint = interdiffusion_step(
                    paint,
                    np.ones((1, width)),
                    coefficient=0.001,
                    dt=1 / pieces,
                    pixel_size=1 / width,
                )
            expected = 0.5 + 0.25 * np.exp(-0.001 * math.pi**2) * np.cos(math.pi * x)
            errors.append(float(np.max(np.abs(paint[0, :, 0] - expected))))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], 2e-6)

    def test_substeps_enforce_stability_without_clamping_the_coefficient(self):
        pieces = diffusion_substeps(0.002, 0.1, 0.01)
        self.assertGreater(pieces, 1)
        self.assertLessEqual(0.002 * (0.1 / pieces) / 0.01**2, DIFFUSION_CFL)
        self.assertEqual(diffusion_substeps(0, 1, 0.1), 0)
        with self.assertRaises(ValueError):
            diffusion_substeps(0.01, 1, 0.0001)
        with self.assertRaises(ValueError):
            interdiffusion_step(
                np.ones((2, 2, 2)), np.ones((2, 2)), coefficient=0.1, dt=1, pixel_size=0.1
            )


class MassBudgetFactorTests(unittest.TestCase):
    def test_factors_are_positive_float32_and_preserve_empty_channels(self):
        factors = mass_budget_factors([1, 2, 0], [1.1, 1.8, 0])
        self.assertEqual(factors.dtype, np.float32)
        np.testing.assert_allclose(factors, [1 / 1.1, 2 / 1.8, 1], rtol=1e-7)
        self.assertTrue(np.all(factors > 0))

    def test_lost_or_invented_pigments_cannot_be_silently_recovered(self):
        for target, current in (([1, 0], [0, 0]), ([1, 0], [1, 0.01])):
            with self.assertRaises(FloatingPointError):
                mass_budget_factors(target, current)
        for target, current in (([1], [-1]), ([float("nan")], [1]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                mass_budget_factors(target, current)


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

    def test_scattered_initial_pools_are_pure_and_all_chromatic_channels_are_visible(self):
        engine = self.engine(
            count=5,
            initial_pattern="scattered",
            underpaint_strength=0,
            settling_scale=0,
            burial_rate=0,
        )
        initial = engine.snapshot()
        self.assertEqual(len(engine.layout["pools"]), 5)
        self.assertEqual(engine.metadata["initial_layout"], engine.layout)
        np.testing.assert_array_equal(initial["mobile"][..., -1], 0)
        np.testing.assert_array_equal(initial["deposit"], 0)
        np.testing.assert_array_equal(initial["underpaint"], 0)
        self.assertTrue(np.all(initial["mobile"][..., :5].sum(axis=(0, 1)) > 0))
        self.assertTrue(np.all((initial["mobile"] > 0).sum(axis=-1) <= 1))
        self.assertTrue(np.isfinite(initial["mobile"]).all())
        engine.advance_to(engine.steps)
        settled = engine.snapshot()
        np.testing.assert_array_equal(settled["deposit"], 0)
        np.testing.assert_array_equal(settled["underpaint"], 0)
        np.testing.assert_allclose(settled["mobile"], initial["mobile"], atol=2e-6)

    def test_scattered_three_and_five_have_identical_primary_pool_rasters(self):
        a = self.engine(count=3, initial_pattern="scattered", underpaint_strength=0)
        first, layout = a.snapshot(), a.layout
        a.close()
        b = self.engine(count=5, initial_pattern="scattered", underpaint_strength=0)
        np.testing.assert_array_equal(first["mobile"][..., :3], b.snapshot()["mobile"][..., :3])
        self.assertEqual(layout["pools"], b.layout["pools"][:3])

    def test_scattered_layout_is_resolution_independent(self):
        a = self.engine(count=5, initial_pattern="scattered", underpaint_strength=0)
        first = a.layout
        a.close()
        b = self.engine(
            count=5, resolution=[192, 144], initial_pattern="scattered", underpaint_strength=0
        )
        self.assertEqual(first, b.layout)

    def test_zero_settling_fast_path_exactly_matches_full_phase_dispatch(self):
        config = {
            "count": 5,
            "initial_pattern": "scattered",
            "underpaint_strength": 0,
            "settling_scale": 0,
            "flow_strength": 0.3,
            "carrier_velocity": [0.2, 0.1],
        }
        fast = self.engine(**config)
        self.assertTrue(fast._gpu.skip_phase)
        self.assertIn("identically zero", fast.metadata["phase_exchange"])
        self.assertIn("no trajectory pigment source", fast.metadata["mass_limitations"])
        fast.advance_to(fast.steps)
        expected = fast.snapshot()
        count = fast._gpu.internal_steps
        fast.close()
        full = self.engine(**config)
        full._gpu.skip_phase = False
        full.advance_to(full.steps)
        for key, value in full.snapshot().items():
            np.testing.assert_array_equal(value, expected[key])
        self.assertEqual(full._gpu.internal_steps, count)

    def test_gpu_interdiffusion_matches_reference_without_changing_the_silhouette(self):
        engine = self.engine(
            count=5,
            initial_load=0,
            underpaint_strength=0,
            settling_scale=0,
            diffusion_coefficient=0.002,
            drying=0,
        )
        gpu = engine._gpu
        paint = np.zeros((72, 96, 6), dtype="f4")
        paint[15:57, 12:48, 0] = 0.1
        paint[15:57, 48:84, 4] = 0.3
        paint[32:40, 12:84, 2] = 0.07
        wet = np.broadcast_to(np.linspace(0.1, 1, 96, dtype="f4"), (72, 96)).copy()
        with gpu.ctx:
            for index, block in enumerate(gpu.blocks):
                packed = np.zeros((72, 96, 4), dtype="f4")
                channels = min(4, 6 - index * 4)
                packed[..., :channels] = paint[..., index * 4 : index * 4 + channels]
                block[0].write(packed.tobytes())
            carrier = np.zeros((72, 96, 4), dtype="f4")
            carrier[..., 0], carrier[..., 2] = wet, 1
            gpu.carrier[0].write(carrier.tobytes())
            gpu._interdiffuse(0.15)
        actual = engine.snapshot()["mobile"]
        expected = interdiffusion_step(
            paint, wet, coefficient=0.002, dt=0.15, pixel_size=2 * gpu.domain / gpu.height
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(actual.sum(axis=-1), paint.sum(axis=-1), rtol=2e-6, atol=3e-8)
        np.testing.assert_allclose(
            actual.sum(axis=(0, 1), dtype="f8"),
            paint.sum(axis=(0, 1), dtype="f8"),
            rtol=2e-7,
            atol=1e-8,
        )
        np.testing.assert_array_equal(actual[paint.sum(axis=-1) == 0], 0)
        self.assertGreater(float(actual[20, 47, 4]), 0)
        self.assertGreaterEqual(float(actual.min()), 0)
        # Exercise the real packed-group ping-pong path over four stability
        # substeps, not only a single shader dispatch.
        expected = actual.astype("f8")
        for _ in range(4):
            expected = interdiffusion_step(
                expected,
                wet,
                coefficient=0.002,
                dt=0.75 / 4,
                pixel_size=2 * gpu.domain / gpu.height,
            )
        with gpu.ctx:
            gpu._interdiffuse(0.75)
        actual = engine.snapshot()["mobile"]
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-8)
        np.testing.assert_allclose(actual.sum(axis=-1), paint.sum(axis=-1), rtol=3e-6, atol=5e-8)
        self.assertEqual(engine.diagnostics["diffusion_substeps"], 5)
        self.assertLessEqual(engine.diagnostics["maximum_diffusion_number"], DIFFUSION_CFL)

    def test_diffusion_capture_cadence_and_diagnostics_are_independent_of_output(self):
        controls = {
            "count": 5,
            "diffusion_coefficient": 0.002,
            "underpaint_strength": 0,
            "settling_scale": 0,
            "flow_strength": 0.3,
            "carrier_velocity": [0.2, 0.1],
        }
        a = self.engine(**controls)
        a.advance_to(17)
        a.snapshot((24, 18))
        a.advance_to(43)
        a.snapshot()
        a.advance_to(60)
        expected, diagnostic = a.snapshot(), a.diagnostics
        a.close()
        b = self.engine(**controls)
        b.advance_to(60)
        for key, value in b.snapshot().items():
            np.testing.assert_array_equal(value, expected[key])
        self.assertEqual(b.diagnostics, diagnostic)
        self.assertEqual(diagnostic["canonical_steps"], 60)
        self.assertGreater(diagnostic["diffusion_substeps"], 0)
        self.assertLessEqual(diagnostic["maximum_diffusion_number"], DIFFUSION_CFL)
        self.assertLessEqual(diagnostic["maximum_courant"], 1.500001)

    def test_confined_shader_matches_analytic_curl_and_is_zero_outside_visible_extent(self):
        from tools.estuary.engine import tool_uniforms
        from tools.estuary.flow_reference import velocity

        engine = self.engine(
            flow_domain_scale=1, flow_strength=0.7, pair_swirl=0.2, carrier_velocity=[0.3, 0.1]
        )
        gpu = engine._gpu
        with gpu.ctx:
            gpu._flow(0.37)
            actual = np.frombuffer(gpu.velocity.read(), dtype="f4").reshape(72, 96, 2)
        x = ((np.arange(96) + 0.5) / 96 * 2 - 1) * gpu.aspect * gpu.domain
        y = ((np.arange(72) + 0.5) / 72 * 2 - 1) * gpu.domain
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=-1)
        tools, pairs = tool_uniforms(gpu.source.frame(0.37), engine.config["stir_radius"])
        expected = velocity(
            points,
            tools,
            pairs,
            aspect=gpu.aspect,
            stir_radius=engine.config["stir_radius"],
            flow_strength=0.7,
            pair_swirl=0.2,
            domain_scale=1,
            carrier_velocity=(0.3, 0.1),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
        outside = (np.abs(xx) >= gpu.aspect) | (np.abs(yy) >= 1)
        np.testing.assert_array_equal(actual[outside], 0)
        self.assertGreater(float(np.linalg.norm(actual[~outside], axis=-1).max()), 0)

    def test_confined_flow_keeps_each_scattered_pigment_inside_the_visible_painting(self):
        engine = self.engine(
            count=5,
            resolution=[192, 144],
            steps=120,
            initial_pattern="scattered",
            underpaint_strength=0,
            settling_scale=0,
            flow_domain_scale=1,
            flow_strength=1.1,
            pair_swirl=0.9,
            carrier_velocity=[2, 0.2],
        )
        engine.advance_to(engine.steps)
        pigment = engine.snapshot()["pigment"][..., :5]
        gpu = engine._gpu
        x = ((np.arange(gpu.width) + 0.5) / gpu.width * 2 - 1) * gpu.aspect * gpu.domain
        y = ((np.arange(gpu.height) + 0.5) / gpu.height * 2 - 1) * gpu.domain
        outside = (np.abs(x)[None, :] >= gpu.aspect) | (np.abs(y)[:, None] >= 1)
        total = pigment.sum(axis=(0, 1))
        self.assertTrue(np.all(total > 0))
        self.assertTrue(np.all(pigment[outside].sum(axis=0) / total < 1e-5))

    def test_scattered_initial_state_ignores_body_mixtures_and_palette_colors(self):
        from tools.estuary.test_engine import SourceFixture

        a = self.engine(count=5, initial_pattern="scattered", underpaint_strength=0)
        initial, layout = a.snapshot(), a.layout
        palette, config = a.palette, a.config
        a.close()
        palette["pigments_srgb"] = [[0.5, 0.1, 0.2]] * 6
        palette["body_mixtures"] = [[1, 0, 0, 0, 0, 0]] * 3
        b = Engine(SourceFixture(), config, palette, [])
        self.addCleanup(b.close)
        self.assertEqual(layout, b.layout)
        np.testing.assert_array_equal(initial["mobile"], b.snapshot()["mobile"])

    def test_legacy_initialization_ignores_new_edge_control(self):
        for pattern in ("pools", "strata"):
            a = self.engine(initial_pattern=pattern, initial_edge_width=0.001)
            initial = a.snapshot()
            a.close()
            b = self.engine(initial_pattern=pattern, initial_edge_width=0.25)
            self.assertIsNone(b.layout)
            for key, value in b.snapshot().items():
                np.testing.assert_array_equal(value, initial[key])
            b.close()

    def test_mass_reduction_matches_cpu_float64_integral(self):
        engine = self.engine(
            count=5,
            resolution=[256, 192],
            initial_pattern="scattered",
            underpaint_strength=0,
            settling_scale=0,
            mass_budget_interval_steps=12,
        )
        gpu = engine._gpu
        state = np.random.default_rng(481).uniform(0, 0.4, (192, 256, 6)).astype("f4")
        state[..., -1] = 0
        with gpu.ctx:
            for index, block in enumerate(gpu.blocks):
                packed = np.zeros((192, 256, 4), dtype="f4")
                channels = min(4, 6 - index * 4)
                packed[..., :channels] = state[..., index * 4 : index * 4 + channels]
                block[0].write(packed.tobytes())
            actual = gpu._mass_amounts()
        expected = state.sum(axis=(0, 1), dtype="f8") * (2 * gpu.domain / gpu.height) ** 2
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)

    def test_mass_restoration_preserves_empty_support_and_original_channel_budgets(self):
        engine = self.engine(
            count=5,
            initial_pattern="scattered",
            underpaint_strength=0,
            settling_scale=0,
            mass_budget_interval_steps=12,
        )
        initial = engine.snapshot()["pigment"]
        inflated = initial * np.array([1.1, 0.9, 1.05, 1.08, 0.95, 1], dtype="f4")
        gpu = engine._gpu
        with gpu.ctx:
            for index, block in enumerate(gpu.blocks):
                packed = np.zeros((72, 96, 4), dtype="f4")
                channels = min(4, 6 - index * 4)
                packed[..., :channels] = inflated[..., index * 4 : index * 4 + channels]
                block[0].write(packed.tobytes())
        engine.advance_to(12)
        result = engine.snapshot()["pigment"]
        np.testing.assert_allclose(result, initial, rtol=3e-6, atol=1e-8)
        np.testing.assert_array_equal(result[initial.sum(axis=-1) == 0], 0)
        np.testing.assert_array_equal(result[..., -1], 0)
        self.assertTrue(np.all(result >= 0))
        report = engine.mass_budget_report
        self.assertEqual([r["step"] for r in report["corrections"]], [12])
        np.testing.assert_allclose(
            report["corrections"][0]["mass_after"], report["initial_mass"], rtol=5e-6
        )

    def test_mass_schedule_forced_endpoint_and_results_do_not_depend_on_capture_cadence(self):
        from .mass_budget import validate_report

        config = {
            "count": 5,
            "steps": 61,
            "initial_pattern": "scattered",
            "underpaint_strength": 0,
            "settling_scale": 0,
            "mass_budget_interval_steps": 12,
            "diffusion_coefficient": 0.0008,
            "flow_strength": 0.3,
            "carrier_velocity": [0.2, 0.1],
        }
        a = self.engine(**config)
        a.advance_to(17)
        a.snapshot((24, 18))
        a.advance_to(43)
        a.snapshot()
        a.advance_to(61)
        expected, ledger, diagnostics = a.snapshot(), a.mass_budget_report, a.diagnostics
        a.advance_to(61)
        self.assertEqual(ledger, a.mass_budget_report)
        a.close()
        b = self.engine(**config)
        b.advance_to(61)
        for key, value in b.snapshot().items():
            np.testing.assert_array_equal(value, expected[key])
        self.assertEqual(ledger, b.mass_budget_report)
        self.assertEqual(diagnostics, b.diagnostics)
        self.assertEqual([r["step"] for r in ledger["corrections"]], [12, 24, 36, 48, 60, 61])
        validate_report(
            ledger, {"simulation": b.config, "chromatic_count": 5}, b.snapshot(), layout=b.layout
        )
        ledger["corrections"][0]["factors"][0] = 0
        self.assertGreater(b.mass_budget_report["corrections"][0]["factors"][0], 0)

    def test_disabled_mass_budget_has_no_ledger(self):
        engine = self.engine()
        self.assertIsNone(engine.mass_budget_report)


if __name__ == "__main__":
    unittest.main()
