"""Initial-shape integration, archive binding and real GPU material budgets."""

from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary.source import Source
from tools.estuary.test_engine import SourceFixture
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_studio.common import read, write

from . import run as runner
from . import test_run as fixtures
from .engine import Engine, _shaped_initial_state, validate_config
from .initial_composition import SETUPS, plan_layout, rasterize
from .laminate import layer_fractions
from .mass_budget import (
    RELATIVE_TOLERANCE,
    VERSION,
    correction_steps,
    initial_material_mass,
    pigment_mass,
    validate_report,
)
from .palette import generate_palette
from .test_interaction_archive import InteractionEngineFixture


def composition_config(setup="random-circles"):
    return {
        "setup": setup,
        "target_mass": [0.03, 0.012, 0.006],
        "reference_radii": [0.22, 0.2, 0.18],
    }


def simulation_config(setup="random-circles", **overrides):
    return validate_config(
        {
            "resolution": [384, 288],
            "steps": 20,
            "material_model": "laminate",
            "initial_pattern": "shaped",
            "initial_composition": composition_config(setup),
            "deposition": 0,
            "settling_scale": 0,
            "underpaint_strength": 0,
            "underpaint_release": 0,
            "burial_rate": 0,
            "initial_pigment_weights": None,
            "carrier_velocity": [0, 0],
            "flow_strength": 0.7,
            "mass_budget_interval_steps": 7,
            "diffusion_coefficient": 0.00002,
            "interaction": {"advection": "maccormack"},
            **overrides,
        }
    )


class CompositionIntegrationTests(unittest.TestCase):
    def test_old_normalized_configs_are_exact_when_composition_is_absent_or_null(self):
        for raw in ({}, {"initial_pattern": "scattered", "underpaint_strength": 0}):
            original = validate_config(raw)
            self.assertNotIn("initial_composition", original)
            self.assertEqual(validate_config({**raw, "initial_composition": None}), original)
        with self.assertRaisesRegex(ValueError, "only valid"):
            validate_config({"initial_composition": composition_config()})

    def test_shapes_require_explicit_budgets_three_pigments_and_source_free_laminate(self):
        settings = simulation_config()
        for key, value in (
            ("initial_composition", None),
            ("material_model", "legacy"),
            ("deposition", 0.01),
            ("settling_scale", 1),
            ("underpaint_strength", 0.01),
            ("underpaint_release", 1),
            ("burial_rate", 1),
            ("initial_pigment_weights", [2.2, 0.9, 0.5]),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**settings, key: value})
        raw = fixtures.small_recipe()
        raw["simulation"] = settings
        raw["render"]["capture_resolution"] = [384, 288]
        for count in (1, 2, 5):
            raw["chromatic_count"] = count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "three chromatic"):
                runner.validate_recipe(raw)
            with self.assertRaisesRegex(ValueError, "exactly three"):
                Engine(SourceFixture(), settings, generate_palette("0x1234", count), [])

    def test_all_six_rasters_reach_declared_integrals_without_initial_load_or_chalk(self):
        for setup in SETUPS:
            settings = simulation_config(setup)
            layout = plan_layout(
                "0x1234",
                3,
                4 / 3,
                settings["initial_composition"],
                source=SourceFixture() if setup == "body-wedges" else None,
            )
            expected = rasterize(layout, settings["resolution"], settings["domain_scale"])
            with self.subTest(setup=setup):
                first = _shaped_initial_state(layout, {**settings, "initial_load": 0})
                second = _shaped_initial_state(layout, {**settings, "initial_load": 10})
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(first[..., :3], expected)
                self.assertFalse(first[..., 3].any())
                actual = pigment_mass(first, settings["domain_scale"])
                np.testing.assert_allclose(
                    actual,
                    [*settings["initial_composition"]["target_mass"], 0],
                    rtol=5e-7,
                    atol=1e-12,
                )
                np.testing.assert_array_equal(actual, initial_material_mass(layout, settings))

    def test_layout_resolution_has_no_engagement_pilot_and_only_wedges_read_time_zero(self):
        class InitialOnly(SourceFixture):
            def __init__(self):
                self.times = []

            def frame(self, fraction):
                self.times.append(fraction)
                if fraction != 0:
                    raise AssertionError("Initial geometry must not sample future trajectories")
                return super().frame(fraction)

        with patch(
            "tools.estuary_confluence.participation_layout.plan_engaged_layout",
            side_effect=AssertionError("No engagement pilot for shapes"),
        ):
            for setup in SETUPS:
                source = InitialOnly()
                settings = simulation_config(setup)
                layout = runner.resolved_layout(
                    {"simulation": settings, "chromatic_count": 3}, "0x1234", source
                )
                self.assertEqual(layout["setup"], setup)
                self.assertEqual(source.times, [0.0] if setup == "body-wedges" else [])
        with self.assertRaisesRegex(ValueError, "initial source positions"):
            runner.resolved_layout(
                {"simulation": simulation_config("body-wedges"), "chromatic_count": 3}, "0x1234"
            )

    def test_material_budget_rejects_declared_amount_drift_without_assuming_pools(self):
        settings = simulation_config()
        layout = plan_layout("0x1234", 3, 4 / 3, settings["initial_composition"])
        self.assertNotIn("pools", layout)
        changed = copy.deepcopy(settings)
        changed["initial_composition"]["target_mass"][0] *= 1.01
        with self.assertRaisesRegex(ValueError, "declared pigment targets"):
            initial_material_mass(layout, changed)
        changed = {**settings, "initial_pigment_weights": [1, 1, 1]}
        with self.assertRaisesRegex(ValueError, "weighted again"):
            initial_material_mass(layout, changed)


class CompositionEngineFixture(InteractionEngineFixture):
    def snapshot(self, resolution=None):
        fields = super().snapshot(resolution)
        settings = {
            **self.config,
            "resolution": self.config["resolution"] if resolution is None else resolution,
        }
        initial = _shaped_initial_state(self.layout, settings)
        fields["mobile"] = initial * layer_fractions(self.palette, initial.shape[-1])
        fields["underpaint"] = initial - fields["mobile"]
        fields["deposit"] = np.zeros_like(initial)
        fields["pigment"] = fields["mobile"] + fields["underpaint"]
        return fields

    @property
    def mass_budget_report(self):
        target = initial_material_mass(self.layout, self.config).tolist()
        return {
            "version": VERSION,
            "interval_steps": self.config["mass_budget_interval_steps"],
            "initial_mass": target,
            "corrections": [
                {"step": step, "mass_before": target, "factors": [1.0] * 4, "mass_after": target}
                for step in correction_steps(self.steps, self.config["mass_budget_interval_steps"])
            ],
        }


class CompositionArchiveTests(unittest.TestCase):
    source_info = staticmethod(Source.read)
    engaged_layout = staticmethod(fixtures.PipelineTests.engaged_layout)
    movie = staticmethod(fixtures.PipelineTests.movie)
    rewrite_artifact_hash = fixtures.PipelineTests.rewrite_artifact_hash
    rewrite_request = fixtures.PipelineTests.rewrite_request

    def setUp(self):
        fixtures.PipelineTests.setUp(self)
        write_orbit(self.source, orbit_points(), seed="0xbc53af1cd380")
        self.raw = fixtures.small_recipe()
        self.raw["palette_mode"] = "composed"
        self.raw["simulation"] = simulation_config(
            resolution=[128, 96], steps=10, diffusion_coefficient=0
        )
        write(self.recipe, self.raw)
        self.patches.enter_context(
            patch("tools.estuary_confluence.engine.Engine", CompositionEngineFixture)
        )

    def test_archive_regenerates_geometry_and_certifies_initial_and_final_amounts(self):
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["layout"]["setup"], "random-circles")
        self.assertNotIn("pools", request["layout"])
        self.assertIn("layout.json", receipt["artifacts"])
        self.assertTrue(
            all(
                f"{look}/initial.png" in receipt["artifacts"] for look in request["recipe"]["looks"]
            )
        )
        report = read(self.args.output / "mass-budget.json")
        np.testing.assert_allclose(report["initial_mass"], [0.03, 0.012, 0.006, 0], rtol=5e-7)
        with np.load(self.args.output / "final.npz", allow_pickle=False) as arrays:
            self.assertEqual(len(arrays.files), 14)
            np.testing.assert_allclose(
                pigment_mass(arrays["pigment"], 1.6),
                report["initial_mass"],
                rtol=RELATIVE_TOLERANCE,
                atol=1e-12,
            )

    def test_rehashed_geometry_cannot_replace_the_seeded_composition(self):
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request["layout"]["primitives"][0]["points"][0][0] += 0.03
        write(self.args.output / "layout.json", request["layout"])
        self.rewrite_artifact_hash("layout.json")
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "Seeded starting layout differs"):
            runner.verify_run(self.args.output)

    def test_rehashed_budget_report_cannot_redefine_the_declared_starting_amounts(self):
        runner.run(self.args)
        report = read(self.args.output / "mass-budget.json")
        report["initial_mass"][0] *= 1.01
        write(self.args.output / "mass-budget.json", report)
        self.rewrite_artifact_hash("mass-budget.json")
        with self.assertRaisesRegex(ValueError, "regenerated initial composition"):
            runner.verify_run(self.args.output)

    def test_body_wedges_regenerate_from_the_archived_source_at_zero(self):
        self.raw["simulation"]["initial_composition"]["setup"] = "body-wedges"
        write(self.recipe, self.raw)
        runner.run(self.args)
        request, _ = runner.verify_run(self.args.output)
        source = Source.read(
            self.args.output / "inputs/source.orbit",
            aspect=4 / 3,
            **request["recipe"]["projection"],
        )
        expected = runner.resolved_layout(request["recipe"], source.seed, source)
        self.assertTrue(runner.equivalent_design(expected, request["layout"]))
        self.assertTrue(all(p["source_fraction"] == 0 for p in request["layout"]["primitives"]))

    def test_still_and_different_film_cadence_keep_the_same_full_material_and_mass_report(self):
        runner.run(self.args)
        _, still = runner.verify_run(self.args.output)
        report = read(self.args.output / "mass-budget.json")
        self.raw["render"].update(formation_frames=11, orbit_frames=4, hold_frames=3)
        write(self.recipe, self.raw)
        self.args.output, self.args.still_only = self.folder / "film", False
        runner.run(self.args)
        _, film = runner.verify_run(self.args.output)
        self.assertEqual(still["physical_state_sha256"], film["physical_state_sha256"])
        self.assertEqual(report, read(self.args.output / "mass-budget.json"))


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class CompositionGPUTests(unittest.TestCase):
    def create(self, setup, source=None, **overrides):
        engine = Engine(
            SourceFixture() if source is None else source,
            simulation_config(setup, **overrides),
            generate_palette("0x1234", 3, mode="composed"),
            [],
        )
        self.addCleanup(engine.close)
        return engine

    def test_all_six_native_initial_states_accumulate_components_and_split_existing_mass(self):
        for setup in SETUPS:
            engine = self.create(setup)
            fields = engine.snapshot()
            raster = rasterize(
                engine.layout, engine.config["resolution"], engine.config["domain_scale"]
            )
            with self.subTest(setup=setup):
                np.testing.assert_allclose(fields["pigment"][..., :3], raster, rtol=2e-7, atol=1e-9)
                np.testing.assert_allclose(
                    fields["mobile"][..., :3],
                    raster * np.asarray(engine.palette["layer_fractions"][:3], dtype="f4"),
                    rtol=2e-7,
                    atol=1e-9,
                )
                self.assertFalse(fields["pigment"][..., 3].any())
                self.assertFalse(fields["deposit"].any())
                np.testing.assert_allclose(
                    pigment_mass(fields["pigment"], engine.config["domain_scale"]),
                    [0.03, 0.012, 0.006, 0],
                    rtol=RELATIVE_TOLERANCE,
                    atol=1e-12,
                )
            engine.close()

    def test_each_setup_restores_all_color_budgets_after_real_transport(self):
        for setup in SETUPS:
            engine = self.create(setup)
            engine.advance_to(engine.steps)
            validate_report(
                engine.mass_budget_report,
                {"simulation": engine.config, "chromatic_count": 3},
                engine.snapshot(),
                layout=engine.layout,
            )
            engine.close()

    def test_first_five_initialize_without_reading_any_source_frame(self):
        class NoInitialSource(SourceFixture):
            def frame(self, fraction):
                raise AssertionError("Seeded shapes must not sample source during initialization")

        for setup in SETUPS[:-1]:
            engine = self.create(setup, source=NoInitialSource())
            self.assertGreater(float(engine.snapshot()["pigment"].sum()), 0)
            engine.close()

    def test_snapshots_and_capture_cadence_do_not_change_shaped_evolution(self):
        a = self.create("scattered-commas")
        a.advance_to(a.steps)
        expected, report = a.snapshot(), a.mass_budget_report
        a.close()
        b = self.create("scattered-commas")
        for step in (0, 3, 7, 11, 16, 20):
            b.advance_to(step)
            b.snapshot()
        self.assertEqual(runner.field_digest(expected), runner.field_digest(b.snapshot()))
        self.assertEqual(report, b.mass_budget_report)
        b.close()


if __name__ == "__main__":
    unittest.main()
