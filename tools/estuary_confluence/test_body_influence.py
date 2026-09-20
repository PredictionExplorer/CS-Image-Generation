"""Paint-forcing selection, source isolation and unchanged initialization contracts."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path

import numpy as np

from tools.estuary.engine import Engine as Transport
from tools.estuary.flow_reference import velocity as reference_velocity
from tools.estuary.source import Source
from tools.estuary.test_engine import SourceFixture
from tools.estuary.test_source import orbit_points, write_orbit

from .body_influence import (
    PAIRS,
    VERSION,
    arc_travel,
    forcing_uniforms,
    initialization_config,
    movement_segments,
    normalize_bodies,
    validate_config,
    validate_event_eligibility,
)
from .engine import Engine
from .engine import validate_config as simulation_config
from .mass_budget import pigment_mass
from .pair_strain import pair_strain_uniforms
from .palette import generate_palette
from .participation_layout import conditioned_uniforms
from .run import field_digest

SUBSETS = tuple(combinations(range(3), 1)) + tuple(combinations(range(3), 2))


def selection(bodies):
    return {"version": VERSION, "bodies": list(bodies)}


def settings(bodies=None, **overrides):
    result = {
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
        "diffusion_coefficient": 0.00002,
        "interaction": {"advection": "maccormack"},
        **overrides,
    }
    if bodies is not None:
        result["body_influence"] = selection(bodies)
    return result


class BodyInfluenceTests(unittest.TestCase):
    def test_selection_is_strict_and_full_selection_is_exact_legacy_omission(self):
        baseline = simulation_config(settings())
        for full in (None, selection((0, 1, 2)), selection((2, 0, 1))):
            self.assertEqual(simulation_config({**settings(), "body_influence": full}), baseline)
        self.assertNotIn("body_influence", baseline)
        self.assertEqual(validate_config({"bodies": [2, 0]}), selection((0, 2)))
        for invalid in (
            False,
            [],
            {},
            {"bodies": None},
            {"bodies": []},
            {"bodies": [0, 0]},
            {"bodies": [True]},
            {"bodies": [3]},
            {"bodies": [-1]},
            {"bodies": [1.0]},
            {"bodies": [0], "version": "future"},
            {"bodies": [0], "gain": 2},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(invalid)
        self.assertIsNone(normalize_bodies((2, 1, 0)))

    def test_every_subset_keeps_active_amplitudes_exact_and_zeroes_entire_disabled_rows(self):
        frame = SourceFixture().frame(0.37)
        tools, pairs = conditioned_uniforms(frame, 0.22)
        strain = pair_strain_uniforms(frame, 0.22)
        original = copy.deepcopy(frame)
        for bodies in SUBSETS:
            actual = forcing_uniforms(frame, 0.22, selection(bodies), include_strain=True)
            pair_mask = [a in bodies and b in bodies for a, b in PAIRS]
            with self.subTest(bodies=bodies):
                np.testing.assert_array_equal(actual[0][list(bodies)], tools[list(bodies)])
                for index in set(range(3)) - set(bodies):
                    np.testing.assert_array_equal(actual[0][index], np.zeros(4))
                for value, expected in zip(actual[1:], (pairs, strain), strict=True):
                    np.testing.assert_array_equal(value[pair_mask], expected[pair_mask])
                    self.assertFalse(value[np.logical_not(pair_mask)].any())
        for name in ("positions", "velocities", "pair_distances", "arc_lengths"):
            np.testing.assert_array_equal(getattr(frame, name), getattr(original, name))

    def test_inactive_nonfinite_data_never_enters_forcing_segments_or_arc_arithmetic(self):
        baseline = SourceFixture().frame(0.3)
        changed = copy.deepcopy(baseline)
        changed.positions[2] = [np.nan, np.inf]
        changed.velocities[2] = [-np.inf, np.nan]
        changed.pair_distances[1:] = np.nan
        config = selection((0, 1))
        with np.errstate(all="raise"):
            actual = forcing_uniforms(changed, 0.22, config, include_strain=True)
            expected = forcing_uniforms(baseline, 0.22, config, include_strain=True)
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(
                arc_travel([0, 2, np.inf], [0.5, 3, np.nan], config), [0.5, 1, 0]
            )
            segments = movement_segments(changed.positions, changed.positions, config)
            self.assertFalse(segments[2].any())
        changed.positions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            forcing_uniforms(changed, 0.22, config)

    def test_reduced_encounters_require_original_pair_indices_with_both_bodies_active(self):
        validate_event_eligibility([{"pair": [2, 0]}], selection((0, 2)))
        for event in ({}, {"pair": [0, 1]}, {"pair": [0, 2]}, {"pair": [True, 0]}):
            with self.subTest(event=event), self.assertRaises(ValueError):
                validate_event_eligibility([event], selection((0, 2)))
        with self.assertRaises(ValueError):
            validate_event_eligibility([{"pair": [0, 1]}], selection((0,)))
        validate_event_eligibility([{}], None)  # Legacy custom schedules remain supported.

    def test_initializer_receives_all_reference_settings_without_the_mask(self):
        baseline = simulation_config(settings())
        for bodies in SUBSETS:
            selected = simulation_config(settings(bodies))
            self.assertEqual(initialization_config(selected), baseline)
            self.assertIn("body_influence", selected)
        self.assertIs(initialization_config(baseline), baseline)

    def test_both_scheduler_limits_ignore_inactive_travel_and_keep_default_call_order(self):
        class ArcSource(SourceFixture):
            def __init__(self, inactive_rate):
                self.inactive_rate, self.calls = inactive_rate, []

            def frame(self, fraction):
                self.calls.append(fraction)
                frame = super().frame(fraction)
                frame.arc_lengths = np.array([0.01, self.inactive_rate, 0.02]) * fraction
                return frame

        class Scheduler(Transport):
            def __init__(self, source, config):
                self.ctx, self.source, self.config = nullcontext(), source, config
                self.step, self.steps, self.height, self.domain = 0, 4, 96, 1.6
                self.recipe = {"simulation": {"brush_radius": 0.035}}
                self.maximum_courant, self.intervals = 0.0, []

            def _source_travel(self, start, end):
                if self.config is None:
                    return super()._source_travel(start, end)
                after, before = self.source.frame(end), self.source.frame(start)
                return arc_travel(before.arc_lengths, after.arc_lengths, self.config)

            def _flow(self, fraction):
                return 0.0

            def _transport(self, start, end):
                self.intervals.append((start, end))
                if len(self.intervals) > 4:
                    raise AssertionError("Inactive travel introduced extra subdivisions")

        results = []
        for rate in (1.0, 1e12):
            scheduler = Scheduler(ArcSource(rate), selection((0, 2)))
            scheduler.advance_to(4)
            results.append(scheduler.intervals)
        self.assertEqual(results[0], [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)])
        self.assertEqual(results[0], results[1])
        default = Scheduler(ArcSource(0.01), None)
        default._source_travel(0.2, 0.4)
        self.assertEqual(default.source.calls, [0.4, 0.2])


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class BodyInfluenceGPUTests(unittest.TestCase):
    def engine(self, bodies=None, source=None, palette=None, **overrides):
        engine = Engine(
            SourceFixture() if source is None else source,
            settings(bodies, **overrides),
            generate_palette("0xbc53af1cd380", 3, mode="composed") if palette is None else palette,
            [],
        )
        self.addCleanup(engine.close)
        return engine

    def test_all_seven_selections_keep_genuine_engaged_layout_and_all_initial_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.orbit"
            write_orbit(path, orbit_points(), seed="0xbc53af1cd380")
            source = Source.read(path, aspect=4 / 3)
            baseline = self.engine(source=source, initial_pattern="engaged")
            initial, layout, palette = baseline.snapshot(), baseline.layout, baseline.palette
            baseline.close()
            for bodies in (*SUBSETS, (2, 0, 1)):
                actual = self.engine(bodies, source=source, initial_pattern="engaged")
                with self.subTest(bodies=bodies):
                    self.assertEqual(actual.layout, layout)
                    self.assertEqual(actual.palette, palette)
                    self.assertEqual(field_digest(actual.snapshot()), field_digest(initial))
                actual.close()

    def test_masked_gpu_flow_matches_original_active_dipoles_and_eligible_pairs(self):
        for bodies in SUBSETS:
            engine = self.engine(bodies)
            frame = SourceFixture().frame(0.37)
            tools, pairs, strains = forcing_uniforms(
                frame, engine.config["stir_radius"], selection(bodies), include_strain=True
            )
            gpu = engine._gpu
            with gpu.ctx:
                gpu._flow(0.37)
                actual = np.frombuffer(gpu.velocity.read(), dtype="f4").reshape(
                    gpu.height, gpu.width, 2
                )
            x = ((np.arange(gpu.width) + 0.5) / gpu.width * 2 - 1) * gpu.aspect * gpu.domain
            y = ((np.arange(gpu.height) + 0.5) / gpu.height * 2 - 1) * gpu.domain
            points = np.stack(np.meshgrid(x, y), -1)
            expected = reference_velocity(
                points,
                tools,
                pairs,
                aspect=gpu.aspect,
                stir_radius=engine.config["stir_radius"],
                flow_strength=engine.config["flow_strength"],
                pair_swirl=engine.config["pair_swirl"],
                carrier_velocity=[0, 0],
                domain_scale=1.0,
                pair_strain=engine.config["pair_strain"],
                strains=strains,
            )
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
            engine.close()

    def test_inactive_motion_cannot_change_fields_substeps_or_output_cadence(self):
        class DifferentInactiveBody(SourceFixture):
            def frame(self, fraction):
                frame = super().frame(fraction)
                frame.positions[1] = [-0.9 + fraction * 0.2, 0.8]
                frame.velocities[1] = [30, -10]
                frame.arc_lengths[1] = fraction * 1e8
                frame.pair_distances[:2] = [20 + fraction, 50 - fraction]
                return frame

        first = self.engine((0, 2))
        first.advance_to(first.steps)
        expected, diagnostics, budget = (
            first.snapshot(),
            first.diagnostics,
            first.mass_budget_report,
        )
        first.close()
        second = self.engine((0, 2), source=DifferentInactiveBody())
        for step in (0, 3, 7, 12, 20):
            second.advance_to(step)
            second.snapshot()
        self.assertEqual(field_digest(second.snapshot()), field_digest(expected))
        self.assertEqual(second.diagnostics, diagnostics)
        self.assertEqual(second.mass_budget_report, budget)
        second.close()

    def test_inactive_brushes_add_neither_pigment_nor_wetting_in_legacy_material(self):
        palette = generate_palette("0xbc53af1cd380", 3, mode="composed")
        palette["body_mixtures"] = np.eye(3, 4).tolist()
        palette["body_weights"] = [1, 1, 1]
        controls = {
            "material_model": "legacy",
            "initial_pattern": "pools",
            "initial_load": 0,
            "interaction": None,
            "deposition": 0.2,
            "mass_budget_interval_steps": 0,
            "diffusion_coefficient": 0,
            "flow_strength": 0,
            "pair_swirl": 0,
            "pair_strain": 0,
            "drying": 2,
            "wetting": 2,
            "fade": 0,
            "brush_radius": 0.08,
        }
        engine = self.engine((0,), palette=palette, **controls)
        engine.advance_to(engine.steps)
        active = engine.snapshot()
        self.assertGreater(float(active["pigment"][..., 0].sum()), 0)
        self.assertFalse(active["pigment"][..., 1:].any())
        engine.close()
        quiet = self.engine((0,), palette=palette, **{**controls, "wetting": 0})
        quiet.advance_to(quiet.steps)
        difference = active["wetness"] - quiet.snapshot()["wetness"]
        self.assertGreater(float(difference.max()), 0.01)
        for point in SourceFixture().frame(0.5).positions[1:]:
            col = round((point[0] / (4 / 3 * 1.6) + 1) * 96 / 2 - 0.5)
            row = round((point[1] / 1.6 + 1) * 72 / 2 - 0.5)
            self.assertEqual(float(difference[row, col]), 0)
        quiet.close()

    def test_explicit_all_body_selection_retains_exact_final_state_and_budgets(self):
        first = self.engine()
        first.advance_to(first.steps)
        expected, budget = first.snapshot(), first.mass_budget_report
        first.close()
        second = self.engine((2, 0, 1))
        second.advance_to(second.steps)
        self.assertEqual(field_digest(second.snapshot()), field_digest(expected))
        self.assertEqual(second.mass_budget_report, budget)
        np.testing.assert_array_equal(
            pigment_mass(second.snapshot()["pigment"], 1.6), pigment_mass(expected["pigment"], 1.6)
        )
        second.close()


if __name__ == "__main__":
    unittest.main()
