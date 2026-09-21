"""Native initialization, layer allocation and complete-source archive binding."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_studio.common import read, write

from . import run as runner
from . import test_run as fixtures
from .body_marker_studies import released_recipe
from .choreography import effective_layer_fractions, plan_layout, rasterize
from .composition_studies import references
from .engine import Engine, _budgeted_initial_state, validate_config
from .mass_budget import initial_material_mass, pigment_mass
from .palette import generate_palette
from .participation_layout import plan_engaged_layout
from .test_initial_composition_integration import CompositionEngineFixture

SEED = "0xb7f327f9f722"
_SOURCE_READ = Source.read
_ENGAGED_LAYOUT = plan_engaged_layout


def simulation(setup="active-pools", **overrides):
    _, recipe = released_recipe(SEED)
    reference = next(row for row in references()["cases"] if row["seed"] == SEED)
    controls = {
        "setup": setup,
        "target_mass": reference["target_mass"],
        "reference_radii": reference["reference_radii"],
    }
    recipe["simulation"].update(
        resolution=[384, 288],
        steps=20,
        mass_budget_interval_steps=5,
        initial_pattern="choreographed",
        initial_pigment_weights=None,
        initial_choreography=controls,
    )
    recipe["simulation"].update(overrides)
    return validate_config(recipe["simulation"])


class ChoreographyIntegrationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "source.orbit"
        write_orbit(self.path, orbit_points(), seed=SEED)
        self.source = _SOURCE_READ(self.path, aspect=4 / 3)
        self.palette = generate_palette(SEED, 3, mode="composed")

    def test_optional_initializer_rejects_conflicting_physics_and_preserves_off_config(self):
        _, rc1 = released_recipe(SEED)
        baseline = validate_config(rc1["simulation"])
        self.assertEqual(validate_config({**baseline, "initial_choreography": None}), baseline)
        self.assertNotIn("initial_choreography", baseline)
        base = simulation()
        for key, value in (
            ("initial_choreography", None),
            ("initial_pattern", "engaged"),
            ("initial_pigment_weights", [1, 1, 1]),
            ("material_model", "legacy"),
            ("deposition", 0.1),
            ("body_influence", {"bodies": [0, 1]}),
            ("rheology", {}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**base, key: value})
        self.assertEqual(validate_config({**base, "body_influence": {"bodies": [0, 1, 2]}}), base)
        controls = copy.deepcopy(base["initial_choreography"])
        controls["mobility_bias"] = [0, 0, 0]
        self.assertEqual(validate_config({**base, "initial_choreography": controls}), base)

    def test_native_amounts_ignore_legacy_loading_and_match_independent_budget(self):
        settings = simulation()
        layout = plan_layout(self.source, settings, self.palette)
        first = _budgeted_initial_state(layout, {**settings, "initial_load": 0})
        second = _budgeted_initial_state(layout, {**settings, "initial_load": 10})
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[..., :3], rasterize(layout, [384, 288], 1.6))
        self.assertFalse(first[..., 3].any())
        np.testing.assert_array_equal(
            pigment_mass(first, 1.6), initial_material_mass(layout, settings)
        )
        np.testing.assert_allclose(
            pigment_mass(first, 1.6),
            [*settings["initial_choreography"]["target_mass"], 0],
            rtol=5e-7,
            atol=1e-12,
        )

    def test_layer_allocation_changes_only_split_and_preserves_geometry_and_palette(self):
        base = simulation()
        changed = copy.deepcopy(base)
        changed["initial_choreography"]["mobility_bias"] = [-0.12, 0.12, 0]
        before_palette = copy.deepcopy(self.palette)
        first = plan_layout(self.source, base, self.palette)
        second = plan_layout(self.source, changed, self.palette)
        self.assertEqual(first["primitives"], second["primitives"])
        self.assertEqual(first["pilot"], second["pilot"])
        self.assertEqual(self.palette, before_palette)
        np.testing.assert_array_equal(
            _budgeted_initial_state(first, base), _budgeted_initial_state(second, changed)
        )
        expected = np.asarray(self.palette["layer_fractions"], "f4")
        expected[:3] = expected[:3].astype("f8") + np.array([-0.12, 0.12, 0])
        np.testing.assert_array_equal(
            effective_layer_fractions(self.palette, changed["initial_choreography"]), expected
        )
        bad = {**base["initial_choreography"], "mobility_bias": [0.15, 0, 0]}
        with self.assertRaisesRegex(ValueError, "within"):
            effective_layer_fractions(self.palette, bad)


class ChoreographyEngineFixture(CompositionEngineFixture):
    def __init__(self, source, config, palette, events):
        self.source, self.config, self.palette, self.events = source, config, palette, events
        self.step, self.steps = 0, config["steps"]
        self.visited, self.snapshots, self.closed = [], [], False
        self.metadata = {"renderer": "CPU choreography archive fixture"}
        self.layout = plan_layout(source, config, palette)
        fixtures.FakeEngine.instances.append(self)

    def snapshot(self, resolution=None):
        from .test_interaction_archive import InteractionEngineFixture

        fields = InteractionEngineFixture.snapshot(self, resolution)
        settings = {
            **self.config,
            "resolution": self.config["resolution"] if resolution is None else resolution,
        }
        initial = _budgeted_initial_state(self.layout, settings)
        fractions = effective_layer_fractions(self.palette, self.config["initial_choreography"])
        fields["mobile"] = initial * fractions
        fields["underpaint"] = initial - fields["mobile"]
        fields["deposit"] = np.zeros_like(initial)
        fields["pigment"] = fields["mobile"] + fields["underpaint"]
        return fields


class ChoreographyArchiveTests(unittest.TestCase):
    source_info = staticmethod(_SOURCE_READ)
    engaged_layout = staticmethod(_ENGAGED_LAYOUT)
    movie = staticmethod(fixtures.PipelineTests.movie)
    rewrite_artifact_hash = fixtures.PipelineTests.rewrite_artifact_hash
    rewrite_request = fixtures.PipelineTests.rewrite_request

    def setUp(self):
        fixtures.PipelineTests.setUp(self)
        write_orbit(self.source, orbit_points(), seed=SEED)
        self.raw = fixtures.small_recipe()
        self.raw["palette_mode"] = "composed"
        self.raw["simulation"] = simulation()
        self.raw["render"]["capture_resolution"] = [384, 288]
        write(self.recipe, self.raw)
        self.patches.enter_context(
            patch("tools.estuary_confluence.engine.Engine", ChoreographyEngineFixture)
        )

    def test_complete_source_layout_and_all_native_fields_are_archived(self):
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["layout"]["source_sha256"], request["source"]["sha256"])
        self.assertEqual(request["layout"]["source_projection"], request["source"]["projection"])
        self.assertEqual(request["layout"]["count"], 3)
        self.assertEqual(receipt["source_fraction"], 1)
        self.assertIn("layout.json", receipt["artifacts"])
        with np.load(self.args.output / "final.npz", allow_pickle=False) as archive:
            self.assertEqual(len(archive.files), 14)
        report = read(self.args.output / "mass-budget.json")
        np.testing.assert_allclose(
            report["initial_mass"], [*request["layout"]["target_mass"], 0], rtol=5e-7
        )

    def test_rehashed_geometry_and_layer_allocation_cannot_replace_the_source_design(self):
        runner.run(self.args)
        original = read(self.args.output / "request.json")
        for change in ("geometry", "allocation"):
            request = copy.deepcopy(original)
            if change == "geometry":
                request["layout"]["primitives"][0]["points"][0][0] += 0.02
            else:
                request["layout"]["effective_layer_fractions"][0] = 0.4
            write(self.args.output / "layout.json", request["layout"])
            self.rewrite_artifact_hash("layout.json")
            self.rewrite_request(request)
            with self.assertRaisesRegex(ValueError, "Seeded starting layout differs"):
                runner.verify_run(self.args.output)

    def test_rehashed_shortened_source_claim_is_rejected_against_complete_recording(self):
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request["source"]["source_last_step"] -= 1
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "Source projection differs"):
            runner.verify_run(self.args.output)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL4.3")
class ChoreographyGPUTests(unittest.TestCase):
    setUp = ChoreographyIntegrationTests.setUp

    def engine(self, **overrides):
        engine = Engine(self.source, simulation(**overrides), self.palette, [])
        self.addCleanup(engine.close)
        return engine

    def test_native_initial_split_and_real_transport_preserve_all_three_amounts(self):
        settings = simulation()
        settings["initial_choreography"]["mobility_bias"] = [-0.12, 0.12, 0]
        engine = self.engine(initial_choreography=settings["initial_choreography"])
        field = engine.snapshot()
        initial = _budgeted_initial_state(engine.layout, engine.config)
        fractions = effective_layer_fractions(self.palette, engine.config["initial_choreography"])
        np.testing.assert_array_equal(field["mobile"], initial * fractions)
        np.testing.assert_array_equal(field["underpaint"], initial - initial * fractions)
        self.assertEqual(engine.palette, self.palette)
        engine.advance_to(engine.steps)
        final, diagnostics = engine.snapshot(), engine.diagnostics
        np.testing.assert_allclose(
            pigment_mass(final["pigment"], 1.6), pigment_mass(initial, 1.6), rtol=5e-6, atol=1e-12
        )
        engine.close()
        split = self.engine(initial_choreography=settings["initial_choreography"])
        for step in (0, 5, 10, 15, 20):
            split.advance_to(step)
            split.snapshot([192, 144])
        self.assertEqual(split.diagnostics, diagnostics)
        for name, value in final.items():
            np.testing.assert_array_equal(split.snapshot()[name], value)


if __name__ == "__main__":
    unittest.main()
