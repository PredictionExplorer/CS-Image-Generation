"""Reduced-body archive binding and independent source-event reconstruction."""

from __future__ import annotations

import copy
import hashlib
import unittest
from unittest.mock import patch

from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_studio.common import encoded, read, write

from . import run as runner
from . import test_run as fixtures
from .body_influence import VERSION
from .events import plan_events as original_plan_events
from .test_interaction_archive import InteractionEngineFixture


class BodyInfluenceArchiveTests(unittest.TestCase):
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
        self.raw["simulation"].update(
            material_model="laminate",
            initial_pattern="engaged",
            deposition=0,
            settling_scale=0,
            underpaint_strength=0,
            underpaint_release=0,
            burial_rate=0,
            interaction={},
            body_influence={"bodies": [1, 0]},
        )
        write(self.recipe, self.raw)
        self.patches.enter_context(
            patch("tools.estuary_confluence.engine.Engine", InteractionEngineFixture)
        )
        self.event_planner = self.patches.enter_context(
            patch("tools.estuary_confluence.events.plan_events", side_effect=original_plan_events)
        )

    def test_archive_binds_canonical_selection_source_and_reconstructed_eligible_events(self):
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        metadata = request["body_influence"]
        self.assertEqual(metadata, receipt["body_influence"])
        self.assertEqual(metadata["config"], {"version": VERSION, "bodies": [0, 1]})
        self.assertEqual(metadata["body_labels"], [1, 2])
        self.assertEqual(metadata["eligible_pairs"], [[0, 1]])
        self.assertEqual(metadata["source"], request["source"])
        self.assertEqual(metadata["events_sha256"], receipt["artifacts"]["events.json"]["sha256"])
        self.assertEqual(metadata["event_selection"]["selected_count"], len(request["events"]))
        source = Source.read(
            self.args.output / "inputs/source.orbit",
            aspect=4 / 3,
            **request["recipe"]["projection"],
        )
        self.assertEqual(request["events"], original_plan_events(source, 3, bodies=[0, 1]))
        self.assertEqual(receipt["source_fraction"], 1.0)
        self.assertEqual(len(request["frames"]), 10)

    def test_single_body_has_no_encounters_despite_unchanged_requested_count(self):
        self.raw["simulation"]["body_influence"] = {"bodies": [2]}
        write(self.recipe, self.raw)
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["recipe"]["encounters"], 3)
        self.assertEqual(request["events"], [])
        self.assertEqual(receipt["body_influence"]["eligible_pairs"], [])
        self.assertEqual(receipt["body_influence"]["event_selection"]["selected_pairs"], [])

    def test_all_body_control_omits_every_feature_key_and_reuses_the_default_archive(self):
        self.raw["simulation"]["body_influence"] = {"bodies": [2, 0, 1]}
        write(self.recipe, self.raw)
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertNotIn("body_influence", request["recipe"]["simulation"])
        self.assertNotIn("body_influence", request)
        self.assertNotIn("body_influence", receipt)
        self.raw["simulation"].pop("body_influence")
        write(self.recipe, self.raw)
        runner.run(self.args)
        self.assertEqual(read(self.args.output / "request.json"), request)
        self.assertEqual(read(self.args.output / "receipt.json"), receipt)
        self.assertEqual(fixtures.FakeEngine.instances[-1].visited, [])

    def test_all_selections_use_identical_initial_layout_metadata_palette_and_simulation_controls(
        self,
    ):
        layouts, palettes = [], []
        for index, bodies in enumerate(([0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2])):
            self.raw["simulation"]["body_influence"] = {"bodies": bodies}
            write(self.recipe, self.raw)
            self.args.output = self.folder / f"selection-{index}"
            runner.run(self.args)
            request, _ = runner.verify_run(self.args.output)
            layouts.append(request["layout"])
            palettes.append(request["palette"])
        self.assertTrue(all(layout == layouts[0] for layout in layouts))
        self.assertTrue(all(palette == palettes[0] for palette in palettes))
        for call in self.planner.call_args_list:
            self.assertNotIn("body_influence", call.args[2])

    def test_rehashed_source_mask_and_event_bindings_cannot_be_substituted(self):
        runner.run(self.args)
        original = read(self.args.output / "receipt.json")
        changes = {
            "config": {"version": VERSION, "bodies": [0]},
            "source": {"seed": "0x1"},
            "events_sha256": "f" * 64,
            "eligible_pairs": [[1, 2]],
            "version": "invented-v2",
        }
        for key, value in changes.items():
            changed = copy.deepcopy(original)
            changed["body_influence"][key] = value
            write(self.args.output / "receipt.json", changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "event binding differs"):
                runner.verify_run(self.args.output)

    def test_self_consistent_forged_active_event_is_rejected_by_source_regeneration(self):
        runner.run(self.args)
        request, receipt = (
            read(self.args.output / "request.json"),
            read(self.args.output / "receipt.json"),
        )
        request["events"] = [
            {
                "version": "confluence-encounters-v1",
                "pair": [0, 1],
                "fraction": 0.37,
                "position": [0.1, -0.2],
                "radius": 0.07,
                "duration": 0.01,
                "strength": 0.5,
                "pair_distance": 0.1,
                "prominence": 0.2,
            }
        ]
        metadata = runner.body_influence_metadata(
            request["recipe"], request["source"], request["events"]
        )
        request["body_influence"] = metadata
        receipt["body_influence"] = metadata
        write(self.args.output / "receipt.json", receipt)
        write(self.args.output / "events.json", request["events"])
        self.rewrite_artifact_hash("events.json")
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "events differ from eligible source encounters"):
            runner.verify_run(self.args.output)

    def test_inactive_or_missing_event_pairs_fail_before_archiving(self):
        recipe = runner.validate_recipe(self.raw)
        source = Source.read(self.source, aspect=4 / 3, **recipe["projection"])
        template = {
            "fraction": 0.5,
            "position": [0, 0],
            "radius": 0.07,
            "duration": 0.01,
            "strength": 0.5,
        }
        for extra in ({}, {"pair": [1, 2]}, {"pair": [0, 2]}):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "both bodies active"),
            ):
                runner.body_influence_metadata(recipe, source.metadata, [{**template, **extra}])

    def test_missing_reduced_metadata_and_unbound_default_metadata_are_rejected(self):
        runner.run(self.args)
        receipt = read(self.args.output / "receipt.json")
        receipt.pop("body_influence")
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "event binding differs"):
            runner.verify_run(self.args.output)
        self.raw["simulation"].pop("body_influence")
        write(self.recipe, self.raw)
        self.args.output = self.folder / "default"
        runner.run(self.args)
        receipt = read(self.args.output / "receipt.json")
        receipt["body_influence"] = {"bodies": [0]}
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "cannot advertise a reduced-body record"):
            runner.verify_run(self.args.output)

    def test_body_event_identity_is_the_actual_portable_event_file_identity(self):
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(
            hashlib.sha256(encoded(request["events"])).hexdigest(),
            receipt["body_influence"]["events_sha256"],
        )
        # The source-aware event schedule remains independent of film sampling.
        expected = request["events"]
        self.raw["render"].update(formation_frames=11, orbit_frames=1, hold_frames=0, fps=30)
        write(self.recipe, self.raw)
        self.args.output, self.args.still_only = self.folder / "film", False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["events"], expected)
        self.assertEqual(
            receipt["body_influence"]["events_sha256"],
            hashlib.sha256(encoded(expected)).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
