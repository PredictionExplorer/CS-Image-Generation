"""Diagnostic annotations use recorded source time and never enter the paint."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_studio.common import artifact, read, write

from . import run as runner
from . import test_run as fixtures
from .body_markers import annotate, project_positions, validate_config
from .test_interaction_archive import InteractionEngineFixture


class BodyMarkerArchiveTests(unittest.TestCase):
    # The archived orbit and its PCA projection are real; only paint/GPU work
    # and movie encoding use the established small archive fixtures.
    source_info = staticmethod(Source.read)
    engaged_layout = staticmethod(fixtures.PipelineTests.engaged_layout)
    movie = staticmethod(fixtures.PipelineTests.movie)
    rewrite_artifact_hash = fixtures.PipelineTests.rewrite_artifact_hash
    rewrite_request = fixtures.PipelineTests.rewrite_request

    def setUp(self):
        fixtures.PipelineTests.setUp(self)
        write_orbit(self.source, orbit_points(), seed="0xbc53af1cd380")
        self.raw = fixtures.small_recipe()
        self.raw["simulation"].update(
            material_model="laminate",
            initial_pattern="scattered",
            deposition=0,
            settling_scale=0,
            underpaint_strength=0,
            underpaint_release=0,
            burial_rate=0,
            interaction={},
        )
        self.raw["render"]["body_markers"] = True
        write(self.recipe, self.raw)
        self.patches.enter_context(
            patch("tools.estuary_confluence.engine.Engine", InteractionEngineFixture)
        )

    def marker_ledger(self):
        return read(self.args.output / "body-markers.json")

    def rewrite_markers(self, ledger):
        write(self.args.output / "body-markers.json", ledger)
        self.rewrite_artifact_hash("body-markers.json")

    def test_opt_in_normalizes_and_disabled_forms_preserve_the_legacy_recipe(self):
        raw = copy.deepcopy(self.raw)
        raw["render"].pop("body_markers")
        legacy = runner.validate_recipe(raw)
        for disabled in (None, False):
            raw["render"]["body_markers"] = disabled
            self.assertEqual(runner.validate_recipe(raw), legacy)
        raw["render"]["body_markers"] = {}
        enabled = runner.validate_recipe(raw)
        self.assertEqual(enabled["render"]["body_markers"], validate_config(True))
        raw["render"]["body_markers"] = True
        self.assertEqual(runner.validate_recipe(raw), enabled)
        for invalid in ("yes", 1, [], {"unknown": 1}):
            raw["render"]["body_markers"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runner.validate_recipe(raw)

    def test_film_uses_true_source_times_and_frozen_endpoint_with_actual_camera(self):
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        ledger = self.marker_ledger()
        source = fixtures.FakeEngine.instances[-1].source
        self.assertEqual(ledger["metadata"], request["body_markers"])
        self.assertEqual(receipt["body_markers"], request["body_markers"])
        self.assertEqual(ledger["metadata"]["source"], source.metadata)
        self.assertEqual(
            [r["source_fraction"] for r in ledger["frames"][:6]], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        for frame, record in zip(request["frames"], ledger["frames"], strict=True):
            np.testing.assert_array_equal(
                record["positions"], source.frame(frame["source_fraction"]).positions
            )
            np.testing.assert_allclose(
                record["pixel_centers"],
                project_positions(
                    source.frame(frame["source_fraction"]).positions,
                    request["recipe"]["render"]["resolution"],
                    tilt_degrees=frame["tilt_degrees"],
                    azimuth_degrees=frame["azimuth_degrees"],
                ),
                rtol=0,
                atol=1e-10,
            )
        for record in ledger["frames"][6:]:
            self.assertEqual(record["source_fraction"], 1)
            np.testing.assert_array_equal(record["positions"], source.frame(1).positions)
        self.assertEqual(ledger["frames"][5]["pixel_centers"], ledger["frames"][6]["pixel_centers"])
        self.assertNotEqual(
            ledger["frames"][5]["pixel_centers"], ledger["frames"][-1]["pixel_centers"]
        )
        np.testing.assert_array_equal(ledger["initial"]["positions"], source.frame(0).positions)
        np.testing.assert_array_equal(ledger["poster"]["positions"], source.frame(1).positions)
        self.assertEqual(ledger["poster"]["resolution"], [256, 192])

    def test_overlay_is_applied_after_every_render_and_hold_copies_the_marked_frame(self):
        self.args.still_only = False
        rendered, written = [], []
        original_render, original_write = fixtures.FakeSurface.render, runner.write_png

        def capture_render(surface, fields, **kwargs):
            pixels = original_render(surface, fields, **kwargs)
            rendered.append((surface.config["mode"], pixels.copy(), kwargs))
            return pixels

        def capture_write(path, pixels, **kwargs):
            written.append((path, pixels.copy()))
            original_write(path, pixels, **kwargs)

        with (
            patch.object(fixtures.FakeSurface, "render", capture_render),
            patch.object(runner, "write_png", side_effect=capture_write),
        ):
            runner.run(self.args)
        request, _ = runner.verify_run(self.args.output)
        ledger = self.marker_ledger()
        self.assertEqual(len(rendered), len(written))
        for (look, unmarked, _), (path, marked) in zip(rendered, written, strict=True):
            record = (
                ledger["poster"] if path.name == "poster.png" else ledger["frames"][int(path.stem)]
            )
            expected = annotate(
                unmarked,
                record["positions"],
                tilt_degrees=record["tilt_degrees"],
                azimuth_degrees=record["azimuth_degrees"],
                config=request["body_markers"]["config"],
            )
            np.testing.assert_array_equal(marked, expected)
            self.assertFalse(np.array_equal(unmarked, marked), str(path))
            if path.name == "poster.png":
                np.testing.assert_array_equal(
                    np.load(self.args.output / look / "poster-unmarked-linear.npy"), unmarked
                )
        for look in request["recipe"]["looks"]:
            frames = self.args.output / look / "frames"
            self.assertEqual(
                (self.args.output / look / "initial.png").read_bytes(),
                (frames / "000000.png").read_bytes(),
            )
            self.assertEqual(
                (frames / "000005.png").read_bytes(), (frames / "000006.png").read_bytes()
            )
            self.assertEqual(
                (frames / "000005.png").read_bytes(), (frames / "000007.png").read_bytes()
            )

    def test_still_initial_and_poster_have_separate_times_and_cameras(self):
        runner.run(self.args)
        request, _ = runner.verify_run(self.args.output)
        ledger = self.marker_ledger()
        self.assertEqual(ledger["frames"], [])
        self.assertEqual(ledger["initial"]["source_fraction"], 0)
        self.assertEqual(ledger["initial"]["tilt_degrees"], 0)
        self.assertEqual(ledger["poster"]["source_fraction"], 1)
        self.assertEqual(
            ledger["poster"]["tilt_degrees"], request["recipe"]["render"]["still_tilt_degrees"]
        )
        self.assertNotEqual(ledger["initial"]["positions"], ledger["poster"]["positions"])

    def test_annotations_preserve_all_fourteen_material_fields_and_unmarked_diagnostics(self):
        self.raw["assessment"] = {
            "interval_steps": 2,
            "resolution": [128, 96],
            "share_threshold": 0.1,
        }
        self.raw["render"]["body_markers"] = False
        write(self.recipe, self.raw)
        runner.run(self.args)
        baseline_request, baseline = runner.verify_run(self.args.output)
        baseline_folder = self.args.output
        self.assertNotIn("body_markers", baseline_request)
        self.assertNotIn("body_markers", baseline)
        self.assertNotIn("body_markers", baseline_request["recipe"]["render"])
        self.assertFalse((baseline_folder / "body-markers.json").exists())
        self.raw["render"]["body_markers"] = True
        write(self.recipe, self.raw)
        self.args.output = self.folder / "marked"
        runner.run(self.args)
        _, marked = runner.verify_run(self.args.output)
        self.assertEqual(marked["physical_state_sha256"], baseline["physical_state_sha256"])
        self.assertEqual(marked["base_material_sha256"], baseline["base_material_sha256"])
        self.assertEqual(
            read(baseline_folder / "assessment.json"), read(self.args.output / "assessment.json")
        )
        for look in baseline["looks"]:
            self.assertEqual(
                marked["looks"][look]["image_balance"], baseline["looks"][look]["image_balance"]
            )
            self.assertNotEqual(marked["looks"][look]["poster"], baseline["looks"][look]["poster"])

    def test_rehashed_positions_and_matching_pixels_cannot_override_archived_source(self):
        runner.run(self.args)
        ledger = self.marker_ledger()
        record = ledger["poster"]
        record["positions"][0][0] += 0.02
        record["pixel_centers"] = project_positions(
            record["positions"],
            record["resolution"],
            tilt_degrees=record["tilt_degrees"],
            azimuth_degrees=record["azimuth_degrees"],
        ).tolist()
        self.rewrite_markers(ledger)
        with self.assertRaisesRegex(ValueError, "positions differ from the archived trajectory"):
            runner.verify_run(self.args.output)

    def test_timing_and_camera_are_exact_even_for_tiny_rehashed_changes(self):
        self.args.still_only = False
        runner.run(self.args)
        original = self.marker_ledger()
        for key in ("source_fraction", "azimuth_degrees", "tilt_degrees"):
            changed = copy.deepcopy(original)
            changed["frames"][2][key] += 1e-13
            self.rewrite_markers(changed)
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "timing or camera differs"),
            ):
                runner.verify_run(self.args.output)

    def test_tiny_projection_roundoff_is_allowed_but_subpixel_relocation_is_rejected(self):
        runner.run(self.args)
        original = self.marker_ledger()
        rounded = copy.deepcopy(original)
        rounded["poster"]["positions"][0][0] += 1e-12
        rounded["poster"]["pixel_centers"][0][0] += 4e-8
        self.rewrite_markers(rounded)
        runner.verify_run(self.args.output)
        moved = copy.deepcopy(original)
        moved["poster"]["pixel_centers"][0][0] += 0.001
        self.rewrite_markers(moved)
        with self.assertRaisesRegex(ValueError, "pixel projection differs"):
            runner.verify_run(self.args.output)

    def test_rehashed_marker_metadata_cannot_change_config_source_or_version(self):
        runner.run(self.args)
        original = read(self.args.output / "receipt.json")
        for key, value in (("version", "invented-v2"), ("source", {}), ("config", {})):
            changed = copy.deepcopy(original)
            changed["body_markers"][key] = value
            write(self.args.output / "receipt.json", changed)
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "source, version or settings"),
            ):
                runner.verify_run(self.args.output)

    def test_final_overlay_cannot_be_removed_even_after_rehashing_the_linear_poster(self):
        runner.run(self.args)
        folder = self.args.output / "layered"
        unmarked = np.load(folder / "poster-unmarked-linear.npy")
        np.save(folder / "poster-linear.npy", unmarked, allow_pickle=False)
        self.rewrite_artifact_hash("layered/poster-linear.npy")
        with self.assertRaisesRegex(ValueError, "poster differs from its unmarked painting"):
            runner.verify_run(self.args.output)

    def test_disabled_run_rejects_unbound_marker_artifacts(self):
        self.raw["render"]["body_markers"] = False
        write(self.recipe, self.raw)
        runner.run(self.args)
        write(self.args.output / "body-markers.json", {})
        receipt = read(self.args.output / "receipt.json")
        receipt["artifacts"]["body-markers.json"] = artifact(self.args.output / "body-markers.json")
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Disabled body markers"):
            runner.verify_run(self.args.output)


if __name__ == "__main__":
    unittest.main()
