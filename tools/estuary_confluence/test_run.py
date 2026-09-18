"""Archive, optical comparison, and complete-film contracts without a GPU."""

from __future__ import annotations

import copy
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from tools.estuary_studio.common import artifact, read, write

from . import run as runner
from .palette import generate_palette
from .surface import validate_fields


def small_recipe():
    return {
        "name": "Confluence comparison fixture",
        "chromatic_count": 3,
        "looks": ["layered", "homogeneous"],
        "simulation": {"resolution": [128, 96], "steps": 10},
        "render": {
            "resolution": [128, 96],
            "still_resolution": [256, 192],
            "capture_resolution": [128, 96],
            "formation_frames": 6,
            "hold_frames": 2,
            "orbit_frames": 3,
        },
    }


def material_fields(width, height, count, step):
    shape = (height, width, count)
    mobile = np.full(shape, 0.006 * step, dtype="f4")
    deposit = np.full(shape, 0.003 * step, dtype="f4")
    underpaint = np.full(shape, 0.001 * step, dtype="f4")
    return {
        "mobile": mobile,
        "deposit": deposit,
        "underpaint": underpaint,
        "pigment": mobile + deposit + underpaint,
        "height": np.full((height, width), 0.0001 * step, dtype="f4"),
        "wetness": np.full((height, width), 0.5, dtype="f4"),
        "mixing": np.full((height, width), 0.25, dtype="f4"),
        "direction": np.zeros((height, width, 2), dtype="f4"),
        "roughness": np.full((height, width), 0.6, dtype="f4"),
        "coverage": np.full((height, width), 0.75, dtype="f4"),
    }


class FakeEngine:
    instances: ClassVar[list] = []

    def __init__(self, source, config, palette, events):
        self.source, self.config, self.palette, self.events = source, config, palette, events
        self.step, self.steps = 0, config["steps"]
        self.visited, self.snapshots = [], []
        self.closed = False
        self.metadata = {"renderer": "CPU physical-history fixture"}
        self.instances.append(self)

    def advance_to(self, step):
        if not self.step <= step <= self.steps:
            raise ValueError("Invalid fixture progress")
        self.step = step
        self.visited.append(step)

    def snapshot(self, resolution=None):
        size = self.config["resolution"] if resolution is None else resolution
        self.snapshots.append((self.step, tuple(size), resolution is None))
        return material_fields(*size, len(self.palette["pigments_srgb"]), self.step)

    def close(self):
        self.closed = True


class FakeSurface:
    instances: ClassVar[list] = []
    fail = False

    def __init__(self, config, palette):
        self.config, self.palette = config, palette
        self.metadata = {"renderer": "CPU optical-view fixture"}
        self.calls, self.closed = [], False
        self.uploaded = None
        self.instances.append(self)

    def render(self, fields, size, tilt_degrees, azimuth_degrees):
        if self.fail:
            raise RuntimeError("Deliberate surface failure")
        cached = fields is None
        if cached:
            if self.uploaded is None:
                raise RuntimeError("Surface cache is empty")
            fields = self.uploaded
        else:
            self.uploaded = fields
        validate_fields(fields, len(self.palette["pigments_srgb"]))
        self.calls.append(
            {
                "physical_sha256": runner.field_digest(fields),
                "input_shape": fields["height"].shape,
                "cached": cached,
                "tilt": tilt_degrees,
                "azimuth": azimuth_degrees,
            }
        )
        value = float(fields["pigment"].mean())
        value += 0.2 if self.config["mode"] == "layered" else 0.3
        return np.full((size[1], size[0], 3), value, dtype="f4")

    def close(self):
        self.closed = True


class RecipeAndTimelineTests(unittest.TestCase):
    def test_source_finishes_before_frozen_hold_and_camera_arc(self):
        recipe = runner.validate_recipe(small_recipe())
        plan = runner.frame_plan(recipe)
        self.assertEqual([frame["step"] for frame in plan[:6]], [0, 2, 4, 6, 8, 10])
        self.assertEqual(
            [frame["source_fraction"] for frame in plan[:6]], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.assertEqual([frame["phase"] for frame in plan[6:]], ["hold", "hold", "orbit", "orbit"])
        self.assertTrue(
            all(frame["step"] == 10 and frame["source_fraction"] == 1 for frame in plan[6:])
        )
        self.assertEqual([frame["tilt_degrees"] for frame in plan[6:]], [0, 0, 6, 12])
        self.assertEqual(plan[-1]["azimuth_degrees"], recipe["render"]["azimuth_end"])

    def test_optical_views_and_movie_cadence_cannot_reconfigure_physics(self):
        first = small_recipe()
        second = copy.deepcopy(first)
        second["looks"] = ["homogeneous"]
        second["surface"] = {"layer_scale": 30, "exposure": 0.7}
        second["render"].update(formation_frames=11, fps=30, orbit_frames=1, hold_frames=0)
        a, b = runner.validate_recipe(first), runner.validate_recipe(second)
        self.assertEqual(a["simulation"], b["simulation"])
        self.assertEqual(a["projection"], b["projection"])
        self.assertEqual(a["chromatic_count"], b["chromatic_count"])
        views = runner.surface_configs(a)
        self.assertEqual(views["homogeneous"]["mix_control"], 0)
        self.assertEqual(views["layered"]["mix_control"], 1)
        self.assertEqual(len(runner.frame_plan(b)), 11)

    def test_capture_resolution_must_exactly_reduce_material_grid(self):
        raw = small_recipe()
        raw["simulation"]["resolution"] = [256, 192]
        self.assertEqual(runner.validate_recipe(raw)["render"]["capture_resolution"], [128, 96])
        for size in ([384, 288], [160, 120], [128, 128]):
            raw["render"]["capture_resolution"] = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                runner.validate_recipe(raw)

    def test_unknown_or_incompatible_controls_fail_before_allocating_resources(self):
        changes = [
            {"chromatic_count": 4},
            {"looks": ["layered", "layered"]},
            {"looks": ["neon"]},
            {"encounters": 4},
            {"encounters": True},
            {"render": {"formation_frames": 4}},
            {"render": {"fps": True}},
            {"render": {"orbit_frames": 0}},
            {"surface": {"domain_scale": 1.5}},
            {"surface": {"tone_map": "none"}},
            {"projection": {"fill": float("nan")}},
            {"schema_version": True},
            {"misspelled": 1},
        ]
        for change in changes:
            raw = small_recipe()
            for key, value in change.items():
                if isinstance(raw.get(key), dict) and isinstance(value, dict):
                    raw[key].update(value)
                else:
                    raw[key] = value
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_recipe(raw)
        for key in ("simulation", "surface", "projection", "render"):
            raw = small_recipe()
            raw[key] = []
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_recipe(raw)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.source = self.folder / "original.orbit"
        self.source.write_bytes(b"immutable full-trajectory fixture")
        self.recipe = self.folder / "recipe.json"
        write(self.recipe, small_recipe())
        self.args = SimpleNamespace(
            source=self.source,
            recipe=self.recipe,
            output=self.folder / "artwork",
            still_only=True,
            resolution=None,
            capture_resolution=None,
            image_size=None,
            video_size=None,
        )
        self.events = [
            {
                "fraction": 0.3,
                "position": [0.1, -0.1],
                "radius": 0.08,
                "duration": 0.01,
                "strength": 0.75,
                "pair": [0, 2],
            }
        ]
        FakeEngine.instances, FakeSurface.instances = [], []
        FakeSurface.fail = False
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("tools.estuary_confluence.engine.Engine", FakeEngine))
        self.patches.enter_context(patch("tools.estuary_confluence.surface.Surface", FakeSurface))
        self.patches.enter_context(
            patch("tools.estuary_confluence.events.plan_events", return_value=self.events)
        )
        self.patches.enter_context(
            patch.object(runner.Source, "read", side_effect=self.source_info)
        )
        self.patches.enter_context(
            patch.object(runner, "runtime_identity", return_value=runner.runtime_identity())
        )
        self.patches.enter_context(
            patch.object(runner.shutil, "which", return_value=sys.executable)
        )
        self.patches.enter_context(patch.object(runner, "encode_movie", side_effect=self.movie))
        self.patches.enter_context(redirect_stdout(io.StringIO()))

    @staticmethod
    def source_info(path, **_):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        seed = "0xbc53af1cd380"
        return SimpleNamespace(
            sha256=sha, seed=seed, metadata={"sha256": sha, "samples": 100, "seed": seed}
        )

    @staticmethod
    def movie(output, recipe, *_):
        (output / "film.mp4").write_bytes(b"encoded-movie-fixture")
        result = {
            **recipe["render"],
            "full_decode_verified": True,
            "artifact": artifact(output / "film.mp4"),
        }
        write(output / "movie.json", result)
        return result

    def rewrite_artifact_hash(self, name):
        receipt = read(self.args.output / "receipt.json")
        receipt["artifacts"][name] = artifact(self.args.output / name)
        write(self.args.output / "receipt.json", receipt)

    def test_paired_looks_share_exactly_one_physical_history_and_genuine_palette(self):
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(len(FakeEngine.instances), 1)
        self.assertEqual(len(FakeSurface.instances), 2)
        self.assertEqual(FakeEngine.instances[0].visited, [10])
        self.assertEqual(request["palette"], generate_palette("0xbc53af1cd380"))
        self.assertEqual(request["events"], self.events)
        self.assertEqual(receipt["source_fraction"], 1)
        self.assertEqual(read(self.args.output / "frame-ledger.json"), [])
        hashes = [result["physical_state_sha256"] for result in receipt["looks"].values()]
        self.assertEqual(hashes, [receipt["physical_state_sha256"]] * 2)
        self.assertNotEqual(
            receipt["looks"]["layered"]["poster"], receipt["looks"]["homogeneous"]["poster"]
        )
        self.assertTrue(all(surface.closed for surface in FakeSurface.instances))
        self.assertTrue(FakeEngine.instances[0].closed)

    def test_verified_reuse_does_not_advance_or_render_again(self):
        runner.run(self.args)
        original = read(self.args.output / "receipt.json")
        runner.run(self.args)
        self.assertEqual(FakeEngine.instances[-1].visited, [])
        self.assertTrue(all(not surface.calls for surface in FakeSurface.instances[-2:]))
        self.assertEqual(read(self.args.output / "receipt.json"), original)

    def test_film_captures_full_source_once_and_freezes_material_for_camera(self):
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(FakeEngine.instances[0].visited, [0, 2, 4, 6, 8, 10])
        self.assertEqual(len(FakeEngine.instances[0].snapshots), 7)
        self.assertEqual(len(request["frames"]), 10)
        for look, surface in zip(request["recipe"]["looks"], FakeSurface.instances, strict=True):
            calls = surface.calls
            self.assertEqual(len(calls), 9)  # Six formation, two orbit, one full-grid still.
            self.assertEqual([call["cached"] for call in calls], [False] * 6 + [True, True, False])
            self.assertEqual(
                [call["physical_sha256"] for call in calls[5:]], [calls[5]["physical_sha256"]] * 4
            )
            frames = self.args.output / look / "frames"
            self.assertEqual(
                (frames / "000005.png").read_bytes(), (frames / "000006.png").read_bytes()
            )
            self.assertEqual(
                (frames / "000005.png").read_bytes(), (frames / "000007.png").read_bytes()
            )
            self.assertEqual(receipt["looks"][look]["movie"]["frames"], 10)

    def test_capture_reduction_never_replaces_full_grid_final_state(self):
        raw = small_recipe()
        raw["simulation"]["resolution"] = [256, 192]
        write(self.recipe, raw)
        self.args.still_only = False
        runner.run(self.args)
        runner.verify_run(self.args.output)
        snapshots = FakeEngine.instances[0].snapshots
        self.assertEqual(snapshots[:6], [(step, (128, 96), False) for step in (0, 2, 4, 6, 8, 10)])
        self.assertEqual(snapshots[-1], (10, (256, 192), True))
        with np.load(self.args.output / "final.npz") as fields:
            self.assertEqual(fields["pigment"].shape, (192, 256, 4))
        for surface in FakeSurface.instances:
            self.assertEqual(surface.calls[-1]["input_shape"], (192, 256))

    def test_still_film_and_optical_view_choices_preserve_final_material_fields(self):
        runner.run(self.args)
        _, expected = runner.verify_run(self.args.output)
        raw = small_recipe()
        raw["looks"] = ["homogeneous"]
        raw["surface"] = {"exposure": 0.7, "layer_scale": 20}
        raw["render"].update(formation_frames=11, fps=30, hold_frames=0, orbit_frames=1)
        write(self.recipe, raw)
        self.args.output, self.args.still_only = self.folder / "film", False
        runner.run(self.args)
        _, actual = runner.verify_run(self.args.output)
        self.assertEqual(expected["physical_state_sha256"], actual["physical_state_sha256"])

    def test_five_color_archive_uses_six_channels_including_chalk(self):
        raw = small_recipe()
        raw["chromatic_count"] = 5
        write(self.recipe, raw)
        runner.run(self.args)
        request, _ = runner.verify_run(self.args.output)
        self.assertEqual(request["palette"]["chalk_index"], 5)
        with np.load(self.args.output / "final.npz") as fields:
            self.assertEqual(fields["pigment"].shape[-1], 6)

    def test_artifact_tampering_and_source_change_are_rejected(self):
        runner.run(self.args)
        path = self.args.output / "layered/poster.png"
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with self.assertRaises(ValueError):
            runner.verify_run(self.args.output)
        path.write_bytes(original)
        self.source.write_bytes(b"different source")
        with self.assertRaisesRegex(ValueError, "different inputs"):
            runner.run(self.args)

    def test_rehashed_source_cannot_replace_bound_source_identity(self):
        runner.run(self.args)
        path = self.args.output / "inputs/source.orbit"
        path.write_bytes(b"replacement source")
        self.rewrite_artifact_hash("inputs/source.orbit")
        with self.assertRaisesRegex(ValueError, "identity"):
            runner.verify_run(self.args.output)

    def test_rehashed_palette_and_events_cannot_replace_resolved_design_inputs(self):
        runner.run(self.args)
        for name in ("palette.json", "events.json"):
            path = self.args.output / name
            original = path.read_bytes()
            value = read(path)
            if name == "palette.json":
                value["pigments_srgb"][0][0] += 0.01
            else:
                value[0]["fraction"] += 0.01
            write(path, value)
            self.rewrite_artifact_hash(name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "design inputs"):
                runner.verify_run(self.args.output)
            path.write_bytes(original)
            self.rewrite_artifact_hash(name)

    def test_rehashed_runtime_is_still_bound_to_request(self):
        runner.run(self.args)
        name = "inputs/code/tools/estuary_confluence/palette.py"
        path = self.args.output / name
        path.write_text(path.read_text() + "\n# changed runtime\n")
        self.rewrite_artifact_hash(name)
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            runner.verify_run(self.args.output)

    def test_rehashed_final_arrays_cannot_keep_stale_physical_identity(self):
        runner.run(self.args)
        path = self.args.output / "final.npz"
        with np.load(path) as saved:
            fields = {name: saved[name].copy() for name in saved.files}
        fields["mobile"][0, 0, 0] += 0.1
        fields["pigment"][0, 0, 0] += 0.1
        runner.record_array(path, fields)
        self.rewrite_artifact_hash("final.npz")
        with self.assertRaises(ValueError):
            runner.verify_run(self.args.output)

    def test_each_view_must_reference_the_same_verified_physical_state(self):
        runner.run(self.args)
        receipt = read(self.args.output / "receipt.json")
        receipt["looks"]["homogeneous"]["physical_state_sha256"] = "0" * 64
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaises(ValueError):
            runner.verify_run(self.args.output)

    def test_surface_failure_leaves_incomplete_archive_and_closes_resources(self):
        FakeSurface.fail = True
        with self.assertRaisesRegex(RuntimeError, "Deliberate surface failure"):
            runner.run(self.args)
        self.assertFalse(read(self.args.output / "receipt.json")["complete"])
        self.assertTrue((self.args.output / "request.json").exists())
        self.assertTrue((self.args.output / "final.npz").exists())
        self.assertTrue(all(surface.closed for surface in FakeSurface.instances))
        self.assertTrue(FakeEngine.instances[-1].closed)
        FakeSurface.fail = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.run(self.args)

    def test_encoder_failure_preserves_both_views_frames_and_incomplete_receipt(self):
        self.args.still_only = False
        with (
            patch.object(runner, "encode_movie", side_effect=RuntimeError("decode failed")),
            self.assertRaisesRegex(RuntimeError, "decode failed"),
        ):
            runner.run(self.args)
        self.assertFalse(read(self.args.output / "receipt.json")["complete"])
        for look in ("layered", "homogeneous"):
            self.assertEqual(len(list((self.args.output / look / "frames").glob("*.png"))), 10)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.verify_run(self.args.output)

    def test_rehashed_frame_ledger_cannot_change_source_timing(self):
        self.args.still_only = False
        runner.run(self.args)
        path = self.args.output / "frame-ledger.json"
        ledger = read(path)
        ledger[-1]["timing"]["source_fraction"] = 0.5
        write(path, ledger)
        self.rewrite_artifact_hash("frame-ledger.json")
        with self.assertRaisesRegex(ValueError, "timing"):
            runner.verify_run(self.args.output)

    def test_each_film_requires_full_decode_evidence(self):
        self.args.still_only = False
        runner.run(self.args)
        name = "homogeneous/movie.json"
        path = self.args.output / name
        movie = read(path)
        movie["full_decode_verified"] = False
        write(path, movie)
        receipt = read(self.args.output / "receipt.json")
        receipt["looks"]["homogeneous"]["movie"] = movie
        receipt["artifacts"][name] = artifact(path)
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "timeline"):
            runner.verify_run(self.args.output)


if __name__ == "__main__":
    unittest.main()
