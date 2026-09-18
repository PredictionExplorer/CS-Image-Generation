"""Timeline, archive, and failure contracts without a GPU or video dependency."""

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

from . import run as studio
from .common import artifact, read, write
from .test_common import material_fields


def small_recipe():
    return {
        "family": "fresco",
        "simulation": {"resolution": [64, 64], "steps": 10},
        "render": {
            "resolution": [64, 64],
            "still_resolution": [64, 64],
            "formation_frames": 6,
            "hold_frames": 2,
            "orbit_frames": 3,
        },
    }


class TimelineTests(unittest.TestCase):
    def test_full_source_precedes_frozen_hold_and_orbit(self):
        recipe = studio.validate_recipe(small_recipe())
        plan = studio.frame_plan(recipe)
        formation = [f for f in plan if f["phase"] == "formation"]
        self.assertEqual([f["step"] for f in formation], [0, 2, 4, 6, 8, 10])
        self.assertEqual(formation[0]["source_fraction"], 0)
        self.assertEqual(formation[-1]["source_fraction"], 1)
        frozen = plan[len(formation) :]
        self.assertEqual([f["phase"] for f in frozen], ["hold", "hold", "orbit", "orbit"])
        self.assertTrue(all(f["step"] == 10 and f["source_fraction"] == 1 for f in frozen))
        self.assertEqual([f["tilt_degrees"] for f in frozen], [0, 0, 9, 18])
        self.assertEqual(plan[-1]["azimuth_degrees"], recipe["render"]["azimuth_end"])

    def test_minimal_orbit_does_not_repeat_or_divide_by_zero(self):
        raw = small_recipe()
        raw["render"].update(orbit_frames=1, hold_frames=0)
        plan = studio.frame_plan(studio.validate_recipe(raw))
        self.assertEqual(len(plan), 6)
        self.assertEqual(plan[-1]["source_fraction"], 1)

    def test_movie_cadence_does_not_change_the_resolved_simulation(self):
        a = small_recipe()
        b = copy.deepcopy(a)
        b["render"].update(formation_frames=11, fps=30, orbit_frames=15)
        self.assertEqual(
            studio.validate_recipe(a)["simulation"], studio.validate_recipe(b)["simulation"]
        )

    def test_nocturne_can_use_either_authored_process_with_explicit_identity(self):
        for dynamics in ("fresco", "monotype"):
            raw = small_recipe()
            raw.update(family="nocturne", dynamics=dynamics)
            resolved = studio.validate_recipe(raw)
            self.assertEqual(resolved["dynamics"], dynamics)
            self.assertEqual(resolved["surface"]["family"], "nocturne")
            self.assertEqual("settling" in resolved["simulation"], dynamics == "fresco")
        raw = small_recipe()
        raw["dynamics"] = "monotype"
        with self.assertRaisesRegex(ValueError, "dynamics"):
            studio.validate_recipe(raw)

    def test_invalid_recipe_clock_geometry_or_family_fails(self):
        changes = [
            {"render": {"formation_frames": 4}},
            {"render": {"orbit_frames": 0}},
            {"render": {"resolution": [64, 96]}},
            {"render": {"fps": True}},
            {"surface": {"domain_scale": 1.3}},
            {"surface": {"family": "monotype"}},
            {"surface": {"tone_map": "none"}},
            {"projection": {"fill": float("nan")}},
            {"schema_version": True},
            {"family": "watercolor"},
            {"spelling_error": 1},
        ]
        for change in changes:
            raw = small_recipe()
            for key, value in change.items():
                if key in raw and isinstance(raw[key], dict) and isinstance(value, dict):
                    raw[key].update(value)
                else:
                    raw[key] = value
            with self.subTest(change=change), self.assertRaises(ValueError):
                studio.validate_recipe(raw)

    def test_nested_nonobjects_fail_as_validation_errors(self):
        for family, key in (
            ("fresco", "surface"),
            ("monotype", "simulation"),
            ("fresco", "render"),
            ("fresco", "projection"),
        ):
            raw = small_recipe()
            raw["family"] = family
            raw[key] = []
            with self.subTest(family=family, key=key), self.assertRaises(ValueError):
                studio.validate_recipe(raw)


class FakeEngine:
    instances: ClassVar[list] = []

    def __init__(self, source, config):
        self.config = config
        self.steps = config["steps"]
        self.step = 0
        self.visited = []
        self.closed = False
        self.metadata = {"renderer": "CPU contract fixture"}
        self.instances.append(self)

    def advance_to(self, step):
        if not self.step <= step <= self.steps:
            raise ValueError("Invalid fixture progress")
        self.step = step
        self.visited.append(step)

    def snapshot(self):
        fields = material_fields(*self.config["resolution"])
        fields["pigment"][:] = 0.01 * self.step
        fields["mobile"] = fields["pigment"].copy()
        return fields

    def close(self):
        self.closed = True


class FakeSurface:
    instances: ClassVar[list] = []
    fail = False

    def __init__(self, config):
        self.metadata = {"renderer": "CPU surface fixture"}
        self.closed = False
        self.calls = []
        self.instances.append(self)

    def render(self, fields, size, tilt_degrees, azimuth_degrees):
        if self.fail:
            raise RuntimeError("Deliberate surface failure")
        if fields is None:
            fields = self.uploaded
        else:
            self.uploaded = fields
        if set(fields) != set(material_fields()):
            raise ValueError("Surface received nonrender fields")
        amount = float(fields["pigment"].mean())
        self.calls.append((amount, tilt_degrees, azimuth_degrees))
        return np.full((size[1], size[0], 3), 0.2 + amount, dtype="f4")

    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.source = self.folder / "original.orbit"
        self.source.write_bytes(b"source fixture")
        self.recipe = self.folder / "recipe.json"
        write(self.recipe, small_recipe())
        self.args = SimpleNamespace(
            source=self.source,
            recipe=self.recipe,
            output=self.folder / "artwork",
            still_only=True,
            resolution=None,
            image_size=None,
            video_size=None,
            formation_frames=None,
            orbit_frames=None,
        )
        FakeEngine.instances = []
        FakeSurface.instances = []
        FakeSurface.fail = False
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("tools.estuary_studio.fresco.Fresco", FakeEngine))
        self.patches.enter_context(patch("tools.estuary_studio.surface.Surface", FakeSurface))
        self.patches.enter_context(
            patch.object(studio.Source, "read", side_effect=self.source_info)
        )
        self.patches.enter_context(
            patch.object(studio, "runtime_identity", return_value=studio.runtime_identity())
        )
        self.patches.enter_context(
            patch.object(studio.shutil, "which", return_value=sys.executable)
        )
        self.patches.enter_context(patch.object(studio, "encode_movie", side_effect=self.movie))
        self.patches.enter_context(redirect_stdout(io.StringIO()))

    def source_info(self, path, **_):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return SimpleNamespace(sha256=sha, metadata={"sha256": sha, "samples": 100})

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

    def test_complete_still_is_archived_verified_and_reused_without_advancing(self):
        studio.run(self.args)
        request, receipt = studio.verified_run(self.args.output)
        self.assertEqual(request["mode"], "still")
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["final_step"], 10)
        self.assertEqual(read(self.args.output / "frame-ledger.json"), [])
        with np.load(self.args.output / "final.npz") as state:
            self.assertIn("mobile", state.files)
            np.testing.assert_allclose(state["pigment"], 0.1)
        identity = receipt["identity_sha256"]
        studio.run(self.args)
        self.assertEqual(FakeEngine.instances[-1].visited, [])
        self.assertEqual(FakeSurface.instances[-1].calls, [])
        self.assertTrue(all(e.closed for e in FakeEngine.instances))
        self.assertTrue(all(s.closed for s in FakeSurface.instances))
        self.assertEqual(read(self.args.output / "receipt.json")["identity_sha256"], identity)

    def test_film_uses_complete_source_and_reuses_frozen_state_for_camera_motion(self):
        self.args.still_only = False
        studio.run(self.args)
        request, receipt = studio.verified_run(self.args.output)
        self.assertEqual(receipt["movie"]["frames"], 10)
        self.assertEqual(FakeEngine.instances[0].visited, [0, 2, 4, 6, 8, 10])
        calls = FakeSurface.instances[0].calls
        self.assertEqual(len(calls), 9)  # Six formation, two orbit, one final still.
        self.assertEqual([c[0] for c in calls[-3:]], [calls[5][0]] * 3)
        frames = self.args.output / "frames"
        self.assertEqual((frames / "000005.png").read_bytes(), (frames / "000006.png").read_bytes())
        self.assertEqual((frames / "000005.png").read_bytes(), (frames / "000007.png").read_bytes())
        self.assertEqual(request["frames"][-1]["source_fraction"], 1)

    def test_still_and_film_archive_the_identical_final_material_fields(self):
        studio.run(self.args)
        with np.load(self.args.output / "final.npz") as data:
            expected = {key: data[key].copy() for key in data.files}
        self.args.output = self.folder / "film"
        self.args.still_only = False
        studio.run(self.args)
        with np.load(self.args.output / "final.npz") as data:
            self.assertEqual(set(data.files), set(expected))
            for key in expected:
                np.testing.assert_array_equal(data[key], expected[key])

    def test_nocturne_with_fresco_process_uses_the_fresco_engine(self):
        raw = small_recipe()
        raw.update(family="nocturne", dynamics="fresco")
        write(self.recipe, raw)
        studio.run(self.args)
        request, _ = studio.verified_run(self.args.output)
        self.assertEqual(request["recipe"]["dynamics"], "fresco")
        self.assertEqual(len(FakeEngine.instances), 1)

    def test_artifact_tampering_and_changed_source_are_rejected(self):
        studio.run(self.args)
        poster = self.args.output / "poster.png"
        original = poster.read_bytes()
        poster.write_bytes(original + b"tampered")
        with self.assertRaises(ValueError):
            studio.verified_run(self.args.output)
        poster.write_bytes(original)
        self.source.write_bytes(b"another trajectory")
        with self.assertRaisesRegex(ValueError, "another recipe, source or runtime"):
            studio.run(self.args)

    def test_updated_artifact_hash_cannot_rebind_the_archived_source(self):
        studio.run(self.args)
        archived = self.args.output / "inputs/source.orbit"
        archived.write_bytes(b"replacement")
        receipt = read(self.args.output / "receipt.json")
        receipt["artifacts"]["inputs/source.orbit"] = artifact(archived)
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "recorded source"):
            studio.verified_run(self.args.output)

    def test_runtime_copies_are_bound_to_request_even_if_receipt_hash_is_updated(self):
        studio.run(self.args)
        name = "inputs/code/tools/estuary_studio/surface.py"
        path = self.args.output / name
        path.write_text(path.read_text() + "\n# replaced runtime\n")
        receipt = read(self.args.output / "receipt.json")
        receipt["artifacts"][name] = artifact(path)
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            studio.verified_run(self.args.output)

    def test_incomplete_run_is_preserved_and_never_reported_complete(self):
        FakeSurface.fail = True
        with self.assertRaisesRegex(RuntimeError, "Deliberate surface failure"):
            studio.run(self.args)
        self.assertFalse(read(self.args.output / "receipt.json")["complete"])
        self.assertTrue((self.args.output / "request.json").exists())
        self.assertTrue((self.args.output / "final.npz").exists())
        self.assertTrue(FakeEngine.instances[-1].closed)
        self.assertTrue(FakeSurface.instances[-1].closed)
        FakeSurface.fail = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            studio.run(self.args)
        self.assertFalse(read(self.args.output / "receipt.json")["complete"])

    def test_failed_encoder_keeps_frames_and_incomplete_receipt(self):
        self.args.still_only = False
        with (
            patch.object(studio, "encode_movie", side_effect=RuntimeError("decode failed")),
            self.assertRaisesRegex(RuntimeError, "decode failed"),
        ):
            studio.run(self.args)
        self.assertEqual(len(list((self.args.output / "frames").glob("*.png"))), 10)
        self.assertFalse(read(self.args.output / "receipt.json")["complete"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            studio.verified_run(self.args.output)

    def test_ledger_timing_is_enforced(self):
        self.args.still_only = False
        studio.run(self.args)
        ledger_path = self.args.output / "frame-ledger.json"
        ledger = read(ledger_path)
        ledger[-1]["timing"]["source_fraction"] = 0.5
        write(ledger_path, ledger)
        receipt = read(self.args.output / "receipt.json")
        receipt["artifacts"]["frame-ledger.json"] = artifact(ledger_path)
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "timing"):
            studio.verified_run(self.args.output)

    def test_movie_requires_verified_full_decode(self):
        self.args.still_only = False
        studio.run(self.args)
        movie_path = self.args.output / "movie.json"
        movie = read(movie_path)
        movie["full_decode_verified"] = False
        write(movie_path, movie)
        receipt = read(self.args.output / "receipt.json")
        receipt["movie"] = movie
        receipt["artifacts"]["movie.json"] = artifact(movie_path)
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "timeline"):
            studio.verified_run(self.args.output)

    def test_explicit_zero_frame_override_is_rejected(self):
        for key in ("formation_frames", "orbit_frames"):
            args = copy.copy(self.args)
            setattr(args, key, 0)
            with self.subTest(key=key), self.assertRaises(ValueError):
                studio.run(args)


if __name__ == "__main__":
    unittest.main()
