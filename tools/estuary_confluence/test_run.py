"""Archive, optical comparison, and complete-film contracts without a GPU."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import linear_to_srgb, srgb_to_linear
from tools.estuary_studio.common import artifact, read, write

from . import run as runner
from .layout import plan_layout
from .palette import generate_palette
from .spectral import build_spectral_material, validate_spectral_material
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
        self.layout = runner.resolved_layout(
            {"simulation": config, "chromatic_count": palette["chromatic_count"]},
            source.seed,
            source,
        )
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

    @property
    def diagnostics(self):
        diffuses = self.config.get("diffusion_coefficient", 0) > 0
        return {
            "canonical_steps": self.step,
            "actual_transport_substeps": 2 * self.step,
            "diffusion_substeps": 2 * self.step if diffuses else 0,
            "maximum_courant": 0.75 if self.step else 0.0,
            "maximum_diffusion_number": 0.18 if diffuses and self.step else 0.0,
        }

    def close(self):
        self.closed = True


class FakeSurface:
    instances: ClassVar[list] = []
    fail = False

    def __init__(self, config, palette, *, spectral=None):
        self.config, self.palette = config, palette
        self.spectral = spectral
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
                "output_size": tuple(size),
            }
        )
        value = float(fields["pigment"].mean())
        value += 0.2 if self.config["mode"] == "layered" else 0.3
        return np.full((size[1], size[0], 3), value, dtype="f4")

    def close(self):
        self.closed = True


class RecipeAndTimelineTests(unittest.TestCase):
    def test_design_round_trip_accepts_numpy_floats_but_preserves_schema_types(self):
        self.assertTrue(runner.equivalent_design({"scale": np.float64(1.25)}, {"scale": 1.25}))
        self.assertFalse(runner.equivalent_design({"step": True}, {"step": 1}))
        self.assertFalse(runner.equivalent_design({"step": 1.0}, {"step": 1}))
        self.assertFalse(
            runner.equivalent_design({"scale": np.float64("nan")}, {"scale": float("nan")})
        )
        self.assertFalse(runner.equivalent_design({"scale": np.float64(1.25)}, {"scale": 1.2501}))

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
            {"palette_mode": "unknown"},
            {"palette_mode": True},
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

    def test_frame_supersampling_validates_intermediate_resolution_without_changing_physics(self):
        base = runner.validate_recipe(small_recipe())
        self.assertEqual(base["render"]["frame_supersampling"], 1)
        raw = small_recipe()
        raw["render"]["frame_supersampling"] = 2
        doubled = runner.validate_recipe(raw)
        self.assertEqual(runner.frame_raster_resolution(doubled["render"]), [256, 192])
        self.assertEqual(doubled["simulation"], base["simulation"])
        self.assertEqual(doubled["render"]["still_resolution"], base["render"]["still_resolution"])
        for value in (0, 3, True, 1.5, None):
            raw["render"]["frame_supersampling"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_recipe(raw)
        raw["render"]["frame_supersampling"] = 2
        raw["render"]["resolution"] = [3840, 2880]
        with self.assertRaises(ValueError):
            runner.validate_recipe(raw)

    def test_frame_filter_averages_linear_samples_before_the_display_transfer_curve(self):
        class CheckerSurface:
            def render(self, fields, size, tilt_degrees, azimuth_degrees):
                self.call = (fields, size, tilt_degrees, azimuth_degrees)
                rows, columns = np.indices((size[1], size[0]))
                values = np.where((rows + columns) % 2, 0.05, 0.65).astype("f4")
                self.pixels = np.repeat(values[..., None], 3, axis=-1)
                return self.pixels

        surface = CheckerSurface()
        render = {"resolution": [64, 64], "frame_supersampling": 2}
        pixels = runner.render_frame(surface, None, render, tilt_degrees=12, azimuth_degrees=35)
        self.assertEqual(surface.call, (None, (128, 128), 12, 35))
        self.assertEqual(pixels.shape, (64, 64, 3))
        np.testing.assert_allclose(pixels, 0.35, atol=3e-8, rtol=0)
        wrong_encoded = linear_to_srgb(surface.pixels).reshape(64, 2, 64, 2, 3).mean(axis=(1, 3))
        wrong_linear = srgb_to_linear(wrong_encoded)
        self.assertGreater(float(np.mean(pixels - wrong_linear)), 0.05)
        self.assertGreater(float(np.mean(linear_to_srgb(pixels) - wrong_encoded)), 0.05)

    def test_default_frame_sampling_returns_the_original_surface_pixels(self):
        class SurfaceFixture:
            pixels = np.arange(64 * 64 * 3, dtype="f4").reshape(64, 64, 3) / 20000

            def render(self, fields, **kwargs):
                self.call = (fields, kwargs)
                return self.pixels

        surface = SurfaceFixture()
        result = runner.render_frame(
            surface, None, {"resolution": [64, 64]}, tilt_degrees=0, azimuth_degrees=-35
        )
        self.assertIs(result, surface.pixels)
        self.assertEqual(
            surface.call, (None, {"size": (64, 64), "tilt_degrees": 0, "azimuth_degrees": -35})
        )

    def test_capture_metadata_distinguishes_material_reduction_and_frame_antialiasing(self):
        native = runner.validate_recipe(small_recipe())
        metadata = runner.capture_metadata(native)
        self.assertEqual(metadata["material_sampling"], "full native material grid")
        self.assertEqual(metadata["material_reduction_factor"], 1)
        self.assertEqual(metadata["frame_filter"], "none")
        raw = small_recipe()
        raw["simulation"]["resolution"] = [256, 192]
        raw["render"]["frame_supersampling"] = 2
        reduced = runner.capture_metadata(runner.validate_recipe(raw))
        self.assertIn("area averages before pigment optics", reduced["material_sampling"])
        self.assertEqual(reduced["material_reduction_factor"], 2)
        self.assertEqual(reduced["frame_raster_resolution"], [256, 192])
        self.assertEqual(reduced["frame_output_resolution"], [128, 96])
        self.assertIn("linear display RGB before sRGB", reduced["frame_filter"])


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
        self.planner = self.patches.enter_context(
            patch(
                "tools.estuary_confluence.participation_layout.plan_engaged_layout",
                side_effect=self.engaged_layout,
            )
        )
        self.source_reader = self.patches.enter_context(
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
    def source_info(path, **supplied):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        seed = "0xbc53af1cd380"
        projection = {
            "aspect": supplied.get("aspect", 4 / 3),
            "fill": supplied.get("fill", 0.78),
            "rotation_degrees": supplied.get("rotation_degrees", 0.0),
        }
        return SimpleNamespace(
            sha256=sha,
            seed=seed,
            aspect=projection["aspect"],
            projection=projection,
            metadata={"sha256": sha, "samples": 100, "seed": seed, "projection": projection},
        )

    @staticmethod
    def engaged_layout(source, count, settings):
        """Small source-dependent stand-in; the real pilot has separate tests."""
        layout = plan_layout(
            source.seed,
            count,
            source.aspect,
            load_radius=settings["load_radius"],
            initial_load=settings["initial_load"],
            edge_width=settings["initial_edge_width"],
        )
        token = hashlib.sha256(runner.encoded(source.metadata)).digest()
        offset = (int.from_bytes(token[:4], "big") / 2**32 - 0.5) * 0.005
        for pool in layout["pools"]:
            pool["position"][0] += offset
        layout.update(
            version="runner-engaged-fixture-v1",
            source_sha256=source.sha256,
            source_projection=copy.deepcopy(source.projection),
        )
        return layout

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

    def rewrite_request(self, request):
        write(self.args.output / "request.json", request)
        receipt = read(self.args.output / "receipt.json")
        receipt["identity_sha256"] = hashlib.sha256(runner.encoded(request)).hexdigest()
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

    def scatter_recipe(self, mode="harmonic"):
        raw = small_recipe()
        raw.update(chromatic_count=5, palette_mode=mode, looks=["layered"])
        raw["simulation"].update(
            initial_pattern="scattered",
            underpaint_strength=0,
            settling_scale=0,
            burial_rate=0,
            deposition=0,
        )
        raw["surface"] = {"finish": "crisp"}
        write(self.recipe, raw)
        return raw

    def convergence_recipe(self, *, spectral=False, assessment=False):
        raw = self.scatter_recipe()
        raw["simulation"].update(
            initial_pattern="engaged",
            carrier_velocity=[0, 0],
            flow_domain_scale=1,
            diffusion_coefficient=0.0001,
        )
        if spectral:
            raw["surface"]["optics_model"] = "spectral"
        if assessment:
            raw["simulation"]["resolution"] = [256, 192]
            raw["assessment"] = {"interval_steps": 3, "resolution": [128, 96]}
        write(self.recipe, raw)
        return raw

    def test_engaged_layout_is_bound_to_archived_source_and_reconstructed_projection(self):
        self.convergence_recipe()
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["layout"]["source_sha256"], request["source"]["sha256"])
        self.assertEqual(request["layout"]["source_projection"], request["source"]["projection"])
        self.assertEqual(request["layout"], FakeEngine.instances[-1].layout)
        self.assertEqual(read(self.args.output / "layout.json"), request["layout"])
        self.assertTrue(
            any(
                call.args[0] == self.args.output / "inputs/source.orbit"
                for call in self.source_reader.call_args_list
            )
        )
        self.assertEqual(receipt["solver_diagnostics"], FakeEngine.instances[-1].diagnostics)
        with np.load(self.args.output / "final.npz") as archive:
            fields = {key: archive[key] for key in archive.files}
        self.assertEqual(receipt["physical_state_sha256"], runner.field_digest(fields))
        self.assertIn("layered/initial.png", receipt["artifacts"])

    def test_engaged_layout_requires_source_and_changes_when_source_changes(self):
        raw = self.convergence_recipe()
        recipe = runner.validate_recipe(raw)
        with self.assertRaisesRegex(ValueError, "complete source"):
            runner.resolved_layout(recipe, "0xbc53af1cd380")
        first = self.source_info(self.source)
        self.source.write_bytes(b"different full trajectory with the same palette seed")
        second = self.source_info(self.source)
        a = runner.resolved_layout(recipe, first.seed, first)
        b = runner.resolved_layout(recipe, second.seed, second)
        self.assertNotEqual(a["pools"], b["pools"])

    def test_rebound_engaged_pool_edit_cannot_pass_source_regeneration(self):
        self.convergence_recipe()
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request["layout"]["pools"][0]["position"][0] += 0.05
        write(self.args.output / "layout.json", request["layout"])
        self.rewrite_artifact_hash("layout.json")
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "Seeded starting layout"):
            runner.verify_run(self.args.output)

    def test_source_projection_metadata_cannot_be_rebound_without_changing_recording(self):
        self.convergence_recipe()
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request["source"]["projection"]["fill"] = 0.6
        self.rewrite_request(request)
        receipt = read(self.args.output / "receipt.json")
        receipt["source"] = request["source"]
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Source projection"):
            runner.verify_run(self.args.output)

    def test_spectral_coefficients_are_archived_validated_and_shared_with_each_surface(self):
        raw = self.convergence_recipe(spectral=True)
        raw["looks"] = ["layered", "homogeneous"]
        write(self.recipe, raw)
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["spectral"], read(self.args.output / "spectral.json"))
        self.assertEqual(request["spectral"], build_spectral_material(request["palette"]))
        validate_spectral_material(request["spectral"], request["palette"])
        self.assertIn("spectral.json", receipt["artifacts"])
        self.assertTrue(
            all(surface.spectral == request["spectral"] for surface in FakeSurface.instances)
        )
        self.assertEqual(len(request["spectral"]["pigment_reflectance"]), 6)
        self.assertEqual(len(request["spectral"]["wavelengths_nm"]), 38)

    def test_spectral_artifact_rehash_cannot_replace_bound_coefficients(self):
        self.convergence_recipe(spectral=True)
        runner.run(self.args)
        record = read(self.args.output / "spectral.json")
        record["pigment_reflectance"][0][0] *= 0.9
        write(self.args.output / "spectral.json", record)
        self.rewrite_artifact_hash("spectral.json")
        with self.assertRaisesRegex(ValueError, "pigment spectra"):
            runner.verify_run(self.args.output)

    def test_self_consistent_spectra_from_another_palette_are_rejected(self):
        self.convergence_recipe(spectral=True)
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request["spectral"] = build_spectral_material(generate_palette(17, 5, mode="harmonic"))
        write(self.args.output / "spectral.json", request["spectral"])
        self.rewrite_artifact_hash("spectral.json")
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "another palette"):
            runner.verify_run(self.args.output)

    def test_assessment_collects_between_movie_captures_and_preserves_capture_states(self):
        self.convergence_recipe(assessment=True)
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        report = read(self.args.output / "assessment.json")
        self.assertEqual([row["step"] for row in report["samples"]], [0, 3, 6, 9, 10])
        self.assertEqual(FakeEngine.instances[-1].visited, [0, 2, 3, 4, 6, 8, 9, 10])
        for row in report["samples"]:
            self.assertEqual(row["source_fraction"], row["step"] / 10)
            self.assertEqual(
                row["metrics"],
                runner.measure(material_fields(128, 96, 6, row["step"]), request["recipe"]),
            )
        calls = FakeSurface.instances[-1].calls
        expected = [
            runner.field_digest(material_fields(128, 96, 6, step)) for step in (0, 2, 4, 6, 8, 10)
        ]
        self.assertEqual([call["physical_sha256"] for call in calls[:6]], expected)
        self.assertEqual(report["final"]["grid_resolution"], [256, 192])
        self.assertEqual(report["samples"][-1]["metrics"]["grid_resolution"], [128, 96])
        self.assertEqual(
            receipt["artifacts"]["layered/initial.png"],
            receipt["artifacts"]["layered/frames/000000.png"],
        )
        self.assertEqual(receipt["solver_diagnostics"]["canonical_steps"], 10)

    def test_assessment_schedule_metrics_and_final_state_do_not_depend_on_movie_cadence(self):
        raw = self.convergence_recipe(assessment=True)
        runner.run(self.args)
        expected = read(self.args.output / "assessment.json")
        _, still = runner.verify_run(self.args.output)
        raw["render"].update(formation_frames=11, fps=30, hold_frames=0, orbit_frames=1)
        write(self.recipe, raw)
        self.args.output, self.args.still_only = self.folder / "different-cadence", False
        runner.run(self.args)
        self.assertEqual(read(self.args.output / "assessment.json"), expected)
        _, film = runner.verify_run(self.args.output)
        self.assertEqual(still["physical_state_sha256"], film["physical_state_sha256"])
        self.assertEqual(still["solver_diagnostics"], film["solver_diagnostics"])

    def test_rehashed_final_assessment_values_are_recomputed_from_native_arrays(self):
        self.convergence_recipe(assessment=True)
        runner.run(self.args)
        path = self.args.output / "assessment.json"
        report = read(path)
        report["final"]["pigments"][0]["mass"] += 0.1
        write(path, report)
        self.rewrite_artifact_hash("assessment.json")
        with self.assertRaisesRegex(ValueError, "Final participation metrics"):
            runner.verify_run(self.args.output)

    def test_mass_budget_archive_binds_starting_pools_final_paint_and_correction_history(self):
        from .mass_budget import VERSION, correction_steps, initial_pool_mass

        class BudgetFixture(FakeEngine):
            def target(self):
                return initial_pool_mass(
                    self.layout, self.config["resolution"], self.config["domain_scale"]
                )

            def snapshot(self, resolution=None):
                fields = super().snapshot(resolution)
                h, w = fields["pigment"].shape[:2]
                area = 4 * self.config["domain_scale"] ** 2 * w / h
                fields["mobile"][...] = self.target() / area
                fields["deposit"][...] = fields["underpaint"][...] = 0
                fields["pigment"] = fields["mobile"].copy()
                return fields

            @property
            def mass_budget_report(self):
                target = self.target().tolist()
                return {
                    "version": VERSION,
                    "interval_steps": self.config["mass_budget_interval_steps"],
                    "initial_mass": target,
                    "corrections": [
                        {
                            "step": step,
                            "mass_before": target,
                            "factors": [1.0] * len(target),
                            "mass_after": target,
                        }
                        for step in correction_steps(
                            self.steps, self.config["mass_budget_interval_steps"]
                        )
                    ],
                }

        raw = self.scatter_recipe()
        raw["simulation"]["mass_budget_interval_steps"] = 3
        write(self.recipe, raw)
        self.args.still_only = False
        with patch("tools.estuary_confluence.engine.Engine", BudgetFixture):
            runner.run(self.args)
        _, receipt = runner.verify_run(self.args.output)
        self.assertIn("mass-budget.json", receipt["artifacts"])
        report_path = self.args.output / "mass-budget.json"
        report = read(report_path)
        self.assertEqual([row["step"] for row in report["corrections"]], [3, 6, 9, 10])
        report["corrections"][0]["factors"][0] = 0.5
        write(report_path, report)
        self.rewrite_artifact_hash("mass-budget.json")
        with self.assertRaisesRegex(ValueError, "restore the initial"):
            runner.verify_run(self.args.output)

    def test_rehashed_checkpoint_schedule_cannot_skip_between_capture_samples(self):
        self.convergence_recipe(assessment=True)
        self.args.still_only = False
        runner.run(self.args)
        path = self.args.output / "assessment.json"
        report = read(path)
        del report["samples"][1]
        write(path, report)
        self.rewrite_artifact_hash("assessment.json")
        with self.assertRaisesRegex(ValueError, "canonical checkpoint"):
            runner.verify_run(self.args.output)

    def test_solver_diagnostic_tampering_is_rejected(self):
        self.convergence_recipe()
        runner.run(self.args)
        path = self.args.output / "receipt.json"
        original = read(path)
        for key, value in (
            ("canonical_steps", 0),
            ("actual_transport_substeps", 9),
            ("maximum_courant", 2.0),
            ("maximum_diffusion_number", 0.5),
            ("diffusion_substeps", 0),
        ):
            receipt = copy.deepcopy(original)
            receipt["solver_diagnostics"][key] = value
            write(path, receipt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.verify_run(self.args.output)
        write(path, original)
        runner.verify_run(self.args.output)

    def test_zero_diffusion_recipe_cannot_advertise_diffusion_work(self):
        runner.run(self.args)
        path = self.args.output / "receipt.json"
        receipt = read(path)
        receipt["solver_diagnostics"]["diffusion_substeps"] = 20
        receipt["solver_diagnostics"]["maximum_diffusion_number"] = 0.18
        write(path, receipt)
        with self.assertRaises(ValueError):
            runner.verify_run(self.args.output)

    def test_legacy_absent_spectra_assessment_and_diagnostics_remain_verifiable(self):
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request.pop("spectral")
        request["recipe"].pop("assessment")
        write(self.args.output / "recipe.json", request["recipe"])
        self.rewrite_artifact_hash("recipe.json")
        self.rewrite_request(request)
        receipt = read(self.args.output / "receipt.json")
        receipt.pop("solver_diagnostics")
        write(self.args.output / "receipt.json", receipt)
        runner.verify_run(self.args.output)

    def test_scattered_still_archives_exact_seeded_geometry_and_initial_image(self):
        self.scatter_recipe()
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["palette"]["mode"], "harmonic")
        self.assertEqual(request["layout"], FakeEngine.instances[-1].layout)
        self.assertEqual(request["layout"], read(self.args.output / "layout.json"))
        self.assertEqual(len(request["layout"]["pools"]), 5)
        self.assertIn("layered/initial.png", receipt["artifacts"])
        self.assertEqual(FakeEngine.instances[-1].visited, [10])
        self.assertEqual(FakeEngine.instances[-1].snapshots[0][0], 0)

    def test_scattered_film_initial_image_is_exact_first_frame(self):
        self.scatter_recipe("random")
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(request["palette"]["mode"], "random")
        self.assertEqual(
            receipt["artifacts"]["layered/initial.png"],
            receipt["artifacts"]["layered/frames/000000.png"],
        )
        self.assertEqual(FakeEngine.instances[-1].visited, [0, 2, 4, 6, 8, 10])

    def test_supersampled_initial_previews_equal_film_frame_zero_and_stills_stay_native(self):
        self.scatter_recipe()
        raw = read(self.recipe)
        raw["render"]["frame_supersampling"] = 2
        raw["render"]["still_resolution"] = [384, 288]
        write(self.recipe, raw)
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        film_initial = (self.args.output / "layered/initial.png").read_bytes()
        self.assertEqual(
            film_initial, (self.args.output / "layered/frames/000000.png").read_bytes()
        )
        self.assertEqual(request["capture"]["frame_raster_resolution"], [256, 192])
        self.assertEqual(request["capture"]["material_sampling"], "full native material grid")
        for surface in FakeSurface.instances:
            self.assertTrue(all(call["output_size"] == (256, 192) for call in surface.calls[:-1]))
            self.assertEqual(surface.calls[-1]["output_size"], (384, 288))
        physical = receipt["physical_state_sha256"]
        poster = (self.args.output / "layered/poster.png").read_bytes()
        self.args.output = self.folder / "supersampled-still"
        self.args.still_only = True
        runner.run(self.args)
        _, still_receipt = runner.verify_run(self.args.output)
        self.assertEqual((self.args.output / "layered/initial.png").read_bytes(), film_initial)
        self.assertEqual((self.args.output / "layered/poster.png").read_bytes(), poster)
        self.assertEqual(still_receipt["physical_state_sha256"], physical)
        self.assertEqual(FakeSurface.instances[-1].calls[0]["output_size"], (256, 192))
        self.assertEqual(FakeSurface.instances[-1].calls[-1]["output_size"], (384, 288))

    def test_rehashed_capture_metadata_cannot_misstate_material_sampling_or_filtering(self):
        runner.run(self.args)
        original = read(self.args.output / "request.json")
        for key, value in [
            ("material_sampling", "area-averaged material"),
            ("frame_supersampling", 2),
            ("frame_raster_resolution", [512, 384]),
        ]:
            request = copy.deepcopy(original)
            request["capture"][key] = value
            self.rewrite_request(request)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Capture resolution"):
                runner.verify_run(self.args.output)
            self.rewrite_request(original)

    def test_legacy_capture_description_is_accepted_only_without_the_new_control(self):
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        request["capture"] = "read-only GPU area averages; final still uses full physical state"
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "legacy capture"):
            runner.verify_run(self.args.output)
        del request["recipe"]["render"]["frame_supersampling"]
        write(self.args.output / "recipe.json", request["recipe"])
        self.rewrite_artifact_hash("recipe.json")
        self.rewrite_request(request)
        runner.verify_run(self.args.output)

    def test_rehashed_layout_cannot_change_bound_positions(self):
        self.scatter_recipe()
        runner.run(self.args)
        path = self.args.output / "layout.json"
        layout = read(path)
        layout["pools"][0]["position"][0] += 0.1
        write(path, layout)
        self.rewrite_artifact_hash("layout.json")
        with self.assertRaisesRegex(ValueError, "starting pools"):
            runner.verify_run(self.args.output)

    def test_self_consistent_palette_edit_cannot_break_seed_derivation(self):
        self.scatter_recipe()
        runner.run(self.args)
        request = read(self.args.output / "request.json")
        palette = request["palette"]
        palette["pigments_srgb"][0][0] += 0.005
        del palette["identity_sha256"]
        palette["identity_sha256"] = hashlib.sha256(
            json.dumps(palette, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        write(self.args.output / "palette.json", palette)
        write(self.args.output / "request.json", request)
        receipt = read(self.args.output / "receipt.json")
        receipt["identity_sha256"] = hashlib.sha256(runner.encoded(request)).hexdigest()
        receipt["artifacts"]["palette.json"] = artifact(self.args.output / "palette.json")
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "derived from its seed"):
            runner.verify_run(self.args.output)

    def test_initial_image_cannot_silently_use_a_later_film_frame(self):
        self.scatter_recipe()
        self.args.still_only = False
        runner.run(self.args)
        name = "layered/initial.png"
        (self.args.output / name).write_bytes(
            (self.args.output / "layered/frames/000005.png").read_bytes()
        )
        self.rewrite_artifact_hash(name)
        with self.assertRaisesRegex(ValueError, "initial state"):
            runner.verify_run(self.args.output)

    def test_palette_modes_and_capture_cadence_preserve_resolved_pool_geometry(self):
        first = self.scatter_recipe()
        second = copy.deepcopy(first)
        second["palette_mode"] = "random"
        second["simulation"]["resolution"] = [256, 192]
        second["render"].update(fps=30, formation_frames=11)
        a, b = runner.validate_recipe(first), runner.validate_recipe(second)
        self.assertEqual(
            runner.resolved_layout(a, "0xbc53af1cd380"),
            runner.resolved_layout(b, "0xbc53af1cd380"),
        )

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
