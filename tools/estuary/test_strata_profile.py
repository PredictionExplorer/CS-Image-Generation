"""Opt-in starting-band geometry and certified state-zero image capture."""

from __future__ import annotations

import copy
import hashlib
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from . import run as runner
from .engine import Engine
from .optics import linear_to_srgb
from .recipe import (
    DEFAULTS,
    STRATA_PROFILE_DEFAULTS,
    canonical_bytes,
    validate_recipe,
    validate_strata_profile,
)
from .test_run import decode_png

OMITTED = object()
RELEASED_INITIAL_SHA256 = "3563c0bae694b25db5749b373cf9f0197684a51bbe851b1d8ac98fd1a6d3c97b"


class HorizontalSource:
    def frame(self, fraction):
        speeds = np.array([[0.3, 0.1], [-0.2, 0.3], [0.2, -0.2]])
        positions = np.array([[-0.5, 0], [0.5, 0], [0, 0]]) + fraction * speeds
        return SimpleNamespace(
            positions=positions,
            velocities=speeds,
            pair_distances=np.array(
                [np.linalg.norm(positions[b] - positions[a]) for a, b in ((0, 1), (1, 2), (2, 0))]
            ),
            arc_lengths=np.linalg.norm(speeds, axis=1) * fraction,
        )


def recipe(profile=OMITTED, *, initial_image=False, size=(512, 384)):
    simulation = {
        "resolution": list(size),
        "steps": 360,
        "domain_scale": 1.6,
        "initial_pattern": "strata",
        "initial_load": 0.6,
        "load_radius": 0.28,
        "deposition": 0,
    }
    if profile is not OMITTED:
        simulation["strata_profile"] = profile
    return validate_recipe(
        {
            "simulation": simulation,
            "render": {"resolution": list(size), "frames": 31, "initial_image": initial_image},
        }
    )


def initial_state(resolved):
    """Execute the actual CPU initializer with an in-memory texture upload sink."""
    engine = Engine.__new__(Engine)
    engine.recipe = resolved
    engine.width, engine.height = resolved["simulation"]["resolution"]
    engine.aspect, engine.domain = (
        engine.width / engine.height,
        resolved["simulation"]["domain_scale"],
    )
    engine.source = HorizontalSource()
    buffers = []
    engine.paint = [SimpleNamespace(write=buffers.append)]
    engine._initialize()
    return np.frombuffer(buffers[0], "f4").reshape(engine.height, engine.width, 4).copy()


class StrataProfileTests(unittest.TestCase):
    def test_neutral_options_preserve_legacy_recipe_bytes_and_released_pixels(self):
        baseline = recipe()
        state = initial_state(baseline)
        self.assertEqual(hashlib.sha256(state.tobytes()).hexdigest(), RELEASED_INITIAL_SHA256)
        self.assertNotIn("strata_profile", DEFAULTS["simulation"])
        self.assertNotIn("initial_image", DEFAULTS["render"])
        for profile in (None, {}, STRATA_PROFILE_DEFAULTS, {"fine_width_scale": 1}):
            current = recipe(profile)
            self.assertEqual(canonical_bytes(current), canonical_bytes(baseline))
            np.testing.assert_array_equal(initial_state(current), state)
        self.assertEqual(
            validate_recipe({"simulation": {"strata_profile": STRATA_PROFILE_DEFAULTS}}),
            validate_recipe({}),
        )

    def test_active_profile_is_versioned_strict_and_only_applies_to_strata(self):
        profile = validate_strata_profile({"fine_width_scale": 0})
        self.assertEqual(profile, {**STRATA_PROFILE_DEFAULTS, "fine_width_scale": 0.0})
        for value in (
            False,
            [],
            {"unknown": 1},
            {"version": "future"},
            {"main_width_scale": 0.24},
            {"accent_width_scale": 2.1},
            {"fine_width_scale": -0.1},
            {"fine_width_scale": True},
            {"main_width_scale": float("nan")},
            {"fine_width_scale": 10**400},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_strata_profile(value)
        with self.assertRaisesRegex(ValueError, "initial_pattern=strata"):
            validate_recipe({"simulation": {"strata_profile": {"fine_width_scale": 0}}})

    def test_all_profiles_remain_a_uniform_nonnegative_three_pigment_partition(self):
        for profile in (
            {"main_width_scale": 0.25, "fine_width_scale": 0, "accent_width_scale": 0.25},
            {"main_width_scale": 2, "fine_width_scale": 2, "accent_width_scale": 2},
            {"main_width_scale": 0.65, "fine_width_scale": 0.4, "accent_width_scale": 0.7},
        ):
            values = initial_state(recipe(profile))
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(float(values.min()), 0)
            self.assertFalse(values[..., 3].any())
            np.testing.assert_allclose(values[..., :3].sum(-1), 0.6, rtol=0, atol=1e-7)

    def test_hairline_removal_does_not_move_or_change_the_main_or_accent_bands(self):
        baseline = initial_state(recipe())
        clean = initial_state(recipe({"fine_width_scale": 0}))
        y = ((np.arange(384) + 0.5) / 384 * 2 - 1) * 1.6
        for center in (-0.28 * 1.65, 0.28 * 2.15):
            row = int(np.argmin(abs(y - center)))
            self.assertGreater(float(baseline[row, 256, 1]), 0.59)
            self.assertEqual(float(clean[row, 256, 1]), 0)
        np.testing.assert_array_equal(clean[abs(y) < 0.3, :, 1], baseline[abs(y) < 0.3, :, 1])
        np.testing.assert_array_equal(clean[..., 2], baseline[..., 2])

    def test_narrower_main_and_accent_stay_centered_and_do_not_change_other_controls(self):
        baseline = recipe()
        narrow = recipe({"main_width_scale": 0.5, "accent_width_scale": 0.5})
        stripped = copy.deepcopy(narrow)
        del stripped["simulation"]["strata_profile"]
        self.assertEqual(stripped, baseline)
        original, changed = initial_state(baseline), initial_state(narrow)
        y = ((np.arange(384) + 0.5) / 384 * 2 - 1) * 1.6
        main = abs(y) < 0.3
        self.assertLess(float(changed[main, :, 1].sum()), 0.55 * float(original[main, :, 1].sum()))
        self.assertLess(float(changed[..., 2].sum()), 0.6 * float(original[..., 2].sum()))
        for field in (original, changed):
            weight = field[:, 256, 2]
            self.assertAlmostEqual(float(y @ weight / weight.sum()), 0.28 * 1.45, delta=0.009)

    def test_initial_image_is_boolean_and_false_is_canonical_omission(self):
        plain = recipe()
        self.assertNotIn("initial_image", plain["render"])
        self.assertTrue(recipe(initial_image=True)["render"]["initial_image"])
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                validate_recipe({"render": {"initial_image": value}})


class InitialImageArchiveTests(unittest.TestCase):
    def test_fresh_initial_image_uses_state_zero_and_is_required_by_its_receipt(self):
        class FakeEngine:
            def __init__(self):
                self.step = self.internal_steps = 0
                self.maximum_courant = 0.0
                self.metadata = {"renderer": "state-zero archive fixture"}
                self.events = []

            def render(self, width, height):
                self.events.append(("render", self.step))
                return np.full((height, width, 3), 0.1 if self.step == 0 else 0.7, dtype="f4")

            def advance_to(self, step):
                self.events.append(("advance", step))
                self.step = self.internal_steps = step

            def read_state(self):
                return np.full((96, 128, 4), self.step / 360, dtype="f4")

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            orbit = folder / "source.orbit"
            orbit.write_bytes(b"bound source fixture")
            source = SimpleNamespace(
                path=orbit, sha256=runner.digest(orbit), seed="0x1", metadata={"seed": "0x1"}
            )
            args = SimpleNamespace(
                output=folder / "enabled", backend="fixture", still_only=True, resume=False
            )
            enabled, plain = FakeEngine(), FakeEngine()
            with (
                patch.object(runner, "code_identity", return_value={}),
                redirect_stdout(io.StringIO()),
            ):
                runner._render_archive(
                    args,
                    enabled,
                    source,
                    recipe(initial_image=True, size=(128, 96)),
                    [],
                    {},
                    time.monotonic(),
                )
                args.output = folder / "plain"
                runner._render_archive(
                    args, plain, source, recipe(size=(128, 96)), [], {}, time.monotonic()
                )
            self.assertEqual(enabled.events, [("render", 0), ("advance", 360), ("render", 360)])
            self.assertEqual(plain.events, [("advance", 360), ("render", 360)])
            self.assertFalse((folder / "plain/initial.png").exists())
            self.assertEqual(
                (folder / "enabled/final-state.npy").read_bytes(),
                (folder / "plain/final-state.npy").read_bytes(),
            )
            pixels, _ = decode_png(folder / "enabled/initial.png")
            expected = np.rint(
                linear_to_srgb(np.full((96, 128, 3), np.float32(0.1))) * 65535
            ).astype("u2")
            np.testing.assert_array_equal(pixels, expected)
            receipt = runner.read_json(folder / "enabled/receipt.json")
            self.assertTrue(runner.completed(folder / "enabled", receipt["identity_sha256"]))
            receipt["artifacts"] = [
                row for row in receipt["artifacts"] if row["path"] != "initial.png"
            ]
            runner.write_json(folder / "enabled/receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "required artifacts"):
                runner.completed(folder / "enabled", receipt["identity_sha256"])

    def test_film_checkpoint_binds_initial_image_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "frames").mkdir()
            runner.write_png(folder / "initial.png", np.full((4, 4, 3), 0.2, dtype="f4"))
            runner.write_png(folder / "frames/000000.png", np.full((4, 4, 3), 0.2, dtype="f4"), 8)
            engine = SimpleNamespace(
                read_state=lambda: np.zeros((4, 4, 4), "f4"),
                internal_steps=0,
                maximum_courant=0,
                restore=Mock(),
            )
            runner.checkpoint(
                folder,
                "identity",
                engine,
                0,
                0,
                [runner.artifact(folder / "frames/000000.png", folder)],
                initial_image=True,
            )
            runner.restore_checkpoint(folder, "identity", engine, [0, 360], initial_image=True)
            engine.restore.assert_called_once()
            (folder / "initial.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                runner.restore_checkpoint(folder, "identity", engine, [0, 360], initial_image=True)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL4.3")
class StrataProfileGPUTests(unittest.TestCase):
    def test_neutral_profile_and_initial_render_preserve_complete_evolution(self):
        baseline = Engine(HorizontalSource(), recipe(size=(128, 96)))
        self.addCleanup(baseline.close)
        initial = baseline.read_state()
        baseline.advance_to(360)
        final = baseline.read_state()
        counters = baseline.internal_steps, baseline.maximum_courant
        baseline.close()
        neutral = Engine(
            HorizontalSource(), recipe(STRATA_PROFILE_DEFAULTS, initial_image=True, size=(128, 96))
        )
        self.addCleanup(neutral.close)
        np.testing.assert_array_equal(neutral.read_state(), initial)
        neutral.render(128, 96)
        self.assertEqual(neutral.step, 0)
        np.testing.assert_array_equal(neutral.read_state(), initial)
        neutral.advance_to(360)
        self.assertEqual((neutral.internal_steps, neutral.maximum_courant), counters)
        np.testing.assert_array_equal(neutral.read_state(), final)

    def test_active_profile_uploads_the_exact_partition_without_changing_flow(self):
        active = recipe(
            {"main_width_scale": 0.65, "fine_width_scale": 0, "accent_width_scale": 0.7},
            size=(128, 96),
        )
        changed = Engine(HorizontalSource(), active)
        self.addCleanup(changed.close)
        np.testing.assert_array_equal(changed.read_state(), initial_state(active))
        with changed.ctx:
            changed._flow(0.37)
            velocity = changed.velocity.read()
        changed.close()
        control = Engine(HorizontalSource(), recipe(size=(128, 96)))
        self.addCleanup(control.close)
        with control.ctx:
            control._flow(0.37)
            self.assertEqual(control.velocity.read(), velocity)


if __name__ == "__main__":
    unittest.main()
