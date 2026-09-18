"""Borrowed native material capture, reference parity and ownership contracts."""

from __future__ import annotations

import gc
import os
import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .gpu_frame import GPUFrame


class ViewContracts(unittest.TestCase):
    @staticmethod
    def fixture():
        class Owner:
            step = 7

        owner = Owner()
        owner._gpu = SimpleNamespace(ctx=object(), internal_steps=11)
        texture = SimpleNamespace(ctx=owner._gpu.ctx, size=(96, 72), components=4, dtype="f4")
        frame = GPUFrame.capture(
            owner=owner,
            context=owner._gpu.ctx,
            token=(7, 11),
            size=(96, 72),
            pigment_count=4,
            mobile=(texture,),
            deposit=(texture,),
            underpaint=(texture,),
            carrier=texture,
            tooth=texture,
            specific_volumes=(0.1, 0.2, 0.3, 0.9),
            height_scale_mm=1.6,
            substrate_um=0,
            chalk_index=3,
        )
        return owner, frame

    def test_view_is_frozen_and_same_step_internal_advance_invalidates_it(self):
        owner, frame = self.fixture()
        self.assertIs(frame.validate(), frame)
        with self.assertRaises(FrozenInstanceError):
            frame.token = (1, 2)
        owner._gpu.internal_steps += 1
        with self.assertRaisesRegex(RuntimeError, "expired"):
            frame.validate()

    def test_canonical_advance_close_and_owner_collection_invalidate_views(self):
        owner, frame = self.fixture()
        owner.step += 1
        with self.assertRaisesRegex(RuntimeError, "expired"):
            frame.validate()
        owner, frame = self.fixture()
        owner._gpu.ctx = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            frame.validate()
        owner, frame = self.fixture()
        del owner
        gc.collect()
        with self.assertRaisesRegex(RuntimeError, "owner was released"):
            frame.validate()

    def test_bad_channel_packing_and_geometry_are_rejected(self):
        owner, frame = self.fixture()
        for changes in (
            {"size": (48, 36)},
            {"pigment_count": 6},
            {"chalk_index": 4},
            {"specific_volumes": (-1, 0.2, 0.3, 0.9)},
            {"height_scale_mm": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(frame, **changes).validate()
        self.assertTrue(frame.owner_alive())
        self.assertIsNotNone(owner)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL4.3")
class NativeCaptureTests(unittest.TestCase):
    def engine(self, count=3, *, mobile_only=False, **overrides):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette

        palette = generate_palette("0xbc53af1cd380", count)
        config = {
            "resolution": [96, 72],
            "steps": 60,
            "flow_strength": 0.3,
            "pair_swirl": 0.2,
            "carrier_velocity": [0.1, 0],
            "deposition": 0,
            **overrides,
        }
        if mobile_only:
            config.update(
                initial_pattern="scattered",
                underpaint_strength=0,
                settling_scale=0,
                burial_rate=0,
                substrate_um=0,
            )
        engine = Engine(SourceFixture(), config, palette, [])
        self.addCleanup(engine.close)
        return engine

    def surface(self, engine, **config):
        from .surface import Surface

        surface = Surface(config, engine.palette, gpu_frame=engine.gpu_frame())
        self.addCleanup(surface.close)
        return surface

    def test_native_images_match_cpu_reference_for_counts_layers_and_models(self):
        from .run import field_digest

        for count in (3, 5):
            for mobile_only in (False, True):
                engine = self.engine(count, mobile_only=mobile_only)
                engine.advance_to(30)
                fields = engine.snapshot()
                expected_state = field_digest(fields)
                if mobile_only:
                    self.assertFalse(np.any(fields["deposit"]))
                    self.assertFalse(np.any(fields["underpaint"]))
                else:
                    self.assertTrue(np.any(fields["underpaint"]))
                for model in ("rgb", "spectral"):
                    surface = self.surface(
                        engine, optics_model=model, finish="crisp" if mobile_only else "fresco"
                    )
                    for factor in (1, 2):
                        for tilt in (0, 18):
                            gpu = surface.render_gpu(
                                engine.gpu_frame(),
                                (128, 96),
                                tilt_degrees=tilt,
                                supersampling=factor,
                            )
                            full = surface.render(
                                fields, (128 * factor, 96 * factor), tilt_degrees=tilt
                            )
                            cpu = (
                                full
                                if factor == 1
                                else full.reshape(96, 2, 128, 2, 3).mean((1, 3), dtype="f4")
                            )
                            np.testing.assert_allclose(gpu, cpu, atol=2e-6, rtol=2e-5)
                    surface.close()
                self.assertEqual(field_digest(engine.snapshot()), expected_state)
                engine.close()

    def test_native_capture_does_not_call_cpu_snapshot_or_mutate_material(self):
        from .run import field_digest

        engine = self.engine(5, mobile_only=True)
        engine.advance_to(20)
        before = field_digest(engine.snapshot())
        surface = self.surface(engine, optics_model="spectral", finish="crisp")
        with patch.object(engine, "snapshot", side_effect=AssertionError("CPU material readback")):
            image = surface.render_gpu(engine.gpu_frame(), (128, 96), supersampling=2)
        self.assertEqual(image.shape, (96, 128, 3))
        self.assertEqual(field_digest(engine.snapshot()), before)

    def test_borrowed_capture_and_close_do_not_change_solver_continuation_or_final_poster(self):
        from .run import field_digest
        from .surface import Surface

        actual = self.engine(
            5, mobile_only=True, mass_budget_interval_steps=20, diffusion_coefficient=0.001
        )
        expected = self.engine(
            5, mobile_only=True, mass_budget_interval_steps=20, diffusion_coefficient=0.001
        )
        native = self.surface(actual, optics_model="spectral", finish="crisp")
        actual.advance_to(17)
        native.render_gpu(actual.gpu_frame(), (128, 96), supersampling=2)
        actual.advance_to(41)
        native.render_gpu(actual.gpu_frame(), (128, 96), supersampling=2)
        actual.advance_to(60)
        expected.advance_to(60)
        a, b = actual.snapshot(), expected.snapshot()
        self.assertEqual(field_digest(a), field_digest(b))
        with Surface({"optics_model": "spectral", "finish": "crisp"}, actual.palette) as reference:
            wanted = reference.render(b, (256, 192), tilt_degrees=8, azimuth_degrees=35)
            received = native.render(a, (256, 192), tilt_degrees=8, azimuth_degrees=35)
            np.testing.assert_array_equal(received, wanted)
        native.close()
        native.close()
        self.assertEqual(field_digest(actual.snapshot()), field_digest(a))

    def test_expired_views_fail_but_frozen_owned_capture_survives_engine_advance(self):
        engine = self.engine(5, mobile_only=True)
        surface = self.surface(engine, optics_model="spectral", finish="crisp")
        old = engine.gpu_frame()
        first = surface.render_gpu(old, (128, 96), supersampling=2)
        engine.advance_to(engine.step)
        old.validate()
        engine.advance_to(20)
        with self.assertRaisesRegex(RuntimeError, "expired"):
            surface.render_gpu(old, (128, 96))
        frozen = surface.render_gpu(None, (128, 96), supersampling=2)
        np.testing.assert_array_equal(frozen, first)
        surface.render_gpu(engine.gpu_frame(), (128, 96), supersampling=2)

    def test_engine_first_close_is_safe_and_borrowed_surfaces_do_not_release_owner(self):
        engine = self.engine(mobile_only=True)
        surface = self.surface(engine, finish="crisp")
        frame = engine.gpu_frame()
        surface.render_gpu(frame, (128, 96), supersampling=2)
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            surface.render_gpu(None, (128, 96))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            surface.render(None, (128, 96))
        surface.close()
        surface.close()
        with self.assertRaises(RuntimeError):
            frame.validate()

    def test_foreign_views_and_invalid_outputs_fail_without_altering_state(self):
        from .run import field_digest
        from .surface import Surface

        engine = self.engine()
        other = self.engine()
        surface = self.surface(engine)
        before = field_digest(engine.snapshot())
        with self.assertRaisesRegex(ValueError, "different borrowed"):
            surface.render_gpu(other.gpu_frame(), (128, 96))
        for size, factor in (
            ((128, 96), True),
            ((128, 96), 3),
            ((10000, 7500), 2),
            ((128.0, 96), 1),
        ):
            with self.subTest(size=size, factor=factor), self.assertRaises(ValueError):
                surface.render_gpu(engine.gpu_frame(), size, supersampling=factor)
        self.assertEqual(field_digest(engine.snapshot()), before)
        with Surface({}, engine.palette) as owned, self.assertRaisesRegex(ValueError, "borrowed"):
            owned.render_gpu(engine.gpu_frame(), (128, 96))

    def test_invalid_gpu_geometry_is_reported_before_drawing(self):
        from .run import field_digest

        engine = self.engine()
        surface = self.surface(engine)
        before = field_digest(engine.snapshot())
        surface.render_gpu(engine.gpu_frame(), (128, 96))
        bad = replace(engine.gpu_frame(), specific_volumes=(1e6,) * 4)
        with self.assertRaisesRegex(FloatingPointError, "outside supported bounds"):
            surface.render_gpu(bad, (128, 96))
        self.assertEqual(field_digest(engine.snapshot()), before)
        with self.assertRaisesRegex(ValueError, "Upload material"):
            surface.render_gpu(None, (128, 96))
        surface.render_gpu(engine.gpu_frame(), (128, 96))
