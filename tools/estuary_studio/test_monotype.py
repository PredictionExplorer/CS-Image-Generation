"""Material accounting, fixed-plane pressure and native GPU clock contracts."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit

from .monotype import (
    PlaneContact,
    contact_pressure,
    contact_texture,
    local_transfer,
    update_load,
    validate_config,
)


class MaterialTests(unittest.TestCase):
    def test_loaded_tool_bridges_tooth_and_dry_tool_breakup_remains_bounded(self):
        tooth = np.linspace(0, 1, 257)
        loaded = contact_texture(tooth, 0.5, 1.0, 0.65, 0.35)
        dry = contact_texture(tooth, 0.5, 0.0, 0.65, 0.35)
        self.assertLess(float(np.ptp(loaded)), 1e-14)
        self.assertGreater(float(loaded.min()), 0.97)
        self.assertGreater(float(np.var(dry)), 0.001)
        self.assertTrue(((dry >= 0) & (dry <= 1)).all())
        np.testing.assert_array_equal(contact_texture(tooth, 0.5, 0, 0.65, 0), np.ones(257))
        for load in (0, 0.25, 0.5, 0.75, 1):
            value = contact_texture(tooth, tooth[::-1], load, 0.65, 1)
            self.assertTrue(((value >= 0) & (value <= 1)).all())

    def test_local_transfer_accounts_for_deposit_and_pickup_without_negative_amounts(self):
        paint = np.array([0.3, 0.8, 0.12])
        deposit = np.array([0.2, 0, 0.01])
        result, lifted = local_transfer(paint, deposit, 0.7)
        np.testing.assert_allclose(result + lifted, paint + deposit, atol=1e-15)
        self.assertTrue((result >= 0).all())
        np.testing.assert_array_equal(local_transfer(paint, deposit, 1)[0], deposit)
        for bad in (-0.1, 1.01, np.nan):
            with self.assertRaises(ValueError):
                local_transfer(paint, deposit, bad)

    def test_reload_only_occurs_while_lifted_and_contact_depletes(self):
        config = validate_config()
        original = np.array([0.3, 0.5, 1.0])
        loaded = update_load(original, np.zeros(3), np.ones(3), 0.4, config)
        depleted = update_load(original, np.ones(3), np.ones(3), 0.4, config)
        self.assertTrue((loaded >= original).all())
        self.assertTrue((loaded <= config["load_capacity"]).all())
        self.assertTrue((depleted < original).all())
        np.testing.assert_array_equal(
            update_load(original, [0, 1, 0.5], [0, 0, 0], 0, config), original
        )

    def test_pressure_is_symmetric_finite_and_vanishes_outside_contact_band(self):
        depth = np.linspace(0, 2, 101)
        pressure = contact_pressure(depth, np.zeros(101))
        np.testing.assert_array_equal(pressure, contact_pressure(-depth, np.zeros(101)))
        self.assertEqual(pressure[0], 1)
        self.assertEqual(pressure[-1], 0)
        self.assertTrue((np.diff(pressure) <= 0).all())
        self.assertLess(contact_pressure(0, 100), contact_pressure(0, 0))
        with self.assertRaises(ValueError):
            contact_pressure(0, -1)

    def test_strict_config_rejects_invalid_inputs_and_does_not_mutate_defaults(self):
        for bad in (
            {"kind": "other"},
            {"brush_width": float("nan")},
            {"steps": True},
            {"resolution": [128, True]},
            {"palette": "blue"},
            {"domain_scale": 0.5},
        ):
            with self.assertRaises(ValueError):
                validate_config(bad)
        first = validate_config()
        first["resolution"][0] = 200
        self.assertEqual(validate_config()["resolution"][0], 1024)


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "source.orbit"
        self.points = orbit_points(257)
        write_orbit(self.path, self.points)
        self.source = Source.read(self.path, aspect=4 / 3)

    def test_depth_uses_raw_third_axis_and_full_recording_scale(self):
        contact = PlaneContact(self.source)
        p = self.source.projection
        axes = np.array(p["axes"])
        normal = np.cross(axes[:, 0], axes[:, 1])
        expected = ((self.points - p["origin"]) / p["extent"] - p["mean"]) @ normal
        expected /= np.quantile(np.abs(expected), 0.90)
        np.testing.assert_allclose(contact.depths, expected, atol=1e-14)
        np.testing.assert_allclose(contact.sample(0), expected[0], atol=1e-14)
        np.testing.assert_allclose(contact.sample(1), expected[-1], atol=1e-14)
        np.testing.assert_allclose(contact.sample(0.1234), contact.sample(0.1234), atol=0)

    def test_pressure_invariant_to_raw_translation_rotation_scale(self):
        first = PlaneContact(self.source)
        rotation, _ = np.linalg.qr(np.array([[1.1, 0.4, 0.7], [0.3, -0.6, 0.8], [-0.8, 0.7, 0.5]]))
        transformed = self.points @ rotation * 7 + [4, -5, 8]
        path = Path(self.temp.name) / "transformed.orbit"
        write_orbit(path, transformed)
        second = PlaneContact(Source.read(path, aspect=4 / 3))
        np.testing.assert_allclose(np.abs(first.depths), np.abs(second.depths), atol=2e-12)
        np.testing.assert_allclose(
            contact_pressure(first.depths, 2), contact_pressure(second.depths, 2), atol=2e-12
        )


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class MonotypeGpuTests(ContactTests):
    def engine(self, **overrides):
        from .monotype import Monotype

        engine = Monotype(self.source, {"resolution": [128, 96], "steps": 120, **overrides})
        self.addCleanup(engine.close)
        return engine

    def test_native_loaded_contact_bridges_tooth_while_depleted_contact_varies(self):
        def contact_row(load):
            engine = self.engine(resolution=[256, 192], initial_load=0, bristle_strength=0.35)
            with engine.ctx:
                engine.shader["u_segments"].write(np.zeros((3, 4), dtype="f4").tobytes())
                tools = np.zeros((3, 4), dtype="f4")
                tools[0] = [1, load, 0.01, 0]
                engine.shader["u_tools"].write(tools.tobytes())
                engine.shader["u_directions"].write(
                    np.tile(np.array([1, 0], dtype="f4"), (3, 1)).tobytes()
                )
                engine.shader["u_dt"].value = 0
                engine.paint[0].use(0)
                engine.surface[0].use(1)
                engine.paint[1].bind_to_image(0, read=False, write=True)
                engine.surface[1].bind_to_image(1, read=False, write=True)
                engine.shader.run(*engine.groups)
                engine.ctx.memory_barrier()
                engine.index = 1
            # Same bristle, well inside the footprint, crossing support tooth.
            return engine.snapshot()["pigment"][96, 124:132, 0]

        loaded, depleted = contact_row(1.0), contact_row(0.2)
        self.assertGreater(float(loaded.mean()), 0)
        self.assertLess(float(np.ptp(loaded)), 1e-8)
        self.assertGreater(float(np.ptp(depleted)), 1e-6)
        self.assertTrue((depleted >= 0).all())

    def test_state_bounds_full_source_and_material_response(self):
        from .common import check_fields

        engine = self.engine()
        initial = engine.snapshot()
        engine.advance_to(engine.steps)
        state = engine.snapshot()
        check_fields(state, (128, 96))
        self.assertEqual(engine.step, 120)
        self.assertFalse(np.array_equal(initial["pigment"], state["pigment"]))
        self.assertGreater(
            float(state["pigment"][:, :, 1].sum()), float(initial["pigment"][:, :, 1].sum())
        )
        self.assertLessEqual(engine.max_segment_travel, engine.config["brush_width"] * 0.20 + 1e-10)
        with self.assertRaises(ValueError):
            engine.advance_to(0)

    def test_render_cadence_and_interleaved_contexts_cannot_change_evolution(self):
        direct = self.engine()
        direct.advance_to(120)
        expected = direct.snapshot()
        segmented = self.engine()
        segmented.advance_to(20)
        segmented.snapshot()
        other = self.engine(kind="nocturne")
        other.advance_to(40)
        segmented.advance_to(73)
        segmented.snapshot()
        segmented.advance_to(120)
        for key, value in expected.items():
            np.testing.assert_array_equal(segmented.snapshot()[key], value)

    def test_lifting_actually_removes_material_and_nocturne_changes_direction(self):
        static = self.engine(deposit_rate=0, drag_strength=0, lift_rate=0)
        static.advance_to(120)
        fixed_mass = static.snapshot()["pigment"].sum(dtype=np.float64)
        lifting = self.engine(deposit_rate=0, drag_strength=0, lift_rate=2)
        lifting.advance_to(120)
        self.assertLess(lifting.snapshot()["pigment"].sum(dtype=np.float64), fixed_mass)
        nocturne = self.engine(kind="nocturne")
        initial = nocturne.snapshot()
        nocturne.advance_to(120)
        self.assertFalse(np.array_equal(initial["direction"], nocturne.snapshot()["direction"]))
        self.assertFalse(np.array_equal(initial["roughness"], nocturne.snapshot()["roughness"]))


if __name__ == "__main__":
    unittest.main()
