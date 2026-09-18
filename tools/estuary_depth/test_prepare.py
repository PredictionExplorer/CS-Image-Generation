"""Actual pigment relief, map orientation and authenticated history contracts."""

from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.estuary.optics import Material
from tools.estuary.recipe import validate_recipe
from tools.estuary.run import artifact, encoded, read_json, write_json
from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_depth.prepare import area_resample, build_bundle, material_maps


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "render"
        (self.run / "inputs").mkdir(parents=True)
        orbit = self.run / "inputs/source.orbit"
        write_orbit(orbit, orbit_points())
        self.source = Source.read(orbit, aspect=4 / 3)
        self.recipe = validate_recipe(
            {
                "simulation": {"resolution": [128, 96], "steps": 360, "domain_scale": 1.6},
                "render": {"resolution": [128, 96], "frames": 181},
            }
        )
        self.request = {
            "recipe": self.recipe,
            "source": self.source.metadata,
            "code": {"engine.py": "f" * 64},
            "frame_steps": list(range(0, 361, 2)),
        }
        self.identity = hashlib.sha256(encoded(self.request)).hexdigest()
        self.state = np.zeros((96, 128, 4), dtype=np.float32)
        self.state[..., 0] = np.arange(128)[None, :] / 128
        self.state[..., 1] = np.arange(96)[:, None] / 96
        self.state[..., 2] = 0.2
        np.save(self.run / "final-state.npy", self.state)
        np.save(self.run / "linear.npy", np.full((96, 128, 3), 0.5, dtype=np.float32))
        write_json(self.run / "request.json", self.request)
        write_json(self.run / "recipe.json", self.recipe)
        self.refresh_receipt()

    def refresh_receipt(self):
        write_json(
            self.run / "receipt.json",
            {
                "complete": True,
                "identity_sha256": self.identity,
                "source": self.source.metadata,
                "final_step": 360,
                "source_fraction": 1.0,
                "artifacts": [
                    artifact(self.run / name, self.run)
                    for name in (
                        "final-state.npy",
                        "linear.npy",
                        "recipe.json",
                        "inputs/source.orbit",
                    )
                ],
            },
        )

    def history(self):
        folder = self.root / "replay"
        folder.mkdir()
        shutil.copyfile(self.run / "final-state.npy", folder / "final-state.npy")
        records = []
        for step, weight in [(126, 0.4), (234, 0.7)]:
            path = folder / f"state-{step}.npy"
            np.save(path, self.state * weight)
            records.append(
                {"step": step, "source_fraction": step / 360, "state": artifact(path, folder)}
            )
        ledger = {
            "schema_version": 1,
            "identity_sha256": self.identity,
            "source_sha256": self.source.sha256,
            "code": self.request["code"],
            "final_state": artifact(folder / "final-state.npy", folder),
            "checkpoints": records,
        }
        write_json(folder / "history.json", ledger)
        return folder / "history.json"

    def build(self, output="bundle", **kwargs):
        return build_bundle(
            self.run, self.root / output, resolution=(64, 48), mesh_resolution=(32, 24), **kwargs
        )

    def test_exact_area_averages_preserve_density_and_bottom_up_rows(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6, 1)
        observed = area_resample(values, (3, 2))
        expected = values.reshape(2, 2, 3, 2, 1).mean(axis=(1, 3))
        np.testing.assert_array_equal(observed, expected)
        self.assertLess(observed[0, 0, 0], observed[-1, 0, 0])
        np.testing.assert_allclose(area_resample(values, (5, 3)).mean(), values.mean(), atol=1e-6)
        np.testing.assert_array_equal(area_resample(values, (6, 4)), values)

    def test_full_guard_domain_and_authored_mass_height(self):
        manifest = self.build()
        with np.load(self.root / "bundle/bundle.npz", allow_pickle=False) as maps:
            expected = self.state[..., :3].reshape(24, 4, 32, 4, 3).mean(axis=(1, 3))
            np.testing.assert_allclose(
                maps["height"], expected @ np.array([0.25, 1, 0.5]), atol=1e-7
            )
            self.assertEqual(maps["density"].shape, (48, 64, 3))
            self.assertEqual(maps["history_height"].shape, (1, 24, 32))
            np.testing.assert_array_equal(maps["history_times"], [1])
            self.assertLess(maps["density"][0, 0, 0], 0.01)  # Guard edge is retained.
        self.assertEqual(manifest["coordinates"]["row_order"], "bottom-to-top")
        self.assertEqual(manifest["domain_scale"], 1.6)
        self.assertEqual(manifest["coordinates"]["domain_bounds"][1][1], 1.6)

    def test_height_ignores_display_rgb_and_scales_with_concentration(self):
        self.build("before")
        np.save(self.run / "linear.npy", np.zeros((96, 128, 3), dtype=np.float32))
        self.refresh_receipt()
        self.build("after")
        with (
            np.load(self.root / "before/bundle.npz") as before,
            np.load(self.root / "after/bundle.npz") as after,
        ):
            np.testing.assert_array_equal(before["height"], after["height"])
            np.testing.assert_array_equal(before["color_linear"], after["color_linear"])
        a = material_maps(self.state, (32, 24), (16, 12), 1, Material(), [0.25, 1, 0.5])
        b = material_maps(
            self.state * 2,
            (32, 24),
            (16, 12),
            1,
            Material(substrate_srgb=(0, 0, 0)),
            [0.25, 1, 0.5],
        )
        np.testing.assert_allclose(b["height"], a["height"] * 2, atol=1e-7)

    def test_verified_replay_preserves_history_order_and_actual_states(self):
        ledger = self.history()
        manifest = self.build(history_manifest=ledger, history_fractions=(0.35, 0.65))
        with np.load(self.root / "bundle/bundle.npz") as maps:
            np.testing.assert_array_equal(maps["history_times"], [0.35, 0.65, 1])
            np.testing.assert_allclose(maps["history_height"][0], maps["height"] * 0.4, atol=1e-7)
            np.testing.assert_allclose(maps["history_height"][1], maps["height"] * 0.7, atol=1e-7)
        self.assertTrue(manifest["history"]["older_history_available"])

    def test_requested_history_requires_evidence(self):
        with self.assertRaisesRegex(ValueError, "verified replay ledger"):
            self.build(history_fractions=(0.35, 0.65))

    def test_rejects_corrupted_source_state_linear_and_history(self):
        for name in ["inputs/source.orbit", "final-state.npy", "linear.npy"]:
            path = self.run / name
            original = path.read_bytes()
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "changed or missing"):
                self.build()
            path.write_bytes(original)
        ledger = self.history()
        path = ledger.parent / "state-126.npy"
        np.save(path, self.state)
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            self.build(history_manifest=ledger, history_fractions=(0.35,))

    def test_rejects_history_from_different_code_or_false_replay(self):
        path = self.history()
        original = read_json(path)
        for field, value in [("code", {}), ("source_sha256", "x"), ("identity_sha256", "x")]:
            changed = {**original, field: value}
            write_json(path, changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(history_manifest=path, history_fractions=(0.35,))
        np.save(path.parent / "final-state.npy", self.state * 0.9)
        original["final_state"] = artifact(path.parent / "final-state.npy", path.parent)
        write_json(path, original)
        with self.assertRaisesRegex(ValueError, "Replay final state differs"):
            self.build(history_manifest=path, history_fractions=(0.35,))

    def test_negative_and_nonfinite_concentrations_fail_even_with_matching_hash(self):
        for value in [-1, np.inf, np.nan]:
            state = self.state.copy()
            state[0, 0, 0] = value
            np.save(self.run / "final-state.npy", state)
            self.refresh_receipt()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "concentrations"):
                self.build()

    def test_reuse_verifies_identity_and_bundle_hash(self):
        original = self.build()
        self.assertEqual(original, self.build())
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.build(specific_volumes=(1, 1, 1))
        path = self.root / "bundle/bundle.npz"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            self.build()

    def test_mapping_and_source_metadata_are_bound_to_preparation_identity(self):
        original = self.build()
        for key in ("domain_scale", "view_aspect", "coordinates"):
            self.assertEqual(original[key], original["request"]["geometry"][key])
        self.assertEqual(original["source"], original["request"]["source"])
        path = self.root / "bundle/manifest.json"
        for key, value in [
            ("domain_scale", 2),
            ("view_aspect", 1),
            ("coordinates", {"row_order": "top-to-bottom"}),
        ]:
            changed = copy.deepcopy(original)
            changed[key] = value
            write_json(path, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mapping metadata"):
                self.build()
        changed = copy.deepcopy(original)
        changed["source"] = {**changed["source"], "seed": "mislabelled"}
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "source metadata"):
            self.build()
        changed = copy.deepcopy(original)
        changed["request"]["geometry"]["domain_scale"] = 2
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "preparation request"):
            self.build()


if __name__ == "__main__":
    unittest.main()
