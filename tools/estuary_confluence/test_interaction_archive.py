"""Persistent interaction state, native-grid identity and playback independence.

These CPU fixtures exercise the production archive/verification path. Numerical
transport and optical correctness are tested separately against the real engine.
"""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary_studio.common import artifact, read, write
from tools.estuary_studio.run import record_array

from . import run as runner
from . import test_run as fixtures
from .interaction import BASE_FIELDS, FIELD_NAMES, VERSION


class InteractionEngineFixture(fixtures.FakeEngine):
    def snapshot(self, resolution=None):
        fields = super().snapshot(resolution)
        if self.config.get("interaction") is None:
            return fields
        height, width = fields["height"].shape
        x, y = np.meshgrid(
            np.linspace(-1, 1, width, dtype="f4"), np.linspace(-1, 1, height, dtype="f4")
        )
        origin = np.stack((x, y), axis=-1)
        history = np.empty((height, width, 4), dtype="f4")
        history[...] = np.array([0.03, 0.018, 0.006, 0.02], dtype="f4") * self.step
        fields.update(
            origin_upper=origin.copy(),
            origin_lower=origin + np.float32(self.step * 0.01),
            interaction_upper=history.copy(),
            interaction_lower=history * np.float32(0.5),
        )
        return fields


class InteractionArchiveTests(unittest.TestCase):
    # Share the established CPU archive harness, without inheriting its tests.
    source_info = staticmethod(fixtures.PipelineTests.source_info)
    engaged_layout = staticmethod(fixtures.PipelineTests.engaged_layout)
    movie = staticmethod(fixtures.PipelineTests.movie)
    rewrite_artifact_hash = fixtures.PipelineTests.rewrite_artifact_hash
    rewrite_request = fixtures.PipelineTests.rewrite_request

    def setUp(self):
        fixtures.PipelineTests.setUp(self)
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
        write(self.recipe, self.raw)
        self.patches.enter_context(
            patch("tools.estuary_confluence.engine.Engine", InteractionEngineFixture)
        )

    def fields(self):
        with np.load(self.args.output / "final.npz", allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def replace_fields(self, fields, *, rebind_state=False):
        record_array(self.args.output / "final.npz", fields)
        receipt = read(self.args.output / "receipt.json")
        receipt["artifacts"]["final.npz"] = artifact(self.args.output / "final.npz")
        if rebind_state:
            receipt["physical_state_sha256"] = runner.field_digest(fields)
            receipt["base_material_sha256"] = runner.base_material_digest(fields)
            for look in receipt["looks"].values():
                look["physical_state_sha256"] = receipt["physical_state_sha256"]
        write(self.args.output / "receipt.json", receipt)

    def test_native_raw_fields_and_two_distinct_material_identities_are_archived(self):
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        actual = self.fields()
        expected = fixtures.FakeEngine.instances[-1].snapshot()
        self.assertEqual(set(actual), set(BASE_FIELDS) | set(FIELD_NAMES))
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(actual[key].dtype, np.float32)
        self.assertEqual(receipt["physical_state_sha256"], runner.field_digest(actual))
        self.assertEqual(receipt["base_material_sha256"], runner.base_material_digest(actual))
        self.assertNotEqual(receipt["physical_state_sha256"], receipt["base_material_sha256"])
        self.assertEqual(receipt["interaction"], request["interaction"])
        self.assertEqual(request["interaction"]["version"], VERSION)
        self.assertEqual(request["interaction"]["seed"], request["palette"]["seed"])
        self.assertEqual(
            request["interaction"]["settings"], request["recipe"]["simulation"]["interaction"]
        )

    def test_base_identity_can_match_while_complete_material_identity_changes(self):
        runner.run(self.args)
        _, microstructure = runner.verify_run(self.args.output)
        self.raw["simulation"].pop("interaction")
        write(self.recipe, self.raw)
        self.args.output = self.folder / "baseline"
        runner.run(self.args)
        request, baseline = runner.verify_run(self.args.output)
        self.assertEqual(microstructure["base_material_sha256"], baseline["physical_state_sha256"])
        self.assertNotEqual(
            microstructure["physical_state_sha256"], baseline["physical_state_sha256"]
        )
        self.assertNotIn("interaction", request)
        self.assertNotIn("interaction", baseline)
        self.assertNotIn("base_material_sha256", baseline)
        self.assertEqual(set(self.fields()), set(BASE_FIELDS))

    def test_every_interaction_array_is_required_and_bound_to_complete_hash(self):
        runner.run(self.args)
        original = self.fields()
        for name in FIELD_NAMES:
            with self.subTest(field=name, problem="missing"):
                absent = {key: value for key, value in original.items() if key != name}
                self.replace_fields(absent)
                with self.assertRaisesRegex(ValueError, "fields differ"):
                    runner.verify_run(self.args.output)
            with self.subTest(field=name, problem="changed"):
                changed = copy.deepcopy(original)
                channel = 0 if name.startswith("origin") else 3
                changed[name][0, 0, channel] += 0.01
                self.replace_fields(changed)
                with self.assertRaisesRegex(ValueError, "Physical state identity"):
                    runner.verify_run(self.args.output)

    def test_rehashed_malformed_history_still_fails_numeric_validation(self):
        runner.run(self.args)
        original = self.fields()
        edits = [
            ("origin_upper", lambda value: value.astype("f8")),
            ("origin_lower", lambda value: value[..., :1]),
            ("interaction_upper", lambda value: value[..., :3]),
            ("interaction_lower", lambda value: np.full_like(value, np.nan)),
            ("interaction_upper", lambda value: np.full_like(value, -0.1)),
            ("interaction_lower", lambda value: np.full_like(value, 1.1)),
        ]
        for name, edit in edits:
            with self.subTest(field=name):
                changed = copy.deepcopy(original)
                changed[name] = edit(changed[name])
                self.replace_fields(changed, rebind_state=True)
                with self.assertRaises(ValueError):
                    runner.verify_run(self.args.output)
        changed = copy.deepcopy(original)
        changed["interaction_upper"][..., 1:3] = 0.9
        self.replace_fields(changed, rebind_state=True)
        with self.assertRaises(ValueError):
            runner.verify_run(self.args.output)

    def test_reduced_snapshot_cannot_replace_native_final_material(self):
        runner.run(self.args)
        reduced = fixtures.FakeEngine.instances[-1].snapshot((64, 48))
        self.replace_fields(reduced, rebind_state=True)
        with self.assertRaisesRegex(ValueError, "full native material grid"):
            runner.verify_run(self.args.output)

    def test_rehashed_metadata_must_match_seed_version_and_settings(self):
        runner.run(self.args)
        original_request = read(self.args.output / "request.json")
        original_receipt = read(self.args.output / "receipt.json")
        edits = {
            "seed": "0x123456",
            "version": "invented-model-v9",
            "settings": {"invented": 1},
            "resolution": [64, 48],
            "dtype": "float64",
            "fields": {},
        }
        for key, replacement in edits.items():
            with self.subTest(key=key):
                request, receipt = copy.deepcopy(original_request), copy.deepcopy(original_receipt)
                request["interaction"][key] = replacement
                receipt["interaction"][key] = replacement
                write(self.args.output / "receipt.json", receipt)
                self.rewrite_request(request)
                with self.assertRaisesRegex(ValueError, "Interaction version, seed"):
                    runner.verify_run(self.args.output)

    def test_base_hash_is_verified_independently_from_complete_hash(self):
        runner.run(self.args)
        receipt = read(self.args.output / "receipt.json")
        receipt["base_material_sha256"] = receipt["physical_state_sha256"]
        write(self.args.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Base material identity"):
            runner.verify_run(self.args.output)

    def test_enabled_metadata_and_base_hash_cannot_be_omitted(self):
        runner.run(self.args)
        original = read(self.args.output / "receipt.json")
        for key in ("interaction", "base_material_sha256"):
            with self.subTest(key=key):
                receipt = copy.deepcopy(original)
                receipt.pop(key)
                write(self.args.output / "receipt.json", receipt)
                with self.assertRaises(ValueError):
                    runner.verify_run(self.args.output)

    def test_disabled_simulation_rejects_unexplained_fields_and_metadata(self):
        runner.run(self.args)
        microstructure = self.fields()
        metadata = read(self.args.output / "request.json")["interaction"]
        self.raw["simulation"].pop("interaction")
        write(self.recipe, self.raw)
        self.args.output = self.folder / "baseline"
        runner.run(self.args)
        self.replace_fields(microstructure)
        with self.assertRaisesRegex(ValueError, "fields differ"):
            runner.verify_run(self.args.output)
        self.replace_fields({key: microstructure[key] for key in BASE_FIELDS})
        request = read(self.args.output / "request.json")
        request["interaction"] = metadata
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "Disabled interaction"):
            runner.verify_run(self.args.output)

    def test_film_cadence_and_camera_views_preserve_full_microstructured_state(self):
        runner.run(self.args)
        _, still = runner.verify_run(self.args.output)
        self.raw["render"].update(formation_frames=11, fps=30, hold_frames=2, orbit_frames=4)
        write(self.recipe, self.raw)
        self.args.output, self.args.still_only = self.folder / "film", False
        runner.run(self.args)
        request, film = runner.verify_run(self.args.output)
        for key in ("physical_state_sha256", "base_material_sha256", "interaction"):
            self.assertEqual(still[key], film[key])
        self.assertEqual(
            set(request["capture"]["interaction_material"]["fields"]), set(FIELD_NAMES)
        )
        engine = fixtures.FakeEngine.instances[-1]
        self.assertEqual(engine.visited, list(range(11)))
        for surface in fixtures.FakeSurface.instances[-2:]:
            final_calls = surface.calls[10:]
            self.assertEqual(
                [call["physical_sha256"] for call in final_calls],
                [film["physical_state_sha256"]] * len(final_calls),
            )
            self.assertEqual(
                [call["cached"] for call in final_calls], [False, True, True, True, False]
            )

    def test_native_gpu_capture_contract_identifies_borrowed_history(self):
        raw = copy.deepcopy(self.raw)
        raw["render"]["capture_pipeline"] = "native-gpu"
        recipe = runner.validate_recipe(raw)
        metadata = runner.capture_metadata(recipe)["interaction_material"]
        self.assertEqual(metadata["version"], VERSION)
        self.assertIn("borrowed native textures", metadata["capture"])
        self.assertIn("no interaction update", metadata["camera"])

    def test_interaction_movies_require_native_material_for_cpu_and_gpu_capture(self):
        for pipeline in ("cpu", "native-gpu"):
            raw = copy.deepcopy(self.raw)
            raw["simulation"]["resolution"] = [256, 192]
            raw["render"].update(capture_pipeline=pipeline, capture_resolution=[128, 96])
            with (
                self.subTest(pipeline=pipeline),
                self.assertRaisesRegex(ValueError, "full native material grid"),
            ):
                runner.validate_recipe(raw)
        recipe = runner.validate_recipe(self.raw)
        recipe["render"]["capture_resolution"] = [64, 48]
        with self.assertRaisesRegex(ValueError, "full native material grid"):
            runner.capture_metadata(recipe)

    def test_native_capture_is_the_default_for_large_interaction_material(self):
        raw = copy.deepcopy(self.raw)
        raw["simulation"]["resolution"] = [4096, 3072]
        raw["render"].pop("capture_resolution")
        self.assertEqual(runner.validate_recipe(raw)["render"]["capture_resolution"], [4096, 3072])

    def test_active_appearance_needs_real_history_but_zero_strength_is_a_valid_control(self):
        raw = copy.deepcopy(self.raw)
        raw["simulation"].pop("interaction")
        raw["surface"] = {"interaction": {"silk_strength": 0.5, "grain_strength": 0}}
        with self.assertRaisesRegex(ValueError, "require transported interaction"):
            runner.validate_recipe(raw)
        raw["surface"]["interaction"]["silk_strength"] = 0
        recipe = runner.validate_recipe(raw)
        self.assertIsNone(runner.interaction_metadata(recipe, "0x1"))

    def test_null_and_omitted_simulation_extension_keep_identical_legacy_identity(self):
        raw = copy.deepcopy(self.raw)
        raw["simulation"].pop("interaction")
        omitted = runner.validate_recipe(raw)
        raw["simulation"]["interaction"] = None
        explicit_null = runner.validate_recipe(raw)
        self.assertEqual(omitted, explicit_null)
        self.assertEqual(runner.capture_metadata(omitted), runner.capture_metadata(explicit_null))

    def test_three_optical_variants_share_one_history_and_every_frame_identity(self):
        self.raw["looks"] = ["control", "silk", "silk-grain"]
        self.raw["surface"] = {
            "interaction": {"silk_strength": 0.7, "grain_strength": 0.3},
            "mix_control": 0.25,
        }
        write(self.recipe, self.raw)
        self.args.still_only = False
        runner.run(self.args)
        request, receipt = runner.verify_run(self.args.output)
        self.assertEqual(len(fixtures.FakeEngine.instances), 1)
        self.assertEqual(len(fixtures.FakeSurface.instances), 3)
        controls = [(0, 0), (0.7, 0), (0.7, 0.3)]
        surfaces = fixtures.FakeSurface.instances
        for surface, (silk, grain) in zip(surfaces, controls, strict=True):
            self.assertEqual(surface.config["mode"], "layered")
            self.assertEqual(surface.config["mix_control"], 1)
            self.assertEqual(
                surface.config["interaction"], {"silk_strength": silk, "grain_strength": grain}
            )
            self.assertEqual(surface.calls, surfaces[0].calls)
        self.assertEqual(
            [entry["physical_state_sha256"] for entry in receipt["looks"].values()],
            [receipt["physical_state_sha256"]] * 3,
        )
        ledger = read(self.args.output / "frame-ledger.json")
        self.assertEqual([entry["timing"] for entry in ledger], request["frames"])
        self.assertTrue(all(set(entry["images"]) == set(self.raw["looks"]) for entry in ledger))
        self.assertEqual(fixtures.FakeEngine.instances[0].visited, [0, 2, 4, 6, 8, 10])

    def test_named_variants_require_simulation_and_explicit_optics(self):
        for look in ("control", "silk", "silk-grain"):
            raw = copy.deepcopy(self.raw)
            raw["looks"] = [look]
            with (
                self.subTest(look=look, missing="surface"),
                self.assertRaisesRegex(ValueError, "explicit surface.interaction"),
            ):
                runner.validate_recipe(raw)
            raw["surface"] = {"interaction": {"silk_strength": 0, "grain_strength": 0}}
            raw["simulation"].pop("interaction")
            with (
                self.subTest(look=look, missing="simulation"),
                self.assertRaisesRegex(ValueError, "views require transported interaction"),
            ):
                runner.validate_recipe(raw)

    def test_all_five_views_are_distinct_and_legacy_controls_are_unchanged(self):
        raw = copy.deepcopy(self.raw)
        raw["surface"] = {
            "interaction": {"silk_strength": 0.7, "grain_strength": 0.3},
            "mix_control": 0.25,
        }
        legacy = runner.validate_recipe(raw)
        expected = runner.surface_configs(legacy)
        raw["looks"] = list(runner.LABELS)
        recipe = runner.validate_recipe(raw)
        before = copy.deepcopy(recipe)
        actual = runner.surface_configs(recipe)
        self.assertEqual(recipe, before)
        self.assertEqual(len(actual), 5)
        self.assertEqual(actual["layered"], expected["layered"])
        self.assertEqual(actual["homogeneous"], expected["homogeneous"])
        actual["control"]["interaction"]["silk_strength"] = 1
        self.assertEqual(recipe, before)
        for looks in (["control", "control"], ["silk-grain"] * 6, ["unknown"]):
            raw["looks"] = looks
            with self.subTest(looks=looks), self.assertRaisesRegex(ValueError, "distinct known"):
                runner.validate_recipe(raw)

    def test_packing_rejects_substrate_height_in_recipes_and_archived_metadata(self):
        raw = copy.deepcopy(self.raw)
        raw["surface"] = {
            "interaction": {"silk_strength": 0, "grain_strength": 0, "packing_strength": 0.5}
        }
        with self.assertRaisesRegex(ValueError, "zero simulation.substrate_um"):
            runner.validate_recipe(raw)
        raw["simulation"]["substrate_um"] = 0
        recipe = runner.validate_recipe(raw)
        self.assertIsNotNone(runner.interaction_metadata(recipe, "0x1"))
        recipe["simulation"]["substrate_um"] = 1.0
        with self.assertRaisesRegex(ValueError, "zero simulation.substrate_um"):
            runner.interaction_metadata(recipe, "0x1")

    def test_named_packing_comparisons_keep_the_length_but_disable_control_displacement(self):
        raw = copy.deepcopy(self.raw)
        raw["simulation"]["substrate_um"] = 0
        raw["looks"] = ["control", "silk", "silk-grain"]
        raw["surface"] = {
            "interaction": {
                "silk_strength": 0.7,
                "grain_strength": 0.4,
                "packing_strength": 0.6,
                "packing_length_um": 800,
            }
        }
        recipe = runner.validate_recipe(raw)
        before = copy.deepcopy(recipe)
        configs = runner.surface_configs(recipe)
        self.assertEqual(recipe, before)
        for name, strength in (("control", 0), ("silk", 0), ("silk-grain", 0.6)):
            controls = configs[name]["interaction"]
            self.assertEqual(controls["packing_strength"], strength)
            self.assertEqual(controls["packing_length_um"], 800)

    def test_zero_packing_and_length_only_controls_allow_unchanged_substrate(self):
        for controls in ({"packing_strength": 0}, {"packing_length_um": 900}):
            raw = copy.deepcopy(self.raw)
            raw["surface"] = {"interaction": controls}
            with self.subTest(controls=controls):
                recipe = runner.validate_recipe(raw)
                self.assertGreater(recipe["simulation"]["substrate_um"], 0)
                self.assertEqual(recipe["surface"]["interaction"]["packing_strength"], 0)

    def test_unsupported_packing_scale_fails_before_allocating_or_simulating(self):
        raw = copy.deepcopy(self.raw)
        raw["simulation"].update(resolution=[4096, 3072], substrate_um=0)
        raw["render"]["capture_resolution"] = [4096, 3072]
        raw["surface"] = {
            "canvas_width_m": 0.05,
            "interaction": {"packing_strength": 0.5, "packing_length_um": 1200},
        }
        with self.assertRaisesRegex(ValueError, "16-cell native coupling radius"):
            runner.validate_recipe(raw)
        self.assertEqual(fixtures.FakeEngine.instances, [])

    def test_named_views_archive_explicit_grain_contrast_without_changing_strength_controls(self):
        self.raw["looks"] = ["control", "silk", "silk-grain"]
        self.raw["surface"] = {
            "interaction": {"silk_strength": 0.7, "grain_strength": 0.4, "grain_contrast": 8}
        }
        write(self.recipe, self.raw)
        runner.run(self.args)
        request, _ = runner.verify_run(self.args.output)
        for name, silk, grain in (("control", 0, 0), ("silk", 0.7, 0), ("silk-grain", 0.7, 0.4)):
            self.assertEqual(
                request["surface_configs"][name]["interaction"],
                {"silk_strength": silk, "grain_strength": grain, "grain_contrast": 8},
            )
        self.assertTrue(
            all(
                surface.config["interaction"]["grain_contrast"] == 8
                for surface in fixtures.FakeSurface.instances
            )
        )
        configs = runner.surface_configs(request["recipe"])
        configs["control"]["interaction"]["grain_contrast"] = 2
        self.assertEqual(configs["silk"]["interaction"]["grain_contrast"], 8)
        self.assertEqual(request["recipe"]["surface"]["interaction"]["grain_contrast"], 8)

    def test_named_views_keep_optional_contrast_and_packing_absent_when_omitted(self):
        self.raw["looks"] = ["control", "silk", "silk-grain"]
        self.raw["surface"] = {"interaction": {"silk_strength": 0.7, "grain_strength": 0.4}}
        recipe = runner.validate_recipe(self.raw)
        configs = runner.surface_configs(recipe)
        for name, silk, grain in (("control", 0, 0), ("silk", 0.7, 0), ("silk-grain", 0.7, 0.4)):
            self.assertEqual(
                configs[name],
                {
                    **recipe["surface"],
                    "mode": "layered",
                    "mix_control": 1.0,
                    "interaction": {"silk_strength": silk, "grain_strength": grain},
                },
            )


if __name__ == "__main__":
    unittest.main()
