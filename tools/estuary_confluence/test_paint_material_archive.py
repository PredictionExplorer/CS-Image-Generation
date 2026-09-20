"""Native structural material, provenance and borrowed-view extension contracts."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from tools.estuary_studio.common import write

from . import run as runner
from . import test_gpu_frame as gpu_fixtures
from . import test_interaction_archive as archive_fixtures
from .gpu_frame import GPUFrame


class MaterialEngineFixture(archive_fixtures.InteractionEngineFixture):
    def snapshot(self, resolution=None):
        fields = super().snapshot(resolution)
        shape = fields["height"].shape
        if self.config.get("interaction", {}).get("material_variation"):
            for layer in ("upper", "lower"):
                fields[f"trait_{layer}"] = np.full((*shape, 2), 0.2, dtype="f4")
        if self.config.get("rheology"):
            for layer in ("upper", "lower"):
                fields[f"structure_{layer}"] = np.full(shape, 0.45, dtype="f4")
        return fields


class MaterialArchiveTests(unittest.TestCase):
    setUp = archive_fixtures.InteractionArchiveTests.setUp
    source_info = staticmethod(archive_fixtures.InteractionArchiveTests.source_info)
    engaged_layout = staticmethod(archive_fixtures.InteractionArchiveTests.engaged_layout)
    movie = staticmethod(archive_fixtures.InteractionArchiveTests.movie)
    rewrite_artifact_hash = archive_fixtures.InteractionArchiveTests.rewrite_artifact_hash
    rewrite_request = archive_fixtures.InteractionArchiveTests.rewrite_request
    fields = archive_fixtures.InteractionArchiveTests.fields
    replace_fields = archive_fixtures.InteractionArchiveTests.replace_fields

    def enabled(self, traits=True, rheology=True):
        self.raw["simulation"]["interaction"] = {"material_variation": {}} if traits else {}
        if rheology:
            self.raw["simulation"]["rheology"] = {}
        write(self.recipe, self.raw)
        self.patches.enter_context(
            patch("tools.estuary_confluence.engine.Engine", MaterialEngineFixture)
        )
        runner.run(self.args)
        return runner.verify_run(self.args.output)

    def test_extension_fields_are_native_and_each_is_bound_to_material_identity(self):
        request, receipt = self.enabled()
        self.assertEqual(len(self.fields()), 18)
        self.assertEqual(request["rheology"], receipt["rheology"])
        self.assertIn("trait_upper", request["interaction"]["fields"])
        self.assertEqual(
            request["capture"]["structural_material"]["fields"],
            ["structure_lower", "structure_upper"],
        )
        original = self.fields()
        for field in ("trait_upper", "trait_lower", "structure_upper", "structure_lower"):
            changed = copy.deepcopy(original)
            changed[field].flat[0] += np.float32(0.01)
            self.replace_fields(changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Physical state"):
                runner.verify_run(self.args.output)
        self.replace_fields(original)
        runner.verify_run(self.args.output)

    def test_missing_extra_or_invalid_material_fields_fail_even_when_hashes_are_rebound(self):
        self.enabled()
        original = self.fields()
        cases = []
        for name in ("trait_upper", "structure_lower"):
            absent = copy.deepcopy(original)
            del absent[name]
            cases.append(absent)
            bad = copy.deepcopy(original)
            bad[name].flat[0] = 1.1
            cases.append(bad)
            wrong_type = copy.deepcopy(original)
            wrong_type[name] = wrong_type[name].astype("f8")
            cases.append(wrong_type)
        extra = copy.deepcopy(original)
        extra["mystery"] = np.zeros_like(extra["height"])
        cases.append(extra)
        for i, fields in enumerate(cases):
            self.replace_fields(fields, rebind_state=True)
            with self.subTest(case=i), self.assertRaises(ValueError):
                runner.verify_run(self.args.output)

    def test_rheology_solver_claims_are_checked_even_with_rebound_request(self):
        request, receipt = self.enabled()
        request["rheology"]["response"]["iterations"] = 1
        receipt["rheology"] = copy.deepcopy(request["rheology"])
        write(self.args.output / "receipt.json", receipt)
        self.rewrite_request(request)
        with self.assertRaisesRegex(ValueError, "Rheology settings"):
            runner.verify_run(self.args.output)

    def test_rheology_only_provenance_requires_canonical_source_free_simulation(self):
        self.raw["simulation"].pop("interaction")
        self.raw["simulation"]["rheology"] = {}
        normalized = runner.validate_recipe(self.raw)
        self.assertIsNotNone(runner.rheology_metadata(normalized))
        for changes in (
            {"material_model": "legacy"},
            {"rheology": {"strength": 0}},
            {"deposition": 1},
        ):
            changed = copy.deepcopy(normalized)
            changed["simulation"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.rheology_metadata(changed)

    def test_zero_features_resolve_to_exact_legacy_recipe_and_capture_contract(self):
        baseline = runner.validate_recipe(self.raw)
        self.raw["simulation"]["interaction"] = {"material_variation": {"amplitude": 0}}
        self.raw["simulation"]["rheology"] = {"strength": 0}
        disabled = runner.validate_recipe(self.raw)
        self.assertEqual(baseline, disabled)
        self.assertEqual(runner.capture_metadata(baseline), runner.capture_metadata(disabled))
        self.assertIsNone(runner.rheology_metadata(disabled))

    def test_optional_extensions_publish_only_bound_portable_claims(self):
        from .gallery import _interaction_metadata, _rheology_metadata

        request, receipt = self.enabled()
        self.assertEqual(
            _rheology_metadata(request, receipt), {"rheology_version": "paint-rheology-v1"}
        )
        self.assertEqual(
            _interaction_metadata(request, receipt)["base_material_sha256"],
            receipt["base_material_sha256"],
        )
        receipt["rheology"]["settings"]["strength"] = 20
        with self.assertRaisesRegex(ValueError, "Published rheology"):
            _rheology_metadata(request, receipt)


class MaterialViewTests(unittest.TestCase):
    def test_optional_native_extensions_are_atomic_and_share_owner_context(self):
        owner, frame = gpu_fixtures.ViewContracts.fixture()
        texture = frame.mobile[0]
        histories = dict(
            material_model="laminate",
            origin_upper=texture,
            origin_lower=texture,
            interaction_upper=texture,
            interaction_lower=texture,
        )
        enabled = replace(
            frame,
            **histories,
            trait_upper=texture,
            trait_lower=texture,
            structure_upper=texture,
            structure_lower=texture,
        )
        self.assertIs(enabled.validate(), enabled)
        for name in ("trait_upper", "trait_lower", "structure_upper", "structure_lower"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(enabled, **{name: None}).validate()
        for pair in (("trait_upper", "trait_lower"), ("structure_upper", "structure_lower")):
            with self.assertRaises(ValueError):
                replace(frame, **dict.fromkeys(pair, texture)).validate()
        with self.assertRaisesRegex(ValueError, "history"):
            replace(
                frame, material_model="laminate", trait_upper=texture, trait_lower=texture
            ).validate()
        self.assertIsNotNone(owner)
        self.assertIsInstance(frame, GPUFrame)
