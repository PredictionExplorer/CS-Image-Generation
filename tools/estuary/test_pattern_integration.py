"""Opt-in composition identity, initialization and output-cadence contracts."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .engine import Engine
from .recipe import canonical_bytes, validate_recipe
from .run import RUNTIME_FILES, code_identity
from .test_engine import SourceFixture

DESIGN = {
    "version": "starting-patterns-v1",
    "pattern": "folded-sash",
    "seed": "0x" + "17" * 32,
}


class SampledSource(SourceFixture):
    def __init__(self):
        self.sampled = []

    def sample(self, fractions):
        self.sampled.append(np.array(fractions))
        return SimpleNamespace(
            positions=np.stack([self.frame(float(f)).positions for f in fractions])
        )


def recipe(*, design=None):
    simulation = {
        "resolution": [128, 96],
        "steps": 360,
        "domain_scale": 1.6,
        "initial_pattern": "strata",
        "initial_load": 0.6,
        "deposition": 0,
        "carrier_velocity": [0.25, -0.1],
    }
    if design is not None:
        simulation.update(initial_pattern="composition", initial_design=copy.deepcopy(design))
    return validate_recipe(
        {"simulation": simulation, "render": {"resolution": [128, 96], "frames": 31}}
    )


class PatternIntegrationTests(unittest.TestCase):
    def test_omitted_and_null_design_preserve_legacy_controls(self):
        plain = recipe()
        disabled = copy.deepcopy(plain)
        disabled["simulation"]["initial_design"] = None
        self.assertEqual(canonical_bytes(plain), canonical_bytes(validate_recipe(disabled)))
        self.assertNotIn("initial_design", plain["simulation"])

    def test_design_requires_its_explicit_pattern_mode_and_strict_controls(self):
        enabled = recipe(design=DESIGN)
        self.assertEqual(enabled["simulation"]["initial_design"], DESIGN)
        for edit in ("no-design", "wrong-mode", "strata-profile", "short-seed", "unknown-field"):
            broken = copy.deepcopy(enabled)
            simulation = broken["simulation"]
            if edit == "no-design":
                del simulation["initial_design"]
            elif edit == "wrong-mode":
                simulation["initial_pattern"] = "pools"
            elif edit == "strata-profile":
                simulation["strata_profile"] = {"fine_width_scale": 0.5}
            elif edit == "short-seed":
                simulation["initial_design"]["seed"] = "0x17"
            else:
                simulation["initial_design"]["jitter"] = 1
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_recipe(broken)

    def test_initializer_samples_full_interval_once_and_scales_partition_once(self):
        engine = Engine.__new__(Engine)
        engine.recipe = recipe(design=DESIGN)
        engine.width, engine.height = 128, 96
        engine.aspect, engine.domain = 4 / 3, 1.6
        engine.source = SampledSource()
        uploads = []
        engine.paint = [SimpleNamespace(write=uploads.append) for _ in range(4)]
        engine._initialize()
        self.assertEqual(len(engine.source.sampled), 1)
        np.testing.assert_array_equal(engine.source.sampled[0], np.linspace(0, 1, 65))
        self.assertEqual(len(uploads), 4)
        self.assertTrue(all(data == uploads[0] for data in uploads))
        state = np.frombuffer(uploads[0], dtype=np.float32).reshape(96, 128, 4)
        np.testing.assert_allclose(state[..., :3].sum(axis=-1), 0.6, atol=1e-7, rtol=0)
        self.assertTrue(np.all(state[..., 3] == 0))
        self.assertTrue(np.all(state >= 0))
        self.assertTrue(np.all(state[..., :3].mean(axis=(0, 1)) > 0.005))

    def test_optional_code_changes_only_the_active_dependency_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in RUNTIME_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            with patch("tools.estuary.run.HERE", root):
                legacy = code_identity(recipe())
                with self.assertRaisesRegex(ValueError, "initial_patterns.py"):
                    code_identity(recipe(design=DESIGN))
                (root / "initial_patterns.py").write_text("first version")
                active = code_identity(recipe(design=DESIGN))
                self.assertEqual(set(active) - set(legacy), {"initial_patterns.py"})
                (root / "initial_patterns.py").write_text("changed geometry")
                self.assertEqual(code_identity(recipe()), legacy)
                self.assertNotEqual(code_identity(recipe(design=DESIGN)), active)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL4.3")
class PatternCadenceGPUTests(unittest.TestCase):
    def test_filming_and_snapshot_restore_preserve_the_same_final_material(self):
        config = recipe(design=DESIGN)
        with_first = Engine(SampledSource(), config)
        direct = Engine(SampledSource(), config)
        resumed = Engine(SampledSource(), config)
        for engine in (with_first, direct, resumed):
            self.addCleanup(engine.close)
        with_first.render(128, 96)
        for step in range(12, 181, 12):
            with_first.advance_to(step)
            with_first.render(128, 96)
        resumed.restore(
            with_first.read_state(),
            with_first.step,
            internal_steps=with_first.internal_steps,
            maximum_courant=with_first.maximum_courant,
        )
        with_first.advance_to(360)
        resumed.advance_to(360)
        direct.advance_to(360)
        np.testing.assert_array_equal(with_first.read_state(), direct.read_state())
        np.testing.assert_array_equal(resumed.read_state(), direct.read_state())
        self.assertLessEqual(with_first.maximum_courant, 1.500001)


if __name__ == "__main__":
    unittest.main()
