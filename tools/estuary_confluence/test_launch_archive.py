"""Source-clock launch observations never change paint advancement or capture."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from tools.estuary.source import Source
from tools.estuary.test_source import orbit_points, write_orbit
from tools.estuary_studio.common import read, write
from tools.estuary_studio.run import record_array

from . import run as runner
from . import test_run as fixtures
from .engine import Engine
from .mass_budget import pigment_mass
from .test_body_influence import settings as material_settings

_SOURCE_READ = Source.read


class LaunchArchiveTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    source_info = staticmethod(fixtures.PipelineTests.source_info)
    engaged_layout = staticmethod(fixtures.PipelineTests.engaged_layout)
    movie = staticmethod(fixtures.PipelineTests.movie)
    rewrite_artifact_hash = fixtures.PipelineTests.rewrite_artifact_hash
    rewrite_request = fixtures.PipelineTests.rewrite_request

    def recipe_with_assessment(self):
        raw = fixtures.small_recipe()
        raw["simulation"]["steps"] = 20
        raw["assessment"] = {"interval_steps": 5, "resolution": [128, 96]}
        return raw

    def test_cpu_area_reduction_preserves_each_pigment_amount(self):
        field = np.zeros((384, 512, 4), dtype="f4")
        field[13:153, 9:219, 0] = 0.2
        field[::3, ::5, 1] = 0.7
        field[200:377, 225:493, 2] = 0.03
        plan = {"sample_resolution": [256, 192], "reduction_factor": 2}
        sample = runner._launch_sample(field, plan)
        self.assertEqual(sample.shape, (192, 256, 4))
        self.assertEqual(sample.dtype, np.float32)
        np.testing.assert_allclose(pigment_mass(sample, 1.6), pigment_mass(field, 1.6), rtol=1e-7)
        self.assertFalse(sample[..., 3].any())

    def test_observation_reuses_identical_advance_calls_and_readbacks(self):
        raw = self.recipe_with_assessment()
        write(self.recipe, raw)
        runner.run(self.args)
        _, before = runner.verify_run(self.args.output)
        baseline = fixtures.FakeEngine.instances[-1]
        self.assertNotIn("launch-assessment.json", before["artifacts"])
        self.args.output = self.folder / "observed"
        raw["launch_assessment"] = {}
        write(self.recipe, raw)
        runner.run(self.args)
        request, after = runner.verify_run(self.args.output)
        observed = fixtures.FakeEngine.instances[-1]
        self.assertEqual(observed.visited, baseline.visited)
        self.assertEqual(observed.snapshots, baseline.snapshots)
        self.assertEqual(after["physical_state_sha256"], before["physical_state_sha256"])
        self.assertEqual(request["launch_assessment"]["steps"], [0, 5, 10, 15, 20])
        report = read(self.args.output / "launch-assessment.json")
        self.assertEqual(report["source"], request["source"])
        self.assertEqual(report["capture_plan"], request["launch_assessment"])

    def test_missing_checkpoints_fail_and_null_option_keeps_exact_recipe(self):
        raw = self.recipe_with_assessment()
        original = runner.validate_recipe(raw)
        self.assertEqual(runner.validate_recipe({**raw, "launch_assessment": None}), original)
        raw["launch_assessment"] = {}
        raw["assessment"]["interval_steps"] = 4
        with self.assertRaisesRegex(ValueError, "coincide"):
            runner.validate_recipe(raw)
        raw["assessment"] = None
        with self.assertRaisesRegex(ValueError, "existing participation"):
            runner.validate_recipe(raw)

    def test_rehashed_report_or_samples_cannot_break_their_association(self):
        raw = self.recipe_with_assessment()
        raw["launch_assessment"] = {}
        write(self.recipe, raw)
        runner.run(self.args)
        with np.load(self.args.output / "launch-samples.npz", allow_pickle=False) as archive:
            samples = {name: archive[name] for name in archive.files}
        samples["step_000010"][0, 0, 0] += np.float32(0.1)
        record_array(self.args.output / "launch-samples.npz", samples)
        self.rewrite_artifact_hash("launch-samples.npz")
        with self.assertRaisesRegex(ValueError, "Launch report differs"):
            runner.verify_run(self.args.output)

    def test_initial_and_final_diagnostics_are_independent_of_film_cadence(self):
        raw = self.recipe_with_assessment()
        raw["launch_assessment"] = {}
        write(self.recipe, raw)
        runner.run(self.args)
        expected = read(self.args.output / "launch-assessment.json")
        self.args.output, self.args.still_only = self.folder / "film", False
        raw["render"].update(formation_frames=11, hold_frames=2, orbit_frames=4)
        write(self.recipe, raw)
        runner.run(self.args)
        runner.verify_run(self.args.output)
        self.assertEqual(read(self.args.output / "launch-assessment.json"), expected)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL4.3")
class LaunchGPUObserverTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    source_info = staticmethod(fixtures.PipelineTests.source_info)
    engaged_layout = staticmethod(fixtures.PipelineTests.engaged_layout)
    movie = staticmethod(fixtures.PipelineTests.movie)
    recipe_with_assessment = LaunchArchiveTests.recipe_with_assessment

    def test_opt_in_hook_preserves_real_gpu_material_and_work_across_cadences(self):
        write_orbit(self.source, orbit_points(), seed="0xbc53af1cd380")
        self.source_reader.side_effect = _SOURCE_READ
        self.patches.enter_context(patch("tools.estuary_confluence.engine.Engine", Engine))
        raw = self.recipe_with_assessment()
        raw["palette_mode"] = "composed"
        raw["simulation"] = material_settings(resolution=[128, 96], steps=20)
        write(self.recipe, raw)
        runner.run(self.args)
        _, baseline = runner.verify_run(self.args.output)
        self.args.output, self.args.still_only = self.folder / "observed-film", False
        raw["launch_assessment"] = {}
        raw["render"].update(formation_frames=11, hold_frames=2, orbit_frames=4)
        write(self.recipe, raw)
        runner.run(self.args)
        _, observed = runner.verify_run(self.args.output)
        self.assertEqual(observed["physical_state_sha256"], baseline["physical_state_sha256"])
        self.assertEqual(observed["solver_diagnostics"], baseline["solver_diagnostics"])


if __name__ == "__main__":
    unittest.main()
