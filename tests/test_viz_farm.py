"""Unit tests for the stdlib random visualization farm."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_viz_batch
import viz_farm


def catalog_flags() -> tuple[str, ...]:
    """A synthetic 69-mode catalog containing every prerequisite flag."""
    required = [
        "editorial-retime",
        "epilogue",
        "corotating",
        "bullet-time",
        "mission-control",
        "broadcast",
        "sonification",
        "trailer",
        "depth-pack",
        "tilt",
        "basin-map",
        "powers-of-fate",
        "celestial-atlas",
        "turntable",
    ]
    filler = [f"mode-{index:02d}" for index in range(1, 70 - len(required) + 1)]
    return tuple(required + filler)[:69]


def catalog_text(flags: tuple[str, ...]) -> str:
    """Render flags in the stable Rust ``--viz-list`` table shape."""
    rows = ["ID   FLAG                      CATEGORY  COST  STATUS       TITLE"]
    for index, flag in enumerate(flags, 1):
        rows.append(f"V{index:02d}  {flag:<25} combos    D     implemented  Mode {index}")
    rows.append(f"\n{len(flags)} modes total, {len(flags)} implemented.")
    return "\n".join(rows)


class CatalogTests(unittest.TestCase):
    def test_parser_requires_exactly_69_unique_implemented_modes(self) -> None:
        flags = catalog_flags()
        self.assertEqual(len(flags), 69)
        self.assertEqual(viz_farm.parse_viz_catalog(catalog_text(flags)), flags)
        with self.assertRaisesRegex(ValueError, "expected 69"):
            viz_farm.parse_viz_catalog(catalog_text(flags[:-1]))

    def test_every_target_is_eligible_including_turntable(self) -> None:
        flags = catalog_flags()
        for expected in flags:
            with mock.patch("viz_farm.secrets.choice", return_value=expected) as chooser:
                self.assertEqual(viz_farm.choose_target(flags), expected)
                chooser.assert_called_once_with(flags)
        self.assertIn("turntable", flags)

    def test_prerequisites_expand_in_catalog_order(self) -> None:
        flags = catalog_flags()
        expanded = viz_farm.expand_viz_flags("broadcast", flags)
        expected_set = {
            "editorial-retime",
            "epilogue",
            "corotating",
            "bullet-time",
            "mission-control",
            "broadcast",
        }
        self.assertEqual(set(expanded), expected_set)
        self.assertEqual(expanded, tuple(flag for flag in flags if flag in expected_set))
        self.assertEqual(
            viz_farm.expand_viz_flags("powers-of-fate", flags),
            ("basin-map", "powers-of-fate"),
        )
        self.assertEqual(
            viz_farm.expand_viz_flags("trailer", flags),
            ("sonification", "trailer"),
        )
        self.assertEqual(viz_farm.expand_viz_flags("tilt", flags), ("depth-pack", "tilt"))

    def test_job_command_is_final_quality_and_atlas_reads_output(self) -> None:
        binary = Path("/tmp/three_body_problem")
        base = viz_farm.build_job_command(
            binary,
            "0x1234",
            "random-job",
            "turntable",
            ("turntable",),
        )
        self.assertEqual(base[:5], (str(binary), "--seed", "0x1234", "--output", "random-job"))
        self.assertIn("--viz-quality", base)
        self.assertIn("final", base)
        self.assertNotIn("--fast-encode", base)
        atlas = viz_farm.build_job_command(
            binary,
            "0x5678",
            "atlas-job",
            "celestial-atlas",
            ("celestial-atlas",),
        )
        self.assertEqual(atlas[-2:], ("--viz-seeds-dir", "output"))

    def test_job_identity_is_readable_and_collision_resistant(self) -> None:
        instant = dt.datetime(2026, 8, 19, 3, 4, 5, tzinfo=dt.timezone.utc)
        identity = viz_farm.job_identity("rose-window", "0xAABB", 42, instant)
        self.assertEqual(identity, "random-20260819T030405Z-000042-rose-window-AABB")
        self.assertNotIn("/", identity)


class PersistenceTests(unittest.TestCase):
    def test_atomic_json_has_no_leftover_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            viz_farm.atomic_write_json(path, {"state": "running", "count": 4})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["count"], 4)
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            viz_farm.atomic_write_json(path, {"state": "stopped"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"state": "stopped"})

    def test_target_validation_requires_manifest_and_nonempty_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            package = output / "job"
            target = package / "viz" / "pond"
            target.mkdir(parents=True)
            self.assertEqual(
                viz_farm.validate_target_artifacts(output, "job", "pond"),
                "viz/manifest.json missing",
            )
            (package / "viz" / "manifest.json").write_text("{}\n", encoding="utf-8")
            self.assertIn(
                "contains no non-empty artifacts",
                viz_farm.validate_target_artifacts(output, "job", "pond") or "",
            )
            (target / "pond.mp4").write_bytes(b"video")
            self.assertIsNone(viz_farm.validate_target_artifacts(output, "job", "pond"))


class GuardTests(unittest.TestCase):
    def make_farm(self, root: Path) -> viz_farm.VizFarm:
        state_dir = root / "orchestrator"
        state_dir.mkdir()
        (state_dir / "jobs").mkdir()
        output_dir = root / "output"
        output_dir.mkdir()
        config = viz_farm.FarmConfig(
            work_dir=root,
            binary=root / "binary",
            state_dir=state_dir,
            output_dir=output_dir,
            concurrency=4,
            threads_per_job=30,
            min_free_gb=500.0,
            max_failure_streak=5,
            timeout_seconds=60.0,
            poll_seconds=0.01,
            state_seconds=0.01,
            max_jobs=None,
        )
        logger = logging.getLogger(f"test-{id(root)}")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        return viz_farm.VizFarm(config, catalog_flags(), logger)

    def test_disk_guard_stops_before_launching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            farm = self.make_farm(Path(directory))
            with (
                mock.patch("viz_farm.disk_free_gb", return_value=499.9),
                mock.patch.object(farm, "_start_job") as start,
            ):
                farm._fill_workers()
            self.assertEqual(farm.stop_reason, "stopped_low_disk")
            start.assert_not_called()

    def test_failure_circuit_breaker_drains_after_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            farm = self.make_farm(Path(directory))
            for sequence in range(1, 6):
                job_id = f"job-{sequence}"
                log_path = farm.config.state_dir / "jobs" / f"{job_id}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                record = viz_farm.JobRecord(
                    job_id=job_id,
                    session_id=farm.session_id,
                    git_head="abc",
                    sequence=sequence,
                    seed="0x1",
                    target_mode="pond",
                    viz_flags=("pond",),
                    output_name=job_id,
                    rayon_threads=30,
                    command=("binary",),
                    status="running",
                    started_at=viz_farm.utc_now(),
                )
                process = mock.Mock()
                process.pid = sequence
                farm.running[sequence] = viz_farm.RunningJob(
                    record=record,
                    process=process,
                    log_handle=log_handle,
                    log_path=log_path,
                    metadata_path=farm._job_metadata_path(job_id),
                    started_monotonic=0.0,
                )
                with mock.patch("viz_farm.time.monotonic", return_value=1.0):
                    farm._finish_job(sequence, 1, "synthetic failure")
            self.assertEqual(farm.failure_streak, 5)
            self.assertEqual(farm.stop_reason, "stopped_failures")


class FarmIntegrationTests(unittest.TestCase):
    def test_finite_fake_farm_refills_workers_and_finishes_cleanly(self) -> None:
        flags = catalog_flags()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fake_three_body_problem"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                f"FLAGS = {flags!r}\n"
                "if '--viz-list' in sys.argv:\n"
                "    for index, flag in enumerate(FLAGS, 1):\n"
                "        print(f'V{index:02d}  {flag:<25} combos    D     implemented  Mode')\n"
                "    raise SystemExit(0)\n"
                "output = sys.argv[sys.argv.index('--output') + 1]\n"
                "target = sys.argv[sys.argv.index('--viz') + 1].split(',')[-1]\n"
                "root = pathlib.Path('output') / output / 'viz'\n"
                "(root / target).mkdir(parents=True)\n"
                "(root / 'manifest.json').write_text('{}\\n')\n"
                "(root / target / 'artifact.txt').write_text('ok\\n')\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            previous = Path.cwd()
            try:
                os.chdir(root)
                exit_code = viz_farm.main(
                    [
                        "--binary",
                        str(binary),
                        "--concurrency",
                        "2",
                        "--threads-per-job",
                        "1",
                        "--min-free-gb",
                        "0.001",
                        "--max-jobs",
                        "3",
                        "--poll-seconds",
                        "0.01",
                        "--state-seconds",
                        "0.01",
                    ]
                )
            finally:
                os.chdir(previous)
            self.assertEqual(exit_code, 0)
            state = json.loads((root / "orchestrator" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["state"], "completed_limit")
            self.assertEqual(state["jobs_started"], 3)
            self.assertEqual(state["jobs_ok"], 3)
            self.assertEqual(state["jobs_failed"], 0)
            self.assertEqual(len(list((root / "orchestrator" / "jobs").glob("*.json"))), 3)
            self.assertEqual(len(list((root / "output").iterdir())), 3)


class LauncherTests(unittest.TestCase):
    def test_launcher_builds_locked_release_and_starts_guarded_farm(self) -> None:
        args = argparse.Namespace(
            concurrency=4,
            threads_per_job=30,
            min_free_gb=500.0,
            max_failure_streak=5,
            timeout_hours=336.0,
            max_jobs=None,
        )
        script = run_viz_batch.build_launch_script(args)
        self.assertIn("cargo build --release --locked", script)
        self.assertIn('if [ "$MODE_COUNT" -ne 69 ]', script)
        self.assertIn("--concurrency 4", script)
        self.assertIn("--threads-per-job 30", script)
        self.assertIn("--min-free-gb 500.000", script)
        self.assertIn("setsid nohup python3 viz_farm.py", script)
        self.assertNotIn("--fast-encode", script)


if __name__ == "__main__":
    unittest.main()
