"""Semantic tests: temporary files and fake process handles only; never signal jobs."""

import contextlib
import copy
import io
import itertools
import json
import os
import signal
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import rebalance_collection as rc


class FakeProcesses:
    def __init__(self):
        self.uid = os.getuid()
        self.boot = "test-boot"
        self.table = {}
        self.logs = {}
        self.signals = []
        self.launches = []
        self.on_stop = None
        self.ignore_term = set()
        self.crash_after_spawn = False

    def read(self, pid):
        return self.table.get(pid)

    def all(self):
        return list(self.table.values())

    def pin(self, expected):
        current = self.read(expected.pid)
        if current is None:
            return None
        rc.same_process(current, expected.identity())
        return rc.Pin(current, current.pid)

    def signal(self, pin, signum):
        current = self.read(pin.process.pid)
        if current is None:
            return
        rc.same_process(current, pin.process.identity(), check_parent=False)
        self.signals.append((current.pid, current.start, signum))
        if signum == signal.SIGSTOP:
            self.table[current.pid] = replace(current, state="T")
            hook, self.on_stop = self.on_stop, None
            if hook:
                hook()
        elif signum == signal.SIGCONT:
            self.table[current.pid] = replace(current, state="S")
        elif signum == signal.SIGKILL or (
            signum == signal.SIGTERM and current.pid not in self.ignore_term
        ):
            self.table.pop(current.pid)

    def close(self, _pin):
        pass

    def exited(self, pin):
        current = self.read(pin.process.pid)
        return current is None or current.start != pin.process.start

    def wait(self, pins, _timeout, stopped=False):
        if stopped:
            value = self.read(pins[0].process.pid)
            return value is not None and value.state == "T"
        return all(self.exited(pin) for pin in pins)

    def environment(self, _process):
        return {"PATH": "/usr/bin:/bin", "PRIVATE_TOKEN": "must-not-enter-events"}

    def output_path(self, process, _descriptor):
        return self.logs[process.pid]

    def spawn(self, command, cwd, environment, log, python):
        pid = 900000 + len(self.launches)
        image = python.stat()
        process = rc.Proc(
            pid,
            999,
            pid * 10,
            self.uid,
            str(python),
            image.st_dev,
            image.st_ino,
            cwd,
            tuple(command),
            "S",
        )
        self.table[pid] = process
        self.logs[pid] = str(log)
        self.launches.append((list(command), cwd, dict(environment)))
        with Path(log).open("ab") as stream:
            stream.write(b"restarted driver output\n")
        if self.crash_after_spawn:
            self.crash_after_spawn = False
            raise RuntimeError("simulated controller death after child launch")
        return process


class Fixture:
    def __init__(self, temporary):
        base = Path(temporary).resolve() / "tidal-silk"
        self.paths = rc.Paths(base, base / "python3.12")
        self.proc = FakeProcesses()
        self.paths.scripts.mkdir(parents=True)
        (base / "bin").mkdir()
        self.paths.orbit.parent.mkdir()
        (self.paths.series / "configs").mkdir(parents=True)
        (self.paths.series / "logs").mkdir()
        self.paths.private.mkdir(mode=0o700)
        self.paths.python.write_text("fixed python")
        self.paths.orbit.write_text("fixed orbit")
        known = {
            str(self.paths.python): rc.digest(self.paths.python),
            str(self.paths.orbit): rc.digest(self.paths.orbit),
        }
        for name in rc.SCRIPT_HASHES:
            path = self.paths.scripts / name
            path.write_text("fixed " + name)
            known[str(path)] = rc.digest(path)
        self.specs = []
        for original in rc.SPECS:
            binary = self.paths.binary(original)
            binary.write_text("fixed " + original.slug)
            spec = replace(original, binary_hash=rc.digest(binary))
            self.specs.append(spec)
            known[str(binary)] = spec.binary_hash
            root = self.paths.output(spec)
            root.mkdir()
            recipe = {
                "kind": spec.slug.split("-", 1)[1],
                "frames": 1802,
                "render": {"aa": 3, "quality": "unchanged"},
            }
            for path in (self.paths.config(spec), root / "render-request.json"):
                path.write_text(json.dumps(recipe))
            command = [
                "python3",
                str(self.paths.scripts / "render_study.py"),
                "--orbit",
                str(self.paths.orbit),
                "--config",
                str(self.paths.config(spec)),
                "--output",
                str(root),
                "--executable",
                str(binary),
                "--workers",
                str(spec.initial_workers),
                "--chunks",
                "16",
                "--parallel-ranges",
                "4",
                "--poster-frame",
                "900",
            ]
            (root / "final-job.json").write_text(
                json.dumps({"pid": spec.initial_pid, "command": command, "preserve_me": "metadata"})
            )
            self.paths.log(spec).write_text("original driver output\n")
            image = self.paths.python.stat()
            self.proc.table[spec.initial_pid] = rc.Proc(
                spec.initial_pid,
                1,
                spec.initial_start,
                self.proc.uid,
                str(self.paths.python),
                image.st_dev,
                image.st_ino,
                str(base.parent),
                tuple(command),
                "S",
            )
            self.proc.logs[spec.initial_pid] = str(self.paths.log(spec))
            for index in range(16):
                chunk = root / f"final-chunk-{index}"
                chunk.mkdir()
                manifest = {
                    "config": recipe,
                    "seed": "0xb7f327f9f722",
                    "executable_sha256": spec.binary_hash,
                    "orbit_sha256": known[str(self.paths.orbit)],
                    "rendered_frames": list(range(1802 * index // 16, 1802 * (index + 1) // 16)),
                    "complete": False,
                }
                (chunk / "render.json").write_text(json.dumps(manifest))
            for suffix in (".png", ".png.json", ".png.partial"):
                (root / "final-chunk-0" / ("frame_000000" + suffix)).write_text("keep these bytes")
        self.collection = rc.Collection(self.paths, self.specs, self.proc, known, "test-boot")
        for spec in self.specs:
            job = self.collection.job(spec)
            for index in (0, 4, 8, 12):
                self.add_renderer(job, index, spec.initial_pid + 1000 + index)
        self.state = rc.bootstrap(self.collection, 720)
        self.controller = rc.Controller(self.collection, self.state, 0.01)
        self.controller.save()

    def add_renderer(self, job, index, pid):
        image = self.paths.binary(job.spec).stat()
        process = rc.Proc(
            pid,
            job.record["pid"],
            pid * 10,
            self.proc.uid,
            str(self.paths.binary(job.spec)),
            image.st_dev,
            image.st_ino,
            str(self.paths.base.parent),
            tuple(self.collection.renderer_command(job, index)),
            "S",
        )
        self.proc.table[pid] = process
        return process

    def job(self, slug="05-engraving"):
        return self.collection.job(self.collection.specs[slug])

    def complete(self, slug):
        spec = self.collection.specs[slug]
        for index in range(16):
            path = self.paths.output(spec) / f"final-chunk-{index}/render.json"
            value = json.loads(path.read_text())
            value["complete"] = True
            path.write_text(json.dumps(value))

    def reloaded(self):
        state = rc.read_json(self.controller.state_path)
        rc.validate_state(state, self.collection)
        return rc.Controller(self.collection, state, 0.01)


class RebalancerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.temporary.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_all_completion_orders_follow_finite_policy(self):
        initial = {spec.slug: spec.initial_workers for spec in rc.SPECS}
        for order in itertools.permutations(initial):
            complete, current = set(), dict(initial)
            for finished in order:
                before = dict(current)
                complete.add(finished)
                target = rc.allocations(complete, current)
                if "03-aurora" not in complete:
                    self.assertEqual(target, {key: before[key] for key in before.keys() - complete})
                else:
                    self.assertLessEqual(sum(target.values()), 128)
                    if len(target) == 1:
                        self.assertEqual(next(iter(target.values())), 128)
                current.update(target)
        self.assertEqual(
            rc.allocations({"03-aurora"}, initial),
            {"04-light": 32, "05-engraving": 32, "06-eclipse": 64},
        )
        self.assertEqual(
            rc.allocations({"03-aurora", "05-engraving"}, initial),
            {"04-light": 64, "06-eclipse": 64},
        )

    def test_dry_plan_and_unchanged_allocations_have_no_mutations(self):
        f = self.fixture
        for _ in range(3):
            plan = rc.dry_plan(f.controller)
            self.assertTrue(all(job["action"] == "keep" for job in plan["jobs"]))
        f.controller.begin(f.job(), 48)
        self.assertEqual(f.proc.signals, [])
        self.assertEqual(f.proc.launches, [])
        f.complete("03-aurora")
        plan = rc.dry_plan(f.controller)
        changed = [job for job in plan["jobs"] if job["action"] == "restart"]
        self.assertEqual(
            [(job["slug"], job["desired_workers"]) for job in changed], [("05-engraving", 32)]
        )

    def test_pid_reuse_and_foreign_child_block_before_signals(self):
        f = self.fixture
        job = f.job()
        original = f.proc.table[job.record["pid"]]
        f.proc.table[original.pid] = replace(original, start=original.start + 1)
        with self.assertRaises(rc.UnsafeJob):
            f.controller.begin(job, 32)
        self.assertEqual(f.proc.signals, [])
        f.proc.table[original.pid] = original
        child = next(value for value in f.proc.all() if value.ppid == original.pid)
        f.proc.table[child.pid] = replace(child, argv=("unrelated", "--output", "elsewhere"))
        with self.assertRaises(rc.UnsafeJob):
            f.controller.begin(job, 32)
        self.assertEqual(f.proc.signals, [])
        self.assertEqual(f.proc.launches, [])

    def test_paths_duplicate_options_and_changed_files_are_rejected(self):
        f = self.fixture
        job = f.job()
        for modified in [
            [*job.command, "--workers", "64"],
            [*job.command, "--overwrite", "true"],
            [value.replace("final", "other") for value in job.command],
        ]:
            if modified == job.command:
                continue
            with self.assertRaises(rc.UnsafeJob):
                f.collection.parse_driver(job.spec, modified)
        command = list(job.command)
        command[command.index("--executable") + 1] = str(f.paths.base / "bin/atelier-v999")
        with self.assertRaises(rc.UnsafeJob):
            f.collection.parse_driver(job.spec, command)
        f.paths.config(job.spec).write_text("changed")
        with self.assertRaises(rc.UnsafeJob):
            f.controller.begin(job, 32)
        self.assertEqual(f.proc.signals, [])

    def test_writer_matching_remains_lexical_without_following_symlinks(self):
        f = self.fixture
        job = f.job()
        outside = f.paths.base.parent / "redirected-output"
        outside.mkdir()
        alias = f.paths.output(job.spec) / "final-chunk-link"
        alias.symlink_to(outside, target_is_directory=True)
        self.assertFalse(alias.resolve().is_relative_to(f.paths.output(job.spec)))
        process = f.add_renderer(job, 0, 777777)
        command = list(process.argv)
        command[command.index("--output") + 1] = os.path.relpath(alias, process.cwd)
        process = replace(process, argv=tuple(command))
        self.assertEqual(f.collection.writers(job, [process]), [process])
        self.assertEqual(f.proc.signals, [])

    def test_descriptor_symlink_target_remains_a_string(self):
        f = self.fixture
        process = f.proc.table[f.job().record["pid"]]
        expected = f.paths.base / "render.log"
        with patch.object(Path, "readlink", return_value=expected):
            actual = rc.Processes.output_path(process, 1)
        self.assertIsInstance(actual, str)
        self.assertEqual(actual, str(expected))

    def test_renderer_started_during_discovery_is_captured_after_freeze(self):
        f = self.fixture
        job = f.job()
        old_child = next(value for value in f.proc.all() if value.ppid == job.record["pid"])
        new_pid = 777777

        def queued_replacement():
            f.proc.table.pop(old_child.pid)
            f.add_renderer(job, 1, new_pid)

        f.proc.on_stop = queued_replacement
        before = {
            path: path.read_bytes()
            for path in (f.paths.output(job.spec) / "final-chunk-0").iterdir()
        }
        f.controller.begin(job, 32)
        signals = f.proc.signals
        self.assertEqual(signals[0][2], signal.SIGSTOP)
        self.assertIn((new_pid, new_pid * 10, signal.SIGTERM), signals)
        self.assertFalse(any(pid == old_child.pid for pid, _, _ in signals))
        self.assertEqual(len(f.proc.launches), 1)
        new_command = f.proc.launches[0][0]
        self.assertEqual(
            [
                i
                for i, pair in enumerate(zip(job.command, new_command, strict=False))
                if pair[0] != pair[1]
            ],
            [job.command.index("--workers") + 1],
        )
        self.assertEqual(new_command[new_command.index("--workers") + 1], "32")
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))
        updated = rc.read_json(f.paths.output(job.spec) / "final-job.json")
        self.assertEqual(updated["preserve_me"], "metadata")
        self.assertEqual(updated["previous_pid"], job.record["pid"])
        self.assertTrue(f.paths.log(job.spec).read_text().startswith("original driver output\n"))
        self.assertNotIn("must-not-enter-events", (f.paths.private / "events.jsonl").read_text())

    def test_finisher_transition_is_resumed_and_not_terminated(self):
        f = self.fixture
        job = f.job()

        def finish_transition():
            for child in list(f.proc.all()):
                if child.ppid == job.record["pid"]:
                    f.proc.table.pop(child.pid)
            command = [
                str(f.paths.python.with_name("python3")),
                str(f.paths.scripts / "finish_study.py"),
                "--output",
                str(f.paths.output(job.spec)),
                "--executable",
                str(f.paths.binary(job.spec)),
                "--poster-frame",
                "900",
                "--encoder-threads",
                "16",
            ]
            command += [
                part
                for i in range(16)
                for part in ("--input", str(f.paths.output(job.spec) / f"final-chunk-{i}"))
            ]
            image = f.paths.python.stat()
            f.proc.table[888888] = rc.Proc(
                888888,
                job.record["pid"],
                888,
                f.proc.uid,
                str(f.paths.python),
                image.st_dev,
                image.st_ino,
                str(f.paths.base.parent),
                tuple(command),
                "S",
            )

        f.proc.on_stop = finish_transition
        with self.assertRaises(rc.Deferred):
            f.controller.begin(job, 32)
        self.assertEqual(
            [signum for _, _, signum in f.proc.signals], [signal.SIGSTOP, signal.SIGCONT]
        )
        self.assertIn(888888, f.proc.table)
        self.assertEqual(f.proc.launches, [])
        self.assertIsNone(f.controller.state["transaction"])

    def test_escalation_targets_only_pinned_owned_children(self):
        f = self.fixture
        job = f.job()
        child = next(value for value in f.proc.all() if value.ppid == job.record["pid"])
        f.proc.ignore_term.add(child.pid)
        others = {
            value.pid
            for value in f.proc.all()
            if value.ppid != job.record["pid"] and value.pid != job.record["pid"]
        }
        f.controller.begin(job, 32)
        self.assertIn((child.pid, child.start, signal.SIGKILL), f.proc.signals)
        self.assertTrue(others.issubset(f.proc.table))

    def test_launch_before_record_crash_recovers_without_duplicate_driver(self):
        f = self.fixture
        f.proc.crash_after_spawn = True
        with self.assertRaises(RuntimeError):
            f.controller.begin(f.job(), 32)
        self.assertEqual(len(f.proc.launches), 1)
        self.assertEqual(rc.read_json(f.controller.state_path)["transaction"]["stage"], "launching")
        recovered = f.reloaded()
        recovered.recover()
        self.assertEqual(len(f.proc.launches), 1)
        self.assertIsNone(recovered.state["transaction"])
        self.assertEqual(f.job().workers, 32)
        self.assertEqual(f.job().record["pid"], 900000)

    def test_record_before_ledger_crash_recovers_same_launched_process(self):
        f = self.fixture
        real_save = f.controller.save

        def fail_after_record():
            if f.controller.state["transaction"] is None and f.job().workers == 32:
                raise RuntimeError("simulated death after job publication")
            real_save()

        f.controller.save = fail_after_record
        with self.assertRaises(RuntimeError):
            f.controller.begin(f.job(), 32)
        self.assertEqual(f.job().workers, 32)
        recovered = f.reloaded()
        recovered.recover()
        self.assertEqual(len(f.proc.launches), 1)
        self.assertIsNone(recovered.state["transaction"])

    def test_pretermination_crash_resumes_owned_driver_without_restart(self):
        f = self.fixture
        job = f.job()
        original = f.controller.finish_transaction
        f.controller.finish_transaction = lambda: None
        f.controller.begin(job, 32)
        f.controller.finish_transaction = original
        process = f.proc.table[job.record["pid"]]
        f.proc.table[process.pid] = replace(process, state="T")
        recovered = f.reloaded()
        recovered.recover()
        self.assertEqual(f.proc.signals, [(process.pid, process.start, signal.SIGCONT)])
        self.assertEqual(f.proc.launches, [])
        self.assertIsNone(recovered.state["transaction"])

    def test_committed_termination_relaunches_even_if_last_manifest_completed(self):
        f = self.fixture
        job = f.job()
        f.controller.finish_transaction = lambda: None
        f.controller.begin(job, 32)
        transaction = f.controller.state["transaction"]
        transaction["stage"] = "terminating"
        for child in list(f.proc.all()):
            if child.ppid == job.record["pid"]:
                f.proc.table.pop(child.pid)
        driver = f.proc.table[job.record["pid"]]
        f.proc.table[driver.pid] = replace(driver, state="T")
        f.complete(job.spec.slug)
        f.controller.save()
        recovered = f.reloaded()
        recovered.recover()
        self.assertEqual(len(f.proc.launches), 1)
        self.assertFalse(any(signum == signal.SIGCONT for _, _, signum in f.proc.signals))
        self.assertEqual(f.job().workers, 32)
        self.assertIsNone(recovered.state["transaction"])

    def test_complete_requires_pinned_full_manifest_ranges(self):
        f = self.fixture
        job = f.job()
        self.assertFalse(f.collection.complete(job))
        f.complete(job.spec.slug)
        self.assertTrue(f.collection.complete(job))
        path = f.paths.output(job.spec) / "final-chunk-4/render.json"
        value = json.loads(path.read_text())
        value["rendered_frames"].pop()
        path.write_text(json.dumps(value))
        with self.assertRaises(rc.UnsafeJob):
            f.collection.complete(job)

    def test_deadline_survives_reload_and_lock_is_exclusive(self):
        f = self.fixture
        deadline = f.state["deadline_unix"]
        self.assertEqual(f.reloaded().state["deadline_unix"], deadline)
        with (
            rc.controller_lock(f.paths.private),
            self.assertRaises(rc.UnsafeJob),
            rc.controller_lock(f.paths.private),
        ):
            self.fail("duplicate controller acquired the lock")
        expired = copy.deepcopy(f.state)
        expired["deadline_unix"] = time.time() - 1
        expired["started_unix"] = expired["deadline_unix"] - 1
        with self.assertRaises(TimeoutError):
            rc.Controller(f.collection, expired).remaining()
        for value in ("nan", "inf", "721", "0"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                rc.main(["--timeout-hours", value])


if __name__ == "__main__":
    unittest.main()
