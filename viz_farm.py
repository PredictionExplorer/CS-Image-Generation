#!/usr/bin/env python3
"""Continuous random visualization farm for the production Rust binary.

The farm is intentionally small and stdlib-only:

* discover the implemented visualization catalog from ``--viz-list``;
* choose one target mode and one cryptographically random seed per job;
* expand the few targets that consume artifacts from earlier modes;
* keep a fixed rolling pool of release-binary processes busy;
* publish atomic JSON state for remote monitoring; and
* stop safely on low disk, repeated failures, signals, or a STOP file.

Every job keeps complete provenance in ``orchestrator/jobs`` while the Rust
package itself lands under ``output/<unique-name>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Final

EXPECTED_MODE_COUNT: Final = 69
DEFAULT_CONCURRENCY: Final = 4
DEFAULT_THREADS_PER_JOB: Final = 30
DEFAULT_MIN_FREE_GB: Final = 500.0
DEFAULT_MAX_FAILURE_STREAK: Final = 5
DEFAULT_TIMEOUT_HOURS: Final = 24.0 * 14.0
DEFAULT_POLL_SECONDS: Final = 5.0
DEFAULT_STATE_SECONDS: Final = 30.0

MODE_LINE_RE: Final = re.compile(r"^V\d+\s+([a-z0-9-]+)\s+\S+\s+[A-D]\s+implemented\s+")

# Only dependencies that materially improve or are required by on-disk
# artifact consumers belong here. Algorithmic dependencies are imported by
# Rust directly and must not be redundantly rendered.
PREREQUISITES: Final[dict[str, tuple[str, ...]]] = {
    "broadcast": (
        "editorial-retime",
        "epilogue",
        "corotating",
        "bullet-time",
        "mission-control",
    ),
    "trailer": ("sonification",),
    "tilt": ("depth-pack",),
    "powers-of-fate": ("basin-map",),
}

_signal_count = 0


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_viz_catalog(text: str) -> tuple[str, ...]:
    """Parse implemented mode flags from the stable ``--viz-list`` table."""
    flags = tuple(
        match.group(1)
        for line in text.splitlines()
        if (match := MODE_LINE_RE.match(line)) is not None
    )
    if len(flags) != EXPECTED_MODE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODE_COUNT} implemented modes from --viz-list, got {len(flags)}"
        )
    if len(set(flags)) != len(flags):
        raise ValueError("--viz-list contained duplicate implemented flags")
    return flags


def discover_viz_catalog(binary: Path, work_dir: Path) -> tuple[str, ...]:
    """Ask the release binary for the authoritative implemented catalog."""
    result = subprocess.run(
        [str(binary), "--viz-list"],
        cwd=work_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_viz_catalog(result.stdout)


def expand_viz_flags(target: str, catalog: Sequence[str]) -> tuple[str, ...]:
    """Return target + artifact prerequisites in Rust catalog order."""
    if target not in catalog:
        raise ValueError(f"target is not implemented: {target}")
    selected = {target, *PREREQUISITES.get(target, ())}
    missing = selected.difference(catalog)
    if missing:
        raise ValueError(f"prerequisites missing from catalog: {sorted(missing)}")
    return tuple(flag for flag in catalog if flag in selected)


def random_seed() -> str:
    """Generate a uniformly random 64-bit hexadecimal seed."""
    return "0x" + secrets.token_hex(8)


def choose_target(catalog: Sequence[str]) -> str:
    """Choose uniformly from every implemented target, with no exclusions."""
    if not catalog:
        raise ValueError("cannot choose from an empty visualization catalog")
    return secrets.choice(catalog)


def job_identity(target: str, seed: str, sequence: int, now: dt.datetime | None = None) -> str:
    """Build a readable, collision-resistant Rust ``--output`` name."""
    stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"random-{stamp}-{sequence:06d}-{target}-{seed.removeprefix('0x')}"


def build_job_command(
    binary: Path,
    seed: str,
    output_name: str,
    target: str,
    viz_flags: Sequence[str],
) -> tuple[str, ...]:
    """Build the exact production Rust invocation for one random target."""
    command = [
        str(binary),
        "--seed",
        seed,
        "--output",
        output_name,
        "--viz",
        ",".join(viz_flags),
        "--viz-quality",
        "final",
    ]
    if target == "celestial-atlas":
        command.extend(["--viz-seeds-dir", "output"])
    return tuple(command)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON through a sibling temporary and atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def disk_free_gb(path: Path) -> float:
    """Return free bytes at ``path`` as binary gigabytes."""
    return shutil.disk_usage(path).free / (1024.0**3)


def validate_target_artifacts(output_dir: Path, output_name: str, target: str) -> str | None:
    """Return an error when the selected target emitted no usable artifact."""
    package = output_dir / output_name
    manifest = package / "viz" / "manifest.json"
    target_dir = package / "viz" / target
    if not manifest.is_file():
        return "viz/manifest.json missing"
    if not target_dir.is_dir():
        return f"viz/{target} missing"
    if not any(path.is_file() and path.stat().st_size > 0 for path in target_dir.iterdir()):
        return f"viz/{target} contains no non-empty artifacts"
    return None


@dataclasses.dataclass(frozen=True)
class FarmConfig:
    """Immutable runtime configuration."""

    work_dir: Path
    binary: Path
    state_dir: Path
    output_dir: Path
    concurrency: int
    threads_per_job: int
    min_free_gb: float
    max_failure_streak: int
    timeout_seconds: float
    poll_seconds: float
    state_seconds: float
    max_jobs: int | None


@dataclasses.dataclass
class JobRecord:
    """Persistent provenance and lifecycle state for one random job."""

    job_id: str
    session_id: str
    git_head: str
    sequence: int
    seed: str
    target_mode: str
    viz_flags: tuple[str, ...]
    output_name: str
    rayon_threads: int
    command: tuple[str, ...]
    status: str
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


@dataclasses.dataclass
class RunningJob:
    """In-memory process state paired with its durable record."""

    record: JobRecord
    process: subprocess.Popen[str]
    log_handle: IO[str]
    log_path: Path
    metadata_path: Path
    started_monotonic: float


def _signal_handler(signum: int, _frame: object) -> None:
    """First signal drains; a second asks the main loop to kill workers."""
    del signum
    global _signal_count
    _signal_count += 1


def current_git_head(work_dir: Path) -> str:
    """Return deployed commit id, or ``unknown`` outside a git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work_dir,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def configure_logging(state_dir: Path) -> logging.Logger:
    """Configure one session file plus stdout for nohup diagnostics."""
    logger = logging.getLogger("viz_farm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(state_dir / "session.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


class VizFarm:
    """Rolling process pool and atomic status publisher."""

    def __init__(
        self, config: FarmConfig, catalog: tuple[str, ...], logger: logging.Logger
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.logger = logger
        self.session_id = secrets.token_hex(8)
        self.git_head = current_git_head(config.work_dir)
        self.running: dict[int, RunningJob] = {}
        self.jobs_started = 0
        self.jobs_ok = 0
        self.jobs_failed = 0
        self.failure_streak = 0
        self.last_error: str | None = None
        self.stop_reason: str | None = None
        self.last_state_write = 0.0
        self._lock_handle: IO[str] | None = None

    def acquire_lock(self) -> None:
        """Prevent duplicate farm supervisors in one checkout."""
        lock_path = self.config.state_dir / "farm.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise RuntimeError(
                "another viz_farm.py supervisor holds orchestrator/farm.lock"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._lock_handle = handle

    def _job_metadata_path(self, job_id: str) -> Path:
        return self.config.state_dir / "jobs" / f"{job_id}.json"

    def _job_log_path(self, job_id: str) -> Path:
        return self.config.state_dir / "jobs" / f"{job_id}.log"

    def _persist_job(self, job: JobRecord, path: Path) -> None:
        atomic_write_json(path, dataclasses.asdict(job))

    def _start_job(self) -> None:
        sequence = self.jobs_started + 1
        target = choose_target(self.catalog)
        seed = random_seed()
        output_name = job_identity(target, seed, sequence)
        viz_flags = expand_viz_flags(target, self.catalog)
        command = build_job_command(self.config.binary, seed, output_name, target, viz_flags)
        job_id = output_name
        metadata_path = self._job_metadata_path(job_id)
        log_path = self._job_log_path(job_id)
        record = JobRecord(
            job_id=job_id,
            session_id=self.session_id,
            git_head=self.git_head,
            sequence=sequence,
            seed=seed,
            target_mode=target,
            viz_flags=viz_flags,
            output_name=output_name,
            rayon_threads=self.config.threads_per_job,
            command=command,
            status="running",
            started_at=utc_now(),
        )
        self._persist_job(record, metadata_path)

        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(
            f"=== {record.started_at} job={job_id} target={target} seed={seed} "
            f"flags={','.join(viz_flags)} git={self.git_head} ===\n"
        )
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(self.config.threads_per_job)
        environment["RUST_BACKTRACE"] = "1"
        process = subprocess.Popen(
            command,
            cwd=self.config.work_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        running = RunningJob(
            record=record,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            metadata_path=metadata_path,
            started_monotonic=time.monotonic(),
        )
        self.running[process.pid] = running
        self.jobs_started += 1
        self.logger.info(
            "START pid=%d job=%s target=%s seed=%s flags=%s",
            process.pid,
            job_id,
            target,
            seed,
            ",".join(viz_flags),
        )

    def _finish_job(self, pid: int, return_code: int, error: str | None = None) -> None:
        running = self.running.pop(pid)
        elapsed = time.monotonic() - running.started_monotonic
        running.log_handle.flush()
        running.log_handle.close()

        artifact_error = None
        if return_code == 0 and error is None:
            artifact_error = validate_target_artifacts(
                self.config.output_dir,
                running.record.output_name,
                running.record.target_mode,
            )
        final_error = error or artifact_error
        success = return_code == 0 and final_error is None
        running.record.status = "ok" if success else "failed"
        running.record.finished_at = utc_now()
        running.record.exit_code = return_code
        running.record.elapsed_seconds = round(elapsed, 3)
        running.record.error = final_error
        self._persist_job(running.record, running.metadata_path)

        if success:
            self.jobs_ok += 1
            self.failure_streak = 0
            self.logger.info(
                "DONE pid=%d job=%s target=%s elapsed=%.1fs",
                pid,
                running.record.job_id,
                running.record.target_mode,
                elapsed,
            )
        else:
            self.jobs_failed += 1
            self.failure_streak += 1
            self.last_error = f"{running.record.job_id}: exit={return_code}" + (
                f", {final_error}" if final_error else ""
            )
            self.logger.error(
                "FAILED pid=%d job=%s target=%s exit=%d elapsed=%.1fs error=%s",
                pid,
                running.record.job_id,
                running.record.target_mode,
                return_code,
                elapsed,
                final_error,
            )
            if self.failure_streak >= self.config.max_failure_streak:
                self.stop_reason = "stopped_failures"
                self.logger.error(
                    "failure circuit breaker reached %d consecutive jobs; draining",
                    self.failure_streak,
                )

    def _terminate_group(self, running: RunningJob, force: bool) -> None:
        try:
            os.killpg(
                running.process.pid,
                signal.SIGKILL if force else signal.SIGTERM,
            )
        except ProcessLookupError:
            return

    def _reap_and_timeout(self) -> None:
        now = time.monotonic()
        completed: list[tuple[int, int, str | None]] = []
        for pid, running in list(self.running.items()):
            return_code = running.process.poll()
            if return_code is not None:
                completed.append((pid, return_code, None))
                continue
            if now - running.started_monotonic > self.config.timeout_seconds:
                self.logger.error(
                    "TIMEOUT pid=%d job=%s; terminating group", pid, running.record.job_id
                )
                self._terminate_group(running, force=False)
                try:
                    return_code = running.process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    self._terminate_group(running, force=True)
                    return_code = running.process.wait()
                completed.append((pid, return_code, "job exceeded wall-clock timeout"))
        for pid, return_code, error in completed:
            self._finish_job(pid, return_code, error)

    def _check_stop_controls(self) -> None:
        if self.stop_reason is not None:
            return
        if _signal_count >= 1:
            self.stop_reason = "stopped_signal"
            self.logger.warning("signal received; draining active jobs")
            return
        if (self.config.state_dir / "STOP").exists():
            self.stop_reason = "stopped_user"
            self.logger.warning("orchestrator/STOP present; draining active jobs")
            return
        if self.config.max_jobs is not None and self.jobs_started >= self.config.max_jobs:
            self.stop_reason = "completed_limit"
            self.logger.info("max-jobs limit reached; draining active jobs")

    def _fill_workers(self) -> None:
        if self.stop_reason is not None:
            return
        while len(self.running) < self.config.concurrency:
            free_gb = disk_free_gb(self.config.output_dir)
            if free_gb < self.config.min_free_gb:
                self.stop_reason = "stopped_low_disk"
                self.logger.error(
                    "free disk %.1f GB is below %.1f GB guard; draining",
                    free_gb,
                    self.config.min_free_gb,
                )
                return
            if self.config.max_jobs is not None and self.jobs_started >= self.config.max_jobs:
                self.stop_reason = "completed_limit"
                return
            self._start_job()

    def _force_stop_if_requested(self) -> None:
        if _signal_count < 2:
            return
        self.stop_reason = "forced_signal"
        self.logger.error("second signal received; killing %d active job(s)", len(self.running))
        for running in self.running.values():
            self._terminate_group(running, force=True)

    def _emergency_stop(self) -> None:
        """Terminate and reap every child after an unexpected supervisor error."""
        self.logger.exception(
            "unexpected supervisor error; terminating %d job(s)", len(self.running)
        )
        for running in self.running.values():
            self._terminate_group(running, force=False)
        deadline = time.monotonic() + 30.0
        while self.running and time.monotonic() < deadline:
            for pid, running in list(self.running.items()):
                return_code = running.process.poll()
                if return_code is not None:
                    self._finish_job(pid, return_code, "supervisor stopped unexpectedly")
            if self.running:
                time.sleep(0.25)
        for running in self.running.values():
            self._terminate_group(running, force=True)
        for pid, running in list(self.running.items()):
            return_code = running.process.wait()
            self._finish_job(pid, return_code, "supervisor stopped unexpectedly")

    def _state_payload(self) -> dict[str, object]:
        now = time.monotonic()
        in_flight: list[dict[str, object]] = []
        for pid, running in sorted(self.running.items()):
            try:
                log_age = max(0.0, time.time() - running.log_path.stat().st_mtime)
            except FileNotFoundError:
                log_age = -1.0
            in_flight.append(
                {
                    "pid": pid,
                    "job_id": running.record.job_id,
                    "seed": running.record.seed,
                    "target_mode": running.record.target_mode,
                    "viz_flags": list(running.record.viz_flags),
                    "elapsed_seconds": round(now - running.started_monotonic, 1),
                    "log_age_seconds": round(log_age, 1),
                    "log": str(running.log_path.relative_to(self.config.work_dir)),
                }
            )
        return {
            "session_id": self.session_id,
            "git_head": self.git_head,
            "state": self.stop_reason or "running",
            "updated_at": utc_now(),
            "supervisor_pid": os.getpid(),
            "concurrency": self.config.concurrency,
            "threads_per_job": self.config.threads_per_job,
            "jobs_started": self.jobs_started,
            "jobs_ok": self.jobs_ok,
            "jobs_failed": self.jobs_failed,
            "failure_streak": self.failure_streak,
            "free_disk_gb": round(disk_free_gb(self.config.output_dir), 1),
            "min_free_gb": self.config.min_free_gb,
            "last_error": self.last_error,
            "in_flight": in_flight,
        }

    def write_state(self, force: bool = False) -> None:
        """Refresh state.json at a bounded cadence or immediately on events."""
        now = time.monotonic()
        if not force and now - self.last_state_write < self.config.state_seconds:
            return
        atomic_write_json(self.config.state_dir / "state.json", self._state_payload())
        self.last_state_write = now

    def run(self) -> int:
        """Run until a stop condition is met and all active jobs are gone."""
        self.acquire_lock()
        self.logger.info(
            "farm start session=%s git=%s modes=%d concurrency=%d threads/job=%d min_free_gb=%.1f",
            self.session_id,
            self.git_head,
            len(self.catalog),
            self.config.concurrency,
            self.config.threads_per_job,
            self.config.min_free_gb,
        )
        self.write_state(force=True)
        try:
            while True:
                self._check_stop_controls()
                self._force_stop_if_requested()
                self._reap_and_timeout()
                self._fill_workers()
                self.write_state()
                if self.stop_reason is not None and not self.running:
                    break
                time.sleep(self.config.poll_seconds)
        except BaseException:
            self.stop_reason = "stopped_supervisor_error"
            self._emergency_stop()
            raise
        finally:
            self.write_state(force=True)
            if self._lock_handle is not None:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
        self.logger.info(
            "farm stopped reason=%s started=%d ok=%d failed=%d",
            self.stop_reason,
            self.jobs_started,
            self.jobs_ok,
            self.jobs_failed,
        )
        return 2 if self.stop_reason == "stopped_failures" else 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/release/three_body_problem")
    parser.add_argument("--state-dir", default="orchestrator")
    parser.add_argument("--concurrency", type=positive_int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--threads-per-job",
        type=positive_int,
        default=DEFAULT_THREADS_PER_JOB,
    )
    parser.add_argument("--min-free-gb", type=positive_float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument(
        "--max-failure-streak",
        type=positive_int,
        default=DEFAULT_MAX_FAILURE_STREAK,
    )
    parser.add_argument(
        "--timeout-hours",
        type=positive_float,
        default=DEFAULT_TIMEOUT_HOURS,
    )
    parser.add_argument("--poll-seconds", type=positive_float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--state-seconds", type=positive_float, default=DEFAULT_STATE_SECONDS)
    parser.add_argument(
        "--max-jobs",
        type=positive_int,
        help="optional finite job count for controlled validation runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    work_dir = Path.cwd().resolve()
    binary = (work_dir / args.binary).resolve()
    state_dir = (work_dir / args.state_dir).resolve()
    output_dir = work_dir / "output"
    if not binary.is_file():
        print(f"ERROR: release binary not found: {binary}", file=sys.stderr)
        return 1
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "jobs").mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    logger = configure_logging(state_dir)
    try:
        catalog = discover_viz_catalog(binary, work_dir)
        config = FarmConfig(
            work_dir=work_dir,
            binary=binary,
            state_dir=state_dir,
            output_dir=output_dir,
            concurrency=args.concurrency,
            threads_per_job=args.threads_per_job,
            min_free_gb=args.min_free_gb,
            max_failure_streak=args.max_failure_streak,
            timeout_seconds=args.timeout_hours * 3600.0,
            poll_seconds=args.poll_seconds,
            state_seconds=args.state_seconds,
            max_jobs=args.max_jobs,
        )
        global _signal_count
        _signal_count = 0
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        return VizFarm(config, catalog, logger).run()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        logger.exception("farm startup failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
