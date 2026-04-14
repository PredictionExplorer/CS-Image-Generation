#!/usr/bin/env python3
"""
Batch runner for Three Body Problem image generation.

Continuously generates images with random seeds using the production
configuration. Uses a rolling pool to keep exactly CONCURRENT_SIMS
slots busy at all times. Runs forever until Ctrl+C.

Screen: compact progress line every few completions.
File:   full subprocess output written to run.log for debugging.
"""

from __future__ import annotations

import concurrent.futures
import logging
import secrets
import signal
import subprocess
import sys
import time
import typing
from pathlib import Path

from _utils import check_ffmpeg, fmt_duration, resolve_binary

CONCURRENT_SIMS = 3
BINARY = "./target/release/three_body_problem"
LOG_FILE = "run.log"
SIM_TIMEOUT = 86400   # seconds per simulation (24 hours)
REPORT_EVERY = 3      # print a status line every N completions

TEST_SIMS = 1_000
TEST_STEPS = 100_000


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("run")


def setup_logging() -> None:
    """Configure file + console logging (called once from main)."""
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SimResult(typing.NamedTuple):
    success: bool
    seed: str
    elapsed: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_prerequisites() -> Path:
    """Validate binary and ffmpeg, ensure output dir exists. Returns binary path."""
    binary = resolve_binary(BINARY)
    check_ffmpeg()
    Path("output").mkdir(exist_ok=True)
    return binary


def random_seed() -> str:
    return "0x" + secrets.token_hex(6)


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------


def run_one(binary: str, seed: str, run_id: int) -> SimResult:
    """Run the generator for a single seed and return the outcome."""
    cmd = [
        binary,
        "--seed", seed,
        "--output", seed,
        "--sims", str(TEST_SIMS),
        "--steps", str(TEST_STEPS),
        "--fast-encode",
    ]

    logger.debug("[%d] START %s  cmd=%s", run_id, seed, " ".join(cmd))
    t0 = time.monotonic()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT)
        elapsed = time.monotonic() - t0

        if proc.stdout:
            logger.debug("[%d] stdout:\n%s", run_id, proc.stdout.rstrip())
        if proc.stderr:
            logger.debug("[%d] stderr:\n%s", run_id, proc.stderr.rstrip())

        if proc.returncode == 0:
            logger.info("[%d] OK    %s  (%s)", run_id, seed, fmt_duration(elapsed))
            return SimResult(True, seed, elapsed)

        logger.warning(
            "[%d] FAIL  %s  exit=%d  (%s)",
            run_id, seed, proc.returncode, fmt_duration(elapsed),
        )
        return SimResult(False, seed, elapsed)

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("[%d] TIMEOUT %s  (%s)", run_id, seed, fmt_duration(elapsed))
        return SimResult(False, seed, elapsed)

    except OSError as exc:
        elapsed = time.monotonic() - t0
        logger.error("[%d] OS ERROR %s: %s", run_id, seed, exc)
        return SimResult(False, seed, elapsed)

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("[%d] UNEXPECTED %s: %s", run_id, seed, exc)
        return SimResult(False, seed, elapsed)


# ---------------------------------------------------------------------------
# Main loop -- rolling pool keeps all slots busy at all times
# ---------------------------------------------------------------------------


def main() -> int:
    binary = check_prerequisites()
    setup_logging()

    binary_str = str(binary)

    logger.info("=" * 60)
    logger.info("Session started  concurrency=%d", CONCURRENT_SIMS)
    logger.info("=" * 60)

    print(f"Three Body Problem batch runner  ({CONCURRENT_SIMS} concurrent)")
    print(f"Detailed logs -> {LOG_FILE}")
    print("Ctrl+C to stop gracefully (twice to force)\n")

    run_id = 0
    ok_total = 0
    fail_total = 0
    completions_since_report = 0
    t_session = time.monotonic()

    in_flight: dict[concurrent.futures.Future[SimResult], tuple[int, str]] = {}

    shutdown = False
    orig_sigint = signal.getsignal(signal.SIGINT)

    def on_sigint(_sig: int, _frame: object) -> None:
        nonlocal shutdown
        if shutdown:
            signal.signal(signal.SIGINT, orig_sigint)
            raise KeyboardInterrupt
        shutdown = True
        print("\n-- stopping: draining in-flight jobs --")

    signal.signal(signal.SIGINT, on_sigint)

    def submit_next(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        nonlocal run_id
        seed = random_seed()
        run_id += 1
        fut = pool.submit(run_one, binary_str, seed, run_id)
        in_flight[fut] = (run_id, seed)

    def print_status() -> None:
        total = ok_total + fail_total
        elapsed = fmt_duration(time.monotonic() - t_session)
        line = f"  completed {total}  (+{ok_total} ok"
        if fail_total:
            line += f"  -{fail_total} fail"
        line += f")  {elapsed}"
        print(line)
        logger.info("%s", line)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_SIMS) as pool:
            for _ in range(CONCURRENT_SIMS):
                submit_next(pool)

            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for fut in done:
                    try:
                        result = fut.result()
                    except Exception:
                        logger.exception("Unhandled exception from worker")
                        result = SimResult(False, in_flight[fut][1], 0.0)

                    del in_flight[fut]

                    if result.success:
                        ok_total += 1
                    else:
                        fail_total += 1

                    completions_since_report += 1
                    if completions_since_report >= REPORT_EVERY:
                        print_status()
                        completions_since_report = 0

                    if not shutdown:
                        submit_next(pool)

    except KeyboardInterrupt:
        pass

    finally:
        signal.signal(signal.SIGINT, orig_sigint)
        total = ok_total + fail_total
        elapsed = fmt_duration(time.monotonic() - t_session)

        summary = f"\nDone: {ok_total} ok, {fail_total} failed / {total} total in {elapsed}"
        print(summary)
        logger.info("%s", summary)
        logger.info("Session ended\n")

    return 1 if fail_total > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
