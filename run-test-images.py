#!/usr/bin/env python3
"""
Batch runner for Three Body Problem image generation.

Continuously generates images with random seeds using the production
configuration. Uses a rolling pool to keep exactly CONCURRENT_SIMS
slots busy at all times. Runs forever until Ctrl+C.

Screen: compact progress line every few completions.
File:   full subprocess output written to run.log for debugging.
"""

import concurrent.futures
import logging
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

CONCURRENT_SIMS = 3
BINARY = "./target/release/three_body_problem"
LOG_FILE = "run.log"
SIM_TIMEOUT = 86400  # seconds per simulation (24 hours)
REPORT_EVERY = 3     # print a status line every N completions

# ---------------------------------------------------------------------------
# Logging setup: file gets everything, console gets one-liners
# ---------------------------------------------------------------------------

logger = logging.getLogger("run")
logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    if not os.path.isfile(BINARY):
        print(f"Error: binary not found at {BINARY}")
        print("Build it first:  cargo build --release")
        sys.exit(1)
    if not os.access(BINARY, os.X_OK):
        print(f"Error: {BINARY} is not executable")
        sys.exit(1)
    for d in ("pics", "vids"):
        Path(d).mkdir(exist_ok=True)


def random_seed() -> str:
    return "0x" + secrets.token_hex(6)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def run_one(seed: str, run_id: int) -> tuple[bool, str, float]:
    """Returns (success, seed, elapsed_secs)."""
    filename = seed[2:]
    cmd = [BINARY, "--seed", seed, "--output", filename]

    logger.debug(f"[{run_id}] START {seed}  cmd={' '.join(cmd)}")
    t0 = time.monotonic()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT)
        elapsed = time.monotonic() - t0

        if proc.stdout:
            logger.debug(f"[{run_id}] stdout:\n{proc.stdout.rstrip()}")
        if proc.stderr:
            logger.debug(f"[{run_id}] stderr:\n{proc.stderr.rstrip()}")

        if proc.returncode == 0:
            logger.info(f"[{run_id}] OK    {seed}  ({fmt_duration(elapsed)})")
            return (True, seed, elapsed)

        logger.warning(f"[{run_id}] FAIL  {seed}  exit={proc.returncode}  ({fmt_duration(elapsed)})")
        return (False, seed, elapsed)

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error(f"[{run_id}] TIMEOUT {seed}  ({fmt_duration(elapsed)})")
        return (False, seed, elapsed)

    except OSError as exc:
        elapsed = time.monotonic() - t0
        logger.error(f"[{run_id}] OS ERROR {seed}: {exc}")
        return (False, seed, elapsed)

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(f"[{run_id}] UNEXPECTED {seed}: {exc}")
        return (False, seed, elapsed)


# ---------------------------------------------------------------------------
# Main loop — rolling pool keeps all slots busy at all times
# ---------------------------------------------------------------------------

def main() -> None:
    check_prerequisites()

    logger.info(f"{'=' * 60}")
    logger.info(f"Session started  concurrency={CONCURRENT_SIMS}")
    logger.info(f"{'=' * 60}")

    print(f"Three Body Problem batch runner  ({CONCURRENT_SIMS} concurrent)")
    print(f"Detailed logs -> {LOG_FILE}")
    print("Ctrl+C to stop gracefully (twice to force)\n")

    run_id = 0
    ok_total = 0
    fail_total = 0
    completions_since_report = 0
    t_session = time.monotonic()

    in_flight: dict[concurrent.futures.Future, tuple[int, str]] = {}

    shutdown = False
    orig_sigint = signal.getsignal(signal.SIGINT)

    def on_sigint(_sig, _frame):
        nonlocal shutdown
        if shutdown:
            signal.signal(signal.SIGINT, orig_sigint)
            raise KeyboardInterrupt
        shutdown = True
        print("\n-- stopping: draining in-flight jobs --")

    signal.signal(signal.SIGINT, on_sigint)

    def submit_next(pool):
        nonlocal run_id
        seed = random_seed()
        run_id += 1
        fut = pool.submit(run_one, seed, run_id)
        in_flight[fut] = (run_id, seed)

    def print_status():
        total = ok_total + fail_total
        elapsed = fmt_duration(time.monotonic() - t_session)
        line = f"  completed {total}  (+{ok_total} ok"
        if fail_total:
            line += f"  -{fail_total} fail"
        line += f")  {elapsed}"
        print(line)
        logger.info(line)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_SIMS) as pool:
            for _ in range(CONCURRENT_SIMS):
                submit_next(pool)

            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for fut in done:
                    success, _seed, _elapsed = fut.result()
                    del in_flight[fut]

                    if success:
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
        logger.info(summary)
        logger.info("Session ended\n")


if __name__ == "__main__":
    main()
