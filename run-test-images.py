#!/usr/bin/env python3
"""
Batch runner for Three Body Problem image generation.

Default mode generates 360-degree orbit previews for visual evaluation:
random seeds, fast preview settings, and a turntable video per seed
(`output/<seed>/videos/web/orbit.mp4`) plus the master still for QA scoring.
Pass `--full-package` to restore the legacy behavior that renders and
validates the complete production asset package per seed.

Uses a rolling pool to keep all worker slots busy at all times. Runs until
Ctrl+C, or stops after `--count N` successful seeds.

Screen: compact progress line every few completions (plus each finished
orbit preview path in the default mode).
File:   full subprocess output written to run.log for debugging.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import secrets
import signal
import subprocess
import sys
import time
import typing
from pathlib import Path

from _utils import check_ffmpeg, compute_aesthetic_metrics, fmt_duration, resolve_binary
from run import EXPECTED_SPECTRAL_BINS, REQUIRED_PACKAGE_FILES, SPECTRAL_FILE_RE

CONCURRENT_SIMS = 3
BINARY = "./target/release/three_body_problem"
LOG_FILE = "run.log"
SIM_TIMEOUT = 86400  # seconds per simulation (24 hours)
REPORT_EVERY = 3  # print a status line every N completions

# Fast preview settings for orbit evaluation batches: skip the main/spectral
# videos, shrink the search and resolutions, and render a short seamless-loop
# turntable per seed. Tuned for throughput while staying representative.
ORBIT_PREVIEW_RUST_ARGS: tuple[str, ...] = (
    "--image-only",
    "--orbit-video",
    "--fast-encode",
    "--sims",
    "30000",
    "--resolution",
    "1920x1242",
    "--orbit-resolution",
    "1280x828",
    "--orbit-seconds",
    "8",
    "--orbit-fps",
    "24",
    "--orbit-step-stride",
    "4",
)

# Minimal artifact contract for orbit preview mode.
ORBIT_PREVIEW_REQUIRED_FILES: tuple[str, ...] = (
    "images/source/master.png",
    "videos/web/orbit.mp4",
    "videos/hq/orbit.mp4",
)


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
    aesthetic_score: float | None = None
    missing_parts: tuple[str, ...] = ()


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


def estimate_aesthetic_score(seed: str, run_id: int) -> float | None:
    """Score a render with real image-space metrics (see `_utils`).

    Decodes a small proxy frame via ffmpeg and measures ink coverage,
    colorfulness, hue entropy, luminance spread, flat-veil fraction, and
    crisp line energy, replacing the old PNG-file-size heuristic. Returns
    ``None`` when the image is missing or undecodable.
    """
    image_path = Path("output") / seed / "images" / "source" / "master.png"
    metrics = compute_aesthetic_metrics(image_path)
    if metrics is None:
        return None

    logger.debug(
        "[%d] QA    %s  coverage=%.3f colorfulness=%.3f hue_entropy=%.3f "
        "spread=%.3f veil=%.3f crisp=%.3f lush=%.3f",
        run_id,
        seed,
        metrics.coverage,
        metrics.colorfulness,
        metrics.hue_entropy,
        metrics.luminance_spread,
        metrics.veil_fraction,
        metrics.crispness,
        metrics.lushness,
    )
    return metrics.score


def missing_local_package_parts(seed_dir: Path) -> list[str]:
    """Return missing files/groups using the same package contract as run.py."""
    missing: list[str] = []
    for filename in REQUIRED_PACKAGE_FILES:
        path = seed_dir / filename
        if not path.is_file():
            missing.append(filename)

    spectral_dir = seed_dir / "spectral"
    spectral_bins: set[int] = set()
    if spectral_dir.is_dir():
        for path in spectral_dir.iterdir():
            if not path.is_file():
                continue
            match = SPECTRAL_FILE_RE.match(path.name)
            if match:
                bin_idx = int(match.group("bin"))
                if bin_idx in EXPECTED_SPECTRAL_BINS:
                    spectral_bins.add(bin_idx)
    else:
        missing.append("spectral/")

    missing_bins = EXPECTED_SPECTRAL_BINS - spectral_bins
    if missing_bins:
        missing.append(f"spectral/*.png ({len(missing_bins)} missing)")

    return missing


def missing_orbit_preview_parts(seed_dir: Path) -> list[str]:
    """Return missing files for the relaxed orbit-preview artifact contract."""
    return [
        filename for filename in ORBIT_PREVIEW_REQUIRED_FILES if not (seed_dir / filename).is_file()
    ]


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------


def run_one(binary: str, seed: str, run_id: int, full_package: bool) -> SimResult:
    """Run the generator for a single seed and return the outcome."""
    cmd = [
        binary,
        "--seed",
        seed,
        "--output",
        seed,
    ]
    if not full_package:
        cmd.extend(ORBIT_PREVIEW_RUST_ARGS)

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
            seed_dir = Path("output") / seed
            missing_parts = (
                missing_local_package_parts(seed_dir)
                if full_package
                else missing_orbit_preview_parts(seed_dir)
            )
            if missing_parts:
                logger.warning(
                    "[%d] PACKAGE INCOMPLETE %s  missing=%s",
                    run_id,
                    seed,
                    ", ".join(missing_parts),
                )
                return SimResult(False, seed, elapsed, None, tuple(missing_parts))

            aesthetic_score = estimate_aesthetic_score(seed, run_id)
            if aesthetic_score is None:
                logger.warning("[%d] QA    %s  images/source/master.png missing", run_id, seed)
                return SimResult(False, seed, elapsed, None, ("images/source/master.png",))
            elif aesthetic_score < 25.0:
                logger.warning(
                    "[%d] QA    %s  low aesthetic_score=%.1f",
                    run_id,
                    seed,
                    aesthetic_score,
                )
            else:
                logger.info(
                    "[%d] QA    %s  aesthetic_score=%.1f",
                    run_id,
                    seed,
                    aesthetic_score,
                )
            if not full_package:
                orbit_path = seed_dir / "videos" / "web" / "orbit.mp4"
                logger.info("[%d] ORBIT %s  %s", run_id, seed, orbit_path)
                print(f"  orbit ready: {orbit_path}  (score {aesthetic_score:.1f})")
            logger.info("[%d] OK    %s  (%s)", run_id, seed, fmt_duration(elapsed))
            return SimResult(True, seed, elapsed, aesthetic_score)

        logger.warning(
            "[%d] FAIL  %s  exit=%d  (%s)",
            run_id,
            seed,
            proc.returncode,
            fmt_duration(elapsed),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse batch-runner CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Continuously generate random seeds with the Rust generator. "
            "Default mode renders fast orbit previews for visual evaluation; "
            "--full-package restores the legacy production package run."
        )
    )
    parser.add_argument(
        "--full-package",
        action="store_true",
        help="render and validate the complete production asset package per seed",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="stop after N successful seeds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENT_SIMS,
        metavar="N",
        help=f"concurrent generator processes (default: {CONCURRENT_SIMS})",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    full_package: bool = bool(args.full_package)
    target_count: int | None = args.count
    concurrency: int = max(1, int(args.concurrency))

    binary = check_prerequisites()
    setup_logging()

    binary_str = str(binary)
    mode = "full-package" if full_package else "orbit-preview"

    logger.info("=" * 60)
    logger.info(
        "Session started  concurrency=%d  mode=%s  target=%s",
        concurrency,
        mode,
        "unbounded" if target_count is None else str(target_count),
    )
    if not full_package:
        logger.info("Preview generator args: %s", " ".join(ORBIT_PREVIEW_RUST_ARGS))
    logger.info("=" * 60)

    print(f"Three Body Problem batch runner  ({concurrency} concurrent, {mode} mode)")
    if full_package:
        print("High-resolution full-package mode: Rust CLI default resolution")
    else:
        print("Orbit preview mode: still + 8s 360-degree turntable per seed")
        if target_count is not None:
            print(f"Stopping after {target_count} successful seed(s)")
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

    def target_reached() -> bool:
        return target_count is not None and ok_total >= target_count

    def submit_next(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        nonlocal run_id
        seed = random_seed()
        run_id += 1
        fut = pool.submit(run_one, binary_str, seed, run_id, full_package)
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(concurrency):
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

                    if target_reached():
                        if not shutdown:
                            print(
                                f"-- target of {target_count} successful seed(s) reached: "
                                "draining in-flight jobs --"
                            )
                        shutdown = True

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
