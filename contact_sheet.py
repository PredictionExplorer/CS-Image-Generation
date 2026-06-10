#!/usr/bin/env python3
"""
Contact-sheet renderer for fast visual curation and look regression.

Two modes:

  Random QA sheet (default):
      python3 contact_sheet.py --count 24
  Golden gallery (fixed seed set, for before/after look regression):
      python3 contact_sheet.py --golden

Each seed is rendered at preview quality into ``output/<prefix>-<seed>/``,
scored with the shared image-space aesthetic metrics, and the stills are
tiled into one PNG with ffmpeg. Stdlib + ffmpeg only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import typing
from pathlib import Path

from _utils import (
    AestheticMetrics,
    check_ffmpeg,
    compute_aesthetic_metrics,
    fmt_duration,
    resolve_binary,
)

BINARY = "./target/release/three_body_problem"
GOLDEN_SEEDS_FILE = Path("ci/golden_seeds.txt")
SIM_TIMEOUT = 7200  # seconds per render
LOW_SCORE_WARNING = 25.0


class SheetEntry(typing.NamedTuple):
    seed: str
    image: Path | None
    metrics: AestheticMetrics | None
    elapsed: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24, help="random seeds to render")
    parser.add_argument(
        "--golden",
        action="store_true",
        help=f"render the fixed golden seed set from {GOLDEN_SEEDS_FILE}",
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=None,
        help="render seeds listed in a file (one hex seed per line, # comments allowed)",
    )
    parser.add_argument("--resolution", default="512x331", help="render resolution WxH")
    parser.add_argument("--sims", type=int, default=300, help="Borda search size per seed")
    parser.add_argument("--steps", type=int, default=40000, help="simulation steps per seed")
    parser.add_argument("--jobs", type=int, default=3, help="concurrent renders")
    parser.add_argument("--columns", type=int, default=0, help="sheet columns (0 = auto)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="sheet path (default contact_sheet.png, or golden_gallery.png with --golden)",
    )
    return parser.parse_args()


def read_seeds_file(path: Path) -> list[str]:
    if not path.is_file():
        print(f"Error: seeds file not found: {path}", file=sys.stderr)
        sys.exit(1)
    seeds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            seeds.append(token)
    if not seeds:
        print(f"Error: no seeds found in {path}", file=sys.stderr)
        sys.exit(1)
    return seeds


def render_one(binary: str, seed: str, prefix: str, args: argparse.Namespace) -> SheetEntry:
    """Render one seed at preview quality and score the still."""
    out_name = f"{prefix}-{seed}"
    cmd = [
        binary,
        "--seed",
        seed,
        "--output",
        out_name,
        "--resolution",
        args.resolution,
        "--sims",
        str(args.sims),
        "--steps",
        str(args.steps),
        "--fast-encode",
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  {seed}  RENDER ERROR: {exc}", file=sys.stderr)
        return SheetEntry(seed, None, None, time.monotonic() - t0)
    elapsed = time.monotonic() - t0

    image = Path("output") / out_name / "image.png"
    if proc.returncode != 0 or not image.exists():
        print(f"  {seed}  FAILED (exit={proc.returncode}, {fmt_duration(elapsed)})")
        return SheetEntry(seed, None, None, elapsed)

    metrics = compute_aesthetic_metrics(image)
    return SheetEntry(seed, image, metrics, elapsed)


def tile_sheet(images: list[Path], columns: int, output: Path) -> None:
    """Tile equally sized stills into one sheet PNG via ffmpeg."""
    rows = math.ceil(len(images) / columns)
    with tempfile.TemporaryDirectory(prefix="contact-sheet-") as tmp:
        tmp_dir = Path(tmp)
        for index, image in enumerate(images):
            shutil.copyfile(image, tmp_dir / f"frame_{index:04d}.png")
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-filter_complex",
            f"tile={columns}x{rows}",
            "-frames:v",
            "1",
            str(output),
        ]
        subprocess.run(cmd, check=True, timeout=600)


def report(entries: list[SheetEntry]) -> None:
    print("\nseed            score  coverage  colorful  hue-ent  spread   veil  crisp   time")
    print("-" * 86)
    scored = [entry for entry in entries if entry.metrics is not None]
    for entry in sorted(scored, key=lambda e: e.metrics.score if e.metrics else 0.0):
        m = entry.metrics
        assert m is not None
        flag = "  <- LOW" if m.score < LOW_SCORE_WARNING else ""
        print(
            f"{entry.seed:<15s} {m.score:5.1f}  {m.coverage:8.3f}  {m.colorfulness:8.3f}"
            f"  {m.hue_entropy:7.3f}  {m.luminance_spread:6.3f}"
            f"  {m.veil_fraction:5.3f}  {m.crispness:5.3f}  {fmt_duration(entry.elapsed):>5s}"
            f"{flag}"
        )
    if scored:
        mean_score = sum(e.metrics.score for e in scored if e.metrics) / len(scored)
        print("-" * 86)
        print(f"{len(scored)} rendered, mean score {mean_score:.1f}")
    failed = [entry.seed for entry in entries if entry.image is None]
    if failed:
        print(f"failed seeds: {', '.join(failed)}")


def main() -> int:
    args = parse_args()
    binary = resolve_binary(BINARY)
    check_ffmpeg()
    Path("output").mkdir(exist_ok=True)

    if args.golden:
        seeds = read_seeds_file(GOLDEN_SEEDS_FILE)
        prefix = "golden"
        output = args.output or Path("golden_gallery.png")
    elif args.seeds_file is not None:
        seeds = read_seeds_file(args.seeds_file)
        prefix = "sheet"
        output = args.output or Path("contact_sheet.png")
    else:
        seeds = ["0x" + secrets.token_hex(6) for _ in range(max(1, args.count))]
        prefix = "sheet"
        output = args.output or Path("contact_sheet.png")

    mode = "golden gallery" if args.golden else "contact sheet"
    print(f"Rendering {len(seeds)} seed(s) for {mode} ({args.jobs} concurrent)...")

    entries: list[SheetEntry] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(render_one, str(binary), seed, prefix, args) for seed in seeds]
        for future in futures:
            entry = future.result()
            entries.append(entry)
            if entry.metrics is not None:
                print(
                    f"  {entry.seed}  score={entry.metrics.score:5.1f}"
                    f"  ({fmt_duration(entry.elapsed)})"
                )

    images = [entry.image for entry in entries if entry.image is not None]
    if not images:
        print("Error: no renders succeeded; nothing to tile.", file=sys.stderr)
        return 1

    columns = args.columns if args.columns > 0 else max(1, math.ceil(math.sqrt(len(images))))
    tile_sheet(images, columns, output)
    report(entries)
    print(f"\nSheet written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
