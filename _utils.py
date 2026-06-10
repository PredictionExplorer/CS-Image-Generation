"""Shared helpers for the Python runner scripts."""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import sys
import typing
from pathlib import Path

GENERATOR_CANDIDATES: list[str] = [
    "./target/release/three_body_problem",
    "./three_body_problem",
]

# Edge length of the downscaled analysis frame used for aesthetic metrics.
ANALYSIS_SIZE = 96
# Rec. 709 luma threshold above which a pixel counts as "lit".
LIT_LUMA_THRESHOLD = 0.02
# Minimum channel spread (0-255) for a pixel to contribute to hue statistics.
CHROMATIC_SPREAD_THRESHOLD = 12
# Number of hue histogram bins for the hue-entropy metric.
HUE_BINS = 12


def fmt_duration(seconds: float) -> str:
    """Format an elapsed-seconds value as a compact human string (e.g. '3m42s')."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def check_ffmpeg() -> None:
    """Exit with a clear message if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH", file=sys.stderr)
        print("The Rust generator requires ffmpeg for MP4 encoding.", file=sys.stderr)
        print("Install it:  brew install ffmpeg  /  apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def resolve_binary(path: str | Path) -> Path:
    """Validate that *path* exists and is executable, or exit."""
    p = Path(path)
    if not p.is_file():
        print(f"Error: binary not found at {p}", file=sys.stderr)
        print("Build it first:  cargo build --release", file=sys.stderr)
        sys.exit(1)
    if not os.access(p, os.X_OK):
        print(f"Error: {p} is not executable", file=sys.stderr)
        sys.exit(1)
    return p


class AestheticMetrics(typing.NamedTuple):
    """Image-space quality metrics for one rendered frame (all in [0, 1] except score)."""

    coverage: float
    colorfulness: float
    hue_entropy: float
    luminance_spread: float
    veil_fraction: float
    crispness: float
    score: float


def _decode_rgb_frame(image_path: Path, size: int) -> bytes | None:
    """Decode *image_path* to a size x size raw RGB24 frame via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(image_path),
        "-vf",
        f"scale={size}:{size}",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or len(proc.stdout) != size * size * 3:
        return None
    return proc.stdout


def _coverage_band_score(coverage: float) -> float:
    """Map an ink-coverage fraction onto [0, 1] with a flat ideal band."""
    floor, band_low, band_high, ceil = 0.02, 0.06, 0.45, 0.85
    if coverage <= floor or coverage >= ceil:
        return 0.0
    if coverage < band_low:
        return (coverage - floor) / (band_low - floor)
    if coverage > band_high:
        return (ceil - coverage) / (ceil - band_high)
    return 1.0


def compute_aesthetic_metrics(image_path: Path) -> AestheticMetrics | None:
    """Compute real image-space aesthetic metrics for a rendered PNG.

    Decodes a small proxy frame with ffmpeg (already a hard dependency of the
    generator) and measures ink coverage, Hasler-Suesstrunk colorfulness, hue
    entropy, luminance spread, low-gradient veil fraction, and crisp line
    energy. Returns ``None`` when the image is missing or cannot be decoded.
    Stdlib + ffmpeg only.
    """
    if not image_path.exists():
        return None
    raw = _decode_rgb_frame(image_path, ANALYSIS_SIZE)
    if raw is None:
        return None

    total_pixels = ANALYSIS_SIZE * ANALYSIS_SIZE
    lit_lumas: list[float] = []
    luma_grid = [0.0] * total_pixels
    rg_values: list[float] = []
    yb_values: list[float] = []
    hue_histogram = [0] * HUE_BINS

    for offset in range(0, len(raw), 3):
        r, g, b = raw[offset], raw[offset + 1], raw[offset + 2]
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        pixel_idx = offset // 3
        luma_grid[pixel_idx] = luma
        if luma <= LIT_LUMA_THRESHOLD:
            continue
        lit_lumas.append(luma)
        rg_values.append(float(r - g))
        yb_values.append(0.5 * (r + g) - b)

        if max(r, g, b) - min(r, g, b) >= CHROMATIC_SPREAD_THRESHOLD:
            hue = math.atan2(math.sqrt(3.0) * (g - b), 2.0 * r - g - b)
            bin_idx = int((hue + math.pi) / (2.0 * math.pi) * HUE_BINS) % HUE_BINS
            hue_histogram[bin_idx] += 1

    coverage = len(lit_lumas) / total_pixels
    if not lit_lumas:
        return AestheticMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Hasler-Suesstrunk colorfulness over lit pixels, normalized to ~[0, 1].
    if len(rg_values) > 1:
        std_term = math.hypot(statistics.pstdev(rg_values), statistics.pstdev(yb_values))
        mean_term = math.hypot(statistics.fmean(rg_values), statistics.fmean(yb_values))
        colorfulness = min((std_term + 0.3 * mean_term) / 255.0 * 2.4, 1.0)
    else:
        colorfulness = 0.0

    chromatic_total = sum(hue_histogram)
    if chromatic_total > 0:
        entropy = -sum(
            (count / chromatic_total) * math.log(count / chromatic_total)
            for count in hue_histogram
            if count > 0
        )
        hue_entropy = entropy / math.log(HUE_BINS)
    else:
        hue_entropy = 0.0

    sorted_lumas = sorted(lit_lumas)
    p50 = sorted_lumas[int(0.50 * (len(sorted_lumas) - 1))]
    p95 = sorted_lumas[int(0.95 * (len(sorted_lumas) - 1))]
    luminance_spread = max(0.0, p95 - p50)

    veiled = 0
    crisp = 0
    lit = 0
    for idx, luma in enumerate(luma_grid):
        if luma <= LIT_LUMA_THRESHOLD:
            continue
        x = idx % ANALYSIS_SIZE
        y = idx // ANALYSIS_SIZE
        gradient = 0.0
        if x > 0:
            gradient = max(gradient, abs(luma - luma_grid[idx - 1]))
        if x + 1 < ANALYSIS_SIZE:
            gradient = max(gradient, abs(luma - luma_grid[idx + 1]))
        if y > 0:
            gradient = max(gradient, abs(luma - luma_grid[idx - ANALYSIS_SIZE]))
        if y + 1 < ANALYSIS_SIZE:
            gradient = max(gradient, abs(luma - luma_grid[idx + ANALYSIS_SIZE]))
        lit += 1
        if gradient < 6.0 / 255.0:
            veiled += 1
        if gradient > 25.0 / 255.0:
            crisp += 1

    veil_fraction = veiled / lit if lit else 0.0
    crispness = crisp / lit if lit else 0.0
    veil_penalty = max(veil_fraction - 0.12, 0.0) / 0.88

    score = 100.0 * (
        0.25 * _coverage_band_score(coverage)
        + 0.20 * colorfulness
        + 0.16 * hue_entropy
        + 0.14 * min(luminance_spread / 0.45, 1.0)
        + 0.20 * crispness
        + 0.05 * (1.0 - veil_fraction)
    )
    score = max(0.0, score - 40.0 * veil_penalty)
    return AestheticMetrics(
        coverage,
        colorfulness,
        hue_entropy,
        luminance_spread,
        veil_fraction,
        crispness,
        score,
    )
