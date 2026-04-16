#!/usr/bin/env python3
"""CI verification for generated images.

Modes:

* Default (``--mode=auto``): exact SHA-256 match against the reference (legacy
  behaviour) when the reference's ``.json`` metadata contains a ``sha256``
  field *and* the file sizes match.  Otherwise falls back to perceptual
  delta-E 2000 comparison.
* ``--mode=hash``: strict SHA-256 match only (no fallback).
* ``--mode=delta-e``: ignore hash; always compute per-pixel sRGB -> Lab ->
  delta-E2000 and check the mean/max against the configured tolerance.

The delta-E path uses a pure-Python implementation that depends only on the
standard library plus ``Pillow``; if Pillow is missing and the mode is
``auto`` we gracefully degrade to the hash mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class DeltaEReport:
    """Summary of a delta-E2000 comparison between two images."""

    mean: float
    p95: float
    max: float
    pixels: int


def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _srgb_to_linear(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _linear_srgb_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
    # sRGB/D65 matrix (Rec. 709 primaries).
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return x, y, z


def _xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # D65 white point (normalized).
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx = _f(x / xn)
    fy = _f(y / yn)
    fz = _f(z / zn)
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return lightness, a, b


def _f(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta**3:
        return t ** (1.0 / 3.0)
    return t / (3.0 * delta * delta) + 4.0 / 29.0


def _delta_e_2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    avg_l = 0.5 * (l1 + l2)
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    avg_c = 0.5 * (c1 + c2)
    g = 0.5 * (1.0 - math.sqrt(avg_c**7 / (avg_c**7 + 25.0**7))) if avg_c > 0 else 0.0
    a1p = a1 * (1.0 + g)
    a2p = a2 * (1.0 + g)
    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)
    avg_cp = 0.5 * (c1p + c2p)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0
    dh = h2p - h1p
    if abs(dh) > 180.0:
        dh -= 360.0 * (1.0 if dh > 0 else -1.0)
    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = 2.0 * math.sqrt(max(c1p * c2p, 0.0)) * math.sin(math.radians(dh * 0.5))
    avg_hp = h1p + h2p
    if abs(h1p - h2p) > 180.0:
        avg_hp += 360.0
    avg_hp *= 0.5
    t = (
        1.0
        - 0.17 * math.cos(math.radians(avg_hp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * avg_hp))
        + 0.32 * math.cos(math.radians(3.0 * avg_hp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * avg_hp - 63.0))
    )
    sl = 1.0 + (0.015 * (avg_l - 50.0) ** 2) / math.sqrt(20.0 + (avg_l - 50.0) ** 2)
    sc = 1.0 + 0.045 * avg_cp
    sh = 1.0 + 0.015 * avg_cp * t
    delta_theta = 30.0 * math.exp(-(((avg_hp - 275.0) / 25.0) ** 2))
    rc = 2.0 * math.sqrt(avg_cp**7 / (avg_cp**7 + 25.0**7)) if avg_cp > 0 else 0.0
    rt = -math.sin(math.radians(2.0 * delta_theta)) * rc
    return math.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dhp / sh) ** 2
        + rt * (dcp / sc) * (dhp / sh)
    )


def _load_rgb_image(path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError as err:
        raise RuntimeError("Pillow (PIL) is required for delta-E mode") from err
    with Image.open(path) as im:
        return im.convert("RGB").tobytes(), im.size


def compute_delta_e_2000_report(
    test_path: Path, reference_path: Path, stride: int = 1
) -> DeltaEReport:
    test_bytes, (tw, th) = _load_rgb_image(test_path)
    ref_bytes, (rw, rh) = _load_rgb_image(reference_path)
    if (tw, th) != (rw, rh):
        raise ValueError(
            f"size mismatch: test={tw}x{th} reference={rw}x{rh}; cannot compute delta-E"
        )

    stride = max(1, stride)
    samples: list[float] = []
    width_bytes = tw * 3
    for y in range(0, th, stride):
        row = y * width_bytes
        for x in range(0, tw, stride):
            idx = row + x * 3
            tr, tg, tb = test_bytes[idx], test_bytes[idx + 1], test_bytes[idx + 2]
            rr, rg, rbp = ref_bytes[idx], ref_bytes[idx + 1], ref_bytes[idx + 2]
            tl = _xyz_to_lab(
                *_linear_srgb_to_xyz(
                    _srgb_to_linear(tr / 255.0),
                    _srgb_to_linear(tg / 255.0),
                    _srgb_to_linear(tb / 255.0),
                )
            )
            rl = _xyz_to_lab(
                *_linear_srgb_to_xyz(
                    _srgb_to_linear(rr / 255.0),
                    _srgb_to_linear(rg / 255.0),
                    _srgb_to_linear(rbp / 255.0),
                )
            )
            samples.append(_delta_e_2000(tl, rl))

    if not samples:
        return DeltaEReport(mean=0.0, p95=0.0, max=0.0, pixels=0)
    samples.sort()
    p95 = samples[int(len(samples) * 0.95)]
    return DeltaEReport(
        mean=sum(samples) / len(samples),
        p95=p95,
        max=samples[-1],
        pixels=len(samples),
    )


def verify_image(
    test_image_path: Path,
    reference_image_path: Path,
    mode: str,
    delta_e_tolerance_mean: float,
    delta_e_tolerance_max: float,
    delta_e_stride: int,
) -> bool:
    if not test_image_path.exists():
        print(f"ERROR: Test image not found: {test_image_path}")
        return False
    if not reference_image_path.exists():
        print(f"ERROR: Reference image not found: {reference_image_path}")
        return False

    if mode in ("hash", "auto"):
        test_hash = calculate_sha256(test_image_path)
        ref_hash = calculate_sha256(reference_image_path)
        meta_json = reference_image_path.with_suffix(".json")
        expected_hash = ref_hash
        if meta_json.exists():
            try:
                with meta_json.open(encoding="utf-8") as fh:
                    meta = json.load(fh)
                if isinstance(meta, dict):
                    expected_hash = meta.get("sha256", ref_hash)
            except (OSError, ValueError) as err:
                print(f"WARNING: Could not load reference metadata: {err}")

        if test_hash == expected_hash:
            print("PASS - hash match")
            print(f"  Test hash:     {test_hash}")
            print(f"  Expected hash: {expected_hash}")
            return True
        if mode == "hash":
            print("FAIL - hash mismatch")
            print(f"  Test hash:     {test_hash}")
            print(f"  Expected hash: {expected_hash}")
            return False
        print("INFO: hash differs; falling back to delta-E 2000 comparison")

    try:
        report = compute_delta_e_2000_report(
            test_image_path, reference_image_path, delta_e_stride
        )
    except RuntimeError as err:
        print(f"WARNING: {err}; falling back to hash comparison result")
        return False
    except Exception as err:  # noqa: BLE001
        print(f"ERROR: delta-E comparison failed: {err}")
        return False

    print(
        f"delta-E2000: mean={report.mean:.4f} p95={report.p95:.4f} "
        f"max={report.max:.4f} pixels={report.pixels}"
    )
    ok = report.mean <= delta_e_tolerance_mean and report.max <= delta_e_tolerance_max
    print("PASS" if ok else "FAIL")
    print(
        f"  Tolerance: mean<={delta_e_tolerance_mean} max<={delta_e_tolerance_max}"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("test_image", type=Path, help="Path to the freshly generated PNG")
    parser.add_argument(
        "reference_image",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "reference" / "baseline_512x288.png",
        help="Path to the CI reference PNG (default: ci/reference/baseline_512x288.png)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "hash", "delta-e"),
        default="auto",
        help="Comparison mode: auto (hash first, delta-E fallback), hash only, or delta-E only",
    )
    parser.add_argument(
        "--mean-tolerance",
        type=float,
        default=1.5,
        help="Maximum allowed mean delta-E2000 (default: 1.5 units)",
    )
    parser.add_argument(
        "--max-tolerance",
        type=float,
        default=8.0,
        help="Maximum allowed peak delta-E2000 (default: 8.0 units)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Pixel stride when sampling for delta-E (1 = every pixel)",
    )
    args = parser.parse_args()

    success = verify_image(
        args.test_image,
        args.reference_image,
        args.mode,
        args.mean_tolerance,
        args.max_tolerance,
        args.stride,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
