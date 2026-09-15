#!/usr/bin/env python3
"""Align decoded normal frames to the silk source-time schedule without rendering.

Normal PNGs must be decoded without scaling to frame_000000.png, etc. This
script selects the nearest existing accumulated checkpoint for each silk frame,
copies or hardlinks its PNG, and carries that checkpoint's exact A/B/C markers.
The desired silk time and the actual selected normal time are both recorded.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import errno
import hashlib
import itertools
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any

BODY_IDS = ["A", "B", "C"]
COORDINATES = "normalized_xy_top_left"
BODY_COLORS = [[248, 113, 113], [251, 191, 36], [34, 211, 238]]


def read_json(path: Path) -> dict[str, Any]:
    """Read an object document, rejecting non-standard NaN/Infinity values."""

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(path.read_text(), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    """Hash a file without loading all of it into memory."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seed_bytes(value: Any) -> bytes:
    """Compare hex seeds by their bytes, retaining the caller's displayed seed."""
    if not isinstance(value, str):
        raise ValueError("Missing source seed")
    result = bytes.fromhex(value.removeprefix("0x").removeprefix("0X"))
    if not result:
        raise ValueError("Empty source seed")
    return result


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Invalid {name}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid {name}: {value}")
    return value


def marker_frames(document: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Check the coordinate convention, identity order, and zero-based schedule."""
    if document.get("body_ids") != BODY_IDS:
        raise ValueError(f"{name} must preserve body order A/B/C (source indices 0/1/2)")
    if document.get("coordinate_system") != COORDINATES:
        raise ValueError(f"{name} has an unsupported coordinate convention")
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{name} has no frame schedule")
    previous = -1.0
    for index, item in enumerate(frames):
        if not isinstance(item, dict) or item.get("frame") != index:
            raise ValueError(f"{name} frames must be consecutive and start at zero")
        time = fraction(item.get("source_fraction"), f"{name} frame {index} source fraction")
        if time < previous:
            raise ValueError(f"{name} source times must not move backward")
        previous = time
        bodies = item.get("bodies")
        if not isinstance(bodies, list) or len(bodies) != 3:
            raise ValueError(f"{name} frame {index} needs exactly three bodies")
        for position in bodies:
            if not isinstance(position, list) or len(position) != 2:
                raise ValueError(f"{name} frame {index} has an invalid position")
            for coordinate in position:
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    or not math.isfinite(coordinate)
                ):
                    raise ValueError(f"{name} frame {index} has a non-finite coordinate")
    return frames


def build_plan(
    normal: dict[str, Any], silk: dict[str, Any], silk_info: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate a shared physical source and choose nearest normal checkpoints."""
    if normal.get("complete") is not True:
        raise ValueError("Normal generation manifest is incomplete")
    source = normal.get("body_markers")
    recipe = silk_info.get("recipe")
    if not isinstance(source, dict) or not isinstance(recipe, dict):
        raise ValueError("Missing normal body markers or silk cache recipe")
    source_frames = marker_frames(source, "normal")
    target_frames = marker_frames(silk, "silk")
    seeds = [normal.get("seed"), source.get("seed"), silk.get("seed"), recipe.get("seed")]
    if len({seed_bytes(value) for value in seeds}) != 1:
        raise ValueError("Normal, silk markers, and silk cache do not share the same seed")
    samples = integer(normal.get("source_sample_count"), "normal source sample count", 2)
    if integer(recipe.get("source_samples"), "silk source sample count", 2) != samples:
        raise ValueError("Normal and silk use different source sample counts")
    source_last = integer(normal.get("source_last_step"), "normal last source index")
    if source_last != samples - 1 or normal.get("source_first_step") != 0:
        raise ValueError("Normal replay does not describe the complete zero-based orbit")
    source_dt = normal.get("source_dt")
    if (
        not isinstance(source_dt, (float, int))
        or not math.isfinite(source_dt)
        or source_dt <= 0
        or source_dt != recipe.get("source_dt")
    ):
        raise ValueError("Normal and silk use different or invalid source timesteps")
    normal_physics = normal.get("source_orbit_provenance", {})
    silk_physics = recipe.get("source_provenance", {})
    physics_keys = [
        "initial_condition_f64_bits",
        "integrator",
        "integration_dt",
        "warmup_steps",
        "recording_steps",
        "sample_stride",
    ]
    for key in physics_keys:
        if normal_physics.get(key) is None or normal_physics[key] != silk_physics.get(key):
            raise ValueError(f"Normal and silk physical provenance differs: {key}")
    if integer(silk_info.get("frames"), "silk cache frames", 1) != len(target_frames):
        raise ValueError("Silk marker count differs from the cloth cache")
    fps = integer(silk.get("fps"), "silk marker frame rate", 1)
    if silk_info.get("fps") != fps:
        raise ValueError("Silk marker frame rate differs from the cloth cache")
    for field, actual in [
        ("first_visible_source_fraction", target_frames[0]["source_fraction"]),
        ("last_visible_source_fraction", target_frames[-1]["source_fraction"]),
    ]:
        if not math.isclose(fraction(recipe.get(field), field), actual, abs_tol=1e-12):
            raise ValueError(f"Silk markers do not match the cloth's {field}")
    if normal.get("frame_count") != len(source_frames):
        raise ValueError("Normal frame count differs from its marker schedule")
    checkpoints = [
        integer(item.get("source_index"), "checkpoint source index") for item in source_frames
    ]
    if checkpoints != normal.get("frame_checkpoint_indices"):
        raise ValueError("Normal marker positions and accumulation checkpoints disagree")
    if any(a >= b for a, b in itertools.pairwise(checkpoints)):
        raise ValueError("Normal checkpoints must be strictly increasing")
    for item, checkpoint in zip(source_frames, checkpoints, strict=True):
        if checkpoint > source_last or not math.isclose(
            item["source_fraction"], checkpoint / source_last, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Normal checkpoint source index and fraction disagree")

    mapping: list[dict[str, Any]] = []
    aligned_frames = []
    for target in target_frames:
        target_index = target["source_fraction"] * source_last
        insertion = bisect.bisect_left(checkpoints, target_index)
        candidates = [
            index for index in [insertion - 1, insertion] if 0 <= index < len(checkpoints)
        ]
        selected = min(
            candidates, key=lambda index: (abs(checkpoints[index] - target_index), index)
        )
        original = source_frames[selected]
        frame = target["frame"]
        error = checkpoints[selected] - target_index
        mapping.append(
            {
                "frame": frame,
                "original_normal_frame": selected,
                "filename": f"frame_{frame:06d}.png",
                "original_filename": f"frame_{selected:06d}.png",
                "target_source_fraction": target["source_fraction"],
                "target_source_index": target_index,
                "actual_source_fraction": original["source_fraction"],
                "actual_source_index": checkpoints[selected],
                "source_index_error": error,
                "source_time_error_seconds": error * source_dt,
            }
        )
        aligned = copy.deepcopy(original)
        aligned.update(
            {
                "frame": frame,
                "original_normal_frame": selected,
                "target_source_fraction": target["source_fraction"],
                "source_index_error": error,
            }
        )
        aligned_frames.append(aligned)
    absolute = [abs(item["source_index_error"]) for item in mapping]
    summary = {
        "source_frame_count": len(source_frames),
        "aligned_frame_count": len(target_frames),
        "fps": fps,
        "encoded_duration_seconds": len(target_frames) / fps,
        "source_sample_count": samples,
        "source_dt": source_dt,
        "max_abs_source_index_error": max(absolute),
        "mean_abs_source_index_error": sum(absolute) / len(absolute),
        "max_abs_source_fraction_error": max(absolute) / source_last,
        "max_abs_source_time_error_seconds": max(absolute) * source_dt,
        "first": mapping[0],
        "last": mapping[-1],
    }
    aligned_markers = {
        key: copy.deepcopy(value) for key, value in source.items() if key != "frames"
    }
    aligned_markers.update(
        {
            "fps": fps,
            "frames": aligned_frames,
            "body_colors": BODY_COLORS,
            "connectors": False,
            "alignment": summary,
            "projection": normal.get("projection"),
            "checkpoint_semantics": normal.get("checkpoint_semantics"),
        }
    )
    event = aligned_markers.get("closest_approach")
    if isinstance(event, dict) and isinstance(event.get("source_fraction"), (int, float)):
        closest = min(
            mapping,
            key=lambda item: (
                abs(item["target_source_fraction"] - event["source_fraction"]),
                item["frame"],
            ),
        )
        event.update(
            {
                "aligned_frame": closest["frame"],
                "aligned_video_seconds": closest["frame"] / fps,
                "aligned_checkpoint_source_index": closest["actual_source_index"],
            }
        )
    return aligned_markers, mapping, summary


def png_dimensions(path: Path) -> tuple[int, int]:
    """Inspect PNG dimensions only; no image decoding or rendering is performed."""
    with path.open("rb") as image:
        header = image.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG frame: {path}")
    return struct.unpack(">II", header[16:24])


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace metadata only after all selected PNGs are ready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    normal_path = args.normal_json.resolve()
    silk_path = args.silk_markers.resolve()
    info_path = args.silk_info.resolve()
    normal = read_json(normal_path)
    silk = read_json(silk_path)
    info = read_json(info_path)
    markers, mapping, summary = build_plan(normal, silk, info)
    decoded = args.decoded_dir.resolve()
    output = args.output_dir.resolve()
    marker_path = args.markers_output.resolve()
    provenance_path = (
        args.provenance_output.resolve()
        if args.provenance_output
        else marker_path.with_name(marker_path.stem + ".provenance.json")
    )
    if decoded == output:
        raise ValueError("Aligned output must differ from the original decoded frame directory")
    if len({normal_path, silk_path, info_path, marker_path, provenance_path}) != 5:
        raise ValueError("Input and output JSON paths must all be distinct")
    expected_dimensions = (
        integer(markers.get("width"), "normal marker width", 1),
        integer(markers.get("height"), "normal marker height", 1),
    )
    # Preflight all selected files before creating any output. Existing files may
    # be reused only if they are byte-identical to this exact selected source.
    hashes: dict[int, str] = {}
    for item in mapping:
        source = decoded / item["original_filename"]
        destination = output / item["filename"]
        if png_dimensions(source) != expected_dimensions:
            raise ValueError(f"Decoded normal frame dimensions changed: {source}")
        source_frame = item["original_normal_frame"]
        if source_frame not in hashes:
            hashes[source_frame] = digest(source)
        item["png_sha256"] = hashes[source_frame]
        if destination.exists() and digest(destination) != item["png_sha256"]:
            raise ValueError(f"Refusing to replace a different aligned frame: {destination}")
    if args.dry_run:
        return {"dry_run": True, **summary}
    output.mkdir(parents=True, exist_ok=True)
    for item in mapping:
        source = decoded / item["original_filename"]
        destination = output / item["filename"]
        if not destination.exists():
            try:
                if args.link_mode == "copy":
                    shutil.copy2(source, destination)
                else:
                    os.link(source, destination)
            except OSError as error:
                if args.link_mode != "auto" or error.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    errno.ENOSYS,
                }:
                    raise
                shutil.copy2(source, destination)
        item["transfer"] = "hardlink" if source.samefile(destination) else "copy"
    provenance = {
        "schema_version": 1,
        "kind": "normal-nearest-checkpoint-alignment",
        "algorithm": "nearest source index; equal-distance ties choose the earlier normal frame",
        "script_sha256": digest(Path(__file__)),
        "inputs": {
            "normal_json": {"path": str(normal_path), "sha256": digest(normal_path)},
            "silk_markers": {"path": str(silk_path), "sha256": digest(silk_path)},
            "silk_cache_info": {"path": str(info_path), "sha256": digest(info_path)},
            "decoded_normal_directory": str(decoded),
        },
        "outputs": {"directory": str(output), "markers": str(marker_path)},
        "identity": {
            "seed": normal["seed"],
            "body_ids": BODY_IDS,
            "body_index_mapping": {"A": 0, "B": 1, "C": 2},
            "source_samples_sha256": normal.get("source_samples_sha256"),
            "silk_bake_sha256": info.get("sha256"),
            "silk_orbit_cache_sha256": info["recipe"].get("orbit_cache_sha256"),
            "validated": [
                "seed",
                "source_sample_count",
                "source_dt",
                "initial_condition_f64_bits",
                "integrator",
                "integration_dt",
                "warmup_steps",
                "recording_steps",
                "sample_stride",
                "body_order",
                "silk_frame_count",
                "silk_fps",
                "silk_visible_endpoints",
            ],
        },
        "normal_projection": normal.get("projection"),
        "silk_projection": silk.get("projection"),
        "note": "Each aligned image and its markers use the actual selected normal checkpoint; "
        "target_source_fraction describes the silk frame. The normal renderer may not "
        "have an exact source-zero frame. Hardlinked originals and aligned frames must "
        "remain read-only; write Rust marker overlays to a separate directory.",
        "summary": summary,
        "mapping": mapping,
    }
    markers["alignment_provenance"] = str(provenance_path)
    write_json(provenance_path, provenance)
    write_json(marker_path, markers)
    return {"markers": str(marker_path), "provenance": str(provenance_path), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-json", type=Path, required=True)
    parser.add_argument("--silk-markers", type=Path, required=True)
    parser.add_argument(
        "--silk-info",
        type=Path,
        required=True,
        help="Matching .silk.json cache sidecar, for physical-source validation",
    )
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markers-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "copy"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args), indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Alignment failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
