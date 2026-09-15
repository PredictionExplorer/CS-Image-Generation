#!/usr/bin/env python3
"""Verify complete artwork films against their archived full-source render recipe."""

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def digest(path: Path) -> str:
    """Hash media without keeping another full copy in memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(frames: Path, movies: list[Path], output: Path) -> None:
    """Require every source frame and decode every movie with errors treated as fatal."""
    manifest = json.loads((frames / "render.json").read_text())
    config = manifest["config"]
    count, fps = config["frames"], config["fps"]
    if not manifest["complete"] or manifest["rendered_frames"] != list(range(count)):
        raise ValueError("This is not a completed film covering the full source interval")
    endpoints = []
    for index in [0, count - 1]:
        path = frames / f"frame_{index:06d}.png"
        receipt = json.loads(path.with_suffix(".png.json").read_text())
        if receipt["png_sha256"] != digest(path):
            raise ValueError(f"Endpoint image failed its integrity check: {path}")
        endpoints.append(receipt["source_fraction"])
    if endpoints != [0.0, 1.0]:
        raise ValueError("Film does not reach both source endpoints")
    verified = []
    for movie in movies:
        sidecar = json.loads(movie.with_suffix(movie.suffix + ".json").read_text())
        actual_hash = digest(movie)
        if actual_hash != sidecar["sha256"] or sidecar["source"] != manifest:
            raise ValueError(f"Movie provenance does not match: {movie}")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt,duration",
                "-of",
                "json",
                str(movie),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if (
            stream["width"] != config["render"]["width"]
            or stream["height"] != config["render"]["height"]
            or int(stream["nb_frames"]) != count
            or Fraction(stream["r_frame_rate"]) != fps
            or abs(float(stream["duration"]) - count / fps) > 1e-5
        ):
            raise ValueError(f"Movie timing or dimensions differ: {movie}")
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "8",
                "-i",
                str(movie),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stderr:
            raise ValueError(f"Movie decode reported errors: {result.stderr}")
        verified.append(
            {
                "file": movie.name,
                "sha256": actual_hash,
                "bytes": movie.stat().st_size,
                "stream": stream,
                "all_frames_decoded_without_errors": True,
            }
        )
    record = {
        "complete": True,
        "seed": manifest["seed"],
        "kind": config["kind"],
        "source_endpoints": endpoints,
        "orbit_sha256": manifest["orbit_sha256"],
        "render_manifest_sha256": digest(frames / "render.json"),
        "movies": verified,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--movie", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.frames, args.movie, args.output)
