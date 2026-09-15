#!/usr/bin/env python3
"""Compose two already aligned PNG sequences into a labeled comparison film.

Only Python's standard library and the existing FFmpeg installation are used.
The inputs must contain frame_000000.png through frame_{frame_count-1:06d}.png.
This tool does not add guides, retime motion, or modify either input sequence.

Example:
    python3 tools/tidal_silk/compose_comparison.py \
        --normal normal-guided --silk silk-guided --output comparison.mp4 \
        --frame-count 901 --fps 30 --seed 0xb7f327f9f722 \
        --closest-approach 27.823857823857827

Optional cue metadata schema:
    {"closest_approach": {"source_fraction": 0.9274619274619275,
                          "bodies": ["A", "C"], "half_window_seconds": 0.65}}
Use time_seconds instead of source_fraction when a playback time is known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

WIDTH = 1920
HEIGHT = 800
PANEL_WIDTH = 960
PANEL_HEIGHT = 621
HEADER_HEIGHT = 80
FOOTER_HEIGHT = 99
BACKGROUND = "080B12"
FOOTER_BACKGROUND = "0B101A"
DIVIDER = "283244"
TEXT = "E8EDF5"
MUTED = "A6B2C6"
BODY_COLORS = {"A": "F87171", "B": "FBBF24", "C": "22D3EE"}
SCALE_FLAGS = "bilinear+accurate_rnd+bitexact"
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
EXPLANATION = "The same three bodies, shown through two different visual interpretations."


@dataclass(frozen=True)
class Cue:
    """A physical event displayed at a known comparison playback time."""

    time_seconds: float
    bodies: tuple[str, str]
    half_window_seconds: float

    @property
    def text(self) -> str:
        return f"Closest physical approach - {self.bodies[0]} + {self.bodies[1]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_header(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        header = stream.read(29)
    if len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0:
        raise ValueError(f"Empty PNG dimensions: {path}")
    return {"width": width, "height": height, "bit_depth": header[24], "color_type": header[25]}


def inspect_sequence(directory: Path, count: int, hash_contents: bool) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValueError(f"Frame directory does not exist: {directory}")
    dimensions = None
    digest = hashlib.sha256()
    total_bytes = 0
    for frame in range(count):
        path = directory / f"frame_{frame:06d}.png"
        if not path.is_file():
            raise ValueError(f"Missing input frame: {path}")
        info = png_header(path)
        if dimensions is None:
            dimensions = info
        elif info != dimensions:
            raise ValueError(f"PNG dimensions or format changed within the sequence: {path}")
        total_bytes += path.stat().st_size
        if hash_contents:
            digest.update(path.name.encode("utf-8") + b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "directory": str(directory),
        "frame_pattern": "frame_%06d.png",
        "frame_count": count,
        "image": dimensions,
        "total_bytes": total_bytes,
        "sequence_sha256": digest.hexdigest() if hash_contents else None,
        "digest_definition": "SHA256 of ordered filename/NUL/SHA256(file) records",
    }


def load_cue(args: argparse.Namespace, fps: Fraction) -> Cue | None:
    if args.closest_approach is not None and args.cue_metadata is not None:
        raise ValueError("Use either --closest-approach or --cue-metadata")
    time_seconds = args.closest_approach
    bodies = tuple(args.cue_pair)
    half_window = args.cue_half_window
    if args.cue_metadata is not None:
        data = json.loads(args.cue_metadata.read_text(encoding="utf-8"))["closest_approach"]
        if ("time_seconds" in data) == ("source_fraction" in data):
            raise ValueError("Cue metadata needs exactly one of time_seconds/source_fraction")
        if "source_fraction" in data:
            fraction = float(data["source_fraction"])
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("Cue source_fraction must be between 0 and 1")
            time_seconds = fraction * float(Fraction(args.frame_count - 1, 1) / fps)
        else:
            time_seconds = float(data["time_seconds"])
        bodies = tuple(data.get("bodies", data.get("body_ids", bodies)))
        half_window = float(data.get("half_window_seconds", half_window))
    if time_seconds is None:
        return None
    duration = float(Fraction(args.frame_count, 1) / fps)
    if not math.isfinite(time_seconds) or not 0.0 <= time_seconds <= duration:
        raise ValueError("Closest-approach time must lie within the composed film")
    if not math.isfinite(half_window) or half_window <= 0.0:
        raise ValueError("Cue half-window must be positive and finite")
    if len(bodies) != 2 or len(set(bodies)) != 2 or any(body not in BODY_COLORS for body in bodies):
        raise ValueError("Cue bodies must be two different labels from A, B, C")
    return Cue(float(time_seconds), (bodies[0], bodies[1]), half_window)


def text_filter(
    filename: str,
    x: int | str,
    y: int,
    size: int,
    color: str = TEXT,
    *,
    bold: bool = False,
    enable: str | None = None,
) -> str:
    # The filenames are fixed ASCII names in the FFmpeg working directory.
    # User text and font paths never enter filter-expression syntax.
    font = "font-bold.ttf" if bold else "font.ttf"
    result = (
        f"drawtext=fontfile={font}:textfile={filename}:expansion=none:"
        f"x={x}:y={y}:fontsize={size}:fontcolor=0x{color}:fix_bounds=1"
    )
    if enable is not None:
        result += f":enable='{enable}'"
    return result


def build_graph(cue: Cue | None, duration: float) -> str:
    panel = (
        f"setpts=PTS-STARTPTS,scale={PANEL_WIDTH}:{PANEL_HEIGHT}:"
        f"force_original_aspect_ratio=decrease:flags={SCALE_FLAGS}:out_range=full,"
        f"format=rgb24,pad={PANEL_WIDTH}:{PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:"
        f"color=0x{BACKGROUND},setsar=1"
    )
    overlays = [
        f"drawbox=x=0:y=701:w=1920:h=99:color=0x{FOOTER_BACKGROUND}:t=fill",
        f"drawbox=x=0:y=79:w=1920:h=1:color=0x{DIVIDER}:t=fill",
        f"drawbox=x=0:y=701:w=1920:h=1:color=0x{DIVIDER}:t=fill",
        f"drawbox=x=959:y=80:w=2:h=621:color=0x{DIVIDER}:t=fill",
        text_filter("normal-title.txt", 32, 12, 30, bold=True),
        text_filter("silk-title.txt", 992, 12, 30, bold=True),
        text_filter("normal-subtitle.txt", 33, 51, 16, MUTED),
        text_filter("silk-subtitle.txt", 993, 51, 16, MUTED),
        text_filter("seed.txt", 32, 716, 17),
        text_filter("legend-title.txt", 1450, 718, 14, MUTED),
        text_filter("explanation.txt", 32, 751, 17, MUTED),
    ]
    for body, x in [("A", 1550), ("B", 1660), ("C", 1770)]:
        overlays.extend(
            [
                f"drawbox=x={x}:y=722:w=8:h=8:color=0x{BODY_COLORS[body]}:t=fill",
                text_filter(f"body-{body}.txt", x + 18, 715, 17, BODY_COLORS[body], bold=True),
            ]
        )
    if cue is not None:
        start = max(0.0, cue.time_seconds - cue.half_window_seconds)
        end = min(duration, cue.time_seconds + cue.half_window_seconds)
        overlays.append(
            text_filter(
                "cue.txt",
                "w-tw-32",
                751,
                17,
                "FBBF24",
                bold=True,
                enable=f"between(t,{start:.9f},{end:.9f})",
            )
        )
    return (
        f"[0:v]{panel}[normal];[1:v]{panel}[silk];"
        f"[normal][silk]hstack=inputs=2:shortest=1,"
        f"pad={WIDTH}:{HEIGHT}:0:{HEADER_HEIGHT}:color=0x{BACKGROUND},"
        + ",".join(overlays)
        + f",scale=iw:ih:in_range=full:out_range=tv:out_color_matrix=bt709:"
        f"flags={SCALE_FLAGS},format=yuv420p[comparison]"
    )


def stage_labels(
    directory: Path, seed: str, cue: Cue | None, font: Path, title_font: Path, explanation: str
) -> None:
    labels = {
        "normal-title.txt": "Normal",
        "silk-title.txt": "Tidal Silk",
        "normal-subtitle.txt": "Accumulated light",
        "silk-subtitle.txt": "Moving fabric",
        "seed.txt": f"Seed  {seed}",
        "legend-title.txt": "Body key",
        "explanation.txt": explanation,
        **{f"body-{body}.txt": body for body in BODY_COLORS},
    }
    if cue is not None:
        labels["cue.txt"] = cue.text
    for filename, value in labels.items():
        (directory / filename).write_text(value, encoding="utf-8")
    shutil.copyfile(font, directory / "font.ttf")
    shutil.copyfile(title_font, directory / "font-bold.ttf")


def command_line(
    args: argparse.Namespace, fps: Fraction, graph: str, temporary_output: Path
) -> list[str]:
    rate = f"{fps.numerator}/{fps.denominator}"
    command = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-filter_complex_threads",
        "1",
        "-sws_dither",
        "none",
    ]
    for directory in [args.normal, args.silk]:
        pattern = str(directory).replace("%", "%%") + "/frame_%06d.png"
        command.extend(["-framerate", rate, "-start_number", "0", "-i", pattern])
    command.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[comparison]",
            "-frames:v",
            str(args.frame_count),
            "-r",
            rate,
            "-fps_mode",
            "cfr",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-threads",
            str(args.threads),
            "-x264-params",
            "cpu-independent=1",
            "-sws_dither",
            "none",
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-colorspace",
            "bt709",
            "-color_range",
            "tv",
            "-flags",
            "+bitexact",
            "-fflags",
            "+bitexact",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary_output),
        ]
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--normal", type=Path, required=True, help="Aligned Normal PNG sequence")
    parser.add_argument("--silk", type=Path, required=True, help="Aligned Tidal Silk PNG sequence")
    parser.add_argument("--output", type=Path, required=True, help="Destination H.264 .mp4 file")
    parser.add_argument("--frame-count", type=int, default=901)
    parser.add_argument(
        "--fps", default="30", help="Integer or rational frame rate, e.g. 30 or 30000/1001"
    )
    parser.add_argument("--seed", default="0xb7f327f9f722")
    parser.add_argument(
        "--explanation", default=EXPLANATION, help="Footer explanation of the two views"
    )
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument(
        "--title-font", type=Path, help="Optional bold title font; defaults to a sibling bold face"
    )
    parser.add_argument(
        "--closest-approach", type=float, help="Optional physical-approach playback time in seconds"
    )
    parser.add_argument("--cue-pair", nargs=2, choices=list(BODY_COLORS), default=["A", "C"])
    parser.add_argument("--cue-half-window", type=float, default=0.65)
    parser.add_argument(
        "--cue-metadata", type=Path, help="JSON containing a closest_approach object"
    )
    parser.add_argument("--metadata-output", type=Path, help="Defaults to OUTPUT.mp4.json")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--preset", choices=["medium", "slow", "slower"], default="slow")
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate headers and print the composition plan"
    )
    return parser.parse_args()


def compose(args: argparse.Namespace) -> dict[str, Any]:
    fps = Fraction(args.fps)
    if fps <= 0 or fps > 240 or args.frame_count < 1 or args.threads < 1 or not 0 <= args.crf <= 51:
        raise ValueError("Invalid frame rate, frame count, thread count, or CRF")
    if not re.fullmatch(r"0x[0-9a-fA-F]{1,64}", args.seed):
        raise ValueError("Seed must be 0x followed by 1-64 hexadecimal digits")
    for name in ["normal", "silk", "output", "font"]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise ValueError(f"Output already exists; use --overwrite to replace it: {args.output}")
    if not args.font.is_file():
        raise ValueError(f"Font not found; supply --font with an existing font file: {args.font}")
    title_font = (
        args.title_font.expanduser().resolve()
        if args.title_font
        else args.font.with_stem(args.font.stem + "-Bold")
    )
    if not title_font.is_file():
        if args.title_font:
            raise ValueError(f"Title font not found: {title_font}")
        title_font = args.font
    cue = load_cue(args, fps)
    duration = float(Fraction(args.frame_count, 1) / fps)
    sources = {
        "normal": inspect_sequence(args.normal, args.frame_count, not args.dry_run),
        "silk": inspect_sequence(args.silk, args.frame_count, not args.dry_run),
    }
    graph = build_graph(cue, duration)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tidal-silk-comparison-v1",
        "output": str(args.output),
        "seed": args.seed,
        "frame_count": args.frame_count,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "expected_duration_seconds": duration,
        "sources": sources,
        "layout": {
            "width": WIDTH,
            "height": HEIGHT,
            "panel_width": PANEL_WIDTH,
            "panel_height": PANEL_HEIGHT,
            "header_height": HEADER_HEIGHT,
            "footer_height": FOOTER_HEIGHT,
        },
        "labels": {
            "normal": ["Normal", "Accumulated light"],
            "silk": ["Tidal Silk", "Moving fabric"],
            "explanation": args.explanation,
        },
        "body_colors": BODY_COLORS,
        "closest_physical_approach": asdict(cue) if cue else None,
        "font": {"path": str(args.font), "sha256": sha256_file(args.font)},
        "title_font": {"path": str(title_font), "sha256": sha256_file(title_font)},
        "encoder": {
            "codec": "libx264",
            "preset": args.preset,
            "crf": args.crf,
            "threads": args.threads,
            "cpu_independent": True,
            "pixel_format": "yuv420p",
            "scale_flags": SCALE_FLAGS,
            "dither": "none",
            "color_matrix": "bt709",
            "transfer": "iec61966-2-1",
            "range": "tv",
        },
        "filter_graph": graph,
        "filter_graph_sha256": hashlib.sha256(graph.encode("utf-8")).hexdigest(),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        return manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else args.output.with_suffix(args.output.suffix + ".json")
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tidal-comparison-", dir=args.output.parent) as working:
        directory = Path(working)
        stage_labels(directory, args.seed, cue, args.font, title_font, args.explanation)
        temporary_output = directory / "comparison.mp4"
        print(
            f"Composing {args.frame_count} frames at {fps} fps into {args.output}",
            file=sys.stderr,
            flush=True,
        )
        command = command_line(args, fps, graph, temporary_output)
        subprocess.run(command, cwd=directory, check=True)
        probe = json.loads(
            subprocess.run(
                [
                    args.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt,duration",
                    "-of",
                    "json",
                    str(temporary_output),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )["streams"][0]
        if (
            probe["width"] != WIDTH
            or probe["height"] != HEIGHT
            or int(probe["nb_frames"]) != args.frame_count
            or Fraction(probe["r_frame_rate"]) != fps
            or probe["codec_name"] != "h264"
            or probe["pix_fmt"] != "yuv420p"
        ):
            raise ValueError(f"Encoded output does not match the requested video: {probe}")
        manifest["verified_output"] = probe
        version = subprocess.run(
            [args.ffmpeg, "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        manifest["encoder"]["ffmpeg_version"] = version
        manifest["output_sha256"] = sha256_file(temporary_output)
        manifest["output_bytes"] = temporary_output.stat().st_size
        temporary_output.replace(args.output)
        staged_metadata = directory / "manifest.json"
        staged_metadata.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Metadata can live on a different filesystem from the video.
        metadata_temporary = metadata.with_name(metadata.name + ".partial")
        shutil.copyfile(staged_metadata, metadata_temporary)
        metadata_temporary.replace(metadata)
    print(f"Video: {args.output}\nMetadata: {metadata}", file=sys.stderr)
    return manifest


def main() -> int:
    try:
        args = parse_args()
        manifest = compose(args)
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        ZeroDivisionError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Comparison composition failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
