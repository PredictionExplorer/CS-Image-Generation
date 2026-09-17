#!/usr/bin/env python3
"""Render and archive one complete recorded trajectory as an Estuary painting.

Use --still-only for a finished painting. Full films include both source endpoints,
with canonical equal-time simulation steps. Resume verifies input, code, GPU and
artifact identities; partial films recover only from verified state checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time
import zlib
from fractions import Fraction
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.optics import linear_to_srgb

HERE = Path(__file__).resolve().parent
RUNTIME_FILES = (
    "__init__.py",
    "source.py",
    "recipe.py",
    "optics.py",
    "engine.py",
    "run.py",
    "optics.glsl",
    "requirements.txt",
    "shaders/flow.glsl",
    "shaders/advect.glsl",
    "shaders/correct.glsl",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    encoded(value)
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encoded(value))
    partial.replace(path)


def commit_file(partial, path):
    """An already archived render can be reused, but never silently replaced."""
    if path.exists():
        if digest(path) != digest(partial):
            raise ValueError(f"Existing artifact differs; both files preserved: {path}, {partial}")
        partial.unlink()
    else:
        partial.replace(path)


def write_png(path, linear, depth=16):
    """Write tagged RGB8/RGB16 PNG with top-down rows and no lossy intermediate."""
    rgb = np.asarray(linear)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or depth not in (8, 16):
        raise ValueError("PNG requires a nonempty HxWx3 array and depth 8 or 16")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError("Linear RGB must be finite and in [0, 1]")
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    with partial.open("wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")

        def chunk(kind, data):
            stream.write(struct.pack(">I", len(data)) + kind + data)
            stream.write(struct.pack(">I", zlib.crc32(kind + data)))

        chunk(b"IHDR", struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], depth, 2, 0, 0, 0))
        chunk(b"sRGB", b"\x00")
        chunk(b"gAMA", struct.pack(">I", 45455))
        compressor = zlib.compressobj(level=6)
        for row in rgb:
            values = np.rint(linear_to_srgb(row) * ((1 << depth) - 1))
            data = compressor.compress(
                b"\x00" + values.astype(">u2" if depth == 16 else "u1").tobytes()
            )
            if data:
                chunk(b"IDAT", data)
        chunk(b"IDAT", compressor.flush())
        chunk(b"IEND", b"")
    commit_file(partial, path)


def write_array(path, array):
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    with partial.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    commit_file(partial, path)


def artifact(path, output):
    return {
        "path": str(path.relative_to(output)),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def checked_artifact(output, record):
    name = Path(record["path"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Artifact paths must remain inside the archive")
    path = output / name
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or digest(path) != record["sha256"]
    ):
        raise ValueError(f"Archived artifact changed or missing: {path}")
    return path


def frame_plan(steps, frames):
    if type(steps) is not int or type(frames) is not int or frames < 2 or steps < frames - 1:
        raise ValueError("Film requires at least two frames and one step between frames")
    if steps % (frames - 1):
        raise ValueError("simulation.steps must be divisible by render.frames - 1 for exact timing")
    return [index * steps // (frames - 1) for index in range(frames)]


def exposure_plan(endpoints, samples):
    """Sample only trailing canonical states, ending each exposure at its frame time."""
    if type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("Temporal samples must be an integer in [1, 8]")
    if len(endpoints) < 2 or endpoints[0] != 0 or any(type(step) is not int for step in endpoints):
        raise ValueError("Exposure endpoints require canonical integer steps starting at zero")
    if any(last - first < samples for first, last in pairwise(endpoints)):
        raise ValueError("Temporal samples must not exceed the canonical frame stride")
    return [[0], *[list(range(endpoint - samples + 1, endpoint + 1)) for endpoint in endpoints[1:]]]


def render_exposure(engine, points, width, height):
    """Uniformly average scene-linear RGB; the simulation finishes at the endpoint."""
    if not points:
        raise ValueError("An exposure requires at least one canonical state")
    accumulated = np.zeros((height, width, 3), dtype=np.float64)
    for step in points:
        engine.advance_to(step)
        accumulated += engine.render(width, height)
    accumulated /= len(points)
    return accumulated.astype(np.float32)


def completed(output, identity):
    path = output / "receipt.json"
    if not path.exists():
        return False
    receipt = read_json(path)
    if receipt.get("identity_sha256") != identity:
        raise ValueError("Archive belongs to a different render identity")
    if receipt.get("complete") is not True:
        return False
    records = receipt.get("artifacts", [])
    required = {"poster.png", "linear.npy", "final-state.npy", "recipe.json", "inputs/source.orbit"}
    if receipt.get("mode") == "film":
        required |= {"film.mp4", "movie.json"}
    if not required <= {item["path"] for item in records}:
        raise ValueError("Completed archive lacks required artifacts")
    for record in records:
        checked_artifact(output, record)
    return True


def code_identity():
    """Fingerprint the rendering runtime, independently of gallery and test code."""
    for name in RUNTIME_FILES:
        if not (HERE / name).is_file():
            raise ValueError(f"Required renderer runtime file is missing: {name}")
    names = set(RUNTIME_FILES) | {
        str(path.relative_to(HERE)) for path in (HERE / "shaders").glob("*.glsl") if path.is_file()
    }
    return {name: digest(HERE / name) for name in sorted(names)}


def stop_child(child):
    """Stop only the process group created for this owned encoder/probe."""

    def terminate_group(signum):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signum)

    terminate_group(signal.SIGTERM)
    try:
        child.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader can exit before a descendant. Kill any remaining members
        # of this dedicated group as well, then reap the direct child.
        terminate_group(signal.SIGKILL)
        child.communicate()


def run_child(args, stream=None):
    """Run an owned process, cleaning up encoders even on KeyboardInterrupt."""
    command = [str(value) for value in args]
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=stream if stream is not None else subprocess.PIPE,
        stderr=subprocess.STDOUT if stream is not None else subprocess.PIPE,
        text=stream is None,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate()
    except BaseException:
        stop_child(child)
        raise
    if child.returncode:
        stop_child(child)
        raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def command(args, log):
    with log.open("ab") as stream:
        stream.write(("\n" + json.dumps([str(x) for x in args]) + "\n").encode())
        stream.flush()
        run_child(args, stream)


def encode_movie(output, recipe, ffmpeg, ffprobe):
    frames, fps = recipe["render"]["frames"], recipe["render"]["fps"]
    dimensions = recipe["render"]["resolution"]
    partial = output / "film.partial.mp4"
    args = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-threads",
        "4",
        "-framerate",
        str(fps),
        "-i",
        output / "frames" / "%06d.png",
        "-frames:v",
        str(frames),
        "-an",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-filter_threads",
        "1",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-threads",
        "4",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        partial,
    ]
    command(args, output / "encode.log")
    probe = run_child(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(partial),
        ],
    )
    info = json.loads(probe.stdout)["streams"][0]
    if (
        [info["width"], info["height"]] != dimensions
        or int(info["nb_read_frames"]) != frames
        or Fraction(info["avg_frame_rate"]) != fps
    ):
        raise ValueError("Encoded movie has incorrect dimensions, frame count or cadence")
    decode = run_child(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "4",
            "-i",
            str(partial),
            "-progress",
            "pipe:1",
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
    )
    counts = [int(line[6:]) for line in decode.stdout.splitlines() if line.startswith("frame=")]
    if not counts or counts[-1] != frames or "progress=end" not in decode.stdout:
        raise ValueError("Movie did not completely decode")
    commit_file(partial, output / "film.mp4")
    result = {
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / fps,
        "resolution": dimensions,
        "full_decode_verified": True,
        "codec": "h264",
        "crf": 18,
        "color": "sRGB transfer; BT.709 primaries and YCbCr matrix; limited range",
        "artifact": artifact(output / "film.mp4", output),
    }
    write_json(output / "movie.json", result)
    return result


def checkpoint(output, identity, engine, index, step, records):
    folder = output / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = folder / f"state-{index:06d}.npy"
    write_array(path, engine.read_state())
    write_json(
        output / "checkpoint.json",
        {
            "identity_sha256": identity,
            "frame": index,
            "step": step,
            "internal_steps": engine.internal_steps,
            "maximum_courant": engine.maximum_courant,
            "state": artifact(path, output),
            "frames": records,
        },
    )


def restore_checkpoint(output, identity, engine, plan):
    path = output / "checkpoint.json"
    if not path.exists():
        raise ValueError(
            "Incomplete archive has no checkpoint; preserved, use a new output directory"
        )
    record = read_json(path)
    index = record["frame"]
    if (
        record.get("identity_sha256") != identity
        or type(index) is not int
        or not 0 <= index < len(plan)
        or record["step"] != plan[index]
        or len(record["frames"]) != index + 1
    ):
        raise ValueError("Checkpoint identity or timing differs")
    for index, frame in enumerate(record["frames"]):
        if frame["path"] != f"frames/{index:06d}.png":
            raise ValueError("Checkpoint frame order differs")
        checked_artifact(output, frame)
    state = np.load(checked_artifact(output, record["state"]), allow_pickle=False)
    if "internal_steps" not in record or "maximum_courant" not in record:
        raise ValueError("Checkpoint lacks cumulative transport diagnostics; archive preserved")
    engine.restore(
        state,
        record["step"],
        internal_steps=record["internal_steps"],
        maximum_courant=record["maximum_courant"],
    )
    return record["frame"] + 1, record["frames"]


def run(args):
    from tools.estuary.engine import Engine
    from tools.estuary.recipe import read_recipe
    from tools.estuary.source import Source

    started, code = time.monotonic(), code_identity()
    recipe = read_recipe(args.recipe)
    grid = recipe["simulation"]["resolution"]
    source = Source.read(args.orbit, aspect=grid[0] / grid[1], **recipe["projection"])
    steps = recipe["simulation"]["steps"]
    plan = [] if args.still_only else frame_plan(steps, recipe["render"]["frames"])
    engine = Engine(source, recipe, backend=args.backend)
    try:
        _render_archive(args, engine, source, recipe, plan, code, started)
    finally:
        engine.close()


def _render_archive(args, engine, source, recipe, plan, code, started):
    width, height = recipe["render"]["resolution"]
    steps = recipe["simulation"]["steps"]
    exposures = [] if args.still_only else exposure_plan(plan, recipe["render"]["temporal_samples"])
    tools = {}
    if not args.still_only:
        for name in ("ffmpeg", "ffprobe"):
            selected = getattr(args, name) or shutil.which(name)
            if not selected:
                raise ValueError(f"Missing {name}; supply an explicit --{name} path")
            path = Path(selected).resolve(strict=True)
            tools[name] = {"path": str(path), "sha256": digest(path)}
    request = {
        "schema_version": 1,
        "mode": "still" if args.still_only else "film",
        "source": source.metadata,
        "recipe": recipe,
        "hardware": engine.metadata,
        "backend": args.backend,
        "code": code,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "glcontext": version("glcontext"),
        },
        "tools": tools,
        "frame_steps": plan,
        "exposure_steps": exposures,
        "temporal_exposure": {
            "sampling": "discrete trailing canonical states",
            "weights": "uniform within each exposure",
            "working_space": "scene-linear sRGB; float64 accumulation",
            "first_frame": "single initial state",
            "poster": "sharp final state",
        },
    }
    identity = hashlib.sha256(encoded(request)).hexdigest()
    output = args.output.resolve()
    existing = output.exists()
    if existing and (not args.resume or not output.is_dir()):
        raise ValueError("Output exists; use --resume with identical inputs, or a new directory")
    if args.resume and not existing:
        raise ValueError("--resume requires an existing archive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".render.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if existing:
            if (
                not (output / "request.json").exists()
                or read_json(output / "request.json") != request
            ):
                raise ValueError("Archive input, code, recipe, or hardware identity differs")
            if completed(output, identity):
                print(json.dumps({"status": "reused", "output": str(output)}))
                return
        else:
            write_json(output / "request.json", request)
            write_json(output / "recipe.json", recipe)
            inputs = output / "inputs"
            inputs.mkdir()
            shutil.copyfile(source.path, inputs / "source.orbit")
            for name in code:
                target = inputs / "code" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(HERE / name, target)
        receipt = {
            "schema_version": 1,
            "identity_sha256": identity,
            "complete": False,
            "mode": request["mode"],
            "seed": source.seed,
            "source": source.metadata,
            "hardware": engine.metadata,
            "resolution": [width, height],
        }
        write_json(output / "receipt.json", receipt)
        try:
            records, first = [], 0
            if existing:
                if args.still_only:
                    raise ValueError("Incomplete still preserved; use a new output directory")
                first, records = restore_checkpoint(output, identity, engine, plan)
            if not args.still_only:
                (output / "frames").mkdir(exist_ok=True)
                for index in range(first, len(plan)):
                    image = render_exposure(engine, exposures[index], width, height)
                    path = output / "frames" / f"{index:06d}.png"
                    write_png(path, image, 8)
                    records.append(artifact(path, output))
                    if index % 30 == 0 or index == len(plan) - 1:
                        checkpoint(output, identity, engine, index, plan[index], records)
                    write_json(
                        output / "progress.json",
                        {
                            "frame": index + 1,
                            "frames": len(plan),
                            "source_fraction": plan[index] / steps,
                        },
                    )
                    print(
                        f"Frame {index + 1}/{len(plan)} · source {plan[index] / steps:.6f}",
                        flush=True,
                    )
            engine.advance_to(steps)
            linear = np.asarray(engine.render(width, height), dtype=np.float32)
            write_png(output / "poster.png", linear)
            write_array(output / "linear.npy", linear)
            write_array(output / "final-state.npy", engine.read_state())
            if not args.still_only:
                receipt["movie"] = encode_movie(
                    output, recipe, tools["ffmpeg"]["path"], tools["ffprobe"]["path"]
                )
            if (
                digest(source.path) != source.sha256
                or digest(output / "inputs" / "source.orbit") != source.sha256
            ):
                raise ValueError("Source changed during rendering or archival")
            if code_identity() != code or any(
                digest(output / "inputs" / "code" / name) != sha for name, sha in code.items()
            ):
                raise ValueError("Renderer code changed during rendering or archival")
            for tool in tools.values():
                if digest(Path(tool["path"])) != tool["sha256"]:
                    raise ValueError("Video tool changed during rendering")
            names = [
                "poster.png",
                "linear.npy",
                "final-state.npy",
                "recipe.json",
                "inputs/source.orbit",
            ]
            if not args.still_only:
                names += ["film.mp4", "movie.json"]
            names += ["inputs/code/" + name for name in code]
            receipt.update(
                complete=True,
                elapsed_seconds=time.monotonic() - started,
                final_step=steps,
                source_fraction=1.0,
                diagnostics={
                    "internal_steps": engine.internal_steps,
                    "maximum_courant": engine.maximum_courant,
                },
                artifacts=[artifact(output / name, output) for name in names] + records,
            )
            write_json(output / "receipt.json", receipt)
            print(json.dumps({"status": "complete", "output": str(output), "identity": identity}))
        except BaseException as error:
            receipt["error"] = str(error) or type(error).__name__
            write_json(output / "receipt.json", receipt)
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("orbit", "recipe", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--still-only", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--backend", default="egl")
    result.add_argument("--ffmpeg", type=Path)
    result.add_argument("--ffprobe", type=Path)
    return result


def interrupted(_signal, _frame):
    raise KeyboardInterrupt("Render interrupted; verified checkpoints remain available")


def main():
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run(parser().parse_args())
        return 0
    except KeyboardInterrupt as error:
        print(
            str(error) or "Render interrupted; verified checkpoints remain available",
            file=sys.stderr,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
