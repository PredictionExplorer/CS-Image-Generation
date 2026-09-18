#!/usr/bin/env python3
"""Photograph a finite set of depth studies with bounded, resumable Blender jobs.

Only exact completed archives are reused. Incomplete attempts belonging to this
experiment are preserved before replacement. Input copies and identities remain
fixed across retries; changed inputs require a new experiment directory.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    value = json.loads(path.read_text())
    encoded(value)
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def write(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encoded(value))
    partial.replace(path)


def preserve(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != digest(source):
            raise ValueError(f"Archived input differs: {path}")
        return
    partial = path.with_name(path.name + ".partial")
    shutil.copyfile(source, partial)
    if digest(partial) != digest(source):
        raise ValueError(f"Input changed during archival: {source}")
    partial.replace(path)


def checked(path, record):
    if (
        not path.is_file()
        or digest(path) != record["sha256"]
        or path.stat().st_size != record["bytes"]
    ):
        raise ValueError(f"Artifact changed or missing: {path}")


def verify_render(folder):
    request, receipt = read(folder / "request.json"), read(folder / "receipt.json")
    if (
        receipt.get("complete") is not True
        or receipt.get("identity_sha256") != hashlib.sha256(encoded(request)).hexdigest()
    ):
        raise ValueError(
            "Blender did not archive a complete render with its exact request identity"
        )
    artifacts = receipt.get("artifacts", {})
    if not {"render.png", "scene.blend"} <= artifacts.keys() or not any(
        name.endswith(".exr") for name in artifacts
    ):
        raise ValueError("Render archive requires its PNG, linear EXR and editable scene")
    for name, record in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact paths must stay inside the render archive")
        checked(folder / name, record)
    return receipt


def recipe_subset(expected, actual):
    """Compare authored controls to the renderer's resolved defaults without executing code."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and recipe_subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(recipe_subset(a, b) for a, b in zip(expected, actual, strict=True))
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    return expected == actual


def verify_case_inputs(folder, identity):
    """Bind this case's name and authored recipe to the immutable experiment inputs."""
    experiment = read(folder.parent / "experiment-request.json")
    if hashlib.sha256(encoded(experiment)).hexdigest() != identity:
        raise ValueError("Parent experiment identity differs")
    files = experiment["files"]
    recipe_name = f"recipes/{folder.name}.json"
    required = (
        recipe_name,
        "render.py",
        "materials.py",
        "bundle/manifest.json",
        "bundle/bundle.npz",
    )
    for name in required:
        path = folder.parent / "inputs" / name
        if name not in files or not path.is_file() or digest(path) != files[name]["sha256"]:
            raise ValueError(f"Case's archived experiment input changed or is missing: {name}")
    requested = read(folder.parent / "inputs" / recipe_name)
    actual = read(folder / "request.json")
    if not recipe_subset(requested, actual.get("recipe")):
        raise ValueError("Case recipe does not match its own archived input")
    if (
        actual.get("bundle_sha256") != files["bundle/bundle.npz"]["sha256"]
        or actual.get("bundle_manifest_sha256") != files["bundle/manifest.json"]["sha256"]
        or actual.get("renderer")
        != {name: files[name]["sha256"] for name in ("render.py", "materials.py")}
    ):
        raise ValueError("Case uses a different bundle or renderer")
    motion = actual.get("motion", {})
    if (
        motion.get("frames") != experiment["motion_frames"]
        or motion.get("fps") != experiment["fps"]
        or motion.get("source_fraction") != 1.0
    ):
        raise ValueError("Case motion or frozen source differs from its experiment")


def finished(folder, identity):
    path = folder / "experiment-result.json"
    if not path.exists():
        return False
    result = read(path)
    if result.get("identity_sha256") != identity:
        raise ValueError("Existing case belongs to different experiment inputs")
    if result.get("complete") is not True:
        return False
    verify_case_inputs(folder, identity)
    if result.get("case", folder.name) != folder.name:
        raise ValueError("Completed result belongs to a different case")
    verify_render(folder)
    if digest(folder / "receipt.json") != result["render_receipt_sha256"]:
        raise ValueError("Completed renderer receipt changed")
    if result.get("movie"):
        checked(folder / "film.mp4", result["movie"])
    return True


def stop(child):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.communicate()


def capture(command):
    child = subprocess.Popen(
        [str(value) for value in command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate()
    except BaseException:
        stop(child)
        raise
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
    return stdout


def encode_movie(folder, frames, fps, ffmpeg, ffprobe):
    receipt = verify_render(folder)
    for index in range(frames):
        name = f"frames/{index:06d}.png"
        if name not in receipt["artifacts"]:
            raise ValueError(f"Missing certified motion frame: {name}")
    first = folder / "frames/000000.png"
    with first.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Motion frames must be PNG files")
    dimensions = list(struct.unpack(">II", header[16:24]))
    partial = folder / "film.partial.mp4"
    capture(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-threads",
            "4",
            "-framerate",
            fps,
            "-i",
            folder / "frames/%06d.png",
            "-frames:v",
            frames,
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
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
    )
    info = json.loads(
        capture(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                partial,
            ]
        )
    )["streams"][0]
    if (
        [info["width"], info["height"]] != dimensions
        or int(info["nb_read_frames"]) != frames
        or Fraction(info["avg_frame_rate"]) != fps
    ):
        raise ValueError("Movie dimensions, frame count or cadence differ")
    progress = capture(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "4",
            "-i",
            partial,
            "-progress",
            "pipe:1",
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )
    counts = [int(line[6:]) for line in progress.splitlines() if line.startswith("frame=")]
    if not counts or counts[-1] != frames or "progress=end" not in progress:
        raise ValueError("Movie failed full decode verification")
    partial.replace(folder / "film.mp4")
    return {
        "sha256": digest(folder / "film.mp4"),
        "bytes": (folder / "film.mp4").stat().st_size,
        "frames": frames,
        "fps": fps,
        "resolution": dimensions,
        "duration_seconds": frames / fps,
        "full_decode_verified": True,
    }


def make_request(args):
    recipes = sorted(args.recipes.glob("*.json"))
    if not 1 <= len(recipes) <= 64 or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", path.stem) for path in recipes
    ):
        raise ValueError("Use one through 64 simply named JSON recipes")
    manifest = read(args.bundle / "manifest.json")
    if (
        manifest.get("complete") is not True
        or manifest["bundle"]["path"] != "bundle.npz"
        or manifest["identity_sha256"] != hashlib.sha256(encoded(manifest["request"])).hexdigest()
    ):
        raise ValueError("Bundle lacks a complete, correctly identified manifest")
    checked(args.bundle / "bundle.npz", manifest["bundle"])
    files = {
        "render.py": args.render_script,
        "materials.py": args.render_script.with_name("materials.py"),
        "bundle/manifest.json": args.bundle / "manifest.json",
        "bundle/bundle.npz": args.bundle / "bundle.npz",
    }
    files.update({f"recipes/{path.name}": path for path in recipes})
    if args.baseline:
        files["baseline" + args.baseline.suffix] = args.baseline
    request = {
        "schema_version": 1,
        "files": {
            name: {"source": str(path), "sha256": digest(path)} for name, path in files.items()
        },
        "blender": {"path": str(args.blender), "sha256": digest(args.blender)},
        "motion_frames": args.motion_frames,
        "fps": args.fps,
        "encoders": {
            name: None
            if getattr(args, name) is None
            else {"path": str(getattr(args, name)), "sha256": digest(getattr(args, name))}
            for name in ("ffmpeg", "ffprobe")
        },
    }
    return request, files, recipes


def blender_command(args, recipe, folder):
    command = [
        str(args.blender),
        "--factory-startup",
        "-b",
        "--threads",
        "4",
        "--python-exit-code",
        "1",
        "--python",
        str(args.output / "inputs/render.py"),
        "--",
        "--bundle",
        str(args.output / "inputs/bundle"),
        "--recipe",
        str(args.output / "inputs/recipes" / recipe.name),
        "--output",
        str(folder),
    ]
    if args.motion_frames > 1:
        command += ["--motion-frames", str(args.motion_frames), "--fps", str(args.fps)]
    return command


def execute(args, request, recipes):
    from tools.estuary_depth.gallery import build_gallery

    identity = hashlib.sha256(encoded(request)).hexdigest()
    status = {
        "schema_version": 1,
        "complete": False,
        "workers": args.workers,
        "cases": {path.stem: {"status": "pending"} for path in recipes},
    }
    active, ready, pending = {}, [], deque(recipes)

    def save():
        status["updated_unix"] = time.time()
        write(args.output / "status.json", status)

    def accept(recipe):
        folder = args.output / recipe.stem
        verify_case_inputs(folder, identity)
        verify_render(folder)
        movie = (
            encode_movie(folder, args.motion_frames, args.fps, args.ffmpeg, args.ffprobe)
            if args.motion_frames > 1
            else None
        )
        write(
            folder / "experiment-result.json",
            {
                "identity_sha256": identity,
                "case": folder.name,
                "complete": True,
                "render_receipt_sha256": digest(folder / "receipt.json"),
                "movie": movie,
            },
        )

    def publish():
        build_gallery(args.output, [path for path in recipes if path in ready], args.baseline)

    save()
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                recipe = pending.popleft()
                folder = args.output / recipe.stem
                stream = None
                try:
                    for name, info in request["files"].items():
                        if digest(args.output / "inputs" / name) != info["sha256"]:
                            raise ValueError("Archived experiment inputs changed")
                    if finished(folder, identity):
                        ready.append(recipe)
                        status["cases"][recipe.stem] = {"status": "complete", "reused": True}
                        publish()
                        save()
                        continue
                    if folder.exists():
                        folder.rename(
                            folder.with_name(f"{folder.name}.incomplete-{time.time_ns()}")
                        )
                    stream = (args.output / "logs" / f"{recipe.stem}.log").open("ab")
                    command = blender_command(args, recipe, folder)
                    stream.write(b"\nCommand: " + encoded(command))
                    stream.flush()
                    child = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[recipe.stem] = (child, stream, recipe)
                    status["cases"][recipe.stem] = {"status": "running", "pid": child.pid}
                except Exception as error:
                    if stream is not None and recipe.stem not in active:
                        stream.close()
                    if recipe in ready:
                        ready.remove(recipe)
                    status["cases"][recipe.stem] = {"status": "failed", "error": str(error)}
                save()
            for name, (child, stream, recipe) in list(active.items()):
                if child.poll() is None:
                    continue
                stream.close()
                del active[name]
                try:
                    if child.returncode:
                        raise RuntimeError(
                            f"Blender exited with status {child.returncode}; see logs/{name}.log"
                        )
                    accept(recipe)
                    ready.append(recipe)
                    publish()
                    status["cases"][name] = {"status": "complete"}
                except Exception as error:
                    if recipe in ready:
                        ready.remove(recipe)
                    status["cases"][name] = {"status": "failed", "error": str(error)}
                save()
            if active:
                time.sleep(0.2)
        status["complete"] = len(ready) == len(recipes)
        save()
        if not status["complete"]:
            raise RuntimeError("Some depth studies failed; inspect status.json and logs")
    except BaseException as error:
        for name, (child, stream, _) in active.items():
            stop(child)
            stream.close()
            status["cases"][name] = {"status": "interrupted"}
        status.update(complete=False, error=str(error) or type(error).__name__)
        save()
        raise


def run(args):
    if not 1 <= args.workers <= 3 or not 1 <= args.motion_frames <= 1441 or not 1 <= args.fps <= 60:
        raise ValueError("Use 1-3 workers, 1-1441 motion frames and 1-60 fps")
    for name in ("blender", "render_script", "bundle", "recipes", "baseline", "ffmpeg", "ffprobe"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    if args.motion_frames > 1:
        for name in ("ffmpeg", "ffprobe"):
            if getattr(args, name) is None:
                found = shutil.which(name)
                if not found:
                    raise ValueError(f"Motion encoding requires --{name}")
                setattr(args, name, Path(found).resolve())
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".experiment.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        request, files, recipes = make_request(args)
        path = args.output / "experiment-request.json"
        if path.exists() and read(path) != request:
            raise ValueError("Experiment belongs to different inputs; use a new output directory")
        if not path.exists() and any(
            item.name != ".experiment.lock" for item in args.output.iterdir()
        ):
            raise ValueError("Unowned output is preserved; use a new experiment directory")
        write(path, request)
        for name, source in files.items():
            preserve(args.output / "inputs" / name, source)
        (args.output / "logs").mkdir(exist_ok=True)
        execute(args, request, recipes)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("blender", "render-script", "bundle", "recipes", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    result.add_argument("--motion-frames", type=int, default=1)
    result.add_argument("--fps", type=int, default=24)
    result.add_argument("--baseline", type=Path)
    result.add_argument("--ffmpeg", type=Path)
    result.add_argument("--ffprobe", type=Path)
    return result


def interrupted(_signal, _frame):
    raise KeyboardInterrupt("Depth experiment interrupted; completed studies remain archived")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(parser().parse_args())
    except KeyboardInterrupt as error:
        print(error, file=sys.stderr)
        raise SystemExit(130) from None
