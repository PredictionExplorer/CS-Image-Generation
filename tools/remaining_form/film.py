#!/usr/bin/env python3
"""Build and photograph a cumulative sculpture, then encode its film.

The fixed studio must specify ground.height_units. Every excavation frame builds
its own geometry at an explicit source fraction; orbit frames reuse the final
mesh. No mesh interpolation, geometry resampling, or automatic camera fitting is
performed here. At most two frame jobs run together, each child using four threads.
--resume requires identical
inputs and delegates mesh and scene receipt checks to their original programs.
"""

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    encoded(value)
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def preserve(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Archived input changed: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def run_process(command, log, cancel=None):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Film work was cancelled")
    started = time.monotonic()
    with log.open("ab") as stream:
        stream.write(("\nCommand: " + json.dumps(command) + "\n").encode())
        stream.flush()
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT
        )
        try:
            while True:
                try:
                    code = child.wait(timeout=0.5 if cancel is not None else None)
                    break
                except subprocess.TimeoutExpired:
                    if cancel.is_set():
                        raise InterruptedError("Film work was cancelled") from None
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    if code:
        raise RuntimeError(f"Child exited with status {code}; see {log}")
    return time.monotonic() - started


def run_phase(frames, workers, job, accept, cancel):
    """Keep at most two frame jobs active and publish through one controller."""
    if workers not in (1, 2):
        raise ValueError("Frame workers must be one or two")
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(job, frame) for frame in frames]
        for future in concurrent.futures.as_completed(futures):
            accept(future.result())
    except BaseException:
        cancel.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=cancel.is_set())


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("remaining_film_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_geometry(builder, supplied):
    with tempfile.TemporaryDirectory(prefix="remaining-film-recipe-") as temporary:
        folder = Path(temporary)
        output = folder / "resolved.json"
        log = folder / "resolve.log"
        try:
            run_process(
                [
                    str(builder),
                    "--threads",
                    "4",
                    "resolve",
                    "--config",
                    str(supplied),
                    "--output",
                    str(output),
                ],
                log,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Geometry recipe resolution failed:\n{log.read_text()[-4000:]}"
            ) from error
        return read_json(output)


def fixed_studio(adapter, supplied):
    studio = adapter.merge_config(adapter.DEFAULTS, supplied)
    studio["render"]["threads"] = 4
    adapter.validate(studio)
    ground = studio["ground"]["height_units"]
    if type(ground) not in (int, float) or not math.isfinite(ground):
        raise ValueError(
            "A film requires fixed numeric ground.height_units from the approved studio"
        )
    if any(value % 2 for value in studio["render"]["resolution"]):
        raise ValueError("H264 yuv420p requires even width and height")
    return studio


def frame_plan(count, turntable_count, degrees, camera):
    if not 24 <= count <= 721 or not 0 <= turntable_count <= 240:
        raise ValueError("Excavation frame count must be24..721 and orbit count0..240")
    if not math.isfinite(degrees) or abs(degrees) > 360:
        raise ValueError("Turntable degrees must be finite and within -360..360")
    frames = [
        {
            "index": index,
            "phase": "excavation",
            "source_fraction": index / (count - 1),
            "geometry_index": index,
            "camera_position": list(camera["position"]),
        }
        for index in range(count)
    ]
    target = camera["target"]
    dx = camera["position"][0] - target[0]
    dy = camera["position"][1] - target[1]
    for index in range(turntable_count):
        u = (index + 1) / turntable_count
        angle = math.radians(degrees) * (u * u * (3.0 - 2.0 * u))
        frames.append(
            {
                "index": count + index,
                "phase": "camera_orbit",
                "source_fraction": 1.0,
                "geometry_index": count - 1,
                "camera_position": [
                    target[0] + dx * math.cos(angle) - dy * math.sin(angle),
                    target[1] + dx * math.sin(angle) + dy * math.cos(angle),
                    camera["position"][2],
                ],
            }
        )
    return frames


def encoding_timeline(frames, fps, start_hold, end_hold):
    if not 1 <= fps <= 120 or any(
        not math.isfinite(v) or not 0 <= v <= 30 for v in (start_hold, end_hold)
    ):
        raise ValueError("Require fps1..120 and finite hold durations0..30 seconds")
    start_count, end_count = (round(value * fps) for value in (start_hold, end_hold))
    slots = [{"kind": "start_hold", "render_frame": 0} for _ in range(start_count)]
    slots.extend({"kind": frame["phase"], "render_frame": frame["index"]} for frame in frames)
    slots.extend(
        {"kind": "final_hold", "render_frame": frames[-1]["index"]} for _ in range(end_count)
    )
    return [{"slot": i, "time_seconds": i / fps, **value} for i, value in enumerate(slots)]


def verify_build(folder, expected_recipe, request):
    record = read_json(folder / "build.json")
    identity = record["identity"]
    if (
        identity["recipe"] != expected_recipe
        or read_json(folder / "recipe.json") != expected_recipe
    ):
        raise ValueError("Geometry receipt or archived recipe differs from the planned source time")
    if (
        identity["orbit_sha256"] != request["inputs"]["orbit"]["sha256"]
        or identity["executable_sha256"] != request["inputs"]["builder"]["sha256"]
    ):
        raise ValueError("Geometry provenance differs from the frozen source or builder")
    if digest(folder / "mesh.ply") != record["mesh_sha256"]:
        raise ValueError("Geometry mesh hash differs from its build receipt")
    return record


def verify_scene(adapter, folder, studio, mesh_hash, request):
    scene = read_json(folder / "request.json")
    if scene["config"] != studio or scene["mesh_sha256"] != mesh_hash or scene["view"] != "front":
        raise ValueError("Scene camera, studio or mesh differs from the planned frame")
    if (
        scene["script_sha256"] != request["inputs"]["render_script"]["sha256"]
        or scene["runtime"]["binary_sha256"] != request["inputs"]["blender"]["sha256"]
    ):
        raise ValueError("Scene uses a different renderer script or Blender binary")
    identity = hashlib.sha256(adapter.encoded(scene)).hexdigest()
    if not adapter.completed_matches(folder, identity):
        raise ValueError("Scene is missing complete, hash-verified render artifacts")
    return read_json(folder / "receipt.json"), scene["runtime"]


def encode_film(args, output, request, manifest):
    movie = output / "film.mp4"
    receipt_path = output / "movie.json"
    if movie.exists():
        if not receipt_path.exists():
            raise ValueError("Movie lacks a completed receipt; preserve it and use a new output")
        receipt = read_json(receipt_path)
        if (
            receipt.get("request_sha256") != manifest["request_sha256"]
            or digest(movie) != receipt.get("sha256")
            or receipt.get("full_decode_verified") is not True
            or receipt.get("frames") != len(request["encoding_timeline"])
            or receipt.get("fps") != args.fps
            or receipt.get("codec") != "h264"
            or receipt.get("pixel_format") != "yuv420p"
            or receipt.get("crf") != 18
        ):
            raise ValueError("Existing movie does not match the verified film request")
        return receipt
    sequence = output / "encode-frames"
    sequence.mkdir(exist_ok=True)
    for slot in request["encoding_timeline"]:
        frame = manifest["frames"][slot["render_frame"]]
        source = output / frame["image"]
        if digest(source) != frame["png_sha256"]:
            raise ValueError("Render image changed before encoding")
        destination = sequence / f"frame_{slot['slot']:06}.png"
        if destination.exists():
            if digest(destination) != frame["png_sha256"]:
                raise ValueError("Encoding sequence contains a different frame")
        else:
            try:
                destination.hardlink_to(source)
            except OSError:
                shutil.copyfile(source, destination)
    partial = output / "film.partial.mp4"
    partial.unlink(missing_ok=True)
    command = [
        str(args.ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-n",
        "-framerate",
        str(args.fps),
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-start_number",
        "0",
        "-i",
        str(sequence / "frame_%06d.png"),
        "-frames:v",
        str(len(request["encoding_timeline"])),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "4",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv:flags=bilinear+accurate_rnd+bitexact",
        "-sws_dither",
        "none",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    seconds = run_process(command, output / "logs" / "encode.log")
    decode_log = output / "logs" / "decode.log"
    decode_log.unlink(missing_ok=True)
    run_process(
        [
            str(args.ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "4",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(partial),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        decode_log,
    )
    progress = decode_log.read_text()
    counts = [
        int(line.split("=", 1)[1]) for line in progress.splitlines() if line.startswith("frame=")
    ]
    if (
        "progress=end" not in progress
        or not counts
        or counts[-1] != len(request["encoding_timeline"])
    ):
        raise ValueError("Encoded film did not completely decode to the expected frame count")
    receipt = {
        "schema_version": 1,
        "request_sha256": manifest["request_sha256"],
        "sha256": digest(partial),
        "bytes": partial.stat().st_size,
        "frames": counts[-1],
        "fps": args.fps,
        "duration_seconds": counts[-1] / args.fps,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "crf": 18,
        "encode_seconds": seconds,
        "full_decode_verified": True,
        "command": command,
    }
    partial.replace(movie)
    write_json(receipt_path, receipt)
    return receipt


def run(args):
    inputs = {}
    for name in (
        "builder",
        "blender",
        "render_script",
        "ffmpeg",
        "orbit",
        "geometry_recipe",
        "studio_recipe",
    ):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_file() or (
            name in ("builder", "blender", "ffmpeg") and not os.access(path, os.X_OK)
        ):
            raise ValueError(f"Invalid explicit input path: {name}")
        setattr(args, name, path)
        inputs[name] = {"path": str(path), "sha256": digest(path)}
    adapter = load_adapter(args.render_script)
    geometry = resolve_geometry(args.builder, args.geometry_recipe)
    studio = fixed_studio(adapter, read_json(args.studio_recipe))
    frames = frame_plan(
        args.frames, args.turntable_frames, args.turntable_degrees, studio["cameras"]["front"]
    )
    timeline = encoding_timeline(frames, args.fps, args.start_hold, args.end_hold)
    for name, source in inputs.items():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Input changed while the film plan was prepared: {name}")
    request = {
        "schema_version": 1,
        "inputs": inputs,
        "runner_sha256": digest(Path(__file__)),
        "threads": 4,
        "geometry_recipe": geometry,
        "studio_recipe": studio,
        "fps": args.fps,
        "frames": frames,
        "encoding_timeline": timeline,
        "geometry_contract": (
            "independent cumulative source-time meshes; final mesh reused for camera orbit; "
            "no vertex interpolation"
        ),
    }
    request_hash = hashlib.sha256(encoded(request)).hexdigest()
    output = args.output.resolve()
    if output.exists() and (not args.resume or not output.is_dir()):
        raise ValueError("Output exists; use a new directory or --resume with identical inputs")
    if args.resume and not output.is_dir():
        raise ValueError("--resume requires an existing film archive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".film.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and not (output / "request.json").is_file():
            raise ValueError("Existing film has no archived request")
        preserve(output / "request.json", encoded(request))
        for folder in ("inputs", "recipes", "geometry", "frames", "logs"):
            (output / folder).mkdir(exist_ok=True)
        preserve(output / "inputs" / "geometry.json", encoded(geometry))
        preserve(output / "inputs" / "studio.json", encoded(studio))
        preserve(output / "inputs" / "render.py", args.render_script.read_bytes())
        if digest(output / "inputs" / "render.py") != inputs["render_script"]["sha256"]:
            raise ValueError("Render script changed while its immutable copy was prepared")
        manifest = {
            "schema_version": 1,
            "request_sha256": request_hash,
            "complete": False,
            "frames": [],
            "encoding_timeline": timeline,
            "render_runtime": None,
            "execution": {"parallel_frame_jobs": args.workers, "threads_per_child": 4},
        }
        write_json(output / "film.json", manifest)
        verified_geometry = {}
        completed_frames = {}
        cancel = threading.Event()

        def work(frame):
            if cancel.is_set():
                raise InterruptedError("Film work was cancelled")
            index = frame["index"]
            geometry_index = frame["geometry_index"]
            geometry_folder = output / "geometry" / f"source_{geometry_index:06}"
            geometry_recipe = copy.deepcopy(geometry)
            geometry_recipe["source_fraction"] = frame["source_fraction"]
            geometry_path = output / "recipes" / f"geometry_{geometry_index:06}.json"
            preserve(geometry_path, encoded(geometry_recipe))
            if frame["phase"] == "excavation":
                command = [
                    str(args.builder),
                    "--threads",
                    "4",
                    "build",
                    "--orbit",
                    str(args.orbit),
                    "--config",
                    str(geometry_path),
                    "--output",
                    str(geometry_folder),
                ]
                if geometry_folder.exists():
                    command.append("--resume")
                run_process(command, output / "logs" / f"build_{geometry_index:06}.log", cancel)
                build = verify_build(geometry_folder, geometry_recipe, request)
            else:
                build = verified_geometry[geometry_index]
            if digest(geometry_folder / "mesh.ply") != build["mesh_sha256"]:
                raise ValueError("Final geometry changed during its camera orbit")
            settings = copy.deepcopy(studio)
            settings["cameras"]["front"]["position"] = frame["camera_position"]
            studio_path = output / "recipes" / f"studio_{index:06}.json"
            preserve(studio_path, encoded(settings))
            scene_folder = output / "frames" / f"frame_{index:06}"
            command = [
                str(args.blender),
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--background",
                "--threads",
                "4",
                "--python",
                str(output / "inputs" / "render.py"),
                "--",
                "--mesh",
                str(geometry_folder / "mesh.ply"),
                "--recipe",
                str(studio_path),
                "--output",
                str(scene_folder),
                "--view",
                "front",
            ]
            if scene_folder.exists():
                command.append("--resume")
            print(
                f"Frame {index + 1}/{len(frames)}: {frame['phase']}, "
                f"source {frame['source_fraction']:.9f}",
                flush=True,
            )
            seconds = run_process(command, output / "logs" / f"render_{index:06}.log", cancel)
            receipt, runtime = verify_scene(
                adapter, scene_folder, settings, build["mesh_sha256"], request
            )
            record = {
                **frame,
                "complete": True,
                "image": str((scene_folder / "render.png").relative_to(output)),
                "png_sha256": receipt["artifacts"]["render.png"]["sha256"],
                "mesh_sha256": build["mesh_sha256"],
                "process_seconds": seconds,
                "build_receipt": str((geometry_folder / "build.json").relative_to(output)),
                "build_receipt_sha256": digest(geometry_folder / "build.json"),
                "render_receipt": str((scene_folder / "receipt.json").relative_to(output)),
                "render_receipt_sha256": digest(scene_folder / "receipt.json"),
            }
            return record, runtime, build

        def accept(result):
            record, runtime, build = result
            if manifest["render_runtime"] is not None and manifest["render_runtime"] != runtime:
                raise ValueError("Blender or color-management runtime changed between frames")
            manifest["render_runtime"] = runtime
            if record["phase"] == "excavation":
                verified_geometry[record["geometry_index"]] = build
            completed_frames[record["index"]] = record
            manifest["frames"] = [completed_frames[index] for index in sorted(completed_frames)]
            write_json(output / "film.json", manifest)

        try:
            run_phase(
                [frame for frame in frames if frame["phase"] == "excavation"],
                args.workers,
                work,
                accept,
                cancel,
            )
            # This barrier guarantees that the final verified mesh exists before
            # either camera-orbit job starts; those jobs never rebuild geometry.
            run_phase(
                [frame for frame in frames if frame["phase"] == "camera_orbit"],
                args.workers,
                work,
                accept,
                cancel,
            )
            if digest(args.ffmpeg) != inputs["ffmpeg"]["sha256"]:
                raise ValueError("FFmpeg changed before encoding")
            manifest["movie"] = encode_film(args, output, request, manifest)
            manifest["complete"] = True
            write_json(output / "film.json", manifest)
        except BaseException as error:
            manifest["error"] = str(error) or type(error).__name__
            write_json(output / "film.json", manifest)
            raise
    print(f"Verified film: {output / 'film.mp4'}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "builder",
        "blender",
        "render-script",
        "ffmpeg",
        "orbit",
        "geometry-recipe",
        "studio-recipe",
        "output",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--frames", type=int, default=144)
    result.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="Independent frame jobs; each child uses four threads",
    )
    result.add_argument("--fps", type=int, default=24)
    result.add_argument("--turntable-frames", type=int, default=0)
    result.add_argument("--turntable-degrees", type=float, default=35.0)
    result.add_argument(
        "--start-hold", type=float, default=1.0, help="Additional opening hold, seconds"
    )
    result.add_argument(
        "--end-hold", type=float, default=2.0, help="Additional final hold, seconds"
    )
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        print(
            "Film interrupted; completed frame archives remain available for --resume",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Film failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
