"""Edit full Estuary formation into a slow examination of its completed painting.

The entire source finishes before a one-second dissolve begins. That editorial
transition mixes two renderings of the same final pigment state; it does not
extend the physical simulation or animate new pigment. Output is 1920x1440/24fps.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import signal
import sys
from fractions import Fraction
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.run import checked_artifact
from tools.estuary_depth.experiment import (
    capture,
    checked,
    digest,
    encoded,
    finished,
    preserve,
    read,
    write,
)
from tools.estuary_depth.prepare import require, verified_run

FPS, RESOLUTION = 24, [1920, 1440]


def timeline(formation_frames, formation_fps, orbit_frames, orbit_fps):
    for value in (formation_frames, formation_fps, orbit_frames, orbit_fps):
        require(type(value) is int and value > 0, "Source movie cadence must use positive integers")
    durations = [Fraction(formation_frames, formation_fps), Fraction(orbit_frames, orbit_fps)]
    counts = [math.ceil(duration * FPS) for duration in durations]
    require(
        counts[0] >= 2 and counts[1] > FPS and max(counts) <= FPS * 600,
        "Movie duration exceeds the edit budget or orbit cannot complete the dissolve",
    )
    result = {
        "fps": FPS,
        "resolution": RESOLUTION,
        "formation_frames": counts[0],
        "orbit_frames": counts[1],
        "original_durations": [str(value) for value in durations],
        "endpoint_hold_frames": FPS,
        "crossfade_frames": FPS,
        "crossfade_start_frame": counts[0],
        "output_frames": sum(counts),
        "duration": str(Fraction(sum(counts), FPS)),
        "semantics": "Full recorded formation completes, then a one-second editorial dissolve "
        "from its held final exposure into a camera orbit of the same frozen painting",
        "resampling": "24fps nearest temporal sample; rational durations round up to whole frames; "
        "endpoint frames are padded before exact frame trimming",
        "crossfade_space": "display-encoded editorial dissolve; not simulated illumination",
    }
    return result


def validate_inputs(formation, orbit):
    request, identity, recipe, _state, records = verified_run(formation)
    require("film.mp4" in records and "movie.json" in records, "Formation needs a completed movie")
    movie_path = checked_artifact(formation, records["film.mp4"])
    movie = read(checked_artifact(formation, records["movie.json"]))
    require(movie.get("full_decode_verified") is True, "Formation movie was not fully verified")
    require(movie["artifact"] == records["film.mp4"], "Formation movie identities differ")
    require(
        movie["frames"] == recipe["render"]["frames"] and movie["fps"] == recipe["render"]["fps"],
        "Formation cadence differs from its source recipe",
    )
    experiment = read(orbit.parent / "experiment-request.json")
    experiment_identity = hashlib.sha256(encoded(experiment)).hexdigest()
    require(finished(orbit, experiment_identity), "Orbit case is incomplete")
    result, receipt = read(orbit / "experiment-result.json"), read(orbit / "receipt.json")
    orbit_request = read(orbit / "request.json")
    orbit_movie = result.get("movie")
    require(
        orbit_movie and orbit_movie.get("full_decode_verified") is True,
        "Orbit needs a verified movie",
    )
    require(
        receipt.get("source_fraction") == 1.0 and orbit_request["motion"]["source_fraction"] == 1.0,
        "Orbit must examine the completed painting",
    )
    for key in ("sha256", "seed"):
        require(receipt["source"][key] == request["source"][key], f"Orbit source {key} differs")
    bundle = read(orbit.parent / "inputs/bundle/manifest.json")
    require(
        bundle["request"]["inputs"]["render_identity"] == identity
        and bundle["request"]["inputs"]["artifacts"]["final-state.npy"]["sha256"]
        == records["final-state.npy"]["sha256"],
        "Orbit uses a different completed painting",
    )
    require(
        orbit_movie["frames"] == orbit_request["motion"]["frames"]
        and orbit_movie["fps"] == orbit_request["motion"]["fps"],
        "Orbit movie cadence differs",
    )
    plan = timeline(movie["frames"], movie["fps"], orbit_movie["frames"], orbit_movie["fps"])
    source = {
        "seed": request["source"]["seed"],
        "sha256": request["source"]["sha256"],
        "render_identity": identity,
        "final_state_sha256": records["final-state.npy"]["sha256"],
    }
    inputs = {
        "formation": {
            "path": str(movie_path),
            "sha256": records["film.mp4"]["sha256"],
            "bytes": records["film.mp4"]["bytes"],
            "frames": movie["frames"],
            "fps": movie["fps"],
            "request_sha256": digest(formation / "request.json"),
            "receipt_sha256": digest(formation / "receipt.json"),
        },
        "orbit": {
            "path": str(orbit / "film.mp4"),
            "sha256": orbit_movie["sha256"],
            "bytes": orbit_movie["bytes"],
            "frames": orbit_movie["frames"],
            "fps": orbit_movie["fps"],
            "request_sha256": digest(orbit / "request.json"),
            "receipt_sha256": digest(orbit / "receipt.json"),
            "experiment_identity": experiment_identity,
        },
    }
    return source, inputs, plan


def command(ffmpeg, output, plan):
    filters = []
    for index, name in enumerate(("formation", "orbit")):
        count = plan[name + "_frames"] + (plan["endpoint_hold_frames"] if index == 0 else 0)
        filters.append(
            f"[{index}:v]tpad=stop_mode=clone:stop_duration=2,setpts=PTS-STARTPTS,"
            f"fps={FPS}:round=near:start_time=0,scale=1920:1440:flags=lanczos:out_color_matrix=bt709:"
            f"out_range=tv,setsar=1,format=yuv444p,trim=end_frame={count},settb=1/{FPS}[{name}]"
        )
    offset = float(Fraction(plan["crossfade_start_frame"], FPS))
    filters.append(
        f"[formation][orbit]xfade=transition=fade:duration=1:offset={offset:.12f},"
        f"trim=end_frame={plan['output_frames']},format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709[film]"
    )
    return [
        str(ffmpeg),
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-threads",
        "4",
        "-i",
        str(output / "inputs/formation.mp4"),
        "-threads",
        "4",
        "-i",
        str(output / "inputs/orbit.mp4"),
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[film]",
        "-frames:v",
        str(plan["output_frames"]),
        "-an",
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
        str(output / "film.partial.mp4"),
    ]


def probe(ffprobe, path):
    return json.loads(
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
                path,
            ]
        )
    )["streams"][0]


def verify_timing(info, frames, fps, resolution=None):
    require(
        int(info["nb_read_frames"]) == frames and Fraction(info["avg_frame_rate"]) == fps,
        "Decoded movie cadence or frame count differs",
    )
    require(
        Fraction(info["duration_ts"]) * Fraction(info["time_base"]) == Fraction(frames, fps),
        "Movie duration differs from its rational frame schedule",
    )
    require(
        resolution is None or [info["width"], info["height"]] == resolution,
        "Movie resolution differs",
    )
    if resolution is not None:
        require(
            info.get("color_primaries") == "bt709"
            and info.get("color_transfer") == "iec61966-2-1"
            and info.get("color_space") == "bt709"
            and info.get("color_range") == "tv",
            "Edited movie color metadata differs",
        )


def verify_complete(output):
    output = Path(output)
    request, receipt = read(output / "request.json"), read(output / "receipt.json")
    require(
        receipt.get("complete") is True
        and receipt.get("identity_sha256") == hashlib.sha256(encoded(request)).hexdigest(),
        "Edited movie identity differs or is incomplete",
    )
    require(receipt.get("source") == request["source"], "Edited movie source metadata differs")
    required = {
        "film.mp4",
        "request.json",
        "command.json",
        "probe.json",
        "decode.txt",
        "inputs/formation.mp4",
        "inputs/orbit.mp4",
    }
    require(
        required <= receipt["artifacts"].keys(), "Edited movie archive lacks required artifacts"
    )
    for name, record in receipt["artifacts"].items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe artifact path"
        )
        checked(output / name, record)
    movie, plan = receipt["movie"], request["timeline"]
    require(
        movie["path"] == "film.mp4"
        and movie["full_decode_verified"] is True
        and movie["frames"] == plan["output_frames"]
        and movie["fps"] == FPS
        and movie["resolution"] == RESOLUTION
        and {key: movie[key] for key in ("sha256", "bytes")} == receipt["artifacts"]["film.mp4"],
        "Edited movie verification metadata differs",
    )
    verify_timing(read(output / "probe.json"), movie["frames"], FPS, RESOLUTION)
    return receipt


def compose(formation_run, orbit_case, output_dir, ffmpeg=None, ffprobe=None):
    formation, orbit = (
        Path(formation_run).resolve(strict=True),
        Path(orbit_case).resolve(strict=True),
    )
    source, inputs, plan = validate_inputs(formation, orbit)
    tools = {}
    for name, selected in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        selected = selected or shutil.which(name)
        require(selected is not None, f"Missing {name}")
        path = Path(selected).resolve(strict=True)
        tools[name] = {"path": str(path), "sha256": digest(path)}
    request = {
        "schema_version": 1,
        "source": source,
        "inputs": inputs,
        "timeline": plan,
        "tools": tools,
        "editor_sha256": digest(Path(__file__)),
    }
    identity = hashlib.sha256(encoded(request)).hexdigest()
    output = Path(output_dir).resolve()
    if output.exists():
        require(
            (output / "receipt.json").is_file(), "Incomplete edit preserved; choose a new output"
        )
        receipt = verify_complete(output)
        require(receipt["identity_sha256"] == identity, "Existing edit belongs to different inputs")
        return receipt
    output.mkdir(parents=True)
    with (output / ".edit.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / "request.json", request)
        write(output / "receipt.json", {"complete": False, "identity_sha256": identity})
        try:
            for name, record in inputs.items():
                preserve(output / f"inputs/{name}.mp4", Path(record["path"]))
                checked(output / f"inputs/{name}.mp4", record)
                verify_timing(
                    probe(tools["ffprobe"]["path"], output / f"inputs/{name}.mp4"),
                    record["frames"],
                    record["fps"],
                )
            args = command(tools["ffmpeg"]["path"], output, plan)
            write(output / "command.json", args)
            capture(args)
            partial = output / "film.partial.mp4"
            info = probe(tools["ffprobe"]["path"], partial)
            verify_timing(info, plan["output_frames"], FPS, RESOLUTION)
            write(output / "probe.json", info)
            progress = capture(
                [
                    tools["ffmpeg"]["path"],
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
            require(
                counts and counts[-1] == plan["output_frames"] and "progress=end" in progress,
                "Edited movie did not fully decode",
            )
            (output / "decode.txt").write_text(progress)
            for record in inputs.values():
                checked(Path(record["path"]), record)
            require(
                digest(Path(__file__)) == request["editor_sha256"], "Editor changed during encoding"
            )
            for tool in tools.values():
                require(
                    digest(Path(tool["path"])) == tool["sha256"], "Encoder changed during encoding"
                )
            partial.replace(output / "film.mp4")
            names = [
                "film.mp4",
                "request.json",
                "command.json",
                "probe.json",
                "decode.txt",
                "inputs/formation.mp4",
                "inputs/orbit.mp4",
            ]
            artifacts = {
                name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                for name in names
            }
            receipt = {
                "schema_version": 1,
                "complete": True,
                "identity_sha256": identity,
                "source": source,
                "movie": {
                    "path": "film.mp4",
                    **artifacts["film.mp4"],
                    "frames": plan["output_frames"],
                    "fps": FPS,
                    "resolution": RESOLUTION,
                    "duration_seconds": float(Fraction(plan["duration"])),
                    "full_decode_verified": True,
                },
                "artifacts": artifacts,
            }
            write(output / "receipt.json", receipt)
            return verify_complete(output)
        except BaseException as error:
            write(
                output / "receipt.json",
                {
                    "complete": False,
                    "identity_sha256": identity,
                    "error": str(error) or type(error).__name__,
                },
            )
            raise


def interrupted(_signal, _frame):
    raise KeyboardInterrupt("Film edit interrupted; intermediate files remain archived")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("formation-run", "orbit-case", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    print(
        encoded(
            compose(args.formation_run, args.orbit_case, args.output, args.ffmpeg, args.ffprobe)
        ).decode()
    )


if __name__ == "__main__":
    main()
