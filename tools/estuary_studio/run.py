#!/usr/bin/env python3
"""Render a complete three-body painting, then examine its frozen surface.

Every formation frame and the final camera orbit use the same material fields
and renderer. Canonical simulation steps never depend on output frame cadence.
Completed runs are immutable and verified before reuse; failed work is retained.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import math
import platform
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.run import encode_movie, write_array, write_png
from tools.estuary.source import Source
from tools.estuary_studio.common import (
    SURFACE_FIELDS,
    artifact,
    check_fields,
    checked,
    digest,
    encoded,
    read,
    require,
    runtime_identity,
    verify_code,
    write,
)

DEFAULT_RENDER = {
    "resolution": [1280, 960],
    "still_resolution": [3840, 2880],
    "formation_frames": 361,
    "orbit_frames": 193,
    "hold_frames": 24,
    "fps": 24,
    "still_tilt_degrees": 12.0,
    "orbit_tilt_degrees": 18.0,
    "azimuth_start": -35.0,
    "azimuth_end": 35.0,
    "still_supersampling": 1,
}


def number(value, name, low, high):
    require(
        type(value) in (float, int) and math.isfinite(value) and low <= value <= high,
        f"{name} must be a finite number in [{low}, {high}]",
    )


def dimensions(value):
    require(
        isinstance(value, list)
        and len(value) == 2
        and all(type(n) is int and 64 <= n <= 7680 and n % 2 == 0 for n in value)
        and math.prod(value) <= 33_177_600,
        "Invalid image dimensions",
    )
    return value


def validate_recipe(raw):
    from tools.estuary_studio.fresco import validate_config as fresco_config
    from tools.estuary_studio.monotype import validate_config as monotype_config
    from tools.estuary_studio.surface import validate_config as surface_config

    require(
        isinstance(raw, dict)
        and not raw.keys()
        - {
            "schema_version",
            "name",
            "family",
            "dynamics",
            "simulation",
            "surface",
            "projection",
            "render",
        },
        "Unknown studio recipe keys",
    )
    require(
        type(raw.get("schema_version", 1)) is int and raw.get("schema_version", 1) == 1,
        "Unsupported studio recipe",
    )
    family = raw.get("family", "fresco")
    require(family in ("fresco", "monotype", "nocturne"), "Unknown painting family")
    dynamics = raw.get("dynamics", "fresco" if family == "fresco" else "monotype")
    require(
        dynamics in ("fresco", "monotype") and (family == "nocturne" or dynamics == family),
        "Invalid family dynamics",
    )
    name = raw.get("name", family.title())
    require(isinstance(name, str) and 0 < len(name) <= 120, "Painting needs a short name")
    simulation = copy.deepcopy(raw.get("simulation", {}))
    require(isinstance(simulation, dict), "Simulation controls must be an object")
    if dynamics != "fresco":
        require(simulation.get("kind", family) == family, "Tool family differs from recipe")
        simulation["kind"] = family
    simulation = (fresco_config if dynamics == "fresco" else monotype_config)(simulation)
    require(math.prod(simulation["resolution"]) <= 40_000_000, "Simulation exceeds pixel budget")
    surface = copy.deepcopy(raw.get("surface", {}))
    require(isinstance(surface, dict), "Surface controls must be an object")
    require(surface.get("family", family) == family, "Surface family differs from recipe")
    surface["family"] = family
    require(
        surface.get("domain_scale", simulation["domain_scale"]) == simulation["domain_scale"],
        "Simulation and surface guard domains differ",
    )
    surface["domain_scale"] = simulation["domain_scale"]
    surface = surface_config(surface)
    require(
        surface["tone_map"] == "reinhard",
        "PNG and film output require display-bounded tone mapping",
    )
    projection = {"fill": 0.78, "rotation_degrees": 0.0}
    supplied = raw.get("projection", {})
    require(
        isinstance(supplied, dict) and not supplied.keys() - projection.keys(),
        "Unknown projection controls",
    )
    projection.update(supplied)
    number(projection["fill"], "fill", 0.05, 0.98)
    number(projection["rotation_degrees"], "rotation", -360, 360)
    render = copy.deepcopy(DEFAULT_RENDER)
    supplied = raw.get("render", {})
    require(
        isinstance(supplied, dict) and not supplied.keys() - render.keys(),
        "Unknown render controls",
    )
    render.update(supplied)
    sw, sh = simulation["resolution"]
    for key in ("resolution", "still_resolution"):
        w, h = dimensions(render[key])
        require(w * sh == h * sw, "Image aspect must match the physical painting")
    for key, low, high in (
        ("formation_frames", 2, 1801),
        ("orbit_frames", 1, 721),
        ("hold_frames", 0, 240),
        ("fps", 1, 60),
        ("still_supersampling", 1, 2),
    ):
        require(type(render[key]) is int and low <= render[key] <= high, f"Invalid {key}")
    for key in ("still_tilt_degrees", "orbit_tilt_degrees"):
        number(render[key], key, 0, 35)
    for key in ("azimuth_start", "azimuth_end"):
        number(render[key], key, -180, 180)
    require(
        simulation["steps"] % (render["formation_frames"] - 1) == 0,
        "Formation cadence must land on canonical simulation steps",
    )
    require(
        math.prod(render["still_resolution"]) * render["still_supersampling"] ** 2 <= 33_177_600,
        "Supersampled still exceeds pixel budget",
    )
    return {
        "schema_version": 1,
        "name": name,
        "family": family,
        "dynamics": dynamics,
        "simulation": simulation,
        "surface": surface,
        "projection": projection,
        "render": render,
    }


def frame_plan(recipe):
    settings = recipe["render"]
    steps = recipe["simulation"]["steps"]
    count = settings["formation_frames"]
    frames = [
        {
            "phase": "formation",
            "step": i * steps // (count - 1),
            "source_fraction": i / (count - 1),
            "tilt_degrees": 0.0,
            "azimuth_degrees": settings["azimuth_start"],
        }
        for i in range(count)
    ]
    for _ in range(settings["hold_frames"]):
        frames.append(dict(frames[-1], phase="hold"))
    for i in range(1, settings["orbit_frames"]):
        u = i / (settings["orbit_frames"] - 1)
        eased = u * u * (3 - 2 * u)
        frames.append(
            {
                "phase": "orbit",
                "step": steps,
                "source_fraction": 1.0,
                "tilt_degrees": settings["orbit_tilt_degrees"] * eased,
                "azimuth_degrees": settings["azimuth_start"]
                + eased * (settings["azimuth_end"] - settings["azimuth_start"]),
            }
        )
    return frames


def verified_run(folder):
    folder = Path(folder)
    receipt, request = read(folder / "receipt.json"), read(folder / "request.json")
    identity = hashlib.sha256(encoded(request)).hexdigest()
    require(
        receipt.get("complete") is True and receipt.get("identity_sha256") == identity,
        "Painting archive is incomplete or its identity differs",
    )
    require(
        {
            "inputs/source.orbit",
            "recipe.json",
            "final.npz",
            "poster.png",
            "poster-linear.npy",
            "frame-ledger.json",
        }
        <= receipt["artifacts"].keys(),
        "Required artwork artifacts missing",
    )
    for name, info in receipt["artifacts"].items():
        checked(folder, name, info)
    verify_code(folder / "inputs/code", request["code"])
    require(
        receipt.get("source_fraction") == 1.0
        and receipt.get("final_step") == request["recipe"]["simulation"]["steps"],
        "Painting does not contain the complete source",
    )
    require(read(folder / "recipe.json") == request["recipe"], "Archived recipe differs")
    require(receipt.get("source") == request["source"], "Archived source differs")
    require(
        receipt["artifacts"]["inputs/source.orbit"]["sha256"] == request["source"]["sha256"],
        "Archived source file differs from the recorded source",
    )
    if request["mode"] == "film":
        require({"film.mp4", "movie.json"} <= receipt["artifacts"].keys(), "Film artifacts missing")
        movie = receipt["movie"]
        require(
            movie == read(folder / "movie.json")
            and movie["fps"] == request["recipe"]["render"]["fps"]
            and movie["resolution"] == request["recipe"]["render"]["resolution"]
            and {k: movie["artifact"][k] for k in ("sha256", "bytes")}
            == receipt["artifacts"]["film.mp4"],
            "Movie metadata or identity differs",
        )
        plan = request["frames"]
        ledger = read(folder / "frame-ledger.json")
        require(
            len(ledger) == len(plan)
            and receipt["movie"]["frames"] == len(plan)
            and receipt["movie"]["full_decode_verified"] is True,
            "Film timeline differs",
        )
        for i, (frame, expected) in enumerate(zip(ledger, plan, strict=True)):
            require(
                frame["frame"] == i
                and frame["timing"] == expected
                and receipt["artifacts"][f"frames/{i:06d}.png"] == frame["artifact"],
                "Film frame timing or image differs",
            )
    return request, receipt


def record_array(path, fields):
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        np.savez(stream, **fields)
    partial.replace(path)


def run(args):
    from tools.estuary_studio.fresco import Fresco
    from tools.estuary_studio.monotype import Monotype
    from tools.estuary_studio.surface import Surface

    config = validate_recipe(read(args.recipe))
    if args.resolution:
        config["simulation"]["resolution"] = list(args.resolution)
    if args.image_size:
        config["render"]["still_resolution"] = list(args.image_size)
    if args.video_size:
        config["render"]["resolution"] = list(args.video_size)
    if args.formation_frames is not None:
        config["render"]["formation_frames"] = args.formation_frames
    if args.orbit_frames is not None:
        config["render"]["orbit_frames"] = args.orbit_frames
    config = validate_recipe(config)
    code = runtime_identity()
    sw, sh = config["simulation"]["resolution"]
    source = Source.read(args.source, aspect=sw / sh, **config["projection"])
    plan = [] if args.still_only else frame_plan(config)
    binaries = {}
    if plan:
        for name in ("ffmpeg", "ffprobe"):
            path = Path(shutil.which(name) or "").resolve(strict=True)
            require(path.is_file(), f"{name} is unavailable")
            binaries[name] = {"path": str(path), **artifact(path)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = (Fresco if config["dynamics"] == "fresco" else Monotype)(
            source, config["simulation"]
        )
        surface = None
        try:
            surface = Surface(config["surface"])
            request = {
                "schema_version": 1,
                "recipe": config,
                "code": code,
                "source": source.metadata,
                "mode": "film" if plan else "still",
                "frames": plan,
                "binaries": binaries,
                "hardware": {"simulation": engine.metadata, "surface": surface.metadata},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            }
            identity = hashlib.sha256(encoded(request)).hexdigest()
            if (output / "request.json").exists():
                require(
                    read(output / "request.json") == request,
                    "Existing output belongs to another recipe, source or runtime",
                )
                verified_run(output)
                print(encoded({"status": "reused", "identity": identity}).decode())
                return
            require(not any(p.name != ".lock" for p in output.iterdir()), "Output must be empty")
            write(output / "request.json", request)
            write(output / "recipe.json", config)
            write(output / "receipt.json", {"complete": False, "identity_sha256": identity})
            (output / "inputs").mkdir()
            shutil.copyfile(args.source, output / "inputs/source.orbit")
            require(
                digest(output / "inputs/source.orbit") == source.sha256,
                "Source changed while archiving",
            )
            # Runtime source is portable alongside the preserved material state.
            code_folder = output / "inputs/code"
            for package in ("estuary", "estuary_studio"):
                shutil.copytree(
                    Path(__file__).parents[1] / package,
                    code_folder / "tools" / package,
                    ignore=shutil.ignore_patterns("__pycache__", "test_*"),
                )
            if "depth" in code:
                dependency = code_folder / "tools/estuary_depth"
                dependency.mkdir()
                for name in code["depth"]:
                    shutil.copyfile(
                        Path(__file__).parents[1] / "estuary_depth" / name, dependency / name
                    )
            verify_code(code_folder, code)
            started = time.monotonic()
            settings = config["render"]
            artifacts = {
                "inputs/source.orbit": artifact(output / "inputs/source.orbit"),
                "recipe.json": artifact(output / "recipe.json"),
            }
            fields, ledger = None, []
            if plan:
                (output / "frames").mkdir()
            for i, frame in enumerate(plan):
                changed = fields is None or frame["step"] != engine.step
                if changed:
                    engine.advance_to(frame["step"])
                    fields = engine.snapshot()
                path = output / f"frames/{i:06d}.png"
                if frame["phase"] == "hold":
                    shutil.copyfile(output / f"frames/{i - 1:06d}.png", path)
                else:
                    pixels = surface.render(
                        {key: fields[key] for key in SURFACE_FIELDS} if changed else None,
                        size=tuple(settings["resolution"]),
                        tilt_degrees=frame["tilt_degrees"],
                        azimuth_degrees=frame["azimuth_degrees"],
                    )
                    write_png(path, pixels, depth=8)
                info = artifact(path)
                artifacts[str(path.relative_to(output))] = info
                ledger.append({"frame": i, "timing": frame, "artifact": info})
                if i % 24 == 0 or i == len(plan) - 1:
                    write(
                        output / "progress.json",
                        {
                            "frames": i + 1,
                            "total": len(plan),
                            "source_fraction": frame["source_fraction"],
                            "phase": frame["phase"],
                        },
                    )
                    print(f"STUDIO_FRAME {i + 1}/{len(plan)} {frame['phase']}", flush=True)
            if not plan:
                engine.advance_to(config["simulation"]["steps"])
                fields = engine.snapshot()
            statistics = check_fields(fields, (sw, sh))
            record_array(output / "final.npz", fields)
            width, height = settings["still_resolution"]
            ss = settings["still_supersampling"]
            pixels = surface.render(
                None if plan else {key: fields[key] for key in SURFACE_FIELDS},
                size=(width * ss, height * ss),
                tilt_degrees=settings["still_tilt_degrees"],
                azimuth_degrees=settings["azimuth_end"],
            )
            if ss > 1:
                pixels = (
                    pixels.reshape(height, ss, width, ss, 3)
                    .mean(axis=(1, 3), dtype=np.float64)
                    .astype(np.float32)
                )
            write_png(output / "poster.png", pixels)
            write_array(output / "poster-linear.npy", pixels)
            write(output / "frame-ledger.json", ledger)
            for name in ("final.npz", "poster.png", "poster-linear.npy", "frame-ledger.json"):
                artifacts[name] = artifact(output / name)
            for path in sorted(code_folder.rglob("*")):
                if path.is_file():
                    artifacts[str(path.relative_to(output))] = artifact(path)
            movie = None
            if plan:
                movie = encode_movie(
                    output,
                    {
                        "render": {
                            "frames": len(plan),
                            "fps": settings["fps"],
                            "resolution": settings["resolution"],
                        }
                    },
                    binaries["ffmpeg"]["path"],
                    binaries["ffprobe"]["path"],
                )
                artifacts["film.mp4"] = artifact(output / "film.mp4")
                artifacts["movie.json"] = artifact(output / "movie.json")
            require(
                runtime_identity() == code and digest(args.source) == source.sha256,
                "Source or runtime changed while painting",
            )
            write(
                output / "receipt.json",
                {
                    "schema_version": 1,
                    "complete": True,
                    "identity_sha256": identity,
                    "source": source.metadata,
                    "source_fraction": 1.0,
                    "final_step": engine.step,
                    "statistics": statistics,
                    "movie": movie,
                    "seconds": time.monotonic() - started,
                    "artifacts": artifacts,
                },
            )
            print(
                encoded(
                    {
                        "status": "complete",
                        "output": str(output),
                        "identity": identity,
                        "seconds": time.monotonic() - started,
                    }
                ).decode()
            )
        finally:
            if surface is not None:
                surface.close()
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "recipe", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--still-only", action="store_true")
    for key in ("resolution", "image-size", "video-size"):
        parser.add_argument("--" + key, nargs=2, type=int)
    parser.add_argument("--formation-frames", type=int)
    parser.add_argument("--orbit-frames", type=int)
    args = parser.parse_args()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == "__main__":
    main()
