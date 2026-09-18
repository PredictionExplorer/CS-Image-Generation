#!/usr/bin/env python3
"""Archive seeded confluences and several optical views of one physical history.

One canonical simulation feeds every requested view. Movie capture may exactly
area-average the GPU state, without changing its evolution. Final stills retain
the full simulation grid. The original orbit, palette, encounters, source code,
all frames and complete decoding evidence remain inspectable together.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
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
from tools.estuary_studio.common import artifact, checked, digest, encoded, read, require, write
from tools.estuary_studio.run import dimensions, frame_plan, number, record_array

ROOT = Path(__file__).resolve().parent
PACKAGES = ("estuary", "estuary_depth", "estuary_studio", "estuary_confluence")
LABELS = {"layered": "Layered confluence", "homogeneous": "Blended comparison"}
DEFAULT_RENDER = {
    "resolution": [1920, 1440],
    "still_resolution": [3840, 2880],
    "capture_resolution": None,
    "formation_frames": 301,
    "orbit_frames": 145,
    "hold_frames": 24,
    "fps": 24,
    "still_tilt_degrees": 8.0,
    "orbit_tilt_degrees": 12.0,
    "azimuth_start": -35.0,
    "azimuth_end": 35.0,
}


def runtime_identity():
    return {
        name: {
            str(p.relative_to(ROOT.parent / name)): digest(p)
            for p in sorted((ROOT.parent / name).rglob("*"))
            if p.is_file()
            and p.suffix in (".py", ".glsl", ".txt", ".html")
            and not p.name.startswith("test_")
        }
        for name in PACKAGES
    }


def verify_code(folder, code):
    require(set(code) == set(PACKAGES), "Incomplete runtime identity")
    for package, files in code.items():
        root = (Path(folder) / "tools" / package).resolve()
        for name, sha in files.items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root) and path.is_file() and digest(path) == sha,
                f"Archived runtime differs: {package}/{name}",
            )


def validate_recipe(raw):
    from tools.estuary_confluence.engine import reduction_factor
    from tools.estuary_confluence.engine import validate_config as simulation_config
    from tools.estuary_confluence.surface import validate_config as surface_config

    require(
        type(raw) is dict
        and not raw.keys()
        - {
            "schema_version",
            "name",
            "chromatic_count",
            "palette_mode",
            "looks",
            "encounters",
            "simulation",
            "surface",
            "projection",
            "render",
        },
        "Unknown confluence recipe keys",
    )
    require(
        type(raw.get("schema_version", 1)) is int and raw.get("schema_version", 1) == 1,
        "Unsupported recipe schema",
    )
    count = raw.get("chromatic_count", 3)
    require(type(count) is int and count in (3, 5), "Use three or five chromatic pigments")
    palette_mode = raw.get("palette_mode", "curated")
    require(
        type(palette_mode) is str and palette_mode in ("curated", "harmonic", "random"),
        "Choose a curated, harmonic, or random palette",
    )
    name = raw.get("name", "Confluence Fresco")
    require(isinstance(name, str) and 0 < len(name) <= 100, "Painting needs a short name")
    looks = raw.get("looks", ["layered"])
    require(
        type(looks) is list
        and 1 <= len(looks) <= 2
        and all(type(x) is str and x in LABELS for x in looks)
        and len(set(looks)) == len(looks),
        "Choose distinct known optical views",
    )
    encounters = raw.get("encounters", 3)
    require(type(encounters) is int and 0 <= encounters <= 3, "Use at most three encounter blooms")
    simulation = simulation_config(raw.get("simulation", {}))
    surface = copy.deepcopy(raw.get("surface", {}))
    require(type(surface) is dict, "Surface controls must be an object")
    require(
        surface.get("domain_scale", simulation["domain_scale"]) == simulation["domain_scale"],
        "Surface and simulation guard domains differ",
    )
    surface["domain_scale"] = simulation["domain_scale"]
    surface = surface_config(surface)
    require(surface["tone_map"] == "reinhard", "Display output needs bounded tone mapping")
    projection = {"fill": 0.78, "rotation_degrees": 0.0}
    supplied = raw.get("projection", {})
    require(
        type(supplied) is dict and not supplied.keys() - projection.keys(),
        "Invalid projection keys",
    )
    projection.update(supplied)
    number(projection["fill"], "fill", 0.05, 0.98)
    number(projection["rotation_degrees"], "rotation", -360, 360)
    render = copy.deepcopy(DEFAULT_RENDER)
    supplied = raw.get("render", {})
    require(type(supplied) is dict and not supplied.keys() - render.keys(), "Invalid render keys")
    render.update(supplied)
    sw, sh = simulation["resolution"]
    if render["capture_resolution"] is None:
        capture = [sw, sh]
        while capture[0] > 2048 and all(n % 2 == 0 for n in capture):
            capture = [n // 2 for n in capture]
        render["capture_resolution"] = capture
    for key in ("resolution", "still_resolution", "capture_resolution"):
        w, h = dimensions(render[key])
        require(w * sh == h * sw, "Image and material aspects must match")
    reduction_factor((sw, sh), tuple(render["capture_resolution"]))
    for key, lo, hi in (
        ("formation_frames", 2, 1801),
        ("orbit_frames", 1, 721),
        ("hold_frames", 0, 240),
        ("fps", 1, 60),
    ):
        require(type(render[key]) is int and lo <= render[key] <= hi, f"Invalid {key}")
    require(
        simulation["steps"] % (render["formation_frames"] - 1) == 0,
        "Movie captures must fall on canonical steps",
    )
    for key in ("still_tilt_degrees", "orbit_tilt_degrees"):
        number(render[key], key, 0, 30)
    for key in ("azimuth_start", "azimuth_end"):
        number(render[key], key, -180, 180)
    return {
        "schema_version": 1,
        "name": name,
        "chromatic_count": count,
        "palette_mode": palette_mode,
        "looks": looks,
        "encounters": encounters,
        "simulation": simulation,
        "surface": surface,
        "projection": projection,
        "render": render,
    }


def surface_configs(recipe):
    return {
        look: dict(
            recipe["surface"],
            mode=look,
            mix_control=0.0 if look == "homogeneous" else recipe["surface"]["mix_control"],
        )
        for look in recipe["looks"]
    }


def resolved_layout(recipe, seed):
    """Resolve initial geometry without a GPU or any dependence on frame cadence."""
    from tools.estuary_confluence.layout import plan_layout

    settings = recipe["simulation"]
    if settings["initial_pattern"] != "scattered":
        return None
    width, height = settings["resolution"]
    return plan_layout(
        seed,
        recipe["chromatic_count"],
        width / height,
        load_radius=settings["load_radius"],
        initial_load=settings["initial_load"],
        edge_width=settings["initial_edge_width"],
    )


def field_digest(fields):
    """Container-independent identity of the actual physical state."""
    sha = hashlib.sha256()
    for key in sorted(fields):
        value = np.ascontiguousarray(fields[key])
        sha.update(encoded({"field": key, "shape": list(value.shape), "dtype": value.dtype.str}))
        sha.update(memoryview(value).cast("B"))
    return sha.hexdigest()


def verify_run(folder):
    from tools.estuary_confluence.palette import generate_palette, normalize_seed
    from tools.estuary_confluence.surface import validate_fields

    folder = Path(folder)
    request, receipt = read(folder / "request.json"), read(folder / "receipt.json")
    identity = hashlib.sha256(encoded(request)).hexdigest()
    require(
        receipt.get("complete") is True and receipt.get("identity_sha256") == identity,
        "Confluence archive is incomplete or identity differs",
    )
    required = {
        "inputs/source.orbit",
        "recipe.json",
        "palette.json",
        "events.json",
        "final.npz",
        "frame-ledger.json",
    }
    for look in request["recipe"]["looks"]:
        required |= {f"{look}/poster.png", f"{look}/poster-linear.npy"}
        if request.get("layout") is not None:
            required |= {"layout.json", f"{look}/initial.png"}
        if request["mode"] == "film":
            required |= {f"{look}/film.mp4", f"{look}/movie.json"}
    require(required <= receipt["artifacts"].keys(), "Required confluence artifacts are missing")
    for name, record in receipt["artifacts"].items():
        checked(folder, name, record)
    verify_code(folder / "inputs/code", request["code"])
    require(request["mode"] in ("film", "still"), "Unknown archive mode")
    expected_frames = frame_plan(request["recipe"]) if request["mode"] == "film" else []
    require(
        request["frames"] == expected_frames, "Film does not cover the canonical complete timeline"
    )
    require(
        read(folder / "recipe.json") == request["recipe"]
        and read(folder / "palette.json") == request["palette"]
        and read(folder / "events.json") == request["events"],
        "Archived design inputs differ",
    )
    expected_layout = resolved_layout(request["recipe"], request["source"]["seed"])
    require(request.get("layout") == expected_layout, "Seeded starting layout differs")
    if expected_layout is not None:
        require(read(folder / "layout.json") == expected_layout, "Archived starting pools differ")
    require(
        request["recipe"].get("palette_mode", "curated")
        == request["palette"].get("mode", "curated"),
        "Palette algorithm differs from its recipe",
    )
    require(
        request["palette"]
        == generate_palette(
            request["source"]["seed"],
            request["recipe"]["chromatic_count"],
            mode=request["recipe"].get("palette_mode", "curated"),
        ),
        "Palette is not derived from its seed and algorithm",
    )
    require(
        receipt["source"] == request["source"]
        and receipt["artifacts"]["inputs/source.orbit"]["sha256"] == request["source"]["sha256"]
        and normalize_seed(request["source"]["seed"]) == request["palette"]["seed"],
        "Palette and trajectory identity differ",
    )
    require(
        receipt["source_fraction"] == 1.0
        and receipt["final_step"] == request["recipe"]["simulation"]["steps"],
        "Painting does not contain the full trajectory",
    )
    require(
        request["surface_configs"] == surface_configs(request["recipe"]),
        "Optical views differ from their recipe",
    )
    require(set(receipt["looks"]) == set(request["recipe"]["looks"]), "View selection differs")
    ledger = read(folder / "frame-ledger.json")
    require(len(ledger) == len(request["frames"]), "Frame ledger length differs")
    for i, (frame, timing) in enumerate(zip(ledger, request["frames"], strict=True)):
        require(
            frame["frame"] == i
            and frame["timing"] == timing
            and set(frame["images"]) == set(receipt["looks"]),
            "Frame ledger timing differs",
        )
        for look, record in frame["images"].items():
            require(
                record == receipt["artifacts"][f"{look}/frames/{i:06d}.png"],
                "Certified frame differs",
            )
            if i == 0 and expected_layout is not None:
                require(
                    record == receipt["artifacts"][f"{look}/initial.png"],
                    "Starting-pool image differs from the film's initial state",
                )
    with np.load(folder / "final.npz", allow_pickle=False) as archive:
        final = {key: archive[key] for key in archive.files}
    validate_fields(final, request["recipe"]["chromatic_count"] + 1)
    actual_state = field_digest(final)
    require(actual_state == receipt["physical_state_sha256"], "Physical state identity differs")
    for look, result in receipt["looks"].items():
        require(
            result["physical_state_sha256"] == actual_state,
            "Optical views use different physical states",
        )
        require(
            result["poster"] == receipt["artifacts"][f"{look}/poster.png"], "View poster differs"
        )
        if request["mode"] == "film":
            movie = read(folder / look / "movie.json")
            require(
                movie == result["movie"]
                and movie["full_decode_verified"] is True
                and movie["frames"] == len(ledger)
                and movie["fps"] == request["recipe"]["render"]["fps"]
                and movie["resolution"] == request["recipe"]["render"]["resolution"]
                and {k: movie["artifact"][k] for k in ("sha256", "bytes")}
                == receipt["artifacts"][f"{look}/film.mp4"],
                "View movie differs from its certified timeline",
            )
        else:
            require(result["movie"] is None, "Still-only view cannot advertise a film")
    return request, receipt


def run(args):
    from tools.estuary_confluence.engine import Engine
    from tools.estuary_confluence.events import plan_events
    from tools.estuary_confluence.palette import generate_palette
    from tools.estuary_confluence.surface import Surface, validate_fields

    recipe = read(args.recipe)
    for arg, key in (
        ("resolution", "resolution"),
        ("capture_resolution", "capture_resolution"),
        ("image_size", "still_resolution"),
        ("video_size", "resolution"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            section = "simulation" if arg == "resolution" else "render"
            recipe.setdefault(section, {})[key] = value
    recipe = validate_recipe(recipe)
    code = runtime_identity()
    sw, sh = recipe["simulation"]["resolution"]
    source = Source.read(args.source, aspect=sw / sh, **recipe["projection"])
    palette = generate_palette(source.seed, recipe["chromatic_count"], mode=recipe["palette_mode"])
    layout = resolved_layout(recipe, source.seed)
    events = plan_events(source, recipe["encounters"])
    frames = [] if args.still_only else frame_plan(recipe)
    binaries = {}
    if frames:
        for name in ("ffmpeg", "ffprobe"):
            path = Path(shutil.which(name) or "").resolve(strict=True)
            require(path.is_file(), f"{name} is unavailable")
            binaries[name] = {"path": str(path), **artifact(path)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(source, recipe["simulation"], palette, events)
        surfaces = {}
        try:
            require(getattr(engine, "layout", None) == layout, "Engine starting layout differs")
            configs = surface_configs(recipe)
            for look, config in configs.items():
                surfaces[look] = Surface(config, palette)
            request = {
                "schema_version": 1,
                "recipe": recipe,
                "surface_configs": configs,
                "code": code,
                "source": source.metadata,
                "palette": palette,
                "events": events,
                "layout": layout,
                "mode": "film" if frames else "still",
                "frames": frames,
                "binaries": binaries,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "hardware": {
                    "simulation": engine.metadata,
                    "surfaces": {k: v.metadata for k, v in surfaces.items()},
                },
                "capture": "read-only GPU area averages; final still uses full physical state",
            }
            identity = hashlib.sha256(encoded(request)).hexdigest()
            if (output / "request.json").exists():
                require(
                    read(output / "request.json") == request, "Existing run has different inputs"
                )
                verify_run(output)
                print(encoded({"status": "reused", "output": str(output)}).decode())
                return
            require(not any(p.name != ".lock" for p in output.iterdir()), "Output must be empty")
            write(output / "request.json", request)
            write(output / "receipt.json", {"complete": False, "identity_sha256": identity})
            for name, value in (("recipe", recipe), ("palette", palette), ("events", events)):
                write(output / f"{name}.json", value)
            if layout is not None:
                write(output / "layout.json", layout)
            (output / "inputs").mkdir()
            shutil.copyfile(args.source, output / "inputs/source.orbit")
            require(
                digest(output / "inputs/source.orbit") == source.sha256,
                "Source changed while archiving",
            )
            code_folder = output / "inputs/code/tools"
            for package, files in code.items():
                for name in files:
                    target = code_folder / package / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT.parent / package / name, target)
            verify_code(output / "inputs/code", code)
            started = time.monotonic()
            render = recipe["render"]
            artifacts = {
                name: artifact(output / name)
                for name in ("recipe.json", "palette.json", "events.json", "inputs/source.orbit")
            }
            if layout is not None:
                artifacts["layout.json"] = artifact(output / "layout.json")
            for look in surfaces:
                (output / look).mkdir()
                if frames:
                    (output / look / "frames").mkdir()
            if layout is not None and not frames:
                initial = engine.snapshot(resolution=tuple(render["capture_resolution"]))
                for look, surface in surfaces.items():
                    pixels = surface.render(
                        initial,
                        size=tuple(render["resolution"]),
                        tilt_degrees=0.0,
                        azimuth_degrees=render["azimuth_start"],
                    )
                    path = output / look / "initial.png"
                    write_png(path, pixels, depth=8)
                    artifacts[f"{look}/initial.png"] = artifact(path)
            fields, ledger = None, []
            for i, frame in enumerate(frames):
                changed = fields is None or frame["step"] != engine.step
                if changed:
                    engine.advance_to(frame["step"])
                    fields = engine.snapshot(resolution=tuple(render["capture_resolution"]))
                images = {}
                for look, surface in surfaces.items():
                    name = f"{look}/frames/{i:06d}.png"
                    path = output / name
                    if frame["phase"] == "hold":
                        shutil.copyfile(output / f"{look}/frames/{i - 1:06d}.png", path)
                    else:
                        pixels = surface.render(
                            fields if changed else None,
                            size=tuple(render["resolution"]),
                            tilt_degrees=frame["tilt_degrees"],
                            azimuth_degrees=frame["azimuth_degrees"],
                        )
                        write_png(path, pixels, depth=8)
                    images[look] = artifacts[name] = artifact(path)
                    if i == 0 and layout is not None:
                        shutil.copyfile(path, output / look / "initial.png")
                        artifacts[f"{look}/initial.png"] = images[look]
                ledger.append({"frame": i, "timing": frame, "images": images})
                if i % 24 == 0 or i == len(frames) - 1:
                    write(
                        output / "progress.json",
                        {
                            "frames": i + 1,
                            "total": len(frames),
                            "source_fraction": frame["source_fraction"],
                            "phase": frame["phase"],
                        },
                    )
                    print(f"CONFLUENCE_FRAME {i + 1}/{len(frames)} {frame['phase']}", flush=True)
            if not frames:
                engine.advance_to(recipe["simulation"]["steps"])
            final = engine.snapshot()
            validate_fields(final, recipe["chromatic_count"] + 1)
            record_array(output / "final.npz", final)
            final_identity = field_digest(final)
            results = {}
            for look, surface in surfaces.items():
                pixels = surface.render(
                    final,
                    size=tuple(render["still_resolution"]),
                    tilt_degrees=render["still_tilt_degrees"],
                    azimuth_degrees=render["azimuth_end"],
                )
                write_png(output / look / "poster.png", pixels)
                write_array(output / look / "poster-linear.npy", pixels)
                for name in (f"{look}/poster.png", f"{look}/poster-linear.npy"):
                    artifacts[name] = artifact(output / name)
                results[look] = {
                    "poster": artifacts[f"{look}/poster.png"],
                    "movie": None,
                    "physical_state_sha256": final_identity,
                }
            write(output / "frame-ledger.json", ledger)
            for name in ("final.npz", "frame-ledger.json"):
                artifacts[name] = artifact(output / name)
            for path in sorted((output / "inputs/code").rglob("*")):
                if path.is_file():
                    artifacts[str(path.relative_to(output))] = artifact(path)
            if frames:
                encoding_recipe = {
                    "render": {
                        "frames": len(frames),
                        "fps": render["fps"],
                        "resolution": render["resolution"],
                    }
                }
                for look in surfaces:
                    results[look]["movie"] = encode_movie(
                        output / look,
                        encoding_recipe,
                        binaries["ffmpeg"]["path"],
                        binaries["ffprobe"]["path"],
                    )
                    for name in (f"{look}/movie.json", f"{look}/film.mp4"):
                        artifacts[name] = artifact(output / name)
            require(
                runtime_identity() == code and digest(args.source) == source.sha256,
                "Runtime or original source changed",
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
                    "physical_state_sha256": final_identity,
                    "looks": results,
                    "seconds": time.monotonic() - started,
                    "artifacts": artifacts,
                },
            )
            print(
                encoded(
                    {
                        "status": "complete",
                        "output": str(output),
                        "seconds": time.monotonic() - started,
                    }
                ).decode()
            )
        finally:
            for surface in surfaces.values():
                surface.close()
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--still-only", action="store_true")
    for name in ("resolution", "capture-resolution", "image-size", "video-size"):
        parser.add_argument("--" + name, nargs=2, type=int)
    args = parser.parse_args()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == "__main__":
    main()
