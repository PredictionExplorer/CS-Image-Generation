"""Compare grounds, lighting and relief on a verified, unchanged painting.

Appearance studies reuse a completed physical archive. They never replay or
modify transport, and never claim a different simulation has been performed.
The parent archive remains the source of the original recording and material.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import platform
import shutil
from pathlib import Path

import numpy as np

from tools.estuary.run import write_array, write_png
from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.run import dimensions

from .assessment import image_balance
from .backgrounds import generate_background, validate_background
from .run import ROOT, equivalent_design, field_digest, runtime_identity, verify_code, verify_run
from .surface import Surface, validate_config

VERSION = "confluence-appearance-v1"
PRESETS = {
    "white": ("White", "white", {}, None),
    "charcoal": ("Charcoal", "charcoal", {}, None),
    "midnight": ("Midnight blue", "midnight-blue", {}, None),
    "aubergine": ("Aubergine", "aubergine", {}, None),
    "palette-night": ("Palette night", "palette-night", {}, None),
    "raking": (
        "Grazing light",
        "palette-night",
        {"key_elevation_degrees": 11, "key_strength": 6.0, "ambient": 0.14, "fill_strength": 0.08},
        14,
    ),
    "relief": (
        "Raised paint",
        "palette-night",
        {
            "height_scale": 5.0,
            "key_elevation_degrees": 14,
            "key_strength": 5.0,
            "ambient": 0.16,
            "fill_strength": 0.08,
            "roughness_scale": 1.15,
        },
        20,
    ),
    "satin": (
        "Satin reflection",
        "charcoal",
        {
            "roughness_scale": 0.65,
            "anisotropy": 0.55,
            "key_elevation_degrees": 32,
            "key_strength": 2.8,
            "ambient": 0.18,
        },
        14,
    ),
}


def presentation(parent, name):
    """Resolve one explicit appearance without changing pigment or geometry data."""
    require(name in PRESETS, "Unknown appearance study")
    label, ground, overrides, tilt = PRESETS[name]
    background = generate_background(ground, parent["palette"])
    require("layered" in parent["surface_configs"], "Appearance studies need a layered view")
    base = copy.deepcopy(parent["surface_configs"]["layered"])
    require(base["finish"] == "crisp", "Ground studies require the crisp silhouette model")
    surface = validate_config({**base, **overrides, "ground_srgb": background["ground_srgb"]})
    render = parent["recipe"]["render"]
    return {
        "id": name,
        "name": label,
        "background": background,
        "surface": surface,
        "camera": {
            "tilt_degrees": render["still_tilt_degrees"] if tilt is None else tilt,
            "azimuth_degrees": render["azimuth_end"],
        },
    }


def film_recipe(parent, name):
    """Use a selected presentation for an ordinary full-trajectory film."""
    look = presentation(parent, name)
    recipe = copy.deepcopy(parent["recipe"])
    recipe["name"] = f"Convergence · {look['name']}"
    recipe["looks"] = ["layered"]
    recipe["surface"] = look["surface"]
    recipe["render"]["still_tilt_degrees"] = look["camera"]["tilt_degrees"]
    if name in ("raking", "relief", "satin"):
        recipe["render"]["orbit_tilt_degrees"] = look["camera"]["tilt_degrees"]
    return recipe


def _preview(pixels):
    h, w = pixels.shape[:2]
    factor = 1
    while max(w, h) / factor > 960 and w % (factor * 2) == h % (factor * 2) == 0:
        factor *= 2
    if factor == 1:
        return pixels
    return pixels.reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3), dtype="f4")


def _png_size(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "Invalid study PNG")
    return [int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")]


def _verify_resolved_study(folder, *, version, resolve):
    """Verify a same-material archive using its versioned presentation resolver."""
    folder = Path(folder)
    request, receipt = read(folder / "request.json"), read(folder / "receipt.json")
    require(
        request["version"] == version
        and receipt.get("complete") is True
        and receipt["identity_sha256"] == hashlib.sha256(encoded(request)).hexdigest(),
        "Appearance archive is incomplete or its identity differs",
    )
    files = receipt["artifacts"]
    require(
        {"parent-request.json", "parent-receipt.json"} <= files.keys(),
        "Missing physical provenance",
    )
    for name, record in files.items():
        checked(folder, name, record)
    verify_code(folder / "inputs/code", request["code"])
    parent, physical = read(folder / "parent-request.json"), read(folder / "parent-receipt.json")
    require(
        physical["complete"] is True
        and physical["identity_sha256"] == hashlib.sha256(encoded(parent)).hexdigest()
        and physical["identity_sha256"] == request["parent_identity_sha256"]
        and physical["physical_state_sha256"] == request["physical_state_sha256"]
        and physical["source_fraction"] == 1.0
        and physical["final_step"] == parent["recipe"]["simulation"]["steps"],
        "Appearance study belongs to another or incomplete physical archive",
    )
    require(request["seed"] == parent["source"]["seed"], "Appearance seed differs")
    require(
        request["chromatic_count"] == parent["recipe"]["chromatic_count"],
        "Appearance pigment count differs",
    )
    require(request["source_sha256"] == parent["source"]["sha256"], "Appearance source differs")
    expected = [resolve(parent, name) for name in request["presentations"]]
    require(len(request["looks"]) == len(expected), "Appearance view count differs")
    for archived, planned in zip(request["looks"], expected, strict=True):
        validate_background(archived["background"], parent["palette"])
        # Exact archived colors remain authoritative across harmless numerical
        # differences when another platform regenerates the derivation.
        planned["background"]["identity_sha256"] = archived["background"]["identity_sha256"]
    require(
        equivalent_design(request["looks"], expected)
        and len(expected) == len(set(request["presentations"])),
        "Appearance controls differ",
    )
    size = dimensions(request["resolution"])
    width, height = parent["recipe"]["simulation"]["resolution"]
    require(size[0] * height == size[1] * width, "Appearance and material aspects differ")
    require(set(receipt["looks"]) == set(request["presentations"]), "Appearance view set differs")
    for look in request["looks"]:
        name = look["id"]
        required = {
            f"{name}/poster.png",
            f"{name}/poster-linear.npy",
            f"{name}/preview.png",
            f"{name}/background.json",
        }
        require(required <= files.keys(), "Missing appearance output")
        require(
            read(folder / name / "background.json") == look["background"],
            "Background record differs",
        )
        require(_png_size(folder / name / "poster.png") == size, "Appearance image size differs")
        pixels = np.load(folder / name / "poster-linear.npy", allow_pickle=False)
        require(pixels.shape == (size[1], size[0], 3), "Appearance linear raster differs")
        require(
            receipt["looks"][name]["image_balance"]
            == image_balance(pixels, look["background"]["ground_linear"]),
            "Appearance balance differs",
        )
    return request, receipt


def verify_study(folder):
    return _verify_resolved_study(folder, version=VERSION, resolve=presentation)


def _render_resolved_study(case, output, *, names, resolution, version, resolve, interpretation):
    """Shared immutable-material capture, locking, provenance and verification."""
    case, output = Path(case).resolve(), Path(output).resolve()
    require(
        output != case and not output.is_relative_to(case), "Keep studies outside physical archives"
    )
    parent, physical = verify_run(case)
    names = list(names)
    require(names and len(names) == len(set(names)), "Use distinct known presentations")
    presentations = [resolve(parent, name) for name in names]
    size = dimensions(
        parent["recipe"]["render"]["still_resolution"] if resolution is None else resolution
    )
    w, h = parent["recipe"]["simulation"]["resolution"]
    require(size[0] * h == size[1] * w, "Appearance and material aspects differ")
    code = runtime_identity()
    request = {
        "version": version,
        "parent_identity_sha256": physical["identity_sha256"],
        "physical_state_sha256": physical["physical_state_sha256"],
        "source_sha256": parent["source"]["sha256"],
        "seed": parent["source"]["seed"],
        "chromatic_count": parent["recipe"]["chromatic_count"],
        "presentations": names,
        "looks": presentations,
        "resolution": size,
        "code": code,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "interpretation": interpretation,
    }
    identity = hashlib.sha256(encoded(request)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "request.json").exists():
            require(
                read(output / "request.json") == request,
                "Existing appearance study has different inputs",
            )
            _verify_resolved_study(output, version=version, resolve=resolve)
            return output
        require(
            not any(p.name != ".lock" for p in output.iterdir()), "Appearance output must be empty"
        )
        write(output / "request.json", request)
        write(output / "receipt.json", {"complete": False, "identity_sha256": identity})
        for name in ("request", "receipt"):
            shutil.copyfile(case / f"{name}.json", output / f"parent-{name}.json")
        files = {
            name: artifact(output / name) for name in ("parent-request.json", "parent-receipt.json")
        }
        for package, paths in code.items():
            for name in paths:
                target = output / "inputs/code/tools" / package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT.parent / package / name, target)
        verify_code(output / "inputs/code", code)
        with np.load(case / "final.npz", allow_pickle=False) as archive:
            fields = {name: archive[name] for name in archive.files}
        require(
            field_digest(fields) == physical["physical_state_sha256"],
            "Parent material changed while loading",
        )
        for value in fields.values():
            value.setflags(write=False)
        looks = {}
        for look in request["looks"]:
            name = look["id"]
            folder = output / name
            folder.mkdir()
            spectral = (
                parent.get("spectral") if look["surface"]["optics_model"] == "spectral" else None
            )
            with Surface(look["surface"], parent["palette"], spectral=spectral) as surface:
                pixels = surface.render(fields, size=tuple(size), **look["camera"])
                hardware = copy.deepcopy(surface.metadata)
            write_png(folder / "poster.png", pixels)
            write_array(folder / "poster-linear.npy", pixels)
            write_png(folder / "preview.png", _preview(pixels), depth=8)
            write(folder / "background.json", look["background"])
            for filename in ("poster.png", "poster-linear.npy", "preview.png", "background.json"):
                files[f"{name}/{filename}"] = artifact(folder / filename)
            looks[name] = {
                "image_balance": image_balance(pixels, look["background"]["ground_linear"]),
                "renderer": hardware,
            }
            print(f"APPEARANCE {request['seed']} {name}", flush=True)
        require(
            field_digest(fields) == physical["physical_state_sha256"],
            "Appearance pass modified physical state",
        )
        require(runtime_identity() == code, "Appearance renderer changed while rendering")
        write(
            output / "receipt.json",
            {
                "complete": True,
                "identity_sha256": identity,
                "artifacts": files,
                "looks": looks,
                "parent_archive": str(case),
            },
        )
        _verify_resolved_study(output, version=version, resolve=resolve)
    return output


def render_study(case, output, *, names=None, resolution=None):
    return _render_resolved_study(
        case,
        output,
        names=list(PRESETS) if names is None else names,
        resolution=resolution,
        version=VERSION,
        resolve=presentation,
        interpretation=(
            "unchanged pigment and material state; changed ground, lighting or display relief; "
            "pigment backing unchanged"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--looks", nargs="+", choices=list(PRESETS))
    parser.add_argument("--resolution", nargs=2, type=int)
    args = parser.parse_args()
    print(render_study(args.case, args.output, names=args.looks, resolution=args.resolution))


if __name__ == "__main__":
    main()
