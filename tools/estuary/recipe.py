"""Strict, versioned controls shared by previews and production Estuary renders.

Recipes may omit fields to use documented defaults. Unknown fields are errors;
the fully resolved recipe, rather than its shorthand input, identifies a render.
The simulation grid and source-time cadence are independent of output size.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .optics import Material

MAX_RECIPE_BYTES = 64 * 1024
STRATA_PROFILE_VERSION = "strata-profile-v1"
STRATA_PROFILE_DEFAULTS = {
    "version": STRATA_PROFILE_VERSION,
    "main_width_scale": 1.0,
    "fine_width_scale": 1.0,
    "accent_width_scale": 1.0,
}
DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "simulation": {
        "resolution": [1024, 768],
        "steps": 3600,
        "domain_scale": 1.0,
        "flow_strength": 0.70,
        "carrier_velocity": [0.0, 0.0],
        "stir_radius": 0.20,
        "pair_swirl": 0.20,
        "brush_radius": 0.035,
        "deposition": 1.0,
        "pigment_weights": [1.0, 1.0, 0.35],
        "initial_load": 0.0,
        "initial_pattern": "pools",
        "load_radius": 0.12,
        "fade": 0.70,
    },
    "projection": {"fill": 0.78, "rotation_degrees": 0.0},
    "optics": json.loads(json.dumps(asdict(Material()))),
    "render": {"resolution": [1600, 1200], "frames": 301, "fps": 10, "temporal_samples": 1},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _merge(values: Any, defaults: dict[str, Any], name: str) -> dict[str, Any]:
    _require(type(values) is dict, f"{name} must be an object")
    _require(
        not (set(values) - set(defaults)), f"Unknown {name} fields: {set(values) - set(defaults)}"
    )
    result = copy.deepcopy(defaults)
    for key, value in values.items():
        if type(defaults[key]) is dict:
            result[key] = _merge(value, defaults[key], f"{name}.{key}")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _integer(value: Any, name: str, low: int, high: int) -> int:
    _require(
        type(value) is int and low <= value <= high, f"{name} must be an integer in [{low}, {high}]"
    )
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    _require(type(value) in (int, float), f"{name} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite range") from exc
    _require(math.isfinite(result) and low <= result <= high, f"{name} must be in [{low}, {high}]")
    return result


def _vector(value: Any, name: str, length: int, low: float, high: float) -> list[float]:
    _require(
        type(value) in (tuple, list) and len(value) == length, f"{name} needs {length} numbers"
    )
    return [_number(item, name, low, high) for item in value]


def _resolution(value: Any, name: str, maximum: int, max_pixels: int = 16_777_216) -> list[int]:
    _require(type(value) in (tuple, list) and len(value) == 2, f"{name} needs width and height")
    width = _integer(value[0], f"{name}.width", 128, maximum)
    height = _integer(value[1], f"{name}.height", 96, maximum)
    _require(width * height <= max_pixels, f"{name} exceeds the pixel budget")
    _require(width % 2 == 0 and height % 2 == 0, f"{name} dimensions must be even for video")
    _require(0.2 <= width / height <= 5, f"{name} aspect must be in [0.2, 5]")
    return [width, height]


def validate_strata_profile(value: Any) -> dict[str, Any] | None:
    """Normalize opt-in band widths without adding keys to legacy recipes."""
    if value is None:
        return None
    result = _merge(value, STRATA_PROFILE_DEFAULTS, "simulation.strata_profile")
    _require(result["version"] == STRATA_PROFILE_VERSION, "Unsupported strata profile version")
    for key, low in (
        ("main_width_scale", 0.25),
        ("fine_width_scale", 0),
        ("accent_width_scale", 0.25),
    ):
        result[key] = _number(result[key], f"strata_profile.{key}", low, 2)
    return None if result == STRATA_PROFILE_DEFAULTS else result


def validate_recipe(value: Any) -> dict[str, Any]:
    """Resolve defaults and reject unsafe, misspelled or incompatible controls."""
    supplied = copy.deepcopy(value)
    profile_value, initial_image = None, False
    if type(supplied) is dict:
        if type(supplied.get("simulation")) is dict:
            profile_value = supplied["simulation"].pop("strata_profile", None)
        if type(supplied.get("render")) is dict and "initial_image" in supplied["render"]:
            initial_image = supplied["render"].pop("initial_image")
            _require(type(initial_image) is bool, "render.initial_image must be boolean")
    recipe = _merge(supplied, DEFAULTS, "recipe")
    _integer(recipe["schema_version"], "schema_version", 1, 1)
    simulation, projection, optics, render = (
        recipe["simulation"],
        recipe["projection"],
        recipe["optics"],
        recipe["render"],
    )
    simulation["resolution"] = _resolution(
        simulation["resolution"], "simulation.resolution", 8192, max_pixels=40_000_000
    )
    simulation["steps"] = _integer(simulation["steps"], "simulation.steps", 360, 40_000)
    limits = {
        "domain_scale": (1, 2.5),
        "flow_strength": (0, 8),
        "stir_radius": (0.02, 2),
        "pair_swirl": (0, 8),
        "brush_radius": (0.002, 1),
        "deposition": (0, 100),
        "initial_load": (0, 10),
        "load_radius": (0.001, 2),
        "fade": (0, 20),
    }
    for key, (low, high) in limits.items():
        simulation[key] = _number(simulation[key], f"simulation.{key}", low, high)
    _require(
        type(simulation["initial_pattern"]) is str
        and simulation["initial_pattern"] in ("pools", "strata"),
        "simulation.initial_pattern must be pools or strata",
    )
    profile = validate_strata_profile(profile_value)
    if profile is not None:
        _require(
            simulation["initial_pattern"] == "strata",
            "strata_profile requires initial_pattern=strata",
        )
        simulation["strata_profile"] = profile
    simulation["pigment_weights"] = _vector(
        simulation["pigment_weights"], "simulation.pigment_weights", 3, 0, 10
    )
    simulation["carrier_velocity"] = _vector(
        simulation["carrier_velocity"], "simulation.carrier_velocity", 2, -8, 8
    )
    projection["fill"] = _number(projection["fill"], "projection.fill", 0.05, 0.98)
    projection["rotation_degrees"] = _number(
        projection["rotation_degrees"], "projection.rotation_degrees", -360, 360
    )
    _require(
        type(optics["pigments_srgb"]) in (list, tuple) and len(optics["pigments_srgb"]) == 3,
        "optics.pigments_srgb needs three RGB colors",
    )
    optics["pigments_srgb"] = [
        _vector(color, "optics.pigments_srgb", 3, 0, 1) for color in optics["pigments_srgb"]
    ]
    optics["substrate_srgb"] = _vector(optics["substrate_srgb"], "optics.substrate_srgb", 3, 0, 1)
    optics["scattering"] = _vector(optics["scattering"], "optics.scattering", 3, 1e-6, 1e3)
    optics["grain_frequency"] = _vector(
        optics["grain_frequency"], "optics.grain_frequency", 2, 1, 16384
    )
    optics["layer_scale"] = _number(optics["layer_scale"], "optics.layer_scale", 0, 1e4)
    optics["grain"] = _number(optics["grain"], "optics.grain", 0, 0.08)
    Material(**optics)
    render["resolution"] = _resolution(render["resolution"], "render.resolution", 8192)
    render["frames"] = _integer(render["frames"], "render.frames", 2, 1801)
    render["fps"] = _integer(render["fps"], "render.fps", 1, 60)
    render["temporal_samples"] = _integer(
        render["temporal_samples"], "render.temporal_samples", 1, 8
    )
    if initial_image:
        render["initial_image"] = True
    _require(
        simulation["steps"] % (render["frames"] - 1) == 0,
        "simulation.steps must be divisible by render.frames - 1",
    )
    _require(
        render["temporal_samples"] <= simulation["steps"] // (render["frames"] - 1),
        "render.temporal_samples must not exceed the canonical steps between frames",
    )
    sw, sh = simulation["resolution"]
    rw, rh = render["resolution"]
    _require(sw * rh == sh * rw, "Simulation and output must have exactly the same aspect")
    return recipe


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate recipe key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite recipe value: {value}")


def read_recipe(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    _require(0 < path.stat().st_size <= MAX_RECIPE_BYTES, "Recipe size exceeds bounds")
    with path.open("rb") as stream:
        content = stream.read(MAX_RECIPE_BYTES + 1)
    _require(len(content) <= MAX_RECIPE_BYTES, "Recipe size exceeds bounds")
    return validate_recipe(
        json.loads(content, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    )


def canonical_bytes(recipe: dict[str, Any]) -> bytes:
    """Deterministic complete recipe serialization, including all implicit defaults."""
    return (
        json.dumps(validate_recipe(recipe), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def recipe_sha256(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(recipe)).hexdigest()
