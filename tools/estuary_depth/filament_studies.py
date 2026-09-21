"""Matched studies of the thin folds in the original ultramarine painting.

The control preserves the full-resolution paint simulation. Individual variants
change the starting bands, the starting pools, or one part of the current. Depth
recipes photograph that paint with the original Folded Tide lighting and relief.
Other seeds keep the same navy, ivory and oxide color roles with small, explicitly
bounded variations; the reference seed retains its original palette exactly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tools.estuary.recipe import read_recipe, validate_recipe
from tools.estuary_depth.render import recipe as validate_depth_recipe

VERSION = "filament-studies-v1"
REFERENCE_SEED = "0xbc53af1cd380"
PAINT_RECIPE = Path(__file__).resolve().parents[1] / "estuary/recipes/ultramarine.json"
DEPTH_RECIPE = Path(__file__).resolve().parent / "recipes-finish/21-deep.json"
PALETTE_OFFSETS = ((0.012, 0.018, 0.055), (0.025, 0.025, 0.035), (0.065, 0.035, 0.025))


@dataclass(frozen=True)
class StudySpec:
    label: str
    description: str
    main_width_scale: float | None = None
    fine_width_scale: float | None = None
    pair_swirl: float | None = None
    carrier_velocity: tuple[float, float] | None = None
    pool_radius: float | None = None
    resolution: tuple[int, int] | None = None


VARIANTS = MappingProxyType(
    {
        "control": StudySpec("Original folds", "The original bands and current."),
        "plain-bands": StudySpec(
            "One ivory band",
            "Remove the two fine ivory starting bands; keep the oxide accent.",
            fine_width_scale=0.0,
        ),
        "fine-bands": StudySpec(
            "Finer starting bands",
            "Halve the width of the two fine ivory bands.",
            fine_width_scale=0.5,
        ),
        "narrow-main": StudySpec(
            "Narrow ivory band", "Reduce the main ivory band's width.", main_width_scale=0.65
        ),
        "stronger-folds": StudySpec(
            "Stronger folding", "Increase the rotation driven by body pairs.", pair_swirl=1.25
        ),
        "calmer-current": StudySpec(
            "Calmer current", "Reduce the steady carrier current.", carrier_velocity=(0.6, 0.06)
        ),
        "three-pools": StudySpec(
            "Three pools", "Start with three round pools instead of bands.", pool_radius=0.28
        ),
        "three-broad-pools": StudySpec(
            "Three broad pools", "Give the three starting pools a wider reach.", pool_radius=0.5
        ),
        "three-pools-folds": StudySpec(
            "Broad pools · stronger folding",
            "Combine broad starting pools with stronger pair-driven rotation.",
            pair_swirl=1.25,
            pool_radius=0.5,
        ),
        "coarse-control": StudySpec(
            "Bands · coarser paint grid",
            "The original bands on a 2048 x 1536 material grid; identical final image size.",
            resolution=(2048, 1536),
        ),
        "coarse-three-pools": StudySpec(
            "Broad pools · coarser paint grid",
            "The same broad pools on a 2048 x 1536 material grid; identical final image size.",
            pool_radius=0.5,
            resolution=(2048, 1536),
        ),
    }
)


def _seed_bytes(seed: str | int) -> bytes:
    if type(seed) is int:
        number = seed
    elif type(seed) is str and re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,64}", seed):
        number = int(seed, 16)
    else:
        raise ValueError("Seed must be a hexadecimal string or a nonnegative integer")
    if not 0 <= number < 1 << 256:
        raise ValueError("Seed must fit in 256 bits")
    return number.to_bytes(32, "big")


def _digest(seed: bytes, stream: str) -> bytes:
    return hashlib.sha256(VERSION.encode() + b"\0" + stream.encode() + b"\0" + seed).digest()


def make_paint_recipe(seed: str | int, variant: str) -> dict[str, Any]:
    """Return complete controls; only explicitly labeled grid probes reduce material resolution."""
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("Unknown filament study")
    seed_bytes, spec = _seed_bytes(seed), VARIANTS[variant]
    result = read_recipe(PAINT_RECIPE)
    simulation = result["simulation"]
    profile = {
        key: getattr(spec, key)
        for key in ("main_width_scale", "fine_width_scale")
        if getattr(spec, key) is not None
    }
    if profile:
        simulation["strata_profile"] = {"version": "strata-profile-v1", **profile}
    if spec.pair_swirl is not None:
        simulation["pair_swirl"] = spec.pair_swirl
    if spec.carrier_velocity is not None:
        simulation["carrier_velocity"] = list(spec.carrier_velocity)
    if spec.pool_radius is not None:
        simulation.update(initial_pattern="pools", load_radius=spec.pool_radius)
    if spec.resolution is not None:
        simulation["resolution"] = list(spec.resolution)
    if seed_bytes != _seed_bytes(REFERENCE_SEED):
        for pigment, offsets in enumerate(PALETTE_OFFSETS):
            for channel, bound in enumerate(offsets):
                bits = _digest(seed_bytes, f"palette/{pigment}/{channel}")[:8]
                signed = 2 * (int.from_bytes(bits, "big") / 2**64) - 1
                result["optics"]["pigments_srgb"][pigment][channel] += signed * bound
    result["render"].update(resolution=[2048, 1536], initial_image=True)
    return validate_recipe(result)


def make_depth_recipe(seed: str | int, name: str, *, proof: bool = True) -> dict[str, Any]:
    """Resolve the reference studio completely; proof changes only output quality."""
    if type(proof) is not bool:
        raise ValueError("proof must be boolean")
    seed_bytes = _seed_bytes(seed)
    result = json.loads(DEPTH_RECIPE.read_text())
    result["name"] = name
    render = result["render"]
    render["seed"] = (
        31037
        if seed_bytes == _seed_bytes(REFERENCE_SEED)
        else int.from_bytes(_digest(seed_bytes, "cycles")[:4], "big") & (2**31 - 1)
    )
    render.update(resolution=[2048, 1536] if proof else [3840, 2880], samples=128 if proof else 256)
    return validate_depth_recipe(result)
