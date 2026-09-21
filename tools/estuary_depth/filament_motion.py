"""Canonical image and full-formation film controls for every fine-fold option.

Lighting options reuse the Control pigment simulation. All source motion finishes
before the existing film editor dissolves into a short examination of the frozen
relief. The camera finishes at the paired photograph's pose; video sampling and
compression do not promise pixel identity with the higher-quality still image.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tools.estuary.recipe import validate_recipe
from tools.estuary_depth.filament_studies import VARIANTS, make_depth_recipe, make_paint_recipe
from tools.estuary_depth.render import recipe as validate_depth_recipe

VERSION = "filament-motion-v1"
FPS = 24
FORMATION_FRAMES = 721
MOTION_FRAMES = 96
FILM_RESOLUTION = (1920, 1440)
MOTION_SAMPLES = 32


@dataclass(frozen=True)
class OptionSpec:
    label: str
    description: str
    material_variant: str
    relief_mm: float | None = None
    light_elevation_degrees: float | None = None


OPTIONS = MappingProxyType(
    {
        **{key: OptionSpec(value.label, value.description, key) for key, value in VARIANTS.items()},
        "light-flat": OptionSpec(
            "Flat pigment",
            "The same Control paint with no authored relief.",
            "control",
            relief_mm=0,
        ),
        "light-shallow": OptionSpec(
            "Shallow folds",
            "The same Control paint with 6 mm authored relief.",
            "control",
            relief_mm=6,
        ),
        "light-deep": OptionSpec(
            "Deeper folds",
            "The same Control paint with 24 mm authored relief.",
            "control",
            relief_mm=24,
        ),
        "light-raking": OptionSpec(
            "Lower light",
            "The same Control paint under a lower key light.",
            "control",
            light_elevation_degrees=25,
        ),
    }
)


def _option(option: str) -> OptionSpec:
    if type(option) is not str or option not in OPTIONS:
        raise ValueError("Unknown fine-fold film option")
    return OPTIONS[option]


def make_formation_recipe(seed: str | int, option: str) -> dict[str, Any]:
    """Full source interval; all lighting options return the same Control recipe."""
    result = make_paint_recipe(seed, _option(option).material_variant)
    result["render"].update(
        resolution=list(FILM_RESOLUTION), frames=FORMATION_FRAMES, fps=FPS, temporal_samples=4
    )
    return validate_recipe(result)


def make_photo_recipe(seed: str | int, option: str, *, master: bool = False) -> dict[str, Any]:
    """Return the paired 2048-pixel photograph, or an explicitly requested 4K master."""
    if type(master) is not bool:
        raise ValueError("master must be boolean")
    spec = _option(option)
    result = make_depth_recipe(seed, spec.label, proof=not master)
    if spec.relief_mm is not None:
        result["relief_mm"] = spec.relief_mm
    if spec.light_elevation_degrees is not None:
        result["lighting"]["elevation_degrees"] = spec.light_elevation_degrees
    return validate_depth_recipe(result)


def make_motion_recipe(seed: str | int, option: str) -> dict[str, Any]:
    """Examine frozen paint over four seconds, ending at the paired still's camera."""
    result = make_photo_recipe(seed, option)
    result["render"].update(resolution=list(FILM_RESOLUTION), samples=MOTION_SAMPLES)
    camera = result["camera"]
    tilt, azimuth = camera["tilt_degrees"], camera["azimuth_degrees"]
    camera.update(orbit_start=[max(0, tilt - 4), azimuth - 4], orbit_end=[tilt, azimuth])
    return validate_depth_recipe(result)
