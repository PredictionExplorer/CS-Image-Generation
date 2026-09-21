"""Controlled comparisons of seeded starting shapes and separated pigment colors.

Each orbit seed uses one palette across all ten patterns. Transport, source clock,
paint budget and studio stay fixed; only the spatial starting composition changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tools.estuary.initial_patterns import PATTERNS
from tools.estuary.initial_patterns import VERSION as PATTERN_VERSION
from tools.estuary.recipe import validate_recipe
from tools.estuary_depth.filament_motion import FILM_RESOLUTION, FORMATION_FRAMES, FPS
from tools.estuary_depth.filament_studies import make_depth_recipe, make_paint_recipe
from tools.estuary_depth.pattern_palette import make_palette
from tools.estuary_depth.render import recipe as validate_depth_recipe

VERSION = "pattern-studies-v1"
PAINT_RUNTIME_EXTENSIONS = ("initial_patterns.py",)
REFERENCE_OPTION = "lacuna-banks"
DEFAULT_OPTION = "folded-sash"


@dataclass(frozen=True)
class PatternSpec:
    label: str
    description: str
    material_variant: str


_LABELS = {
    "lacuna-banks": "Lacuna banks",
    "interlocking-crescents": "Interlocking crescents",
    "braided-ribbons": "Braided ribbons",
    "split-fan": "Split fan",
    "river-confluence": "River confluence",
    "broken-terraces": "Broken terraces",
    "meandering-fault": "Meandering fault",
    "folded-sash": "Folded sash",
    "asymmetric-rosette": "Open rosette",
    "branching-channels": "Branching channels",
}

OPTIONS = MappingProxyType(
    {key: PatternSpec(_LABELS[key], description, key) for key, description in PATTERNS.items()}
)


def _option(option):
    if type(option) is not str or option not in OPTIONS:
        raise ValueError("Unknown starting-pattern study")
    return OPTIONS[option]


def make_formation_recipe(seed: str | int, option: str) -> dict[str, Any]:
    """Keep the native 6K/full-trajectory model and certify the seeded initial field."""
    _option(option)
    palette = make_palette(seed)
    result = make_paint_recipe(seed, "control")
    simulation = result["simulation"]
    simulation.update(
        initial_pattern="composition",
        initial_design={
            "version": PATTERN_VERSION,
            "pattern": option,
            "seed": palette["seed"],
        },
        # Continuing deposition treats every pigment equally. Each motif starts
        # with the same unit-load partition, scaled by the inherited .6 load.
        pigment_weights=[1.0, 1.0, 1.0],
    )
    result["optics"] = palette["optics"]
    result["render"].update(
        resolution=list(FILM_RESOLUTION),
        frames=FORMATION_FRAMES,
        fps=FPS,
        temporal_samples=4,
        initial_image=True,
    )
    return validate_recipe(result)


def make_photo_recipe(seed: str | int, option: str, *, master: bool = False) -> dict[str, Any]:
    if type(master) is not bool:
        raise ValueError("master must be boolean")
    return make_depth_recipe(seed, _option(option).label, proof=not master)


def make_motion_recipe(seed: str | int, option: str) -> dict[str, Any]:
    result = make_photo_recipe(seed, option)
    result["render"].update(resolution=list(FILM_RESOLUTION), samples=32)
    camera = result["camera"]
    tilt, azimuth = camera["tilt_degrees"], camera["azimuth_degrees"]
    camera.update(orbit_start=[max(0, tilt - 4), azimuth - 4], orbit_end=[tilt, azimuth])
    return validate_depth_recipe(result)
