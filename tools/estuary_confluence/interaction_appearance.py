"""Re-shade verified interaction histories without replaying their paint motion."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

from tools.estuary_studio.common import require

from .appearance import _render_resolved_study, _verify_resolved_study
from .backgrounds import validate_background
from .surface import validate_config

VERSION = "contact-appearance-v1"


@dataclass(frozen=True)
class Presentation:
    name: str
    setup: str = "original"
    silk: float = 1.0
    grain: float = 1.0
    packing: bool = False
    contrast: float | None = None


# Named camera/light arrangements are independent of the material response.
# A comparison pair selects the same setup and changes only interaction controls.
SETUPS = {
    "original": ({}, {}),
    "raking": (
        {"key_azimuth_degrees": 35.0},
        {"tilt_degrees": 20.0, "azimuth_degrees": 35.0},
    ),
    "sculpted": (
        {
            "height_scale": 4.0,
            "glaze_relief_strength": 0.2,
            "key_elevation_degrees": 18.0,
            "key_azimuth_degrees": 35.0,
            "key_strength": 4.3,
            "ambient": 0.12,
        },
        {"tilt_degrees": 20.0, "azimuth_degrees": 35.0},
    ),
    "impasto": (
        {
            "height_scale": 60.0,
            "glaze_relief_strength": 0.2,
            "key_elevation_degrees": 18.0,
            "key_azimuth_degrees": 35.0,
            "key_strength": 4.3,
            "ambient": 0.12,
        },
        {"tilt_degrees": 20.0, "azimuth_degrees": 35.0},
    ),
    "glint": (
        {"key_elevation_degrees": 65.0, "key_azimuth_degrees": 215.0, "key_strength": 1.5},
        {"tilt_degrees": 25.0, "azimuth_degrees": 35.0},
    ),
}
PRESETS = {
    "control": Presentation("Control", silk=0, grain=0),
    "silk": Presentation("Satin seams", grain=0),
    "silk-grain": Presentation("Satin + grain"),
    "raking-control": Presentation("Raking view · Control", "raking", silk=0, grain=0),
    "raking-silk": Presentation("Raking view · Satin seams", "raking", grain=0),
    "raking-silk-grain": Presentation("Raking view · Satin + grain", "raking"),
    "packing": Presentation("Contact relief", packing=True),
    "raking-packing": Presentation("Raking view · Contact relief", "raking", packing=True),
    "sculpted-control": Presentation("Raised paint · Control", "sculpted", silk=0, grain=0),
    "sculpted-packing": Presentation("Raised paint · Contact relief", "sculpted", packing=True),
    "glint-control": Presentation("Reflected light · Control", "glint", silk=0, grain=0),
    "glint-silk": Presentation("Reflected light · Satin seams", "glint", grain=0),
    "glint-grain": Presentation("Reflected light · Satin + grain", "glint"),
    "glint-packing": Presentation("Reflected light · Contact relief", "glint", packing=True),
    "impasto-control": Presentation("Thick paint · Control", "impasto", silk=0, grain=0),
    "impasto-packing": Presentation("Thick paint · Contact relief", "impasto", packing=True),
    "impasto-detailed": Presentation(
        "Thick paint · Fine contact", "impasto", packing=True, contrast=4
    ),
    "impasto-mineral": Presentation(
        "Thick paint · Mineral contact", "impasto", packing=True, contrast=8
    ),
}


def presentation(parent, name):
    require(name in PRESETS, "Unknown interaction appearance")
    require(parent["recipe"]["simulation"].get("interaction") is not None, "History is required")
    require(parent.get("background") is not None, "Use a recorded ground for matched studies")
    background = validate_background(parent["background"], parent["palette"])
    surface = copy.deepcopy(parent["recipe"]["surface"])
    require(surface["grain_um"] == 0, "Stationary support grain must remain disabled")
    preset = PRESETS[name]
    surface.update(
        mode="layered",
        mix_control=1.0,
        interaction={"silk_strength": preset.silk, "grain_strength": preset.grain},
    )
    camera = {
        "tilt_degrees": parent["recipe"]["render"]["still_tilt_degrees"],
        "azimuth_degrees": parent["recipe"]["render"]["azimuth_end"],
    }
    surface_overrides, camera_overrides = SETUPS[preset.setup]
    surface.update(surface_overrides)
    camera.update(camera_overrides)
    if preset.packing:
        require(parent["recipe"]["simulation"]["substrate_um"] == 0, "Packing needs flat support")
        surface["interaction"]["packing_strength"] = 1.0
    if preset.contrast is not None:
        surface["interaction"]["grain_contrast"] = preset.contrast
    return {
        "id": name,
        "name": preset.name,
        "background": background,
        "surface": validate_config(surface),
        "camera": camera,
    }


def verify_study(folder):
    return _verify_resolved_study(folder, version=VERSION, resolve=presentation)


def render_study(case, output, *, names=None, resolution=None):
    return _render_resolved_study(
        case,
        output,
        names=list(PRESETS) if names is None else names,
        resolution=resolution,
        version=VERSION,
        resolve=presentation,
        interpretation=(
            "Same complete paint and interaction history in every view. "
            "Original or explicitly paired raking camera/light; "
            "no new transport simulation or pigment. "
            "Appearance reads frozen recorded fabric and aggregation. Packing views reconstruct "
            "conservative shallow relief from this state; raised-paint views explicitly change "
            "the authored height scale. Exact renderer code is archived."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--looks", nargs="+", choices=PRESETS)
    args = parser.parse_args()
    print(render_study(args.case, args.output, names=args.looks))


if __name__ == "__main__":
    main()
