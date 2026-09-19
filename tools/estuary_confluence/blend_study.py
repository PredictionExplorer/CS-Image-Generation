"""Compare spectral paint interpretations on one verified material history.

No transport is replayed, no pigment is added, and no color is invented by a
postprocess. The baseline, intimate mixture and thinner ordered glaze preserve
the parent's camera and ground. Raised paint changes the presentation of the
actual heightfield, not its archived geometry. These are authored appearances,
not measured pigment formulations or a simulation of extra brushwork.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from tools.estuary_studio.common import require

from .appearance import _render_resolved_study, _verify_resolved_study
from .backgrounds import NAMES, generate_background, validate_background
from .run import equivalent_design, validate_recipe
from .surface import validate_config

VERSION = "confluence-blend-study-v1"
PRESETS = {
    "ordered": ("Ordered glaze", {}),
    "intimate": ("Intimate spectral blend", {"mode": "homogeneous", "mix_control": 0.0}),
    "thin-glaze": (
        "Translucent glaze",
        {"glaze_min_mass_ratio": 0.10, "layer_scale": 16.0},
    ),
    "ink": (
        "Ink density",
        {"glaze_min_mass_ratio": 0.02, "layer_scale": 4.0},
    ),
    "relief": (
        "Raised paint",
        {
            "height_scale": 4.0,
            "glaze_relief_strength": 0.35,
            "key_elevation_degrees": 16.0,
            "key_strength": 4.1,
            "ambient": 0.16,
            "fill_strength": 0.10,
        },
    ),
}


def _parent_background(parent, base):
    """Retain exact archived ground values; reject silently changed backgrounds."""
    background = parent.get("background")
    if background is not None:
        background = validate_background(background, parent["palette"])
    else:
        # Early proof archives predate the optional recipe-level background.
        # Resolve only an existing named ground; arbitrary colors are not snapped.
        background = next(
            (
                candidate
                for name in NAMES
                if equivalent_design(
                    (candidate := generate_background(name, parent["palette"]))["ground_srgb"],
                    base["ground_srgb"],
                )
            ),
            None,
        )
    require(background is not None, "Blend studies need an archived or named ground")
    require(
        equivalent_design(background["ground_srgb"], base["ground_srgb"]),
        "Blend study ground differs from the parent painting",
    )
    return background


def presentation(parent, name):
    """Resolve an explicitly labeled appearance of the parent's unchanged paint."""
    require(type(name) is str and name in PRESETS, "Unknown blend study")
    require("layered" in parent["surface_configs"], "Blend studies need a layered parent view")
    base = copy.deepcopy(parent["surface_configs"]["layered"])
    require(
        base["optics_model"] == "spectral" and base["finish"] == "glazed",
        "Blend studies require a spectral glazed parent",
    )
    label, overrides = PRESETS[name]
    render = parent["recipe"]["render"]
    return {
        "id": name,
        "name": label,
        "background": _parent_background(parent, base),
        "surface": validate_config({**base, **overrides}),
        "camera": {
            "tilt_degrees": render["still_tilt_degrees"],
            "azimuth_degrees": render["azimuth_end"],
        },
    }


def film_recipe(parent, name):
    """Apply one optical treatment to an ordinary full-trajectory film recipe.

    The returned recipe leaves simulation, source coverage and frame cadence
    intact. A resulting film is a separately verified archive; matching still
    treatments alone never establish that its material is identical.
    """
    look = presentation(parent, name)
    recipe = copy.deepcopy(parent["recipe"])
    recipe["name"] = f"The Estuary · {look['name']}"
    recipe["looks"] = [look["surface"]["mode"]]
    recipe["surface"] = look["surface"]
    return validate_recipe(recipe)


def verify_study(folder):
    """Verify media, source, material, resolved controls and captured runtime."""
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
            "unchanged pigment, spectral coefficients and material history; "
            "ordered or intimate spectral reflection, optical thickness and displayed relief; "
            "parent ground, backing and still camera retained"
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
