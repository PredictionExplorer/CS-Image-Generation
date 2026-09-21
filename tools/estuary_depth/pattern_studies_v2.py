"""Opt-in optical calibration of the unchanged version-one starting patterns.

The complete formation recipe is inherited before replacing its optical controls.
Geometry, seed-derived placement, transport, feeding and the source clock stay
unchanged. Version-one recipes remain available through their original family.
"""

from __future__ import annotations

from typing import Any

from tools.estuary.recipe import validate_recipe
from tools.estuary_depth.pattern_palette_v2 import make_palette
from tools.estuary_depth.pattern_studies import DEFAULT_OPTION as DEFAULT_OPTION
from tools.estuary_depth.pattern_studies import OPTIONS as OPTIONS
from tools.estuary_depth.pattern_studies import PAINT_RUNTIME_EXTENSIONS as PAINT_RUNTIME_EXTENSIONS
from tools.estuary_depth.pattern_studies import REFERENCE_OPTION as REFERENCE_OPTION
from tools.estuary_depth.pattern_studies import make_formation_recipe as _make_v1_formation
from tools.estuary_depth.pattern_studies import make_motion_recipe as make_motion_recipe
from tools.estuary_depth.pattern_studies import make_photo_recipe as make_photo_recipe

VERSION = "pattern-studies-v2"


def make_formation_recipe(seed: str | int, option: str) -> dict[str, Any]:
    """Change only seed-derived optics; preserve the version-one paint simulation."""
    result = _make_v1_formation(seed, option)
    result["optics"] = make_palette(seed)["optics"]
    return validate_recipe(result)
