"""Seeded palettes with bounded absorption for actual intermingled paint.

Version one preserves perceptual separation of pure colors, but gamut-boundary
RGB values can imply almost infinite absorption in the RGB Kubelka--Munk model.
This version preserves its seed choices and reduces only OKLCH chroma until every
pigment's linear reflectance is at least .003. Neither RGB clipping nor a shader
change is involved. The bound limits K/S to about 165.67 in every color channel.

The authored relative scattering coefficients were selected on real completed
paint, including thin light and medium strands contaminated by dark pigment.
They are not measured coefficients of manufactured pigments. Subtractive mixtures
may still lose chroma; the contract does not promise every mixture stays vivid.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from tools.estuary.optics import (
    absorption_over_scattering,
    linear_to_srgb,
    srgb_to_linear,
)
from tools.estuary_depth import pattern_palette as v1

VERSION = "pattern-palette-v2"
LINEAR_REFLECTANCE_FLOOR = 0.003
SCATTERING = (0.2, 8.0, 3.0)  # dark, light, medium
GAMUT_STEPS = 40
_FLOOR_MARGIN = 1e-10


def _round(values):
    return [round(float(value), 12) for value in values]


def _identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _inset_color(color):
    if min(srgb_to_linear(color)) >= LINEAR_REFLECTANCE_FLOOR:
        return list(color), 1.0
    lightness, a, b = v1.srgb_to_oklab(color)
    low, high = 0.0, 1.0
    threshold = LINEAR_REFLECTANCE_FLOOR + _FLOOR_MARGIN
    for _ in range(GAMUT_STEPS):
        fraction = (low + high) / 2
        linear = v1.oklab_to_linear_srgb((lightness, a * fraction, b * fraction))
        if all(threshold <= value <= 1 for value in linear):
            low = fraction
        else:
            high = fraction
    linear = v1.oklab_to_linear_srgb((lightness, a * low, b * low))
    return _round(linear_to_srgb(linear)), low


def make_palette(seed: str | int) -> dict[str, Any]:
    """Keep v1 seed geometry/color choices; change only pigment absorption and S."""
    parent = v1.make_palette(seed)
    result = copy.deepcopy(parent)
    result.pop("identity_sha256")
    result["version"] = VERSION
    inset = [_inset_color(color) for color in parent["optics"]["pigments_srgb"]]
    colors, fractions = [item[0] for item in inset], [item[1] for item in inset]
    result["optics"].update(pigments_srgb=colors, scattering=list(SCATTERING))
    labs = [v1.srgb_to_oklab(color) for color in colors]
    lchs = [
        (lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360)
        for lightness, a, b in labs
    ]
    pairs = ((0, 1), (0, 2), (1, 2))
    distances = [math.dist(labs[a], labs[b]) for a, b in pairs]
    hue_distances = [abs((lchs[a][2] - lchs[b][2] + 180) % 360 - 180) for a, b in pairs]
    linear = srgb_to_linear(colors)
    if (
        linear.min() < LINEAR_REFLECTANCE_FLOOR
        or min(distances) < v1.MIN_DISTANCE
        or min(hue_distances) < v1.MIN_HUE_SEPARATION
        or any(lch[1] < minimum for lch, minimum in zip(lchs, v1.MIN_CHROMA, strict=True))
    ):
        raise ValueError("Bounded-absorption palette failed its declared separation contract")
    result["metadata"].update(
        oklab=[_round(value) for value in labs],
        oklch=[_round(value) for value in lchs],
        pairwise_oklab_distance=_round(distances),
        pairwise_hue_separation_degrees=_round(hue_distances),
        gamut_mapping={
            "method": "fixed-lightness-and-hue-chroma-bisection-into-inner-linear-sRGB",
            "iterations": GAMUT_STEPS,
            "linear_reflectance_floor": LINEAR_REFLECTANCE_FLOOR,
            "rounding_safety_margin": _FLOOR_MARGIN,
            "chroma_fraction_of_v1": _round(fractions),
        },
        linear_pigment_reflectance=[_round(row) for row in linear],
        absorption_over_scattering=[_round(row) for row in absorption_over_scattering(linear)],
        scattering_policy=(
            "Authored relative dark/light/medium S=[0.2,8,3], qualified on completed folded "
            "paint. Not measured pigment coefficients."
        ),
        randomness=(
            "All authored seed choices retained from pattern-palette-v1; no new random draws"
        ),
        seed_stream_provenance={
            "version": parent["version"],
            "palette_identity_sha256": parent["identity_sha256"],
            "randomness": parent["metadata"]["randomness"],
        },
    )
    result["metadata"]["thresholds"]["minimum_linear_pigment_reflectance"] = (
        LINEAR_REFLECTANCE_FLOOR
    )
    result["identity_sha256"] = _identity(result)
    return result


def verify_palette(record: dict[str, Any]) -> dict[str, Any]:
    """Regenerate the palette and its v1 provenance, rejecting altered controls."""
    if type(record) is not dict or record.get("version") != VERSION or "seed" not in record:
        raise ValueError("Unknown palette record")
    if record != make_palette(record["seed"]):
        raise ValueError("Palette differs from its deterministic seed contract")
    return record
