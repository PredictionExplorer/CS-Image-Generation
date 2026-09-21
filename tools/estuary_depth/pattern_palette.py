"""Seeded, separated pigment colors for controlled initial-pattern comparisons.

OKLab uses Björn Ottosson's 2021 linear-sRGB matrices:
https://bottosson.github.io/posts/oklab/ (public-domain reference implementation).
Gamut mapping reduces chroma at fixed OKLCH lightness and hue, using a fixed
32-step bisection. It is deliberately simple; it is not CSS perceptual gamut
mapping. Palette separation describes authored pure-pigment colors, not every
subtractive mixture or a guarantee of artistic quality.

The three channels retain dark/light/medium value roles, but their hues rotate
and exchange places across seeds. Scattering is an authored, bounded response to
lightness. RGB determines K/S, not S independently; these relative coefficients
are not measured physical pigment properties.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from typing import Any

VERSION = "pattern-palette-v1"
MIN_DISTANCE = 0.20
MIN_HUE_SEPARATION = 75.0
MIN_CHROMA = (0.04, 0.05, 0.07)
ROLE_NAMES = ("dark", "light", "medium")
GAMUT_STEPS = 32
HARMONIES = (("triad", (0.0, 120.0, 240.0)), ("split", (0.0, 135.0, 225.0)))


def _seed_bytes(seed: str | int) -> bytes:
    if type(seed) is int:
        number = seed
    elif type(seed) is str and re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,64}", seed):
        number = int(seed, 16)
    else:
        raise ValueError("Seed must be an integer or a full-width-compatible hexadecimal value")
    if not 0 <= number < 1 << 256:
        raise ValueError("Seed must fit in 256 bits")
    return number.to_bytes(32, "big")


def _bits(seed: bytes, stream: str) -> int:
    return int.from_bytes(
        hashlib.sha256(
            VERSION.encode("ascii") + b"\0" + stream.encode("ascii") + b"\0" + seed
        ).digest(),
        "big",
    )


def _unit(seed: bytes, stream: str) -> float:
    # Exactly representable 53-bit fractions; no ambient RNG or order-dependent
    # stream consumption. Every authored choice has its own domain-separated key.
    return (_bits(seed, stream) >> (256 - 53)) / 2**53


def _triple(value, name):
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite values")
    if any(type(x) not in (int, float) for x in value):
        raise ValueError(f"{name} must contain three finite values")
    try:
        result = tuple(float(x) for x in value)
    except OverflowError as error:
        raise ValueError(f"{name} must contain three finite values") from error
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain three finite values")
    return result


def srgb_to_oklab(value) -> tuple[float, float, float]:
    """Convert display sRGB in [0,1] to OKLab using the published D65 matrices."""
    rgb = _triple(value, "sRGB")
    if any(not 0 <= x <= 1 for x in rgb):
        raise ValueError("sRGB must be in [0,1]")
    r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb)
    cone_l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * cone_l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * cone_l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * cone_l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklab_to_linear_srgb(value) -> tuple[float, float, float]:
    """Return unclipped linear RGB; out-of-gamut values remain observable."""
    L, a, b = _triple(value, "OKLab")
    cone_l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * cone_l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * cone_l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * cone_l - 0.7034186147 * m + 1.7076147010 * s,
    )


def gamut_map_oklch(lightness: float, chroma: float, hue: float) -> tuple[float, float, float]:
    """Reduce C only, preserving L/h; the result is strictly inside sRGB."""
    L, C, h = _triple((lightness, chroma, hue), "OKLCH")
    if not 0 < L < 1 or not 0 <= C <= 0.5:
        raise ValueError("Use lightness in (0,1) and chroma in [0,.5]")
    direction = (math.cos(math.radians(h % 360)), math.sin(math.radians(h % 360)))

    def convert(amount):
        return oklab_to_linear_srgb((L, amount * direction[0], amount * direction[1]))

    def inside(rgb):
        return all(0 <= x <= 1 for x in rgb)

    linear = convert(C)
    if not inside(linear):
        low, high = 0.0, C
        for _ in range(GAMUT_STEPS):
            middle = (low + high) / 2
            if inside(convert(middle)):
                low = middle
            else:
                high = middle
        linear = convert(low)
    return tuple(12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055 for x in linear)


def _lch(lab):
    L, a, b = lab
    return (L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360)


def _hue_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def _rounded(values):
    return [round(float(x), 12) for x in values]


def _identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def make_palette(seed: str | int) -> dict[str, Any]:
    """Create all optics and an independently inspectable, seed-bound palette record."""
    raw = _seed_bytes(seed)
    name, offsets = HARMONIES[_bits(raw, "harmony") % len(HARMONIES)]
    hue = 360 * _unit(raw, "base-hue")
    permutation = tuple(itertools.permutations(range(3)))[_bits(raw, "role-permutation") % 6]
    hues = [
        (hue + offset + 6 * (2 * _unit(raw, f"hue/{i}") - 1)) % 360
        for i, offset in enumerate(offsets)
    ]
    lightness = (
        0.34 + 0.04 * _unit(raw, "dark-L"),
        0.82 + 0.04 * _unit(raw, "light-L"),
        0.58 + 0.04 * _unit(raw, "medium-L"),
    )
    desired_chroma = (
        0.10 + 0.06 * _unit(raw, "dark-C"),
        0.08 + 0.06 * _unit(raw, "light-C"),
        0.14 + 0.07 * _unit(raw, "medium-C"),
    )
    colors = [
        _rounded(gamut_map_oklch(L, C, hues[permutation[i]]))
        for i, (L, C) in enumerate(zip(lightness, desired_chroma, strict=True))
    ]
    labs = [srgb_to_oklab(color) for color in colors]
    lchs = [_lch(lab) for lab in labs]
    distances = [math.dist(labs[a], labs[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
    hue_distances = [_hue_distance(lchs[a][2], lchs[b][2]) for a, b in ((0, 1), (0, 2), (1, 2))]
    if (
        min(distances) < MIN_DISTANCE
        or min(hue_distances) < MIN_HUE_SEPARATION
        or any(lch[1] < limit for lch, limit in zip(lchs, MIN_CHROMA, strict=True))
    ):
        raise ValueError("The bounded palette construction failed its declared separation contract")
    dark_background = _bits(raw, "background-kind") % 8 != 0
    background_L = (
        (0.12 + 0.04 * _unit(raw, "background-L"))
        if dark_background
        else (0.93 + 0.025 * _unit(raw, "background-L"))
    )
    background_C = 0.015 + 0.015 * _unit(raw, "background-C") if dark_background else 0.012
    background = _rounded(gamut_map_oklch(background_L, background_C, hue + 35))
    scattering = _rounded(0.9 + 1.6 * L * L for L, _, _ in labs)
    result = {
        "version": VERSION,
        "seed": "0x" + raw.hex(),
        "optics": {
            "pigments_srgb": colors,
            "substrate_srgb": background,
            "scattering": scattering,
            "layer_scale": 5.0,
            "grain": 0.008,
            "grain_frequency": [450.0, 340.0],
        },
        "metadata": {
            "roles": list(ROLE_NAMES),
            "harmony": name,
            "base_hue_degrees": round(hue, 12),
            "role_permutation": list(permutation),
            "oklab": [_rounded(lab) for lab in labs],
            "oklch": [_rounded(lch) for lch in lchs],
            "requested_chroma": _rounded(desired_chroma),
            "pair_order": [[0, 1], [0, 2], [1, 2]],
            "pairwise_oklab_distance": _rounded(distances),
            "pairwise_hue_separation_degrees": _rounded(hue_distances),
            "thresholds": {
                "minimum_oklab_distance": MIN_DISTANCE,
                "minimum_hue_separation_degrees": MIN_HUE_SEPARATION,
                "minimum_chroma_by_role": list(MIN_CHROMA),
            },
            "background": {
                "kind": "dark" if dark_background else "light",
                "oklch": _rounded(_lch(srgb_to_oklab(background))),
            },
            "gamut_mapping": {
                "method": "fixed-lightness-and-hue-chroma-bisection",
                "iterations": GAMUT_STEPS,
            },
            "scattering_policy": (
                "Relative authored S=0.9+1.6*OKLab_L^2; K/S follows RGB reflectance. "
                "Not measured pigment coefficients."
            ),
            "randomness": (
                "SHA-256(version ASCII + NUL + named stream ASCII + NUL + 32 seed bytes); "
                "53-bit fractions"
            ),
        },
    }
    result["identity_sha256"] = _identity(result)
    return result


def verify_palette(record: dict[str, Any]) -> dict[str, Any]:
    """Regenerate the exact palette, detecting altered optics, metadata or identities."""
    if type(record) is not dict or record.get("version") != VERSION or "seed" not in record:
        raise ValueError("Unknown palette record")
    if record != make_palette(record["seed"]):
        raise ValueError("Palette differs from its deterministic seed contract")
    return record
