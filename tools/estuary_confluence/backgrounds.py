"""Versioned appearance-only grounds for existing, immutable pigment palettes.

The named grounds are fixed display-sRGB colors. Palette Night follows an actual
principal pigment hue, with small full-seed-derived OKLCH variations. It remains
dark and low in chroma so it supports the colored painting rather than competing
with it. No paint concentrations, pigment coefficients, or palette values change.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from tools.estuary.optics import srgb_to_linear

from .palette import _linear_to_oklab, normalize_seed
from .procedural_palette import _gamut_map

VERSION = "confluence-background-v1"
NAMED_SRGB = {
    "white": (1.0, 1.0, 1.0),
    "charcoal": (21 / 255, 23 / 255, 27 / 255),
    "midnight-blue": (7 / 255, 18 / 255, 34 / 255),
    "aubergine": (27 / 255, 15 / 255, 34 / 255),
}
NAMES = (*NAMED_SRGB, "palette-night")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unit(seed, label):
    digest = hashlib.sha256(
        VERSION.encode() + b"\0" + int(seed, 16).to_bytes(32, "big") + b"\0" + label.encode("ascii")
    ).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def generate_background(name: str, palette: dict) -> dict:
    """Return exact sRGB/linear ground values and their reproducible derivation.

    The first chromatic pigment supplies Palette Night's hue. If it is neutral,
    use the most chromatic of the principal three; a wholly neutral principal
    palette receives a neutral night ground. Using only that common prefix keeps
    three/five-color comparisons on the same ground when their principal paints
    match. The record still binds all actual pigment colors by a separate hash.
    """
    if type(name) is not str or name not in NAMES:
        raise ValueError(f"Background must be one of {', '.join(NAMES)}")
    if type(palette) is not dict:
        raise ValueError("Background selection requires a resolved pigment palette")
    seed = normalize_seed(palette.get("seed"))
    count = palette.get("chromatic_count")
    if type(count) is not int or count not in (3, 5):
        raise ValueError("Background palette needs three or five chromatic pigments")
    colors = np.asarray(palette.get("pigments_srgb"), dtype="f8")
    if colors.shape != (count + 1, 3):
        raise ValueError("Background palette dimensions differ from its pigment count")
    linear = srgb_to_linear(colors)
    if name in NAMED_SRGB:
        ground = list(NAMED_SRGB[name])
        method = {"kind": "fixed-display-srgb", "name": name}
    else:
        lab = _linear_to_oklab(linear[:3])
        chroma = np.linalg.norm(lab[:, 1:], axis=1)
        anchor = 0 if chroma[0] > 1e-6 else int(np.argmax(chroma))
        neutral = chroma[anchor] <= 1e-6
        hue = 0.0 if neutral else math.degrees(math.atan2(lab[anchor, 2], lab[anchor, 1])) % 360
        offset = (_unit(seed, "hue-offset") * 2 - 1) * 8
        lightness = 0.155 + 0.06 * _unit(seed, "lightness")
        requested_chroma = 0.0 if neutral else 0.008 + 0.017 * _unit(seed, "chroma")
        ground = _gamut_map(lightness, requested_chroma, (hue + offset) % 360)
        method = {
            "kind": "seeded-oklch",
            "anchor_pigment_index": anchor,
            "anchor_hue_degrees": hue,
            "hue_offset_degrees": offset,
            "lightness": lightness,
            "requested_chroma": requested_chroma,
            "neutral_fallback": bool(neutral),
            "gamut_mapping": "preserve lightness and hue; reduce chroma to fit sRGB",
        }
    record = {
        "version": VERSION,
        "name": name,
        "seed": seed,
        "method": method,
        "palette_colors_sha256": hashlib.sha256(_encoded(colors.tolist())).hexdigest(),
        "ground_srgb": ground,
        "ground_linear": srgb_to_linear(ground).tolist(),
    }
    record["identity_sha256"] = hashlib.sha256(_encoded(record)).hexdigest()
    return record


def _equivalent(first, second):
    """Allow only finite reconstruction roundoff, never changed record structure."""
    if type(first) is dict and type(second) is dict:
        return first.keys() == second.keys() and all(
            _equivalent(first[key], second[key]) for key in first
        )
    if type(first) is list and type(second) is list:
        return len(first) == len(second) and all(
            _equivalent(a, b) for a, b in zip(first, second, strict=True)
        )
    if type(first) is float and type(second) is float:
        return (
            math.isfinite(first)
            and math.isfinite(second)
            and math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(first) is type(second) and first == second


def validate_background(record: dict, palette: dict) -> dict:
    """Verify exact archived bytes' identity and a numerically equivalent derivation.

    The returned copy preserves the archived color values and identity. Only
    recomputed floating-point derivation values receive a 1e-12 tolerance; the
    saved self-hash, seed, palette binding, schema, and algorithm remain exact.
    """
    keys = {
        "version",
        "name",
        "seed",
        "method",
        "palette_colors_sha256",
        "ground_srgb",
        "ground_linear",
        "identity_sha256",
    }
    if type(record) is not dict or set(record) != keys or record["version"] != VERSION:
        raise ValueError("Unsupported or incomplete background record")
    payload = {key: value for key, value in record.items() if key != "identity_sha256"}
    if hashlib.sha256(_encoded(payload)).hexdigest() != record["identity_sha256"]:
        raise ValueError("Background record identity differs")
    expected = generate_background(record["name"], palette)
    expected.pop("identity_sha256")
    if not _equivalent(payload, expected):
        raise ValueError("Background derivation differs from its palette and seed")
    return copy.deepcopy(record)
