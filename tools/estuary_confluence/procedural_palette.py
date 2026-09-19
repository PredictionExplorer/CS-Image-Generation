"""Procedural full-hue pigment relationships over a neutral white ground.

Harmony is expressed as angular relationships, never a list of fixed RGB
anchors. The five candidates are always generated and tested together, so a
smaller study retains the identical color prefix of its five-color
counterpart. All five pigments receive the same chroma range and must remain
distinct as actual finite K--M paint layers, rather than only as RGB swatches.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np

from tools.estuary.optics import linear_to_srgb

from .palette import (
    SUPPORTED_CHROMATIC_COUNTS,
    _body_mixtures,
    _body_weights,
    _digest,
    _linear_to_oklab,
    _master,
    _oklab_to_linear,
    _physical_arrays,
    _range,
    _unit,
    mixture_reflectance,
    normalize_seed,
)
from .palette import (
    VERSION as PHYSICAL_VERSION,
)

VERSION = "scatter-palette-v1"
MAX_ATTEMPTS = 128
HARMONIES = (
    ("split-complement", (0, 150, 210, 30, 330)),
    ("triadic", (0, 120, 240, 60, 300)),
    ("tetradic", (0, 90, 180, 270, 45)),
)
LIGHTNESS_BANDS = ((0.39, 0.54), (0.61, 0.75), (0.48, 0.64), (0.65, 0.79), (0.44, 0.60))
HUE_NAMES = (
    "Carmine",
    "Vermilion",
    "Ochre",
    "Gold",
    "Leaf green",
    "Viridian",
    "Teal",
    "Cerulean",
    "Cobalt",
    "Indigo",
    "Violet",
    "Magenta",
)


def _gamut_map(lightness, chroma, hue_degrees):
    """Preserve authored lightness/hue; reduce chroma to the sRGB boundary."""
    hue = math.radians(hue_degrees)
    direction = np.array([0.0, math.cos(hue), math.sin(hue)])
    neutral = np.array([lightness, 0.0, 0.0])
    low, high = 0.0, chroma
    for _ in range(28):
        candidate = (low + high) * 0.5
        rgb = _oklab_to_linear(neutral + candidate * direction)
        if np.all(rgb >= 0) and np.all(rgb <= 1):
            low = candidate
        else:
            high = candidate
    rgb = np.clip(_oklab_to_linear(neutral + low * direction), 0, 1)
    return [round(float(value), 10) for value in linear_to_srgb(rgb)]


def quality(palette):
    """Check every chromatic pigment, chalk tint, and pairwise K--M mixture.

    The .7 material load tests developed passages; .025 chalk checks visible
    tinting. Complementary mixtures may be subdued, but all original pigments
    must retain chroma and distinguishability. This is a validity guard, not an
    artistic score. Selection always evaluates all five generated candidates.
    """
    count = len(palette["scattering"]) - 1
    pairs = list(itertools.combinations(range(count), 2))
    samples = np.zeros((count * 2 + len(pairs) + 1, count + 1))
    samples[:count, :count] = np.eye(count) * 0.7
    samples[count : count * 2] = samples[:count]
    samples[count : count * 2, -1] = 0.025
    for index, pair in enumerate(pairs, start=count * 2):
        samples[index, list(pair)] = 0.35
    samples[-1, -1] = 0.7
    rgb = mixture_reflectance(
        samples, palette["pigments_srgb"], palette["scattering"], palette["substrate_srgb"]
    )
    lab = _linear_to_oklab(rgb)
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    separation = min(float(np.linalg.norm(lab[a] - lab[b])) for a, b in pairs)
    lightness_span = float(lab[:, 0].max() - lab[:, 0].min())
    mixed_chroma = chroma[count * 2 : -1]
    passes = (
        separation >= 0.085
        and lightness_span >= 0.28
        and float(chroma[:count].min()) >= 0.047
        and float(chroma[:count].mean()) >= 0.09
        and float(np.median(mixed_chroma)) >= 0.025
        and float(mixed_chroma.max()) >= 0.065
        and bool(np.all(lab[count : count * 2, 0] > lab[:count, 0]))
    )
    return {
        "passes": bool(passes),
        "evaluated_chromatic_count": count,
        "minimum_pigment_separation": round(separation, 8),
        "lightness_span": round(lightness_span, 8),
        "minimum_pigment_chroma": round(float(chroma[:count].min()), 8),
        "mean_pigment_chroma": round(float(chroma[:count].mean()), 8),
        "median_mixture_chroma": round(float(np.median(mixed_chroma)), 8),
        "maximum_mixture_chroma": round(float(mixed_chroma.max()), 8),
    }


def build_palette(seed, chromatic_count, mode):
    """Keep phase behavior fixed while changing only the chromatic experiment."""
    if type(chromatic_count) is not int or chromatic_count not in SUPPORTED_CHROMATIC_COUNTS:
        raise ValueError("chromatic_count must be 1, 2, 3, or 5")
    if mode not in ("harmonic", "random"):
        raise ValueError("Procedural mode must be harmonic or random")
    seed = normalize_seed(seed)
    physical_master = _master(seed)
    full = _physical_arrays(physical_master)
    indices = [*range(chromatic_count), 5]
    result = {key: [values[i] for i in indices] for key, values in full.items()}
    master = hashlib.sha256(VERSION.encode() + b"\0" + int(seed, 16).to_bytes(32, "big")).digest()
    base_hue = _range(master, "base-hue", 0, 360)
    harmony, offsets = HARMONIES[int(_unit(master, "harmony") * len(HARMONIES))]
    if mode == "random":
        harmony = "independent"
    chalk = [0.975, 0.975, 0.975]
    for attempt in range(MAX_ATTEMPTS):
        hues, colors = [], []
        for pigment, bounds in enumerate(LIGHTNESS_BANDS):
            prefix = f"{mode}/{attempt}/{pigment}"
            if pigment == 0:
                hue = base_hue
            elif mode == "harmonic":
                hue = base_hue + offsets[pigment] + _range(master, prefix + "/angle", -7, 7)
            else:
                hue = _range(master, prefix + "/angle", 0, 360)
            hue %= 360
            hues.append(hue)
            colors.append(
                _gamut_map(
                    _range(master, prefix + "/lightness", *bounds),
                    _range(master, prefix + "/chroma", 0.16, 0.25),
                    hue,
                )
            )
        full_candidate = {
            "pigments_srgb": [*colors, chalk],
            "scattering": full["scattering"],
            "substrate_srgb": [1.0, 1.0, 1.0],
        }
        checks = quality(full_candidate)
        if checks["passes"]:
            break
    else:
        raise ValueError("Procedural palette exceeded its bounded material-quality search")
    result.update(
        {
            "version": VERSION,
            "mode": mode,
            "physical_version": PHYSICAL_VERSION,
            "seed": seed,
            "seed_sha256": hashlib.sha256(int(seed, 16).to_bytes(32, "big")).hexdigest(),
            "chromatic_count": chromatic_count,
            "pigment_ids": [*(f"chromatic-{i}" for i in range(chromatic_count)), "chalk"],
            "chalk_index": chromatic_count,
            "underpaint_index": min(2, chromatic_count - 1),
            "body_weights": _body_weights(physical_master),
            "body_mixtures": _body_mixtures(physical_master, chromatic_count),
            "substrate_seed": "0x" + _digest(physical_master, "substrate").hex(),
            "coefficient_provenance": (
                "authored RGB K-M approximation; not measured spectral pigments"
            ),
            "family": f"{mode}-{harmony}",
            "pigments_srgb": [*colors[:chromatic_count], chalk],
            "pigment_names": [
                f"{i + 1} · {HUE_NAMES[int((hues[i] + 15) % 360 // 30)]}"
                for i in range(chromatic_count)
            ]
            + ["Shared chalk"],
            "pigment_roles": ["chromatic"] * chromatic_count + ["chalk"],
            "substrate_srgb": [1.0, 1.0, 1.0],
            "generator_attempt": attempt,
            "quality": checks,
            "color_generation": {
                "relationship": harmony,
                "base_hue_degrees": round(base_hue, 8),
                "candidate_hues_degrees": [round(hue, 8) for hue in hues],
                "quality_scope": "all five chromatic candidates and shared chalk",
                "gamut_mapping": "constant lightness and hue; chroma reduction",
            },
        }
    )
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["identity_sha256"] = hashlib.sha256(payload).hexdigest()
    return result
