"""Seeded color compositions with distinct pigment and layer roles.

Hue relationships range from near-monochrome to a complementary accent. Roles
have deliberately unequal lightness, chroma, and scattering; five equally vivid
swatches are not the objective. Acceptance measures actual finite spectral paint
and its mixtures, using the same synthetic optical model as the renderer. It is
a numerical guardrail, not a score for artistic quality or measured paint data.

Five colors are always resolved together. Smaller paintings retain the same
prefix, with unchanged material coefficients and layer allocations: one dominant
pigment, then light support, then accent. The dark anchor and quiet bridge extend
that composition to five. One pigment varies through thickness and lighting,
without introducing additional chromatic materials.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from tools.estuary.optics import srgb_to_linear

from .palette import (
    SUPPORTED_CHROMATIC_COUNTS,
    _body_mixtures,
    _body_weights,
    _digest,
    _linear_to_oklab,
    _master,
    _physical_arrays,
    _range,
    _unit,
    normalize_seed,
)
from .procedural_palette import HUE_NAMES, _gamut_map
from .spectral import build_spectral_material, reflectance

VERSION = "composed-palette-v1"
PHYSICAL_VERSION = "composed-material-v1"
MAX_ATTEMPTS = 96
ROLES = ("dominant", "support", "accent", "deep-anchor", "quiet-bridge")
ROLE_NAMES = ("Dominant", "Light support", "Luminous accent", "Deep anchor", "Quiet bridge")
# Offsets are relative to a full-seed-derived principal hue. Narrow harmonies
# remain distinguishable by value and opacity rather than compulsory hue gaps.
RELATIONSHIPS = (
    ("tonal", (0, 16, -24, 8, -10)),
    ("analogous", (0, 35, -48, -18, 62)),
    ("complementary-accent", (0, 24, 178, -16, 70)),
    ("split-accent", (0, -34, 145, 20, -70)),
)
LIGHTNESS_BANDS = ((0.53, 0.62), (0.77, 0.85), (0.65, 0.74), (0.29, 0.37), (0.63, 0.71))
CHROMA_BANDS = ((0.12, 0.20), (0.045, 0.095), (0.16, 0.24), (0.035, 0.070), (0.018, 0.045))
SCATTERING_BANDS = ((0.17, 0.24), (0.66, 0.85), (0.35, 0.49), (0.12, 0.18), (0.78, 1.00))
UPPER_FRACTION_BANDS = ((0.65, 0.82), (0.18, 0.30), (0.45, 0.65), (0.76, 0.90), (0.12, 0.24))


def _color(lightness, chroma, hue):
    """Leave chroma headroom so tiny additions cannot act like ideal absorbers.

    Exact gamut-boundary channels can approach zero reflectance. In synthetic
    K--M reconstruction that creates disproportionately strong absorbers and
    steep color changes at the ends of mixture ramps. An eight-percent chroma
    reserve preserves lightness and hue while making those mixtures gentler.
    """
    mapped = _gamut_map(lightness, chroma, hue)
    lab = _linear_to_oklab(srgb_to_linear(mapped))
    return _gamut_map(lightness, float(np.linalg.norm(lab[1:])) * 0.92, hue)


def quality(palette):
    """Check developed color, value hierarchy, and finite spectral mixtures.

    The .18 mass and scale 12 match the released filled finish. A thinner .045
    sample checks that a glaze can change the substrate; .7 checks thicker paint.
    All five candidates are assessed even when only the first three will be used.
    """
    count = len(palette["scattering"]) - 1
    if count != 5:
        raise ValueError("Composed quality needs all five chromatic candidates")
    pairs = list(itertools.combinations(range(count), 2))
    samples = np.zeros((count * 3 + len(pairs), count + 1))
    for block, mass in enumerate((0.18, 0.045, 0.7)):
        samples[block * count : (block + 1) * count, :count] = np.eye(count) * mass
    for row, pair in enumerate(pairs, start=count * 3):
        samples[row, list(pair)] = 0.09
    material = build_spectral_material(palette)
    colors = reflectance(samples, material, layer_scale=12)
    lab = _linear_to_oklab(colors)
    pure, thin, thick, mixed = np.split(lab, (count, 2 * count, 3 * count))
    chroma = np.linalg.norm(pure[:, 1:], axis=1)
    mixed_chroma = np.linalg.norm(mixed[:, 1:], axis=1)
    separation = min(float(np.linalg.norm(pure[a] - pure[b])) for a, b in pairs)
    principal_separation = min(
        float(np.linalg.norm(pure[a] - pure[b])) for a, b in itertools.combinations(range(3), 2)
    )
    span = float(np.ptp(pure[:, 0]))
    # Light support remains legible on a dark ground; anchor contributes depth;
    # the bridge is allowed to be nearly neutral. Mixing is not forced to stay
    # highly saturated when complementary paints physically neutralize.
    passes = (
        bool(np.isfinite(colors).all())
        and separation >= 0.045
        and principal_separation >= 0.09
        and span >= 0.27
        and pure[1, 0] >= 0.76
        and pure[1, 0] - pure[3, 0] >= 0.27
        and float(chroma[:3].mean()) >= 0.055
        and float(chroma.max()) >= 0.08
        and float(mixed_chroma.max()) >= 0.045
        and bool(np.all(thin[:, 0] >= pure[:, 0]))
        and bool(np.all(pure[:, 0] >= thick[:, 0]))
    )
    return {
        "passes": bool(passes),
        "evaluated_chromatic_count": count,
        "optics": "38-band synthetic spectral finite paint",
        "developed_mass": 0.18,
        "layer_scale": 12.0,
        "minimum_pigment_separation": round(separation, 8),
        "minimum_principal_separation": round(principal_separation, 8),
        "lightness_span": round(span, 8),
        "light_support_lightness": round(float(pure[1, 0]), 8),
        "mean_principal_chroma": round(float(chroma[:3].mean()), 8),
        "median_mixture_chroma": round(float(np.median(mixed_chroma)), 8),
        "maximum_mixture_chroma": round(float(mixed_chroma.max()), 8),
    }


def build_palette(seed, chromatic_count):
    """Resolve a complete composition with stable, independently named streams."""
    if type(chromatic_count) is not int or chromatic_count not in SUPPORTED_CHROMATIC_COUNTS:
        raise ValueError("chromatic_count must be 1, 2, 3, or 5")
    seed = normalize_seed(seed)
    master = hashlib.sha256(VERSION.encode() + b"\0" + int(seed, 16).to_bytes(32, "big")).digest()
    physical_master = _master(seed)
    full = _physical_arrays(physical_master)
    # Non-optical phase behavior keeps the existing seeded parameter streams.
    # Only scattering and vertical allocation introduce new material roles.
    full["scattering"][:5] = [
        round(_range(master, f"material/{role}/scattering", *bounds), 10)
        for role, bounds in zip(ROLES, SCATTERING_BANDS, strict=True)
    ]
    full["layer_fractions"] = [
        round(_range(master, f"material/{role}/upper-fraction", *bounds), 10)
        for role, bounds in zip(ROLES, UPPER_FRACTION_BANDS, strict=True)
    ] + [round(_range(master, "material/chalk/upper-fraction", 0.04, 0.08), 10)]
    base_hue = _range(master, "base-hue", 0, 360)
    relationship, offsets = RELATIONSHIPS[int(_unit(master, "relationship") * len(RELATIONSHIPS))]
    chalk = [0.975, 0.975, 0.975]
    for attempt in range(MAX_ATTEMPTS):
        hues, colors = [], []
        for index, role in enumerate(ROLES):
            prefix = f"color/{attempt}/{role}"
            offset = 0 if index == 0 else _range(master, prefix + "/angle", -6, 6)
            hue = (base_hue + offsets[index] + offset) % 360
            hues.append(hue)
            colors.append(
                _color(
                    _range(master, prefix + "/lightness", *LIGHTNESS_BANDS[index]),
                    _range(master, prefix + "/chroma", *CHROMA_BANDS[index]),
                    hue,
                )
            )
        checks = quality(
            {
                "pigments_srgb": [*colors, chalk],
                "scattering": full["scattering"],
                "substrate_srgb": [1.0, 1.0, 1.0],
            }
        )
        if checks["passes"]:
            break
    else:
        raise ValueError("Composed palette exceeded its bounded material-quality search")
    indices = [*range(chromatic_count), 5]
    result = {key: [values[i] for i in indices] for key, values in full.items()}
    result.update(
        {
            "version": VERSION,
            "mode": "composed",
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
                "authored role-dependent scattering and synthetic RGB-derived spectra; "
                "not measured artist pigments"
            ),
            "family": f"composed-{relationship}",
            "pigments_srgb": [*colors[:chromatic_count], chalk],
            "pigment_names": [
                f"{ROLE_NAMES[i]} · {HUE_NAMES[int((hues[i] + 15) % 360 // 30)]}"
                for i in range(chromatic_count)
            ]
            + ["Shared chalk"],
            "pigment_roles": [*ROLES[:chromatic_count], "chalk"],
            "substrate_srgb": [1.0, 1.0, 1.0],
            "generator_attempt": attempt,
            "quality": checks,
            "color_generation": {
                "relationship": relationship,
                "base_hue_degrees": round(base_hue, 8),
                "candidate_hues_degrees": [round(hue, 8) for hue in hues],
                "quality_scope": "all five chromatic candidates and shared chalk",
                "gamut_mapping": (
                    "constant lightness and hue; chroma reduction plus eight-percent reserve"
                ),
            },
            "layer_allocation": {
                "version": "role-layers-v1",
                "meaning": "upper-layer fraction of each pigment's initial mass",
                "lower_fraction": "one minus layer_fractions",
                "activation": "laminate material model only; ignored by legacy material model",
            },
        }
    )
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["identity_sha256"] = hashlib.sha256(payload).hexdigest()
    return result
