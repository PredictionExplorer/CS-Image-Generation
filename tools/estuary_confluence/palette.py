"""Reproducible pigment relationships for Confluence Fresco.

These are authored RGB K--M pigments, not measured spectral artist materials.
The palette is generated once from the complete 256-bit seed. Named hash streams
keep changes in one choice from perturbing all the others; changing the pigment
count preserves the principal three colors and chalk. The released curated
default retains its exact v1 contract; procedural full-hue alternatives live in
``procedural_palette`` and share the unchanged physical parameter streams.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from tools.estuary.optics import absorption_over_scattering, linear_to_srgb, srgb_to_linear

VERSION = "confluence-palette-v1"
MAX_ATTEMPTS = 8

# Every row describes a relationship, rather than independent colors selected
# from a hue wheel. Additional colors remain related and receive less material.
FAMILIES = (
    (
        "mineral-tide",
        ("Deep blue", "Jade", "Coral", "Petrol", "Old gold"),
        ("243F75", "4D7C6B", "C06954", "285C69", "99804A"),
        "F1E9D8",
        "D6CCB6",
    ),
    (
        "saffron-delta",
        ("Ochre", "Petrol teal", "Oxide red", "Dusty rose", "Blue slate"),
        ("B8862E", "245461", "9D4334", "AA7876", "526C81"),
        "F1E8D7",
        "D7CABB",
    ),
    (
        "violet-estuary",
        ("Indigo", "Violet", "Green gold", "Plum", "Copper"),
        ("303A67", "725579", "969A54", "764664", "A56C4B"),
        "EEE8DB",
        "D3CED0",
    ),
    (
        "rose-mineral",
        ("Aubergine", "Dusty rose", "Verdigris", "Wine", "Stone green"),
        ("533A4D", "B37B7D", "4D8276", "873E56", "839383"),
        "F0E6D8",
        "DACBBE",
    ),
    (
        "winter-copper",
        ("Midnight blue", "Glacier turquoise", "Copper", "Cerulean", "Ochre"),
        ("24364F", "6B9E9A", "AD6541", "457887", "AC8A57"),
        "F0EEE5",
        "D6DADD",
    ),
    (
        "red-earth",
        ("Burgundy", "Viridian", "Burnt sienna", "Aubergine", "Sage"),
        ("6D3137", "426658", "AF7346", "674F67", "7E947A"),
        "EFE4CF",
        "CFC3B0",
    ),
)


def normalize_seed(seed: str | int) -> str:
    """Canonical lowercase hexadecimal; strings are hex with an optional prefix."""
    if type(seed) is int:
        number = seed
    elif type(seed) is str and re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,256}", seed):
        number = int(seed, 16)
    else:
        raise ValueError("Seed must be a hexadecimal string or a nonnegative integer")
    if not 0 <= number < 1 << 256:
        raise ValueError("Seed must fit in 256 bits")
    return f"0x{number:x}"


def _master(seed: str) -> bytes:
    return hashlib.sha256(VERSION.encode() + b"\0" + int(seed, 16).to_bytes(32, "big")).digest()


def _digest(master: bytes, name: str) -> bytes:
    return hashlib.sha256(master + b"\0" + name.encode("ascii")).digest()


def _unit(master: bytes, name: str) -> float:
    # Exactly representable 53-bit [0,1) values, independently addressed by name.
    return (int.from_bytes(_digest(master, name)[:8], "big") >> 11) / (1 << 53)


def _range(master: bytes, name: str, low: float, high: float) -> float:
    return low + (high - low) * _unit(master, name)


def _hex(value: str) -> np.ndarray:
    return np.array([int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    # Public-domain 2021 sRGB matrices from Björn Ottosson:
    # https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
    lms = np.asarray(rgb) @ np.array(
        [
            [0.4122214708, 0.2119034982, 0.0883024619],
            [0.5363325363, 0.6806995451, 0.2817188376],
            [0.0514459929, 0.1073969566, 0.6299787005],
        ]
    )
    return np.cbrt(lms) @ np.array(
        [
            [0.2104542553, 1.9779984951, 0.0259040371],
            [0.7936177850, -2.4285922050, 0.7827717662],
            [-0.0040720468, 0.4505937099, -0.8086757660],
        ]
    )


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = np.asarray(lab) @ np.array(
        [
            [1.0, 1.0, 1.0],
            [0.3963377774, -0.1055613458, -0.0894841775],
            [0.2158037573, -0.0638541728, -1.2914855480],
        ]
    )
    return lms**3 @ np.array(
        [
            [4.0767416621, -1.2684380046, -0.0041960863],
            [-3.3077115913, 2.6097574011, -0.7034186147],
            [0.2309699292, -0.3413193965, 1.7076147010],
        ]
    )


def _vary(base: str, master: bytes, label: str, *, quiet: bool = False) -> list[float]:
    lab = _linear_to_oklab(srgb_to_linear(_hex(base)))
    light = float(
        np.clip(
            lab[0]
            + _range(
                master, label + "/light", -0.012 if quiet else -0.035, 0.012 if quiet else 0.035
            ),
            0.12,
            0.97,
        )
    )
    chroma = float(np.hypot(lab[1], lab[2])) * _range(master, label + "/chroma", 0.86, 1.14)
    hue = math.atan2(lab[2], lab[1]) + math.radians(_range(master, label + "/hue", -8, 8))
    direction = np.array([0.0, math.cos(hue), math.sin(hue)])
    neutral = np.array([light, 0.0, 0.0])
    # Gamut mapping preserves lightness and hue, lowering only chroma.
    lower, upper = 0.0, chroma
    for _ in range(28):
        mid = (lower + upper) / 2
        rgb = _oklab_to_linear(neutral + mid * direction)
        if np.all(rgb >= 0) and np.all(rgb <= 1):
            lower = mid
        else:
            upper = mid
    rgb = np.clip(_oklab_to_linear(neutral + lower * direction), 0, 1)
    return [round(float(v), 10) for v in linear_to_srgb(rgb)]


def mixture_reflectance(density, pigments_srgb, scattering, substrate_srgb, layer_scale=5.0):
    """Finite K--M reference for any pigment count, returning linear sRGB.

    Used to assess real paint mixtures and tints, not just isolated swatches.
    This follows the original Estuary finite-layer equations and black floor.
    """
    density = np.asarray(density, dtype=np.float64)
    pigments = np.asarray(pigments_srgb, dtype=np.float64)
    scatter = np.asarray(scattering, dtype=np.float64)
    if (
        pigments.ndim != 2
        or pigments.shape[1] != 3
        or scatter.shape != (len(pigments),)
        or density.ndim == 0
        or density.shape[-1] != len(pigments)
        or not np.isfinite(density).all()
        or np.any(density < 0)
        or not np.isfinite(scatter).all()
        or np.any(scatter <= 0)
        or np.any(scatter > 1e6)
        or type(layer_scale) not in (float, int)
        or not math.isfinite(layer_scale)
        or not 0 < layer_scale <= 1e6
    ):
        raise ValueError("Invalid finite pigment mixture")
    colors = srgb_to_linear(pigments)
    substrate = srgb_to_linear(substrate_srgb)
    if substrate.shape != (3,):
        raise ValueError("Substrate must have three channels")
    strengths = np.minimum(density, 1e6) * scatter
    total = strengths.sum(axis=-1, keepdims=True)
    absorption = strengths @ absorption_over_scattering(colors)
    ratio = absorption / np.where(total > 0, total, 1)
    a, b = 1 + ratio, np.sqrt(ratio * (ratio + 2))
    thickness = total * layer_scale
    exponent = b * thickness
    one_minus_exp = -np.expm1(-2 * exponent)
    denominator = b * (2 - one_minus_exp) + a * one_minus_exp
    regular = b > 1e-6
    safe = np.where(regular & (denominator > 0), denominator, 1)
    reflection = np.where(regular, one_minus_exp / safe, thickness / (1 + thickness))
    transmission = np.where(regular, 2 * b * np.exp(-exponent) / safe, 1 / (1 + thickness))
    result = reflection + transmission**2 * substrate / np.maximum(
        1 - reflection * substrate, 1e-12
    )
    return np.clip(result, 0, 1)


def palette_quality(palette: dict) -> dict[str, float | bool]:
    """Bounded basic checks; these are numerical guardrails, not an art score."""
    n, chalk = len(palette["scattering"]), palette["chalk_index"]
    density = np.zeros((10, n))
    density[:3, :3] = np.eye(3) * 0.55
    density[3:6] = density[:3]
    density[3:6, chalk] = 0.035
    density[6, :2] = 0.275
    density[7, 1:3] = 0.275
    density[8, [0, 2]] = 0.275
    density[9, chalk] = 0.55
    rgb = mixture_reflectance(
        density, palette["pigments_srgb"], palette["scattering"], palette["substrate_srgb"]
    )
    lab = _linear_to_oklab(rgb)
    contrast = float(lab[:, 0].max() - lab[:, 0].min())
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    separation = min(float(np.linalg.norm(lab[i] - lab[j])) for i, j in ((0, 1), (1, 2), (0, 2)))
    valid = (
        contrast >= 0.20
        and separation >= 0.055
        and float(chroma[:3].mean()) >= 0.035
        and float(chroma[6:9].max()) >= 0.018
        and bool(np.all(lab[3:6, 0] > lab[:3, 0]))
    )
    return {
        "passes": bool(valid),
        "lightness_span": round(contrast, 8),
        "minimum_pigment_separation": round(separation, 8),
        "mean_pigment_chroma": round(float(chroma[:3].mean()), 8),
        "maximum_mixture_chroma": round(float(chroma[6:9].max()), 8),
    }


def _physical_arrays(master: bytes) -> dict:
    """The released physical parameters, independent of any color selection."""
    physical_bases = {
        "scattering": [0.15, 0.40, 0.70, 0.30, 0.45, 6.0],
        "settling": [0.25, 1.1, 3.6, 1.5, 2.3, 2.2],
        "release": [0.60, 0.35, 0.08, 0.25, 0.15, 0.24],
        "specific_volumes": [0.05, 0.25, 0.38, 0.20, 0.30, 1.0],
        "granulation": [0.08, 0.32, 0.80, 0.36, 0.58, 0.42],
    }
    return {
        key: [
            round(base * _range(master, f"{key}/{i}", 0.88, 1.12), 10)
            for i, base in enumerate(values)
        ]
        for key, values in physical_bases.items()
    }


def _body_weights(master: bytes) -> list[float]:
    return [
        round(base * _range(master, f"body/{body}/weight", 0.9, 1.1), 10)
        for body, base in enumerate((0.8, 0.7, 0.18))
    ]


def _body_mixtures(master: bytes, chromatic_count: int) -> list[list[float]]:
    body_mixtures = []
    for body, chalk_base in enumerate((0.008, 0.075, 0.018)):
        row = [0.0] * (chromatic_count + 1)
        row[-1] = chalk_base * _range(master, f"body/{body}/chalk", 0.8, 1.2)
        if chromatic_count == 5:
            row[3] = _range(master, f"body/{body}/extra-3", 0.065, 0.09)
            row[4] = _range(master, f"body/{body}/extra-4", 0.035, 0.055)
        row[body] = 1 - sum(row)
        body_mixtures.append(row)
    return body_mixtures


def _curated_palette(seed: str | int, chromatic_count: int = 3) -> dict:
    """Resolve one seed into a complete, JSON-ready, versioned material palette."""
    if type(chromatic_count) is not int or chromatic_count not in (3, 5):
        raise ValueError("chromatic_count must be 3 or 5")
    seed = normalize_seed(seed)
    master = _master(seed)
    family, names, anchors, chalk, ground = FAMILIES[int(_unit(master, "family") * len(FAMILIES))]
    roles = ["dominant", "support", "accent", "undertone", "mineral-note"]
    # Resolve five every time so count comparisons use exactly the same colors,
    # material traits, quality decision, substrate, and chalk.
    full = _physical_arrays(master)
    for attempt in range(MAX_ATTEMPTS):
        prefix = f"color/{attempt}"
        colors = [_vary(color, master, f"{prefix}/{i}") for i, color in enumerate(anchors)]
        colors.append(_vary(chalk, master, "chalk", quiet=True))
        ground_linear = srgb_to_linear(_vary(ground, master, "ground", quiet=True))
        substrate = [
            round(float(value), 10) for value in linear_to_srgb(1 - (1 - ground_linear) * 0.55)
        ]
        candidate = {**full, "pigments_srgb": colors, "substrate_srgb": substrate, "chalk_index": 5}
        quality = palette_quality(candidate)
        if quality["passes"]:
            break
    else:
        raise ValueError("No palette within this version's bounded mixture-quality limits")
    indices = [*range(chromatic_count), 5]
    palette = {key: [values[i] for i in indices] for key, values in full.items()}
    palette.update(
        {
            "version": VERSION,
            "seed": seed,
            "seed_sha256": hashlib.sha256(int(seed, 16).to_bytes(32, "big")).hexdigest(),
            "family": family,
            "chromatic_count": chromatic_count,
            "pigments_srgb": [colors[i] for i in indices],
            "pigment_ids": [*(f"chromatic-{i}" for i in range(chromatic_count)), "chalk"],
            "pigment_names": [*names[:chromatic_count], "Shared chalk"],
            "pigment_roles": [*roles[:chromatic_count], "chalk"],
            "substrate_srgb": substrate,
            "chalk_index": chromatic_count,
            "underpaint_index": 2,
            "body_weights": _body_weights(master),
            # JSON numbers cannot retain 256-bit integers in browser clients.
            # Hex preserves every bit while int(value, 16) recovers the exact
            # original entropy used by the numerical substrate generator.
            "substrate_seed": "0x" + _digest(master, "substrate").hex(),
            "generator_attempt": attempt,
            "quality": quality,
            "coefficient_provenance": (
                "authored RGB K-M approximation; not measured spectral pigments"
            ),
        }
    )
    palette["body_mixtures"] = _body_mixtures(master, chromatic_count)
    payload = json.dumps(palette, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    palette["identity_sha256"] = hashlib.sha256(payload).hexdigest()
    return palette


def generate_palette(seed: str | int, chromatic_count: int = 3, *, mode: str = "curated") -> dict:
    """Resolve a reproducible palette; the default preserves released v1 bytes.

    ``harmonic`` derives related hues from a seed-selected angle across the full
    color circle. ``random`` independently selects chromatic hues for comparison.
    Both procedural modes share physical coefficients and use a white ground.
    """
    if type(mode) is not str or mode not in ("curated", "harmonic", "random"):
        raise ValueError("Palette mode must be curated, harmonic, or random")
    if mode == "curated":
        return _curated_palette(seed, chromatic_count)
    from .procedural_palette import build_palette

    return build_palette(seed, chromatic_count, mode)
