"""Seeded material properties initialized once, then transported with paint.

The two signed traits describe relative aggregation affinity and fabric response.
They are authored bounded heterogeneity, not measured pigment parameters. Their
initial fields combine two smooth physical scales with independent seed streams;
reactions read the transported properties rather than re-sampling an origin map.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from .palette import normalize_seed

VERSION = "paint-material-variation-v1"
FIELD_NAMES = ("trait_upper", "trait_lower")
DEFAULTS = {
    "version": VERSION,
    "amplitude": 0.08,
    "coarse_scale": 0.16,
    "fine_scale": 0.035,
    "fine_fraction": 0.25,
}
_BOUNDS = {
    "amplitude": (0, 0.25),
    "coarse_scale": (0.001, 2),
    "fine_scale": (0.001, 2),
    "fine_fraction": (0, 0.5),
}


def validate_config(value):
    """Zero amplitude resolves to absence, including the archive dictionary."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Material variation must contain only documented fields")
    result = {**DEFAULTS, **copy.deepcopy(value)}
    if result["version"] != VERSION:
        raise ValueError(f"Material variation version must be {VERSION}")
    for name, (low, high) in _BOUNDS.items():
        number = result[name]
        try:
            valid = type(number) in (int, float) and math.isfinite(number) and low <= number <= high
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"Material variation {name} must be finite and in [{low}, {high}]")
        result[name] = float(number)
    if result["fine_scale"] > result["coarse_scale"]:
        raise ValueError("Material variation fine scale must not exceed coarse scale")
    return result if result["amplitude"] else None


def seed_key(seed, channel, scale):
    """Versioned independent streams; every one of the seed's 256 bits participates."""
    if channel not in ("aggregation", "fabric") or scale not in ("coarse", "fine"):
        raise ValueError("Unknown material trait stream")
    key = hashlib.sha256(
        VERSION.encode("ascii")
        + b"\0"
        + channel.encode("ascii")
        + b"\0"
        + scale.encode("ascii")
        + b"\0"
        + int(normalize_seed(seed), 16).to_bytes(32, "big")
    ).digest()
    return tuple(int.from_bytes(key[i : i + 4], "big") for i in range(0, 16, 4))


def _noise(points, key):
    cells = np.floor(points).astype(np.int64)
    fraction = points - cells
    fraction *= fraction * (3 - 2 * fraction)
    mask = np.uint64(0xFFFFFFFF)

    def lattice(dx, dy):
        x = (cells[..., 0] + dx).astype(np.uint64) & mask
        y = (cells[..., 1] + dy).astype(np.uint64) & mask
        value = ((x ^ np.uint64(key[0])) * np.uint64(0x9E3779B9)) & mask
        value ^= ((y ^ np.uint64(key[1])) * np.uint64(0x85EBCA6B)) & mask
        value ^= np.uint64(key[2])
        value = ((value ^ (value >> np.uint64(16))) * np.uint64(0x7FEB352D)) & mask
        value = ((value ^ (value >> np.uint64(15))) * np.uint64(0x846CA68B)) & mask
        value ^= (value >> np.uint64(16)) ^ np.uint64(key[3])
        return (value >> np.uint64(8)).astype("f8") / 8388608.0 - 1

    a, b, c, d = (lattice(dx, dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)))
    return (a * (1 - fraction[..., 0]) + b * fraction[..., 0]) * (1 - fraction[..., 1]) + (
        c * (1 - fraction[..., 0]) + d * fraction[..., 0]
    ) * fraction[..., 1]


def initial_traits(world_positions, seed, config):
    """Evaluate the initialization map once, before masking it to occupied paint."""
    settings = validate_config(config)
    if settings is None:
        raise ValueError("Initial traits require enabled material variation")
    positions = np.asarray(world_positions, dtype="f8")
    if positions.ndim != 3 or positions.shape[-1] != 2 or not np.isfinite(positions).all():
        raise ValueError("Initial material coordinates require finite H x W x 2 fields")
    if np.any(np.abs(positions) / settings["fine_scale"] > (1 << 30)):
        raise ValueError("Material coordinates exceed the supported lattice range")
    channels = []
    for channel in ("aggregation", "fabric"):
        coarse = _noise(positions / settings["coarse_scale"], seed_key(seed, channel, "coarse"))
        fine = _noise(positions / settings["fine_scale"], seed_key(seed, channel, "fine"))
        channels.append(coarse * (1 - settings["fine_fraction"]) + fine * settings["fine_fraction"])
    return np.stack(channels, -1).astype("f4")


def validate_fields(fields, shape, *, enabled, pigment_amounts=None, minimum_concentration=1e-5):
    """Require the optional pair atomically; no renderer or simulation mutation."""
    present = set(FIELD_NAMES) & set(fields)
    if present != (set(FIELD_NAMES) if enabled else set()):
        raise ValueError("Material traits must match the enabled configuration as an atomic pair")
    for name in present:
        field = np.asarray(fields[name])
        if field.shape != (*shape, 2) or field.dtype != np.dtype("f4"):
            raise ValueError("Material traits require native float32 H x W x 2 arrays")
        if not np.isfinite(field).all() or np.any(np.abs(field) > 1):
            raise ValueError("Material traits must be finite and in [-1, 1]")
    if enabled and pigment_amounts is not None:
        if len(pigment_amounts) != 2:
            raise ValueError("Trait support requires both layer amounts")
        if not math.isfinite(minimum_concentration) or minimum_concentration < 0:
            raise ValueError("Trait support requires a finite nonnegative threshold")
        for name, amount in zip(FIELD_NAMES, pigment_amounts, strict=True):
            amount = np.asarray(amount)
            if amount.shape != tuple(shape) or not np.isfinite(amount).all() or np.any(amount < 0):
                raise ValueError("Trait support requires finite nonnegative matching layer amounts")
            if np.any(fields[name][amount <= minimum_concentration] != 0):
                raise ValueError("Material traits must be zero outside occupied pigment support")


def rate_multipliers(traits, contact, wetness, config):
    """Return aggregation, breakup, fabric factors; dry/non-contact sites are neutral."""
    settings = validate_config(config)
    if settings is None:
        raise ValueError("Material rate factors require enabled variation")
    contact, wetness, traits = (
        np.asarray(value, dtype="f8") for value in (contact, wetness, traits)
    )
    if (
        traits.shape != (*contact.shape, 2)
        or wetness.shape != contact.shape
        or any(not np.isfinite(v).all() for v in (contact, wetness, traits))
        or np.any(np.abs(traits) > 1)
        or any(np.any(v < 0) or np.any(v > 1) for v in (contact, wetness))
    ):
        raise ValueError("Invalid material traits or contact/wetness fields")
    exposure = settings["amplitude"] * contact * wetness
    aggregation = exposure * traits[..., 0]
    return 1 + aggregation, 1 - aggregation, 1 + exposure * traits[..., 1]
