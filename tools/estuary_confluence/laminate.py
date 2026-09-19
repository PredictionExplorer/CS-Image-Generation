"""Positive, species-conservative exchange between two occupied wet layers.

This is an authored co-moving paint model. It does not solve vertical fluid
motion: horizontal velocities share a prescribed field at two bounded speeds.
"""

from __future__ import annotations

import math

import numpy as np


def layer_fractions(palette, count):
    """Validate archived upper shares without inventing defaults for new art."""
    values = palette.get("layer_fractions")
    if (
        type(values) is not list
        or len(values) != count
        or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values)
    ):
        raise ValueError(
            "Laminate layer_fractions must contain one upper share in [0, 1] per pigment"
        )
    return np.asarray(values, dtype="f4")


def exchange_layers(upper, lower, wetness, *, rate, dt, minimum_concentration=0.001):
    """Exact constant-wetness relaxation at fixed local layer amounts (float64).

    Two positive convex combinations approach the common local color fractions.
    Each species' upper+lower amount and each layer's total remain unchanged.
    Dry cells and air gaps exchange no pigment; no limiter or renormalization
    hides a mass loss. Numerical transport is a separate approximation.
    """
    a, b, wet = (np.asarray(value, dtype="f8") for value in (upper, lower, wetness))
    if (
        a.ndim != 3
        or min(a.shape, default=0) < 1
        or b.shape != a.shape
        or wet.shape != a.shape[:2]
        or any(not np.isfinite(value).all() for value in (a, b, wet))
        or np.any(a < 0)
        or np.any(b < 0)
        or np.any(wet < 0)
        or np.any(wet > 1)
    ):
        raise ValueError("Invalid laminate layer concentrations or wetness")
    for name, value in (
        ("rate", rate),
        ("dt", dt),
        ("minimum_concentration", minimum_concentration),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if minimum_concentration == 0:
        raise ValueError("minimum_concentration must be positive")
    exposure = rate * dt
    if not math.isfinite(exposure):
        raise ValueError("Laminate exchange exposure exceeds numerical limits")
    with np.errstate(over="ignore"):
        amount_a, amount_b = a.sum(axis=-1), b.sum(axis=-1)
        total = amount_a + amount_b
    if not np.isfinite(total).all():
        raise ValueError("Laminate layer amount exceeds numerical limits")
    active = (amount_a > minimum_concentration) & (amount_b > minimum_concentration)
    q = -np.expm1(-exposure * wet) * active
    to_upper = np.divide(q * amount_a, total, out=np.zeros_like(q), where=total > 0)
    to_lower = np.divide(q * amount_b, total, out=np.zeros_like(q), where=total > 0)
    return (
        a * (1 - to_lower[..., None]) + b * to_upper[..., None],
        b * (1 - to_upper[..., None]) + a * to_lower[..., None],
    )
