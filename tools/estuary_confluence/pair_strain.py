"""Bounded pair extension measurements for the optional incompressible strain.

This adds no random or ambient force. Each descriptor follows one real pair's
projected separation and relative source motion. The denominator retains the
pair's physical 3D distance, so a crossing created only by projection cannot
produce an artificial singularity. A projected coincidence produces zero strain.
"""

from __future__ import annotations

import math

import numpy as np

VERSION = "source-pair-strain-v1"
PAIRS = ((0, 1), (1, 2), (2, 0))
RATE_LIMIT = 20.0


def pair_strain_uniforms(frame, radius):
    """Return float32 ``(..., 3, 3)`` rows of ``(axis.x, axis.y, rate)``.

    Positive rate stretches along the pair axis and contracts perpendicular to
    it. The velocity conditioning is identical to Estuary's dipole/spin adapter;
    rate is a derivative with respect to the complete recording's unit interval.
    One frame or a batch from ``Source.sample`` is accepted.
    """
    if type(radius) not in (int, float) or not math.isfinite(radius) or not 0.02 <= radius <= 2:
        raise ValueError("stir_radius must be finite and in [0.02, 2]")
    position = np.asarray(frame.positions, dtype=np.float64)
    speed = np.asarray(frame.velocities, dtype=np.float64)
    distance = np.asarray(frame.pair_distances, dtype=np.float64)
    if (
        position.shape[-2:] != (3, 2)
        or speed.shape != position.shape
        or distance.shape != (*position.shape[:-2], 3)
        or not all(np.isfinite(a).all() for a in (position, speed, distance))
        or np.any(distance < 0)
    ):
        raise ValueError("Invalid source measurements for pair strain")
    speed = speed / (1 + np.linalg.norm(speed, axis=-1, keepdims=True) / 24)
    rows = []
    for index, (a, b) in enumerate(PAIRS):
        delta = position[..., b, :] - position[..., a, :]
        relative = speed[..., b, :] - speed[..., a, :]
        length = np.linalg.norm(delta, axis=-1, keepdims=True)
        axis = np.divide(delta, length, out=np.zeros_like(delta), where=length > 1e-12)
        radial = np.sum(delta * relative, axis=-1)
        radial /= distance[..., index] ** 2 + radius**2
        rate = RATE_LIMIT * np.tanh(radial / RATE_LIMIT)
        rate = np.where(length[..., 0] > 1e-12, rate, 0)
        rows.append(np.concatenate((axis, rate[..., None]), axis=-1))
    result = np.stack(rows, axis=-2).astype("f4")
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite pair strain response")
    return result
