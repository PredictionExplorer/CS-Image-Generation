"""Float64 reference for Estuary's orbit-driven, incompressible stirring field.

A sum of translating Gaussian dipoles and rotating Gaussian pair kernels defines
one streamfunction. Multiplying that scalar by a squared boundary envelope makes
its curl tangent to the painting boundary; including both product-rule terms is
essential. This is a prescribed two-dimensional kinematic field, not a solution
of the Navier--Stokes equations.

Tool rows are ``(x, y, vx, vy)``; pair rows are ``(center_x, center_y, spin)``.
Velocities and spins must already be conditioned by the source adapter. In
particular, a source-velocity soft bound belongs *before* this function. A
pointwise clamp of its resulting velocity would destroy incompressibility.
``domain_scale`` enlarges the simulated support around the visible canvas; it
does not rescale the source positions, velocities, or stirring radii.
``carrier_velocity`` is an optional authored background current, independent of
the recording, included in the same boundary-conditioned streamfunction.
Optional ``pair_strain`` adds pair-aligned Gaussian quadrupoles using bounded
``(axis.x, axis.y, rate)`` descriptors from the source adapter. Its analytic
gradient participates in the same boundary product rule, preserving zero
divergence and the no-through-flow boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

PAIR_RADIUS_RATIO = 1.7


def _array(value: ArrayLike, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must have shape {shape} and contain only finite values")
    return array


@dataclass(frozen=True)
class _Evaluation:
    potential: NDArray[np.float64]
    gradient: NDArray[np.float64]


def _evaluate(
    points: ArrayLike,
    tools: ArrayLike,
    pairs: ArrayLike,
    *,
    aspect: float,
    stir_radius: float,
    flow_strength: float,
    pair_swirl: float,
    domain_scale: float = 1.0,
    carrier_velocity: tuple[float, float] = (0.0, 0.0),
    pair_strain: float = 0.0,
    strains: ArrayLike | None = None,
) -> _Evaluation:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must have shape (..., 2) and contain finite values")
    tools = _array(tools, (3, 4), "tools")
    pairs = _array(pairs, (3, 3), "pairs")
    carrier = _array(carrier_velocity, (2,), "carrier_velocity")
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("aspect must be finite and positive")
    if not np.isfinite(domain_scale) or domain_scale <= 0:
        raise ValueError("domain_scale must be finite and positive")
    if not np.isfinite(stir_radius) or stir_radius <= 0:
        raise ValueError("stir_radius must be finite and positive")
    if not np.isfinite(flow_strength) or not np.isfinite(pair_swirl):
        raise ValueError("flow_strength and pair_swirl must be finite")
    if not np.isfinite(pair_strain) or not 0 <= pair_strain <= 4:
        raise ValueError("pair_strain must be finite and in [0, 4]")

    potential = carrier[0] * points[..., 1] - carrier[1] * points[..., 0]
    gradient = np.broadcast_to(np.array([-carrier[1], carrier[0]]), points.shape).copy()
    radius_squared = stir_radius**2
    for x, y, vx, vy in tools:
        delta = points - np.array([x, y])
        dx, dy = delta[..., 0], delta[..., 1]
        gaussian = np.exp(-(dx**2 + dy**2) / (2 * radius_squared))
        dipole = vx * dy - vy * dx
        potential += flow_strength * dipole * gaussian
        gradient[..., 0] += flow_strength * gaussian * (-vy - dipole * dx / radius_squared)
        gradient[..., 1] += flow_strength * gaussian * (vx - dipole * dy / radius_squared)

    pair_radius_squared = (PAIR_RADIUS_RATIO * stir_radius) ** 2
    for x, y, spin in pairs:
        delta = points - np.array([x, y])
        gaussian = np.exp(-np.sum(delta**2, axis=-1) / (2 * pair_radius_squared))
        potential += pair_swirl * spin * pair_radius_squared * gaussian
        gradient -= pair_swirl * spin * gaussian[..., None] * delta

    if pair_strain:
        descriptors = _array(strains, (3, 3), "strains")
        lengths = np.linalg.norm(descriptors[:, :2], axis=-1)
        if np.any(np.abs(descriptors[:, 2]) > 20) or np.any(
            (np.abs(lengths - 1) > 2e-6) & ~((lengths == 0) & (descriptors[:, 2] == 0))
        ):
            raise ValueError("Strains require unit axes and bounded rates, or zero rows")
        for (cx, cy, _), (ax, ay, rate) in zip(pairs, descriptors, strict=True):
            delta = points - np.array([cx, cy])
            axis, normal = np.array([ax, ay]), np.array([-ay, ax])
            x, y = np.sum(delta * axis, axis=-1), np.sum(delta * normal, axis=-1)
            gaussian = np.exp(-np.sum(delta**2, axis=-1) / (2 * pair_radius_squared))
            amplitude = pair_strain * rate * gaussian
            potential += amplitude * x * y
            gradient += amplitude[..., None] * (
                y[..., None] * axis
                + x[..., None] * normal
                - (x * y / pair_radius_squared)[..., None] * delta
            )

    # Extend the compact domain by zero. Squaring makes both the envelope and
    # its first derivative vanish on every edge, including corner points.
    x, y = points[..., 0], points[..., 1]
    half_width = aspect * domain_scale
    half_height = domain_scale
    ex = np.maximum(1 - (x / half_width) ** 2, 0)
    ey = np.maximum(1 - (y / half_height) ** 2, 0)
    boundary = ex**2 * ey**2
    boundary_gradient = np.stack(
        (-4 * x / half_width**2 * ex * ey**2, -4 * y / half_height**2 * ey * ex**2), axis=-1
    )
    full_gradient = gradient * boundary[..., None] + potential[..., None] * boundary_gradient
    return _Evaluation(potential * boundary, full_gradient)


def streamfunction(
    points: ArrayLike,
    tools: ArrayLike,
    pairs: ArrayLike,
    *,
    aspect: float,
    stir_radius: float,
    flow_strength: float,
    pair_swirl: float,
    domain_scale: float = 1.0,
    carrier_velocity: tuple[float, float] = (0.0, 0.0),
    pair_strain: float = 0.0,
    strains: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Evaluate the complete boundary-conditioned streamfunction."""
    return _evaluate(
        points,
        tools,
        pairs,
        aspect=aspect,
        stir_radius=stir_radius,
        flow_strength=flow_strength,
        pair_swirl=pair_swirl,
        domain_scale=domain_scale,
        carrier_velocity=carrier_velocity,
        pair_strain=pair_strain,
        strains=strains,
    ).potential


def velocity(
    points: ArrayLike,
    tools: ArrayLike,
    pairs: ArrayLike,
    *,
    aspect: float,
    stir_radius: float,
    flow_strength: float,
    pair_swirl: float,
    domain_scale: float = 1.0,
    carrier_velocity: tuple[float, float] = (0.0, 0.0),
    pair_strain: float = 0.0,
    strains: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Evaluate ``(d(streamfunction)/dy, -d(streamfunction)/dx)`` analytically."""
    gradient = _evaluate(
        points,
        tools,
        pairs,
        aspect=aspect,
        stir_radius=stir_radius,
        flow_strength=flow_strength,
        pair_swirl=pair_swirl,
        domain_scale=domain_scale,
        carrier_velocity=carrier_velocity,
        pair_strain=pair_strain,
        strains=strains,
    ).gradient
    return np.stack((gradient[..., 1], -gradient[..., 0]), axis=-1)
