"""White-albedo EON rough-diffuse angular redistribution.

This independently derived implementation specializes equations 10, 15 and 19
of Portsmouth, Kutz and Hill, *EON: A practical energy-preserving rough diffuse
BRDF* (2024), https://arxiv.org/abs/2410.18026, to a white single-scatter albedo.
Multiplying this normalized scalar kernel by the existing pigment reflectance
preserves that reflectance's directional-hemispherical integral. No pigment
coefficient, spectrum, material quantity, height, or normal is changed.

The return value is pi times the BRDF. Its projected hemisphere integral is pi,
and its zero-roughness value is exactly one (Lambert). This conservation claim
applies to the diffuse kernel; the renderer's existing dielectric/diffuse lobe
coupling and approximate illumination are not a full OpenPBR implementation.
"""

from __future__ import annotations

import numpy as np

EON_C1 = 0.5 - 2.0 / (3.0 * np.pi)
EON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * np.pi)


def aggregate_response(aggregate, contrast=1.0):
    """Authored crowding response g**c / (g**c + (1-g)**c), c in [1,8].

    This percolation-inspired optical response is not a measured threshold or
    particle-connectivity model. It changes only the renderer's response to the
    existing aggregate fraction; material quantities and packing still use raw g.
    The symmetric ratio formulation has no small denominator or overflow, and
    the identity branch deliberately preserves the existing linear response.
    """
    g, c = np.asarray(aggregate, dtype="f8"), np.asarray(contrast, dtype="f8")
    if (
        not np.isfinite(g).all()
        or np.any(g < 0)
        or np.any(g > 1)
        or not np.isfinite(c).all()
        or np.any(c < 1)
        or np.any(c > 8)
    ):
        raise ValueError("Aggregate must lie in [0,1] and optical contrast in [1,8]")
    ratio = np.minimum(g, 1 - g) / np.maximum(g, 1 - g)
    power = np.power(ratio, c)
    response = np.where(g <= 0.5, power / (1 + power), 1 / (1 + power))
    return np.where(c == 1, g, response)


def _direction(value, name):
    result = np.asarray(value, dtype="f8")
    if (
        result.shape[-1:] != (3,)
        or not np.isfinite(result).all()
        or np.any(np.abs(result) > 1.0000000001)
        or np.any(result[..., 2] < 0)
        or not np.allclose(np.linalg.norm(result, axis=-1), 1, rtol=0, atol=2e-7)
    ):
        raise ValueError(f"{name} must contain unit directions in the upper hemisphere")
    return result


def _roughness(value):
    result = np.asarray(value, dtype="f8")
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError("Diffuse roughness must be finite and in [0, 1]")
    return result


def _albedo_deficit(cosine):
    """Return the roughness-independent factor in the FON albedo deficit.

    Rewriting (sin(theta)/cos(theta)) * (1-sin(theta)^3) using
    1-sin(theta)=cos(theta)^2/(1+sin(theta)) removes both the grazing division
    and subtractive cancellation. Its limit at cosine=0 is evaluated directly.
    """
    cosine = np.clip(cosine, 0, 1)
    sine = np.sqrt(np.maximum(0, 1 - cosine * cosine))
    factor = sine * (np.arccos(cosine) - sine * cosine)
    factor += (2.0 / 3.0) * (sine * cosine * (1 + sine + sine * sine) / (1 + sine) - sine)
    return np.clip(EON_C1 - factor / np.pi, 0, EON_C1)


def white_eon(incoming, outgoing, roughness):
    """Evaluate pi times the reciprocal white EON BRDF in a local normal frame.

    Directions point away from the surface and broadcast over their leading
    axes. At the exactly tangential/tangential singular configuration we return
    zero: the projected incident contribution has zero measure there. All
    strictly positive angle cosines use the analytical expression without an
    epsilon cap. The explicit smooth branch returns exactly one.
    """
    incoming = _direction(incoming, "incoming")
    outgoing = _direction(outgoing, "outgoing")
    r = _roughness(roughness)
    ci, co = incoming[..., 2], outgoing[..., 2]
    tangent_dot = np.sum(incoming[..., :2] * outgoing[..., :2], axis=-1)
    denominator = np.maximum(ci, co)
    positive = np.divide(
        tangent_dot,
        denominator,
        out=np.zeros_like(tangent_dot),
        where=denominator > 0,
    )
    angular = np.where(tangent_dot > 0, positive, tangent_dot)
    # For white albedo, rho_ms=1. Cancelling the common A*r factor in
    # (1-Ei)*(1-Eo)/(1-Eavg) yields this stable zero-roughness limit.
    multiple = _albedo_deficit(ci) * _albedo_deficit(co) / (EON_C1 - EON_C2)
    result = (1 + r * (angular + multiple)) / (1 + EON_C1 * r)
    return np.where(r == 0, 1.0, np.where(denominator == 0, 0.0, result))
