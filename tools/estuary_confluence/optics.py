"""N-pigment finite-layer RGB Kubelka--Munk reference.

These are authored three-band optical coefficients, not measured spectra. The
mixture parameter interpolates passive reflection/transmission operators between
an intimate pigment mixture and unresolved adjacent single-pigment columns.
The areal columns have equal total pigment mass per area and mass-proportional
area fractions: changing mixedness cannot create or remove a pigment. Applying
an effective areal operator in a stack assumes lateral statistical independence
between layers; it does not resolve individual subpixel strands or refraction.
"""

from __future__ import annotations

import math

import numpy as np

from tools.estuary.optics import absorption_over_scattering, srgb_to_linear

from .palette import SUPPORTED_CHROMATIC_COUNTS


def palette_coefficients(palette):
    """Validate the optical subset of an archived palette without changing it."""
    if not isinstance(palette, dict):
        raise ValueError("palette must be a dictionary")
    colors = np.asarray(palette.get("pigments_srgb"), dtype=np.float64)
    if colors.ndim != 2 or colors.shape not in tuple(
        (n + 1, 3) for n in SUPPORTED_CHROMATIC_COUNTS
    ):
        raise ValueError("palette needs one, two, three, or five colors plus chalk")
    colors = srgb_to_linear(colors)
    substrate = srgb_to_linear(palette.get("substrate_srgb"))
    if substrate.shape != (3,):
        raise ValueError("substrate_srgb must be an RGB triple")
    scatter = np.asarray(palette.get("scattering"), dtype=np.float64)
    if (
        scatter.shape != (len(colors),)
        or not np.isfinite(scatter).all()
        or np.any(scatter < 1e-6)
        or np.any(scatter > 1e3)
    ):
        raise ValueError("scattering needs one finite strength in [1e-6, 1000] per pigment")
    return absorption_over_scattering(colors), scatter, substrate


def _scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e4:
        raise ValueError("layer_scale must be finite and in [0, 10000]")
    return float(value)


def _density(value, count):
    density = np.asarray(value, dtype=np.float64)
    if (
        density.ndim == 0
        or density.shape[-1] != count
        or not np.isfinite(density).all()
        or np.any(density < 0)
        or np.any(density > 1e6)
    ):
        raise ValueError("pigment density must have N finite concentrations in [0, 1e6]")
    return density


def _mixedness(value, shape):
    mixed = np.asarray(value, dtype=np.float64)
    if not np.isfinite(mixed).all() or np.any(mixed < 0) or np.any(mixed > 1):
        raise ValueError("mixedness must be finite and in [0, 1]")
    try:
        return np.broadcast_to(mixed, shape)[..., None]
    except ValueError as exc:
        raise ValueError("mixedness must broadcast to the material field") from exc


def glazed_density_scale(
    total_mass, *, mass_reference=0.15, min_mass_ratio=0.35, max_mass_ratio=2.5
):
    """Bound actual optical thickness while preserving pigment/layer fractions.

    A glaze uses actual mass between two explicit bounds. The lower bound keeps
    a colored material edge; the upper bound limits opaque saturation. This is
    an authored optical interpretation, never a change to stored pigment mass.
    """
    for value, name, low, high in (
        (mass_reference, "mass_reference", 1e-4, 10),
        (min_mass_ratio, "min_mass_ratio", 0.01, 1),
        (max_mass_ratio, "max_mass_ratio", 1, 8),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    mass = np.asarray(total_mass, dtype="f8")
    if not np.isfinite(mass).all() or np.any(mass < 0) or np.any(mass > 6e6):
        raise ValueError("total_mass must be finite and in [0, 6000000]")
    target = np.clip(mass, mass_reference * min_mass_ratio, mass_reference * max_mass_ratio)
    return np.divide(target, mass, out=np.zeros_like(mass), where=mass > 1e-20)


def finite_layer_rt(ratio, thickness):
    """Return symmetric layer reflection and transmission in linear RGB.

    Uses an analytic pure-scattering limit and negative exponentials, avoiding
    overflow in opaque layers and cancellation in almost transparent ones.
    """
    ratio, thickness = np.broadcast_arrays(
        np.asarray(ratio, dtype="f8"), np.asarray(thickness, dtype="f8")
    )
    if (
        not np.isfinite(ratio).all()
        or not np.isfinite(thickness).all()
        or np.any(ratio < 0)
        or np.any(thickness < 0)
    ):
        raise ValueError("ratio and thickness must be finite and nonnegative")
    b = np.sqrt(ratio * (ratio + 2.0))
    x = b * thickness
    difference = -np.expm1(-2.0 * x)
    denominator = b * (2.0 - difference) + (1.0 + ratio) * difference
    regular = b > 1e-6
    safe = np.where(regular, denominator, 1.0)
    # Both formula branches are evaluated; no 0/0 is evaluated in either.
    r = np.where(regular, difference / np.maximum(safe, 1e-300), thickness / (1.0 + thickness))
    t = np.where(regular, 2.0 * b * np.exp(-x) / np.maximum(safe, 1e-300), 1.0 / (1.0 + thickness))
    zero = thickness == 0
    return np.where(zero, 0.0, r), np.where(zero, 1.0, t)


def layer_rt(density, palette, *, layer_scale=5.0, mixedness=1.0):
    """Passive effective layer operator; mixedness=1 intimate, 0 areal."""
    ratios, strengths, _ = palette_coefficients(palette)
    density = _density(density, len(strengths))
    scale = _scale(layer_scale)
    mixed = _mixedness(mixedness, density.shape[:-1])
    scattering = density * strengths
    total_scattering = scattering.sum(axis=-1, keepdims=True)
    ratio = (scattering @ ratios) / np.maximum(total_scattering, 1e-30)
    intimate_r, intimate_t = finite_layer_rt(ratio, total_scattering * scale)
    total_mass = density.sum(axis=-1, keepdims=True)
    fractions = density / np.maximum(total_mass, 1e-30)
    areal_r = np.zeros_like(intimate_r)
    areal_t = np.zeros_like(intimate_t)
    for index in range(len(strengths)):
        r, t = finite_layer_rt(ratios[index], total_mass * strengths[index] * scale)
        areal_r += fractions[..., index, None] * r
        areal_t += fractions[..., index, None] * t
    areal_t = np.where(total_mass > 0, areal_t, 1.0)
    return mixed * intimate_r + (1 - mixed) * areal_r, mixed * intimate_t + (1 - mixed) * areal_t


def add_layer(bottom, reflection, transmission):
    """Add a symmetric passive layer above an opaque reflective substrate."""
    return reflection + transmission * transmission * bottom / np.maximum(
        1.0 - reflection * bottom, 1e-12
    )


def reflectance(pigment, palette, *, layer_scale=5.0, mixedness=1.0):
    """Evaluate a single homogeneous material layer over the archived ground."""
    _, _, substrate = palette_coefficients(palette)
    r, t = layer_rt(pigment, palette, layer_scale=layer_scale, mixedness=mixedness)
    return np.clip(add_layer(substrate, r, t), 0.0, 1.0)


def layered_reflectance(underpaint, deposit, mobile, palette, *, layer_scale=5.0, mixedness=1.0):
    """Stack real retained phases bottom-to-top, including finite transmission."""
    _, scatter, color = palette_coefficients(palette)
    layers = [_density(value, len(scatter)) for value in (underpaint, deposit, mobile)]
    if any(value.shape != layers[0].shape for value in layers):
        raise ValueError("all phase arrays must have exactly the same shape")
    for layer in layers:
        r, t = layer_rt(layer, palette, layer_scale=layer_scale, mixedness=mixedness)
        color = add_layer(color, r, t)
    return np.clip(color, 0.0, 1.0)
