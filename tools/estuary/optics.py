"""Reflective pigment optics for the Estuary's three concentration channels.

This is a controllable Kubelka--Munk approximation, not a spectral reconstruction
of actual artists' pigments. RGB reflectances are converted to absorption over
scattering (K/S), mixed there, and evaluated as a finite layer over a substrate.
All calculations and the GLSL output are in linear sRGB; display encoding belongs
to the image writer. Neither the reference model nor this configuration imports a
GPU library.

The shader's optional micrograin belongs to the stationary substrate. It is a
continuous spatial field, with no frame number, screen-size, or random-seed input.
It is intentionally excluded from ``reflectance`` so numerical tests describe the
pigment model itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

RGB = tuple[float, float, float]
REFLECTANCE_FLOOR = 1.0e-6
MAX_CONCENTRATION = 1.0e6


def _triple(value: ArrayLike, name: str, minimum: float, maximum: float) -> RGB:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite numbers")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return float(array[0]), float(array[1]), float(array[2])


def srgb_to_linear(value: ArrayLike) -> NDArray[np.float64]:
    """Decode normalized sRGB display values without changing their shape."""
    encoded = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(encoded)) or np.any(encoded < 0) or np.any(encoded > 1):
        raise ValueError("sRGB values must be finite and in [0, 1]")
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: ArrayLike) -> NDArray[np.float64]:
    """Encode normalized linear sRGB reflectance for a display or PNG writer."""
    linear = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(linear)) or np.any(linear < 0) or np.any(linear > 1):
        raise ValueError("linear RGB values must be finite and in [0, 1]")
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def absorption_over_scattering(reflectance: ArrayLike) -> NDArray[np.float64]:
    """Return F(R)=(1-R)^2/(2R), with an explicit ideal-black regularization.

    The lower reflectance bound avoids an infinite absorption coefficient for an
    ideal black. It is shared verbatim with the float32 shader and is well below
    the visible black level of the intended display output.
    """
    color = np.asarray(reflectance, dtype=np.float64)
    if not np.all(np.isfinite(color)) or np.any(color < 0) or np.any(color > 1):
        raise ValueError("reflectance must be finite and in [0, 1]")
    safe = np.maximum(color, REFLECTANCE_FLOOR)
    return (1 - safe) ** 2 / (2 * safe)


@dataclass(frozen=True)
class Material:
    """Three pigment colors, scattering strengths, and a reflective substrate.

    Palette colors are authored in sRGB and decoded when bound to the shader.
    Concentrations and ``layer_scale`` define relative optical thickness, not
    millimeters. Pigment order is ultramarine, mineral white, and vermilion.
    """

    pigments_srgb: tuple[RGB, RGB, RGB] = (
        (0.055, 0.105, 0.36),
        (0.93, 0.915, 0.86),
        (0.62, 0.165, 0.105),
    )
    substrate_srgb: RGB = (0.93, 0.915, 0.86)
    scattering: RGB = (0.72, 1.0, 0.82)
    layer_scale: float = 7.0
    grain: float = 0.01
    grain_frequency: tuple[float, float] = (1400.0, 1400.0)

    def __post_init__(self) -> None:
        if len(self.pigments_srgb) != 3:
            raise ValueError("exactly three pigments are required")
        for index, color in enumerate(self.pigments_srgb):
            _triple(color, f"pigment {index}", 0.0, 1.0)
        _triple(self.substrate_srgb, "substrate", 0.0, 1.0)
        _triple(self.scattering, "scattering", 1.0e-6, 1.0e3)
        if not np.isfinite(self.layer_scale) or not 0 <= self.layer_scale <= 1.0e4:
            raise ValueError("layer_scale must be finite and in [0, 10000]")
        if not np.isfinite(self.grain) or not 0 <= self.grain <= 0.08:
            raise ValueError("grain must be finite and in [0, 0.08]")
        frequency = np.asarray(self.grain_frequency, dtype=np.float64)
        if (
            frequency.shape != (2,)
            or not np.all(np.isfinite(frequency))
            or np.any(frequency < 1)
            or np.any(frequency > 16384)
        ):
            raise ValueError("grain_frequency must contain two finite values in [1, 16384]")

    def uniforms(self, texture_unit: int = 0) -> dict[str, Any]:
        """Return ordinary values ready for ``program[name].value = value``.

        ``u_paint`` contains nonnegative RGB pigment concentrations. The alpha
        channel is reserved for the simulation and never sampled by the optics.
        The vertex shader must emit normalized field coordinates as ``v_uv``.
        The render target receives opaque linear-sRGB RGBA.
        """
        if isinstance(texture_unit, bool) or not isinstance(texture_unit, int) or texture_unit < 0:
            raise ValueError("texture_unit must be a nonnegative integer")
        colors = srgb_to_linear(self.pigments_srgb)
        return {
            "u_paint": texture_unit,
            "u_pigment_0": tuple(colors[0]),
            "u_pigment_1": tuple(colors[1]),
            "u_pigment_2": tuple(colors[2]),
            "u_scattering": tuple(self.scattering),
            "u_substrate": tuple(srgb_to_linear(self.substrate_srgb)),
            "u_layer_scale": float(self.layer_scale),
            "u_grain": float(self.grain),
            "u_grain_frequency": tuple(self.grain_frequency),
        }


def shader_source() -> str:
    """Load the versioned standalone GLSL 330 fragment shader."""
    return Path(__file__).with_suffix(".glsl").read_text(encoding="utf-8")


def reflectance(concentrations: ArrayLike, material: Material | None = None) -> NDArray[np.float64]:
    """Evaluate a finite pigment layer over the substrate, returning linear RGB.

    ``concentrations`` has shape ``(..., 3)``. Values beyond 1e6 are already far
    beyond optical saturation and are clamped, identically to the shader, to keep
    all arithmetic finite. Zero thickness returns the substrate exactly. The
    pure-scattering limit is analytic, including a perfectly white pigment.
    """
    material = material if material is not None else Material()
    density = np.asarray(concentrations, dtype=np.float64)
    if density.ndim == 0 or density.shape[-1] != 3:
        raise ValueError("concentrations must have shape (..., 3)")
    if not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("concentrations must be finite and nonnegative")
    density = np.minimum(density, MAX_CONCENTRATION)
    strengths = density * np.asarray(material.scattering)
    scattering = strengths.sum(axis=-1, keepdims=True)
    absorption = strengths @ absorption_over_scattering(srgb_to_linear(material.pigments_srgb))
    ratio = absorption / np.where(scattering > 0, scattering, 1.0)
    a = 1 + ratio
    b = np.sqrt(ratio * (ratio + 2))
    thickness = scattering * material.layer_scale
    exponent = b * thickness
    one_minus_exp = -np.expm1(-2 * exponent)
    denominator = b * (2 - one_minus_exp) + a * one_minus_exp
    regular = b > 1.0e-6
    safe_denominator = np.where(regular & (denominator > 0), denominator, 1.0)
    layer_r = np.where(regular, one_minus_exp / safe_denominator, thickness / (1 + thickness))
    layer_t = np.where(regular, 2 * b * np.exp(-exponent) / safe_denominator, 1 / (1 + thickness))
    # For a non-scattering empty layer, the limiting transmission is exactly 1.
    empty = thickness == 0
    layer_r = np.where(empty, 0.0, layer_r)
    layer_t = np.where(empty, 1.0, layer_t)
    substrate = srgb_to_linear(material.substrate_srgb)
    result = layer_r + layer_t**2 * substrate / np.maximum(1 - layer_r * substrate, 1.0e-12)
    return np.clip(result, 0.0, 1.0)
