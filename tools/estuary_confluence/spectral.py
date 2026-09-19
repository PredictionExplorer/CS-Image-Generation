"""Deterministic 38-band finite-layer pigment optics, 380--750 nm.

Spectral reflectances are synthetic reconstructions of each seeded sRGB color,
using the MIT-licensed Spectral.js seven-basis data. They are NOT measured paint
spectra or a calibrated pigment database. We retain actual material fractions
and the palette's authored, wavelength-independent scattering strengths; the
upstream luminance-squared concentration heuristic is deliberately not used.

Explicit adaptations keep the reconstruction passive and colorimetrically
consistent: neutral white uses unit reflectance; D65-weighted CIE XYZ sums are
normalized to the exact sRGB D65 white point; a small least-norm correction of
the seven basis curves restores their exact target RGB colors under that
integration. The corrected bases remain entirely in [0, 1]. Final reflectance
has a 1e-6 floor to regularize ideal black. Colors outside sRGB
are compressed toward their neutral luminance in linear RGB. No RGB correction
is added to a spectral mixture, and no randomness enters the optical model.

Version 2 composes each independent pigment column over its lower reflector
before averaging the returned spectra. This is an exact separated-column
endpoint under the stated two-flux model, without cross-column light exchange.
Partial mixedness is an authored ensemble interpolation between those columns
and intimate paint, not measured microstructure or reconstructed spatial detail.
Earlier proof implementations retain their own archived runtimes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.estuary.optics import srgb_to_linear

from .optics import _density, _mixedness, _scale, add_layer, finite_layer_rt, palette_coefficients
from .palette import SUPPORTED_CHROMATIC_COUNTS
from .spectral_data import (
    BASE_SPECTRA,
    D65_CIE_XYZ,
    UPSTREAM_COMMIT,
    UPSTREAM_SOURCE_SHA256,
    XYZ_TO_LINEAR_SRGB,
)

VERSION = "confluence-spectral-v2"
WAVELENGTHS = list(range(380, 751, 10))
ROOT = Path(__file__).parent
REFLECTANCE_FLOOR = 1e-6


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def reconstruction(colors_srgb):
    """Reconstruct passive reflectances with the pinned seven-basis method."""
    linear = srgb_to_linear(colors_srgb)
    if linear.ndim == 0 or linear.shape[-1] != 3:
        raise ValueError("Spectral reconstruction needs RGB triples")
    white = linear.min(axis=-1)
    red, green, blue = np.moveaxis(linear - white[..., None], -1, 0)
    weights = np.stack(
        [
            white,
            np.minimum(green, blue),
            np.minimum(red, blue),
            np.minimum(red, green),
            np.maximum(0, np.minimum(red - blue, red - green)),
            np.maximum(0, np.minimum(green - blue, green - red)),
            np.maximum(0, np.minimum(blue - green, blue - red)),
        ],
        axis=-1,
    )
    basis = np.asarray([BASE_SPECTRA[key] for key in ("W", "C", "M", "Y", "R", "G", "B")])
    basis[0] = 1.0
    observer = _integration()[1].T
    targets = np.asarray(
        [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="f8"
    )
    # Minimum Euclidean correction under the three exact color constraints.
    # With these pinned data the largest adjustment is 0.001202 reflectance;
    # every chromatic basis stays strictly inside the passive [0,1] bounds.
    inverse = np.linalg.solve(np.einsum("ij,kj->ik", observer, observer), observer)
    residual = targets - np.einsum("ij,kj->ik", basis, observer)
    basis += np.einsum("ij,jk->ik", residual, inverse)
    if np.any(basis < -1e-12) or np.any(basis > 1 + 1e-12):
        raise ValueError("Pinned spectral basis correction is not passive")
    basis = np.clip(basis, 0, 1)
    return np.clip(np.einsum("...i,ij->...j", weights, basis), REFLECTANCE_FLOOR, 1.0)


def _integration():
    xyz_to_rgb = np.asarray(XYZ_TO_LINEAR_SRGB, dtype="f8")
    xyz = np.asarray(D65_CIE_XYZ, dtype="f8")
    white_xyz = np.linalg.solve(xyz_to_rgb, np.ones(3))
    correction = white_xyz / xyz.sum(axis=1)
    xyz = xyz * correction[:, None]
    rgb = np.einsum("ij,jk->ik", xyz_to_rgb, xyz).T
    return xyz.T, rgb, correction


def _optical_input(palette):
    palette_coefficients(palette)
    return {key: palette[key] for key in ("pigments_srgb", "scattering", "substrate_srgb")}


def build_spectral_material(palette):
    """Return exact, self-identifying spectral coefficients for an artwork archive."""
    optical_input = _optical_input(palette)
    reflectance = reconstruction(palette["pigments_srgb"])
    substrate = reconstruction(palette["substrate_srgb"])
    scattering = np.repeat(np.asarray(palette["scattering"], dtype="f8")[:, None], 38, axis=1)
    ratio = (1 - reflectance) ** 2 / (2 * reflectance)
    xyz, rgb, correction = _integration()
    data = {
        "schema_version": 1,
        "version": VERSION,
        "optical_input_sha256": _identity(optical_input),
        "palette_identity_sha256": palette.get("identity_sha256"),
        "provenance": {
            "upstream": "https://github.com/rvanwijnen/spectral.js",
            "upstream_ref": "3.0.0",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "data_sha256": hashlib.sha256((ROOT / "spectral_data.py").read_bytes()).hexdigest(),
            "license": "MIT",
            "copyright": "2025 Ronald van Wijnen",
            "license_file": "licenses/spectral-js-MIT.txt",
            "license_sha256": hashlib.sha256(
                (ROOT / "licenses/spectral-js-MIT.txt").read_bytes()
            ).hexdigest(),
            "reflectance": (
                "synthetic LHTSS-derived seven-basis reconstruction; not measured pigments"
            ),
            "adaptations": [
                "unit neutral-white basis",
                "reflectance floor 1e-6 and ceiling 1",
                "D65 XYZ discrete-white normalization",
                "passive least-norm basis correction restoring exact RGB colorimetry",
            ],
            "scattering": "authored palette strengths, constant with wavelength",
            "concentrations": "actual physical pigment fractions; no luminance-squared reweighting",
            "gamut_mapping": "linear RGB chroma compression toward neutral luminance",
        },
        "wavelengths_nm": list(WAVELENGTHS),
        "pigment_reflectance": reflectance.tolist(),
        "absorption": (ratio * scattering).tolist(),
        "scattering": scattering.tolist(),
        "substrate_reflectance": substrate.tolist(),
        "xyz_weights_d65": xyz.tolist(),
        "linear_srgb_weights_d65": rgb.tolist(),
        "xyz_to_linear_srgb": copy.deepcopy(XYZ_TO_LINEAR_SRGB),
        "xyz_white_correction": correction.tolist(),
    }
    data["provenance"]["partial_mixing"] = (
        "v2: compose each independent pigment column with the lower-layer "
        "reflectance before area averaging; blend that spectrum with intimate "
        "mixture reflection using recorded mixedness; no resolved strand geometry"
    )
    data["identity_sha256"] = _identity(data)
    return data


def validate_spectral_material(record, palette=None):
    """Validate passive archived coefficients and, when supplied, their palette binding."""
    if type(record) is not dict:
        raise ValueError("Spectral material must be an archived dictionary")
    required = {
        "schema_version",
        "version",
        "optical_input_sha256",
        "palette_identity_sha256",
        "provenance",
        "wavelengths_nm",
        "pigment_reflectance",
        "absorption",
        "scattering",
        "substrate_reflectance",
        "xyz_weights_d65",
        "linear_srgb_weights_d65",
        "xyz_to_linear_srgb",
        "xyz_white_correction",
        "identity_sha256",
    }
    if (
        set(record) != required
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["version"] != VERSION
        or record["wavelengths_nm"] != WAVELENGTHS
    ):
        raise ValueError("Unsupported or incomplete spectral material")
    payload = {key: value for key, value in record.items() if key != "identity_sha256"}
    if _identity(payload) != record["identity_sha256"]:
        raise ValueError("Spectral material identity differs")
    reflection = np.asarray(record["pigment_reflectance"], dtype="f8")
    if reflection.shape not in tuple((n + 1, 38) for n in SUPPORTED_CHROMATIC_COUNTS):
        raise ValueError(
            "Spectral material needs supported chromatic pigments plus chalk, each with 38 bands"
        )
    arrays = {}
    for key, shape, low, high in (
        ("pigment_reflectance", reflection.shape, REFLECTANCE_FLOOR, 1),
        ("absorption", reflection.shape, 0, 1e9),
        ("scattering", reflection.shape, 1e-6, 1000),
        ("substrate_reflectance", (38,), REFLECTANCE_FLOOR, 1),
        ("xyz_weights_d65", (38, 3), 0, 1),
        ("linear_srgb_weights_d65", (38, 3), -2, 2),
    ):
        value = np.asarray(record[key], dtype="f8")
        if (
            value.shape != shape
            or not np.isfinite(value).all()
            or np.any(value < low)
            or np.any(value > high)
        ):
            raise ValueError(f"Invalid spectral {key}")
        arrays[key] = value
    ratio = (1 - reflection) ** 2 / (2 * reflection)
    if not np.allclose(arrays["absorption"], ratio * arrays["scattering"], rtol=1e-12, atol=1e-12):
        raise ValueError("Absorption does not match passive spectral reflectance")
    if palette is not None:
        expected = build_spectral_material(palette)
        if (
            record["optical_input_sha256"] != expected["optical_input_sha256"]
            or record["palette_identity_sha256"] != expected["palette_identity_sha256"]
            or record["provenance"] != expected["provenance"]
        ):
            raise ValueError("Spectral material belongs to another palette or reconstruction")
        for key in (
            "pigment_reflectance",
            "absorption",
            "scattering",
            "substrate_reflectance",
            "xyz_weights_d65",
            "linear_srgb_weights_d65",
            "xyz_to_linear_srgb",
            "xyz_white_correction",
        ):
            if not np.allclose(record[key], expected[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Archived spectral coefficients differ: {key}")
    return copy.deepcopy(record)


def gamut_map(linear_rgb):
    """Bound display RGB by reducing chroma at fixed linear luminance."""
    rgb = np.asarray(linear_rgb, dtype="f8")
    if rgb.ndim == 0 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Linear RGB must be finite triples")
    neutral = np.clip(np.einsum("...i,i->...", rgb, [0.2126, 0.7152, 0.0722]), 0, 1)[..., None]
    delta = rgb - neutral
    limits = np.where(
        delta < 0, neutral / np.maximum(-delta, 1e-30), (1 - neutral) / np.maximum(delta, 1e-30)
    )
    amount = np.minimum(1, limits.min(axis=-1, keepdims=True))
    return np.clip(neutral + amount * delta, 0, 1)


def to_linear_rgb(spectrum, material=None, *, map_gamut=True):
    spectrum = np.asarray(spectrum, dtype="f8")
    if (
        spectrum.ndim == 0
        or spectrum.shape[-1] != 38
        or not np.isfinite(spectrum).all()
        or np.any(spectrum < 0)
        or np.any(spectrum > 1)
    ):
        raise ValueError("Reflectance needs 38 passive finite samples")
    weights = (
        _integration()[1] if material is None else np.asarray(material["linear_srgb_weights_d65"])
    )
    rgb = np.einsum("...i,ij->...j", spectrum, weights)
    return gamut_map(rgb) if map_gamut else rgb


def layer_rt(density, material, *, layer_scale=5.0):
    """Passive R/T of a homogeneous, intimately mixed material phase."""
    scattering = np.asarray(material["scattering"], dtype="f8")
    absorption = np.asarray(material["absorption"], dtype="f8")
    density = _density(density, len(scattering))
    scale = _scale(layer_scale)
    total_scattering = np.einsum("...i,ij->...j", density, scattering)
    total_absorption = np.einsum("...i,ij->...j", density, absorption)
    return finite_layer_rt(
        total_absorption / np.maximum(total_scattering, 1e-30), total_scattering * scale
    )


def independent_layer_reflectance(density, bottom, material, *, layer_scale=5.0, mixedness=1.0):
    """Blend complete reflected spectra of intimate and independent-column paint.

    Each pure column contains the same total mass as the original material;
    its area fraction is that pigment's mass fraction. Both endpoint ensembles
    therefore retain the same mean amount of every pigment. The lower layer
    is treated as a homogeneous reflector at this scale; no unknown alignment
    of microscopic structures across layers is invented.
    """
    scattering = np.asarray(material["scattering"], dtype="f8")
    absorption = np.asarray(material["absorption"], dtype="f8")
    density = _density(density, len(scattering))
    scale = _scale(layer_scale)
    mixed = _mixedness(mixedness, density.shape[:-1])
    bottom = np.asarray(bottom, dtype="f8")
    if (
        bottom.shape[-1:] != (38,)
        or not np.isfinite(bottom).all()
        or np.any(bottom < 0)
        or np.any(bottom > 1)
    ):
        raise ValueError("The lower spectral reflector must contain 38 passive samples")
    ir, it = layer_rt(density, material, layer_scale=scale)
    intimate = add_layer(bottom, ir, it)
    if np.all(mixed == 1):
        return np.clip(intimate, 0, 1)
    mass = density.sum(axis=-1, keepdims=True)
    fractions = density / np.maximum(mass, 1e-30)
    area = np.zeros_like(intimate)
    for index in range(len(scattering)):
        pr, pt = finite_layer_rt(
            absorption[index] / scattering[index], mass * scattering[index] * scale
        )
        area += fractions[..., index, None] * add_layer(bottom, pr, pt)
    area = np.where(mass > 0, area, bottom)
    reflected = mixed * intimate + (1 - mixed) * area
    return np.clip(np.where(mass > 0, reflected, bottom), 0, 1)


def layer_spectra(layers, material, *, layer_scale=5.0, mixedness=1.0, normalized_mass=None):
    """Evaluate real phases bottom-to-top; return bounded wavelength reflectance."""
    count = len(material["pigment_reflectance"])
    layers = [_density(layer, count) for layer in layers]
    if not layers or any(layer.shape != layers[0].shape for layer in layers):
        raise ValueError("Spectral phases must have matching dimensions")
    if normalized_mass is not None:
        if (
            type(normalized_mass) not in (int, float)
            or not np.isfinite(normalized_mass)
            or not 1e-4 <= normalized_mass <= 10
        ):
            raise ValueError("Normalized optical mass must be in [1e-4,10]")
        mass = sum(layer.sum(axis=-1) for layer in layers)
        factor = np.divide(normalized_mass, mass, out=np.zeros_like(mass), where=mass > 1e-20)
        layers = [layer * factor[..., None] for layer in layers]
    color = np.asarray(material["substrate_reflectance"], dtype="f8")
    for layer in layers:
        color = independent_layer_reflectance(
            layer, color, material, layer_scale=layer_scale, mixedness=mixedness
        )
    return np.clip(color, 0, 1)


def reflectance(density, material, *, layer_scale=5.0, mixedness=1.0, normalized_mass=None):
    return to_linear_rgb(
        layer_spectra(
            [density],
            material,
            layer_scale=layer_scale,
            mixedness=mixedness,
            normalized_mass=normalized_mass,
        ),
        material,
    )


def layered_reflectance(
    underpaint, deposit, mobile, material, *, layer_scale=5.0, mixedness=1.0, normalized_mass=None
):
    return to_linear_rgb(
        layer_spectra(
            [underpaint, deposit, mobile],
            material,
            layer_scale=layer_scale,
            mixedness=mixedness,
            normalized_mass=normalized_mass,
        ),
        material,
    )
