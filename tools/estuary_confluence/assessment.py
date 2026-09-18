"""Read-only, palette-independent participation and composition measurements.

These describe pigment concentrations, not artistic quality. Coarse history
samples can merge unresolved strands; the final report always uses the native
material grid. No metric feeds back into or changes the simulation.
"""

from __future__ import annotations

import math

import numpy as np

VERSION = "pigment-participation-v2"


def _value(value):
    return round(float(value), 10)


def assess(pigment, chromatic_count, domain_scale, *, mass_threshold=0.008, share_threshold=0.1):
    """Measure each pigment's visible mass, shared mass, motion anchor and spread.

    ``contact_mass_fraction`` counts a pigment's mass where it and at least one
    other chromatic pigment each represent ``share_threshold`` of local paint.
    Its denominator includes all of that pigment on the visible canvas, so
    diluted, invisible remnants cannot inflate participation.
    """
    p = np.asarray(pigment)
    if (
        p.ndim != 3
        or p.dtype.kind not in "fiu"
        or type(chromatic_count) is not int
        or chromatic_count not in (3, 5)
        or p.shape[2] != chromatic_count + 1
        or min(p.shape[:2]) < 1
        or not np.isfinite(p).all()
        or np.any(p < 0)
    ):
        raise ValueError("Assessment requires finite, nonnegative pigment channels")
    for name, value, low, high in (
        ("domain_scale", domain_scale, 1, 3),
        ("mass_threshold", mass_threshold, 1e-8, 1),
        ("share_threshold", share_threshold, 0.001, 0.5),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid assessment {name}")
    h, w = p.shape[:2]
    aspect = w / h
    x = ((np.arange(w, dtype="f8") + 0.5) / w * 2 - 1) * aspect * domain_scale
    y = ((np.arange(h, dtype="f8") + 0.5) / h * 2 - 1) * domain_scale
    ix = np.flatnonzero(np.abs(x) <= aspect)
    iy = np.flatnonzero(np.abs(y) <= 1)
    if not len(ix) or not len(iy):
        raise ValueError("Assessment grid has no visible samples")
    visible = p[iy[0] : iy[-1] + 1, ix[0] : ix[-1] + 1, :chromatic_count]
    x, y = x[ix], y[iy]
    total = visible.sum(axis=2, dtype="f8")
    painted = total > mass_threshold
    pixels = int(painted.sum())
    cell_area = (2 * domain_scale / h) ** 2
    global_mass = p[..., :chromatic_count].sum(axis=(0, 1), dtype="f8") * cell_area
    visible_mass = visible.sum(axis=(0, 1), dtype="f8") * cell_area

    def moments(weight):
        mass = weight.sum(dtype="f8")
        if mass <= 0:
            return None
        wx = weight.sum(axis=0, dtype="f8")
        wy = weight.sum(axis=1, dtype="f8")
        cx, cy = float(wx @ x / mass), float(wy @ y / mass)
        return {
            "centroid": [_value(cx), _value(cy)],
            "normalized_centroid": [_value(cx / aspect), _value(cy)],
            "rms_radius": _value(
                math.sqrt(max(0, float(wx @ (x - cx) ** 2 + wy @ (y - cy) ** 2) / mass))
            ),
        }

    significant = visible >= total[..., None] * share_threshold
    significant &= painted[..., None]
    contributors = significant.sum(axis=2)
    shared = contributors >= 2
    # Split exact ties rather than awarding equal mixtures to channel zero.
    dominant = visible == visible.max(axis=2, keepdims=True)
    dominant_share = dominant / dominant.sum(axis=2, keepdims=True)
    reports = []
    for index in range(chromatic_count):
        own = visible[..., index]
        own_visible = float(visible_mass[index])
        display_mass = own[painted].sum(dtype="f8") * cell_area
        contact_mass = own[shared & significant[..., index]].sum(dtype="f8") * cell_area
        reports.append(
            {
                "pigment_index": index,
                "mass": _value(global_mass[index]),
                "visible_mass_fraction": _value(own_visible / global_mass[index])
                if global_mass[index] > 0
                else 0.0,
                "displayed_mass_fraction": _value(display_mass / own_visible)
                if own_visible > 0
                else 0.0,
                "contact_mass_fraction": _value(contact_mass / own_visible)
                if own_visible > 0
                else 0.0,
                "dominant_area_fraction": _value(
                    dominant_share[..., index][painted].sum(dtype="f8") / pixels
                )
                if pixels
                else 0.0,
                "moments": moments(own),
            }
        )
    return {
        "version": VERSION,
        "grid_resolution": [w, h],
        "visible_resolution": [len(ix), len(iy)],
        "domain_scale": domain_scale,
        "mass_threshold": mass_threshold,
        "share_threshold": share_threshold,
        "painted_area_fraction": _value(pixels / painted.size),
        "shared_painted_area_fraction": _value(np.count_nonzero(shared) / pixels)
        if pixels
        else 0.0,
        "painted_area_moments": moments(painted),
        "paint_mass_moments": moments(total),
        "pigments": reports,
        "minimum_contact_mass_fraction": min(r["contact_mass_fraction"] for r in reports),
    }


def image_balance(linear_rgb):
    """Perceived image balance using darkness against the intended white ground."""
    rgb = np.asarray(linear_rgb)
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or not np.isfinite(rgb).all()
        or np.any((rgb < 0) | (rgb > 1))
    ):
        raise ValueError("Image balance requires bounded finite linear RGB")
    h, w = rgb.shape[:2]
    weight = 1 - np.einsum("...i,i->...", rgb, [0.2126, 0.7152, 0.0722])
    mass = weight.sum(dtype="f8")
    if mass <= h * w * 1e-10:
        return None
    x = (np.arange(w) + 0.5) / w * 2 - 1
    y = 1 - (np.arange(h) + 0.5) / h * 2
    return [
        _value(weight.sum(axis=0, dtype="f8") @ x / mass),
        _value(weight.sum(axis=1, dtype="f8") @ y / mass),
    ]
