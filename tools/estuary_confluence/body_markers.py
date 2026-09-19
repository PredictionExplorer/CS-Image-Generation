"""Screen-space diagnostic glyphs anchored to the three forcing-plane positions.

Source positions use the visible canvas coordinates x∈[-aspect,aspect], y∈[-1,1].
They already include the source's fixed projection/fitting; applying its rotation
again or applying the simulation guard-domain scale would misplace these guides.
We project z=0 through Surface's orthographic camera and draw after rendering.
Glyphs therefore remain visible over paint and never enter material state.

Input/output images are top-down float32 linear display RGB, after tone mapping
and before display encoding. Fixed vector strokes avoid fonts, assets, random
state and platform text rasterizers. Pixel centers are measured at n+0.5 from
the top-left image edge; projected marker centers use that same edge convention.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from tools.estuary_studio.surface import _number, camera_basis

VERSION = "body-markers-v1"
DEFAULTS = {
    "version": VERSION,
    "labels": True,
    "size_px_1080": 36.0,
    "stroke_px_1080": 3.0,
}

# Fixed centerline paths in a unit-height label box. Labels are intentionally
# diagnostic, using open vector strokes rather than an installed font.
_LABEL_PATHS = (
    (((0.0, 0.25), (0.45, 0.0)), ((0.45, 0.0), (0.45, 1.0)), ((0.1, 1.0), (0.85, 1.0))),
    (
        ((0.0, 0.0), (0.85, 0.0)),
        ((0.85, 0.0), (0.85, 0.5)),
        ((0.85, 0.5), (0.0, 0.5)),
        ((0.0, 0.5), (0.0, 1.0)),
        ((0.0, 1.0), (0.85, 1.0)),
    ),
    (
        ((0.0, 0.0), (0.85, 0.0)),
        ((0.85, 0.0), (0.85, 1.0)),
        ((0.0, 0.5), (0.85, 0.5)),
        ((0.0, 1.0), (0.85, 1.0)),
    ),
)


def validate_config(value=None):
    """Normalize the opt-in extension without changing legacy/disabled recipes."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Body marker config must contain only documented controls")
    result = {**DEFAULTS, **copy.deepcopy(value)}
    if type(result["version"]) is not str or result["version"] != VERSION:
        raise ValueError(f"Body marker version must be {VERSION}")
    if type(result["labels"]) is not bool:
        raise ValueError("Body marker labels must be a boolean")
    result["size_px_1080"] = _number(result["size_px_1080"], "marker size", 12, 96)
    result["stroke_px_1080"] = _number(result["stroke_px_1080"], "marker stroke", 1, 8)
    if result["stroke_px_1080"] > result["size_px_1080"] / 3:
        raise ValueError("Marker stroke must not exceed one third of marker size")
    return result


def _size(value):
    if (
        type(value) not in (list, tuple)
        or len(value) != 2
        or any(type(v) is not int or not 4 <= v <= 12288 for v in value)
        or value[0] * value[1] > 50_000_000
    ):
        raise ValueError("Marker output requires supported integer width and height")
    return tuple(value)


def project_positions(positions, size, *, tilt_degrees=0.0, azimuth_degrees=-30.0):
    """Project exactly three z=0 source positions to top-left pixel-edge centers.

    Source aspect must equal the output aspect, as required by the renderer.
    Out-of-frame positions are retained (not clamped, shifted or occluded).
    Canvas metres and the guard-domain scale cancel in this plane projection.
    """
    width, height = _size(size)
    p = np.asarray(positions)
    if p.shape != (3, 2) or p.dtype.kind not in "fiu":
        raise ValueError("Body positions must be a numeric array with shape (3, 2)")
    p = p.astype("f8", copy=False)
    if not np.isfinite(p).all():
        raise ValueError("Body positions must be finite")
    basis = camera_basis(tilt_degrees, azimuth_degrees)
    with np.errstate(over="ignore", invalid="ignore"):
        projected = np.column_stack(
            (
                width / 2 + (height / 2) * np.einsum("ij,j->i", p, basis[:2, 0]),
                height / 2 - (height / 2) * np.einsum("ij,j->i", p, basis[:2, 1]),
            )
        )
    if not np.isfinite(projected).all():
        raise ValueError("Body positions exceed finite projection range")
    return projected


def _glyphs(centers, settings, image_height):
    scale = image_height / 1080.0
    size, stroke = settings["size_px_1080"] * scale, settings["stroke_px_1080"] * scale
    half = (size - stroke) / 2
    cross = np.array([[[-half, -half], [half, half]], [[-half, half], [half, -half]]], dtype="f8")
    glyphs = []
    for index, center in enumerate(centers):
        glyphs.append((cross + center, stroke))
        if settings["labels"]:
            label = np.asarray(_LABEL_PATHS[index], dtype="f8")
            label_width, label_height = size * 0.24, size * 0.36
            label = label * np.array([label_width, label_height])
            # Fixed NE/NW/SE label quadrants distinguish coincident bodies.
            # Only labels use these offsets: the X centers never move.
            right = size / 2 + 4 * scale
            offsets = (
                (right, -size / 2),
                (-right - 0.85 * label_width, -size / 2),
                (right, size / 2 - label_height),
            )
            label += center + np.asarray(offsets[index])
            glyphs.append((label, stroke * 0.7))
    return glyphs, 1.2 * scale


def _stroke(image, segments, width, color):
    """Bounded local distance-to-segment rasterization with one-pixel AA."""
    height, image_width = image.shape[:2]
    radius = width / 2
    low = segments.min(axis=(0, 1)) - radius - 0.5
    high = segments.max(axis=(0, 1)) + radius + 0.5
    # Clip before constructing any coordinate arrays, including wholly offscreen
    # bodies. No coordinate clamping or marker-center displacement is performed.
    if high[0] <= 0 or high[1] <= 0 or low[0] >= image_width or low[1] >= height:
        return
    x0, y0 = max(0, math.floor(low[0])), max(0, math.floor(low[1]))
    x1, y1 = min(image_width, math.ceil(high[0])), min(height, math.ceil(high[1]))
    x = np.arange(x0, x1, dtype="f8")[None, :] + 0.5
    y = np.arange(y0, y1, dtype="f8")[:, None] + 0.5
    coverage = np.zeros((y1 - y0, x1 - x0), dtype="f8")
    for first, last in segments:
        dx, dy = last - first
        fraction = np.clip(((x - first[0]) * dx + (y - first[1]) * dy) / (dx * dx + dy * dy), 0, 1)
        distance = np.hypot(x - (first[0] + fraction * dx), y - (first[1] + fraction * dy))
        coverage = np.maximum(coverage, np.clip(radius + 0.5 - distance, 0, 1))
    affected = coverage > 0
    if not affected.any():
        return
    region = image[y0:y1, x0:x1]
    alpha = coverage[affected].astype("f4")[:, None]
    marked = region[affected] * (1 - alpha) + np.asarray(color, dtype="f4") * alpha
    region[affected] = np.clip(marked, 0, 1)


def annotate(pixels, positions, *, tilt_degrees=0.0, azimuth_degrees=-30.0, config=None):
    """Return an independent float32 image; every unmarked pixel stays bit-identical.

    Disabled overlays copy pixels exactly and do not inspect source positions or
    camera parameters. Enabled overlays validate bounded linear display RGB.
    All halos precede all red strokes so nearby bodies cannot erase each other's
    red centers. Intersecting trajectories keep their exact projected centers.
    """
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.float32
        or pixels.ndim != 3
        or pixels.shape[-1] != 3
    ):
        raise ValueError("Body marker pixels must be an H by W by 3 float32 array")
    settings = validate_config(config)
    result = pixels.copy()
    if settings is None:
        return result
    _size((pixels.shape[1], pixels.shape[0]))
    low, high = pixels.min(), pixels.max()
    if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high > 1:
        raise ValueError("Body marker pixels must be finite linear RGB in [0,1]")
    centers = project_positions(
        positions,
        (pixels.shape[1], pixels.shape[0]),
        tilt_degrees=tilt_degrees,
        azimuth_degrees=azimuth_degrees,
    )
    glyphs, halo = _glyphs(centers, settings, pixels.shape[0])
    # Black outer and white inner outlines remain distinguishable on bright,
    # dark and red paint. The inner diagnostic stroke is always pure red.
    for extra, color in ((4 * halo, (0, 0, 0)), (2 * halo, (1, 1, 1)), (0, (1, 0, 0))):
        for segments, width in glyphs:
            _stroke(result, segments, width + extra, color)
    return result
