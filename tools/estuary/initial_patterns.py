"""Versioned spatial paint motifs, with no simulation or ambient random state.

The initializer writes a unit-load partition of the existing three pigments.
Every pixel keeps the same total amount, so overlaps never multiply paint loading.
A caller applies initial_load once. Pigment proportions vary intentionally between
motifs; this module does not claim equal per-pigment mass or a dynamic flow pilot.

Coordinates are world-space pixel centers, including the solver's guard domain.
The optional full-interval source samples guide the center, orientation and scale.
They are a spatial participation heuristic, not proof of future pigment motion.
Only row-sized working arrays are allocated; the caller owns the output state.
"""

from __future__ import annotations

import hashlib
import math
import re
from types import MappingProxyType

import numpy as np

VERSION = "starting-patterns-v1"
PATTERNS = MappingProxyType(
    {
        "lacuna-banks": "Joined light and medium banks surround two unequal dark openings.",
        "interlocking-crescents": "Opposed light and medium crescents retain open dark interiors.",
        "braided-ribbons": "Broad light and medium ribbons cross beside one fine return.",
        "split-fan": "A medium stem opens between two tapered light tongues.",
        "river-confluence": "Light and medium tributaries join a divided, bending river.",
        "broken-terraces": (
            "Unequal light and medium contour shelves carry deliberate interruptions."
        ),
        "meandering-fault": "A broad medium seam winds between a light bank and a dark field.",
        "folded-sash": "A two-color S-shaped cloth band narrows through its middle.",
        "asymmetric-rosette": "Unequal light and medium petals surround an open dark aperture.",
        "branching-channels": "Forked dark channels divide light and medium peninsula regions.",
    }
)

_SEED = re.compile(r"0x[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_controls(value):
    """Return a fresh strict control record; None preserves the omitted legacy path."""
    if value is None:
        return None
    _require(type(value) is dict, "starting pattern controls must be an object")
    _require(
        set(value) == {"version", "pattern", "seed"}, "Unknown or missing starting pattern keys"
    )
    _require(value["version"] == VERSION, "Unsupported starting pattern version")
    _require(
        type(value["pattern"]) is str and value["pattern"] in PATTERNS,
        "Unknown starting pattern",
    )
    _require(
        type(value["seed"]) is str and _SEED.fullmatch(value["seed"]) is not None,
        "Starting pattern seed must be 0x followed by 64 lowercase hex digits",
    )
    return dict(value)


def _unit(seed, pattern, name):
    payload = f"{VERSION}\0{pattern}\0{name}\0{seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _positions(value, name, shape):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite coordinates") from exc
    _require(points.shape == shape, f"{name} must have shape {shape}")
    _require(np.isfinite(points).all() and np.max(np.abs(points)) < 1e6, f"Invalid {name}")
    return points


def _frame(initial_positions, support_positions, controls):
    initial = _positions(initial_positions, "initial_positions", (3, 2))
    if support_positions is None:
        points = initial
    else:
        try:
            support = np.asarray(support_positions, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("support_positions must contain finite coordinates") from exc
        _require(
            support.ndim == 3 and 1 <= len(support) <= 4096 and support.shape[1:] == (3, 2),
            "support_positions must have shape (1..4096,3,2)",
        )
        points = _positions(support, "support_positions", support.shape).reshape(-1, 2)
    center = np.median(points, axis=0)
    centered = points - center
    # Closed-form 2D principal orientation avoids eigenvector sign conventions.
    xx, yy = np.mean(centered * centered, axis=0)
    xy = np.mean(centered[:, 0] * centered[:, 1])
    if math.hypot(float(xx - yy), float(2 * xy)) < 1e-12:
        edge = initial[1] - initial[0]
        angle = math.atan2(float(edge[1]), float(edge[0]))
    else:
        angle = 0.5 * math.atan2(float(2 * xy), float(xx - yy))
    direction = np.array([math.cos(angle), math.sin(angle)])
    anchor = initial[1] - initial[0]
    if float(np.dot(anchor, direction)) < 0:
        angle += math.pi
    seed, pattern = controls["seed"], controls["pattern"]
    angle += (_unit(seed, pattern, "rotation") - 0.5) * 0.44
    radius = float(np.quantile(np.sqrt(np.sum(centered * centered, axis=1)), 0.75))
    radius = float(np.clip(radius, 0.4, 0.85))
    phase = _unit(seed, pattern, "phase") * 2 * math.pi
    skew = (_unit(seed, pattern, "skew") - 0.5) * 0.28
    stretch = 0.9 + 0.2 * _unit(seed, pattern, "stretch")
    return center, angle, radius, phase, skew, stretch


def _inside(distance, edge):
    t = np.clip(0.5 - distance / edge, 0, 1)
    return t * t * (3 - 2 * t)


def _band(distance, width, edge):
    return _inside(np.abs(distance) - width, edge)


def _oval(u, v, cx, cy, rx, ry):
    return (np.sqrt(((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2) - 1) * min(rx, ry)


def _segment(u, v, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = np.clip(((u - a[0]) * dx + (v - a[1]) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(u - a[0] - t * dx, v - a[1] - t * dy), t


def _masks(pattern, u, v, phase, edge):
    """Return light/medium coverage; caller enforces a single pigment partition."""
    wobble = 0.055 * np.sin(2.2 * u + phase)
    if pattern == "lacuna-banks":
        bank = _oval(u, v, 0.05, 0, 1.65, 0.86)
        holes = np.minimum(
            _oval(u, v, -0.50, 0.13, 0.48, 0.35),
            _oval(u, v, 0.61, -0.19, 0.62, 0.27),
        )
        body = _inside(bank, edge) * (1 - _inside(holes, edge))
        division = _inside(u + 0.15 * np.sin(2.2 * v + phase), edge)
        white = body * (1 - division)
        red = np.maximum(body * division, _band(holes + 0.027, 0.018, edge))
    elif pattern == "interlocking-crescents":
        left = _oval(u, v, -0.47, 0.12, 0.89, 0.91)
        right = _oval(u, v, 0.49, -0.10, 0.82, 0.69)
        left_cut = _oval(u, v, -0.10, 0.20, 0.79, 0.75)
        right_cut = _oval(u, v, 0.19, -0.20, 0.72, 0.57)
        first = _inside(left, edge) * (1 - _inside(left_cut, edge))
        second = _inside(right, edge) * (1 - _inside(right_cut, edge))
        white = first
        red = np.maximum(
            second,
            np.maximum(
                _band(left_cut + 0.035, 0.020, edge) * _inside(left + 0.035, edge),
                _band(right_cut + 0.035, 0.016, edge) * _inside(right + 0.035, edge),
            ),
        )
    elif pattern == "braided-ribbons":
        a = v - 0.38 * np.sin(1.65 * u + 0.30 * math.sin(phase))
        b = v + 0.38 * np.sin(1.65 * u + 0.30 * math.sin(phase))
        c = v - 0.68 * np.sin(1.10 * u + 0.35)
        white = np.maximum(_band(a, 0.18, edge), _band(c, 0.035, edge))
        red = np.maximum(_band(b, 0.18, edge), _band(a - 0.25, 0.023, edge))
    elif pattern == "split-fan":
        trunk = v - 0.08 * np.sin(2 * u + phase)
        spread = np.maximum(u + 0.5, 0)
        width = 0.115 + 0.08 * np.clip(spread, 0, 1.8)
        red = _band(trunk - 0.03 * spread, width * 1.15, edge)
        white = np.maximum(
            _band(trunk + 0.65 * spread, width * 0.85, edge),
            _band(trunk - 0.72 * spread, width * 0.72, edge),
        ) * (1 - red)
        white = np.maximum(white, _band(trunk - 0.03 * spread - width - 0.10, 0.022, edge))
    elif pattern == "river-confluence":
        bend = 0.21 * np.sin(1.35 * u + 0.20 * math.sin(phase))
        fork = 0.64 * np.maximum(-u - 0.1, 0)
        first, second = v - bend - fork - 0.13, v - bend + fork + 0.13
        white = _band(first, 0.20, edge)
        red = np.maximum(_band(second, 0.20, edge), _band(first - 0.27, 0.023, edge))
    elif pattern == "broken-terraces":
        # Unequal curved shelves with two localized interruptions, not a uniform grid.
        contour = np.sqrt((u + 0.60) ** 2 + (1.30 * v) ** 2)
        shelves = np.maximum(_band(contour - 0.49, 0.15, edge), _band(contour - 1.63, 0.075, edge))
        cuts = np.maximum(
            _inside(_oval(u, v, 0.30, 0.37, 0.22, 0.35), edge),
            _inside(_oval(u, v, -0.75, -0.49, 0.33, 0.14), edge),
        )
        bank = _band(v + 0.17 + wobble, 0.11, edge)
        white = np.maximum(shelves * (1 - cuts), bank)
        red = np.maximum(_band(contour - 0.98, 0.19, edge), _band(contour - 0.25, 0.022, edge)) * (
            1 - cuts
        )
    elif pattern == "meandering-fault":
        boundary = 0.46 * np.sin(1.65 * u + 0.17 * math.sin(phase)) + 0.12 * np.sin(3.5 * u)
        distance = (v - boundary) / np.sqrt(1 + (0.76 * np.cos(1.65 * u)) ** 2)
        white = _inside(distance + 0.16, edge)
        red = _band(distance - 0.02 - 0.025 * np.sin(3 * u + phase), 0.17, edge)
    elif pattern == "folded-sash":
        curve = 0.62 * np.tanh(1.55 * u) + 0.10 * np.sin(2.4 * u + phase)
        width = 0.29 + 0.13 * (1 - np.exp(-1.5 * u * u))
        body = _band(v - curve, width, edge)
        division = _inside(v - curve + 0.025 * np.sin(2 * u + phase), edge)
        white = np.maximum(body * (1 - division), _band(v - curve + width + 0.24, 0.035, edge))
        red = np.maximum(body * division, _band(v - curve - width - 0.075, 0.025, edge))
    elif pattern == "asymmetric-rosette":
        theta = np.arctan2(v, u)
        radial = np.hypot(u, v)
        radius = 0.90 + 0.20 * np.cos(3 * theta + 0.18 * math.sin(phase)) + 0.10 * np.sin(theta)
        outside = _inside(radial - radius, edge)
        aperture = _oval(u, v, 0.04, -0.06, 0.35, 0.25)
        center = _inside(aperture, edge)
        slit = _band(v + 0.13 * u, 0.070, edge) * _inside(-u, edge)
        body = outside * (1 - np.maximum(center, slit))
        division = _inside(0.42 - np.cos(theta - 2.1), edge)
        white = body * (1 - division)
        red = np.maximum(
            body * division, _band(radial - radius - 0.06, 0.027, edge) * _inside(-v - 0.18, edge)
        )
        red = np.maximum(red, _band(aperture + 0.04, 0.021, edge) * _inside(u - 0.18, edge))
    else:  # branching-channels
        bank = _inside(_oval(u, v, -0.20, 0, 1.42, 0.93), edge)
        trunk, _ = _segment(u, v, (-1.7, -0.16), (0.22, 0.10))
        upper, t = _segment(u, v, (0.02, 0.07), (1.45, 0.85))
        lower, _ = _segment(u, v, (0.13, 0.08), (1.30, -0.65))
        distance = np.minimum.reduce([trunk - 0.105, upper - (0.09 - 0.035 * t), lower - 0.07])
        body = bank * (1 - _inside(distance, edge))
        division = _inside(v - 0.10 + 0.12 * u, edge)
        white = body * (1 - division)
        red = np.maximum(body * division, _band(distance + 0.033, 0.018, edge) * bank)
    return white, red


def fill_pattern(
    state, x, y, initial_positions, controls, *, support_positions=None, tile_rows=128
):
    """Fill an existing float32 HxWx4 array with a deterministic unit-load motif.

    x/y are strictly increasing 1D world-space pixel centers. Geometry does not
    depend on output shape, tile boundaries, frame number, camera or global RNG.
    support_positions, when supplied, is Nx3x2 projected positions sampled across
    the complete source interval. None controls is a no-op for legacy callers.
    """
    controls = validate_controls(controls)
    if controls is None:
        return
    _require(
        isinstance(state, np.ndarray)
        and state.dtype == np.float32
        and state.ndim == 3
        and state.shape[2] == 4
        and state.flags.writeable,
        "state must be a writable float32 HxWx4 array",
    )
    _require(type(tile_rows) is int and 1 <= tile_rows <= 512, "tile_rows must be in [1,512]")
    try:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x/y must be finite coordinate vectors") from exc
    _require(
        x.shape == (state.shape[1],)
        and y.shape == (state.shape[0],)
        and len(x) >= 2
        and len(y) >= 2
        and np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.max(np.abs(x)) < 1e6
        and np.max(np.abs(y)) < 1e6
        and np.all(np.diff(x) > 0)
        and np.all(np.diff(y) > 0),
        "x/y must be increasing finite pixel-center vectors matching state",
    )
    center, angle, radius, phase, skew, stretch = _frame(
        initial_positions, support_positions, controls
    )
    cosine, sine = math.cos(angle), math.sin(angle)
    edge = max(0.006, min(float(np.min(np.diff(x))), float(np.min(np.diff(y)))) / radius * 1.2)
    for start in range(0, len(y), tile_rows):
        stop = min(start + tile_rows, len(y))
        dx, dy = x[None, :] - center[0], y[start:stop, None] - center[1]
        u = (cosine * dx + sine * dy) / (radius * stretch)
        v = (-sine * dx + cosine * dy) * stretch / radius
        v = v + skew * np.tanh(u)
        white, red = _masks(controls["pattern"], u, v, phase, edge)
        white = np.asarray(np.clip(white, 0, 1), dtype=np.float32)
        red = np.minimum(np.asarray(np.clip(red, 0, 1), dtype=np.float32), 1 - white)
        tile = state[start:stop]
        tile[..., 1] = white
        tile[..., 2] = red
        tile[..., 0] = np.maximum(1 - white - red, 0)
        tile[..., 3] = 0


def coverage_stats(
    state, x, y, initial_positions, controls, *, support_positions=None, aspect=4 / 3, tile_rows=128
):
    """Measure spatial pigment shares in the visible view and central source region.

    The active corridor is the source-frame ellipse with semiaxes radius and
    0.7*radius. It is a reproducible spatial diagnostic, not a flow simulation or
    assurance that a given source visits every marked pixel. Shares use paint
    amount, including antialiased edges; support counts also report pixels with
    at least 10% of the local total. Statistics require a unit-load partition.
    """
    controls = validate_controls(controls)
    _require(controls is not None, "Coverage statistics require an enabled starting pattern")
    _require(
        isinstance(state, np.ndarray)
        and state.dtype == np.float32
        and state.ndim == 3
        and state.shape[2] == 4,
        "state must be float32 HxWx4",
    )
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    _require(
        x.shape == (state.shape[1],)
        and y.shape == (state.shape[0],)
        and np.isfinite(x).all()
        and np.isfinite(y).all(),
        "Coverage coordinates must match state",
    )
    _require(
        type(aspect) in (int, float) and math.isfinite(aspect) and aspect > 0, "Invalid aspect"
    )
    _require(type(tile_rows) is int and 1 <= tile_rows <= 512, "tile_rows must be in [1,512]")
    center, angle, radius, *_ = _frame(initial_positions, support_positions, controls)
    cosine, sine = math.cos(angle), math.sin(angle)
    totals = {key: np.zeros(3, dtype=np.float64) for key in ("visible", "active_corridor")}
    supports = {key: np.zeros(3, dtype=np.int64) for key in totals}
    counts = dict.fromkeys(totals, 0)
    for start in range(0, len(y), tile_rows):
        stop = min(start + tile_rows, len(y))
        values = state[start:stop, :, :3]
        _require(
            np.isfinite(values).all()
            and np.min(values) >= 0
            and np.max(values) <= 1
            and np.max(np.abs(values.sum(axis=-1) - 1)) <= 2e-7
            and not state[start:stop, :, 3].any(),
            "Coverage statistics require a finite unit-load pigment partition",
        )
        dx, dy = x[None, :] - center[0], y[start:stop, None] - center[1]
        u, v = (cosine * dx + sine * dy) / radius, (-sine * dx + cosine * dy) / radius
        visible = (np.abs(x[None, :]) <= aspect) & (np.abs(y[start:stop, None]) <= 1)
        regions = {"visible": visible, "active_corridor": visible & (u * u + (v / 0.7) ** 2 <= 1)}
        for name, mask in regions.items():
            counts[name] += int(mask.sum())
            totals[name] += np.sum(values, axis=(0, 1), where=mask[..., None], dtype=np.float64)
            supports[name] += np.sum((values >= 0.1) & mask[..., None], axis=(0, 1))
    _require(all(counts.values()), "Visible view and active corridor must contain pixels")
    return {
        name: {
            "pixels": counts[name],
            "pigment_fractions": (totals[name] / counts[name]).tolist(),
            "support_fractions": (supports[name] / counts[name]).tolist(),
        }
        for name in totals
    }
