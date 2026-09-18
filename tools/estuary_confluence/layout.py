"""Versioned, seed-driven placement of separate pure pigment pools.

Five positions are resolved first, regardless of requested pigment count. The
locations are drawn continuously from the visible canvas using bounded
blue-noise rejection. Radius and load come from independent named SHA-256
streams over the complete seed. No color,
trajectory sample, image dimensions, frame rate or global random state enters
the construction.

Coordinates use the fixed visible canvas: x in [-aspect, aspect], y in [-1, 1].
Circular radii scale with the short half-extent. The documented radius bound
leaves a real separation between pools and a margin inside the visible canvas;
the planner rejects larger requests rather than silently shrinking pools.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .palette import normalize_seed

VERSION = "scattered-pigment-layout-v1"
MAX_LOAD_RADIUS = 0.34
MAX_LAYOUT_ATTEMPTS = 8
CANDIDATES_PER_POOL = 256


def _number(value, name, low, high):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return result


def _unit(seed_bytes, label):
    digest = hashlib.sha256(VERSION.encode() + b"\0" + seed_bytes + b"\0" + label.encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def plan_layout(seed, count, aspect, *, load_radius=0.28, initial_load=0.18, edge_width=0.02):
    """Return exact JSON-ready pool placement and resolved material amounts.

    ``load_radius`` scales each seed-specific radius by a factor in [0.65, 0.8].
    ``initial_load`` scales independent peak concentrations in [0.82, 1.18].
    ``edge_width`` is the fraction of radius over which concentration smoothly
    falls from its full value to zero. It is not an output-pixel blur.
    """
    seed = normalize_seed(seed)
    if type(count) is not int or count not in (3, 5):
        raise ValueError("Scattered pigment count must be 3 or 5")
    aspect = _number(aspect, "aspect", 0.2, 5)
    radius_scale = _number(load_radius, "load_radius", 0.001, MAX_LOAD_RADIUS)
    load_scale = _number(initial_load, "initial_load", 0, 10)
    edge_width = _number(edge_width, "edge_width", 0.001, 0.25)
    seed_bytes = int(seed, 16).to_bytes(32, "big")

    def unit(label):
        return _unit(seed_bytes, label)

    short_extent = min(aspect, 1)
    pools = []
    for index in range(5):
        radius = radius_scale * short_extent * (0.65 + 0.15 * unit(f"pool/{index}/radius"))
        load = load_scale * (0.82 + 0.36 * unit(f"pool/{index}/load"))
        pools.append(
            {
                "pigment_index": index,
                "position": None,
                "radius": radius,
                "load": load,
                "edge_width": edge_width,
                "edge_width_world": radius * edge_width,
            }
        )

    def admissible(point, previous, radius):
        for other in previous:
            if math.dist(point, other["position"]) < radius + other["radius"] + 0.06 * short_extent:
                return False
        # The main three colors span an area rather than forming a tight group
        # or a nearly straight line. This does not force prescribed corners.
        normalized = [point[0] / aspect, point[1]]
        points = [[p["position"][0] / aspect, p["position"][1]] for p in previous]
        if len(previous) == 1 and math.dist(normalized, points[0]) < 1.0:
            return False
        if len(previous) == 2:
            a, b = points
            area = (
                abs((b[0] - a[0]) * (normalized[1] - a[1]) - (b[1] - a[1]) * (normalized[0] - a[0]))
                * 0.5
            )
            if area < 0.22:
                return False
        return True

    placement_method = "blue-noise-rejection"
    placement_attempt = 0
    for attempt in range(MAX_LAYOUT_ATTEMPTS):
        placed = []
        for index, pool in enumerate(pools):
            margin = pool["radius"] + 0.04 * short_extent
            for candidate in range(CANDIDATES_PER_POOL):
                prefix = f"placement/{attempt}/pool/{index}/candidate/{candidate}"
                point = [
                    (unit(prefix + "/x") * 2 - 1) * (aspect - margin),
                    (unit(prefix + "/y") * 2 - 1) * (1 - margin),
                ]
                if admissible(point, placed, pool["radius"]):
                    placed.append({**pool, "position": point})
                    break
            else:
                break
        if len(placed) == 5:
            pools = placed
            placement_attempt = attempt
            break
    else:
        # A finite, tested packing remains available for exceptionally unlucky
        # rejection streams. It is not the normal layout and is archived as
        # a fallback rather than silently advertised as unrestricted sampling.
        placement_method = "bounded-fallback"
        placement_attempt = MAX_LAYOUT_ATTEMPTS
        corners = [(-0.58, -0.55), (0.58, -0.55), (-0.58, 0.55), (0.58, 0.55)]
        order = sorted(range(4), key=lambda i: (unit(f"fallback/corner/{i}"), i))
        anchors = [*corners, (0.0, 0.0)]
        angle = (unit("fallback/rotation") * 2 - 1) * 0.14
        cosine, sine = math.cos(angle), math.sin(angle)
        for index, anchor in enumerate([*order[:3], 4, order[3]]):
            x, y = anchors[anchor]
            px = cosine * x - sine * y + (unit(f"fallback/{index}/x") * 2 - 1) * 0.055
            py = sine * x + cosine * y + (unit(f"fallback/{index}/y") * 2 - 1) * 0.055
            pools[index]["position"] = [px * aspect, py]
    # Verify physical support, including the full outer edge. These checks are
    # independent of raster resolution and requested count.
    for index, pool in enumerate(pools):
        x, y = pool["position"]
        if abs(x) + pool["radius"] >= aspect or abs(y) + pool["radius"] >= 1:
            raise RuntimeError("Internal scattered layout violated its canvas margin")
        for previous in pools[:index]:
            dx = x - previous["position"][0]
            dy = y - previous["position"][1]
            if math.hypot(dx, dy) <= pool["radius"] + previous["radius"]:
                raise RuntimeError("Internal scattered layout produced overlapping pools")
    return {
        "version": VERSION,
        "seed": seed,
        "count": count,
        "aspect": aspect,
        "coordinate_system": "fixed visible canvas: x [-aspect, aspect], y [-1, 1]",
        "load_radius": radius_scale,
        "initial_load": load_scale,
        "edge_width": edge_width,
        "construction": "five separated positions resolved before selecting the count prefix",
        "placement_method": placement_method,
        "placement_attempt": placement_attempt,
        "pools": pools[:count],
    }


def pool_profile(distance, radius, edge_width):
    """Finite-support pool with a narrow, physically sized C1 transition."""
    values = np.asarray(distance)
    transition = np.clip((values / radius - (1 - edge_width)) / edge_width, 0, 1)
    return 1 - transition * transition * (3 - 2 * transition)
