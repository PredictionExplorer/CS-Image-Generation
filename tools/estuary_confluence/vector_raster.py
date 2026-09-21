"""Shared native vector masks and explicit per-pigment world-area normalization.

Legacy shape callers retain the original arithmetic and outputs. Choreography
can add explicit component budgets and practical concentration limits; neither
changes the old path when omitted. All rasters are float32 and bottom-up.
"""

from __future__ import annotations

import math

import numpy as np

from tools.estuary_studio.surface import _number


def bounds(primitive):
    points = np.asarray(primitive["points"], dtype="f8")
    radii = np.asarray(primitive.get("radii", [0] * len(points)), dtype="f8")
    return np.stack(((points - radii[:, None]).min(0), (points + radii[:, None]).max(0)))


def validate_primitives(parts, aspect):
    """Validate three-pigment stroke/polygon geometry without layout metadata."""
    if type(parts) is not list or not 3 <= len(parts) <= 9:
        raise ValueError("Composition requires three to nine vector primitives")
    seen = set()
    for p in parts:
        if (
            type(p) is not dict
            or p.get("kind") not in ("stroke", "polygon")
            or type(p.get("pigment_index")) is not int
            or p["pigment_index"] not in (0, 1, 2)
        ):
            raise ValueError("Invalid composition primitive")
        points = np.asarray(p.get("points"), dtype="f8")
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 1 <= len(points) <= 128
            or not np.isfinite(points).all()
            or np.any(np.abs(points) > max(aspect, 1) + 1)
        ):
            raise ValueError("Primitive needs bounded finite vector points")
        if len(points) > 1 and np.any(np.sum(np.diff(points, axis=0) ** 2, axis=1) < 1e-20):
            raise ValueError("Primitive cannot contain repeated adjacent points")
        if p["kind"] == "stroke":
            radii = np.asarray(p.get("radii"), dtype="f8")
            if (
                radii.shape != (len(points),)
                or not np.isfinite(radii).all()
                or np.any(radii <= 0)
                or np.any(radii > 2 * max(aspect, 1))
            ):
                raise ValueError("Stroke radii must be finite and positive")
        elif len(points) < 3:
            raise ValueError("Polygon must be nondegenerate")
        if p["kind"] == "polygon":
            if float(np.sum((points[-1] - points[0]) ** 2)) < 1e-20:
                raise ValueError("Polygon closing edge cannot have zero length")
            a, b = points[1] - points[0], points[2] - points[0]
            if abs(float(a[0] * b[1] - a[1] * b[0])) < 1e-15:
                raise ValueError("Polygon must be nondegenerate")
        box = bounds(p)
        if np.any(box[0] < -np.array([aspect, 1]) - 1e-12) or np.any(
            box[1] > np.array([aspect, 1]) + 1e-12
        ):
            raise ValueError("Initial paint geometry leaves the visible canvas")
        seen.add(p["pigment_index"])
    if seen != {0, 1, 2}:
        raise ValueError("Every actual pigment needs at least one primitive")


def _profile(distance, edge):
    t = np.clip((distance + edge) / (2 * edge), 0, 1)
    return 1 - t * t * (3 - 2 * t)


def rasterize(
    primitives, resolution, domain_scale, *, aspect, target_mass, edge_width_world, peak_limits=None
):
    """Accumulate vector masks, then match each target's world-area pigment mass."""
    aspect = _number(aspect, "aspect", 0.2, 5)
    validate_primitives(primitives, aspect)
    if type(target_mass) not in (list, tuple) or len(target_mass) != 3:
        raise ValueError("Vector raster needs three target pigment amounts")
    targets = [_number(v, "target mass", 1e-9, 100) for v in target_mass]
    edge_width_world = _number(edge_width_world, "edge width", 1e-6, 0.05)
    component_budgets = ["target_mass" in p for p in primitives]
    if any(component_budgets):
        if not all(component_budgets):
            raise ValueError("Component budgets must be supplied for every vector primitive")
        declared = np.zeros(3, dtype="f8")
        for p in primitives:
            declared[p["pigment_index"]] += _number(p["target_mass"], "component mass", 1e-12, 100)
        if not np.allclose(declared, targets, rtol=5e-12, atol=1e-15):
            raise ValueError("Component budgets differ from total pigment targets")
    if peak_limits is not None:
        limits = np.asarray(peak_limits, dtype="f8")
        if limits.shape != (3,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("Peak concentration limits need three finite positive values")
    if (
        type(resolution) not in (list, tuple)
        or len(resolution) != 2
        or any(type(v) is not int or not 4 <= v <= 12288 for v in resolution)
        or resolution[0] * resolution[1] > 50_000_000
    ):
        raise ValueError("Invalid composition raster dimensions")
    width, height = resolution
    if not math.isclose(width / height, aspect, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Composition raster must match its source aspect")
    domain = _number(domain_scale, "domain scale", 1, 3)
    dx, dy = 2 * aspect * domain / width, 2 * domain / height
    edge = max(edge_width_world, 0.5 * math.hypot(dx, dy))
    x = ((np.arange(width, dtype="f8") + 0.5) / width * 2 - 1) * aspect * domain
    y = ((np.arange(height, dtype="f8") + 0.5) / height * 2 - 1) * domain
    masks = np.zeros((height, width, 3), dtype="f8")
    for p in primitives:
        box = bounds(p)
        x0, x1 = np.searchsorted(x, (box[0, 0] - edge, box[1, 0] + edge), side="left")
        y0, y1 = np.searchsorted(y, (box[0, 1] - edge, box[1, 1] + edge), side="left")
        if x0 == x1 or y0 == y1:
            raise ValueError("Initial geometry is unresolved at this material resolution")
        xx, yy = x[x0:x1][None, :], y[y0:y1][:, None]
        points = np.asarray(p["points"])
        if p["kind"] == "stroke":
            radii = np.asarray(p["radii"])
            if 2 * float(radii.min()) < 1.25 * max(dx, dy):
                raise ValueError("Initial stroke tips require a finer material grid")
            coverage = np.zeros((y1 - y0, x1 - x0), dtype="f8")
            if len(points) == 1:
                coverage = _profile(np.hypot(xx - points[0, 0], yy - points[0, 1]) - radii[0], edge)
            else:
                for i in range(len(points) - 1):
                    a, b = points[i], points[i + 1]
                    radius = max(radii[i], radii[i + 1]) + edge
                    sx0, sx1 = np.searchsorted(
                        x[x0:x1], (min(a[0], b[0]) - radius, max(a[0], b[0]) + radius)
                    )
                    sy0, sy1 = np.searchsorted(
                        y[y0:y1], (min(a[1], b[1]) - radius, max(a[1], b[1]) + radius)
                    )
                    if sx0 == sx1 or sy0 == sy1:
                        continue
                    vx, vy = b - a
                    qx, qy = xx[:, sx0:sx1], yy[sy0:sy1]
                    t = np.clip(((qx - a[0]) * vx + (qy - a[1]) * vy) / (vx * vx + vy * vy), 0, 1)
                    d = np.hypot(qx - a[0] - t * vx, qy - a[1] - t * vy) - (
                        radii[i] + t * (radii[i + 1] - radii[i])
                    )
                    region = coverage[sy0:sy1, sx0:sx1]
                    np.maximum(region, _profile(d, edge), out=region)
        else:
            edges = np.roll(points, -1, axis=0) - points
            if float(np.linalg.norm(edges, axis=1).min()) < 1.25 * max(dx, dy):
                raise ValueError("Initial polygon tips require a finer material grid")
            inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            distance = np.full_like(inside, np.inf, dtype="f8")
            for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
                vx, vy = b - a
                t = np.clip(((xx - a[0]) * vx + (yy - a[1]) * vy) / (vx * vx + vy * vy), 0, 1)
                distance = np.minimum(distance, np.hypot(xx - a[0] - t * vx, yy - a[1] - t * vy))
                if b[1] != a[1]:
                    inside ^= ((a[1] > yy) != (b[1] > yy)) & (
                        xx < (b[0] - a[0]) * (yy - a[1]) / (b[1] - a[1]) + a[0]
                    )
            coverage = _profile(np.where(inside, -distance, distance), edge)
        coverage *= (np.abs(xx) <= aspect) & (np.abs(yy) <= 1)
        if all(component_budgets):
            area = coverage.sum(dtype="f8") * dx * dy
            if not math.isfinite(area) or area <= 0:
                raise ValueError("A budgeted component has no resolved native paint area")
            coverage *= p["target_mass"] / area
        masks[y0:y1, x0:x1, p["pigment_index"]] += coverage
    result = np.empty_like(masks, dtype="f4")
    for i, target in enumerate(target_mass):
        mass = masks[..., i].sum(dtype="f8") * dx * dy
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("An initial pigment has no resolved native paint area")
        result[..., i] = masks[..., i] * (target / mass)
        actual = result[..., i].sum(dtype="f8") * dx * dy
        if abs(actual - target) > target * 5e-7:
            result[..., i] *= np.float32(target / actual)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1e6):
        raise ValueError("Initial composition exceeds supported pigment concentration")
    if peak_limits is not None and np.any(result.max(axis=(0, 1)) > limits * (1 + 1e-6)):
        raise ValueError("Initial geometry exceeds its declared peak concentration limits")
    return result
