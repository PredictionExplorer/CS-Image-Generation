"""Six reproducible vector initial conditions with matched native pigment mass.

Only three actual pigments are generated. Every choice has an independently
addressed SHA-256 stream over the complete seed. Random variants share placement
quantiles, mapped into their own shape-fit bounds; the resulting centers need not
coincide because a long ribbon needs different margins than a circle. Overlap is
allowed and repeated same-pigment primitives accumulate before normalization.

Curved strokes are bounded vector polylines with rounded, varying-width sections;
they contain no bitmap noise. Rasterization works on each segment's local bounds.
Native world-area integration sets the requested mass explicitly, independently
of the old initial_load and pigment-weight controls. All arrays are bottom-up.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from tools.estuary_studio.surface import _number

from .palette import normalize_seed

VERSION = "initial-composition-v1"
SETUPS = (
    "random-circles",
    "random-ribbons",
    "random-crescents",
    "facing-shores",
    "scattered-commas",
    "body-wedges",
)


def validate_config(value):
    if value is None:
        return None
    names = {"version", "setup", "target_mass", "reference_radii"}
    if type(value) is not dict or set(value) - names or not names - {"version"} <= set(value):
        raise ValueError("Initial composition requires setup, target_mass and reference_radii")
    result = {"version": VERSION, **copy.deepcopy(value)}
    if (
        type(result["version"]) is not str
        or result["version"] != VERSION
        or type(result["setup"]) is not str
        or result["setup"] not in SETUPS
    ):
        raise ValueError("Unknown initial-composition version or setup")
    for name, bounds in (("target_mass", (1e-9, 100)), ("reference_radii", (0.01, 0.65))):
        values = result[name]
        if type(values) not in (list, tuple) or len(values) != 3:
            raise ValueError(f"{name} needs exactly three values")
        result[name] = [_number(v, name, *bounds) for v in values]
    return result


def _unit(seed, label):
    material = VERSION.encode() + b"\0" + int(seed, 16).to_bytes(32, "big") + b"\0" + label.encode()
    return (int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 11) / 2**53


def _rotate(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(points, dtype="f8") @ np.array([[c, s], [-s, c]])


def _curve(controls, widths, samples=65):
    t = np.linspace(0, 1, samples)
    q = 1 - t
    points = (
        q[:, None] ** 3 * controls[0]
        + 3 * (q * q * t)[:, None] * controls[1]
        + 3 * (q * t * t)[:, None] * controls[2]
        + t[:, None] ** 3 * controls[3]
    )
    # A smooth quadratic width profile keeps both tips finite and resolvable.
    radius = q * q * widths[0] + 2 * q * t * widths[1] + t * t * widths[2]
    return {"kind": "stroke", "points": points.tolist(), "radii": radius.tolist()}


def _bounds(primitive):
    points = np.asarray(primitive["points"], dtype="f8")
    radii = np.asarray(primitive.get("radii", [0] * len(points)), dtype="f8")
    return np.stack(((points - radii[:, None]).min(0), (points + radii[:, None]).max(0)))


def _transform(primitive, scale, center):
    result = copy.deepcopy(primitive)
    result["points"] = (np.asarray(result["points"]) * scale + center).tolist()
    if "radii" in result:
        result["radii"] = (np.asarray(result["radii"]) * scale).tolist()
    result["center"] = list(map(float, center))
    result["fit_scale"] = float(scale)
    return result


def _place(primitive, aspect, quantiles):
    margin = 0.025 * min(1, aspect)
    available = np.array([aspect, 1]) - margin
    bounds = _bounds(primitive)
    scale = min(1.0, float(np.min(2 * available / (bounds[1] - bounds[0]))))
    bounds *= scale
    low, high = -available - bounds[0], available - bounds[1]
    center = low + (high - low) * quantiles
    return _transform(primitive, scale, center)


def _stroke(seed, setup, pigment, patch, radius, angle):
    bend = 2 * _unit(seed, f"shape/{setup}/{pigment}/{patch}/bend") - 1
    if setup == "random-ribbons":
        controls = (
            np.array(
                [[-2.45, -0.15], [-0.8, 1.35 + 0.6 * bend], [0.9, -1.1 + 0.4 * bend], [2.4, 0.2]]
            )
            * radius
        )
        widths = np.array([0.08, 0.34, 0.075]) * radius
    else:
        controls = (
            np.array([[-0.9, -0.45], [-1.25, 0.9 + 0.2 * bend], [0.65, 1.0], [0.7, -0.7]]) * radius
        )
        widths = np.array([0.3, 0.58, 0.11]) * radius
    return _curve(_rotate(controls, angle), widths)


def _polygon_centroid(points):
    points = np.asarray(points, dtype="f8")
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    return ((points + following) * cross[:, None]).sum(0) / (3 * cross.sum())


def plan_layout(seed, count, aspect, config, source=None):
    """Plan three actual colors; only body-wedges samples Source.frame(0)."""
    settings = validate_config(config)
    if settings is None or type(count) is not int or count != 3:
        raise ValueError("These composition studies require exactly three chromatic pigments")
    aspect = _number(aspect, "aspect", 0.2, 5)
    seed = normalize_seed(seed)
    setup = settings["setup"]
    references = settings["reference_radii"]
    quantiles = [
        [
            [_unit(seed, f"placement/{i}/{j}/{axis}") for axis in ("x", "y")]
            for j in range(3 if setup == "scattered-commas" else 1)
        ]
        for i in range(3)
    ]
    primitives = []
    if setup in ("random-circles", "random-ribbons", "random-crescents", "scattered-commas"):
        for i in range(3):
            for j, quantile in enumerate(quantiles[i]):
                radius = references[i] * (1, 0.68, 0.43)[j]
                angle = 2 * math.pi * _unit(seed, f"orientation/{i}/{j}")
                if setup == "random-circles":
                    p = {"kind": "stroke", "points": [[0.0, 0.0]], "radii": [radius]}
                elif setup == "random-crescents":
                    gap = math.radians(80 + 45 * _unit(seed, f"shape/crescent/{i}/gap"))
                    t = np.linspace(0, 1, 97)
                    theta = angle + gap / 2 + t * (2 * math.pi - gap)
                    points = radius * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
                    widths = radius * (0.11 + 0.19 * np.sin(math.pi * t) ** 0.75)
                    p = {"kind": "stroke", "points": points.tolist(), "radii": widths.tolist()}
                else:
                    p = _stroke(seed, setup, i, j, radius, angle)
                p = _place(p, aspect, np.asarray(quantile))
                primitives.append({**p, "pigment_index": i, "component": j})
    elif setup == "facing-shores":
        radius = float(np.mean(references))
        angle = math.radians(-22 + 44 * _unit(seed, "shore/angle"))
        bend = 0.7 + 0.5 * _unit(seed, "shore/bend")
        coast = np.array([[-0.15, -2.45], [bend, -0.8], [-bend, 0.7], [0.15, 2.1]]) * radius
        grouped = []
        for i, side in ((0, -1), (1, 1)):
            half_width = references[i] * (0.72 if i == 0 else 0.86)
            path = coast.copy()
            path[:, 0] += side * (0.28 * radius + half_width)
            p = _curve(_rotate(path, angle), [0.62 * half_width, half_width, 0.7 * half_width])
            grouped.append({**p, "pigment_index": i, "component": 0})
        for j, t in enumerate((0.22, 0.53, 0.81)):
            q = 1 - t
            center = (
                q**3 * coast[0]
                + 3 * q * q * t * coast[1]
                + 3 * q * t * t * coast[2]
                + t**3 * coast[3]
            )
            center[0] += (-0.1, 0.18, -0.06)[j] * radius
            center = _rotate(center, angle)
            grouped.append(
                {
                    "kind": "stroke",
                    "points": [center.tolist()],
                    "radii": [references[2] * (0.44, 0.3, 0.2)[j]],
                    "pigment_index": 2,
                    "component": j,
                }
            )
        group_bounds = np.stack([_bounds(p) for p in grouped])
        low, high = group_bounds[:, 0].min(0), group_bounds[:, 1].max(0)
        margin = 0.035 * min(aspect, 1)
        available = np.array([aspect, 1]) - margin
        scale = min(1.0, float(np.min(2 * available / (high - low))))
        group_quantile = np.array([_unit(seed, "shore/center/x"), _unit(seed, "shore/center/y")])
        center = -available - low * scale + (2 * available - (high - low) * scale) * group_quantile
        primitives = [_transform(p, scale, center) for p in grouped]
    else:
        if source is None:
            raise ValueError("Body wedges require the initial source frame")
        if hasattr(source, "aspect") and not math.isclose(
            source.aspect, aspect, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("Body wedge source aspect differs from its canvas")
        frame = source.frame(0.0)
        positions, velocities = (
            np.asarray(frame.positions, dtype="f8"),
            np.asarray(frame.velocities, dtype="f8"),
        )
        if (
            positions.shape != (3, 2)
            or velocities.shape != (3, 2)
            or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
        ):
            raise ValueError("Initial bodies need three finite planar positions and velocities")
        bounds = np.array([aspect, 1]) - 0.015 * min(aspect, 1)
        if np.any(np.abs(positions) >= bounds):
            raise ValueError("A body lies too close to the visible boundary for a centered wedge")
        for i in range(3):
            speed = math.hypot(*map(float, velocities[i]))
            angle = (
                math.atan2(velocities[i, 1], velocities[i, 0])
                if speed > 1e-12
                else 2 * math.pi * _unit(seed, f"body/{i}/stationary")
            )
            width = 0.65 + 0.25 * _unit(seed, f"body/{i}/width")
            points = (
                np.array(
                    [
                        [1.45, -0.12],
                        [1.45, 0.12],
                        [-0.65, width],
                        [-1.0, 0.42],
                        [-1.0, -0.42],
                        [-0.65, -width],
                    ]
                )
                * references[i]
            )
            points -= _polygon_centroid(points)
            points = _rotate(points, angle)
            scale = 1.0
            for axis in (0, 1):
                for side in (-1, 1):
                    extent = float(np.max(points[:, axis] * side))
                    if extent > 0:
                        scale = min(scale, (bounds[axis] - side * positions[i, axis]) / extent)
            p = _transform({"kind": "polygon", "points": points.tolist()}, scale, positions[i])
            primitives.append(
                {
                    **p,
                    "pigment_index": i,
                    "component": 0,
                    "body_index": i,
                    "initial_velocity": velocities[i].tolist(),
                    "heading_radians": angle,
                    "source_fraction": 0.0,
                }
            )
    return {
        "version": VERSION,
        "seed": seed,
        "count": 3,
        "aspect": aspect,
        "setup": setup,
        "target_mass": settings["target_mass"],
        "reference_radii": references,
        "edge_width_world": min(references) * 0.012,
        "placement_quantiles": quantiles,
        "primitives": primitives,
    }


def _validated_layout(layout):
    required = {
        "version",
        "seed",
        "count",
        "aspect",
        "setup",
        "target_mass",
        "reference_radii",
        "edge_width_world",
        "placement_quantiles",
        "primitives",
    }
    if (
        type(layout) is not dict
        or set(layout) != required
        or type(layout["count"]) is not int
        or layout["count"] != 3
    ):
        raise ValueError("Invalid three-pigment composition layout")
    validate_config({k: layout[k] for k in ("version", "setup", "target_mass", "reference_radii")})
    normalize_seed(layout["seed"])
    aspect = _number(layout["aspect"], "aspect", 0.2, 5)
    _number(layout["edge_width_world"], "edge width", 1e-6, 0.05)
    parts = layout["primitives"]
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
        bounds = _bounds(p)
        if np.any(bounds[0] < -np.array([aspect, 1]) - 1e-12) or np.any(
            bounds[1] > np.array([aspect, 1]) + 1e-12
        ):
            raise ValueError("Initial paint geometry leaves the visible canvas")
        seen.add(p["pigment_index"])
    if seen != {0, 1, 2}:
        raise ValueError("Every actual pigment needs at least one primitive")
    return aspect


def _profile(distance, edge):
    t = np.clip((distance + edge) / (2 * edge), 0, 1)
    return 1 - t * t * (3 - 2 * t)


def rasterize(layout, resolution, domain_scale):
    """Accumulate vector masks, then match each target's world-area pigment mass."""
    aspect = _validated_layout(layout)
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
    edge = max(layout["edge_width_world"], 0.5 * math.hypot(dx, dy))
    x = ((np.arange(width, dtype="f8") + 0.5) / width * 2 - 1) * aspect * domain
    y = ((np.arange(height, dtype="f8") + 0.5) / height * 2 - 1) * domain
    masks = np.zeros((height, width, 3), dtype="f8")
    for p in layout["primitives"]:
        bounds = _bounds(p)
        x0, x1 = np.searchsorted(x, (bounds[0, 0] - edge, bounds[1, 0] + edge), side="left")
        y0, y1 = np.searchsorted(y, (bounds[0, 1] - edge, bounds[1, 1] + edge), side="left")
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
        masks[y0:y1, x0:x1, p["pigment_index"]] += coverage
    result = np.empty_like(masks, dtype="f4")
    for i, target in enumerate(layout["target_mass"]):
        mass = masks[..., i].sum(dtype="f8") * dx * dy
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("An initial pigment has no resolved native paint area")
        result[..., i] = masks[..., i] * (target / mass)
        actual = result[..., i].sum(dtype="f8") * dx * dy
        if abs(actual - target) > target * 5e-7:
            result[..., i] *= np.float32(target / actual)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1e6):
        raise ValueError("Initial composition exceeds supported pigment concentration")
    return result
