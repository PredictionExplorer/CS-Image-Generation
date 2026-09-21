"""Bounded source-aware starting paint, qualified by a three-color tracer pilot.

The prescribed RC1 flow chooses geometry, never colors or new forces. Tracers
approximate both original layer speeds but omit diffusion and layer exchange;
their participation gates are a candidate filter, not a claim of artistic merit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import OrderedDict

import numpy as np

from tools.estuary.flow_reference import velocity

from . import participation_layout as engaged
from . import vector_raster
from .laminate import layer_fractions
from .pair_strain import pair_strain_uniforms
from .palette import normalize_seed

VERSION = "choreographed-paint-v1"
SETUPS = (
    "active-pools",
    "compact-pools",
    "broad-pools",
    "unequal-pools",
    "stretch-ovals",
    "cross-strokes",
    "facing-banks",
    "split-lobes",
    "long-ribbons",
    "swept-crescents",
)
CANDIDATE_LIMIT = 36
PILOT_STEPS = 1024
PARTICLES_PER_COMPONENT = 32
OBSERVATION_STRIDE = 16
PEAK_REFERENCE_MULTIPLIER = 4.0
GATES = {
    "minimum_displacement_in_radii": 0.45,
    "minimum_peak_stretch": 1.08,
    "minimum_moved_mass_fraction": 0.60,
    "minimum_neighbor_exposure": 0.008,
    "minimum_contacted_mass_fraction": 0.10,
}
_PREPARED = {}
_LAYOUTS = OrderedDict()


def _number(value, label, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid choreography {label}")
    return float(value)


def validate_config(value):
    if value is None:
        return None
    required = {"setup", "target_mass", "reference_radii"}
    if (
        type(value) is not dict
        or not required <= value.keys()
        or value.keys() - required - {"version", "mobility_bias"}
    ):
        raise ValueError("Choreography requires setup, target_mass and reference_radii")
    if (
        type(value.get("version", VERSION)) is not str
        or value.get("version", VERSION) != VERSION
        or type(value["setup"]) is not str
        or value["setup"] not in SETUPS
    ):
        raise ValueError("Unknown choreography version or setup")
    result = {"version": VERSION, "setup": value["setup"]}
    for name, low, high in (("target_mass", 1e-9, 100), ("reference_radii", 0.01, 0.65)):
        row = value[name]
        if type(row) not in (list, tuple) or len(row) != 3:
            raise ValueError(f"Choreography {name} needs exactly three values")
        result[name] = [_number(v, name, low, high) for v in row]
    if "mobility_bias" in value:
        row = value["mobility_bias"]
        if type(row) not in (list, tuple) or len(row) != 3:
            raise ValueError("mobility_bias needs three values")
        bias = [_number(v, "mobility_bias", -0.15, 0.15) for v in row]
        if any(bias):
            result["mobility_bias"] = bias
    return result


def effective_layer_fractions(palette, config):
    """Apply an optional bounded upper/lower redistribution without palette writes."""
    config = validate_config(config)
    result = layer_fractions(palette, 4).copy()
    if config is not None and "mobility_bias" in config:
        shifted = result[:3].astype("f8") + config["mobility_bias"]
        if np.any((shifted < 0.15) | (shifted > 0.85)):
            raise ValueError("Biased upper layer fractions must stay within [0.15, 0.85]")
        result[:3] = shifted
    return result


def _json_finite(value):
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Choreography metadata must be finite JSON") from exc


def _hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype="f8"))))

    def half(rows):
        hull = []
        for point in rows:
            while len(hull) > 1:
                a, b = np.asarray(hull[-2]), np.asarray(hull[-1])
                v, w = b - a, np.asarray(point) - b
                if v[0] * w[1] - v[1] * w[0] > 0:
                    break
                hull.pop()
            hull.append(point)
        return hull

    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def _outline(primitive):
    if primitive["kind"] == "polygon":
        return np.asarray(primitive["points"])
    theta = np.arange(32) * (2 * math.pi / 32)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1) / math.cos(math.pi / 32)
    return _hull(
        np.concatenate(
            [
                np.asarray(p) + radius * ring
                for p, radius in zip(primitive["points"], primitive["radii"], strict=True)
            ]
        )
    )


def _separation(a, b, gap):
    """Return a separating translation, or None for disjoint convex supports."""
    axes = []
    for points in (a, b):
        edges = np.roll(points, -1, axis=0) - points
        axes.extend(np.stack([-edges[:, 1], edges[:, 0]], axis=1))
    axes = np.asarray(axes)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    aa, bb = np.einsum("ni,mi->nm", a, axes), np.einsum("ni,mi->nm", b, axes)
    overlap = np.minimum(aa.max(0) - bb.min(0), bb.max(0) - aa.min(0)) + gap
    if np.any(overlap <= 0):
        return None
    index = int(np.argmin(overlap))
    direction = axes[index]
    if np.sum((b.mean(0) - a.mean(0)) * direction) < 0:
        direction = -direction
    return direction * overlap[index]


def _separate(primitives, aspect, gap):
    result = copy.deepcopy(primitives)
    for _ in range(24):
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                shift = _separation(_outline(result[i]), _outline(result[j]), gap)
                if shift is not None:
                    changed = True
                    result[i]["points"] = (np.asarray(result[i]["points"]) - shift * 0.501).tolist()
                    result[j]["points"] = (np.asarray(result[j]["points"]) + shift * 0.501).tolist()
        if not changed:
            break
    else:
        return None
    margin = 0.025 * min(aspect, 1)
    for p in result:
        bounds = vector_raster.bounds(p)
        if np.any(bounds[0] < -np.array([aspect, 1]) + margin) or np.any(
            bounds[1] > np.array([aspect, 1]) - margin
        ):
            return None
    return result


def _ellipse(center, radius, angle, aspect_ratio=1):
    t = np.arange(64) * (2 * math.pi / 64)
    points = radius * np.stack(
        [np.cos(t) * math.sqrt(aspect_ratio), np.sin(t) / math.sqrt(aspect_ratio)], axis=1
    )
    c, s = math.cos(angle), math.sin(angle)
    return {
        "kind": "polygon",
        "points": (np.einsum("ni,ij->nj", points, [[c, s], [-s, c]]) + center).tolist(),
    }


def _primitive(center, radius, angle, setup, index):
    if setup in {"stretch-ovals", "facing-banks"}:
        return _ellipse(center, radius, angle, 2.5 if setup == "stretch-ovals" else 2.0)
    if setup == "cross-strokes":
        axis = np.array([math.cos(angle), math.sin(angle)])
        return {
            "kind": "stroke",
            "points": (center + np.array([-1.55, 0, 1.55])[:, None] * radius * axis).tolist(),
            "radii": (radius * np.array([0.23, 0.55, 0.23])).tolist(),
        }
    if setup == "long-ribbons":
        axis = np.array([math.cos(angle), math.sin(angle)])
        return {
            "kind": "stroke",
            "points": (center + np.array([-2.8, 0, 2.8])[:, None] * radius * axis).tolist(),
            "radii": (radius * np.array([0.08, 0.32, 0.08])).tolist(),
        }
    if setup == "swept-crescents":
        t = np.linspace(0, 1, 33)
        theta = angle + (t - 0.5) * math.radians(200)
        widths = radius * (0.10 + 0.22 * np.sin(math.pi * t) ** 0.85)
        points = 1.2 * radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        # Keep the approximate painted centroid at the activity anchor. The
        # open side faces upstream; the conservative convex hull is used only
        # for separation, never as the paint mask or tracer support.
        points -= np.average(points, axis=0, weights=widths)
        return {
            "kind": "stroke",
            "points": (points + center).tolist(),
            "radii": widths.tolist(),
        }
    factor = {"compact-pools": 0.75, "broad-pools": 1.2}.get(setup, 1.0)
    if setup == "unequal-pools":
        factor = (0.72, 1.0, 1.28)[index]
    return {"kind": "stroke", "points": [np.asarray(center).tolist()], "radii": [radius * factor]}


def _source_data(source, simulation):
    flow, pools = engaged._settings(source, simulation)
    key_data = [
        VERSION,
        normalize_seed(source.seed),
        source.sha256,
        source.projection,
        flow,
        pools,
        PILOT_STEPS,
        engaged.PILOT_STEPS,
        engaged.CANDIDATE_LAYOUTS,
    ]
    _json_finite(key_data)
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    if key in _PREPARED:
        return _PREPARED[key]
    fractions = (np.arange(PILOT_STEPS) + 0.5) / PILOT_STEPS
    frames = source.sample(fractions)
    tools, pairs = engaged.conditioned_uniforms(frames, flow["stir_radius"])
    strains = (
        pair_strain_uniforms(frames, flow["stir_radius"]) if flow.get("pair_strain", 0) else None
    )
    if tools.shape != (PILOT_STEPS, 3, 4):
        raise ValueError("Invalid choreography source sampling dimensions")
    # These are permitted safe anchors only. New ranking uses three actual
    # colors; no unused fourth/fifth pool enters the choreography pilot.
    safe = engaged.plan_engaged_layout(source, 3, simulation)
    centers = np.asarray([p["position"] for p in safe["pools"]])
    bank = []
    for step in np.linspace(0, PILOT_STEPS - 1, 32, dtype=int):
        current = flow if strains is None else {**flow, "strains": strains[step]}
        anchors = np.concatenate([tools[step, :, :2], pairs[step, :, :2]])
        v = velocity(anchors, tools[step], pairs[step], **current)
        h = flow["stir_radius"] * 0.08
        dx = (
            velocity(anchors + np.array([h, 0]), tools[step], pairs[step], **current)
            - velocity(anchors - np.array([h, 0]), tools[step], pairs[step], **current)
        ) / (2 * h)
        dy = (
            velocity(anchors + np.array([0, h]), tools[step], pairs[step], **current)
            - velocity(anchors - np.array([0, h]), tools[step], pairs[step], **current)
        ) / (2 * h)
        for point, speed, gx, gy in zip(anchors, v, dx, dy, strict=True):
            gradient = np.stack([gx, gy], axis=1)
            strain = (gradient + gradient.T) * 0.5
            values, vectors = np.linalg.eigh(strain)
            direction = vectors[:, -1]
            activity = np.linalg.norm(speed) + flow["stir_radius"] * max(float(values[-1]), 0)
            angle = (
                math.atan2(speed[1], speed[0])
                if np.linalg.norm(speed) > 1e-9
                else math.atan2(direction[1], direction[0])
            )
            bank.append(
                (float(activity), point.copy(), angle, math.atan2(direction[1], direction[0]))
            )
    bank.sort(key=lambda row: -row[0])
    chosen = []
    for row in bank:
        if row[0] <= 1e-10:
            continue
        if np.any(np.abs(row[1]) > np.array([flow["aspect"], 1]) - 0.18):
            continue
        if all(np.linalg.norm(row[1] - old[1]) >= flow["stir_radius"] * 0.65 for old in chosen):
            chosen.append(row)
        if len(chosen) >= 24:
            break
    if not chosen:
        raise ValueError("No active source anchors for choreography")
    result = flow, tools, pairs, strains, centers, chosen, key
    if len(_PREPARED) >= 12:
        _PREPARED.pop(next(iter(_PREPARED)))
    _PREPARED[key] = result
    return result


def _starting_directions(centers, prepared):
    """Measure initial directions stretched over the first half of the recording."""
    flow, tools, pairs, strains, _, _, _ = prepared
    h = flow["stir_radius"] * 0.04
    offsets = np.array([[h, 0], [-h, 0], [0, h], [0, -h]])
    points = np.asarray(centers)[..., None, :] + offsets
    for step in range(max(1, len(tools) // 2)):
        current = flow if strains is None else {**flow, "strains": strains[step]}
        first = velocity(points, tools[step], pairs[step], **current)
        points += velocity(
            points + first / (2 * len(tools)), tools[step], pairs[step], **current
        ) / len(tools)
    dx, dy = (
        (points[..., 0, :] - points[..., 1, :]) / (2 * h),
        (points[..., 2, :] - points[..., 3, :]) / (2 * h),
    )
    jacobian = np.stack([dx, dy], axis=-1)
    values, vectors = np.linalg.eigh(np.einsum("...ji,...jk->...ik", jacobian, jacobian))
    direction = vectors[..., -1]
    angles = np.arctan2(direction[..., 1], direction[..., 0])
    # An isotropic map has no physical preferred axis; use measured paint flow.
    initial_velocity = velocity(
        np.asarray(centers),
        tools[0],
        pairs[0],
        **(flow if strains is None else {**flow, "strains": strains[0]}),
    )
    fallback = np.arctan2(initial_velocity[..., 1], initial_velocity[..., 0])
    return np.where(values[..., 1] - values[..., 0] > 1e-5, angles, fallback)


def _candidates(config, prepared):
    flow, _, _, _, safe, bank, _ = prepared
    setup, radii = config["setup"], np.asarray(config["reference_radii"])
    dominant = int(np.argmax(config["target_mass"]))
    spacing = float(radii.max()) * 2.05 + 0.012
    proposals = []
    tangent = np.array([math.cos(bank[0][2]), math.sin(bank[0][2])])
    normal = np.array([-tangent[1], tangent[0]])
    for shift in (
        np.zeros(2),
        0.45 * radii.mean() * tangent,
        -0.45 * radii.mean() * tangent,
        0.45 * radii.mean() * normal,
        -0.45 * radii.mean() * normal,
        0.85 * radii.mean() * normal,
        -0.85 * radii.mean() * normal,
    ):
        proposals.append((safe + shift, bank[0][2], bank[0][3], "RC1 anchor neighborhood"))
    for _, center, angle, stretch in bank:
        tangent = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        for side in (1, -1):
            offsets = np.array([[-0.58, -0.3], [0.48, -0.3], [0.0, 0.62 * side]]) * spacing
            points = center + offsets[:, :1] * tangent + offsets[:, 1:] * normal
            proposals.append((points, angle, stretch, "conditioned flow/strain anchor"))
    directions = (
        _starting_directions(np.asarray([p[0] for p in proposals]), prepared)
        if setup == "stretch-ovals"
        else None
    )
    candidates, descriptions = [], []
    for proposal_index, (centers, angle, _stretch_angle, label) in enumerate(proposals):
        if setup == "facing-banks":
            center = centers.mean(0)
            tangent = np.array([math.cos(angle), math.sin(angle)])
            normal = np.array([-tangent[1], tangent[0]])
            centers = np.array(
                [
                    center - spacing * 0.42 * normal,
                    center + spacing * 0.42 * normal,
                    center - spacing * 0.85 * tangent,
                ]
            )
        primitives = []
        for index in range(3):
            heading = float(directions[proposal_index, index]) if directions is not None else angle
            if setup in {"cross-strokes", "long-ribbons"}:
                heading += math.pi / 2
            parts = [(centers[index], 1.0)]
            if setup == "split-lobes" and index == dominant:
                axis = np.array([math.cos(angle), math.sin(angle)])
                parts = [
                    (centers[index] - radii[index] * 0.6 * axis, 0.7),
                    (centers[index] + radii[index] * 1.35 * axis, 0.3),
                ]
            for component, (center, share) in enumerate(parts):
                p = _primitive(center, radii[index] * math.sqrt(share), heading, setup, index)
                p.update(
                    pigment_index=index,
                    component=component,
                    target_mass=config["target_mass"][index] * share,
                )
                primitives.append(p)
        primitives = _separate(primitives, flow["aspect"], 0.008 * min(flow["aspect"], 1))
        if primitives is None:
            continue
        candidates.append(primitives)
        descriptions.append(label)
        if len(candidates) >= CANDIDATE_LIMIT:
            break
    if not candidates:
        raise ValueError("No separated choreography geometry fits inside the visible canvas")
    return candidates, descriptions


def _samples(primitive):
    """Deterministic equal-area samples of the actual convex painted support."""
    outline = _outline(primitive)
    low, high = outline.min(0), outline.max(0)
    q = (np.arange(24) + 0.5) / 24
    xx, yy = np.meshgrid(low[0] + q * (high[0] - low[0]), low[1] + q * (high[1] - low[1]))
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    if primitive["kind"] == "polygon":
        inside = np.ones(len(points), dtype=bool)
        for a, b in zip(outline, np.roll(outline, -1, axis=0), strict=True):
            edge = b - a
            inside &= edge[0] * (points[:, 1] - a[1]) - edge[1] * (points[:, 0] - a[0]) >= 0
    else:
        pp, rr = np.asarray(primitive["points"]), np.asarray(primitive["radii"])
        distance = np.full(len(points), np.inf)
        if len(pp) == 1:
            distance = np.linalg.norm(points - pp[0], axis=1) - rr[0]
        for j in range(len(pp) - 1):
            edge = pp[j + 1] - pp[j]
            t = np.clip(np.sum((points - pp[j]) * edge, axis=-1) / np.sum(edge**2), 0, 1)
            distance = np.minimum(
                distance,
                np.linalg.norm(points - pp[j] - t[:, None] * edge, axis=1)
                - (rr[j] + t * (rr[j + 1] - rr[j])),
            )
        inside = distance <= 0
    support = points[inside]
    if len(support) < PARTICLES_PER_COMPONENT:
        raise ValueError("Insufficient choreography quadrature support")
    # Serpentine ordering prevents the quantiles from selecting the same x on
    # every lattice row; each selected point represents equal support area.
    rows = np.repeat(np.arange(24), 24)[inside]
    order = np.lexsort((np.where(rows % 2, -support[:, 0], support[:, 0]), rows))
    selected = np.floor(
        (np.arange(PARTICLES_PER_COMPONENT) + 0.5) * len(support) / PARTICLES_PER_COMPONENT
    ).astype(int)
    area = float(np.prod(high - low) * inside.mean())
    return support[order[selected]], math.sqrt(area / math.pi)


def _stretch(points, weights, initial_covariance, initial_points):
    """Largest affine singular value from fixed material-point correspondence.

    A rigidly rotated ellipse must stay at one. Comparing world-axis covariance
    matrices instead incorrectly counts that rotation as stretching.
    """
    center = np.sum(points * weights[..., None], axis=-2)
    delta = points - center[..., None, :]
    covariance = np.einsum("...ni,...nj,...n->...ij", delta, delta, weights)
    initial_delta = (
        initial_points - np.sum(initial_points * weights[..., None], axis=-2)[..., None, :]
    )
    cross = np.einsum("...ni,...nj,...n->...ij", delta, initial_delta, weights)
    values, vectors = np.linalg.eigh(initial_covariance)
    inverse = np.einsum("...ik,...k,...jk->...ij", vectors, 1 / np.maximum(values, 1e-12), vectors)
    affine = np.einsum("...ij,...jk->...ik", cross, inverse)
    relative = np.einsum("...ji,...jk->...ik", affine, affine)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(relative)[..., -1], 0)), covariance


def _pilot(candidates, prepared, fractions, lower_speed):
    flow, tools, pairs, strains, _, _, _ = prepared
    samples = [[_samples(p) for p in row] for row in candidates]
    initial = np.asarray([[s[0] for s in row] for row in samples])
    radii = np.asarray([[s[1] for s in row] for row in samples])
    pigments = np.asarray([p["pigment_index"] for p in candidates[0]])
    n = PARTICLES_PER_COMPONENT
    points = np.repeat(initial[:, :, None], 2, axis=2)
    origin = points.copy()
    weights = np.broadcast_to(
        np.stack([fractions[pigments], 1 - fractions[pigments]], axis=1)[None, :, :, None] / n,
        points.shape[:-1],
    ).copy()
    flat_weights = weights.reshape(*weights.shape[:2], -1)
    centered = initial - initial.mean(axis=-2, keepdims=True)
    initial_covariance = np.einsum("...ni,...nj->...ij", centered, centered) / n
    motion = np.zeros_like(radii)
    peak_stretch = np.ones_like(radii)
    exposure = np.zeros_like(radii)
    contacted = np.zeros(points.shape[:-1], dtype=bool)
    moved = np.zeros(points.shape[:-1], dtype=bool)
    leaked = np.zeros(len(candidates), dtype=bool)
    speeds = np.array([1.0, lower_speed])[None, None, :, None, None]
    observations = 0
    for step, (force, swirl) in enumerate(zip(tools, pairs, strict=True)):
        current = flow if strains is None else {**flow, "strains": strains[step]}
        dt = 1 / len(tools)
        first = velocity(points, force, swirl, **current) * speeds
        points += velocity(points + first * (dt * 0.5), force, swirl, **current) * speeds * dt
        if (step + 1) % OBSERVATION_STRIDE and step != len(tools) - 1:
            continue
        observations += 1
        distance = np.linalg.norm(points - origin, axis=-1) / radii[:, :, None, None]
        motion = np.maximum(motion, np.sqrt(np.sum(weights * distance**2, axis=(-2, -1))))
        moved |= distance >= GATES["minimum_displacement_in_radii"]
        current_stretch, _covariance = _stretch(
            points.reshape(*points.shape[:2], -1, 2),
            flat_weights,
            initial_covariance,
            origin.reshape(*origin.shape[:2], -1, 2),
        )
        peak_stretch = np.maximum(peak_stretch, current_stretch)
        outside = np.any(np.abs(points) >= [flow["aspect"], 1], axis=-1)
        leaked |= np.any(outside & (weights > 0), axis=(1, 2, 3))
        proximity = np.zeros(points.shape[:-1])
        for a in range(len(pigments)):
            for b in range(a + 1, len(pigments)):
                if pigments[a] == pigments[b]:
                    continue
                aa, bb = (
                    points[:, a].reshape(len(points), -1, 2),
                    points[:, b].reshape(len(points), -1, 2),
                )
                squared = np.sum((aa[:, :, None] - bb[:, None, :]) ** 2, axis=-1)
                scale = (0.28 * np.minimum(radii[:, a], radii[:, b]))[:, None] ** 2
                # Zero-weight layers cannot confer participation to active paint.
                to_b = np.where(flat_weights[:, b, None, :] > 0, squared, np.inf)
                to_a = np.where(flat_weights[:, a, :, None] > 0, squared, np.inf)
                pa = np.exp(-to_b.min(axis=2) / scale).reshape(len(points), 2, n)
                pb = np.exp(-to_a.min(axis=1) / scale).reshape(len(points), 2, n)
                proximity[:, a] = np.maximum(proximity[:, a], pa)
                proximity[:, b] = np.maximum(proximity[:, b], pb)
        exposure += np.sum(weights * proximity, axis=(-2, -1))
        contacted |= proximity >= 0.1
    exposure /= observations
    reports = []
    for index, primitives in enumerate(candidates):
        rows, scores = [], []
        for component, p in enumerate(primitives):
            record = {
                "pigment_index": p["pigment_index"],
                "component": p["component"],
                "target_mass": p["target_mass"],
                "maximum_rms_displacement_in_radii": float(motion[index, component]),
                "peak_stretch": float(peak_stretch[index, component]),
                "final_stretch": float(current_stretch[index, component]),
                "moved_mass_fraction": float(
                    np.sum(weights[index, component] * moved[index, component])
                ),
                "neighbor_exposure": float(exposure[index, component]),
                "contacted_mass_fraction": float(
                    np.sum(weights[index, component] * contacted[index, component])
                ),
            }
            record["eligible"] = _qualifies(record)
            scores.append(_participation_score(record))
            rows.append(record)
        mass = np.asarray([p["target_mass"] for p in primitives])
        all_weights = (weights[index] * mass[:, None, None]).ravel() / mass.sum()
        all_points = points[index].reshape(-1, 2)
        center = np.sum(all_points * all_weights[:, None], axis=0)
        delta = all_points - center
        cov = np.einsum("ni,nj,n->ij", delta, delta, all_weights)
        eigen = np.linalg.eigvalsh(cov)
        aspect = float(math.sqrt(eigen[-1] / max(eigen[0], 1e-12)))
        balance = float(np.exp(-np.sum((center / [flow["aspect"], 1]) ** 2) / 0.5))
        reports.append(
            {
                "eligible": bool(not leaked[index] and all(r["eligible"] for r in rows)),
                "visible_boundary_crossed": bool(leaked[index]),
                "components": rows,
                "minimum_participation_score": float(min(scores)),
                "final_mass_centroid": center.tolist(),
                "final_mass_aspect_ratio": aspect,
                "centroid_balance": balance,
                "shape_tiebreak": float(
                    min(math.log(max(aspect, 1)) / math.log(3), 1) + 0.2 * balance
                ),
            }
        )
    _json_finite(reports)
    return reports


def _qualifies(row):
    return bool(
        row["maximum_rms_displacement_in_radii"] >= GATES["minimum_displacement_in_radii"]
        and row["peak_stretch"] >= GATES["minimum_peak_stretch"]
        and row["moved_mass_fraction"] >= GATES["minimum_moved_mass_fraction"]
        and row["neighbor_exposure"] >= GATES["minimum_neighbor_exposure"]
        and row["contacted_mass_fraction"] >= GATES["minimum_contacted_mass_fraction"]
    )


def _participation_score(row):
    return (
        0.45 * min(row["maximum_rms_displacement_in_radii"] / 1.5, 1)
        + 0.35 * min(row["neighbor_exposure"] / 0.08, 1)
        + 0.2 * min((row["peak_stretch"] - 1) / 0.8, 1)
    )


def plan_layout(source, simulation_config, palette):
    config = validate_config(simulation_config.get("initial_choreography"))
    if config is None:
        raise ValueError("Missing initial_choreography")
    seed = normalize_seed(source.seed)
    if (
        normalize_seed(palette["seed"]) != seed
        or palette.get("chalk_index") != 3
        or len(palette.get("pigments_srgb", [])) != 4
    ):
        raise ValueError("Choreography needs the source's three-pigment palette followed by chalk")
    baseline = layer_fractions(palette, 4)
    effective = effective_layer_fractions(palette, config)
    lower_speed = _number(
        simulation_config.get("lower_transport_scale", 0.82), "lower speed", 0.25, 1
    )
    prepared = _source_data(source, simulation_config)
    geometric_config = {k: v for k, v in config.items() if k != "mobility_bias"}
    cache_key = hashlib.sha256(
        json.dumps(
            [
                prepared[-1],
                geometric_config,
                baseline.tolist(),
                lower_speed,
                CANDIDATE_LIMIT,
                PARTICLES_PER_COMPONENT,
                OBSERVATION_STRIDE,
                GATES,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if cache_key in _LAYOUTS:
        _LAYOUTS.move_to_end(cache_key)
        result = copy.deepcopy(_LAYOUTS[cache_key])
        result["config"] = config
        result["effective_layer_fractions"] = effective.tolist()
        validate_layout(result)
        return result
    candidates, descriptions = _candidates(config, prepared)
    reports = _pilot(candidates, prepared, baseline[:3].astype("f8"), lower_speed)
    eligible = [i for i, r in enumerate(reports) if r["eligible"]]
    if not eligible:
        best = max(reports, key=lambda r: r["minimum_participation_score"])
        raise ValueError(
            "No qualified choreography layout: every candidate has an inactive component "
            "or boundary crossing; " + json.dumps(best, sort_keys=True, allow_nan=False)
        )
    chosen = max(
        eligible,
        key=lambda i: (
            round(reports[i]["minimum_participation_score"], 2),
            reports[i]["shape_tiebreak"],
            -i,
        ),
    )
    result = {
        "version": VERSION,
        "seed": seed,
        "count": 3,
        "aspect": float(source.aspect),
        "setup": config["setup"],
        "config": config,
        "target_mass": config["target_mass"],
        "reference_radii": config["reference_radii"],
        "edge_width_world": min(config["reference_radii"]) * 0.012,
        "source_sha256": source.sha256,
        "source_projection": copy.deepcopy(source.projection),
        "baseline_layer_fractions": baseline.tolist(),
        "effective_layer_fractions": effective.tolist(),
        "primitives": candidates[chosen],
        "pilot": {
            "steps": PILOT_STEPS,
            "particles_per_component_per_layer": PARTICLES_PER_COMPONENT,
            "observation_stride": OBSERVATION_STRIDE,
            "candidate_limit": CANDIDATE_LIMIT,
            "evaluated_candidates": len(candidates),
            "eligible_candidates": len(eligible),
            "selected_index": chosen,
            "anchor_method": descriptions[chosen],
            "source_settings_sha256": prepared[-1],
            "flow": copy.deepcopy(prepared[0]),
            "lower_transport_scale": lower_speed,
            "gates": copy.deepcopy(GATES),
            "selection": reports[chosen],
            "orientation_model": (
                "measured flow direction; ovals use finite-time deformation from t0 to 0.5"
            ),
            "selection_rule": (
                "qualified components first; rounded minimum participation; "
                "noncircular covariance/balance tiebreak; earliest candidate"
            ),
            "limitations": (
                "Equal-area geometry tracers at two fixed baseline layer speeds; no diffusion, "
                "layer exchange or native paint reaction. Covariance describes shape, not beauty. "
                "Stretch is a rotation-invariant affine fit and can miss nonlinear folding. "
                "Mobility bias does not change layout selection."
            ),
        },
        "peak_concentration_limits": (
            PEAK_REFERENCE_MULTIPLIER
            * np.asarray(config["target_mass"])
            / (math.pi * np.asarray(config["reference_radii"]) ** 2)
        ).tolist(),
    }
    validate_layout(result)
    cached = copy.deepcopy(result)
    cached["config"] = geometric_config
    cached["effective_layer_fractions"] = baseline.tolist()
    _LAYOUTS[cache_key] = cached
    if len(_LAYOUTS) > 128:
        _LAYOUTS.popitem(last=False)
    return result


def validate_layout(layout):
    keys = {
        "version",
        "seed",
        "count",
        "aspect",
        "setup",
        "config",
        "target_mass",
        "reference_radii",
        "edge_width_world",
        "source_sha256",
        "source_projection",
        "baseline_layer_fractions",
        "effective_layer_fractions",
        "primitives",
        "pilot",
        "peak_concentration_limits",
    }
    if (
        type(layout) is not dict
        or set(layout) != keys
        or layout["version"] != VERSION
        or type(layout["count"]) is not int
        or layout["count"] != 3
    ):
        raise ValueError("Invalid choreography layout schema")
    _json_finite(layout)
    config = validate_config(layout["config"])
    if config != layout["config"] or any(
        layout[k] != config[k] for k in ("setup", "target_mass", "reference_radii")
    ):
        raise ValueError("Choreography layout/config mismatch")
    if normalize_seed(layout["seed"]) != layout["seed"] or not re.fullmatch(
        r"[0-9a-f]{64}", layout["source_sha256"]
    ):
        raise ValueError("Invalid choreography source identity")
    aspect = _number(layout["aspect"], "aspect", 0.2, 5)
    if type(layout["source_projection"]) is not dict:
        raise ValueError("Choreography source projection must be an object")
    if layout["edge_width_world"] != min(config["reference_radii"]) * 0.012:
        raise ValueError("Choreography edge width differs from its config")
    baseline = layer_fractions({"layer_fractions": layout["baseline_layer_fractions"]}, 4)
    effective = effective_layer_fractions({"layer_fractions": baseline.tolist()}, config)
    if layout["effective_layer_fractions"] != effective.tolist():
        raise ValueError("Choreography effective layer fractions differ")
    expected_peak = (
        PEAK_REFERENCE_MULTIPLIER
        * np.asarray(config["target_mass"])
        / (math.pi * np.asarray(config["reference_radii"]) ** 2)
    ).tolist()
    if layout["peak_concentration_limits"] != expected_peak:
        raise ValueError("Choreography peak limits differ from reference disk density")
    parts = layout["primitives"]
    if type(parts) is not list or len(parts) != (4 if config["setup"] == "split-lobes" else 3):
        raise ValueError("Invalid choreography component count")
    masses = np.zeros(3)
    indices = {i: [] for i in range(3)}
    for p in parts:
        expected = {"kind", "points", "pigment_index", "component", "target_mass"}
        if p.get("kind") == "stroke":
            expected.add("radii")
        if (
            set(p) != expected
            or type(p["pigment_index"]) is not int
            or p["pigment_index"] not in indices
            or type(p["component"]) is not int
        ):
            raise ValueError("Invalid choreography primitive schema")
        masses[p["pigment_index"]] += _number(p["target_mass"], "component target mass", 1e-12, 100)
        indices[p["pigment_index"]].append(p["component"])
    if any(
        sorted(row) != list(range(len(row))) or not row for row in indices.values()
    ) or not np.allclose(masses, config["target_mass"], rtol=1e-12, atol=1e-14):
        raise ValueError("Choreography component identity or mass mismatch")
    vector_raster.validate_primitives(parts, aspect)
    for i, p in enumerate(parts):
        for q in parts[i + 1 :]:
            if _separation(_outline(p), _outline(q), 0.0) is not None:
                raise ValueError("Choreography components overlap initially")
    pilot = layout["pilot"]
    pkeys = {
        "steps",
        "particles_per_component_per_layer",
        "observation_stride",
        "candidate_limit",
        "evaluated_candidates",
        "eligible_candidates",
        "selected_index",
        "anchor_method",
        "source_settings_sha256",
        "flow",
        "lower_transport_scale",
        "gates",
        "selection",
        "selection_rule",
        "orientation_model",
        "limitations",
    }
    if type(pilot) is not dict or set(pilot) != pkeys or pilot["gates"] != GATES:
        raise ValueError("Invalid choreography pilot schema")
    for name, expected in (
        ("steps", PILOT_STEPS),
        ("particles_per_component_per_layer", PARTICLES_PER_COMPONENT),
        ("observation_stride", OBSERVATION_STRIDE),
        ("candidate_limit", CANDIDATE_LIMIT),
    ):
        if type(pilot[name]) is not int or pilot[name] != expected:
            raise ValueError("Choreography pilot constants differ from its version")
    if type(pilot["source_settings_sha256"]) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", pilot["source_settings_sha256"]
    ):
        raise ValueError("Invalid choreography source settings hash")
    _number(pilot["lower_transport_scale"], "pilot lower speed", 0.25, 1)
    for name in (
        "steps",
        "particles_per_component_per_layer",
        "observation_stride",
        "candidate_limit",
        "evaluated_candidates",
        "eligible_candidates",
    ):
        if type(pilot[name]) is not int or pilot[name] < 1:
            raise ValueError("Invalid choreography pilot dimensions")
    if (
        not 1
        <= pilot["eligible_candidates"]
        <= pilot["evaluated_candidates"]
        <= pilot["candidate_limit"]
        or type(pilot["selected_index"]) is not int
        or not 0 <= pilot["selected_index"] < pilot["evaluated_candidates"]
    ):
        raise ValueError("Invalid choreography candidate selection")
    selection = pilot["selection"]
    if (
        type(selection) is not dict
        or set(selection)
        != {
            "eligible",
            "visible_boundary_crossed",
            "components",
            "minimum_participation_score",
            "final_mass_centroid",
            "final_mass_aspect_ratio",
            "centroid_balance",
            "shape_tiebreak",
        }
        or selection.get("eligible") is not True
        or selection.get("visible_boundary_crossed") is not False
        or len(selection.get("components", [])) != len(parts)
    ):
        raise ValueError("Choreography did not qualify every component")
    for part, report in zip(parts, selection["components"], strict=True):
        if type(report) is not dict or set(report) != {
            "pigment_index",
            "component",
            "target_mass",
            "maximum_rms_displacement_in_radii",
            "peak_stretch",
            "final_stretch",
            "moved_mass_fraction",
            "neighbor_exposure",
            "contacted_mass_fraction",
            "eligible",
        }:
            raise ValueError("Invalid choreography component report schema")
        for name in ("moved_mass_fraction", "neighbor_exposure", "contacted_mass_fraction"):
            _number(report[name], name, 0, 1)
        for name in ("maximum_rms_displacement_in_radii", "peak_stretch", "final_stretch"):
            _number(
                report[name],
                name,
                0 if name == "maximum_rms_displacement_in_radii" else 1e-12,
                1e12,
            )
        if (
            report.get("eligible") is not True
            or any(report.get(k) != part[k] for k in ("pigment_index", "component", "target_mass"))
            or not _qualifies(report)
            or report["peak_stretch"] < max(1, report["final_stretch"])
        ):
            raise ValueError("Invalid choreography component participation report")
        if type(report["pigment_index"]) is not int or type(report["component"]) is not int:
            raise ValueError("Choreography report indices must be integers")
        _number(report["target_mass"], "reported component mass", 1e-12, 100)
    minimum = min(_participation_score(r) for r in selection["components"])
    _number(selection["minimum_participation_score"], "participation score", 0, 1)
    centroid = selection["final_mass_centroid"]
    if type(centroid) is not list or len(centroid) != 2:
        raise ValueError("Invalid choreography final centroid")
    center = np.asarray([_number(v, "centroid", -5, 5) for v in centroid])
    shape_aspect = _number(selection["final_mass_aspect_ratio"], "final aspect", 1, 1e12)
    balance = float(np.exp(-np.sum((center / [aspect, 1]) ** 2) / 0.5))
    shape = min(math.log(shape_aspect) / math.log(3), 1) + 0.2 * balance
    for name, expected in (
        ("minimum_participation_score", minimum),
        ("centroid_balance", balance),
        ("shape_tiebreak", shape),
    ):
        if type(selection[name]) not in (int, float) or not math.isclose(
            selection[name], expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Inconsistent choreography selection score")
    return layout


def rasterize(layout, resolution, domain_scale):
    validate_layout(layout)
    return vector_raster.rasterize(
        layout["primitives"],
        resolution,
        domain_scale,
        aspect=layout["aspect"],
        target_mass=layout["target_mass"],
        edge_width_world=layout["edge_width_world"],
        peak_limits=layout["peak_concentration_limits"],
    )
