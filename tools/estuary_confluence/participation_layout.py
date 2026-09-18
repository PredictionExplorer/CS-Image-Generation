"""Source-aware pure-pool placement with a bounded, approximate particle pilot.

The pilot uses Estuary's actual conditioned streamfunction, but samples it on a
coarser fixed clock and transports area-distributed tracers rather than paint. It is a
candidate filter, not a guarantee of GPU pigment participation or mixing. The
chosen layout records its diagnostics even when no candidate meets every gate.
No palette, diffusion, paint resolution, or output cadence enters selection.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from tools.estuary.flow_reference import velocity

from .layout import MAX_LOAD_RADIUS, _number
from .palette import normalize_seed

VERSION = "engaged-pigment-layout-v2"
# Keep placement and material amounts fixed when comparing the v1/v2 pilots.
SEED_NAMESPACE = "engaged-pigment-layout-v1"
CANDIDATE_LAYOUTS = 32
MAX_PLACEMENT_ATTEMPTS = 128
PROPOSALS_PER_POOL = 256
PILOT_STEPS = 1024
PARTICLES_PER_POOL = 17
OBSERVATION_STRIDE = 8


def _settings(source, config):
    if type(config) is not dict:
        raise ValueError("Engaged layout config must be an object")
    aspect = _number(source.aspect, "aspect", 0.2, 5)
    carrier = config.get("carrier_velocity", [0.0, 0.0])
    if type(carrier) not in (list, tuple) or len(carrier) != 2:
        raise ValueError("carrier_velocity needs two values")
    flow_domain = config.get("flow_domain_scale")
    if flow_domain is None:
        flow_domain = config.get("domain_scale", 1.6)
    flow = {
        "aspect": aspect,
        "stir_radius": _number(config.get("stir_radius", 0.22), "stir_radius", 0.02, 2),
        "flow_strength": _number(config.get("flow_strength", 1.1), "flow_strength", 0, 8),
        "pair_swirl": _number(config.get("pair_swirl", 0.9), "pair_swirl", 0, 8),
        "domain_scale": _number(flow_domain, "flow_domain_scale", 1, 2.5),
        "carrier_velocity": [_number(v, "carrier_velocity", -8, 8) for v in carrier],
    }
    pools = {
        "load_radius": _number(
            config.get("load_radius", 0.28), "load_radius", 0.001, MAX_LOAD_RADIUS
        ),
        "initial_load": _number(config.get("initial_load", 0.18), "initial_load", 1e-9, 10),
        "edge_width": _number(
            config.get("initial_edge_width", 0.02), "initial_edge_width", 0.001, 0.25
        ),
    }
    return flow, pools


def conditioned_uniforms(frame, radius):
    """CPU-only version of estuary.engine.tool_uniforms, including f32 uniforms.

    Accepts one frame or a batch from Source.sample. The pair denominator retains
    physical 3D distance, so projected crossings do not manufacture rotation.
    """
    radius = _number(radius, "stir_radius", 0.02, 2)
    position = np.asarray(frame.positions, dtype=np.float64)
    speed = np.asarray(frame.velocities, dtype=np.float64)
    distance = np.asarray(frame.pair_distances, dtype=np.float64)
    if (
        position.shape[-2:] != (3, 2)
        or speed.shape != position.shape
        or distance.shape != (*position.shape[:-2], 3)
        or not all(np.isfinite(a).all() for a in (position, speed, distance))
        or np.any(distance < 0)
    ):
        raise ValueError("Invalid source measurements for participation pilot")
    speed = speed / (1 + np.linalg.norm(speed, axis=-1, keepdims=True) / 24)
    tools = np.concatenate([position, speed], axis=-1).astype("f4")
    pairs = []
    for index, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        delta, relative = (
            position[..., b, :] - position[..., a, :],
            speed[..., b, :] - speed[..., a, :],
        )
        spin = delta[..., 0] * relative[..., 1] - delta[..., 1] * relative[..., 0]
        spin /= distance[..., index] ** 2 + radius**2
        spin = 20 * np.tanh(spin / 20)
        pairs.append(
            np.concatenate(
                [(position[..., a, :] + position[..., b, :]) * 0.5, spin[..., None]], axis=-1
            )
        )
    return tools, np.stack(pairs, axis=-2).astype("f4")


def flow_velocity(source, points, fraction, config):
    """Evaluate the pilot's exact analytic force model for independent diagnostics."""
    flow, _ = _settings(source, config)
    tools, pairs = conditioned_uniforms(source.frame(fraction), flow["stir_radius"])
    return velocity(points, tools, pairs, **flow)


def _unit(seed, label):
    digest = hashlib.sha256(
        SEED_NAMESPACE.encode() + b"\0" + seed + b"\0" + label.encode()
    ).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def _candidates(seed, tools, pairs, flow, settings):
    aspect, radius = flow["aspect"], flow["stir_radius"]
    short = min(aspect, 1)
    anchors = np.concatenate([tools[..., :2], pairs[..., :2]], axis=1).reshape(-1, 2)
    weights = np.concatenate(
        [
            np.linalg.norm(tools[..., 2:], axis=-1) * flow["flow_strength"],
            np.abs(pairs[..., 2]) * (1.7 * radius) * flow["pair_swirl"],
        ],
        axis=1,
    ).reshape(-1)
    envelope = np.maximum(
        1 - (anchors / [aspect * flow["domain_scale"], flow["domain_scale"]]) ** 2, 0
    )
    weights *= np.prod(envelope**2, axis=-1)
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        raise ValueError("The recording has no active source stirring under these flow settings")
    cdf = np.cumsum(weights, dtype=np.float64)
    cdf /= cdf[-1]
    pools = []
    for index in range(5):
        r = settings["load_radius"] * short * (0.60 + 0.14 * _unit(seed, f"pool/{index}/radius"))
        pools.append(
            {
                "pigment_index": index,
                "radius": r,
                "load": settings["initial_load"]
                * (0.90 + 0.20 * _unit(seed, f"pool/{index}/load")),
                "edge_width": settings["edge_width"],
                "edge_width_world": r * settings["edge_width"],
            }
        )
    minimum_gap, contact_gap = 0.008 * short, 0.35 * radius
    candidates, attempts = [], []
    for attempt in range(MAX_PLACEMENT_ATTEMPTS):
        placed = []
        for index, pool in enumerate(pools):
            r = pool["radius"]
            for proposal in range(PROPOSALS_PER_POOL):
                name = f"placement/{attempt}/{index}/{proposal}"
                selected = min(
                    int(np.searchsorted(cdf, _unit(seed, name + "/anchor"))), len(cdf) - 1
                )
                angle = 2 * math.pi * _unit(seed, name + "/angle")
                offset = (r + 0.85 * radius) * math.sqrt(_unit(seed, name + "/offset"))
                point = anchors[selected] + offset * np.array([math.cos(angle), math.sin(angle)])
                if (
                    abs(point[0]) + r >= aspect - 0.025 * short
                    or abs(point[1]) + r >= 1 - 0.025 * short
                ):
                    continue
                gaps = [
                    float(np.linalg.norm(point - p["position"])) - r - p["radius"] for p in placed
                ]
                if gaps and (min(gaps) < minimum_gap or min(gaps) > contact_gap):
                    continue
                placed.append({**pool, "position": point.tolist()})
                break
            else:
                break
        if len(placed) == 5:
            candidates.append(placed)
            attempts.append(attempt)
            if len(candidates) == CANDIDATE_LAYOUTS:
                break
    if not candidates:
        raise ValueError(
            "No separated engaged layout fits this source and pool radius within the bounded search"
        )
    return candidates, attempts


def _particle_offsets():
    """Equal-weight area quadrature: center, eight inner, eight outer tracers.

    Each ring represents eight seventeenths of the disk area; its squared
    radius is the midpoint of that annulus's squared-radius interval. Angular
    staggering avoids aligned radial spokes while preserving a zero centroid.
    """
    angles = np.arange(8) * 2 * math.pi / 8
    return np.concatenate(
        [
            np.zeros((1, 2)),
            math.sqrt(5 / 17) * np.stack([np.cos(angles), np.sin(angles)], axis=1),
            math.sqrt(13 / 17)
            * np.stack([np.cos(angles + math.pi / 8), np.sin(angles + math.pi / 8)], axis=1),
        ]
    )


def _tracer_proximity(points, radii):
    """Directed nearest-other proximity for every tracer and neighboring color.

    Retaining the tracer axis matters: one edge contact must not award contact
    to the whole pigment pool. The caller first chooses each tracer's nearest
    other color, then averages the pool's equal-area tracers.
    """
    result = np.zeros((*points.shape[:-1], 5))
    for a in range(5):
        for b in range(a + 1, 5):
            separation = points[:, a, :, None, :] - points[:, b, None, :, :]
            squared = np.sum(separation**2, axis=-1)
            scale = (0.28 * np.minimum(radii[:, a], radii[:, b]))[:, None]
            result[:, a, :, b] = np.exp(-squared.min(axis=2) / scale**2)
            result[:, b, :, a] = np.exp(-squared.min(axis=1) / scale**2)
    return result


def _pilot(candidates, tools, pairs, flow):
    centers = np.array([[p["position"] for p in layout] for layout in candidates])
    radii = np.array([[p["radius"] for p in layout] for layout in candidates])
    offsets = _particle_offsets()
    unit_initial_variance = float(np.linalg.eigvalsh(offsets.T @ offsets / len(offsets))[-1])
    initial = centers[:, :, None, :] + radii[:, :, None, None] * offsets[None, None]
    points = initial.copy()
    travel, displacement, stretch = (np.zeros_like(radii) for _ in range(3))
    contact = {count: np.zeros((len(candidates), count)) for count in (3, 5)}
    leaked = np.zeros(len(candidates), dtype=bool)
    observations = 0
    dt = 1 / len(tools)
    for step, (force, swirl) in enumerate(zip(tools, pairs, strict=True)):
        first = velocity(points, force, swirl, **flow)
        midpoint = points + first * (dt * 0.5)
        delta = velocity(midpoint, force, swirl, **flow) * dt
        points += delta
        travel += np.linalg.norm(delta, axis=-1).mean(axis=-1) / radii
        if (step + 1) % OBSERVATION_STRIDE != 0 and step != len(tools) - 1:
            continue
        observations += 1
        leaked |= np.any(
            (np.abs(points[..., 0]) >= flow["aspect"]) | (np.abs(points[..., 1]) >= 1), axis=(1, 2)
        )
        displacement = np.maximum(
            displacement,
            np.sqrt(np.mean(np.sum((points - initial) ** 2, axis=-1), axis=-1)) / radii,
        )
        centered = points - points.mean(axis=-2, keepdims=True)
        xx = np.mean(centered[..., 0] ** 2, axis=-1)
        yy = np.mean(centered[..., 1] ** 2, axis=-1)
        xy = np.mean(centered[..., 0] * centered[..., 1], axis=-1)
        largest = 0.5 * (xx + yy + np.sqrt((xx - yy) ** 2 + 4 * xy**2))
        initial_variance = radii**2 * unit_initial_variance
        stretch = np.maximum(stretch, np.sqrt(np.maximum(largest, 0) / initial_variance))
        proximity = _tracer_proximity(points, radii)
        for count in (3, 5):
            contact[count] += proximity[:, :count, :, :count].max(axis=-1).mean(axis=-1)
    if not all(
        np.isfinite(a).all() for a in (points, travel, displacement, stretch, *contact.values())
    ):
        raise FloatingPointError("Nonfinite engaged particle pilot")
    for count in contact:
        contact[count] /= observations
    masses = np.array([[p["load"] * p["radius"] ** 2 for p in layout] for layout in candidates])
    centroid = (
        np.sum(points.mean(axis=-2) * masses[..., None], axis=1) / masses.sum(axis=1)[:, None]
    )
    balance = np.exp(-np.sum((centroid / [flow["aspect"], 1]) ** 2, axis=-1) / 0.25)
    reports = []
    for index in range(len(candidates)):
        prefixes = {}
        scores, eligible = [], not bool(leaked[index])
        for count in (3, 5):
            exposure = contact[count][index]
            moved = displacement[index, :count]
            movement = moved / (moved + 1.2)
            strained = np.log(np.maximum(stretch[index, :count], 1))
            deformation = strained / (strained + 0.7)
            score = 0.35 * movement + 0.25 * deformation + 0.4 * exposure / (exposure + 0.10)
            flags = (
                (displacement[index, :count] >= 0.45)
                & (stretch[index, :count] >= 1.08)
                & (exposure >= 0.015)
            )
            eligible &= bool(np.all(flags))
            prefix_mass = masses[index, :count]
            prefix_centroid = (
                np.sum(points[index, :count].mean(axis=-2) * prefix_mass[:, None], axis=0)
                / prefix_mass.sum()
            )
            prefix_balance = float(
                np.exp(-np.sum((prefix_centroid / [flow["aspect"], 1]) ** 2) / 0.25)
            )
            scores.append(float(score.min() + 0.20 * score.mean() + 0.10 * prefix_balance))
            rows = [
                {
                    "pigment_index": pigment,
                    "maximum_rms_displacement_in_radii": round(
                        float(displacement[index, pigment]), 8
                    ),
                    "travel_in_radii": round(float(travel[index, pigment]), 8),
                    "maximum_stretch": round(float(stretch[index, pigment]), 8),
                    "neighbor_exposure": round(float(exposure[pigment]), 8),
                    "eligible": bool(flags[pigment]),
                }
                for pigment in range(count)
            ]
            prefixes[str(count)] = {
                "minimum_score": round(float(score.min()), 8),
                "final_centroid": prefix_centroid.tolist(),
                "centroid_balance": round(prefix_balance, 8),
                "per_pool": rows,
            }
        reports.append(
            {
                "eligible": eligible,
                "score": round(min(scores), 8),
                "visible_boundary_crossed": bool(leaked[index]),
                "final_centroid": centroid[index].tolist(),
                "centroid_balance": round(float(balance[index]), 8),
                "prefix_three": prefixes["3"],
                "all_five": prefixes["5"],
            }
        )
    return reports


def plan_engaged_layout(source, count, config):
    """Resolve five source-aware pools, then select the identical 3/5 prefix."""
    if type(count) is not int or count not in (3, 5):
        raise ValueError("Engaged pigment count must be 3 or 5")
    seed = normalize_seed(source.seed)
    flow, settings = _settings(source, config)
    fractions = (np.arange(PILOT_STEPS, dtype=np.float64) + 0.5) / PILOT_STEPS
    sampled = source.sample(fractions)
    tools, pairs = conditioned_uniforms(sampled, flow["stir_radius"])
    if tools.shape != (PILOT_STEPS, 3, 4):
        raise ValueError("Source sampling dimensions differ from the pilot clock")
    candidates, attempts = _candidates(
        int(seed, 16).to_bytes(32, "big"), tools, pairs, flow, settings
    )
    reports = _pilot(candidates, tools, pairs, flow)
    chosen = max(
        range(len(reports)), key=lambda i: (reports[i]["eligible"], reports[i]["score"], -i)
    )
    return {
        "version": VERSION,
        "seed_namespace": SEED_NAMESPACE,
        "seed": seed,
        "count": count,
        "aspect": flow["aspect"],
        "source_sha256": source.sha256,
        "source_projection": copy.deepcopy(source.projection),
        "coordinate_system": "fixed visible canvas: x [-aspect, aspect], y [-1, 1]",
        **settings,
        "construction": (
            "five source-aware separated pools selected jointly for three and five colors"
        ),
        "placement_method": "source-activity proposals and bounded particle pilot",
        "placement_attempt": attempts[chosen],
        "flow": flow,
        "pilot": {
            "steps": PILOT_STEPS,
            "integration": "midpoint RK2 with frozen interval-midpoint source",
            "particles_per_pool": PARTICLES_PER_POOL,
            "tracer_geometry": (
                "equal-weight area quadrature: center + two staggered eight-point rings"
            ),
            "tracer_offsets_in_radii": _particle_offsets().tolist(),
            "contact_operator": (
                "temporal mean of each tracer's nearest-other-color proximity, "
                "averaged over pool area tracers"
            ),
            "observation_stride": OBSERVATION_STRIDE,
            "candidate_limit": CANDIDATE_LAYOUTS,
            "evaluated_candidates": len(candidates),
            "placement_attempt_limit": MAX_PLACEMENT_ATTEMPTS,
            "proposals_per_pool": PROPOSALS_PER_POOL,
            "selection_model": {
                "per_pool_score": "0.35*d/(d+1.2)+0.25*log(s)/(log(s)+0.7)+0.40*e/(e+0.10)",
                "quantities": (
                    "d=maximum RMS displacement/radius; s=maximum stretch; e=neighbor exposure"
                ),
                "joint_score": (
                    "minimum over 3/5 prefixes of min(score)+0.20*mean(score)+0.10*centroid_balance"
                ),
                "centroid_balance": "exp(-squared normalized centroid distance/0.25)",
                "contact_radius_fraction": 0.28,
                "minimum_displacement_in_radii": 0.45,
                "minimum_stretch": 1.08,
                "minimum_neighbor_exposure": 0.015,
                "priority": "eligible first; then larger joint score; then earliest candidate",
            },
            "placement_controls": {
                "radius_factors": [0.60, 0.74],
                "load_factors": [0.90, 1.10],
                "minimum_gap_short_extent": 0.008,
                "maximum_neighbor_gap_stir_radius": 0.35,
                "canvas_margin_short_extent": 0.025,
                "anchor_offset": "uniform-area disk of pool radius + 0.85*stir radius",
                "anchor_weights": (
                    "conditioned source speeds and pair spins, times boundary envelope"
                ),
            },
            "selection": reports[chosen],
            "eligible_candidates": sum(report["eligible"] for report in reports),
            "limitations": (
                "coarse area tracers; no diffusion or pigment phase exchange; "
                "GPU participation must be checked"
            ),
        },
        "pools": candidates[chosen][:count],
    }
