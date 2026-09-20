"""Select recorded bodies that may force paint, without changing the recording.

The complete three-body trajectories, fitted source coordinates and initial
painting remain fixed. Proper subsets gate material forcing only. The full set
is represented by omission, preserving the existing numerical path exactly.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

VERSION = "body-influence-v1"
PAIRS = ((0, 1), (1, 2), (2, 0))


def normalize_bodies(value):
    """Return a sorted proper subset, or None for the original all-body case."""
    if value is None:
        return None
    if (
        type(value) not in (list, tuple)
        or not 1 <= len(value) <= 3
        or any(type(body) is not int or body not in (0, 1, 2) for body in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("Body influence requires one to three distinct body indices in [0, 2]")
    bodies = tuple(sorted(value))
    return None if bodies == (0, 1, 2) else bodies


def validate_config(value=None):
    """Normalize the optional versioned selection without adding a default key."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - {"version", "bodies"} or "bodies" not in value:
        raise ValueError("Body influence config requires bodies and an optional version")
    if value.get("version", VERSION) != VERSION:
        raise ValueError(f"Body influence version must be {VERSION}")
    bodies = normalize_bodies(value["bodies"])
    if value["bodies"] is None:
        raise ValueError("An explicit body influence config needs a body list")
    return None if bodies is None else {"version": VERSION, "bodies": list(bodies)}


def active_bodies(config):
    settings = validate_config(config)
    return (0, 1, 2) if settings is None else tuple(settings["bodies"])


def eligible_pairs(config):
    selected = set(active_bodies(config))
    return tuple(pair for pair in PAIRS if set(pair) <= selected)


def initialization_config(config):
    """Exclude the experimental mask from the unchanged full-source layout pilot."""
    if "body_influence" not in config:
        return config
    return {key: copy.deepcopy(value) for key, value in config.items() if key != "body_influence"}


def validate_event_eligibility(events, config):
    """Require reduced schedules to contain only original-index active pairs."""
    settings = validate_config(config)
    if settings is None:
        return
    allowed = eligible_pairs(settings)
    if type(events) is not list:
        raise ValueError("Body-influence events must be a list")
    for event in events:
        pair = event.get("pair") if type(event) is dict else None
        if (
            type(pair) not in (list, tuple)
            or len(pair) != 2
            or any(type(body) is not int for body in pair)
            or tuple(pair) not in allowed
        ):
            raise ValueError(
                "Body-influence events require a canonical pair with both bodies active"
            )


def arc_travel(start, end, config):
    """Compute active-only arc differences before any unused arithmetic occurs."""
    a, b = np.asarray(start), np.asarray(end)
    selected = active_bodies(config)
    if any(value.shape != (3,) or value.dtype.kind not in "fiu" for value in (a, b)):
        raise ValueError("Body travel requires three numeric source distances")
    if not all(np.isfinite(value[list(selected)]).all() for value in (a, b)):
        raise ValueError("Active source travel must be finite")
    result = np.zeros(3, dtype="f8")
    with np.errstate(over="raise", invalid="raise"):
        result[list(selected)] = b[list(selected)].astype("f8") - a[list(selected)].astype("f8")
    return result


def movement_segments(start, end, config):
    """Use actual active paths and entirely zero unused brush/wetting segments."""
    a, b = np.asarray(start), np.asarray(end)
    selected = active_bodies(config)
    if any(value.shape != (3, 2) or value.dtype.kind not in "fiu" for value in (a, b)) or not all(
        np.isfinite(value[list(selected)]).all() for value in (a, b)
    ):
        raise ValueError("Active body segments need finite planar source positions")
    result = np.zeros((3, 4), dtype="f4")
    with np.errstate(over="raise", invalid="raise"):
        result[list(selected)] = np.concatenate((a[list(selected)], b[list(selected)]), axis=1)
    return result


def forcing_uniforms(frame, radius, config, *, include_strain=False):
    """Build unchanged active descriptors and zero every disabled descriptor row.

    Source validates the full recording upstream. At this adapter, unused rows
    are replaced before conditioning so they cannot introduce overflow/NaNs into
    an otherwise valid active contribution. No source arrays are modified.
    """
    from .pair_strain import pair_strain_uniforms
    from .participation_layout import conditioned_uniforms

    if type(include_strain) is not bool:
        raise ValueError("include_strain must be a boolean")
    selected = active_bodies(config)
    pair_mask = np.array([a in selected and b in selected for a, b in PAIRS], dtype=bool)
    position, velocity, distance = (
        np.asarray(value) for value in (frame.positions, frame.velocities, frame.pair_distances)
    )
    if (
        position.shape != (3, 2)
        or velocity.shape != (3, 2)
        or distance.shape != (3,)
        or any(value.dtype.kind not in "fiu" for value in (position, velocity, distance))
        or not np.isfinite(position[list(selected)]).all()
        or not np.isfinite(velocity[list(selected)]).all()
        or not np.isfinite(distance[pair_mask]).all()
        or np.any(distance[pair_mask] < 0)
    ):
        raise ValueError("Invalid active source forcing measurements")
    p, v, d = np.zeros((3, 2), "f8"), np.zeros((3, 2), "f8"), np.zeros(3, "f8")
    p[list(selected)], v[list(selected)], d[pair_mask] = (
        position[list(selected)],
        velocity[list(selected)],
        distance[pair_mask],
    )
    sanitized = SimpleNamespace(positions=p, velocities=v, pair_distances=d)
    tools, pairs = conditioned_uniforms(sanitized, radius)
    disabled = [body for body in range(3) if body not in selected]
    tools[disabled] = 0
    pairs[~pair_mask] = 0
    strains = pair_strain_uniforms(sanitized, radius) if include_strain else None
    if strains is not None:
        strains[~pair_mask] = 0
    if not all(np.isfinite(value).all() for value in (tools, pairs)) or (
        strains is not None and not np.isfinite(strains).all()
    ):
        raise ValueError("Active source forcing exceeds finite float32 uniforms")
    return tools, pairs, strains
