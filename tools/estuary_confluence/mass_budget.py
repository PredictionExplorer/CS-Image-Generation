"""Archive contracts for explicit, global pigment-budget restoration.

This checks integral amounts, not local transport accuracy. Uniform channel
scaling cannot recover misplaced paint or unresolved spatial detail.
"""

from __future__ import annotations

import math

import numpy as np

from tools.estuary_studio.common import require

from .palette import SUPPORTED_CHROMATIC_COUNTS

VERSION = "global-pigment-budget-v1"
RELATIVE_TOLERANCE = 5e-6


def validate_initial_weights(value, count=None):
    """Resolve explicit chromatic load multipliers; chalk is never included."""
    if value is None:
        return None
    if (
        type(value) is not list
        or len(value) not in SUPPORTED_CHROMATIC_COUNTS
        or (count is not None and len(value) != count)
        or any(type(v) not in (int, float) or not 0.05 <= v <= 5 for v in value)
    ):
        raise ValueError(
            "initial_pigment_weights needs one finite multiplier in [0.05, 5] "
            "per chromatic pigment (one, two, three, or five, without chalk)"
        )
    return list(map(float, value))


def correction_steps(steps, interval):
    """Canonical correction schedule, including a non-divisible final step."""
    if type(steps) is not int or steps < 1 or type(interval) is not int or interval < 0:
        raise ValueError("Invalid mass-budget schedule")
    return sorted({*range(interval, steps + 1, interval), steps}) if interval else []


def pigment_mass(pigment, domain_scale):
    """World-area integral from full, bottom-up material concentrations."""
    pigment = np.asarray(pigment)
    if (
        pigment.ndim != 3
        or min(pigment.shape) < 1
        or not np.isfinite(pigment).all()
        or np.any(pigment < 0)
        or not math.isfinite(domain_scale)
        or domain_scale <= 0
    ):
        raise ValueError("Invalid mass-budget material")
    return pigment.sum(axis=(0, 1), dtype="f8") * (2 * domain_scale / pigment.shape[0]) ** 2


def initial_pool_mass(layout, resolution, domain_scale, *, weights=None):
    """Independently integrate the same sampled pure-pool initial condition."""
    from .layout import pool_profile

    weights = validate_initial_weights(weights, layout["count"])
    width, height = resolution
    aspect = width / height
    x = ((np.arange(width, dtype="f4") + 0.5) / width * 2 * aspect - aspect) * domain_scale
    y = ((np.arange(height, dtype="f4") + 0.5) / height * 2 - 1) * domain_scale
    result = np.zeros(layout["count"] + 1, dtype="f8")
    for pool in layout["pools"]:
        load = pool["load"]
        if weights is not None:
            load *= weights[pool["pigment_index"]]
        squared_x = (x - pool["position"][0]) ** 2
        for start in range(0, height, 256):
            distance = np.sqrt(
                squared_x[None, :] + (y[start : start + 256, None] - pool["position"][1]) ** 2
            )
            sampled = np.asarray(
                load * pool_profile(distance, pool["radius"], pool["edge_width"]),
                dtype="f4",
            )
            result[pool["pigment_index"]] += sampled.sum(dtype="f8")
    return result * (2 * domain_scale / height) ** 2


def validate_report(report, recipe, fields, *, layout):
    """Validate the recorded corrections and independently integrate final paint."""
    simulation = recipe["simulation"]
    interval = simulation.get("mass_budget_interval_steps", 0)
    if interval == 0:
        require(report is None, "Disabled mass restoration cannot advertise a report")
        return
    require(
        simulation.get("initial_pattern") in ("scattered", "engaged")
        and all(
            type(simulation.get(key)) in (int, float) and simulation[key] == 0
            for key in ("deposition", "settling_scale", "underpaint_strength")
        ),
        "Mass budgets require source-free separated pools",
    )
    require(
        type(report) is dict
        and set(report) == {"version", "interval_steps", "initial_mass", "corrections"}
        and report["version"] == VERSION
        and type(report["interval_steps"]) is int
        and report["interval_steps"] == interval,
        "Invalid mass-budget report schema",
    )
    count = recipe["chromatic_count"] + 1

    def vector(value, name):
        require(
            type(value) is list
            and len(value) == count
            and all(type(v) in (int, float) for v in value),
            f"Invalid mass-budget {name}",
        )
        try:
            values = np.asarray(value, dtype="f8")
        except (OverflowError, ValueError) as error:
            raise ValueError(f"Invalid mass-budget {name}") from error
        require(np.isfinite(values).all() and np.all(values >= 0), f"Invalid mass-budget {name}")
        return values

    target = vector(report["initial_mass"], "initial mass")
    require(type(layout) is dict, "Mass restoration needs archived starting pools")
    initial = initial_pool_mass(
        layout,
        simulation["resolution"],
        simulation["domain_scale"],
        weights=simulation.get("initial_pigment_weights"),
    )
    empty = initial == 0
    require(
        np.allclose(initial, target, rtol=RELATIVE_TOLERANCE, atol=1e-12)
        and np.all(target[empty] == 0),
        "Pigment budgets differ from the regenerated starting pools",
    )
    rows = report["corrections"]
    expected_steps = correction_steps(simulation["steps"], interval)
    require(type(rows) is list and len(rows) == len(expected_steps), "Missing mass corrections")
    for row, step in zip(rows, expected_steps, strict=True):
        require(
            type(row) is dict
            and set(row) == {"step", "mass_before", "factors", "mass_after"}
            and type(row["step"]) is int
            and row["step"] == step,
            "Mass correction differs from its canonical schedule",
        )
        before = vector(row["mass_before"], "pre-correction mass")
        factors = vector(row["factors"], "factors")
        after = vector(row["mass_after"], "post-correction mass")
        with np.errstate(over="ignore", under="ignore"):
            applied = factors.astype("f4").astype("f8")
        require(
            np.all(factors > 0)
            and np.array_equal(factors, applied)
            and np.all(before[empty] == 0)
            and np.all(after[empty] == 0)
            and np.all(factors[empty] == 1),
            "Mass factors must be positive float32 values and preserve empty channels",
        )
        require(
            np.allclose(before * factors, target, rtol=RELATIVE_TOLERANCE, atol=1e-12)
            and np.allclose(after, target, rtol=RELATIVE_TOLERANCE, atol=1e-12),
            "Correction does not restore the initial pigment budgets",
        )
    actual = pigment_mass(fields["pigment"], simulation["domain_scale"])
    require(
        np.allclose(actual, target, rtol=RELATIVE_TOLERANCE, atol=1e-12)
        and np.allclose(actual, after, rtol=RELATIVE_TOLERANCE, atol=1e-12)
        and np.all(actual[empty] == 0),
        "Native final paint differs from the certified pigment budgets",
    )
    laminate = simulation.get("material_model", "legacy") == "laminate"
    if laminate:
        require(
            all(simulation.get(key) == 0 for key in ("underpaint_release", "burial_rate")),
            "Laminate mass budgets require disabled legacy pigment phases",
        )
        require(
            not np.any(fields["deposit"])
            and all(
                fields[key].shape == fields["pigment"].shape
                and np.isfinite(fields[key]).all()
                and np.all(fields[key] >= 0)
                for key in ("mobile", "underpaint")
            )
            and np.allclose(
                fields["pigment"],
                fields["mobile"] + fields["underpaint"],
                rtol=RELATIVE_TOLERANCE,
                atol=0,
            ),
            "Laminate pigment budgets must include both moving layers and no deposits",
        )
    else:
        require(
            not np.any(fields["deposit"]) and not np.any(fields["underpaint"]),
            "Source-free mass restoration requires empty stationary pigment phases",
        )
