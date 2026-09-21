"""Read-only Eulerian launch diagnostics; neither causal tracking nor beauty scores.

Retained footprint measures where pigment is found, not which material parcels
stayed there. Even low sampled concentration change cannot prove inactivity:
uniform translating paint, exchange with similar paint, and motion between
samples can be invisible to this measure. Review the labeled motion as well.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np

from .assessment import assess

VERSION = "paint-launch-assessment-v1"
SOURCE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULTS = {
    "version": VERSION,
    "mass_threshold": 0.008,
    "share_threshold": 0.1,
    "footprint_margin": 0.02,
    "quiet_change_threshold": 0.15,
    "minimum_component_fraction": 0.01,
    "minimum_component_area": 0.0005,
}
_BOUNDS = {
    "mass_threshold": (1e-8, 1),
    "share_threshold": (0.001, 0.5),
    "footprint_margin": (0, 0.25),
    "quiet_change_threshold": (0, 2),
    "minimum_component_fraction": (0.0001, 1),
    "minimum_component_area": (0, 1),
}


def validate_config(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Launch assessment requires documented settings")
    result = {**DEFAULTS, **copy.deepcopy(value)}
    if result["version"] != VERSION:
        raise ValueError("Unsupported launch assessment version")
    for name, (low, high) in _BOUNDS.items():
        number = result[name]
        try:
            valid = type(number) in (int, float) and math.isfinite(number) and low <= number <= high
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"Invalid launch assessment {name}")
        result[name] = float(number)
    return result


def _grid(shape, domain):
    h, w = shape
    cell = 2 * domain / h
    x = ((np.arange(w, dtype="f8") + 0.5) / w * 2 - 1) * (w / h) * domain
    y = ((np.arange(h, dtype="f8") + 0.5) / h * 2 - 1) * domain
    visible = (np.abs(y[:, None]) <= 1) & (np.abs(x[None, :]) <= w / h)
    return x, y, visible, cell


def _dilate(mask, radius):
    if not radius:
        return mask.copy()
    summed = np.pad(mask, radius).cumsum(axis=0, dtype="i8").cumsum(axis=1, dtype="i8")
    summed = np.pad(summed, ((1, 0), (1, 0)))
    width = 2 * radius + 1
    return (
        summed[width:, width:]
        - summed[:-width, width:]
        - summed[width:, :-width]
        + summed[:-width, :-width]
    ) > 0


def _components(mask, x, y, cell, settings):
    """Eight-connected components, including cell area in covariance moments.

    Ellipse fill = area / area of same-covariance ellipse; aspect separates a
    disk from an elongated ellipse. Neither needs pixel perimeter.
    """
    parents, runs, previous = [], [], []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for row, values in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.pad(values.astype("i1"), (1, 1))))
        current, start_previous = [], 0
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            index = len(parents)
            parents.append(index)
            current.append((start, end, index))
            while start_previous < len(previous) and previous[start_previous][1] < start:
                start_previous += 1
            for prior in range(start_previous, len(previous)):
                prior_start, prior_end, prior_index = previous[prior]
                if prior_start > end:
                    break
                if prior_end >= start:
                    parents[find(index)] = find(prior_index)
            count = int(end - start)
            cx, cy = (x[start] + x[end - 1]) / 2, y[row]
            runs.append(
                (
                    count,
                    count * cx,
                    count * cy,
                    count * (cx * cx + cell * cell * (count * count - 1) / 12),
                    count * cx * cy,
                    count * cy * cy,
                )
            )
        previous = current
    groups = {}
    for index, moments in enumerate(runs):
        root = find(index)
        if root not in groups:
            groups[root] = np.zeros(6)
        groups[root] += moments
    pixels = int(np.count_nonzero(mask))
    meaningful = []
    for moments in groups.values():
        n, sx, sy, sxx, sxy, syy = moments
        area = n * cell * cell
        if (
            n < pixels * settings["minimum_component_fraction"]
            or area < settings["minimum_component_area"]
        ):
            continue
        cx, cy = sx / n, sy / n
        covariance = np.array(
            [[sxx / n - cx * cx, sxy / n - cx * cy], [sxy / n - cx * cy, syy / n - cy * cy]]
        )
        covariance += np.eye(2) * cell * cell / 12
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), cell * cell / 12)
        meaningful.append(
            {
                "area_world_squared": float(area),
                "area_fraction": float(n / pixels),
                "centroid_world": [float(cx), float(cy)],
                "covariance_aspect": float(np.sqrt(eigenvalues[1] / eigenvalues[0])),
                "covariance_ellipse_fill": float(
                    area / (4 * np.pi * np.sqrt(np.prod(eigenvalues)))
                ),
            }
        )
    meaningful.sort(key=lambda row: row["area_world_squared"], reverse=True)
    return {
        "connectivity": 8,
        "component_count": len(groups),
        "meaningful_components": meaningful,
        "meaningful_area_fraction": sum(row["area_fraction"] for row in meaningful),
        "support_area_world_squared": pixels * cell * cell,
    }


class LaunchAssessment:
    """Fixed-grid total-pigment collector. Native final improves shape/contact;
    footprint and temporal comparisons stay on the source-clock sampling grid.
    """

    def __init__(
        self, initial_pigment, chromatic_count, domain_scale, *, source_metadata=None, config=None
    ):
        self.settings = validate_config({} if config is None else config)
        self.count, self.domain = chromatic_count, domain_scale
        self.initial_metrics = self._assess(initial_pigment)
        self.initial = np.asarray(initial_pigment, dtype="f8").copy()
        self.previous = self.initial.copy()
        self.change = np.zeros_like(self.initial)
        self.source = json.loads(
            json.dumps({} if source_metadata is None else source_metadata, allow_nan=False)
        )
        if type(self.source) is not dict:
            raise ValueError("Launch source metadata must be a JSON object")
        self.samples = [{"source_fraction": 0.0, "participation": self.initial_metrics}]

    def _assess(self, pigment):
        return assess(
            pigment,
            self.count,
            self.domain,
            mass_threshold=self.settings["mass_threshold"],
            share_threshold=self.settings["share_threshold"],
        )

    def sample(self, pigment, source_fraction):
        if (
            type(source_fraction) not in (int, float)
            or not self.samples[-1]["source_fraction"] < source_fraction <= 1
        ):
            raise ValueError("Launch samples must advance the source fraction within (0, 1]")
        metrics = self._assess(pigment)
        current = np.asarray(pigment, dtype="f8")
        if current.shape != self.initial.shape:
            raise ValueError("Launch temporal samples must use one fixed diagnostic grid")
        self.change += np.abs(current - self.previous)
        self.previous = current.copy()
        self.samples.append({"source_fraction": float(source_fraction), "participation": metrics})

    def report(self, final_native=None):
        if self.samples[-1]["source_fraction"] != 1:
            raise ValueError("Launch report requires the completed source fraction 1")
        final = self.previous if final_native is None else np.asarray(final_native)
        final_metrics = self._assess(final)
        h, w = self.initial.shape[:2]
        if final.shape[0] * w != final.shape[1] * h:
            raise ValueError("Native final aspect must match the diagnostic grid")
        _, _, visible, cell = _grid((h, w), self.domain)
        x, y, native_visible, native_cell = _grid(final.shape[:2], self.domain)
        total, initial_total = (
            final[..., : self.count].sum(-1, dtype="f8"),
            self.initial[..., : self.count].sum(-1),
        )
        radius = math.floor(self.settings["footprint_margin"] / cell)
        painted = native_visible & (total > self.settings["mass_threshold"])
        pigments = []
        for index in range(self.count):
            start, current = self.initial[..., index], self.previous[..., index]
            support = (initial_total > self.settings["mass_threshold"]) & (
                start >= initial_total * self.settings["share_threshold"]
            )
            near = _dilate(support, radius) & visible
            quiet = self.change[..., index] <= self.settings["quiet_change_threshold"] * np.maximum(
                start, self.settings["mass_threshold"]
            )
            denominator = current[visible].sum()
            initial_visible_mass = start[visible].sum()
            initial_row, final_row = (
                self.initial_metrics["pigments"][index],
                final_metrics["pigments"][index],
            )
            a, b = initial_row["moments"], final_row["moments"]
            drift = np.linalg.norm(np.subtract(b["centroid"], a["centroid"])) if a and b else None
            spread = b["rms_radius"] / a["rms_radius"] if a and b and a["rms_radius"] else None
            retained = current[near].sum() / denominator if denominator else 0.0
            quiet_retained = current[near & quiet].sum() / denominator if denominator else 0.0
            change = self.change[..., index][visible].sum()
            change_ratio = change / initial_visible_mass if initial_visible_mass else None
            mask = painted & (final[..., index] >= total * self.settings["share_threshold"])
            pigments.append(
                {
                    "pigment_index": index,
                    "initial_mass": initial_row["mass"],
                    "final_mass": final_row["mass"],
                    "initial_visible_mass_fraction": initial_row["visible_mass_fraction"],
                    "final_visible_mass_fraction": final_row["visible_mass_fraction"],
                    "final_displayed_mass_fraction": final_row["displayed_mass_fraction"],
                    "final_contact_mass_fraction": final_row["contact_mass_fraction"],
                    "maximum_sampled_contact_mass_fraction": max(
                        sample["participation"]["pigments"][index]["contact_mass_fraction"]
                        for sample in self.samples
                    ),
                    "retained_initial_footprint_fraction": float(retained),
                    "retained_low_change_footprint_fraction": float(quiet_retained),
                    "sampled_cumulative_change_per_initial_visible_mass": change_ratio,
                    "visible_centroid_displacement_world": drift,
                    "visible_rms_radius_ratio": spread,
                    "final_components": _components(mask, x, y, native_cell, self.settings),
                }
            )
        return {
            "version": VERSION,
            "settings": copy.deepcopy(self.settings),
            "source": copy.deepcopy(self.source),
            "source_clock": {
                "initial": 0.0,
                "final": 1.0,
                "sample_fractions": [row["source_fraction"] for row in self.samples],
            },
            "sample_grid": [w, h],
            "final_grid": [final.shape[1], final.shape[0]],
            "domain_scale": self.domain,
            "actual_footprint_margin_world": radius * cell,
            "footprint_scope": "Final sampled visible mass near initial support; square margin.",
            "temporal_scope": "Sum |delta density| / max(initial density, threshold); Eulerian.",
            "limitation": "Uniform flow or missed motion can look unchanged; inspect films.",
            "phase_scope": "Total pigment: mobile + deposit + underpaint; not phase-specific.",
            "shape_scope": "Thresholded support covariance; no pixel perimeter or beauty score.",
            "samples": copy.deepcopy(self.samples),
            "final_participation": final_metrics,
            "pigments": pigments,
            "final_silhouette": _components(painted, x, y, native_cell, self.settings),
        }
