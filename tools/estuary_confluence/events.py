"""Plan sparse blooms from real three-dimensional close approaches.

The event schedule is prepared from the entire immutable trajectory on a fixed
sampling lattice and refined around actual distance minima. Rendering cadence
and solver resolution never enter this calculation. A trajectory with no
distinct close approaches yields no events, rather than invented timed pulses.
"""

from __future__ import annotations

import math

import numpy as np

from tools.estuary.source import PAIRS

VERSION = "confluence-encounters-v1"
SAMPLE_COUNT = 4097
REFRACTORY_FRACTION = 0.12
MIN_CLOSENESS = 0.35
MIN_PROMINENCE = 0.08


def plan_events(source, count: int = 3) -> list[dict]:
    """Return up to ``count`` meaningful encounters in chronological order.

    Position and radius use the source's fixed projected coordinate system.
    ``duration`` is a Gaussian standard deviation in complete-source fractions.
    Strength is the dimensionless 3D closeness at the selected distance minimum.
    Selection suppresses nearby competing pairs as one compositional event.
    """
    if type(count) is not int or not 0 <= count <= 12:
        raise ValueError("Event count must be an integer in [0, 12]")
    if count == 0:
        return []
    fractions = np.linspace(0, 1, SAMPLE_COUNT)
    sample = source.sample(fractions)
    distance = np.asarray(sample.pair_distances, dtype=np.float64)
    positions = np.asarray(sample.positions, dtype=np.float64)
    projection = source.projection
    scale = float(projection["proximity_distance_scale_normalized_3d"]) * float(projection["scale"])
    if (
        distance.shape != (SAMPLE_COUNT, 3)
        or positions.shape != (SAMPLE_COUNT, 3, 2)
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
        or not np.isfinite(positions).all()
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid source encounter measurements")
    closeness = 1 / (1 + np.minimum(distance / scale, 1e50) ** 4)
    shoulder = int(0.035 * (SAMPLE_COUNT - 1))
    candidates = []
    for pair in range(3):
        curve = closeness[:, pair]
        peaks = np.flatnonzero((curve[1:-1] > curve[:-2]) & (curve[1:-1] >= curve[2:])) + 1
        for index in peaks:
            # Both shoulders must fall: a constant orbit or monotonic approach
            # cannot become a bloom merely because it remains close for a while.
            prominence = float(
                curve[index]
                - max(
                    curve[max(0, index - shoulder) : index].min(),
                    curve[index + 1 : min(SAMPLE_COUNT, index + shoulder + 1)].min(),
                )
            )
            if curve[index] < MIN_CLOSENESS or prominence < MIN_PROMINENCE:
                continue
            candidates.append(
                (float(curve[index]) * math.sqrt(prominence), index, pair, prominence)
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = []
    for _, index, pair, prominence in candidates:
        if any(abs(fractions[index] - event["fraction"]) < REFRACTORY_FRACTION for event in chosen):
            continue
        refinement = np.linspace(fractions[index - 1], fractions[index + 1], 65)
        fine = source.sample(refinement)
        fine_distances = np.asarray(fine.pair_distances, dtype=np.float64)
        fine_positions = np.asarray(fine.positions, dtype=np.float64)
        if (
            fine_distances.shape != (65, 3)
            or fine_positions.shape != (65, 3, 2)
            or not np.isfinite(fine_distances).all()
            or np.any(fine_distances < 0)
            or not np.isfinite(fine_positions).all()
        ):
            raise ValueError("Invalid refined source encounter measurements")
        minimum = int(np.argmin(fine_distances[:, pair]))
        fraction = float(refinement[minimum])
        if any(abs(fraction - event["fraction"]) < REFRACTORY_FRACTION for event in chosen):
            continue
        distance_at_peak = float(fine_distances[minimum, pair])
        strength = float(1 / (1 + (min(distance_at_peak / scale, 1e50)) ** 4))
        half_height = closeness[index, pair] - prominence / 2
        left, right = index, index
        while left > 0 and closeness[left, pair] > half_height:
            left -= 1
        while right < SAMPLE_COUNT - 1 and closeness[right, pair] > half_height:
            right += 1
        # FWHM to sigma; only the material response is bounded, not encounter time.
        duration = float(np.clip((fractions[right] - fractions[left]) / 2.355, 0.003, 0.025))
        bodies = PAIRS[pair]
        position = fine_positions[minimum, bodies].mean(axis=0)
        chosen.append(
            {
                "version": VERSION,
                "fraction": fraction,
                "position": position.tolist(),
                "pair": bodies.tolist(),
                "strength": strength,
                "duration": duration,
                "radius": 0.055 + 0.045 * strength,
                "pair_distance": distance_at_peak,
                "prominence": prominence,
            }
        )
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda event: event["fraction"])
