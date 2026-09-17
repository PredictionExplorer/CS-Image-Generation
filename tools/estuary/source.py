"""Immutable, full-recording projection of the original three-body source.

The recording is never integrated again or fitted per frame. Every source knot
undergoes the same translation, PCA projection and uniform scale. Interpolation
is linear between original knots, so sampling cannot overshoot the fitted bounds.
PCA axis signs and repeated-eigenvalue directions come from ordered source points,
not from world coordinate axes. A source that does not span a plane is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"CSORBIT1"
MAX_HEADER_BYTES = 1 << 20
MAX_SAMPLES = 10_000_000
MAX_PRECOMPUTE = 1_000_001
CHUNK_SAMPLES = 32_768
PAIRS = np.array([[0, 1], [1, 2], [2, 0]])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, lower: float, upper: float) -> float:
    _require(type(value) in (float, int), f"{name} must be a number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"Invalid {name}") from exc
    _require(math.isfinite(value) and lower <= value <= upper, f"Invalid {name}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def _chunks(values: np.ndarray):
    for start in range(0, len(values), CHUNK_SAMPLES):
        yield values[start : start + CHUNK_SAMPLES]


def _project(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # These are tiny 3-column transforms. Explicit contraction avoids platform
    # BLAS startup/thread overhead and spurious floating-point flags on Accelerate.
    return np.einsum("...i,ij->...j", values, matrix, optimize=False)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    # Access time can legitimately change while hashing or reading the mapping.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@dataclass(frozen=True)
class SourceFrame:
    """Positions, source-fraction derivatives, and 3D encounter measurements.

    ``positions`` and ``velocities`` have shape ``(..., 3, 2)``. ``proximity``
    has shape ``(..., 3)`` in body order. Proximity measures physical 3D separation,
    so overlapping projected trajectories do not create false encounters.
    """

    positions: np.ndarray
    velocities: np.ndarray
    proximity: np.ndarray
    arc_lengths: np.ndarray
    pair_distances: np.ndarray


@dataclass(frozen=True)
class Source:
    path: Path
    seed: str
    sha256: str
    samples_sha256: str
    samples: int
    dt: float
    masses: tuple[float, float, float]
    provenance: Any
    aspect: float
    fill: float
    rotation_degrees: float
    _raw: np.ndarray
    _origin: np.ndarray
    _extent: float
    _mean: np.ndarray
    _axes: np.ndarray
    _projected_center: np.ndarray
    _scale: float
    _distance_scale: float
    _eigenvalues: np.ndarray
    _bounds: np.ndarray
    _arc_lengths: np.ndarray

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        aspect: float = 1.25,
        fill: float = 0.78,
        rotation_degrees: float = 0.0,
    ) -> Source:
        """Validate every sample, bind its exact bytes and prepare one fixed plane.

        ``fill`` is the fraction of the canvas extent occupied by source points.
        The read-only mapping retains raw f64 coordinates; preparation uses bounded
        chunks. Whole-file and sample-only hashes are compatible with Rust caches.
        Files must remain immutable for the lifetime of this object.
        """
        aspect = _number(aspect, "aspect", 0.2, 5.0)
        fill = _number(fill, "fill", 0.05, 0.98)
        rotation_degrees = _number(rotation_degrees, "rotation_degrees", -360, 360)
        path = Path(path).resolve(strict=True)
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            _require(stream.read(8) == MAGIC, "Unrecognized orbit cache type or version")
            size_bytes = stream.read(8)
            _require(len(size_bytes) == 8, "Truncated orbit header length")
            length = struct.unpack("<Q", size_bytes)[0]
            _require(0 < length <= MAX_HEADER_BYTES, "Orbit header exceeds size limit")
            header_bytes = stream.read(length)
            _require(len(header_bytes) == length, "Truncated orbit header")
            header = json.loads(
                header_bytes, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
            )
            _require(type(header) is dict, "Orbit header must be an object")
            _require(
                set(header) == {"seed", "dt", "masses", "count", "provenance"},
                "Unexpected orbit header fields",
            )
            count = header["count"]
            _require(type(count) is int and 2 <= count <= MAX_SAMPLES, "Invalid orbit sample count")
            _require(info.st_size == 16 + length + count * 72, "Orbit payload size differs")
            seed = header["seed"]
            _require(type(seed) is str and 1 <= len(seed) <= 256, "Invalid orbit seed")
            dt = _number(header["dt"], "source timestep", 1e-300, 1e300)
            _require(
                type(header["masses"]) is list and len(header["masses"]) == 3, "Invalid masses"
            )
            masses = tuple(_number(m, "body mass", 1e-300, 1e300) for m in header["masses"])
            _require(math.isfinite(dt * (count - 1)), "Source duration exceeds finite range")
            digest = hashlib.sha256(MAGIC + size_bytes + header_bytes)
            samples_digest = hashlib.sha256(struct.pack("<4d", dt, *masses))
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
                samples_digest.update(block)
            _require(_identity(path.stat()) == _identity(info), "Orbit changed while reading")

        raw = np.memmap(path, mode="r", dtype="<f8", offset=16 + length, shape=(count, 3, 3))
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for block in _chunks(raw):
            _require(bool(np.isfinite(block).all()), "Orbit contains non-finite coordinates")
            low = np.minimum(low, block.min(axis=(0, 1)))
            high = np.maximum(high, block.max(axis=(0, 1)))
        origin = low * 0.5 + high * 0.5
        with np.errstate(over="ignore", invalid="ignore"):
            extent = float((high - low).max())
        _require(
            math.isfinite(extent) and extent > 1e-150, "Orbit extent is degenerate or excessive"
        )

        # Normalize before covariance to keep source units from degrading conditioning.
        mean = sum(((block - origin) / extent).sum(axis=(0, 1)) for block in _chunks(raw))
        mean /= count * 3
        covariance = np.zeros((3, 3))
        for block in _chunks(raw):
            centered = ((block - origin) / extent - mean).reshape(-1, 3)
            covariance += np.einsum("ni,nj->ij", centered, centered, optimize=False)
        covariance /= count * 3
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
        _require(eigenvalues[1] > eigenvalues[0] * 1e-12, "Orbit does not span a stable plane")
        axes: list[np.ndarray] = []
        for axis_index in range(2):
            tied = np.abs(eigenvalues - eigenvalues[axis_index]) <= eigenvalues[0] * 1e-10
            projector = eigenvectors[:, tied] @ eigenvectors[:, tied].T
            for previous in axes:
                contained = projector @ previous
                projector -= np.outer(contained, contained)
            best = None
            maximum = -1.0
            for block in _chunks(raw):
                candidates = _project(((block - origin) / extent - mean).reshape(-1, 3), projector)
                lengths = np.einsum("ij,ij->i", candidates, candidates)
                local_maximum = float(lengths.max())
                # Stable first anchor among numerical ties, in time then body order.
                if local_maximum > maximum * (1 + 1e-12):
                    index = int(np.flatnonzero(lengths >= local_maximum * (1 - 1e-12))[0])
                    best, maximum = candidates[index], local_maximum
            _require(best is not None and maximum > 1e-20, "PCA orientation is degenerate")
            for previous in axes:
                best -= previous * np.dot(previous, best)
            axes.append(best / np.linalg.norm(best))
        basis = np.stack(axes, axis=1)
        angle = math.radians(rotation_degrees)
        basis = basis @ np.array(
            [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
        )
        projected_low, projected_high = np.full(2, np.inf), np.full(2, -np.inf)
        pair_sum = 0.0
        for block in _chunks(raw):
            centered = (block - origin) / extent - mean
            points = _project(centered, basis)
            projected_low = np.minimum(projected_low, points.min(axis=(0, 1)))
            projected_high = np.maximum(projected_high, points.max(axis=(0, 1)))
            edges = centered[:, PAIRS[:, 0]] - centered[:, PAIRS[:, 1]]
            pair_sum += float(np.linalg.norm(edges, axis=2).sum())
        center = (projected_low + projected_high) * 0.5
        half_extent = (projected_high - projected_low) * 0.5
        scale = fill / max(half_extent[0] / aspect, half_extent[1])
        _require(math.isfinite(scale), "Projection scale exceeds finite range")
        distance_scale = pair_sum / (3 * count) * 0.3
        _require(distance_scale > 1e-15, "Body separations are degenerate")
        # Original-knot polyline lengths, not chords between display frames. These
        # measurements allow the renderer to limit source travel per solver step.
        arc_lengths = np.zeros((count, 3), dtype=np.float64)
        previous = None
        accumulated = np.zeros(3)
        for start in range(0, count, CHUNK_SAMPLES):
            block = raw[start : start + CHUNK_SAMPLES]
            points = (_project((block - origin) / extent - mean, basis) - center) * scale
            differences = np.diff(
                points, axis=0, prepend=points[:1] if previous is None else previous
            )
            lengths = np.linalg.norm(differences, axis=2)
            arc_lengths[start : start + len(block)] = np.cumsum(lengths, axis=0) + accumulated
            accumulated = arc_lengths[start + len(block) - 1]
            previous = points[-1:]
        arc_lengths.flags.writeable = False
        _require(
            _identity(path.stat()) == _identity(info), "Orbit changed while preparing projection"
        )
        return cls(
            path,
            seed,
            digest.hexdigest(),
            samples_digest.hexdigest(),
            count,
            dt,
            masses,
            header["provenance"],
            aspect,
            fill,
            rotation_degrees,
            raw,
            origin,
            extent,
            mean,
            basis,
            center,
            scale,
            distance_scale,
            eigenvalues,
            np.stack([(projected_low - center) * scale, (projected_high - center) * scale]),
            arc_lengths,
        )

    @property
    def duration(self) -> float:
        return self.dt * (self.samples - 1)

    @property
    def projection(self) -> dict[str, Any]:
        """JSON-ready transform; positions equal the equation recorded here."""
        return {
            "method": "full-source-pca-v1",
            "interpolation": "piecewise-linear-original-knots",
            "orientation": "ordered-maximal-source-anchors-within-eigenspaces",
            "equation": "((((p-origin)/extent-mean)@axes)-projected_center)*scale",
            "origin": self._origin.tolist(),
            "extent": self._extent,
            "mean": self._mean.tolist(),
            "axes": self._axes.tolist(),
            "projected_center": self._projected_center.tolist(),
            "scale": self._scale,
            "pca_eigenvalues": self._eigenvalues.tolist(),
            "bounds": self._bounds.tolist(),
            "aspect": self.aspect,
            "fill": self.fill,
            "rotation_degrees": self.rotation_degrees,
            "proximity_distance_scale_normalized_3d": self._distance_scale,
            "projected_arc_lengths": self._arc_lengths[-1].tolist(),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "sha256": self.sha256,
            "samples_sha256": self.samples_sha256,
            "samples": self.samples,
            "dt": self.dt,
            "duration": self.duration,
            "masses": list(self.masses),
            "provenance": self.provenance,
            "source_first_step": 0,
            "source_last_step": self.samples - 1,
            "projection": self.projection,
        }

    def sample(self, fractions: np.ndarray) -> SourceFrame:
        """Vectorized evaluation, rejecting extrapolation and preserving endpoint knots.

        At an interior knot the derivative is the outgoing segment derivative.
        At the final knot it is the incoming derivative. Units are distance per
        source fraction, not per simulation second or video second.
        """
        fractions = np.asarray(fractions, dtype=np.float64)
        _require(
            fractions.ndim == 1 and 1 <= len(fractions) <= MAX_PRECOMPUTE, "Invalid time array"
        )
        _require(
            bool(
                np.isfinite(fractions).all() and (fractions >= 0).all() and (fractions <= 1).all()
            ),
            "Source fractions must be finite and inside [0, 1]",
        )
        fractional_index = fractions * (self.samples - 1)
        left = np.minimum(fractional_index.astype(np.int64), self.samples - 2)
        weight = (fractional_index - left)[:, None, None]
        first = (self._raw[left] - self._origin) / self._extent - self._mean
        second = (self._raw[left + 1] - self._origin) / self._extent - self._mean
        positions_3d = first * (1 - weight) + second * weight
        positions = (_project(positions_3d, self._axes) - self._projected_center) * self._scale
        velocities = _project(second - first, self._axes) * (self._scale * (self.samples - 1))
        edges = positions_3d[:, PAIRS[:, 0]] - positions_3d[:, PAIRS[:, 1]]
        raw_distances = np.linalg.norm(edges, axis=2)
        distances = raw_distances / self._distance_scale
        pair_closeness = 1 / (1 + distances**4)
        proximity = np.stack(
            [
                np.maximum(pair_closeness[:, 0], pair_closeness[:, 2]),
                np.maximum(pair_closeness[:, 0], pair_closeness[:, 1]),
                np.maximum(pair_closeness[:, 1], pair_closeness[:, 2]),
            ],
            axis=1,
        )
        arc_weight = weight[:, :, 0]
        arc_lengths = (
            self._arc_lengths[left] * (1 - arc_weight) + self._arc_lengths[left + 1] * arc_weight
        )
        return SourceFrame(
            positions, velocities, proximity, arc_lengths, raw_distances * self._scale
        )

    def frame(self, fraction: float) -> SourceFrame:
        fraction = _number(fraction, "source fraction", 0.0, 1.0)
        sampled = self.sample(np.array([fraction]))
        return SourceFrame(
            sampled.positions[0],
            sampled.velocities[0],
            sampled.proximity[0],
            sampled.arc_lengths[0],
            sampled.pair_distances[0],
        )

    def precompute(self, steps: int) -> SourceFrame:
        """Evaluate ``steps+1`` uniformly spaced source fractions, including both ends."""
        _require(type(steps) is int and 1 <= steps < MAX_PRECOMPUTE, "Invalid source step count")
        return self.sample(np.linspace(0, 1, steps + 1))
