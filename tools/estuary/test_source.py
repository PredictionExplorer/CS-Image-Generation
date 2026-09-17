"""Source integrity, fixed geometry and full-recording sampling contracts."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .source import Source


def orbit_points(count: int = 65) -> np.ndarray:
    time = np.linspace(0, 1, count)
    return np.stack(
        [
            np.stack([2 * np.cos(3 * time), np.sin(5 * time), 0.2 * time], axis=1),
            np.stack([np.sin(4 * time), 0.7 * np.cos(6 * time), -0.3 * time], axis=1),
            np.stack([-1 + time, np.sin(2 * time), 0.1 * np.cos(time)], axis=1),
        ],
        axis=1,
    )


def write_orbit(path: Path, points: np.ndarray, **overrides) -> bytes:
    header = {
        "seed": "0x1234",
        "dt": 0.001,
        "masses": [1, 2, 3],
        "count": len(points),
        "provenance": {"integration": "recorded", "steps": len(points)},
    }
    header.update(overrides)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    payload = np.asarray(points, dtype="<f8").tobytes()
    content = b"CSORBIT1" + struct.pack("<Q", len(encoded)) + encoded + payload
    path.write_bytes(content)
    return content


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "original.orbit"
        self.points = orbit_points()
        self.content = write_orbit(self.path, self.points)

    def test_exact_identity_and_source_metadata(self):
        source = Source.read(self.path)
        self.assertEqual(source.sha256, hashlib.sha256(self.content).hexdigest())
        expected = hashlib.sha256(
            struct.pack("<4d", 0.001, 1, 2, 3) + self.points.astype("<f8").tobytes()
        )
        self.assertEqual(source.samples_sha256, expected.hexdigest())
        self.assertEqual(source.samples, len(self.points))
        self.assertEqual(source.duration, 0.064)
        self.assertEqual(source.metadata["source_first_step"], 0)
        self.assertEqual(source.metadata["source_last_step"], 64)
        self.assertEqual(source.provenance, {"integration": "recorded", "steps": 65})
        self.assertIsInstance(source._raw, np.memmap)
        self.assertFalse(source._raw.flags.writeable)
        json.dumps(source.metadata, allow_nan=False)

    def test_first_last_and_every_original_knot_under_one_transform(self):
        source = Source.read(self.path, aspect=4 / 3, rotation_degrees=27)
        p = source.projection
        expected = ((self.points - p["origin"]) / p["extent"] - p["mean"]) @ np.array(p["axes"])
        expected = (expected - p["projected_center"]) * p["scale"]
        observed = source.precompute(len(self.points) - 1).positions
        np.testing.assert_allclose(observed, expected, atol=2e-15)
        np.testing.assert_array_equal(source.frame(0).positions, observed[0])
        np.testing.assert_array_equal(source.frame(1).positions, observed[-1])
        self.assertLessEqual(np.abs(observed[..., 0]).max(), (4 / 3) * 0.78 + 1e-14)
        self.assertLessEqual(np.abs(observed[..., 1]).max(), 0.78 + 1e-14)

    def test_translation_rotation_positive_scale_invariance(self):
        source = Source.read(self.path)
        q, _ = np.linalg.qr(np.array([[1.1, 0.4, 0.7], [0.3, -0.6, 0.8], [-0.8, 0.7, 0.5]]))
        transformed = self.points @ q * 7.3 + np.array([12, -4, 8])
        other = self.root / "transformed.orbit"
        write_orbit(other, transformed)
        transformed_source = Source.read(other)
        a, b = source.precompute(170), transformed_source.precompute(170)
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-12)
        np.testing.assert_allclose(a.velocities, b.velocities, atol=2e-11)
        np.testing.assert_allclose(a.proximity, b.proximity, atol=1e-12)
        np.testing.assert_allclose(a.arc_lengths, b.arc_lengths, atol=1e-12)
        np.testing.assert_allclose(a.pair_distances, b.pair_distances, atol=1e-12)

    def test_pair_distances_retain_depth_when_projected_bodies_overlap(self):
        # Most points span xy, so the two endpoint points along z project onto one
        # location. Their physical separation remains nonzero in the pair feature.
        points = np.tile(
            np.array([[-3.0, -2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 2.0, 0.0]]), (64, 1, 1)
        )
        points[0] = [[0, 0, -0.1], [0, 0, 0.1], [0, 2, 0]]
        write_orbit(self.path, points)
        source = Source.read(self.path)
        first = source.frame(0)
        np.testing.assert_allclose(first.positions[0], first.positions[1], atol=1e-14)
        expected = 0.2 / source.projection["extent"] * source.projection["scale"]
        self.assertAlmostEqual(first.pair_distances[0], expected)
        self.assertGreater(first.pair_distances[0], 0)

    def test_repeated_pca_eigenvalues_use_source_orientation(self):
        cube = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)
        points = np.repeat(cube, 3, axis=0).reshape(8, 3, 3)
        a, b = self.root / "cube.orbit", self.root / "rotated.orbit"
        q, _ = np.linalg.qr(np.array([[1.1, 0.4, 0.7], [0.3, -0.6, 0.8], [-0.8, 0.7, 0.5]]))
        # Separate bodies without changing cube isotropy; each traverses vertices.
        points = np.stack([cube, np.roll(cube, 2, axis=0), np.roll(cube, 4, axis=0)], axis=1)
        write_orbit(a, points)
        write_orbit(b, points @ q)
        np.testing.assert_allclose(
            Source.read(a).precompute(70).positions,
            Source.read(b).precompute(70).positions,
            atol=2e-12,
        )

    def test_linear_derivative_and_endpoint_convention(self):
        source = Source.read(self.path)
        t = 17.5 / 64
        h = 1e-7
        expected = (source.frame(t + h).positions - source.frame(t - h).positions) / (2 * h)
        np.testing.assert_allclose(source.frame(t).velocities, expected, rtol=1e-8, atol=1e-8)
        incoming = (source.frame(1).positions - source.frame(63 / 64).positions) * 64
        np.testing.assert_allclose(source.frame(1).velocities, incoming, atol=1e-13)

    def test_arc_lengths_measure_all_raw_segments(self):
        source = Source.read(self.path)
        knots = source.precompute(64)
        expected = np.vstack(
            [
                np.zeros(3),
                np.cumsum(np.linalg.norm(np.diff(knots.positions, axis=0), axis=2), axis=0),
            ]
        )
        np.testing.assert_allclose(knots.arc_lengths, expected, atol=1e-14)
        fine = source.precompute(300)
        self.assertTrue((np.diff(fine.arc_lengths, axis=0) >= 0).all())
        np.testing.assert_array_equal(fine.arc_lengths[0], np.zeros(3))
        np.testing.assert_array_equal(fine.arc_lengths[-1], knots.arc_lengths[-1])
        half = source.frame(17.5 / 64).arc_lengths
        np.testing.assert_allclose(half, (knots.arc_lengths[17] + knots.arc_lengths[18]) * 0.5)

    def test_rejects_extrapolation_and_invalid_times(self):
        source = Source.read(self.path)
        for value in [-0.001, 1.001, float("nan"), float("inf"), True, "0.5"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                source.frame(value)
        for value in [0, -1, True, 1_000_001]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                source.precompute(value)

    def test_rejects_bad_header_and_payload(self):
        cases = [
            b"no",
            b"CSORBIT1",
            b"CSORBIT1" + struct.pack("<Q", 2**32),
            self.content[:-1],
            self.content + b"x",
        ]
        for index, content in enumerate(cases):
            path = self.root / f"bad-{index}.orbit"
            path.write_bytes(content)
            with self.subTest(index=index), self.assertRaises(ValueError):
                Source.read(path)
        for overrides in [
            {"dt": 0},
            {"dt": True},
            {"masses": [1, 0, 1]},
            {"count": True},
            {"count": 1},
            {"seed": ""},
            {"other": 3},
        ]:
            write_orbit(self.path, self.points, **overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Source.read(self.path)

    def test_rejects_nonfinite_and_degenerate_coordinates(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            points = self.points.copy()
            points[-1, 2, 2] = invalid
            write_orbit(self.path, points)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "non-finite"):
                Source.read(self.path)
        points = np.zeros_like(self.points)
        for collinear in [False, True]:
            if collinear:
                points[..., 0] = self.points[..., 0]
            write_orbit(self.path, points)
            with self.subTest(collinear=collinear), self.assertRaises(ValueError):
                Source.read(self.path)


if __name__ == "__main__":
    unittest.main()
