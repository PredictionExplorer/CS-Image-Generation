"""Behavioral checks for deterministic, bounded three-pigment spatial motifs."""

from __future__ import annotations

import copy
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from . import initial_patterns as patterns

SEED = "0x" + "0123456789abcdef" * 4
INITIAL = np.array([[-0.61, -0.11], [0.58, 0.06], [0.03, 0.37]])
TIME = np.linspace(0, 2 * np.pi, 65)
SUPPORT = np.stack(
    [
        np.stack((0.71 * np.cos(TIME + phase), 0.43 * np.sin(TIME + phase)), axis=-1)
        for phase in (0, 2.1, 4.2)
    ],
    axis=1,
)


def controls(pattern="lacuna-banks", seed=SEED):
    return {"version": patterns.VERSION, "pattern": pattern, "seed": seed}


def raster(pattern="lacuna-banks", *, seed=SEED, size=(321, 243), tile_rows=128, support=SUPPORT):
    width, height = size
    x = ((np.arange(width) + 0.5) / width * 2 * 4 / 3 - 4 / 3) * 1.6
    y = ((np.arange(height) + 0.5) / height * 2 - 1) * 1.6
    state = np.full((height, width, 4), np.nan, dtype=np.float32)
    patterns.fill_pattern(
        state,
        x,
        y,
        INITIAL,
        controls(pattern, seed),
        support_positions=support,
        tile_rows=tile_rows,
    )
    return state, x, y


class StartingPatternTests(unittest.TestCase):
    def test_none_is_omitted_and_does_not_touch_legacy_state(self):
        self.assertIsNone(patterns.validate_controls(None))
        state = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        original = state.copy()
        patterns.fill_pattern(state, None, None, None, None)
        np.testing.assert_array_equal(state, original)

    def test_strict_versioned_controls_use_the_full_seed(self):
        value = controls()
        normalized = patterns.validate_controls(value)
        self.assertEqual(normalized, value)
        self.assertIsNot(normalized, value)
        for invalid in (
            {},
            False,
            [],
            {**value, "unknown": 1},
            {**value, "version": "future"},
            {**value, "pattern": "strata"},
            {**value, "pattern": []},
            {**value, "seed": SEED.upper()},
            {**value, "seed": SEED[:-1]},
            {**value, "seed": "f" * 64},
            {**value, "seed": True},
            {**value, "seed": SEED + "\n"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                patterns.validate_controls(invalid)

    def test_each_pattern_partitions_paint_and_reaches_the_central_corridor(self):
        for pattern in patterns.PATTERNS:
            with self.subTest(pattern=pattern):
                state, x, y = raster(pattern)
                self.assertEqual(state.dtype, np.float32)
                self.assertTrue(np.isfinite(state).all())
                self.assertGreaterEqual(float(state.min()), 0)
                self.assertLessEqual(float(state.max()), 1)
                self.assertFalse(state[..., 3].any())
                np.testing.assert_allclose(state[..., :3].sum(-1), 1, atol=1.2e-7, rtol=0)
                stats = patterns.coverage_stats(
                    state, x, y, INITIAL, controls(pattern), support_positions=SUPPORT
                )
                for region in ("visible", "active_corridor"):
                    shares = stats[region]["pigment_fractions"]
                    # All three pigments own substantial central regions; this is
                    # a spatial participation test, not a motion or beauty score.
                    self.assertGreater(min(shares), 0.06 if region == "active_corridor" else 0.005)
                    self.assertLess(max(shares), 0.96)
                    self.assertAlmostEqual(sum(shares), 1, places=7)

    def test_irregular_tiles_are_byte_exact_and_never_allocate_full_geometry(self):
        expected, _, _ = raster("braided-ribbons", tile_rows=128)
        for rows in (1, 7, 64, 243, 512):
            actual, _, _ = raster("braided-ribbons", tile_rows=rows)
            np.testing.assert_array_equal(actual, expected)
        original = patterns._masks
        observed = []

        def record(*args):
            observed.append(args[1].shape)
            return original(*args)

        with patch.object(patterns, "_masks", side_effect=record):
            state, _, _ = raster("branching-channels", size=(6144, 97), tile_rows=32)
        self.assertEqual(observed, [(32, 6144), (32, 6144), (32, 6144), (1, 6144)])
        self.assertTrue(np.isfinite(state).all())

    def test_all_seed_bits_matter_without_ambient_rng_or_input_mutation(self):
        before_initial, before_support = INITIAL.copy(), SUPPORT.copy()
        for pattern in patterns.PATTERNS:
            value = controls(pattern)
            before = copy.deepcopy(value)
            first, x, y = raster(pattern)
            with patch.object(np.random, "default_rng", side_effect=AssertionError("ambient RNG")):
                again = np.empty_like(first)
                patterns.fill_pattern(again, x, y, INITIAL, value, support_positions=SUPPORT)
            np.testing.assert_array_equal(first, again)
            self.assertEqual(value, before)
            for seed in ("0xf" + SEED[3:], SEED[:-1] + "0"):
                changed, _, _ = raster(pattern, seed=seed)
                self.assertGreater(float(np.mean(np.abs(first - changed))), 0.002)
        np.testing.assert_array_equal(INITIAL, before_initial)
        np.testing.assert_array_equal(SUPPORT, before_support)

    def test_motifs_are_not_recolorings_or_aliases(self):
        # Coarse dominant-pigment maps test spatial arrangement, even allowing an
        # adversarial permutation of pigment identities. No aesthetic ranking.
        maps = {
            name: np.argmax(raster(name, size=(160, 120))[0][..., :3], axis=-1)
            for name in patterns.PATTERNS
        }
        for first, second in itertools.combinations(maps, 2):
            mismatch = min(
                np.mean(maps[first] != np.asarray(order)[maps[second]])
                for order in itertools.permutations(range(3))
            )
            with self.subTest(first=first, second=second):
                self.assertGreater(mismatch, 0.025)

    def test_source_translation_moves_geometry_without_changing_it(self):
        baseline, x, y = raster("river-confluence")
        offset = np.array([0.23, -0.19])
        translated = np.empty_like(baseline)
        patterns.fill_pattern(
            translated,
            x + offset[0],
            y + offset[1],
            INITIAL + offset,
            controls("river-confluence"),
            support_positions=SUPPORT + offset,
        )
        np.testing.assert_allclose(translated, baseline, atol=1e-6, rtol=0)
        moved, _, _ = raster("river-confluence", support=SUPPORT + offset)
        self.assertGreater(float(np.mean(np.abs(moved - baseline))), 0.01)

    def test_source_rotation_rotates_each_motif_and_preserves_amount(self):
        coordinates = (np.arange(193) + 0.5) / 193 * 2.6 - 1.3
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        for pattern in patterns.PATTERNS:
            first = np.empty((193, 193, 4), dtype=np.float32)
            second = np.empty_like(first)
            patterns.fill_pattern(
                first,
                coordinates,
                coordinates,
                INITIAL,
                controls(pattern),
                support_positions=SUPPORT,
            )
            patterns.fill_pattern(
                second,
                coordinates,
                coordinates,
                INITIAL @ rotation.T,
                controls(pattern),
                support_positions=SUPPORT @ rotation.T,
            )
            with self.subTest(pattern=pattern):
                np.testing.assert_allclose(second, np.rot90(first, -1), atol=1e-6, rtol=0)

    def test_resolution_does_not_invent_or_remove_pigment_area(self):
        for pattern in patterns.PATTERNS:
            shares = []
            for size in ((320, 240), (640, 480)):
                state, x, y = raster(pattern, size=size)
                stats = patterns.coverage_stats(
                    state, x, y, INITIAL, controls(pattern), support_positions=SUPPORT
                )
                shares.append(stats["visible"]["pigment_fractions"])
            with self.subTest(pattern=pattern):
                np.testing.assert_allclose(shares[0], shares[1], atol=0.002, rtol=0)

    def test_lacuna_has_two_enclosed_dark_openings(self):
        # A topology claim on one canonical fixture, not a universal beauty gate.
        from collections import deque

        state, _, _ = raster("lacuna-banks", size=(160, 120))
        remaining = state[..., 0] > 0.8
        height, width = remaining.shape
        enclosed = 0
        while remaining.any():
            start = tuple(np.argwhere(remaining)[0])
            queue = deque([start])
            remaining[start] = False
            boundary = False
            area = 0
            while queue:
                row, column = queue.popleft()
                area += 1
                boundary |= row in (0, height - 1) or column in (0, width - 1)
                for nr, nc in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if 0 <= nr < height and 0 <= nc < width and remaining[nr, nc]:
                        remaining[nr, nc] = False
                        queue.append((nr, nc))
            enclosed += int(not boundary and area > 10)
        self.assertEqual(enclosed, 2)

    def test_stationary_sources_and_optional_support_are_finite(self):
        for support in (None, np.zeros((1, 3, 2))):
            state, _, _ = raster("asymmetric-rosette", support=support)
            self.assertTrue(np.isfinite(state).all())
        state, x, y = raster()
        patterns.fill_pattern(state, x, y, np.zeros((3, 2)), controls())
        self.assertTrue(np.isfinite(state).all())

    def test_invalid_arrays_and_support_fail_before_state_is_written(self):
        state, x, y = raster()
        baseline = state.copy()
        invalid = (
            {"x": x[::-1]},
            {"y": y[:-1]},
            {"initial_positions": np.zeros((2, 2))},
            {"support_positions": np.zeros((0, 3, 2))},
            {"support_positions": np.full((2, 3, 2), np.nan)},
            {"tile_rows": True},
        )
        for change in invalid:
            args = {
                "state": state,
                "x": x,
                "y": y,
                "initial_positions": INITIAL,
                "controls": controls(),
                "support_positions": SUPPORT,
                **change,
            }
            with self.subTest(change=change.keys()), self.assertRaises(ValueError):
                patterns.fill_pattern(**args)
            np.testing.assert_array_equal(state, baseline)
        for wrong in (state.astype(np.float64), state[..., :3]):
            with self.assertRaises(ValueError):
                patterns.fill_pattern(wrong, x, y, INITIAL, controls())


if __name__ == "__main__":
    unittest.main()
