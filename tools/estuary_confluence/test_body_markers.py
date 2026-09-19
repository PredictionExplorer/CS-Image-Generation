"""Forcing-plane marker projection and non-destructive diagnostic rasterization."""

from __future__ import annotations

import unittest

import numpy as np

from tools.estuary_studio.surface import camera_basis

from .body_markers import VERSION, annotate, project_positions, validate_config


class BodyMarkerContracts(unittest.TestCase):
    def test_configuration_is_versioned_opt_in_and_copied(self):
        self.assertIsNone(validate_config())
        self.assertIsNone(validate_config(False))
        self.assertEqual(validate_config(True), validate_config({}))
        config = validate_config(True)
        self.assertEqual(config["version"], VERSION)
        self.assertEqual(config["size_px_1080"], 36)
        self.assertEqual(validate_config(config), config)
        changed = validate_config(config)
        changed["labels"] = False
        self.assertTrue(config["labels"])
        for invalid in (
            0,
            1,
            [],
            "yes",
            {"enabled": True},
            {"version": "future"},
            {"labels": 1},
            {"size_px_1080": True},
            {"size_px_1080": 0},
            {"stroke_px_1080": float("nan")},
            {"size_px_1080": 12, "stroke_px_1080": 8},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_frontal_projection_uses_pixel_edges_and_flips_vertical_axis(self):
        points = np.array([[0, 0], [4 / 3, 1], [-4 / 3, -1]], dtype="f8")
        expected = np.array([[720, 540], [1440, 0], [0, 1080]], dtype="f8")
        for azimuth in (-90, 0, 35, 180):
            np.testing.assert_allclose(
                project_positions(points, (1440, 1080), azimuth_degrees=azimuth),
                expected,
                atol=1e-12,
            )
        points = np.array([[1, 0], [0, 1], [0, -1]])
        np.testing.assert_array_equal(
            project_positions(points, (1080, 1080)), [[1080, 540], [540, 0], [540, 1080]]
        )

    def test_tilt_foreshortens_correct_axis_without_image_plane_rotation(self):
        points = np.array([[1, 0], [0, 1], [-1, -1]], dtype="f8")
        cosine = np.sqrt(3) / 2
        horizontal = project_positions(points, (1440, 1080), tilt_degrees=30, azimuth_degrees=0)
        np.testing.assert_allclose(
            horizontal,
            [[720 + 540 * cosine, 540], [720, 0], [720 - 540 * cosine, 1080]],
            atol=1e-12,
        )
        vertical = project_positions(points, (1440, 1080), tilt_degrees=30, azimuth_degrees=90)
        np.testing.assert_allclose(
            vertical,
            [[1260, 540], [720, 540 - 540 * cosine], [180, 540 + 540 * cosine]],
            atol=1e-12,
        )

    def test_projection_inverts_the_renderers_orthographic_z_zero_intersection(self):
        points = np.array([[0.7, -0.4], [-0.2, 0.3], [0, 0]], dtype="f8")
        basis = camera_basis(31, 43)
        centers = project_positions(points, (1440, 1080), tilt_degrees=31, azimuth_degrees=43)
        visible = np.array([0.4, 0.3])
        screen = np.column_stack((centers[:, 0] / 1440 - 0.5, 0.5 - centers[:, 1] / 1080)) * visible
        origin = screen[:, 0, None] * basis[:, 0] + screen[:, 1, None] * basis[:, 1]
        hit = origin - origin[:, 2, None] / basis[2, 2] * basis[:, 2]
        np.testing.assert_allclose(hit[:, :2] / (0.3 / 2), points, atol=3e-16)
        np.testing.assert_allclose(hit[:, 2], 0, atol=1e-16)

    def test_output_size_scales_positions_without_using_guard_domain(self):
        points = np.array([[0.7, -0.4], [-0.2, 0.3], [0, 0]], dtype="f8")
        small = project_positions(points, (1440, 1080), tilt_degrees=20, azimuth_degrees=35)
        large = project_positions(points, (2880, 2160), tilt_degrees=20, azimuth_degrees=35)
        np.testing.assert_array_equal(large, small * 2)

    def test_invalid_positions_camera_and_output_are_rejected(self):
        points = np.zeros((3, 2))
        for size in ((0, 1080), (True, 1080), (1440.0, 1080), (1440,), (12288, 12288)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                project_positions(points, size)
        for bad in (
            np.zeros((2, 2)),
            np.ones((3, 2), dtype=bool),
            np.ones((3, 2), dtype=complex),
            [["x", "y"]] * 3,
            np.full((3, 2), float("nan")),
            np.full((3, 2), 1e308),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                project_positions(bad, (1440, 1080))
        with self.assertRaises(ValueError):
            project_positions(points, (1440, 1080), tilt_degrees=43)


class BodyMarkerRasterTests(unittest.TestCase):
    def test_disabled_is_an_independent_exact_copy_without_source_validation(self):
        pixels = np.array([[[0.2, 0.3, float("nan")]]], dtype="f4")
        for config in (None, False):
            result = annotate(pixels, None, tilt_degrees="unused", config=config)
            self.assertFalse(np.shares_memory(pixels, result))
            np.testing.assert_array_equal(result.view("u4"), pixels.view("u4"))

    def test_three_exact_centers_are_red_and_only_local_pixels_change(self):
        pixels = np.full((1080, 1440, 3), 0.37, dtype="f4")
        positions = np.array([[-0.8, -0.4], [0.1, 0.2], [0.8, -0.1]], dtype="f8")
        original_positions = positions.copy()
        result = annotate(pixels, positions, config=True)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, pixels.shape)
        np.testing.assert_array_equal(pixels, np.full_like(pixels, 0.37))
        np.testing.assert_array_equal(positions, original_positions)
        allowed = np.zeros(pixels.shape[:2], dtype=bool)
        for x, y in project_positions(positions, (1440, 1080)):
            ix, iy = int(x), int(y)
            np.testing.assert_array_equal(result[iy, ix], [1, 0, 0])
            allowed[iy - 60 : iy + 60, ix - 60 : ix + 70] = True
        np.testing.assert_array_equal(result[~allowed], pixels[~allowed])
        self.assertGreater(np.count_nonzero(result != pixels), 100)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

    def test_halo_is_visible_on_light_dark_and_red_ground(self):
        positions = np.array([[0, 0], [5, 5], [-5, -5]], dtype="f8")
        for color in ((0, 0, 0), (1, 1, 1), (1, 0, 0), (1, 0.3, 0.5)):
            pixels = np.empty((1080, 1440, 3), dtype="f4")
            pixels[:] = color
            result = annotate(pixels, positions, config={"labels": False})
            crop = result[515:565, 695:745]
            self.assertTrue(np.any(np.all(crop == (1, 0, 0), axis=-1)))
            self.assertTrue(np.any(np.all(crop == (1, 1, 1), axis=-1)))
            self.assertLess(float(np.min(crop @ [0.2126, 0.7152, 0.0722])), 0.2)

    def test_clipped_guides_do_not_wrap_or_move_onto_the_canvas(self):
        pixels = np.full((1080, 1440, 3), 0.37, dtype="f4")
        positions = np.array([[-4 / 3, 0], [10, 0], [0, 10]], dtype="f8")
        result = annotate(pixels, positions, config={"labels": False})
        self.assertTrue(np.any(result[515:565, :24] != pixels[515:565, :24]))
        np.testing.assert_array_equal(result[:, 25:], pixels[:, 25:])
        outside = np.full((3, 2), 1e8)
        np.testing.assert_array_equal(annotate(pixels, outside, config=True), pixels)

    def test_labels_are_optional_and_rendering_is_deterministic(self):
        pixels = np.full((540, 720, 3), 0.3, dtype="f4")
        positions = np.array([[-0.8, -0.4], [0, 0], [0.8, 0.4]], dtype="f8")
        marked = annotate(pixels, positions, config=True)
        np.testing.assert_array_equal(marked, annotate(pixels, positions, config=True))
        no_labels = annotate(pixels, positions, config={"labels": False})
        self.assertGreater(np.count_nonzero(marked != no_labels), 0)
        np.testing.assert_array_equal(pixels, np.full_like(pixels, 0.3))

    def test_coincident_bodies_keep_one_exact_center_and_three_label_quadrants(self):
        pixels = np.full((540, 720, 3), 0.3, dtype="f4")
        positions = np.zeros((3, 2))
        labeled = annotate(pixels, positions, config=True)
        bare = annotate(pixels, positions, config={"labels": False})
        np.testing.assert_array_equal(labeled[270, 360], [1, 0, 0])
        for y, x in (
            (slice(254, 270), slice(370, 384)),
            (slice(254, 270), slice(336, 350)),
            (slice(270, 286), slice(370, 384)),
        ):
            self.assertTrue(np.any(labeled[y, x] != bare[y, x]))
        np.testing.assert_array_equal(
            project_positions(positions, (720, 540)), np.tile([360, 270], (3, 1))
        )

    def test_invalid_pixels_are_rejected_when_enabled(self):
        positions = np.zeros((3, 2))
        for pixels in (
            np.zeros((10, 10, 3), dtype="f8"),
            np.zeros((10, 10), dtype="f4"),
            np.zeros((10, 10, 4), dtype="f4"),
            np.zeros((1, 1, 3), dtype="f4"),
            np.full((10, 10, 3), -0.1, dtype="f4"),
            np.full((10, 10, 3), 1.1, dtype="f4"),
            np.full((10, 10, 3), float("nan"), dtype="f4"),
        ):
            with self.subTest(shape=pixels.shape), self.assertRaises(ValueError):
                annotate(pixels, positions, config=True)


if __name__ == "__main__":
    unittest.main()
