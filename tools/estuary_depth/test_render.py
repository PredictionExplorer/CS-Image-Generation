"""Geometry, camera, and archive checks without starting a render or importing bpy."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from tools.estuary_depth import render


def faces(loops, starts, totals):
    return [loops[start : start + total] for start, total in zip(starts, totals, strict=True)]


def signed_volume(vertices, polygons):
    volume = 0.0
    vertices = vertices.astype(np.float64)
    for polygon in polygons:
        a = vertices[polygon[0]]
        for index in range(1, len(polygon) - 1):
            b, c = vertices[polygon[index : index + 2]]
            volume += float(np.dot(a, np.cross(b, c))) / 6.0
    return volume


class SolidGeometryTests(unittest.TestCase):
    def test_physical_relief_smoothing_preserves_mass_and_quiets_cell_edges(self):
        for index in ((0, 0), (4, 5), (8, 11)):
            field = np.zeros((9, 12), np.float32)
            field[index] = 1
            result = render.smooth_height(field, 0.012, 4 / 3, 0.75)
            self.assertAlmostEqual(float(result.sum()), 1, places=6)
            self.assertTrue((result >= 0).all())
            self.assertLess(float(result.max()), 1)
            self.assertGreater(np.count_nonzero(result), 1)
        field = np.full((9, 12), 0.3, np.float32)
        np.testing.assert_array_equal(render.smooth_height(field, 0.012, 4 / 3, 0.75), field)
        self.assertIs(render.smooth_height(field, 0.012, 4 / 3, 0), field)

    def test_nonuniform_relief_is_closed_oriented_and_genus_zero(self):
        height = np.random.default_rng(916).uniform(0, 2, (7, 11)).astype(np.float32)
        vertices, loops, starts, totals, uv, top_count = render.solid_arrays(
            height, 0.64, 4 / 3, 0.003, 0.0004, 0.012
        )
        polygons = faces(loops, starts, totals)
        edges = Counter()
        for polygon in polygons:
            for a, b in zip(polygon, np.roll(polygon, -1), strict=True):
                edges[int(a), int(b)] += 1
        self.assertTrue(all(count == 1 and edges[b, a] == 1 for (a, b), count in edges.items()))
        self.assertEqual(len(vertices) - len(edges) // 2 + len(polygons), 2)
        self.assertGreater(signed_volume(vertices, polygons), 0)
        self.assertEqual(top_count, (height.shape[0] - 1) * (height.shape[1] - 1))
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.all((uv >= 0) & (uv <= 1)))
        for polygon in polygons[:top_count]:
            a, b, c = vertices[polygon[:3]]
            self.assertGreater(np.cross(b - a, c - a)[2], 0)
        for polygon in polygons[-2 * (sum(height.shape) - 2) :]:
            a, b, c = vertices[polygon[:3]]
            self.assertLess(np.cross(b - a, c - a)[2], 0)

    def test_constant_relief_has_expected_physical_volume(self):
        height = np.full((9, 13), 0.65, dtype=np.float32)
        width, aspect, relief, base = 0.64, 4 / 3, 0.004, 0.0004
        vertices, loops, starts, totals, _, _ = render.solid_arrays(
            height, width, aspect, relief, base, 0.1
        )
        # Use the actual mesh footprint so a future cell-centred guard mesh
        # remains valid without changing the physical-volume invariant.
        footprint = float(np.ptp(vertices[:, 0])) * float(np.ptp(vertices[:, 1]))
        expected = footprint * (0.65 * relief + base)
        self.assertAlmostEqual(
            signed_volume(vertices, faces(loops, starts, totals)), expected, delta=expected * 2e-5
        )

    def test_minimal_grid_and_zero_relief_are_valid_solids(self):
        vertices, loops, starts, totals, _, top_count = render.solid_arrays(
            np.zeros((2, 2), np.float32), 0.4, 4 / 3, 0, 0.0004, 0
        )
        self.assertEqual(top_count, 1)
        self.assertGreater(signed_volume(vertices, faces(loops, starts, totals)), 0)
        self.assertTrue(np.all(np.diff(starts) == totals[:-1]))
        self.assertEqual(int(starts[-1] + totals[-1]), len(loops))

    def test_translation_preserves_relief_shape(self):
        height = np.arange(15, dtype=np.float32).reshape(3, 5) / 10
        first = render.solid_arrays(height, 0.4, 4 / 3, 0.002, 0.0004, 0)[0]
        second = render.solid_arrays(height, 0.4, 4 / 3, 0.002, 0.0004, 0.031)[0]
        np.testing.assert_allclose(
            second - first, np.broadcast_to([0, 0, 0.031], first.shape), atol=3e-9
        )

    def test_uv_coordinates_correspond_to_world_position(self):
        width, aspect = 0.64, 4 / 3
        vertices, _, _, _, uv, _ = render.solid_arrays(
            np.ones((5, 8), np.float32), width, aspect, 0.002, 0.0004, 0
        )
        np.testing.assert_allclose(uv, vertices[:, :2] / [width, width / aspect] + 0.5, atol=6e-8)

    def test_linear_cell_averages_remain_aligned_with_the_color_field(self):
        h, w = 5, 8
        u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
        # For a linear field, each exact cell average equals its centre value.
        height = (1 + 0.3 * u + 0.7 * v).astype(np.float32)
        vertices, _, _, _, uv, _ = render.solid_arrays(height, 0.64, 4 / 3, 0.002, 0.0004, 0)
        recovered_height = vertices[: h * w, 2] / 0.002
        expected_at_uv = 1 + 0.3 * uv[: h * w, 0] + 0.7 * uv[: h * w, 1]
        np.testing.assert_allclose(recovered_height, expected_at_uv, atol=3e-7)

    def test_invalid_height_fields_are_rejected(self):
        for height in (
            np.zeros((1, 4)),
            np.zeros((0, 4)),
            np.zeros((4,)),
            np.full((2, 2), np.nan),
            np.full((2, 2), np.inf),
            np.full((2, 2), -1),
        ):
            with self.subTest(shape=height.shape), self.assertRaises(ValueError):
                render.solid_arrays(height, 0.4, 4 / 3, 0.002, 0.0004, 0)


class CameraTests(unittest.TestCase):
    def test_camera_frames_are_proper_orthonormal_and_point_at_target(self):
        target = np.array([0.023, -0.041, 0.017])
        for tilt in (0, 6, 20, 35):
            for azimuth in (-360, -90, -30, 0, 45, 180, 360):
                with self.subTest(tilt=tilt, azimuth=azimuth):
                    matrix = render.camera_pose(tilt, azimuth, target)
                    rotation = matrix[:3, :3]
                    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
                    self.assertAlmostEqual(np.linalg.det(rotation), 1)
                    np.testing.assert_allclose(
                        matrix[:3, 3] - target, 0.8 * rotation[:, 2], atol=1e-14
                    )
                    np.testing.assert_array_equal(matrix[3], [0, 0, 0, 1])
                    self.assertGreater(np.dot(rotation[:, 1], [0, 1, 0]), 0)

    def test_overhead_camera_does_not_spin_with_azimuth(self):
        first = render.camera_pose(0, -180, [0, 0, 0])
        for azimuth in (-90, 0, 35, 180):
            np.testing.assert_allclose(render.camera_pose(0, azimuth, [0, 0, 0]), first, atol=1e-15)

    def test_motion_reaches_exact_endpoints_and_static_uses_static_camera(self):
        camera = render.recipe({})["camera"]
        self.assertEqual(
            render.motion_angles(camera, 0, 1), (camera["tilt_degrees"], camera["azimuth_degrees"])
        )
        self.assertEqual(render.motion_angles(camera, 0, 241), tuple(camera["orbit_start"]))
        self.assertEqual(render.motion_angles(camera, 240, 241), tuple(camera["orbit_end"]))
        np.testing.assert_allclose(
            render.motion_angles(camera, 120, 241),
            np.mean([camera["orbit_start"], camera["orbit_end"]], axis=0),
        )
        angles = np.array([render.motion_angles(camera, index, 241) for index in range(241)])
        deltas = np.diff(angles, axis=0)
        self.assertTrue((deltas >= 0).all())
        self.assertLess(deltas[0, 0], deltas[119, 0] / 50)
        self.assertLess(deltas[-1, 0], deltas[119, 0] / 50)


class RecipeTests(unittest.TestCase):
    def test_defaults_and_all_three_families_are_valid(self):
        for family in ("relief", "layered", "hybrid"):
            result = render.recipe({"family": family})
            self.assertEqual(result["family"], family)
            self.assertEqual(result["material"]["ior"], 1.47)

    def test_unknown_nested_keys_and_wrong_types_are_rejected(self):
        for supplied in (
            {"unexpected": 1},
            {"camera": {"roll": 1}},
            {"material": {"gloss": 1}},
            {"render": []},
            {"camera": {"target": [0]}},
            {"name": ""},
            {"family": "volume"},
            {"schema_version": True},
        ):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                render.recipe(supplied)

    def test_numeric_bounds_and_types_are_enforced(self):
        for supplied in (
            {"relief_mm": -0.1},
            {"canvas_width_m": 0},
            {"layer_gap_mm": 0},
            {"base_thickness_mm": 0},
            {"relief_mm": np.nan},
            {"camera": {"tilt_degrees": 35.1}},
            {"camera": {"zoom": True}},
            {"camera": {"orbit_end": [5, 361]}},
            {"lighting": {"key_watts": np.inf}},
            {"render": {"samples": 1.5}},
            {"render": {"seed": -1}},
        ):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                render.recipe(supplied)

    def test_render_budget_and_color_transform_are_enforced(self):
        for render_config in (
            {"resolution": [100, 101]},
            {"resolution": [32, 32]},
            {"resolution": [7680, 7680]},
            {"resolution": [128.0, 96]},
            {"exposure": 5.1},
            {"view_transform": "Filmic"},
            {"view_transform": "Standard", "look": "AgX - Medium High Contrast"},
        ):
            with self.subTest(render_config=render_config), self.assertRaises(ValueError):
                render.recipe({"render": render_config})


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.maps = {
            "history_color_linear": np.full((2, 6, 8, 3), 0.25, np.float32),
            "history_fractions": np.full((2, 6, 8, 3), 1 / 3, np.float32),
            "history_height": np.full((2, 3, 4), 0.2, np.float32),
            "history_times": np.array([0.5, 1.0]),
        }
        self.manifest = {
            "schema_version": 1,
            "complete": True,
            "request": {"schema_version": 1, "test": "verified fixture"},
            "coordinates": {"row_order": "bottom-to-top"},
            "history": {"source_fractions": [0.5, 1.0], "older_history_available": True},
            "view_aspect": 4 / 3,
            "domain_scale": 1.6,
            "source": {"seed": "0xbc53af1cd380"},
        }
        self.manifest["request"].update(
            source=copy.deepcopy(self.manifest["source"]),
            parameters={"resolution": [8, 6], "mesh_resolution": [4, 3]},
            geometry={
                key: copy.deepcopy(self.manifest[key])
                for key in ("domain_scale", "view_aspect", "coordinates")
            },
        )

    def tearDown(self):
        self.temp.cleanup()

    def save(self, maps=None, manifest=None):
        maps = self.maps if maps is None else maps
        manifest = copy.deepcopy(self.manifest if manifest is None else manifest)
        np.savez(self.folder / "bundle.npz", **maps)
        manifest["bundle"] = {"path": "bundle.npz", **render.record(self.folder / "bundle.npz")}
        manifest["identity_sha256"] = hashlib.sha256(
            render.encoded(manifest["request"])
        ).hexdigest()
        render.write(self.folder / "manifest.json", manifest)
        return manifest

    def test_valid_archive_loads_all_requested_history(self):
        self.save()
        for family in ("relief", "layered", "hybrid"):
            manifest, maps = render.load_bundle(self.folder, render.recipe({"family": family}))
            self.assertEqual(manifest["history"]["source_fractions"], [0.5, 1.0])
            np.testing.assert_array_equal(maps["history_height"], self.maps["history_height"])

    def test_payload_tampering_is_rejected_before_array_loading(self):
        self.save()
        with (self.folder / "bundle.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            render.load_bundle(self.folder, render.recipe({}))

    def test_identity_tampering_and_incomplete_archive_are_rejected(self):
        manifest = self.save()
        manifest["request"]["test"] = "changed"
        render.write(self.folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            render.load_bundle(self.folder, render.recipe({}))
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        self.save(manifest=manifest)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            render.load_bundle(self.folder, render.recipe({}))

    def test_history_order_end_and_declared_metadata_must_agree(self):
        for times in ([0.7, 0.6], [0.5, 0.9], [0, 1], [np.nan, 1], [0.7, 1]):
            maps = dict(self.maps, history_times=np.array(times))
            self.save(maps)
            with self.subTest(times=times), self.assertRaises(ValueError):
                render.load_bundle(self.folder, render.recipe({}))

    def test_single_state_cannot_be_relabelled_as_temporal_lamination(self):
        maps = {key: value[-1:] for key, value in self.maps.items()}
        manifest = copy.deepcopy(self.manifest)
        manifest["history"] = {"source_fractions": [1.0], "older_history_available": False}
        self.save(maps, manifest)
        render.load_bundle(self.folder, render.recipe({"family": "relief"}))
        with self.assertRaisesRegex(ValueError, "earlier states"):
            render.load_bundle(self.folder, render.recipe({"family": "layered"}))

    def test_invalid_array_values_shapes_and_precision_are_rejected(self):
        bad_arrays = (
            ("history_height", np.full((2, 3, 4), -1, np.float32)),
            ("history_height", np.full((2, 3, 4), np.inf, np.float32)),
            ("history_height", np.ones((2, 3, 4), np.float64)),
            ("history_color_linear", np.ones((2, 6, 8, 4), np.float32)),
            ("history_color_linear", np.full((2, 6, 8, 3), 1.1, np.float32)),
            ("history_fractions", np.ones((2, 5, 8, 3), np.float32)),
        )
        for key, value in bad_arrays:
            self.save(dict(self.maps, **{key: value}))
            with self.subTest(key=key, shape=value.shape), self.assertRaises(ValueError):
                render.load_bundle(self.folder, render.recipe({}))

    def test_scalar_or_empty_geometric_fields_fail_during_bundle_validation(self):
        for height in (
            np.array(0.2, np.float32),
            np.empty((2, 0, 4), np.float32),
            np.empty((2, 1, 4), np.float32),
        ):
            self.save(dict(self.maps, history_height=height))
            with self.subTest(shape=height.shape), self.assertRaises(ValueError):
                render.load_bundle(self.folder, render.recipe({}))

    def test_row_orientation_and_requested_aspect_must_match(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["coordinates"]["row_order"] = "top-to-bottom"
        self.save(manifest=manifest)
        with self.assertRaisesRegex(ValueError, "orientation"):
            render.load_bundle(self.folder, render.recipe({}))
        self.save()
        with self.assertRaisesRegex(ValueError, "aspect"):
            render.load_bundle(self.folder, render.recipe({"render": {"resolution": [1024, 1024]}}))

    def test_guard_domain_and_map_dimensions_are_bound_to_preparation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["domain_scale"] = 2.0
        self.save(manifest=manifest)
        with self.assertRaisesRegex(ValueError, "Geometry metadata differs"):
            render.load_bundle(self.folder, render.recipe({}))
        maps = dict(self.maps, history_height=np.ones((2, 6, 8), np.float32))
        self.save(maps)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            render.load_bundle(self.folder, render.recipe({}))


if __name__ == "__main__":
    unittest.main()
