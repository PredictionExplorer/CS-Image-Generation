"""Geometric audit regressions, including legal and illegal adjacent contact."""

import importlib.util
import itertools
import random
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "shell_audit", Path(__file__).with_name("audit_mesh.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def relation(self, vertices, a, b):
        return audit.pair_relation(audit.Triangle(a, vertices), audit.Triangle(b, vertices))

    def test_underflowed_predicate_bounds_require_exact_arithmetic(self):
        scale_2d = 1e-160
        self.assertGreater(scale_2d * scale_2d, 0.0)
        self.assertEqual(64 * audit.EPSILON * (scale_2d * scale_2d), 0.0)
        with mock.patch.object(audit, "exact_point", wraps=audit.exact_point) as exact:
            self.assertEqual(audit.orient2d((0.0, 0.0), (scale_2d, 0.0), (0.0, scale_2d)), 1)
            self.assertEqual(exact.call_count, 3)

        scale_3d = 1e-107
        self.assertGreater(scale_3d * scale_3d * scale_3d, 0.0)
        self.assertEqual(64 * audit.EPSILON * (scale_3d * scale_3d * scale_3d), 0.0)
        points = ((0.0, 0.0, 0.0), (scale_3d, 0.0, 0.0), (0.0, scale_3d, 0.0))
        with mock.patch.object(audit, "exact_plane", wraps=audit.exact_plane) as exact:
            self.assertEqual(audit.orient3d(points, (0.0, 0.0, scale_3d)), 1)
            exact.assert_called_once_with(points)

    def test_shared_edge_is_legal_only_without_coplanar_overlap(self):
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0.2, 0.4, 0), (0, 0, 1)]
        self.assertEqual(self.relation(points, (0, 1, 2), (1, 0, 3)), "legal_contact")
        self.assertEqual(self.relation(points, (0, 1, 2), (0, 1, 4)), "intersection")
        self.assertEqual(self.relation(points, (0, 1, 2), (0, 1, 5)), "legal_contact")

    def test_shared_vertex_does_not_hide_a_crossing_beyond_that_vertex(self):
        points = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (-1, -1, -1),
            (-1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0.2, 0.2, 0),
            (-0.2, 0.2, 0),
        ]
        self.assertEqual(self.relation(points, (0, 1, 2), (0, 3, 4)), "legal_contact")
        self.assertEqual(self.relation(points, (0, 1, 2), (0, 5, 6)), "intersection")
        self.assertEqual(self.relation(points, (0, 1, 2), (0, 7, 8)), "legal_contact")
        self.assertEqual(self.relation(points, (0, 1, 2), (0, 9, 10)), "intersection")

    def test_nonadjacent_crossings_coplanar_overlap_and_unwelded_touch_fail(self):
        points = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0.2, 0.2, -1),
            (0.2, 0.2, 1),
            (0.8, 0.2, 0),
            (0.2, 0.2, 0),
            (0.4, 0.2, 0),
            (0.2, 0.4, 0),
            (1, 0, 0),
            (2, 0, 0),
            (1, -1, 0),
        ]
        for triangle in [(3, 4, 5), (6, 7, 8), (9, 10, 11)]:
            self.assertEqual(self.relation(points, (0, 1, 2), triangle), "intersection")

    def test_exact_predicates_resolve_tiny_parallel_clearance(self):
        epsilon = 2.0**-48
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, epsilon),
            (1.0, 0.0, epsilon),
            (0.0, 1.0, epsilon),
        ]
        self.assertEqual(self.relation(points, (0, 1, 2), (3, 4, 5)), "separate")
        points[3] = (0.25, 0.25, -epsilon)
        points[4] = (0.25, 0.25, epsilon)
        points[5] = (0.6, 0.25, 0.0)
        self.assertEqual(self.relation(points, (0, 1, 2), (3, 4, 5)), "intersection")

    def test_streamed_bvh_matches_brute_box_pairs_without_duplicates(self):
        random_source = random.Random(42)
        points = [tuple(random_source.uniform(-2, 2) for _ in range(3)) for _ in range(90)]
        triangles = [audit.Triangle((i, i + 1, i + 2), points) for i in range(0, 90, 3)]
        actual = list(audit.candidate_pairs(triangles))
        expected = {
            (i, j)
            for i, j in itertools.combinations(range(30), 2)
            if audit.overlaps(triangles[i].bounds, triangles[j].bounds)
        }
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)

    def test_pair_cap_and_degenerate_input_never_report_a_pass(self):
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        triangles = [audit.Triangle((0, 1, 2), points) for _ in range(12)]
        result = audit.audit(triangles, maximum_pairs=3)
        self.assertFalse(result["complete"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["tested_pairs"], 3)
        self.assertEqual(result["candidate_pairs"], 4)
        invalid = audit.audit([audit.Triangle((0, 0, 2), points)])
        self.assertEqual(invalid["ambiguous"], 1)
        self.assertFalse(invalid["passed"])

    def test_closed_tetrahedron_permits_only_its_shared_topological_contacts(self):
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        result = audit.audit([audit.Triangle(face, points) for face in faces])
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["legal_contacts"], 6)

    def test_canonical_ply_loading_and_native_precision_are_explicit(self):
        points = [(1.0, 1.0, 1.0), (1.0 + 1e-9, 1.0, 1.0), (1.0, 2.0, 1.0)]
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
            + "".join(f"property double {name}\n" for name in ("x", "y", "z", "nx", "ny", "nz"))
            + "element face 1\nproperty list uchar uint vertex_indices\nend_header\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.ply"
            path.write_bytes(
                header.encode()
                + b"".join(struct.pack("<6d", *p, 0, 0, 1) for p in points)
                + struct.pack("<B3I", 3, 0, 1, 2)
            )
            vertices, triangles = audit.read_mesh(path)
            self.assertEqual(vertices, points)
            self.assertFalse(triangles[0].degenerate())
            _, native = audit.read_mesh(path, native_f32=True)
            self.assertTrue(native[0].degenerate())
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                audit.read_mesh(path)


if __name__ == "__main__":
    unittest.main()
