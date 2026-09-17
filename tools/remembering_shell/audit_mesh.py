#!/usr/bin/env python3
"""Bounded geometric self-intersection audit of the canonical binary PLY.

Uses a double-precision BVH and streams unique overlapping-box pairs. Adaptive
orientation predicates fall back to exact rational arithmetic on the supplied
binary64 coordinates; possible noncoplanar crossings use exact plane intervals.
Only contact confined to a shared topological vertex/edge is permitted. Shared
neighbors are tested, not discarded. The tool changes no mesh or renderer state.
"""

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

EPSILON = sys.float_info.epsilon
DEFAULT_MAX_PAIRS = 10_000_000
MAX_PAIRS = 50_000_000


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def sign(value):
    return (value > 0) - (value < 0)


@lru_cache(maxsize=8192)
def exact_point(point):
    return tuple(Fraction(value) for value in point)


@lru_cache(maxsize=8192)
def exact_plane(points):
    a, b, c = (exact_point(point) for point in points)
    return a, cross(sub(b, a), sub(c, a))


def exact_plane_value(plane, point):
    origin, normal = plane
    return dot(normal, sub(exact_point(point), origin))


def orient3d(points, point):
    a, b, c = points
    u, v, w = sub(b, a), sub(c, a), sub(point, a)
    value = dot(cross(u, v), w)
    permanent = sum(
        abs(w[k]) * (abs(u[(k + 1) % 3] * v[(k + 2) % 3]) + abs(u[(k + 2) % 3] * v[(k + 1) % 3]))
        for k in range(3)
    )
    error_bound = 64 * EPSILON * permanent
    if (
        math.isfinite(value)
        and math.isfinite(error_bound)
        and error_bound > 0.0
        and abs(value) > error_bound
    ):
        return sign(value)
    return sign(exact_plane_value(exact_plane(points), point))


def orient2d(a, b, c):
    u, v = sub(b, a), sub(c, a)
    left, right = u[0] * v[1], u[1] * v[0]
    value = left - right
    permanent = abs(left) + abs(right)
    error_bound = 64 * EPSILON * permanent
    if (
        math.isfinite(value)
        and math.isfinite(error_bound)
        and error_bound > 0.0
        and abs(value) > error_bound
    ):
        return sign(value)
    a, b, c = exact_point(a), exact_point(b), exact_point(c)
    return sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


class Triangle:
    __slots__ = ("bounds", "indices", "normal", "points")

    def __init__(self, indices, vertices):
        self.indices = tuple(indices)
        self.points = tuple(vertices[i] for i in indices)
        self.bounds = tuple(min(p[k] for p in self.points) for k in range(3)) + tuple(
            max(p[k] for p in self.points) for k in range(3)
        )
        self.normal = cross(
            sub(self.points[1], self.points[0]), sub(self.points[2], self.points[0])
        )

    def degenerate(self):
        if len(set(self.indices)) != 3:
            return True
        a, b = sub(self.points[1], self.points[0]), sub(self.points[2], self.points[0])
        scale = max(abs(value) for value in a + b)
        if (
            any(not math.isfinite(v) for v in self.normal)
            or math.hypot(*self.normal) <= 128 * EPSILON * scale * scale
        ):
            return all(value == 0 for value in exact_plane(self.points)[1])
        return False


def overlaps(a, b):
    return all(a[k] <= b[k + 3] and b[k] <= a[k + 3] for k in range(3))


class Node:
    __slots__ = ("bounds", "indices", "left", "right", "size")

    def __init__(self, indices, triangles):
        self.size = len(indices)
        self.bounds = tuple(min(triangles[i].bounds[k] for i in indices) for k in range(3)) + tuple(
            max(triangles[i].bounds[k + 3] for i in indices) for k in range(3)
        )
        self.left = self.right = None
        self.indices = None
        if len(indices) <= 8:
            self.indices = tuple(indices)
        else:
            axis = max(range(3), key=lambda k: self.bounds[k + 3] - self.bounds[k])
            indices.sort(
                key=lambda i: (
                    triangles[i].bounds[axis] * 0.5 + triangles[i].bounds[axis + 3] * 0.5,
                    i,
                )
            )
            half = len(indices) // 2
            self.left = Node(indices[:half], triangles)
            self.right = Node(indices[half:], triangles)


def candidate_pairs(triangles):
    """Stream each candidate once; never create a potentially quadratic pair list."""
    root = Node(list(range(len(triangles))), triangles)
    stack = [(root, root)]
    while stack:
        a, b = stack.pop()
        if not overlaps(a.bounds, b.bounds):
            continue
        if a is b:
            if a.indices is not None:
                for i, j in itertools.combinations(a.indices, 2):
                    if overlaps(triangles[i].bounds, triangles[j].bounds):
                        yield min(i, j), max(i, j)
            else:
                stack.extend([(a.right, a.right), (a.left, a.right), (a.left, a.left)])
        elif a.indices is not None and b.indices is not None:
            for i in a.indices:
                for j in b.indices:
                    if overlaps(triangles[i].bounds, triangles[j].bounds):
                        yield min(i, j), max(i, j)
        elif b.indices is not None or (a.indices is None and a.size >= b.size):
            stack.extend([(a.right, b), (a.left, b)])
        else:
            stack.extend([(a, b.right), (a, b.left)])


def projection_axis(triangle):
    normal = exact_plane(triangle.points)[1]
    return max(range(3), key=lambda k: abs(normal[k]))


def projected(points, dropped):
    return tuple(tuple(value for k, value in enumerate(point) if k != dropped) for point in points)


def on_segment(a, b, p):
    return orient2d(a, b, p) == 0 and all(
        min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in range(2)
    )


def inside_triangle(point, triangle):
    signs = [orient2d(triangle[k], triangle[(k + 1) % 3], point) for k in range(3)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def coplanar_relation(a, b, shared):
    axis = projection_axis(a)
    pa, pb = projected(a.points, axis), projected(b.points, axis)
    if orient2d(*pa) == 0 or orient2d(*pb) == 0:
        return "ambiguous"
    if len(shared) == 2:
        edge = [pa[a.indices.index(index)] for index in shared]
        ua = pa[next(i for i, index in enumerate(a.indices) if index not in shared)]
        ub = pb[next(i for i, index in enumerate(b.indices) if index not in shared)]
        return "legal_contact" if orient2d(*edge, ua) * orient2d(*edge, ub) < 0 else "intersection"
    permitted = pa[a.indices.index(shared[0])] if shared else None
    found = False
    for points, other in [(pa, pb), (pb, pa)]:
        for point in points:
            if inside_triangle(point, other):
                if point != permitted:
                    return "intersection"
                found = True
    for i in range(3):
        x, y = pa[i], pa[(i + 1) % 3]
        for j in range(3):
            u, v = pb[j], pb[(j + 1) % 3]
            first = orient2d(x, y, u), orient2d(x, y, v)
            second = orient2d(u, v, x), orient2d(u, v, y)
            if first[0] * first[1] < 0 and second[0] * second[1] < 0:
                return "intersection"
            for p, start, end in [(x, u, v), (y, u, v), (u, x, y), (v, x, y)]:
                if on_segment(start, end, p):
                    if p != permitted:
                        return "intersection"
                    found = True
    return "legal_contact" if found else "separate"


def plane_interval(points, plane, axis):
    values = [exact_plane_value(plane, p) for p in points]
    coordinates = [exact_point(p)[axis] for p in points]
    intersections = [coordinates[i] for i, value in enumerate(values) if value == 0]
    for i in range(3):
        j = (i + 1) % 3
        if sign(values[i]) * sign(values[j]) < 0:
            intersections.append(
                (coordinates[i] * values[j] - coordinates[j] * values[i]) / (values[j] - values[i])
            )
    return (min(intersections), max(intersections)) if intersections else None


def pair_relation(a, b):
    """Test actual contact, allowing only each pair's shared topological feature."""
    shared = sorted(set(a.indices).intersection(b.indices))
    if len(shared) == 3:
        return "intersection"
    if len(shared) == 2:
        unique_b = b.points[next(i for i, index in enumerate(b.indices) if index not in shared)]
        # Two distinct planes through the same edge intersect only on that line.
        if orient3d(a.points, unique_b) != 0:
            return "legal_contact"
        return coplanar_relation(a, b, shared)
    signs_b = [
        0 if index in shared else orient3d(a.points, point)
        for index, point in zip(b.indices, b.points, strict=True)
    ]
    if all(s > 0 for s in signs_b) or all(s < 0 for s in signs_b):
        return "separate"
    if all(s == 0 for s in signs_b):
        return coplanar_relation(a, b, shared)
    signs_a = [
        0 if index in shared else orient3d(b.points, point)
        for index, point in zip(a.indices, a.points, strict=True)
    ]
    if all(s > 0 for s in signs_a) or all(s < 0 for s in signs_a):
        return "separate"
    if shared:
        for signs in (signs_a, signs_b):
            nonzero = [s for s in signs if s != 0]
            if len(nonzero) == 2 and nonzero[0] == nonzero[1]:
                return "legal_contact"
    # Exact intervals avoid guessing whether an almost-zero overlap extends
    # beyond a shared vertex. This path is limited to potential plane crossings.
    plane_a, plane_b = exact_plane(a.points), exact_plane(b.points)
    direction = cross(plane_a[1], plane_b[1])
    if all(value == 0 for value in direction):
        return "ambiguous"
    axis = max(range(3), key=lambda k: abs(direction[k]))
    ia = plane_interval(a.points, plane_b, axis)
    ib = plane_interval(b.points, plane_a, axis)
    if ia is None or ib is None:
        return "separate"
    low, high = max(ia[0], ib[0]), min(ia[1], ib[1])
    if low > high:
        return "separate"
    if shared and low == high == exact_point(a.points[a.indices.index(shared[0])])[axis]:
        return "legal_contact"
    return "intersection"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_mesh(path, maximum_triangles=2_000_000, native_f32=False):
    with path.open("rb") as stream:
        lines = []
        while stream.tell() < 8192:
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete PLY header")
            lines.append(line.decode("ascii").strip())
            if line == b"end_header\n":
                break
        else:
            raise ValueError("PLY header exceeds8192bytes")
        if lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Require canonical binary-little-endian PLY")
        vertices = [int(line.split()[-1]) for line in lines if line.startswith("element vertex ")]
        faces = [int(line.split()[-1]) for line in lines if line.startswith("element face ")]
        properties = [line for line in lines if line.startswith("property ")]
        expected = [f"property double {name}" for name in ("x", "y", "z", "nx", "ny", "nz")]
        expected.append("property list uchar uint vertex_indices")
        if len(vertices) != 1 or len(faces) != 1 or properties != expected:
            raise ValueError(
                "Unexpected PLY layout; require f64XYZ/normals and triangular u32faces"
            )
        nv, nf = vertices[0], faces[0]
        if not 3 <= nv <= maximum_triangles * 3 or not 1 <= nf <= maximum_triangles:
            raise ValueError("PLY vertex or face count exceeds the bounded audit")
        if path.stat().st_size != stream.tell() + nv * 48 + nf * 13:
            raise ValueError("PLY length differs from its declared canonical layout")
        points = []
        for _ in range(nv):
            point = struct.unpack("<6d", stream.read(48))[:3]
            if native_f32:
                point = struct.unpack("<3f", struct.pack("<3f", *point))
            if not all(math.isfinite(value) for value in point):
                raise ValueError("PLY positions are not finite in the requested precision")
            points.append(point)
        triangles = []
        for _ in range(nf):
            length, *indices = struct.unpack("<B3I", stream.read(13))
            if length != 3 or any(index >= nv for index in indices):
                raise ValueError("Invalid PLY triangle indices")
            triangles.append(Triangle(indices, points))
    return points, triangles


def audit(triangles, maximum_pairs=DEFAULT_MAX_PAIRS, examples=16):
    if not 1 <= maximum_pairs <= MAX_PAIRS:
        raise ValueError(f"Candidate-pair cap must be between 1 and {MAX_PAIRS}")
    result = {
        "triangle_count": len(triangles),
        "candidate_pairs": 0,
        "tested_pairs": 0,
        "intersections": 0,
        "ambiguous": 0,
        "legal_contacts": 0,
        "complete": False,
        "passed": False,
        "examples": [],
    }
    for index, triangle in enumerate(triangles):
        if triangle.degenerate():
            result["ambiguous"] += 1
            if len(result["examples"]) < examples:
                result["examples"].append(
                    {"triangle": index, "reason": "degenerate input triangle"}
                )
    if result["ambiguous"]:
        result["reason"] = "invalid input geometry; pair search not performed"
        return result
    try:
        for i, j in candidate_pairs(triangles):
            result["candidate_pairs"] += 1
            if result["candidate_pairs"] > maximum_pairs:
                result["reason"] = "candidate-pair limit exceeded; result is incomplete"
                return result
            relation = pair_relation(triangles[i], triangles[j])
            result["tested_pairs"] += 1
            if relation == "legal_contact":
                result["legal_contacts"] += 1
            elif relation in ("intersection", "ambiguous"):
                result["intersections" if relation == "intersection" else "ambiguous"] += 1
                if len(result["examples"]) < examples:
                    result["examples"].append(
                        {
                            "triangles": [i, j],
                            "relation": relation,
                            "shared_vertices": sorted(
                                set(triangles[i].indices).intersection(triangles[j].indices)
                            ),
                            "positions": [triangles[i].points, triangles[j].points],
                        }
                    )
    except KeyboardInterrupt:
        result["reason"] = "interrupted; result is incomplete"
        return result
    result["complete"] = True
    result["passed"] = result["intersections"] == 0 and result["ambiguous"] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--max-triangles", type=int, default=2_000_000)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--native-f32", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    if args.output.exists():
        parser.error("Audit output already exists; use a new report path")
    if not 1 <= args.max_triangles <= 2_000_000 or not 0 <= args.examples <= 128:
        parser.error("Triangle cap must be1..2000000 and examples0..128")
    started = time.monotonic()
    mesh = args.mesh.resolve(strict=True)
    identity = digest(mesh)
    report = {
        "schema_version": 1,
        "mesh": str(mesh),
        "mesh_sha256": identity,
        "script_sha256": digest(Path(__file__)),
        "coordinate_precision": "native-f32-promoted-to-f64"
        if args.native_f32
        else "canonical-f64",
        "method": "streamed double BVH; adaptive exact orientation and rational plane intervals",
        "maximum_candidate_pairs": args.max_pairs,
    }
    try:
        _, triangles = read_mesh(mesh, args.max_triangles, args.native_f32)
        if digest(mesh) != identity:
            raise ValueError("Mesh changed during loading")
        report.update(audit(triangles, args.max_pairs, args.examples))
        if digest(mesh) != identity:
            raise ValueError("Mesh changed during the audit")
    except (OSError, ValueError, ArithmeticError, struct.error) as error:
        report.update(complete=False, passed=False, error=str(error))
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "passed",
                    "complete",
                    "triangle_count",
                    "candidate_pairs",
                    "intersections",
                    "ambiguous",
                    "elapsed_seconds",
                )
            }
        )
    )
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
