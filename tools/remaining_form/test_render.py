"""Input/archive contract checks without importing or rendering with Blender."""

import copy
import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "remaining_render", Path(__file__).with_name("render.py")
)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


class RenderContractTests(unittest.TestCase):
    def test_thin_film_defaults_preserve_uncoated_clay_and_porcelain(self):
        config = copy.deepcopy(renderer.DEFAULTS)
        self.assertEqual(config["material"]["thin_film_nm"], 0.0)
        self.assertEqual(config["material"]["thin_film_ior"], 1.35)
        renderer.validate(config)
        porcelain = renderer.merge_config(
            config,
            {"material": {"kind": "porcelain", "subsurface_radius_mm": 0.55}},
        )
        renderer.validate(porcelain)
        porcelain["material"]["thin_film_nm"] = 280.0
        renderer.validate(porcelain)
        porcelain["material"].update(thin_film_nm=1500.0, thin_film_ior=2.5)
        renderer.validate(porcelain)

    def test_thin_film_rejects_nonfinite_out_of_range_and_nonporcelain_settings(self):
        porcelain = renderer.merge_config(
            renderer.DEFAULTS,
            {"material": {"kind": "porcelain", "subsurface_radius_mm": 0.55}},
        )
        for name, values in [
            ("thin_film_nm", [-0.01, 1500.01, math.nan, math.inf, True]),
            ("thin_film_ior", [1.0, 2.51, math.nan, math.inf, False]),
        ]:
            for value in values:
                invalid = copy.deepcopy(porcelain)
                invalid["material"][name] = value
                with self.assertRaises(ValueError):
                    renderer.validate(invalid)
        clay = copy.deepcopy(renderer.DEFAULTS)
        clay["material"]["thin_film_nm"] = 280.0
        with self.assertRaisesRegex(ValueError, "porcelain sculpture"):
            renderer.validate(clay)

    def shell_fixture(self, directory):
        nv, nl, rows = 32, 4, 3
        ring = 2 * nv + 2 * nl
        vertices = rows * ring + 2 * (nv + 1)
        triangles = 2 * (rows - 1) * ring + 8 * nv + 4 * nl
        mesh = directory / "mesh.ply"
        mesh.write_text(
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {vertices}\nelement face {triangles}\nend_header\n"
        )
        record = {
            "identity": {
                "schema_version": 1,
                "recipe": {
                    "transverse_segments": nv,
                    "lip_segments": nl,
                    "longitudinal_segments": 64,
                },
            },
            "mesh_sha256": renderer.digest(mesh),
            "vertices": vertices,
            "triangles": triangles,
            "normal_count": vertices,
            "shell": {"rows": rows, "ring_vertices": ring},
        }
        renderer.write_json(directory / "build.json", record)
        return mesh, record

    def test_shell_material_weights_follow_both_walls_rounded_lips_and_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh, _ = self.shell_fixture(Path(directory))
            metadata = renderer.shell_material_metadata(mesh, renderer.digest(mesh))
            layout = metadata["layout"]
            weights = renderer.shell_interior_weights(layout)
            nv, nl, nr = (
                layout["transverse_segments"],
                layout["lip_segments"],
                layout["ring_vertices"],
            )
            self.assertEqual(len(weights), layout["vertices"])
            self.assertTrue(all(0.0 <= value <= 1.0 for value in weights))
            self.assertEqual(list(weights[: nv + 1]), [0.0] * (nv + 1))
            self.assertEqual(list(weights[nv + nl : 2 * nv + nl + 1]), [1.0] * (nv + 1))
            for j in range(1, nl):
                self.assertAlmostEqual(weights[nv + j], 0.5 - 0.5 * math.cos(math.pi * j / nl))
                self.assertAlmostEqual(weights[nv + j] + weights[2 * nv + nl + j], 1.0)
            for row in range(1, layout["rows"]):
                self.assertEqual(weights[row * nr : (row + 1) * nr], weights[:nr])
            self.assertEqual(list(weights[layout["rows"] * nr :]), [0.5] * (2 * (nv + 1)))

    def test_shell_layout_is_bound_to_mesh_and_raw_build_receipt_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mesh, record = self.shell_fixture(directory)
            metadata = renderer.shell_material_metadata(mesh, renderer.digest(mesh))
            self.assertEqual(
                metadata["build_receipt_sha256"], renderer.digest(directory / "build.json")
            )
            with self.assertRaisesRegex(ValueError, "mesh SHA256"):
                renderer.shell_material_metadata(mesh, "0" * 64)
            record["total_seconds"] = 0.25
            renderer.write_json(directory / "build.json", record)
            updated = renderer.shell_material_metadata(mesh, renderer.digest(mesh))
            self.assertNotEqual(metadata["build_receipt_sha256"], updated["build_receipt_sha256"])
            self.assertEqual(metadata["layout"], updated["layout"])

    def test_inconsistent_shell_layout_and_native_vertex_count_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mesh, original = self.shell_fixture(directory)
            for path, value in [
                (("shell", "rows"), True),
                (("shell", "rows"), 66),
                (("shell", "ring_vertices"), 74),
                (("identity", "recipe", "lip_segments"), 0),
                (("vertices",), original["vertices"] + 1),
                (("normal_count",), original["normal_count"] - 1),
            ]:
                broken = copy.deepcopy(original)
                target = broken
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                renderer.write_json(directory / "build.json", broken)
                with self.assertRaises(ValueError):
                    renderer.shell_material_metadata(mesh, renderer.digest(mesh))
            mesh.write_text(mesh.read_text().replace("element vertex 282", "element vertex 283"))
            original["mesh_sha256"] = renderer.digest(mesh)
            renderer.write_json(directory / "build.json", original)
            with self.assertRaisesRegex(ValueError, "PLY counts"):
                renderer.shell_material_metadata(mesh, renderer.digest(mesh))

    def test_interior_material_controls_are_optional_and_strictly_validated(self):
        config = copy.deepcopy(renderer.DEFAULTS)
        self.assertIsNone(config["material"]["interior_color"])
        renderer.validate(config)
        config["material"].update(interior_color=[0.08, 0.3, 0.22], interior_roughness=0.2)
        renderer.validate(config)
        for color in [[0.2, 0.3], [0.2, -0.1, 0.3], [0.2, math.nan, 0.3]]:
            config["material"]["interior_color"] = color
            with self.assertRaises(ValueError):
                renderer.validate(config)

    def test_imported_tiny_slivers_are_measured_without_float32_cancellation(self):
        def imported(point):
            return tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in point)

        a = imported((0.5, 0.5, 0.5))
        b = imported((0.5 + 2**-24, 0.5, 0.5))
        c = imported((0.5 + 2**-23, 0.5 + 2**-24, 0.5))
        self.assertEqual(renderer.triangle_area(a, b, c), 2**-49)
        skinny = [
            imported(point) for point in [(1e-4, 1e-4, 0), (2e-4, 2e-4, 0), (2e-4, 2e-4 + 1e-11, 0)]
        ]
        self.assertGreater(renderer.triangle_area(*skinny), 0.0)
        self.assertEqual(renderer.triangle_area(a, b, b), 0.0)
        self.assertEqual(renderer.triangle_area(a, b, imported((0.5 + 2**-23, 0.5, 0.5))), 0.0)
        self.assertFalse(math.isfinite(renderer.triangle_area(a, b, (math.nan, 0, 0))))

    def test_defaults_and_solid_material_units_are_valid(self):
        clay = copy.deepcopy(renderer.DEFAULTS)
        renderer.validate(clay)
        porcelain = renderer.merge_config(
            clay, {"material": {"kind": "porcelain", "subsurface_radius_mm": 0.65}}
        )
        renderer.validate(porcelain)
        self.assertEqual(clay["material"]["subsurface_radius_mm"], 0.0)
        porcelain["material"]["subsurface_radius_mm"] = 0.0
        with self.assertRaises(ValueError):
            renderer.validate(porcelain)

    def test_unknown_geometry_changes_and_unbounded_resources_are_rejected(self):
        for extra in [{"model_scale_xyz": [1, 2, 3]}, {"material": {"displacement": 0.1}}]:
            with self.assertRaises(ValueError):
                renderer.merge_config(renderer.DEFAULTS, extra)
        for key, value in [("threads", 129), ("samples", True), ("resolution", [1600, -1])]:
            config = copy.deepcopy(renderer.DEFAULTS)
            config["render"][key] = value
            with self.assertRaises(ValueError):
                renderer.validate(config)
        config = copy.deepcopy(renderer.DEFAULTS)
        config["meters_per_unit"] = float("nan")
        with self.assertRaises(ValueError):
            renderer.validate(config)

    def test_invalid_camera_and_light_directions_are_rejected(self):
        for section in ("camera", "light"):
            config = copy.deepcopy(renderer.DEFAULTS)
            item = config["cameras"]["front"] if section == "camera" else config["lights"][0]
            item["position"] = item["target"][:]
            with self.assertRaises(ValueError):
                renderer.validate(config)

    def test_resume_requires_every_archived_artifact_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names = ("render.png", "render.exr", "scene.blend", "recipe.json")
            for name in names:
                (output / name).write_bytes(name.encode())
            receipt = {
                "complete": True,
                "identity_sha256": "identity-a",
                "artifacts": {name: {"sha256": renderer.digest(output / name)} for name in names},
            }
            renderer.write_json(output / "receipt.json", receipt)
            self.assertTrue(renderer.completed_matches(output, "identity-a"))
            self.assertFalse(renderer.completed_matches(output, "identity-b"))
            (output / "render.exr").write_bytes(b"changed linear master")
            self.assertFalse(renderer.completed_matches(output, "identity-a"))

    def test_mesh_header_requires_canonical_format_and_bounded_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            valid = (
                "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
                "element face 4\nend_header\n"
            )
            path.write_text(valid)
            self.assertEqual(renderer.ply_header(path), {"vertex": 4, "face": 4})
            for broken in [
                valid.replace("binary_little_endian", "ascii"),
                valid.replace("4", "-1"),
            ]:
                path.write_text(broken)
                with self.assertRaises(ValueError):
                    renderer.ply_header(path)


if __name__ == "__main__":
    unittest.main()
