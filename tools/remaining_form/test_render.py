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
