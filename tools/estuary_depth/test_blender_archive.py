"""Native Blender archive precision and physical framing; no GPU render required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.estuary_depth import render

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipIf(bpy is None, "requires native Blender Python")
class BlenderArchiveTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_packed_linear_textures_survive_blend_reload_without_quantization(self):
        fields = {
            "color": np.array(
                [
                    [[0.000013, 0.1234567, 0.999991], [0.015789, 0.8765432, 0.333333]],
                    [[0.0010123, 0.555555, 0.90909], [0.3141592, 0.271828, 0.000071]],
                ],
                dtype=np.float32,
            ),
            "fractions": np.array(
                [
                    [[0.125125, 0.875, 0.000125], [0.041234, 0.700021, 0.258745]],
                    [[0.000017, 0.999972, 0.000011], [0.123451, 0.321765, 0.554784]],
                ],
                dtype=np.float32,
            ),
        }
        images = {}
        for name, values in fields.items():
            image = render.float_image(bpy, name, values)
            images[name] = image
            self.assertTrue(image.is_float)
            self.assertIsNotNone(image.packed_file)
            np.testing.assert_array_equal(
                np.asarray(image.pixels[:], dtype=np.float32).reshape(2, 2, 4)[..., :3], values
            )
        # Preserve the image through a real scene material user, exactly as the
        # production renderer does; unused Blender datablocks are not archived.
        bpy.ops.mesh.primitive_plane_add()
        bpy.context.object.data.materials.append(render.build_material(bpy, images, {}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear-textures.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False, compress=True)
            bpy.ops.wm.open_mainfile(filepath=str(path))
            for name, expected in fields.items():
                image = bpy.data.images[name]
                self.assertTrue(image.is_float, f"Packed {name} was quantized to integer pixels")
                self.assertEqual(image.colorspace_settings.name, "Non-Color")
                actual = np.asarray(image.pixels[:], dtype=np.float32).reshape(2, 2, 4)[..., :3]
                np.testing.assert_array_equal(actual, expected)
                self.assertGreater(np.count_nonzero((actual * 255) % 1), 0)

    def test_cell_centered_solid_and_exact_40_by_30_centimetre_camera_frame(self):
        vertices, loops, starts, totals, _, _ = render.solid_arrays(
            np.zeros((3, 4), dtype=np.float32),
            0.4,
            4 / 3,
            0.008,
            0.0004,
            0,
        )
        mesh = bpy.data.meshes.new("Measured solid")
        mesh.vertices.add(len(vertices))
        mesh.vertices.foreach_set("co", vertices.ravel())
        mesh.loops.add(len(loops))
        mesh.loops.foreach_set("vertex_index", loops)
        mesh.polygons.add(len(totals))
        mesh.polygons.foreach_set("loop_start", starts)
        mesh.polygons.foreach_set("loop_total", totals)
        mesh.update(calc_edges=True)
        coordinates = np.asarray([vertex.co[:] for vertex in mesh.vertices])
        np.testing.assert_allclose(
            np.ptp(coordinates, axis=0)[:2], [0.4 * 3 / 4, 0.3 * 2 / 3], rtol=0, atol=1e-7
        )
        scene = bpy.context.scene
        scene.render.resolution_x, scene.render.resolution_y = 1024, 768
        camera = bpy.data.cameras.new("Measured camera")
        camera.type, camera.ortho_scale = "ORTHO", 0.4
        points = np.asarray([tuple(point) for point in camera.view_frame(scene=scene)])
        np.testing.assert_allclose(np.ptp(points, axis=0)[:2], [0.4, 0.3], rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
