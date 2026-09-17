#!/usr/bin/env python3
"""Photograph an unchanged Remaining Form mesh with pinned Blender/Cycles CPU.

Run inside the official Blender 4.5.14 build, for example:
  blender --factory-startup -b --threads 4 --python-exit-code 1 --python render.py -- \
    --mesh sculpture.ply --recipe studio-clay.json --output proof --view front

The mesh receives one explicit rigid rotation and one uniform conversion from
canonical units to metres. Its vertex data and topology are never edited.
The scene, display PNG16, scene-linear RGB32 EXR, resolved recipe and receipts
are archived together. --resume requires identical inputs and renderer identity.
Fixed seeds improve repeatability; cross-platform pixel identity is not claimed.

For animation, freeze ground.height_units in the recipe after the initial scene
is chosen. A null height fits the ground below the current mesh for still proofs.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from array import array
from pathlib import Path

DEFAULTS = {
    "schema_version": 1,
    "blender_version": "4.5.14",
    "meters_per_unit": 0.05,
    "model_rotation_degrees": [90.0, 0.0, 0.0],
    "cameras": {
        "front": {"position": [7.0, -11.0, 6.0], "target": [0.0, 0.0, 0.0], "lens_mm": 65.0},
        "side": {"position": [12.0, -6.0, 5.0], "target": [0.0, 0.0, 0.0], "lens_mm": 65.0},
        "back": {"position": [-7.0, 11.0, 6.0], "target": [0.0, 0.0, 0.0], "lens_mm": 65.0},
    },
    "camera": {"sensor_width_mm": 36.0, "depth_of_field": False, "f_stop": 11.0},
    "lights": [
        {
            "name": "Key",
            "position": [-4.0, -5.0, 8.0],
            "target": [0.0, 0.0, 0.3],
            "size": [5.0, 4.0],
            "power_watts": 2.4,
            "color": [1.0, 0.96, 0.89],
        },
        {
            "name": "Fill",
            "position": [5.0, -2.0, 4.0],
            "target": [0.0, 0.0, 0.1],
            "size": [5.0, 5.0],
            "power_watts": 0.22,
            "color": [0.81, 0.9, 1.0],
        },
        {
            "name": "Shoulder",
            "position": [1.0, 4.0, 6.0],
            "target": [0.0, 0.0, 0.3],
            "size": [3.0, 4.0],
            "power_watts": 0.7,
            "color": [1.0, 0.96, 0.9],
        },
    ],
    "world": {"color": [0.26, 0.25, 0.23], "strength": 0.055},
    "ground": {
        "height_units": None,
        "gap_mm": 0.15,
        "size_units": 80.0,
        "color": [0.19, 0.175, 0.153],
        "roughness": 0.8,
    },
    "material": {
        "kind": "clay",
        "base_color": [0.58, 0.54, 0.47],
        "roughness": 0.64,
        "interior_color": None,
        "interior_roughness": 0.2,
        "thin_film_nm": 0.0,
        "thin_film_ior": 1.35,
        "ior": 1.48,
        "subsurface_radius_mm": 0.0,
        "subsurface_rgb": [1.0, 0.88, 0.72],
        "smooth_normals": True,
    },
    "render": {
        "resolution": [1600, 1200],
        "samples": 128,
        "seed": 17017,
        "threads": 4,
        "denoise": False,
        "exposure": 0.0,
        "view_transform": "AgX",
        "look": "AgX - Medium High Contrast",
        "max_bounces": 12,
    },
}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    encoded(value)  # Also rejects finite JSON syntax whose exponent overflows.
    return value


def merge_config(default, supplied, location="recipe"):
    if not isinstance(supplied, dict):
        raise ValueError(f"{location} must be an object")
    unknown = supplied.keys() - default.keys()
    if unknown:
        raise ValueError(f"Unknown {location} keys: {sorted(unknown)}")
    result = {}
    for key, value in default.items():
        provided = supplied.get(key, value)
        result[key] = (
            merge_config(value, provided, f"{location}.{key}")
            if isinstance(value, dict)
            else provided
        )
    return result


def number(value, name, minimum, maximum):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")


def vector(value, name, length=3, minimum=-1000.0, maximum=1000.0):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    for component in value:
        number(component, name, minimum, maximum)


def validate(config):
    if config["schema_version"] != 1 or type(config["schema_version"]) is not int:
        raise ValueError("Unsupported studio recipe schema")
    if config["blender_version"] != "4.5.14":
        raise ValueError("This scene adapter is pinned to Blender 4.5.14")
    number(config["meters_per_unit"], "meters_per_unit", 0.001, 1.0)
    vector(config["model_rotation_degrees"], "model_rotation_degrees", minimum=-360, maximum=360)
    for name, camera in config["cameras"].items():
        vector(camera["position"], f"{name} camera position")
        vector(camera["target"], f"{name} camera target")
        if math.dist(camera["position"], camera["target"]) < 0.01:
            raise ValueError(f"{name} camera position must differ from its target")
        number(camera["lens_mm"], "lens_mm", 15.0, 250.0)
    optics = config["camera"]
    number(optics["sensor_width_mm"], "sensor_width_mm", 10.0, 80.0)
    number(optics["f_stop"], "f_stop", 1.0, 64.0)
    if type(optics["depth_of_field"]) is not bool:
        raise ValueError("depth_of_field must be true or false")
    lights = config["lights"]
    if not isinstance(lights, list) or not 1 <= len(lights) <= 8:
        raise ValueError("Use one through eight explicit area lights")
    names = set()
    for light in lights:
        if not isinstance(light, dict) or light.keys() != DEFAULTS["lights"][0].keys():
            raise ValueError("Each area light must specify exactly the documented light fields")
        if not isinstance(light["name"], str) or not light["name"] or light["name"] in names:
            raise ValueError("Lights require distinct nonempty names")
        names.add(light["name"])
        vector(light["position"], "light position")
        vector(light["target"], "light target")
        if math.dist(light["position"], light["target"]) < 0.01:
            raise ValueError("Each light position must differ from its target")
        vector(light["size"], "light size", 2, 0.01, 100.0)
        vector(light["color"], "light color", minimum=0.0, maximum=1.0)
        number(light["power_watts"], "light power_watts", 0.0, 1000.0)
    vector(config["world"]["color"], "world color", minimum=0.0, maximum=1.0)
    number(config["world"]["strength"], "world strength", 0.0, 10.0)
    ground = config["ground"]
    if ground["height_units"] is not None:
        number(ground["height_units"], "ground height_units", -1000.0, 1000.0)
    number(ground["gap_mm"], "ground gap_mm", 0.0, 20.0)
    number(ground["size_units"], "ground size_units", 5.0, 1000.0)
    vector(ground["color"], "ground color", minimum=0.0, maximum=1.0)
    number(ground["roughness"], "ground roughness", 0.05, 1.0)
    material = config["material"]
    if material["kind"] not in ("clay", "porcelain"):
        raise ValueError("Material kind must be clay or porcelain")
    vector(material["base_color"], "material base_color", minimum=0.0, maximum=1.0)
    number(material["roughness"], "material roughness", 0.05, 1.0)
    if material["interior_color"] is not None:
        vector(material["interior_color"], "material interior_color", minimum=0.0, maximum=1.0)
    number(material["interior_roughness"], "material interior_roughness", 0.05, 1.0)
    number(material["thin_film_nm"], "material thin_film_nm", 0.0, 1500.0)
    number(material["thin_film_ior"], "material thin_film_ior", 1.01, 2.5)
    if material["thin_film_nm"] != 0.0 and material["kind"] != "porcelain":
        raise ValueError("A nonzero thin film is supported only on the porcelain sculpture")
    number(material["ior"], "material ior", 1.01, 2.0)
    number(material["subsurface_radius_mm"], "subsurface_radius_mm", 0.0, 10.0)
    vector(material["subsurface_rgb"], "subsurface_rgb", minimum=0.01, maximum=4.0)
    if (material["kind"] == "clay") != (material["subsurface_radius_mm"] == 0.0):
        raise ValueError(
            "Clay requires zero subsurface radius; porcelain requires a positive radius"
        )
    if type(material["smooth_normals"]) is not bool:
        raise ValueError("smooth_normals must be true or false")
    render = config["render"]
    resolution = render["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("render.resolution must contain width and height")
    for value, label, lower, upper in [
        (resolution[0], "width", 32, 16384),
        (resolution[1], "height", 32, 16384),
        (render["samples"], "samples", 1, 16384),
        (render["seed"], "seed", 0, 2147483647),
        (render["threads"], "threads", 1, 128),
        (render["max_bounces"], "max_bounces", 4, 32),
    ]:
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")
    if type(render["denoise"]) is not bool:
        raise ValueError("denoise must be true or false")
    number(render["exposure"], "exposure", -12.0, 12.0)
    if render["view_transform"] != "AgX" or render["look"] not in (
        "AgX - Base Contrast",
        "AgX - Medium High Contrast",
        "AgX - Medium Low Contrast",
    ):
        raise ValueError("Use AgX with an explicit supported photographic contrast look")


def ply_header(path):
    with path.open("rb") as stream:
        lines = []
        total = 0
        while total < 65536:
            line = stream.readline(4096)
            total += len(line)
            if not line:
                break
            text = line.decode("ascii").strip()
            lines.append(text)
            if text == "end_header":
                break
    if not lines or lines[0] != "ply" or lines[-1] != "end_header":
        raise ValueError("Mesh must have a bounded, complete PLY header")
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("Use the canonical binary little-endian PLY mesh")
    counts = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[:2] in (["element", "vertex"], ["element", "face"]):
            counts[parts[1]] = int(parts[2])
    if not 4 <= counts.get("vertex", 0) <= 10_000_000:
        raise ValueError("PLY vertex count must be in [4, 10000000]")
    if not 4 <= counts.get("face", 0) <= 20_000_000:
        raise ValueError("PLY triangle count must be in [4, 20000000]")
    return counts


def runtime_identity(bpy):
    def text(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

    binary = Path(bpy.app.binary_path).resolve(strict=True)
    if os.environ.get("OCIO"):
        raise ValueError("Use the pinned Blender color configuration without an OCIO override")
    color_folder = binary.parent / "4.5" / "datafiles" / "colormanagement"
    color_files = sorted(path for path in color_folder.rglob("*") if path.is_file())
    if not (color_folder / "config.ocio").is_file() or not color_files:
        raise ValueError("Pinned Blender color-management files are missing")
    color_identity = [[str(path.relative_to(color_folder)), digest(path)] for path in color_files]
    return {
        "version": bpy.app.version_string,
        "version_tuple": list(bpy.app.version),
        "build_hash": text(bpy.app.build_hash),
        "build_date": text(bpy.app.build_date),
        "build_time": text(bpy.app.build_time),
        "binary_sha256": digest(binary),
        "color_management_sha256": hashlib.sha256(encoded(color_identity)).hexdigest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def shell_material_metadata(mesh_path, mesh_sha256):
    """Bind the semantic wall layout to this exact shell mesh and build receipt."""
    receipt_path = mesh_path.with_name("build.json")
    raw = receipt_path.read_bytes()
    record = json.loads(raw)
    encoded(record)  # Reject nonfinite metadata, including overflowed exponents.
    if not isinstance(record, dict) or record.get("mesh_sha256") != mesh_sha256:
        raise ValueError("Shell build receipt does not match the supplied mesh SHA256")
    identity = record.get("identity")
    shell = record.get("shell")
    if (
        not isinstance(identity, dict)
        or type(identity.get("schema_version")) is not int
        or identity["schema_version"] != 1
        or not isinstance(identity.get("recipe"), dict)
        or not isinstance(shell, dict)
    ):
        raise ValueError("Interior material requires a version-one shell build receipt")
    recipe = identity["recipe"]
    nv = recipe.get("transverse_segments")
    nl = recipe.get("lip_segments")
    longitudinal = recipe.get("longitudinal_segments")
    rows = shell.get("rows")
    ring = shell.get("ring_vertices")
    for value, label, minimum, maximum in [
        (nv, "transverse_segments", 32, 512),
        (nl, "lip_segments", 3, 16),
        (longitudinal, "longitudinal_segments", 64, 4096),
        (rows, "rows", 2, 4097),
        (ring, "ring_vertices", 70, 1056),
    ]:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid shell layout {label}")
    if ring != 2 * nv + 2 * nl or rows > longitudinal + 1:
        raise ValueError("Shell ring or row count disagrees with its construction recipe")
    vertices = rows * ring + 2 * (nv + 1)
    triangles = 2 * (rows - 1) * ring + 8 * nv + 4 * nl
    for key, expected in [
        ("vertices", vertices),
        ("normal_count", vertices),
        ("triangles", triangles),
    ]:
        if type(record.get(key)) is not int or record[key] != expected:
            raise ValueError(f"Shell build receipt has an inconsistent {key} count")
    if ply_header(mesh_path) != {"vertex": vertices, "face": triangles}:
        raise ValueError("PLY counts do not match the shell wall and end-cap layout")
    return {
        "build_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "layout": {
            "version": 1,
            "rows": rows,
            "ring_vertices": ring,
            "transverse_segments": nv,
            "lip_segments": nl,
            "vertices": vertices,
            "triangles": triangles,
            "attribute": "shell_interior",
            "mapping": "outer=0; inner=1; cosine-rounded-lips; cap-midpoints=0.5",
        },
    }


def shell_interior_weights(layout):
    """Continuous material coordinates around each closed shell-wall ring."""
    nv, nl = layout["transverse_segments"], layout["lip_segments"]
    ring = array("f", [0.0]) * (nv + 1)
    ring.extend(0.5 - 0.5 * math.cos(math.pi * j / nl) for j in range(1, nl))
    ring.extend([1.0] * (nv + 1))
    ring.extend(0.5 + 0.5 * math.cos(math.pi * j / nl) for j in range(1, nl))
    if len(ring) != layout["ring_vertices"]:
        raise ValueError("Interior material ring weights disagree with the validated layout")
    weights = ring * layout["rows"]
    weights.extend([0.5] * (2 * (nv + 1)))
    if len(weights) != layout["vertices"]:
        raise ValueError("Interior material weights do not match the shell vertex count")
    return weights


def add_shell_material_attribute(mesh, material, bsdf, settings, metadata):
    """Tint the existing wall through one BSDF without changing its geometry."""
    layout = metadata["layout"]
    if len(mesh.vertices) != layout["vertices"]:
        raise ValueError("Imported vertex count does not match the verified shell material layout")
    name = layout["attribute"]
    if mesh.attributes.get(name) is not None:
        raise ValueError(
            "The reserved shell material attribute already exists on the imported mesh"
        )
    attribute = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    attribute.data.foreach_set("value", shell_interior_weights(layout))
    nodes, links = material.node_tree.nodes, material.node_tree.links
    coordinate = nodes.new("ShaderNodeAttribute")
    coordinate.name = "Verified shell interior"
    coordinate.attribute_name = name
    color = nodes.new("ShaderNodeMixRGB")
    color.name = "Ivory exterior to colored interior"
    color.blend_type = "MIX"
    color.inputs[1].default_value = (*settings["base_color"], 1.0)
    color.inputs[2].default_value = (*settings["interior_color"], 1.0)
    links.new(coordinate.outputs["Fac"], color.inputs[0])
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    roughness = nodes.new("ShaderNodeMath")
    roughness.name = "Exterior to interior roughness"
    roughness.operation = "MULTIPLY_ADD"
    roughness.inputs[1].default_value = settings["interior_roughness"] - settings["roughness"]
    roughness.inputs[2].default_value = settings["roughness"]
    links.new(coordinate.outputs["Fac"], roughness.inputs[0])
    links.new(roughness.outputs[0], bsdf.inputs["Roughness"])


def point_at(obj, target, Vector):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def material_nodes(bpy, name, color, roughness, ior):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Transmission Weight"].default_value = 0.0
    bsdf.inputs["Coat Weight"].default_value = 0.0
    return material, bsdf


def triangle_area(a, b, c):
    """Measure imported coordinates in double precision without editing them.

    Blender stores mesh coordinates as float32. Promote before subtracting and
    crossing; bmesh's area calculation can cancel to zero on valid small slivers.
    """
    a, b, c = (tuple(float(component) for component in point) for point in (a, b, c))
    u = tuple(b[axis] - a[axis] for axis in range(3))
    v = tuple(c[axis] - a[axis] for axis in range(3))
    cross = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    return 0.5 * math.hypot(*cross)


def build_scene(bpy, mesh_path, config, view, shell_material=None):
    import bmesh
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.threads_mode = "FIXED"
    scene.render.threads = config["render"]["threads"]
    counts = ply_header(mesh_path)
    bpy.ops.wm.ply_import(
        filepath=str(mesh_path),
        global_scale=1.0,
        use_scene_unit=False,
        forward_axis="Y",
        up_axis="Z",
        merge_verts=False,
    )
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise ValueError("Canonical PLY must import as exactly one mesh object")
    sculpture = meshes[0]
    sculpture.name = "The Remaining Form"
    mesh = sculpture.data
    if len(mesh.vertices) != counts["vertex"] or len(mesh.polygons) != counts["face"]:
        raise ValueError("PLY import changed the declared vertex or face counts")
    if any(len(face.vertices) != 3 for face in mesh.polygons):
        raise ValueError("Canonical mesh must already be triangulated")
    if any(not all(math.isfinite(v) for v in vertex.co) for vertex in mesh.vertices):
        raise ValueError("Imported mesh contains nonfinite coordinates")
    audit = bmesh.new()
    try:
        audit.from_mesh(mesh)
        invalid_edges = sum(not edge.is_manifold or not edge.is_contiguous for edge in audit.edges)
        invalid_vertices = sum(not vertex.is_manifold for vertex in audit.verts)
        degenerate_faces = 0
        for face in mesh.polygons:
            area = triangle_area(*(mesh.vertices[index].co for index in face.vertices))
            degenerate_faces += int(not math.isfinite(area) or area == 0.0)
        signed_volume = audit.calc_volume(signed=True)
    finally:
        audit.free()
    if invalid_edges or invalid_vertices or degenerate_faces or not signed_volume > 0.0:
        raise ValueError(
            "Closed oriented mesh required: "
            f"bad_edges={invalid_edges}, bad_vertices={invalid_vertices}, "
            f"degenerate_faces={degenerate_faces}, signed_volume={signed_volume}"
        )
    scale = config["meters_per_unit"]
    sculpture.location = (0.0, 0.0, 0.0)
    sculpture.rotation_euler = tuple(math.radians(v) for v in config["model_rotation_degrees"])
    sculpture.scale = (scale, scale, scale)
    settings = config["material"]
    for face in mesh.polygons:
        face.use_smooth = settings["smooth_normals"]
    material, bsdf = material_nodes(
        bpy,
        settings["kind"].title(),
        settings["base_color"],
        settings["roughness"],
        settings["ior"],
    )
    # A constant dielectric film changes the sculpture's specular transport.
    # Zero thickness disables it. The studio floor uses a separate material.
    bsdf.inputs["Thin Film Thickness"].default_value = settings["thin_film_nm"]
    bsdf.inputs["Thin Film IOR"].default_value = settings["thin_film_ior"]
    if settings["interior_color"] is not None:
        if shell_material is None:
            raise ValueError("Interior material requires verified shell metadata")
        add_shell_material_attribute(mesh, material, bsdf, settings, shell_material)
    if settings["kind"] == "porcelain":
        bsdf.subsurface_method = "RANDOM_WALK"
        bsdf.inputs["Subsurface Weight"].default_value = 1.0
        bsdf.inputs["Subsurface Radius"].default_value = tuple(settings["subsurface_rgb"])
        bsdf.inputs["Subsurface Scale"].default_value = settings["subsurface_radius_mm"] / 1000.0
        # The separate Subsurface IOR socket belongs to Blender's skin method;
        # it is removed when the ordinary solid-material Random Walk is chosen.
        bsdf.inputs["Subsurface Anisotropy"].default_value = 0.0
    else:
        bsdf.inputs["Subsurface Weight"].default_value = 0.0
    mesh.materials.clear()
    mesh.materials.append(material)
    bpy.context.view_layer.update()
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    for vertex in mesh.vertices:
        point = sculpture.matrix_world @ vertex.co
        for axis in range(3):
            minimum[axis] = min(minimum[axis], point[axis])
            maximum[axis] = max(maximum[axis], point[axis])
    ground_config = config["ground"]
    ground_z = (
        minimum[2] - ground_config["gap_mm"] / 1000.0
        if ground_config["height_units"] is None
        else ground_config["height_units"] * scale
    )
    if ground_z > minimum[2] + 1e-7 * scale:
        raise ValueError("The explicitly positioned ground intersects the sculpture")
    bpy.ops.mesh.primitive_plane_add(
        size=ground_config["size_units"] * scale, location=(0.0, 0.0, ground_z)
    )
    ground = bpy.context.object
    ground.name = "Warm grey studio floor"
    floor_material, _ = material_nodes(
        bpy, "Studio floor", ground_config["color"], ground_config["roughness"], 1.45
    )
    ground.data.materials.append(floor_material)
    world = bpy.data.worlds.new("Quiet photographic studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        *config["world"]["color"],
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = config["world"][
        "strength"
    ]
    scene.world = world
    for light in config["lights"]:
        data = bpy.data.lights.new(light["name"], "AREA")
        data.shape = "RECTANGLE"
        data.energy = light["power_watts"]
        data.color = light["color"]
        data.size, data.size_y = (value * scale for value in light["size"])
        obj = bpy.data.objects.new(light["name"], data)
        scene.collection.objects.link(obj)
        obj.location = tuple(value * scale for value in light["position"])
        point_at(obj, [value * scale for value in light["target"]], Vector)
    for name, camera in config["cameras"].items():
        data = bpy.data.cameras.new(f"Camera {name}")
        data.lens = camera["lens_mm"]
        data.sensor_width = config["camera"]["sensor_width_mm"]
        data.sensor_fit = "HORIZONTAL"
        data.clip_start = 0.01 * scale
        data.clip_end = 200.0 * scale
        data.dof.use_dof = config["camera"]["depth_of_field"]
        data.dof.aperture_fstop = config["camera"]["f_stop"]
        data.dof.focus_distance = math.dist(camera["position"], camera["target"]) * scale
        obj = bpy.data.objects.new(f"Camera {name}", data)
        scene.collection.objects.link(obj)
        obj.location = tuple(value * scale for value in camera["position"])
        point_at(obj, [value * scale for value in camera["target"]], Vector)
        if name == view:
            scene.camera = obj
    render = config["render"]
    scene.render.resolution_x, scene.render.resolution_y = render["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_persistent_data = False
    scene.cycles.samples = render["samples"]
    scene.cycles.seed = render["seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = render["denoise"]
    if render["denoise"]:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_use_gpu = False
    scene.cycles.max_bounces = render["max_bounces"]
    scene.cycles.diffuse_bounces = 6
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_bounces = 2
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = render["view_transform"]
    scene.view_settings.look = render["look"]
    scene.view_settings.exposure = render["exposure"]
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 15
    scene.render.filepath = "//render.png"
    scene.frame_set(1)
    return scene, {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.polygons),
        "signed_volume_canonical": signed_volume,
        "signed_volume_m3": signed_volume * scale**3,
        "bounds_m": [minimum, maximum],
        "ground_height_m": ground_z,
        "ground_height_units": ground_z / scale,
        "geometry_changes": "none; one explicit rigid rotation and uniform metre conversion",
        "normal_shading": "smooth" if settings["smooth_normals"] else "flat",
        "import_precision": "Blender mesh positions use native single-precision storage",
    }


def completed_matches(output, identity):
    path = output / "receipt.json"
    if not path.is_file():
        return False
    receipt = read_json(path)
    if receipt.get("identity_sha256") != identity or receipt.get("complete") is not True:
        return False
    artifacts = receipt.get("artifacts", {})
    for name in ("render.png", "render.exr", "scene.blend", "recipe.json"):
        record = artifacts.get(name, {})
        file = output / name
        if not file.is_file() or digest(file) != record.get("sha256"):
            return False
    return True


def verify_image_headers(png_path, exr_path, resolution):
    with png_path.open("rb") as stream:
        header = stream.read(26)
    if (
        len(header) != 26
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
        or list(struct.unpack(">II", header[16:24])) != resolution
        or header[24:26] != bytes((16, 2))
    ):
        raise RuntimeError("Display output is not the requested RGB16 PNG")

    def string(stream):
        result = bytearray()
        for _ in range(256):
            value = stream.read(1)
            if value == b"\0":
                return bytes(result)
            if not value:
                break
            result.extend(value)
        raise RuntimeError("Invalid OpenEXR header string")

    with exr_path.open("rb") as stream:
        if stream.read(4) != b"\x76\x2f\x31\x01":
            raise RuntimeError("Linear output is not an OpenEXR file")
        stream.read(4)  # Version and flags.
        attributes = {}
        while stream.tell() < 1_048_576:
            name = string(stream)
            if not name:
                break
            kind = string(stream)
            size_bytes = stream.read(4)
            if len(size_bytes) != 4:
                raise RuntimeError("Incomplete OpenEXR header")
            size = struct.unpack("<I", size_bytes)[0]
            if size > 1_048_576:
                raise RuntimeError("Unexpectedly large OpenEXR attribute")
            payload = stream.read(size)
            if len(payload) != size:
                raise RuntimeError("Incomplete OpenEXR attribute")
            attributes[name] = (kind, payload)
        window = attributes.get(b"dataWindow")
        if window is None or window[0] != b"box2i" or len(window[1]) != 16:
            raise RuntimeError("Linear output has no valid pixel window")
        x0, y0, x1, y1 = struct.unpack("<4i", window[1])
        if [x1 - x0 + 1, y1 - y0 + 1] != resolution:
            raise RuntimeError("Linear output dimensions differ from the recipe")
        channels = attributes.get(b"channels")
        if channels is None or channels[0] != b"chlist":
            raise RuntimeError("Linear output has no channel description")
        payload = channels[1]
        offset = 0
        names = set()
        while offset < len(payload) and payload[offset] != 0:
            end = payload.find(b"\0", offset)
            if end < offset or end + 17 > len(payload):
                raise RuntimeError("Malformed OpenEXR channel description")
            name = payload[offset:end]
            pixel_type, _, x_sampling, y_sampling = struct.unpack(
                "<iB3xii", payload[end + 1 : end + 17]
            )
            if name in names or pixel_type != 2 or (x_sampling, y_sampling) != (1, 1):
                raise RuntimeError("Linear output must contain full-resolution float32 channels")
            names.add(name)
            offset = end + 17
        if names != {b"R", b"G", b"B"} or offset != len(payload) - 1:
            raise RuntimeError("Linear output must contain exactly RGB channels")
        if attributes.get(b"compression") != (b"compression", b"\x03"):
            raise RuntimeError("Linear output must use the recorded lossless ZIP compression")


def run(args):
    import fcntl

    import bpy

    started = time.monotonic()
    mesh_path = args.mesh.resolve(strict=True)
    recipe_path = args.recipe.resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    if not mesh_path.is_file() or not recipe_path.is_file():
        raise ValueError("Mesh and recipe must be regular files")
    config = merge_config(DEFAULTS, read_json(recipe_path))
    validate(config)
    runtime = runtime_identity(bpy)
    if runtime["version_tuple"] != [4, 5, 14]:
        raise ValueError(f"Expected Blender 4.5.14, found {runtime['version']}")
    if not bpy.app.background:
        raise ValueError("Use a dedicated background Blender process (-b)")
    request = {
        "schema_version": 1,
        "mesh_sha256": digest(mesh_path),
        "script_sha256": digest(script_path),
        "recipe_sha256": hashlib.sha256(encoded(config)).hexdigest(),
        "runtime": runtime,
        "view": args.view,
        "config": config,
    }
    shell_material = None
    if config["material"]["interior_color"] is not None:
        shell_material = shell_material_metadata(mesh_path, request["mesh_sha256"])
        request["shell_material"] = shell_material
    identity = hashlib.sha256(encoded(request)).hexdigest()
    output = args.output.resolve()
    if output.exists() and (not args.resume or not output.is_dir()):
        raise ValueError("Output exists; use a new directory or --resume with identical inputs")
    if args.resume and not output.is_dir():
        raise ValueError("--resume requires an existing scene archive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".render.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_path = output / "request.json"
        if request_path.exists():
            if read_json(request_path) != request:
                raise ValueError("Scene archive belongs to different inputs, settings, or renderer")
        elif args.resume:
            raise ValueError("Existing scene archive has no matching request")
        else:
            write_json(request_path, request)
        if completed_matches(output, identity):
            print(json.dumps({"status": "reused", "output": str(output), "identity": identity}))
            return
        write_json(output / "recipe.json", config)
        receipt = {"schema_version": 1, "identity_sha256": identity, "complete": False}
        write_json(output / "receipt.json", receipt)
        scene, geometry = build_scene(bpy, mesh_path, config, args.view, shell_material)
        if digest(mesh_path) != request["mesh_sha256"]:
            raise ValueError("Mesh changed while the scene was being prepared")
        if shell_material is not None:
            if digest(mesh_path.with_name("build.json")) != shell_material["build_receipt_sha256"]:
                raise ValueError("Shell build receipt changed while the scene was being prepared")
            geometry["shell_material"] = shell_material
        text = bpy.data.texts.new("remaining-form-render.py")
        text.write(script_path.read_text(encoding="utf-8"))
        scene["remaining_form_identity"] = identity
        scene["remaining_form_recipe"] = encoded(config).decode()
        temporary_blend = output / "scene.partial.blend"
        bpy.ops.wm.save_as_mainfile(
            filepath=str(temporary_blend), check_existing=False, compress=True
        )
        temporary_blend.replace(output / "scene.blend")
        render_started = time.monotonic()
        bpy.ops.render.render(write_still=False)
        render_seconds = time.monotonic() - render_started
        result = bpy.data.images.get("Render Result")
        if result is None:
            raise RuntimeError("Cycles did not produce a Render Result")
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
        temporary_exr = output / "render.partial.exr"
        result.save_render(filepath=str(temporary_exr), scene=scene)
        temporary_exr.replace(output / "render.exr")
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "16"
        scene.render.image_settings.compression = 15
        temporary_png = output / "render.partial.png"
        result.save_render(filepath=str(temporary_png), scene=scene)
        temporary_png.replace(output / "render.png")
        # Render Result.size can be zero in background Blender even when the
        # render slot contains a complete image. Check the saved formats instead.
        verify_image_headers(
            output / "render.png", output / "render.exr", config["render"]["resolution"]
        )
        artifacts = {
            name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
            for name in ("render.png", "render.exr", "scene.blend", "recipe.json")
        }
        receipt.update(
            complete=True,
            geometry=geometry,
            artifacts=artifacts,
            render_seconds=render_seconds,
            elapsed_seconds=time.monotonic() - started,
            pixel_encoding={
                "render.png": "16-bit RGB, sRGB display encoding with the recorded AgX view",
                "render.exr": "32-bit RGB, scene-linear Rec.709/sRGB primaries, lossless ZIP",
            },
            determinism="fixed CPU renderer, seed and samples; no cross-platform identity promise",
        )
        write_json(output / "receipt.json", receipt)
        print(json.dumps({"status": "complete", "output": str(output), "identity": identity}))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mesh", type=Path, required=True)
    result.add_argument("--recipe", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--view", choices=("front", "side", "back"), default="front")
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        run(parser().parse_args(arguments))
    except Exception as error:
        print(f"Remaining Form render failed: {error}", file=sys.stderr)
        raise
