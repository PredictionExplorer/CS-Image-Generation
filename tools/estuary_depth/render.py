#!/usr/bin/env python3
"""Photograph actual Estuary pigment as closed relief and temporal laminations.

Run with pinned Blender 4.5.14, using --factory-startup -b --threads 4
--python-exit-code 1 --python render.py -- --bundle DIR --recipe FILE --output DIR.
Motion freezes every pigment field at the completed painting; only the camera
moves. Metre geometry, linear textures, fixed sampling and full receipts make
each interpretation inspectable. This is not a three-dimensional fluid solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from materials import build_material, validate_config

DEFAULTS = {
    "schema_version": 1,
    "name": "Porcelain current",
    "family": "relief",
    "canvas_width_m": 0.4,
    "relief_mm": 8.0,
    "relief_smoothing_mm": 0.0,
    "layer_gap_mm": 0.5,
    "base_thickness_mm": 0.4,
    "material": {},
    "camera": {
        "tilt_degrees": 12.0,
        "azimuth_degrees": -30.0,
        "zoom": 1.0,
        "target": [0.0, 0.0],
        "orbit_start": [0.0, -40.0],
        "orbit_end": [20.0, 25.0],
    },
    "lighting": {
        "azimuth_degrees": 140.0,
        "elevation_degrees": 32.0,
        "key_watts": 4.0,
        "key_size_m": 0.35,
        "fill_watts": 0.4,
        "world_strength": 0.08,
    },
    "render": {
        "resolution": [1024, 768],
        "samples": 48,
        "seed": 31037,
        "exposure": 0.0,
        "view_transform": "Standard",
        "look": "None",
    },
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    value = json.loads(path.read_text())
    encoded(value)
    return value


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encoded(value))
    partial.replace(path)


def record(path):
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, lo, hi):
    require(
        type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi,
        f"{name} must be finite in [{lo}, {hi}]",
    )


def merge(default, supplied):
    require(isinstance(supplied, dict), "Configuration must be an object")
    require(not supplied.keys() - default.keys(), "Unknown configuration keys")
    return {
        key: merge(value, supplied.get(key, {}))
        if isinstance(value, dict) and key != "material"
        else supplied.get(key, value)
        for key, value in default.items()
    }


def recipe(value):
    config = merge(DEFAULTS, value)
    require(
        type(config["schema_version"]) is int and config["schema_version"] == 1,
        "Unsupported recipe schema",
    )
    require(
        isinstance(config["name"], str) and 0 < len(config["name"]) <= 100,
        "Recipe needs a short name",
    )
    require(config["family"] in ("relief", "layered", "hybrid"), "Unknown study family")
    for key, lo, hi in (
        ("canvas_width_m", 0.1, 2),
        ("relief_mm", 0, 60),
        ("relief_smoothing_mm", 0, 2),
        ("layer_gap_mm", 0.05, 10),
        ("base_thickness_mm", 0.05, 5),
    ):
        number(config[key], key, lo, hi)
    config["material"] = validate_config(config["material"])
    camera = config["camera"]
    number(camera["tilt_degrees"], "camera tilt", 0, 35)
    number(camera["azimuth_degrees"], "camera azimuth", -360, 360)
    number(camera["zoom"], "camera zoom", 0.9, 3)
    for key in ("target", "orbit_start", "orbit_end"):
        require(isinstance(camera[key], list) and len(camera[key]) == 2, f"Invalid {key}")
        for i, component in enumerate(camera[key]):
            bounds = (-0.2, 0.2) if key == "target" else ((0, 35) if i == 0 else (-360, 360))
            number(component, key, *bounds)
    light = config["lighting"]
    for key, lo, hi in (
        ("azimuth_degrees", -360, 360),
        ("elevation_degrees", 5, 85),
        ("key_watts", 0.1, 200),
        ("key_size_m", 0.01, 2),
        ("fill_watts", 0, 50),
        ("world_strength", 0, 2),
    ):
        number(light[key], key, lo, hi)
    render = config["render"]
    require(
        isinstance(render["resolution"], list) and len(render["resolution"]) == 2,
        "Invalid render resolution",
    )
    for size in render["resolution"]:
        require(type(size) is int and 64 <= size <= 7680 and size % 2 == 0, "Invalid pixel size")
    require(math.prod(render["resolution"]) <= 33_177_600, "Render pixel budget exceeded")
    for key, lo, hi in (("samples", 1, 4096), ("seed", 0, 2**31 - 1)):
        require(type(render[key]) is int and lo <= render[key] <= hi, f"Invalid {key}")
    number(render["exposure"], "exposure", -5, 5)
    require(
        (render["view_transform"], render["look"])
        in (("Standard", "None"), ("AgX", "AgX - Medium High Contrast"), ("AgX", "None")),
        "Unsupported display transform",
    )
    return config


def smooth_height(height, width, aspect, millimetres):
    """Spread authored relief at a declared physical scale, preserving total mass.

    Symmetric boundaries keep convolution conservative at the guard edges. This
    changes geometry only; color and the archived concentration field stay exact.
    """
    if millimetres == 0:
        return height
    result = height.astype(np.float64)
    for axis, span in ((0, width / aspect), (1, width)):
        sigma = millimetres / 1000 * height.shape[axis] / span
        radius = max(1, math.ceil(3 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="symmetric")
        result = np.zeros_like(result)
        for index, weight in enumerate(kernel):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(index, index + height.shape[axis])
            result += padded[tuple(slices)] * weight
    return result.astype(np.float32)


def solid_arrays(height, width, aspect, relief, base, offset):
    """Closed oriented solid: grid top, perimeter walls, planar bottom fan."""
    require(
        height.ndim == 2
        and min(height.shape) >= 2
        and np.isfinite(height).all()
        and (height >= 0).all(),
        "Invalid pigment height",
    )
    h, w = height.shape
    # Prepared heights are cell averages, located at cell centers. Align geometry
    # with texture texel centers rather than stretching the values to the edges.
    x, y = np.meshgrid(
        ((np.arange(w) + 0.5) / w - 0.5) * width, ((np.arange(h) + 0.5) / h - 0.5) * width / aspect
    )
    top = np.column_stack((x.ravel(), y.ravel(), offset + height.ravel() * relief))
    ids = np.arange(h * w).reshape(h, w)
    perimeter = np.concatenate((ids[0], ids[1:, -1], ids[-1, -2::-1], ids[-2:0:-1, 0]))
    bottom = top[perimeter].copy()
    bottom[:, 2] = offset - base
    vertices = np.vstack((top, bottom, [0, 0, offset - base])).astype(np.float32)
    a = ids[:-1, :-1].ravel()
    quads = np.column_stack((a, a + 1, a + w + 1, a + w))
    p = len(perimeter)
    b = np.arange(p) + h * w
    walls = np.column_stack((perimeter, b, np.roll(b, -1), np.roll(perimeter, -1)))
    floor = np.column_stack((np.full(p, len(vertices) - 1), np.roll(b, -1), b))
    loops = np.concatenate((quads.ravel(), walls.ravel(), floor.ravel())).astype(np.int32)
    totals = np.concatenate((np.full(len(quads) + p, 4), np.full(p, 3))).astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(totals[:-1]))).astype(np.int32)
    uv = vertices[:, :2] / [width, width / aspect] + 0.5
    return vertices, loops, starts, totals, uv.astype(np.float32), len(quads)


def camera_pose(tilt, azimuth, target):
    """A fixed world-up axis avoids a spurious spin at the overhead endpoint."""
    theta, phi = math.radians(tilt), math.radians(azimuth)
    position = np.asarray(target) + np.asarray(
        [
            0.8 * math.sin(theta) * math.cos(phi),
            0.8 * math.sin(theta) * math.sin(phi),
            0.8 * math.cos(theta),
        ]
    )
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 1, 0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, up, backward))
    matrix[:3, 3] = position
    return matrix


def motion_angles(camera, index, frames):
    if frames == 1:
        return camera["tilt_degrees"], camera["azimuth_degrees"]
    u = index / (frames - 1)
    u = u * u * (3 - 2 * u)
    return tuple(
        a + (b - a) * u for a, b in zip(camera["orbit_start"], camera["orbit_end"], strict=True)
    )


def load_bundle(folder, config):
    manifest = read(folder / "manifest.json")
    require(manifest.get("complete") is True, "Bundle is incomplete")
    require(manifest["bundle"]["path"] == "bundle.npz", "Unexpected bundle path")
    require(
        record(folder / "bundle.npz") == {k: manifest["bundle"][k] for k in ("sha256", "bytes")},
        "Bundle hash differs",
    )
    require(
        hashlib.sha256(encoded(manifest["request"])).hexdigest() == manifest["identity_sha256"],
        "Bundle request identity differs",
    )
    require(manifest["coordinates"]["row_order"] == "bottom-to-top", "Unexpected row orientation")
    require(
        manifest.get("source") == manifest["request"].get("source")
        and isinstance(manifest.get("source"), dict),
        "Source metadata differs from preparation",
    )
    geometry = manifest["request"].get("geometry", {})
    require(
        all(
            manifest[key] == geometry.get(key)
            for key in ("domain_scale", "view_aspect", "coordinates")
        ),
        "Geometry metadata differs from the authenticated preparation request",
    )
    number(manifest["domain_scale"], "domain scale", 1, 4)
    number(manifest["view_aspect"], "view aspect", 0.25, 4)
    dimensions = manifest["request"]["parameters"]
    for key in ("resolution", "mesh_resolution"):
        size = dimensions[key]
        require(
            isinstance(size, list)
            and len(size) == 2
            and all(type(v) is int and 2 <= v <= 4096 for v in size),
            "Invalid map dimensions",
        )
        require(
            abs(size[0] / size[1] - manifest["view_aspect"]) < 1e-9,
            "Map aspect differs from the painting",
        )
    with np.load(folder / "bundle.npz", allow_pickle=False) as archive:
        maps = {
            key: archive[key]
            for key in (
                "history_color_linear",
                "history_fractions",
                "history_height",
                "history_times",
            )
        }
    times = maps["history_times"]
    require(
        times.ndim == 1
        and 1 <= len(times) <= 7
        and np.isfinite(times).all()
        and (times > 0).all()
        and np.all(np.diff(times) > 0)
        and times[-1] == 1,
        "Invalid frozen history times",
    )
    require(times.tolist() == manifest["history"]["source_fractions"], "History metadata differs")
    for key in ("history_color_linear", "history_fractions", "history_height"):
        array = maps[key]
        require(
            array.ndim >= 1
            and array.dtype == np.float32
            and len(array) == len(times)
            and np.isfinite(array).all()
            and (array >= 0).all(),
            f"Invalid {key}",
        )
        if key != "history_height":
            require(
                array.ndim == 4
                and min(array.shape[1:3]) >= 2
                and array.shape[-1] == 3
                and (array <= 1.00001).all(),
                f"Invalid {key} layout or range",
            )
        else:
            require(array.ndim == 3 and min(array.shape[1:]) >= 2, "Invalid height layout")
        expected = dimensions["mesh_resolution" if key == "history_height" else "resolution"]
        require(
            array.shape[1:3] == tuple(reversed(expected)),
            "Array dimensions differ from the authenticated preparation request",
        )
    require(
        maps["history_color_linear"].shape == maps["history_fractions"].shape,
        "Texture shapes differ",
    )
    if config["family"] != "relief":
        require(
            len(times) >= 2 and manifest["history"]["older_history_available"],
            "Lamination needs verified earlier states",
        )
    aspect = manifest["view_aspect"]
    rw, rh = config["render"]["resolution"]
    require(abs(rw / rh - aspect) < 1e-9, "Render aspect differs from painting")
    return manifest, maps


def float_image(bpy, name, value):
    h, w, _ = value.shape
    image = bpy.data.images.new(name, width=w, height=h, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = value
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    image.pack()
    return image


def build_scene(bpy, config, manifest, maps, frames, fps):
    from mathutils import Matrix, Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.refresh_devices()
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
    devices = [{"id": d.id, "name": d.name} for d in preferences.devices if d.use]
    require(bool(devices), "No OptiX device available; CPU fallback is disabled")
    scene.cycles.device = "GPU"
    scene.cycles.samples = config["render"]["samples"]
    scene.cycles.seed = config["render"]["seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = True
    scene.cycles.max_bounces = 16
    scene.cycles.transmission_bounces = 12
    scene.cycles.diffuse_bounces = 6
    scene.cycles.glossy_bounces = 8
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.use_persistent_data = frames > 1
    scene.render.resolution_x, scene.render.resolution_y = config["render"]["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, frames
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = config["render"]["view_transform"]
    scene.view_settings.look = config["render"]["look"]
    scene.view_settings.exposure = config["render"]["exposure"]
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False
    width = config["canvas_width_m"] * manifest["domain_scale"]
    aspect = manifest["view_aspect"]
    indices = (
        [len(maps["history_times"]) - 1]
        if config["family"] == "relief"
        else range(len(maps["history_times"]))
    )
    offset, geometry = 0.0, []
    relief = 0 if config["family"] == "layered" else config["relief_mm"] / 1000
    for order, index in enumerate(indices):
        height = smooth_height(
            maps["history_height"][index], width, aspect, config["relief_smoothing_mm"]
        )
        vertices, loops, starts, totals, uv, top_count = solid_arrays(
            height, width, aspect, relief, config["base_thickness_mm"] / 1000, offset
        )
        mesh = bpy.data.meshes.new(f"Pigment solid {order}")
        mesh.vertices.add(len(vertices))
        mesh.vertices.foreach_set("co", vertices.ravel())
        mesh.loops.add(len(loops))
        mesh.loops.foreach_set("vertex_index", loops)
        mesh.polygons.add(len(totals))
        mesh.polygons.foreach_set("loop_start", starts)
        mesh.polygons.foreach_set("loop_total", totals)
        mesh.polygons.foreach_set("use_smooth", np.arange(len(totals)) < top_count)
        mesh.uv_layers.new(name="Pigment coordinates").data.foreach_set("uv", uv[loops].ravel())
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new(f"Source time {maps['history_times'][index]:.2f}", mesh)
        scene.collection.objects.link(obj)
        material = dict(config["material"])
        if order == 0:
            material.update(blue_transmission=0.0, ivory_transmission=0.0)
        images = {
            key: float_image(bpy, f"{key} {order}", maps[field][index])
            for key, field in (
                ("color", "history_color_linear"),
                ("fractions", "history_fractions"),
            )
        }
        mesh.materials.append(build_material(bpy, images, material, layer_index=order))
        geometry.append(
            {
                "source_fraction": float(maps["history_times"][index]),
                "vertices": len(vertices),
                "faces": len(totals),
                "base_z_m": offset,
                "peak_z_m": float(vertices[:, 2].max()),
                "closed": True,
            }
        )
        offset = (
            float(vertices[:, 2].max())
            + (config["base_thickness_mm"] + config["layer_gap_mm"]) / 1000
        )
    top_z = geometry[-1]["base_z_m"] + float(np.median(maps["history_height"][-1])) * relief
    target = [*config["camera"]["target"], top_z]
    camera = bpy.data.objects.new("Slow examination", bpy.data.cameras.new("Orthographic camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = config["canvas_width_m"] / config["camera"]["zoom"]
    camera.data.clip_start, camera.data.clip_end = 0.01, 10
    camera.rotation_mode = "QUATERNION"
    poses = []
    for index in range(frames):
        tilt, azimuth = motion_angles(config["camera"], index, frames)
        matrix = camera_pose(tilt, azimuth, target)
        camera.matrix_world = Matrix(matrix.tolist())
        if frames > 1:
            camera.keyframe_insert(data_path="location", frame=index + 1)
            camera.keyframe_insert(data_path="rotation_quaternion", frame=index + 1)
        poses.append(
            {
                "frame": index,
                "tilt_degrees": tilt,
                "azimuth_degrees": azimuth,
                "matrix_world": matrix.tolist(),
            }
        )
    light = config["lighting"]
    az, el = math.radians(light["azimuth_degrees"]), math.radians(light["elevation_degrees"])

    def area(name, position, power, size, color):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size = power, "DISK", size
        data.color = color
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area(
        "Broad raking key",
        [
            0.55 * math.cos(el) * math.cos(az),
            0.55 * math.cos(el) * math.sin(az),
            top_z + 0.55 * math.sin(el),
        ],
        light["key_watts"],
        light["key_size_m"],
        (1, 0.96, 0.90),
    )
    area("Quiet fill", [0.12, -0.1, 0.65 + top_z], light["fill_watts"], 0.6, (0.85, 0.92, 1))
    world = bpy.data.worlds.new("Quiet studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.7, 0.8, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = light["world_strength"]
    scene.world = world
    scene.frame_set(1)
    return scene, geometry, devices, poses


def save_image(bpy, scene, path, *, linear=False):
    scene.render.image_settings.file_format = "OPEN_EXR" if linear else "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32" if linear else "16"
    if linear:
        scene.render.image_settings.exr_codec = "ZIP"
    else:
        scene.render.image_settings.compression = 15
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    bpy.data.images["Render Result"].save_render(filepath=str(partial), scene=scene)
    if not linear:
        with partial.open("rb") as stream:
            header = stream.read(29)
        require(
            header[:8] == b"\x89PNG\r\n\x1a\n"
            and struct.unpack(">II", header[16:24]) == tuple(configured_resolution(scene))
            and header[24:26] == bytes([16, 2]),
            "Saved PNG format differs",
        )
    partial.replace(path)


def configured_resolution(scene):
    return [scene.render.resolution_x, scene.render.resolution_y]


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "recipe", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--motion-frames", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    require(bpy.app.version == (4, 5, 14), "Use pinned Blender 4.5.14")
    require(1 <= args.motion_frames <= 1441 and 1 <= args.fps <= 60, "Invalid motion budget")
    config = recipe(read(args.recipe))
    manifest, maps = load_bundle(args.bundle, config)
    request = {
        "schema_version": 1,
        "recipe": config,
        "bundle_manifest_sha256": digest(args.bundle / "manifest.json"),
        "bundle_sha256": digest(args.bundle / "bundle.npz"),
        "renderer": {
            p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("materials.py"))
        },
        "blender": {"version": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode()},
        "motion": {
            "frames": args.motion_frames,
            "fps": args.fps,
            "source_fraction": 1.0,
            "semantics": "frozen completed painting; camera only",
        },
    }
    identity = hashlib.sha256(encoded(request)).hexdigest()
    out = args.output.resolve()
    require(not out.exists() or not any(out.iterdir()), "Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "request.json", request)
    write(out / "recipe.json", config)
    write(out / "receipt.json", {"complete": False, "identity_sha256": identity})
    started = time.monotonic()
    scene, geometry, devices, poses = build_scene(
        bpy, config, manifest, maps, args.motion_frames, args.fps
    )
    write(out / "camera.json", poses)
    scene["estuary_depth_identity"] = identity
    scene["estuary_depth_request"] = encoded(request).decode()
    scene["estuary_depth_geometry"] = encoded(geometry).decode()
    for name in ("render.py", "materials.py"):
        bpy.data.texts.new(name).write(Path(__file__).with_name(name).read_text())
    bpy.ops.wm.save_as_mainfile(
        filepath=str(out / "scene.partial.blend"), check_existing=False, compress=True
    )
    (out / "scene.partial.blend").replace(out / "scene.blend")
    artifacts = ["scene.blend", "recipe.json", "camera.json"]
    if args.motion_frames > 1:
        (out / "frames").mkdir()
    for index in range(args.motion_frames):
        scene.frame_set(index + 1)
        bpy.ops.render.render(write_still=False)
        if index == 0:
            save_image(bpy, scene, out / "render.exr", linear=True)
            save_image(bpy, scene, out / "render.png")
            artifacts.extend(("render.exr", "render.png"))
        if args.motion_frames > 1:
            name = f"frames/{index:06d}.png"
            if index == 0:
                shutil.copyfile(out / "render.png", out / name)
            else:
                save_image(bpy, scene, out / name)
            artifacts.append(name)
        print(f"ESTUARY_DEPTH_FRAME {index + 1}/{args.motion_frames}", flush=True)
    require(
        digest(args.bundle / "bundle.npz") == request["bundle_sha256"]
        and digest(args.bundle / "manifest.json") == request["bundle_manifest_sha256"],
        "Bundle changed during rendering",
    )
    write(
        out / "receipt.json",
        {
            "schema_version": 1,
            "complete": True,
            "identity_sha256": identity,
            "geometry": geometry,
            "compute": {"backend": "OPTIX", "devices": devices, "cpu_fallback": False},
            "source": manifest["source"],
            "source_fraction": 1.0,
            "history_fractions": [item["source_fraction"] for item in geometry],
            "motion_frames": args.motion_frames,
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": {name: record(out / name) for name in artifacts},
        },
    )


if __name__ == "__main__":
    main()
