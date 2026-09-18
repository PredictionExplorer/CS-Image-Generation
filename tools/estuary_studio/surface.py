"""Deterministic, physically scaled heightfield painting renderer.

The renderer intersects orthographic rays with the supplied material height, so
camera motion reveals actual parallax and occlusion. Lighting uses an authored
area-light quadrature, anisotropic GGX and heightfield shadows. It is a real-time
surface approximation, not a volumetric fluid renderer or a path tracer. Pigment
reflectance is the project's RGB Kubelka--Munk model, not measured spectral paint.

Input arrays use bottom-up rows over the *entire* guarded canvas. Output is
top-down linear sRGB, ready for a single display encoding by the image writer.
Only a small, stationary support microtexture is procedural; the painting's
structure, relief and directional finish come from the supplied material fields.
"""

from __future__ import annotations

import copy
import math
from functools import wraps
from pathlib import Path

import numpy as np

from tools.estuary.optics import Material

ROOT = Path(__file__).parent
DEFAULTS = {
    "family": "fresco",
    "canvas_width_m": 0.4,
    "domain_scale": 1.6,
    "height_scale": 1.0,
    "exposure": 1.4,
    "ambient": 0.48,
    "key_strength": 1.7,
    "key_elevation_degrees": 42.0,
    "key_azimuth_degrees": -45.0,
    "fill_strength": 0.2,
    "anisotropy": 0.15,
    "roughness_scale": 1.0,
    "roughness_bias": 0.0,
    "grain_um": 2.0,
    "grain_scale_um": 350.0,
    "shadow_strength": 0.6,
    "occlusion_strength": 0.2,
    "tone_map": "reinhard",
    "pigments_srgb": [[0.035, 0.08, 0.32], [0.94, 0.91, 0.82], [0.58, 0.17, 0.075]],
    "substrate_srgb": [0.86, 0.82, 0.72],
    "scattering": [0.3, 8.0, 0.8],
    "layer_scale": 5.0,
}
FAMILY_DEFAULTS = {
    "fresco": {},
    "monotype": {
        "anisotropy": 0.42,
        "grain_um": 0.8,
        "key_elevation_degrees": 48.0,
        "pigments_srgb": [[0.025, 0.065, 0.24], [0.95, 0.92, 0.82], [0.55, 0.12, 0.055]],
        "substrate_srgb": [0.86, 0.82, 0.73],
    },
    "nocturne": {
        "anisotropy": 0.85,
        "grain_um": 0.5,
        "ambient": 0.25,
        "key_strength": 2.1,
        "fill_strength": 0.1,
        "key_elevation_degrees": 65.0,
        "pigments_srgb": [[0.018, 0.03, 0.075], [0.095, 0.09, 0.075], [0.085, 0.025, 0.018]],
        "substrate_srgb": [0.015, 0.022, 0.035],
        "scattering": [0.5, 2.0, 0.8],
    },
}


def _number(value, name, low, high):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite range") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _vector(value, name, low, high):
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f"{name} needs three numbers")
    return [_number(v, name, low, high) for v in value]


def validate_config(value=None):
    """Resolve a complete family preset; unknown or unbounded controls fail."""
    value = {} if value is None else value
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Surface config must be an object containing only known controls")
    family = value.get("family", DEFAULTS["family"])
    if type(family) is not str or family not in FAMILY_DEFAULTS:
        raise ValueError("family must be fresco, monotype or nocturne")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(FAMILY_DEFAULTS[family]))
    result.update(copy.deepcopy(value))
    limits = {
        "canvas_width_m": (0.05, 4.0),
        "domain_scale": (1.25, 3.0),
        "height_scale": (0.0, 10.0),
        "exposure": (0.01, 16.0),
        "ambient": (0.0, 4.0),
        "key_strength": (0.0, 8.0),
        "key_elevation_degrees": (10.0, 90.0),
        "key_azimuth_degrees": (-360.0, 360.0),
        "fill_strength": (0.0, 4.0),
        "anisotropy": (0.0, 0.95),
        "roughness_scale": (-4.0, 4.0),
        "roughness_bias": (-2.0, 2.0),
        "grain_um": (0.0, 30.0),
        "grain_scale_um": (80.0, 4000.0),
        "shadow_strength": (0.0, 1.0),
        "occlusion_strength": (0.0, 1.0),
        "layer_scale": (0.0, 10000.0),
    }
    for key, bounds in limits.items():
        result[key] = _number(result[key], key, *bounds)
    if result["tone_map"] not in ("reinhard", "none"):
        raise ValueError("tone_map must be reinhard or none")
    if type(result["pigments_srgb"]) not in (tuple, list) or len(result["pigments_srgb"]) != 3:
        raise ValueError("pigments_srgb needs exactly three RGB triples")
    result["pigments_srgb"] = [_vector(v, "pigments_srgb", 0, 1) for v in result["pigments_srgb"]]
    result["substrate_srgb"] = _vector(result["substrate_srgb"], "substrate_srgb", 0, 1)
    result["scattering"] = _vector(result["scattering"], "scattering", 1e-6, 1000)
    return result


def camera_basis(tilt_degrees, azimuth_degrees):
    """Fixed world-up orthographic basis, with no artificial image-plane spin."""
    tilt = math.radians(_number(tilt_degrees, "tilt_degrees", -42, 42))
    azimuth = math.radians(_number(azimuth_degrees, "azimuth_degrees", -360, 360))
    view = np.array(
        [math.sin(tilt) * math.cos(azimuth), math.sin(tilt) * math.sin(azimuth), math.cos(tilt)]
    )
    right = np.cross([0.0, 1.0, 0.0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    return np.column_stack([right, up, view])


def validate_fields(fields):
    """Check finite material fields before submitting data to the GPU.

    Heights are physical metres (0..5 cm), not normalized display brightness.
    An axial direction of zero means no coherent orientation. Unit vectors are
    accepted with float32 tolerance; subunit length carries directional coherence.
    """
    names = {"pigment", "height", "wetness", "direction", "roughness", "coverage"}
    if type(fields) is not dict or set(fields) != names:
        raise ValueError(f"Surface fields must contain exactly {sorted(names)}")
    pigment = fields["pigment"]
    if not isinstance(pigment, np.ndarray) or pigment.ndim != 3 or pigment.shape[-1] != 3:
        raise ValueError("pigment must be an H by W by 3 array")
    h, w = pigment.shape[:2]
    if not 4 <= h <= 12288 or not 4 <= w <= 12288 or h * w > 50_000_000:
        raise ValueError("Material grid exceeds supported dimensions")
    bounds = {
        "pigment": (0, 1e6),
        "height": (0, 0.05),
        "wetness": (0, 1),
        "direction": (-1.00001, 1.00001),
        "roughness": (0, 1),
        "coverage": (0, 1),
    }
    arrays = {}
    for name in names:
        array = fields[name]
        shape = (h, w, 3) if name == "pigment" else (h, w, 2) if name == "direction" else (h, w)
        if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"{name} must be float32 with shape {shape}")
        low, high = bounds[name]
        if not np.isfinite(array).all() or np.any(array < low) or np.any(array > high):
            raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        arrays[name] = np.ascontiguousarray(array)
    if np.any(np.sum(arrays["direction"] ** 2, axis=-1) > 1.0001):
        raise ValueError("direction must be an axial unit vector or subunit coherence vector")
    return arrays


def _current(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.ctx is None:
            raise RuntimeError("The painting surface context has been closed")
        with self.ctx:
            return method(self, *args, **kwargs)

    return wrapped


class Surface:
    """Dedicated hardware context; safe to interleave with simulation contexts."""

    def __init__(self, config=None, backend="egl"):
        import moderngl

        self.config = validate_config(config)
        self.ctx = moderngl.create_standalone_context(require=430, backend=backend)
        self._textures = []
        self._target = self._framebuffer = self._vao = self._quad = self._program = None
        self._grid_size = self._target_size = None
        self._scaled_max_height = None
        try:
            if any(
                v in self.ctx.info["GL_RENDERER"].lower()
                for v in ("llvmpipe", "softpipe", "software")
            ):
                raise RuntimeError("Painting experiments require a hardware GPU")
            self.metadata = {
                "renderer": self.ctx.info["GL_RENDERER"],
                "vendor": self.ctx.info["GL_VENDOR"],
                "version": self.ctx.info["GL_VERSION"],
                "moderngl": moderngl.__version__,
                "backend": backend,
                "geometry": "orthographic heightfield ray intersection, 128 steps and 8 bisections",
                "lighting": (
                    "deterministic 5-point area-light quadrature; "
                    "anisotropic GGX; heightfield shadows"
                ),
                "optics": "authored RGB finite-layer Kubelka-Munk; not measured spectral paint",
                "limitations": (
                    "single-valued surface; approximate direct lighting; "
                    "no overhangs, refraction or multiple scattering"
                ),
                "output": (
                    "top-down float32 linear sRGB; "
                    "extended luminance Reinhard (white=4) or unbounded HDR"
                ),
                "microtexture": (
                    "stationary support grain, physical metres, pixel-footprint filtered"
                ),
                "cross_device_pixel_identity": False,
                "config": copy.deepcopy(self.config),
            }
            self._program = self.ctx.program(
                vertex_shader=(ROOT / "shaders/surface.vert.glsl").read_text(),
                fragment_shader=(ROOT / "shaders/surface.frag.glsl").read_text(),
            )
            self._quad = self.ctx.buffer(
                np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
            )
            self._vao = self.ctx.vertex_array(self._program, [(self._quad, "2f", "in_position")])
            self._bind_config()
        except BaseException:
            self.close()
            raise

    def _bind_config(self):
        c, p = self.config, self._program
        material = Material(
            pigments_srgb=c["pigments_srgb"],
            substrate_srgb=c["substrate_srgb"],
            scattering=c["scattering"],
            layer_scale=c["layer_scale"],
            grain=0,
        )
        for name, value in material.uniforms().items():
            if name in p:
                p[name].value = value
        for name in (
            "height_scale",
            "ambient",
            "key_strength",
            "fill_strength",
            "anisotropy",
            "roughness_scale",
            "roughness_bias",
            "shadow_strength",
            "occlusion_strength",
            "exposure",
        ):
            p["u_" + name].value = c[name]
        p["u_grain_height"].value = c["grain_um"] * 1e-6
        p["u_grain_scale"].value = c["grain_scale_um"] * 1e-6
        p["u_tone_map"].value = int(c["tone_map"] == "reinhard")
        p["u_family"].value = {"fresco": 0, "monotype": 1, "nocturne": 2}[c["family"]]
        elevation, azimuth = map(
            math.radians, (c["key_elevation_degrees"], c["key_azimuth_degrees"])
        )
        p["u_key_direction"].value = (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
        for unit, name in enumerate(("u_paint", "u_geometry", "u_finish")):
            p[name].value = unit

    @_current
    def render(self, fields=None, size=(1280, 960), tilt_degrees=0.0, azimuth_degrees=-30.0):
        """Render supplied fields, or explicitly reuse the last uploaded fields.

        ``fields=None`` is for a frozen-painting camera orbit or a larger still.
        No CPU arrays are retained and no object-identity caching is performed:
        mutating an earlier caller array cannot silently alter the uploaded paint.
        """
        import moderngl

        if fields is None:
            if self._scaled_max_height is None:
                raise ValueError("Upload material fields before rendering a frozen surface")
            arrays = None
            grid_width, grid_height = self._grid_size
            scaled_max = self._scaled_max_height
        else:
            arrays = validate_fields(fields)
            grid_height, grid_width = arrays["height"].shape
            scaled_max = float(arrays["height"].max()) * self.config["height_scale"]
        if (
            type(size) not in (list, tuple)
            or len(size) != 2
            or any(type(v) is not int or not 4 <= v <= 12288 for v in size)
        ):
            raise ValueError("Output size needs two integer dimensions in [4, 12288]")
        width, height = size
        if width * height > 50_000_000:
            raise ValueError("Output exceeds the pixel budget")
        if width * grid_height != height * grid_width:
            raise ValueError("Output and material grid must have exactly the same aspect ratio")
        basis = camera_basis(tilt_degrees, azimuth_degrees)
        if max(width, height, grid_width, grid_height) > self.ctx.info["GL_MAX_TEXTURE_SIZE"]:
            raise ValueError("Texture dimensions exceed GPU limits")
        if scaled_max > self.config["canvas_width_m"] * 0.15:
            raise ValueError("Scaled relief must remain below 15% of visible canvas width")
        visible_width = self.config["canvas_width_m"]
        visible_size = (visible_width, visible_width * height / width)
        full_size = np.asarray(visible_size) * self.config["domain_scale"]
        # The picture must fill the view at every legal height. Reject a camera
        # that could expose a guard-band boundary instead of inventing a border.
        for x in (-visible_size[0] / 2, visible_size[0] / 2):
            for y in (-visible_size[1] / 2, visible_size[1] / 2):
                origin = basis[:, 0] * x + basis[:, 1] * y
                for z in (0, scaled_max):
                    point = origin + basis[:, 2] * ((z - origin[2]) / basis[2, 2])
                    if np.any(np.abs(point[:2]) > full_size / 2):
                        raise ValueError(
                            "Camera leaves the guarded painting; reduce tilt or relief"
                        )
        if arrays is not None:
            if self._grid_size != (grid_width, grid_height):
                for texture in self._textures:
                    texture.release()
                self._textures = [
                    self.ctx.texture((grid_width, grid_height), n, dtype="f4") for n in (3, 4, 2)
                ]
                for texture in self._textures:
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    texture.repeat_x = texture.repeat_y = False
                self._grid_size = (grid_width, grid_height)
            geometry = np.empty((grid_height, grid_width, 4), dtype="f4")
            geometry[..., 0], geometry[..., 1] = arrays["height"], arrays["wetness"]
            geometry[..., 2], geometry[..., 3] = arrays["roughness"], arrays["coverage"]
            for texture, array in zip(
                self._textures, (arrays["pigment"], geometry, arrays["direction"]), strict=True
            ):
                texture.write(array)
            self._scaled_max_height = scaled_max
        for unit, texture in enumerate(self._textures):
            texture.use(unit)
        if self._target_size != tuple(size):
            if self._framebuffer is not None:
                self._framebuffer.release()
                self._target.release()
            self._target = self.ctx.texture(tuple(size), 3, dtype="f4")
            self._framebuffer = self.ctx.framebuffer(color_attachments=[self._target])
            self._target_size = tuple(size)
        p = self._program
        p["u_visible_size"].value = visible_size
        p["u_full_size"].value = tuple(full_size)
        p["u_grid_size"].value = (grid_width, grid_height)
        p["u_output_size"].value = tuple(size)
        p["u_max_height"].value = max(scaled_max, 1e-7)
        p["u_camera_right"].value = tuple(basis[:, 0])
        p["u_camera_up"].value = tuple(basis[:, 1])
        p["u_camera_view"].value = tuple(basis[:, 2])
        self._framebuffer.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._vao.render(mode=moderngl.TRIANGLE_STRIP)
        result = (
            np.frombuffer(self._target.read(alignment=1), dtype="f4")
            .reshape(height, width, 3)[::-1]
            .copy()
        )
        if not np.isfinite(result).all() or np.any(result < 0):
            raise FloatingPointError("Surface rendering produced nonfinite or negative radiance")
        return result

    def close(self):
        """Release owned GPU resources; repeated cleanup is safe."""
        if self.ctx is None:
            return
        with self.ctx:
            for resource in (
                self._framebuffer,
                self._target,
                *self._textures,
                self._vao,
                self._quad,
                self._program,
            ):
                if resource is not None:
                    resource.release()
        self.ctx.release()
        self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
