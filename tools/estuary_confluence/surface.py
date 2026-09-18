"""Phase-resolved pigment optics on a physically scaled fresco heightfield.

The geometry/light adapter derives from estuary_studio's explicitly approximate
heightfield renderer. Color is evaluated once on the material grid and cached on
GPU; camera changes cannot alter pigment mixing or temporal layers. Inputs are
bottom-up fullguard fields, outputs top-down bounded linear RGB. Illumination is
an analytic direct-light approximation; the material is not a volumetric fluid.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from tools.estuary_studio.surface import _current, _number, camera_basis

from .optics import palette_coefficients

ROOT = Path(__file__).parent
DEFAULTS = {
    "mode": "layered",
    "mix_control": 1.0,
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
    "layer_scale": 5.0,
}


def validate_config(value=None):
    value = {} if value is None else value
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Surface config must contain only known controls")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))
    if result["mode"] not in ("homogeneous", "layered"):
        raise ValueError("mode must be homogeneous or layered")
    if result["tone_map"] not in ("reinhard", "none"):
        raise ValueError("tone_map must be reinhard or none")
    limits = {
        "mix_control": (0, 1),
        "canvas_width_m": (0.05, 4),
        "domain_scale": (1.25, 3),
        "height_scale": (0, 10),
        "exposure": (0.01, 16),
        "ambient": (0, 4),
        "key_strength": (0, 8),
        "key_elevation_degrees": (10, 90),
        "key_azimuth_degrees": (-360, 360),
        "fill_strength": (0, 4),
        "anisotropy": (0, 0.95),
        "roughness_scale": (-4, 4),
        "roughness_bias": (-2, 2),
        "grain_um": (0, 30),
        "grain_scale_um": (80, 4000),
        "shadow_strength": (0, 1),
        "occlusion_strength": (0, 1),
        "layer_scale": (0, 10000),
    }
    for name, bounds in limits.items():
        result[name] = _number(result[name], name, *bounds)
    return result


def validate_fields(fields, pigment_count=None):
    names = {
        "pigment",
        "mobile",
        "deposit",
        "underpaint",
        "height",
        "wetness",
        "mixing",
        "direction",
        "roughness",
        "coverage",
    }
    if type(fields) is not dict or set(fields) != names:
        raise ValueError(f"Surface fields must contain exactly {sorted(names)}")
    pigment = fields["pigment"]
    if not isinstance(pigment, np.ndarray) or pigment.ndim != 3 or pigment.shape[-1] not in (4, 6):
        raise ValueError("pigment must be an H by W by N array; N=4 or 6")
    h, w, count = pigment.shape
    if pigment_count is not None and count != pigment_count:
        raise ValueError("Material pigment count must match the archived palette")
    if not 4 <= h <= 12288 or not 4 <= w <= 12288 or h * w > 50_000_000:
        raise ValueError("Material grid exceeds supported dimensions")
    bounds = {
        "pigment": (0, 1e6),
        "mobile": (0, 1e6),
        "deposit": (0, 1e6),
        "underpaint": (0, 1e6),
        "height": (0, 0.05),
        "wetness": (0, 1),
        "mixing": (0, 1),
        "direction": (-1.00001, 1.00001),
        "roughness": (0, 1),
        "coverage": (0, 1),
    }
    arrays = {}
    for name in names:
        array = fields[name]
        shape = (
            (h, w, count)
            if name in ("pigment", "mobile", "deposit", "underpaint")
            else (h, w, 2)
            if name == "direction"
            else (h, w)
        )
        if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"{name} must be float32 with shape {shape}")
        low, high = bounds[name]
        if not np.isfinite(array).all() or np.any(array < low) or np.any(array > high):
            raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        arrays[name] = np.ascontiguousarray(array)
    if np.any(np.sum(arrays["direction"] ** 2, axis=-1) > 1.0001):
        raise ValueError("direction must be an axial unit vector or subunit coherence vector")
    # A stale total is an archival/rendering bug, not an alternate optical input.
    if not np.allclose(
        arrays["pigment"],
        arrays["underpaint"] + arrays["deposit"] + arrays["mobile"],
        rtol=2e-6,
        atol=1e-7,
    ):
        raise ValueError("pigment must equal the sum of the three real phases")
    return arrays


class Surface:
    """Dedicated hardware context; safe to interleave with simulation contexts."""

    def __init__(self, config, palette, backend="egl"):
        import moderngl

        self.config = validate_config(config)
        self.palette = copy.deepcopy(palette)
        self._ratios, self._scattering, self._substrate = palette_coefficients(self.palette)
        self._pigment_count = len(self._scattering)
        self.ctx = moderngl.create_standalone_context(require=430, backend=backend)
        self._textures = []
        self._phase_texture = self._mix_texture = self._optics_program = None
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
                "optics": (
                    "authored RGB finite-layer Kubelka-Munk; "
                    "phase-resolved reflection/transmission stacking"
                ),
                "mixing": (
                    "bounded intimate and unresolved areal operator mixture; "
                    "statistically independent layer patches; "
                    "not resolved filaments or measured spectral paint"
                ),
                "limitations": (
                    "single-valued surface; approximate direct lighting; "
                    "no overhangs, refraction or multiple scattering"
                ),
                "output": (
                    "top-down float32 linear sRGB; "
                    "extended luminance Reinhard (white=4) or clipped linear; bounded [0,1]"
                ),
                "microtexture": (
                    "stationary support grain, physical metres, pixel-footprint filtered"
                ),
                "cross_device_pixel_identity": False,
                "config": copy.deepcopy(self.config),
                "pigment_count": self._pigment_count,
                "palette": copy.deepcopy(self.palette),
            }
            self._program = self.ctx.program(
                vertex_shader=(ROOT / "shaders/surface.vert.glsl").read_text(),
                fragment_shader=(ROOT / "shaders/surface.frag.glsl").read_text(),
            )
            self._quad = self.ctx.buffer(
                np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
            )
            self._vao = self.ctx.vertex_array(self._program, [(self._quad, "2f", "in_position")])
            optics_source = (ROOT / "shaders/optics.comp.glsl").read_text()
            optics_source = optics_source.replace(
                "#version 430 core",
                f"#version 430 core\n#define PIGMENT_COUNT {self._pigment_count}",
                1,
            )
            self._optics_program = self.ctx.compute_shader(optics_source)
            self._bind_config()
        except BaseException:
            self.close()
            raise

    def _bind_config(self):
        c, p = self.config, self._program
        p["u_substrate"].value = tuple(self._substrate)
        optics = self._optics_program
        optics["u_phases"].value = 3
        optics["u_mixing"].value = 4
        optics["u_substrate"].value = tuple(self._substrate)
        optics["u_ratios"].write(self._ratios.astype("f4").tobytes())
        optics["u_scattering"].write(self._scattering.astype("f4").tobytes())
        optics["u_layer_scale"].value = c["layer_scale"]
        optics["u_mix_control"].value = c["mix_control"]
        optics["u_layered"].value = int(c["mode"] == "layered")
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
            arrays = validate_fields(fields, self._pigment_count)
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
                    self.ctx.texture((grid_width, grid_height), n, dtype="f4") for n in (4, 4, 2)
                ]
                for texture in self._textures:
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    texture.repeat_x = texture.repeat_y = False
                if self._phase_texture is not None:
                    self._phase_texture.release()
                    self._mix_texture.release()
                groups = (self._pigment_count + 3) // 4
                self._phase_texture = self.ctx.texture_array(
                    (grid_width, grid_height, groups * 3), 4, dtype="f4"
                )
                self._mix_texture = self.ctx.texture((grid_width, grid_height), 1, dtype="f4")
                self._grid_size = (grid_width, grid_height)
            geometry = np.empty((grid_height, grid_width, 4), dtype="f4")
            geometry[..., 0], geometry[..., 1] = arrays["height"], arrays["wetness"]
            geometry[..., 2], geometry[..., 3] = arrays["roughness"], arrays["coverage"]
            self._textures[1].write(geometry)
            self._textures[2].write(arrays["direction"])
            self._upload_optics(arrays, grid_width, grid_height)
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

    def _upload_optics(self, arrays, width, height):
        # One RGBA group per four pigments per true material phase. The only
        # color readback is the final display-sized image, never this prepass.
        groups = (self._pigment_count + 3) // 4
        packed = np.zeros((groups * 3, height, width, 4), dtype="f4")
        for phase_index, name in enumerate(("underpaint", "deposit", "mobile")):
            for pigment_index in range(self._pigment_count):
                packed[phase_index * groups + pigment_index // 4, ..., pigment_index % 4] = arrays[
                    name
                ][..., pigment_index]
        self._phase_texture.write(packed)
        self._mix_texture.write(arrays["mixing"])
        self._phase_texture.use(3)
        self._mix_texture.use(4)
        self._textures[0].bind_to_image(0, read=False, write=True)
        self._optics_program.run(group_x=(width + 15) // 16, group_y=(height + 15) // 16)
        self.ctx.memory_barrier()

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
                self._phase_texture,
                self._mix_texture,
                self._optics_program,
            ):
                if resource is not None:
                    resource.release()
        self.ctx.release()
        self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
