"""Phase-resolved pigment optics on a physically scaled fresco heightfield.

The geometry/light adapter derives from estuary_studio's explicitly approximate
heightfield renderer. Color is evaluated once on the material grid and cached on
GPU; camera changes cannot alter pigment mixing or temporal layers. Inputs are
bottom-up fullguard fields, outputs top-down bounded linear RGB. Illumination is
an analytic direct-light approximation; the material is not a volumetric fluid.

The opt-in crisp finish is an authored print interpretation: actual pigment mass
sets a hard silhouette, and a single optical mass reference gives filled colors
without a dilute fringe. Geometry and illumination stay inside that silhouette;
the outside is an unlit constant ground. Antialiasing spans one output pixel,
not a physical feather or animated threshold. The archived state is untouched.

The opt-in glazed finish retains that silhouette with bounded actual interior
optical thickness. Higher-mass banks carry most relief; wet upper-layer coverage
and pigment scattering control restrained sheen. These are authored material
interpretations, not a three-dimensional fluid.

Height scaling is an explicit appearance interpretation (0..100), not additional
simulated paint. The actual scaled extent still must fit within 15% of the
visible canvas width and the guarded camera volume. Optional packing conserves
this displayed native-grid volume, rather than the unchanged archived geometry.
"""

from __future__ import annotations

import copy
import math
from functools import wraps
from pathlib import Path

import numpy as np

from tools.estuary.optics import srgb_to_linear
from tools.estuary_studio.surface import _number, camera_basis

from .gpu_frame import GPUFrame
from .interaction import BASE_FIELDS, FIELD_NAMES
from .interaction import VERSION as INTERACTION_VERSION
from .optics import palette_coefficients
from .packing import (
    DEFAULT_LENGTH_UM,
    FILM_FRACTION,
    LENGTH_BOUNDS_UM,
    PackingRelief,
    validate_directional,
)
from .packing import VERSION as PACKING_VERSION
from .palette import SUPPORTED_CHROMATIC_COUNTS

ROOT = Path(__file__).parent
DEFAULTS = {
    "mode": "layered",
    "optics_model": "rgb",
    "finish": "fresco",
    "paint_mass_threshold": 0.002,
    "paint_mass_reference": 0.15,
    "ground_srgb": [1.0, 1.0, 1.0],
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
GLAZE_DEFAULTS = {
    "glaze_min_mass_ratio": 0.35,
    "glaze_max_mass_ratio": 2.5,
    "glaze_relief_strength": 0.8,
}
INTERACTION_DEFAULTS = {"silk_strength": 0.65, "grain_strength": 0.45}
PACKING_DEFAULTS = {"packing_strength": 0.0, "packing_length_um": DEFAULT_LENGTH_UM}


def interaction_enabled(config):
    """Zero appearance strengths compile the original rendering path unchanged."""
    controls = config.get("interaction")
    return controls is not None and any(
        controls.get(name, 0) for name in ("silk_strength", "grain_strength", "packing_strength")
    )


def validate_config(value=None):
    value = {} if value is None else value
    if type(value) is not dict or set(value) - (
        set(DEFAULTS) | set(GLAZE_DEFAULTS) | {"interaction"}
    ):
        raise ValueError("Surface config must contain only known controls")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))
    if result.get("interaction") is not None:
        controls = result["interaction"]
        if type(controls) is not dict or set(controls) - (
            set(INTERACTION_DEFAULTS)
            | set(PACKING_DEFAULTS)
            | {"grain_contrast", "directional_relief"}
        ):
            raise ValueError(
                "Surface interaction must contain only silk, grain and packing controls"
            )
        result["interaction"] = {
            name: _number(controls.get(name, default), f"interaction.{name}", 0, 1)
            for name, default in INTERACTION_DEFAULTS.items()
        }
        if "grain_contrast" in controls:
            result["interaction"]["grain_contrast"] = _number(
                controls["grain_contrast"], "interaction.grain_contrast", 1, 8
            )
        if set(controls) & set(PACKING_DEFAULTS):
            result["interaction"].update(
                {
                    "packing_strength": _number(
                        controls.get("packing_strength", 0), "interaction.packing_strength", 0, 1
                    ),
                    "packing_length_um": _number(
                        controls.get("packing_length_um", DEFAULT_LENGTH_UM),
                        "interaction.packing_length_um",
                        *LENGTH_BOUNDS_UM,
                    ),
                }
            )
        directional = validate_directional(controls.get("directional_relief"))
        if directional is not None:
            if result["interaction"].get("packing_strength", 0) <= 0:
                raise ValueError("Directional relief requires positive packing_strength")
            result["interaction"]["directional_relief"] = directional
    # Do not add controls to old archived surface dictionaries.
    if result["finish"] == "glazed":
        result = {**GLAZE_DEFAULTS, **result}
    if result["mode"] not in ("homogeneous", "layered"):
        raise ValueError("mode must be homogeneous or layered")
    if result["optics_model"] not in ("rgb", "spectral"):
        raise ValueError("optics_model must be rgb or spectral")
    if result["finish"] not in ("fresco", "crisp", "glazed"):
        raise ValueError("finish must be fresco, crisp or glazed")
    ground = result["ground_srgb"]
    if type(ground) not in (list, tuple) or len(ground) != 3:
        raise ValueError("ground_srgb needs three display-sRGB values")
    result["ground_srgb"] = [_number(value, "ground_srgb", 0, 1) for value in ground]
    if result["tone_map"] not in ("reinhard", "none"):
        raise ValueError("tone_map must be reinhard or none")
    limits = {
        "paint_mass_threshold": (1e-6, 1),
        "paint_mass_reference": (1e-4, 10),
        "mix_control": (0, 1),
        "canvas_width_m": (0.05, 4),
        "domain_scale": (1.25, 3),
        "height_scale": (0, 100),
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
    for name, bounds in {
        "glaze_min_mass_ratio": (0.01, 1),
        "glaze_max_mass_ratio": (1, 8),
        "glaze_relief_strength": (0, 1),
    }.items():
        if name in result:
            result[name] = _number(result[name], name, *bounds)
    return result


def validate_fields(fields, pigment_count=None):
    names = set(BASE_FIELDS)
    extensions = (
        set(FIELD_NAMES),
        {"trait_upper", "trait_lower"},
        {"structure_upper", "structure_lower"},
    )
    if (
        type(fields) is not dict
        or not names <= set(fields)
        or set(fields) - names - set.union(*extensions)
        or any(set(fields) & group and not group <= set(fields) for group in extensions)
        or ("trait_upper" in fields and not set(FIELD_NAMES) <= set(fields))
    ):
        raise ValueError(
            "Surface fields require the base material and either all or no interaction fields; "
            "material traits and structure fields must occur as complete pairs"
        )
    has_interaction = set(FIELD_NAMES).issubset(fields)
    pigment = fields["pigment"]
    if (
        not isinstance(pigment, np.ndarray)
        or pigment.ndim != 3
        or pigment.shape[-1] - 1 not in SUPPORTED_CHROMATIC_COUNTS
    ):
        raise ValueError(
            "pigment must be an H by W by N array; N is a supported color count plus chalk"
        )
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
    if has_interaction:
        for name in FIELD_NAMES:
            array = fields[name]
            shape = (h, w, 2 if name.startswith("origin_") else 4)
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.shape != shape
            ):
                raise ValueError(f"{name} must be float32 with shape {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            if name.startswith("origin_"):
                if np.any(np.abs(array) > 1e6):
                    raise ValueError(f"{name} exceeds the supported coordinate bounds")
            elif (
                np.any(array[..., (0, 3)] < 0)
                or np.any(array[..., (0, 3)] > 1)
                or np.any(np.abs(array[..., 1:3]) > 1.00001)
                or np.any(np.linalg.norm(array[..., 1:3], axis=-1) > array[..., 0] + 1e-5)
            ):
                raise ValueError(
                    f"{name} must have bounded contact, fabric and aggregate fractions"
                )
            arrays[name] = np.ascontiguousarray(array)
    for prefix, channels, low, high in (("trait", 2, -1, 1), ("structure", None, 0, 1)):
        for layer in ("upper", "lower"):
            name = f"{prefix}_{layer}"
            if name not in fields:
                continue
            array = fields[name]
            shape = (h, w, channels) if channels is not None else (h, w)
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.shape != shape
                or not np.isfinite(array).all()
                or np.any(array < low)
                or np.any(array > high)
            ):
                raise ValueError(
                    f"{name} must be finite float32 with shape {shape} in [{low}, {high}]"
                )
            arrays[name] = np.ascontiguousarray(array)
    return arrays


def _current(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.ctx is None:
            raise RuntimeError("The painting surface context has been closed")
        if self._borrowed_frame is not None:
            self._borrowed_frame.require_owner()
        with self.ctx:
            return method(self, *args, **kwargs)

    return wrapped


class Surface:
    """Own a hardware context or safely borrow an Engine's native context."""

    def __init__(self, config, palette, backend="egl", *, spectral=None, gpu_frame=None):
        import moderngl

        self.config = validate_config(config)
        self._interaction_enabled = interaction_enabled(self.config)
        self._packing_strength = (self.config.get("interaction") or {}).get("packing_strength", 0)
        self._directional_relief = (self.config.get("interaction") or {}).get("directional_relief")
        self._grain_contrast = (self.config.get("interaction") or {}).get("grain_contrast", 1)
        self.palette = copy.deepcopy(palette)
        self._ratios, self._scattering, self._substrate = palette_coefficients(self.palette)
        self._pigment_count = len(self._scattering)
        self.spectral = None
        if self.config["optics_model"] == "spectral":
            from .spectral import build_spectral_material, validate_spectral_material

            self.spectral = (
                build_spectral_material(self.palette)
                if spectral is None
                else validate_spectral_material(spectral, self.palette)
            )
        elif spectral is not None:
            raise ValueError("RGB optics cannot consume a spectral material archive")
        self._borrowed_frame = None
        self._owns_context = gpu_frame is None
        if gpu_frame is not None:
            if not isinstance(gpu_frame, GPUFrame):
                raise ValueError("A borrowed surface requires an Engine GPUFrame")
            gpu_frame.validate()
            if gpu_frame.pigment_count != self._pigment_count:
                raise ValueError("GPU frame and palette have different pigment counts")
            self._borrowed_frame = gpu_frame
            self.ctx = gpu_frame.context
        else:
            self.ctx = moderngl.create_standalone_context(require=430, backend=backend)
        self._textures = []
        self._gpu_pack = self._gpu_summary = self._gpu_reduce = self._gpu_reduced = None
        self._gpu_reduced_size = None
        self._phase_texture = self._mix_texture = self._optics_program = None
        self._target = self._framebuffer = self._vao = self._quad = self._program = None
        self._grid_size = self._target_size = None
        self._interaction_texture = None
        self._packing_relief = None
        self._gpu_pack_interaction = None
        self._scaled_max_height = None
        try:
            with self.ctx:
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
                    "context_ownership": "owned"
                    if self._owns_context
                    else "borrowed simulation context",
                    "native_gpu_capture": (
                        "read-only live phase textures copied to Surface-owned staging; "
                        "GPU geometry and "
                        "existing pigment optics; optional GPU 2x2 linear display-RGB filtering; "
                        + (
                            "only 12 bytes of validation/height-bound summaries and final image "
                            "return to CPU"
                            if self._packing_strength > 0
                            else "only an 8-byte validation summary and final image return to CPU"
                        )
                    )
                    if not self._owns_context
                    else None,
                    "geometry": (
                        "orthographic heightfield ray intersection, 128 steps and 8 bisections"
                    ),
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
                    "finish": (
                        "fresco: physical phase amount controls optical thickness; "
                        "crisp: total-mass contour with one-pixel antialiasing and "
                        "mass-normalized filled pigment colors over an unlit constant ground; "
                        "crisp deliberately removes dilution cues without changing simulation state"
                    ),
                    "mass_units": (
                        "stored pigment concentration per material area, before RGB optics"
                    ),
                    "cross_device_pixel_identity": False,
                    "config": copy.deepcopy(self.config),
                    "pigment_count": self._pigment_count,
                    "palette": copy.deepcopy(self.palette),
                    "optics_model": self.config["optics_model"],
                    "spectral_material_identity": None
                    if self.spectral is None
                    else self.spectral["identity_sha256"],
                }
                if self.spectral is not None:
                    self.metadata["optics"] = (
                        "38-band finite-layer Kubelka-Munk, D65/CIE 1931 integration; "
                        "synthetic RGB-reconstructed reflectance, not measured pigments"
                    )
                if (
                    self.spectral is not None
                    and self.spectral["version"] == "confluence-spectral-v2"
                ):
                    self.metadata["mixing"] = (
                        "recorded mixedness blends intimate spectral reflection with "
                        "independent pigment-column reflection composed over the lower layer; "
                        "no resolved microscopic strands or measured mixing calibration"
                    )
                if self.config["finish"] == "glazed":
                    self.metadata["finish"] = (
                        "glazed: the crisp real-mass silhouette and flat exterior ground; "
                        "actual interior optical mass clamped to explicit reference ratios, "
                        "preserving pigment and phase fractions; concentration-selected relief "
                        "and wet upper-layer sheen; no change to archived material fields"
                    )
                if self._interaction_enabled:
                    self.metadata["interaction"] = {
                        "version": INTERACTION_VERSION,
                        "silk": "bounded GGX anisotropy and roughness from frozen material fabric",
                        "grain": (
                            "aggregate-fraction microfacet roughness and white-albedo EON "
                            "rough-diffuse angular redistribution; unresolved optical granulation"
                        ),
                        "limits": (
                            "no added pigment or procedural marks; optional packing "
                            "redistributes displayed height without adding native-grid volume; "
                            "no measured-particle claim"
                        ),
                        "capture": (
                            "Surface-owned copy; camera movement does not evolve material history"
                        ),
                    }
                if self._packing_strength > 0:
                    self.metadata["packing"] = {
                        "version": PACKING_VERSION,
                        "film_fraction": FILM_FRACTION,
                        "volume": "redistributed displayed paint thickness; zero substrate only",
                        "solver": (
                            "28 bounded multiscale pair relaxations plus preparation/finalization"
                        ),
                        "limits": (
                            "finite packing equilibrium; no solvent clock or pigment transport"
                        ),
                        "memory_bytes_per_native_pixel": 12,
                        "maximum_coupling_cells": 16,
                        "source_material_unchanged": True,
                    }
                    if self._directional_relief is not None:
                        self.metadata["packing"].update(
                            directional_relief={
                                "config": copy.deepcopy(self._directional_relief),
                                "affinity": (
                                    "contact/fabric-weighted aggregate smoothing "
                                    "along material fabric"
                                ),
                                "paths": "four directions with wholly occupied supercover paths",
                                "response": (
                                    "finite authored affinity reconstruction; "
                                    "no new pigment or motion"
                                ),
                            },
                            solver=(
                                "28 affinity passes and 28 packing passes "
                                "plus preparation/finalization"
                            ),
                            memory_bytes_per_native_pixel=20,
                        )
                if self._interaction_enabled and self._grain_contrast > 1:
                    self.metadata["interaction"]["grain_response"] = {
                        "formula": "g^c / (g^c + (1-g)^c)",
                        "contrast": self._grain_contrast,
                        "scope": (
                            "authored percolation-inspired optics; no particle-connectivity model"
                        ),
                        "raw_aggregate_and_packing_unchanged": True,
                    }
                self._program = self.ctx.program(
                    vertex_shader=(ROOT / "shaders/surface.vert.glsl").read_text(),
                    fragment_shader=(ROOT / "shaders/surface.frag.glsl")
                    .read_text()
                    .replace(
                        "// INTERACTION_SCATTERING",
                        (ROOT / "shaders/scattering.glsl").read_text()
                        if self._interaction_enabled
                        else "",
                    )
                    .replace(
                        "#version 430 core",
                        f"#version 430 core\n#define PIGMENT_COUNT {self._pigment_count}"
                        + ("\n#define INTERACTION_SURFACE" if self._interaction_enabled else "")
                        + (
                            "\n#define GRAIN_CONTRAST"
                            if self._interaction_enabled and self._grain_contrast > 1
                            else ""
                        )
                        + ("\n#define PACKING_SURFACE" if self._packing_strength > 0 else ""),
                        1,
                    ),
                )
                self._quad = self.ctx.buffer(
                    np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
                )
                self._vao = self.ctx.vertex_array(
                    self._program, [(self._quad, "2f", "in_position")]
                )
                if self.spectral is None:
                    optics_file = "optics.comp.glsl"
                else:
                    optics_file = "optics-spectral.comp.glsl"
                optics_source = (ROOT / "shaders" / optics_file).read_text()
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
        if self._interaction_enabled:
            p["u_interaction"].value = 5
            p["u_silk_strength"].value = c["interaction"]["silk_strength"]
            p["u_grain_strength"].value = c["interaction"]["grain_strength"]
        if self._packing_strength > 0:
            p["u_packing"].value = 6
        if self._interaction_enabled and self._grain_contrast > 1:
            p["u_grain_contrast"].value = self._grain_contrast
        p["u_substrate"].value = tuple(self._substrate)
        crisp = int(c["finish"] in ("crisp", "glazed"))
        glazed = int(c["finish"] == "glazed")
        p["u_crisp"].value = crisp
        p["u_glazed"].value = glazed
        p["u_phases"].value = 3
        p["u_scattering"].write(self._scattering.astype("f4").tobytes())
        p["u_mass_reference"].value = c["paint_mass_reference"]
        p["u_glaze_relief_strength"].value = c.get(
            "glaze_relief_strength", GLAZE_DEFAULTS["glaze_relief_strength"]
        )
        p["u_mass_threshold"].value = c["paint_mass_threshold"]
        p["u_ground"].value = tuple(srgb_to_linear(c["ground_srgb"]))
        optics = self._optics_program
        optics["u_phases"].value = 3
        optics["u_mixing"].value = 4
        if self.spectral is None:
            optics["u_substrate"].value = tuple(self._substrate)
            optics["u_ratios"].write(self._ratios.astype("f4").tobytes())
        else:
            from .spectral import to_linear_rgb

            spectra = self.spectral
            ratio = np.asarray(spectra["absorption"]) / np.asarray(spectra["scattering"])
            optics["u_spectral_ks"].write(ratio.astype("f4").tobytes())
            optics["u_substrate_spectrum"].write(
                np.asarray(spectra["substrate_reflectance"], dtype="f4").tobytes()
            )
            optics["u_rgb_weights"].write(
                np.asarray(spectra["linear_srgb_weights_d65"], dtype="f4").tobytes()
            )
            optics["u_substrate"].value = tuple(
                to_linear_rgb(spectra["substrate_reflectance"], spectra)
            )
        optics["u_scattering"].write(self._scattering.astype("f4").tobytes())
        optics["u_layer_scale"].value = c["layer_scale"]
        optics["u_mix_control"].value = c["mix_control"]
        optics["u_layered"].value = int(c["mode"] == "layered")
        optics["u_crisp"].value = crisp
        optics["u_glazed"].value = glazed
        for name in ("glaze_min_mass_ratio", "glaze_max_mass_ratio"):
            optics["u_" + name].value = c.get(name, GLAZE_DEFAULTS[name])
        optics["u_mass_reference"].value = c["paint_mass_reference"]
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
        return self._render_buffer(fields, size, tilt_degrees, azimuth_degrees, readback=True)

    def _render_buffer(self, fields, size, tilt_degrees, azimuth_degrees, *, readback):
        import moderngl

        if fields is None:
            if self._scaled_max_height is None:
                raise ValueError("Upload material fields before rendering a frozen surface")
            arrays = None
            grid_width, grid_height = self._grid_size
            scaled_max = self._scaled_max_height
        else:
            arrays = validate_fields(fields, self._pigment_count)
            if self._interaction_enabled and "interaction_upper" not in arrays:
                raise ValueError("Interaction appearance requires archived material history")
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
        visible_width = self.config["canvas_width_m"]
        visible_size = (visible_width, visible_width * height / width)
        full_size = np.asarray(visible_size) * self.config["domain_scale"]
        self._validate_extent(scaled_max, basis, visible_size, full_size)
        if arrays is not None:
            self._scaled_max_height = None
            self._ensure_grid(grid_width, grid_height)
            geometry = np.empty((grid_height, grid_width, 4), dtype="f4")
            geometry[..., 0], geometry[..., 1] = arrays["height"], arrays["wetness"]
            geometry[..., 2], geometry[..., 3] = arrays["roughness"], arrays["coverage"]
            self._textures[1].write(geometry)
            self._textures[2].write(arrays["direction"])
            self._upload_optics(arrays, grid_width, grid_height)
            if self._interaction_enabled:
                self._ensure_interaction_grid()
                self._interaction_texture.write(
                    np.stack((arrays["interaction_upper"], arrays["interaction_lower"]))
                )
            self._prepare_packing()
            if self._packing_strength > 0:
                scaled_max = float(arrays["height"].max()) * self.config["height_scale"]
                scaled_max *= 1 + self._packing_relief.maximum_relative_height
                self._validate_extent(scaled_max, basis, visible_size, full_size)
            self._scaled_max_height = scaled_max
        for unit, texture in enumerate(self._textures):
            texture.use(unit)
        if self.config["finish"] == "glazed" or self._interaction_enabled:
            self._phase_texture.use(3)
        if self._interaction_enabled:
            self._interaction_texture.use(5)
        if self._packing_strength > 0:
            self._packing_relief.relative_height.use(6)
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
        if not readback:
            return None
        result = (
            np.frombuffer(self._target.read(alignment=1), dtype="f4")
            .reshape(height, width, 3)[::-1]
            .copy()
        )
        if not np.isfinite(result).all() or np.any(result < 0):
            raise FloatingPointError("Surface rendering produced nonfinite or negative radiance")
        return result

    def _validate_extent(self, scaled_max, basis, visible_size, full_size):
        if scaled_max > self.config["canvas_width_m"] * 0.15:
            raise ValueError("Scaled relief must remain below 15% of visible canvas width")
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

    def _ensure_grid(self, grid_width, grid_height):
        import moderngl

        if self._grid_size == (grid_width, grid_height):
            return
        for texture in self._textures:
            texture.release()
        if self._interaction_texture is not None:
            self._interaction_texture.release()
            self._interaction_texture = None
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
        self._phase_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._phase_texture.repeat_x = self._phase_texture.repeat_y = False
        self._mix_texture = self.ctx.texture((grid_width, grid_height), 1, dtype="f4")
        self._grid_size = (grid_width, grid_height)

    def _ensure_interaction_grid(self):
        import moderngl

        if self._interaction_texture is None:
            self._interaction_texture = self.ctx.texture_array((*self._grid_size, 2), 4, dtype="f4")
            self._interaction_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._interaction_texture.repeat_x = self._interaction_texture.repeat_y = False

    def _prepare_packing(self):
        if self._packing_strength <= 0:
            return
        if self._packing_relief is None:
            self._packing_relief = PackingRelief(
                self.ctx, self._pigment_count, directional=self._directional_relief is not None
            )
        self._packing_relief.prepare(
            self._textures[1],
            self._textures[0],
            self._phase_texture,
            self._interaction_texture,
            self.config,
        )

    def _prepare_gpu_frame(self, frame):
        frame.validate()
        if frame.context is not self.ctx or frame._owner() is not self._borrowed_frame._owner():
            raise ValueError("GPU frame belongs to a different borrowed simulation context")
        if frame.pigment_count != self._pigment_count:
            raise ValueError("GPU frame and palette have different pigment counts")
        if self._interaction_enabled and not frame.has_interaction:
            raise ValueError("Interaction appearance requires native material history")
        if self._packing_strength > 0 and frame.substrate_um != 0:
            raise ValueError("Packing relief requires a zero-height substrate")
        self._scaled_max_height = None
        width, height = frame.size
        self._ensure_grid(width, height)
        if self._gpu_pack is None or self._gpu_pack_interaction != frame.has_interaction:
            if self._gpu_pack is not None:
                self._gpu_pack.release()
            source = (
                (ROOT / "shaders/gpu-frame.comp.glsl")
                .read_text()
                .replace(
                    "#version 430 core",
                    f"#version 430 core\n#define PIGMENT_COUNT {self._pigment_count}"
                    + ("\n#define INTERACTION_CAPTURE" if frame.has_interaction else ""),
                    1,
                )
            )
            self._gpu_pack = self.ctx.compute_shader(source)
            self._gpu_pack_interaction = frame.has_interaction
            if self._gpu_summary is None:
                self._gpu_summary = self.ctx.buffer(reserve=8)
        phases = (*frame.underpaint, *frame.deposit, *frame.mobile)
        for unit, texture in enumerate(phases):
            texture.use(unit)
        program = self._gpu_pack
        program["u_phase"].value = tuple(range(len(phases)))
        frame.carrier.use(len(phases))
        frame.tooth.use(len(phases) + 1)
        program["u_carrier"].value = len(phases)
        program["u_tooth"].value = len(phases) + 1
        if frame.has_interaction:
            self._ensure_interaction_grid()
            for offset, name in enumerate(FIELD_NAMES, start=2):
                unit = len(phases) + offset
                getattr(frame, name).use(unit)
                program["u_" + name].value = unit
            self._interaction_texture.bind_to_image(4, read=False, write=True)
        program["u_specific_volumes"].write(
            np.asarray(frame.specific_volumes, dtype="f4").tobytes()
        )
        program["u_height_scale_mm"].value = frame.height_scale_mm
        program["u_substrate_height_m"].value = frame.substrate_um * 1e-6
        program["u_chalk_index"].value = frame.chalk_index
        program["u_material_model"].value = int(frame.material_model == "laminate")
        self._phase_texture.bind_to_image(0, read=False, write=True)
        self._textures[1].bind_to_image(1, read=False, write=True)
        self._textures[2].bind_to_image(2, read=False, write=True)
        self._mix_texture.bind_to_image(3, read=False, write=True)
        self._gpu_summary.write(b"\0" * 8)
        self._gpu_summary.bind_to_storage_buffer(0)
        program.run((width + 15) // 16, (height + 15) // 16)
        self.ctx.memory_barrier()
        summary = np.frombuffer(self._gpu_summary.read(), dtype="u4")
        if summary[1]:
            self._scaled_max_height = None
            raise FloatingPointError(
                "Native GPU material fields are nonfinite or outside supported bounds"
            )
        scaled_max = float(summary[:1].view("f4")[0]) * self.config["height_scale"]
        self._phase_texture.use(3)
        self._mix_texture.use(4)
        self._textures[0].bind_to_image(0, read=False, write=True)
        self._optics_program.run((width + 15) // 16, (height + 15) // 16)
        self.ctx.memory_barrier()
        self._prepare_packing()
        if self._packing_strength > 0:
            scaled_max *= 1 + self._packing_relief.maximum_relative_height
        frame.validate()
        self._scaled_max_height = scaled_max

    @_current
    def render_gpu(
        self,
        frame=None,
        size=(1280, 960),
        tilt_degrees=0.0,
        azimuth_degrees=-30.0,
        *,
        supersampling=1,
    ):
        """Capture a live Engine view without reading material arrays back to CPU.

        The view is consumed synchronously. After capture, camera-only calls may
        pass None to reuse Surface-owned frozen material even if Engine advances.
        The borrowed engine context must remain alive until this surface closes.
        Pixel calculations use float32 GPU arithmetic; the CPU snapshot/render
        path remains the numerical reference and unchanged final-still path.
        """
        if self._borrowed_frame is None:
            raise ValueError("Native GPU rendering requires a borrowed GPUFrame at construction")
        if type(supersampling) is not int or supersampling not in (1, 2):
            raise ValueError("GPU frame supersampling must be 1 or 2")
        if (
            type(size) not in (tuple, list)
            or len(size) != 2
            or any(type(value) is not int or value < 4 for value in size)
        ):
            raise ValueError("GPU output needs two integer dimensions")
        raster = tuple(value * supersampling for value in size)
        if max(raster) > 12288 or raster[0] * raster[1] > 50_000_000:
            raise ValueError("Supersampled GPU frame exceeds the image budget")
        camera_basis(tilt_degrees, azimuth_degrees)
        if frame is not None:
            if not isinstance(frame, GPUFrame):
                raise ValueError("Native material input must be an Engine GPUFrame")
            self._prepare_gpu_frame(frame)
        self._render_buffer(None, raster, tilt_degrees, azimuth_degrees, readback=False)
        if supersampling == 1:
            result = (
                np.frombuffer(self._target.read(alignment=1), dtype="f4")
                .reshape(size[1], size[0], 3)[::-1]
                .copy()
            )
        else:
            if self._gpu_reduce is None:
                self._gpu_reduce = self.ctx.compute_shader(
                    (ROOT / "shaders/frame-reduce.comp.glsl").read_text()
                )
                self._gpu_reduce["u_input"].value = 0
            if self._gpu_reduced_size != tuple(size):
                if self._gpu_reduced is not None:
                    self._gpu_reduced.release()
                self._gpu_reduced = self.ctx.texture(tuple(size), 4, dtype="f4")
                self._gpu_reduced_size = tuple(size)
            self._target.use(0)
            self._gpu_reduced.bind_to_image(0, read=False, write=True)
            self._gpu_reduce.run((size[0] + 15) // 16, (size[1] + 15) // 16)
            self.ctx.memory_barrier()
            result = (
                np.frombuffer(self._gpu_reduced.read(alignment=1), dtype="f4")
                .reshape(size[1], size[0], 4)[::-1, :, :3]
                .copy()
            )
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise FloatingPointError("Native frame produced invalid linear RGB")
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
        """Release owned images/programs, never borrowed Engine resources."""
        if self.ctx is None:
            return
        resources = (
            self._framebuffer,
            self._target,
            *self._textures,
            self._vao,
            self._quad,
            self._program,
            self._phase_texture,
            self._mix_texture,
            self._interaction_texture,
            self._optics_program,
            self._gpu_pack,
            self._gpu_summary,
            self._gpu_reduce,
            self._gpu_reduced,
        )
        # Closing the engine first destroys its GL context. Do not issue any GL
        # calls through the released context; its driver already owns reclamation.
        alive = self._borrowed_frame is None or self._borrowed_frame.owner_alive()
        if alive:
            with self.ctx:
                for resource in resources:
                    if resource is not None:
                        resource.release()
                if self._packing_relief is not None:
                    self._packing_relief.close()
        if self._owns_context:
            self.ctx.release()
        self.ctx = None
        self._textures = []
        self._framebuffer = self._target = self._vao = self._quad = self._program = None
        self._phase_texture = self._mix_texture = self._optics_program = None
        self._interaction_texture = None
        self._packing_relief = None
        self._gpu_pack = self._gpu_summary = self._gpu_reduce = self._gpu_reduced = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
