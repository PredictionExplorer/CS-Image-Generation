"""Conservative, contact-driven reconstruction of small paint-surface relief.

This finite packing-equilibrium approximation redistributes 12% of existing
displayed paint thickness. It does not model elapsed solvent flow, move pigment,
or add geometric volume. Aggregate affinity drives bounded pair exchanges.
Longer solver pairs accelerate equilibration only along wholly occupied paths.
Four decreasing physical scales and symmetric face orders bound preparation to
28 exchange passes. Results are native-grid approximations, not a claim of exact
resolution independence or fully converged equilibrium.

Input thickness is in metres after the existing authored height/glaze transforms.
The substrate must be flat at zero: current archives do not separate its height.
Native-grid displacements conserve volume. GPU staging stores relative changes
for bounded interpolation against the unchanged legacy surface evaluation;
no exact continuous-surface volume claim is made between native samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = "aggregate-packing-v1"
FILM_FRACTION = 0.12
MAX_STRIDE = 16
DEFAULT_LENGTH_UM = 1200.0
LENGTH_BOUNDS_UM = (80.0, 2400.0)


@dataclass(frozen=True)
class PackingPass:
    axis: int
    stride: int
    parity: int
    relaxation: float


def _number(value, name, low, high):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite range") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def packing_plan(size, canvas_width_m, length_um=DEFAULT_LENGTH_UM):
    """Return a deterministic, bounded multiscale native-grid relaxation plan.

    ``size`` is width,height of the full guarded material. Pixels are square in
    world space. Length selects the coarsest coupling distance; subpixel lengths
    attenuate relaxation quadratically. Oversized strides fail explicitly.
    """
    if (
        type(size) not in (tuple, list)
        or len(size) != 2
        or any(type(v) is not int or v < 4 for v in size)
    ):
        raise ValueError("Packing size needs two integer dimensions of at least four")
    width_m = _number(canvas_width_m, "Packing canvas width", 0.001, 100)
    length = _number(length_um, "Packing length", *LENGTH_BOUNDS_UM) * 1e-6
    pixels = length / (width_m / size[0])
    if pixels > MAX_STRIDE:
        raise ValueError("Packing length exceeds the supported 16-cell native coupling radius")
    attenuation = min(1.0, pixels * pixels)
    plan = []
    for level, factor in enumerate((1.0, 0.5, 0.25, 0.0)):
        stride = max(1, math.floor(pixels * factor + 0.5))
        a, b = (1, 0) if level % 2 == 0 else (0, 1)
        sequence = ((a, 0), (a, 1), (b, 0), (b, 1), (b, 0), (a, 1), (a, 0))
        for index, (axis, parity) in enumerate(sequence):
            plan.append(
                PackingPass(axis, stride, parity, attenuation * (1.0 if index == 3 else 0.5))
            )
    return tuple(plan)


def packing_displacement(
    height, aggregate, occupied, *, canvas_width_m, strength=1.0, length_um=DEFAULT_LENGTH_UM
):
    """Float64 conservative reference, leaving every caller array unchanged."""
    strength = _number(strength, "Packing strength", 0, 1)
    h, g = np.asarray(height, dtype="f8"), np.asarray(aggregate, dtype="f8")
    mask = np.asarray(occupied)
    if (
        h.ndim != 2
        or g.shape != h.shape
        or mask.shape != h.shape
        or mask.dtype != np.bool_
        or not np.isfinite(h).all()
        or not np.isfinite(g).all()
        or np.any(h < 0)
        or np.any(g < 0)
        or np.any(g > 1)
    ):
        raise ValueError(
            "Packing requires matching nonnegative thickness, bounded aggregate and boolean mask"
        )
    plan = packing_plan((h.shape[1], h.shape[0]), canvas_width_m, length_um)
    if strength == 0:
        return np.zeros_like(h)
    base = np.where(mask, h * FILM_FRACTION, 0)
    film = base.copy()
    for step in plan:
        axis, stride, parity = step.axis, step.stride, step.parity
        starts = np.arange(h.shape[axis] - stride)
        starts = starts[(starts // stride) % 2 == parity]
        for start in starts:
            first = [slice(None), slice(None)]
            second = first.copy()
            first[axis], second[axis] = start, start + stride
            first, second = tuple(first), tuple(second)
            a, b = base[first], base[second]
            fa, fb = film[first].copy(), film[second].copy()
            path = [slice(None), slice(None)]
            path[axis] = slice(start, start + stride + 1)
            allowed = np.all(base[tuple(path)] > 0, axis=axis)
            difference = fa * b - fb * a - strength * a * b * (g[first] - g[second])
            transfer = np.divide(difference, a + b, out=np.zeros_like(a), where=allowed)
            transfer *= step.relaxation
            low = np.maximum(fa - (1 + strength) * a, (1 - strength) * b - fb)
            high = np.minimum(fa - (1 - strength) * a, (1 + strength) * b - fb)
            transfer = np.clip(transfer, low, high)
            transfer = np.where(allowed, transfer, 0)
            film[first], film[second] = fa - transfer, fb + transfer
    return film - base


class PackingRelief:
    """Surface-owned staging: 12 bytes/pixel, 30 bounded compute passes/capture.

    All methods run inside the caller's current OpenGL context. The renderer
    never calls prepare during a frozen camera move. No CPU material readback,
    atomic float accumulation, random field or screen-space normal is involved.
    """

    def __init__(self, context, pigment_count):
        self.ctx = context
        self._program = self.potential = self.relative_height = self._summary = None
        self._size = None
        self.last_plan = ()
        self.maximum_relative_height = 0.0
        try:
            source = (Path(__file__).parent / "shaders/packing.comp.glsl").read_text()
            self._program = context.compute_shader(
                source.replace(
                    "#version 430 core",
                    f"#version 430 core\n#define PIGMENT_COUNT {pigment_count}",
                    1,
                )
            )
            self._summary = context.buffer(reserve=4)
        except BaseException:
            self.close()
            raise

    def prepare(self, geometry, paint, phases, interaction, config):
        import moderngl

        size = geometry.size
        controls = config["interaction"]
        strength = controls["packing_strength"]
        plan = packing_plan(
            size, config["canvas_width_m"] * config["domain_scale"], controls["packing_length_um"]
        )
        if self._size != size:
            new_potential = new_relative = None
            try:
                new_potential = self.ctx.texture(size, 2, dtype="f4")
                new_relative = self.ctx.texture(size, 1, dtype="f4")
                new_relative.filter = (moderngl.LINEAR, moderngl.LINEAR)
                new_relative.repeat_x = new_relative.repeat_y = False
            except BaseException:
                for texture in (new_potential, new_relative):
                    if texture is not None:
                        texture.release()
                raise
            for texture in (self.potential, self.relative_height):
                if texture is not None:
                    texture.release()
            self.potential, self.relative_height = new_potential, new_relative
            self._size = size
        p = self._program
        for unit, (name, texture) in enumerate(
            (
                ("u_geometry", geometry),
                ("u_paint", paint),
                ("u_phases", phases),
                ("u_interaction", interaction),
            )
        ):
            texture.use(unit)
            p[name].value = unit
        self.potential.bind_to_image(0, read=True, write=True)
        self.relative_height.bind_to_image(1, read=True, write=True)
        p["u_strength"].value = strength
        p["u_height_scale"].value = config["height_scale"]
        p["u_mass_threshold"].value = config["paint_mass_threshold"]
        p["u_mass_reference"].value = config["paint_mass_reference"]
        p["u_glazed"].value = int(config["finish"] == "glazed")
        p["u_glaze_relief_strength"].value = config.get("glaze_relief_strength", 0.8)
        p["u_mode"].value = 0
        groups = ((size[0] + 15) // 16, (size[1] + 15) // 16)
        p.run(*groups)
        self.ctx.memory_barrier()
        p["u_mode"].value = 1
        for step in plan:
            p["u_axis"].value = step.axis
            p["u_stride"].value = step.stride
            p["u_parity"].value = step.parity
            p["u_relaxation"].value = step.relaxation
            p.run(*groups)
            self.ctx.memory_barrier()
        p["u_mode"].value = 2
        self._summary.write(b"\0" * 4)
        self._summary.bind_to_storage_buffer(1)
        p.run(*groups)
        self.ctx.memory_barrier()
        self.maximum_relative_height = float(np.frombuffer(self._summary.read(), dtype="f4")[0])
        self.last_plan = plan

    def close(self):
        for resource in (self._program, self.potential, self.relative_height, self._summary):
            if resource is not None:
                resource.release()
        self._program = self.potential = self.relative_height = self._summary = None
