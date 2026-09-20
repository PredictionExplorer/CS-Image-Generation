"""Conservative, contact-driven reconstruction of small paint-surface relief.

This finite packing-equilibrium approximation redistributes 12% of existing
displayed paint thickness. It does not model elapsed solvent flow, move pigment,
or add geometric volume. Aggregate affinity drives bounded pair exchanges.
Longer solver pairs accelerate equilibration only along wholly occupied paths.
Four decreasing physical scales and symmetric face orders bound preparation to
28 exchange passes. Results are native-grid approximations, not a claim of exact
resolution independence or fully converged equilibrium.

Optional directional contact relief first reconstructs aggregate affinity with
28 bounded fabric-aligned smoothing passes. Its four-direction occupied paths
include diagonal corner cells; the legacy packing stage and volume budget stay
the same. This is a finite surface response, not additional material motion.

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
DIRECTIONAL_VERSION = "directional-contact-relief-v1"
DIRECTIONAL_DEFAULTS = {"strength": 0.35, "anisotropy": 0.75}
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


def validate_directional(value=None):
    """Normalize an opt-in affinity reconstruction; zero preserves the old path."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - {"version", *DIRECTIONAL_DEFAULTS}:
        raise ValueError("Directional relief requires version, strength and anisotropy controls")
    if value.get("version", DIRECTIONAL_VERSION) != DIRECTIONAL_VERSION:
        raise ValueError(f"Directional relief version must be {DIRECTIONAL_VERSION}")
    result = {"version": DIRECTIONAL_VERSION}
    for name, default in DIRECTIONAL_DEFAULTS.items():
        result[name] = _number(
            value.get(name, default),
            f"Directional relief {name}",
            0,
            0.95 if name == "anisotropy" else 1,
        )
    return result if result["strength"] > 0 else None


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


def affinity_plan(size, canvas_width_m, length_um=DEFAULT_LENGTH_UM):
    """28 paired smoothing passes over two physical scales and four directions.

    Axes 0/1 are y/x; 2/3 are the two diagonals. Diagonal strides account for
    their longer physical step. Parity uses x except for the vertical axis,
    making every pair disjoint within a pass, including across workgroups.
    """
    packing_plan(size, canvas_width_m, length_um)
    pixels = length_um * 1e-6 / (canvas_width_m / size[0])
    result = []
    for level, factor in enumerate((1.0, 0.5)):
        for directions, step_length in (((0, 1), 1.0), ((2, 3), math.sqrt(2))):
            a, b = directions if level % 2 == 0 else directions[::-1]
            stride = max(1, math.floor(pixels * factor / step_length + 0.5))
            attenuation = min(1.0, (pixels / step_length) ** 2)
            sequence = ((a, 0), (a, 1), (b, 0), (b, 1), (b, 0), (a, 1), (a, 0))
            result.extend(
                PackingPass(axis, stride, parity, attenuation * (1.0 if i == 3 else 0.5))
                for i, (axis, parity) in enumerate(sequence)
            )
    return tuple(result)


def _fabric_confidence(contact, fabric):
    # Fabric/contact is alignment coherence. sqrt(contact) keeps lightly worked
    # paint quiet without counting contact twice through the stored fabric dose.
    return np.divide(
        fabric,
        np.sqrt(np.maximum(contact, 1e-12))[..., None],
        out=np.zeros_like(fabric),
        where=contact[..., None] > 0,
    )


def directional_affinity(
    aggregate,
    contact,
    fabric,
    occupied,
    *,
    canvas_width_m,
    directional,
    length_um=DEFAULT_LENGTH_UM,
):
    """Reconstruct elongated aggregate affinity from frozen material history.

    Paired convex smoothing follows the nematic fabric, so a rotated fabric
    rotates the affinity's elongation. Constant affinity creates no pattern.
    Every cardinal and diagonal path is wholly occupied, including both sides
    of each diagonal corner. No camera, time, random field or pigment is read.
    """
    controls = validate_directional(directional)
    g, c, q = (np.asarray(value, dtype="f8") for value in (aggregate, contact, fabric))
    mask = np.asarray(occupied)
    if (
        g.ndim != 2
        or c.shape != g.shape
        or q.shape != (*g.shape, 2)
        or mask.shape != g.shape
        or mask.dtype != np.bool_
        or not all(np.isfinite(value).all() for value in (g, c, q))
        or np.any(g < 0)
        or np.any(g > 1)
        or np.any(c < 0)
        or np.any(c > 1)
        or np.any(np.linalg.norm(q, axis=-1) > c + 1e-5)
    ):
        raise ValueError("Directional relief requires bounded matching aggregate/contact/fabric")
    plan = affinity_plan((g.shape[1], g.shape[0]), canvas_width_m, length_um)
    result = g.copy()
    if controls is None:
        return result
    direction = _fabric_confidence(c, q)
    yy, xx = np.indices(g.shape)
    for step in plan:
        dx, dy = ((0, 1), (1, 0), (1, 1), (1, -1))[step.axis]
        tx, ty = xx + dx * step.stride, yy + dy * step.stride
        coordinate = yy if step.axis == 0 else xx
        selected = (
            ((coordinate // step.stride) % 2 == step.parity)
            & (tx >= 0)
            & (tx < g.shape[1])
            & (ty >= 0)
            & (ty < g.shape[0])
        )
        py, px = yy[selected], xx[selected]
        qy, qx = ty[selected], tx[selected]
        allowed = mask[py, px] & mask[qy, qx]
        for i in range(1, step.stride):
            allowed &= mask[py + i * dy, px + i * dx]
        if step.axis >= 2:
            for i in range(step.stride):
                allowed &= mask[py + i * dy, px + (i + 1) * dx]
                allowed &= mask[py + (i + 1) * dy, px + i * dx]
        a, b = result[py, px], result[qy, qx]
        mean = 0.5 * (direction[py, px] + direction[qy, qx])
        confidence = np.minimum(np.linalg.norm(mean, axis=-1), 1.0)
        axial = np.array((dx * dx - dy * dy, 2 * dx * dy)) / (dx * dx + dy * dy)
        alignment = np.clip(np.sum(mean * axial, axis=-1) / np.maximum(confidence, 1e-12), -1, 1)
        along = (0.5 + 0.5 * alignment) ** 4
        weight = (
            0.5
            * step.relaxation
            * controls["strength"]
            * confidence
            * (1 - controls["anisotropy"] + controls["anisotropy"] * along)
        )
        transfer = np.where(allowed, weight * (a - b), 0)
        result[py, px], result[qy, qx] = a - transfer, b + transfer
    return result


def directional_packing_displacement(
    height,
    aggregate,
    occupied,
    *,
    contact=None,
    fabric=None,
    directional=None,
    canvas_width_m,
    strength=1.0,
    length_um=DEFAULT_LENGTH_UM,
):
    """Conservative displayed relief driven by directionally reconstructed affinity."""
    controls = validate_directional(directional)
    affinity = aggregate
    if controls is not None:
        affinity = directional_affinity(
            aggregate,
            contact,
            fabric,
            np.asarray(occupied) & (np.asarray(height) > 0),
            canvas_width_m=canvas_width_m,
            directional=controls,
            length_um=length_um,
        )
    return packing_displacement(
        height,
        affinity,
        occupied,
        canvas_width_m=canvas_width_m,
        strength=strength,
        length_um=length_um,
    )


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
    """Surface-owned staging: 12 bytes/pixel and 30 bounded passes by default.

    Directional reconstruction uses 20 bytes/pixel and 58 bounded passes.

    All methods run inside the caller's current OpenGL context. The renderer
    never calls prepare during a frozen camera move. No CPU material readback,
    atomic float accumulation, random field or screen-space normal is involved.
    """

    def __init__(self, context, pigment_count, *, directional=False):
        self.ctx = context
        self.directional = directional
        self._program = self.potential = self.relative_height = self._summary = None
        self._size = None
        self.last_plan = ()
        self.last_affinity_plan = ()
        self.maximum_relative_height = 0.0
        try:
            source = (Path(__file__).parent / "shaders/packing.comp.glsl").read_text()
            self._program = context.compute_shader(
                source.replace(
                    "#version 430 core",
                    f"#version 430 core\n#define PIGMENT_COUNT {pigment_count}"
                    + ("\n#define DIRECTIONAL_RELIEF" if directional else ""),
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
        directional = validate_directional(controls.get("directional_relief"))
        if (directional is not None) != self.directional:
            raise ValueError("Packing directional controls differ from its compiled mode")
        strength = controls["packing_strength"]
        plan = packing_plan(
            size, config["canvas_width_m"] * config["domain_scale"], controls["packing_length_um"]
        )
        if self._size != size:
            new_potential = new_relative = None
            try:
                new_potential = self.ctx.texture(size, 4 if self.directional else 2, dtype="f4")
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
        if directional is not None:
            p["u_directional_strength"].value = directional["strength"]
            p["u_directional_anisotropy"].value = directional["anisotropy"]
        p["u_mode"].value = 0
        groups = ((size[0] + 15) // 16, (size[1] + 15) // 16)
        p.run(*groups)
        self.ctx.memory_barrier()
        self.last_affinity_plan = ()
        if directional is not None:
            self.last_affinity_plan = affinity_plan(
                size,
                config["canvas_width_m"] * config["domain_scale"],
                controls["packing_length_um"],
            )
            p["u_mode"].value = 3
            for step in self.last_affinity_plan:
                p["u_axis"].value = step.axis
                p["u_stride"].value = step.stride
                p["u_parity"].value = step.parity
                p["u_relaxation"].value = step.relaxation
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
