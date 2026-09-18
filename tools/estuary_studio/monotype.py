"""Partially loaded tools that deposit, drag and lift a persistent paint layer.

This is a bounded, kinematic thin-paint model, not a bristle or fluid solver.
Paint amount, tool marks and wetness are state; illumination is a separate stage.
Semi-Lagrangian dragging is deliberately dissipative and is not mass conserving.
The tool's finite *loading fraction* controls transfer, rather than pretending to
measure a volumetric reservoir from the two-dimensional contact approximation.
"""

from __future__ import annotations

import math
from functools import wraps
from pathlib import Path

import numpy as np

DEFAULTS = {
    "kind": "monotype",
    "resolution": [1024, 768],
    "steps": 3600,
    "domain_scale": 1.6,
    "brush_width": 0.24,
    "deposit_rate": 0.65,
    "drag_strength": 0.85,
    "lift_rate": 0.45,
    "dry_rate": 3.2,
    "reload_rate": 4.0,
    "load_capacity": 1.0,
    "depletion_rate": 0.18,
    "height_mm": 1.4,
    "initial_load": 0.14,
    "bristle_strength": 0.28,
    "contact_depth": 0.70,
    "speed_response": 0.006,
}
_RANGES = {
    "domain_scale": (1.0, 2.5),
    "brush_width": (0.03, 0.8),
    "deposit_rate": (0.0, 5.0),
    "drag_strength": (0.0, 2.0),
    "lift_rate": (0.0, 5.0),
    "dry_rate": (0.0, 30.0),
    "reload_rate": (0.0, 30.0),
    "load_capacity": (0.0, 2.0),
    "depletion_rate": (0.0, 10.0),
    "height_mm": (0.0, 10.0),
    "initial_load": (0.0, 2.0),
    "bristle_strength": (0.0, 1.0),
    "contact_depth": (0.05, 3.0),
    "speed_response": (0.0, 0.1),
}
MAX_INTERNAL_STEPS = 100_000


def validate_config(config=None):
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Unknown Monotype configuration fields")
    result = {**DEFAULTS, **config}
    if result["kind"] not in ("monotype", "nocturne"):
        raise ValueError("kind must be monotype or nocturne")
    size = result["resolution"]
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(x) is not int or not 32 <= x <= 8192 for x in size)
        or not 0.2 <= size[0] / size[1] <= 5
    ):
        raise ValueError("Invalid simulation resolution")
    result["resolution"] = list(size)
    if type(result["steps"]) is not int or not 1 <= result["steps"] <= 50_000:
        raise ValueError("Invalid canonical step count")
    for name, (low, high) in _RANGES.items():
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {name}")
        result[name] = float(value)
    return result


def contact_pressure(depth, speed, *, contact_depth=0.70, speed_response=0.006):
    """Smooth contact with the fixed source plane; depth uses one global scale."""
    depth, speed = np.asarray(depth, dtype=float), np.asarray(speed, dtype=float)
    if not np.isfinite(depth).all() or not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("Contact inputs must be finite, with nonnegative speed")
    if (
        not math.isfinite(contact_depth)
        or contact_depth <= 0
        or not math.isfinite(speed_response)
        or speed_response < 0
    ):
        raise ValueError("Invalid contact response")
    ramp = np.clip((np.abs(depth) / contact_depth - 0.08) / 0.92, 0, 1)
    return (1 - ramp * ramp * (3 - 2 * ramp)) / (1 + speed_response * speed)


def update_load(load, pressure, travel, dt, config):
    """Finite loading fraction; reload only during lift and deplete on contact."""
    load, pressure, travel = (np.asarray(x, dtype=np.float64) for x in (load, pressure, travel))
    if (
        any(not np.isfinite(x).all() for x in (load, pressure, travel))
        or np.any(travel < 0)
        or np.any((pressure < 0) | (pressure > 1))
        or not math.isfinite(dt)
        or dt < 0
    ):
        raise ValueError("Invalid loading update")
    capacity = config["load_capacity"]
    if np.any((load < 0) | (load > capacity)):
        raise ValueError("Loading fraction exceeds capacity")
    depleted = load * np.exp(-config["depletion_rate"] * pressure * travel)
    return depleted + (capacity - depleted) * -np.expm1(
        -config["reload_rate"] * (1 - pressure) * dt
    )


def local_transfer(pigment, deposit, pickup):
    """Exact local deposit/lift accounting used by the shader after dragging."""
    pigment, deposit, pickup = (np.asarray(x, dtype=np.float64) for x in (pigment, deposit, pickup))
    if (
        any(not np.isfinite(x).all() for x in (pigment, deposit, pickup))
        or np.any(pigment < 0)
        or np.any(deposit < 0)
        or np.any((pickup < 0) | (pickup > 1))
    ):
        raise ValueError("Invalid material transfer")
    lifted = pigment * pickup
    return pigment - lifted + deposit, lifted


class PlaneContact:
    """Third coordinate from the source's immutable full-recording PCA frame."""

    def __init__(self, source):
        projection = source.projection
        axes = np.asarray(projection["axes"], dtype=np.float64)
        normal = np.cross(axes[:, 0], axes[:, 1])
        normal /= np.linalg.norm(normal)
        origin = np.asarray(projection["origin"])
        mean = np.asarray(projection["mean"])
        extent = projection["extent"]
        depths = np.empty((source.samples, 3), dtype=np.float64)
        # Read the original immutable mapping, never infer Z from projected XY.
        for start in range(0, source.samples, 32768):
            raw = source._raw[start : start + 32768]
            depths[start : start + len(raw)] = np.einsum(
                "...i,i->...", (raw - origin) / extent - mean, normal, optimize=False
            )
        self.scale = max(float(np.quantile(np.abs(depths), 0.90)), 1e-12)
        self.depths = depths / self.scale
        self.metadata = {
            "normal": normal.tolist(),
            "normalization": "one full-recording 90th percentile of absolute PCA-normal distance",
            "depth_scale_normalized_source": self.scale,
            "plane": "full-recording PCA centroid; no per-frame refit",
        }

    def sample(self, fraction):
        if (
            type(fraction) not in (float, int)
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("Source fraction must be in [0,1]")
        x = fraction * (len(self.depths) - 1)
        left = min(int(x), len(self.depths) - 2)
        return self.depths[left] * (1 - (x - left)) + self.depths[left + 1] * (x - left)


def _current(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        if self.ctx is None:
            raise RuntimeError("The Monotype context is closed")
        with self.ctx:
            return method(self, *args, **kwargs)

    return call


class Monotype:
    """A dedicated GPU state and deterministic full-trajectory tool clock."""

    def __init__(self, source, config=None, backend="egl"):
        import moderngl

        self.config = validate_config(config)
        self.source = source
        self.width, self.height = self.config["resolution"]
        self.aspect = self.width / self.height
        if not math.isclose(self.aspect, source.aspect, rel_tol=1e-9):
            raise ValueError("Source projection and painting aspect ratios differ")
        self.domain = self.config["domain_scale"]
        self.steps, self.step, self.internal_steps = self.config["steps"], 0, 0
        self.contact = PlaneContact(source)
        self.load = np.full(3, self.config["load_capacity"], dtype=np.float64)
        self.max_segment_travel = 0.0
        self.ctx = None
        try:
            self.ctx = moderngl.create_standalone_context(require=430, backend=backend)
            if any(
                x in self.ctx.info["GL_RENDERER"].lower()
                for x in ("llvmpipe", "softpipe", "software")
            ):
                raise RuntimeError("A hardware GPU is required")
            if max(self.width, self.height) > self.ctx.info["GL_MAX_TEXTURE_SIZE"]:
                raise ValueError("Simulation exceeds the GPU texture-size limit")
            self.shader = self.ctx.compute_shader(
                (Path(__file__).parent / "shaders/monotype.glsl").read_text()
            )
            self.paint, self.surface = [], []
            for collection in (self.paint, self.surface):
                for _ in range(2):
                    texture = self.ctx.texture((self.width, self.height), 4, dtype="f4")
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    texture.repeat_x = texture.repeat_y = False
                    collection.append(texture)
            self.index = 0
            self.groups = ((self.width + 15) // 16, (self.height + 15) // 16)
            values = {
                "u_size": (self.width, self.height),
                "u_aspect": self.aspect,
                "u_domain": self.domain,
                "u_paint": 0,
                "u_surface": 1,
                "u_width": self.config["brush_width"],
                "u_drag": self.config["drag_strength"],
                "u_lift": self.config["lift_rate"],
                "u_deposit": self.config["deposit_rate"],
                "u_dry": self.config["dry_rate"],
                "u_height": self.config["height_mm"] / 1000,
                "u_bristles": self.config["bristle_strength"],
                "u_nocturne": int(self.config["kind"] == "nocturne"),
            }
            for key, value in values.items():
                self.shader[key].value = value
            self._initialize()
            self.metadata = {
                "model": "bounded thin-paint deposition, semi-Lagrangian drag and fractional lift",
                "kind": self.config["kind"],
                "vendor": self.ctx.info["GL_VENDOR"],
                "renderer": self.ctx.info["GL_RENDERER"],
                "version": self.ctx.info["GL_VERSION"],
                "moderngl": moderngl.__version__,
                "backend": backend,
                "cross_device_pixel_identity": False,
                "mass_conserving": False,
                "tool_load": "finite loading fraction, not a resolved volumetric bristle reservoir",
                "direction": "persistent axial cos(2 theta), sin(2 theta), carried by paint drag",
                "height_units": "metres; authored specific volume of local pigment amount",
                "visible_canvas_width_metres": 0.4,
                "contact": self.contact.metadata,
                "simulation_bounds": [
                    [-self.aspect * self.domain, -self.domain],
                    [self.aspect * self.domain, self.domain],
                ],
                "view_bounds": [[-self.aspect, -1], [self.aspect, 1]],
                "brush_roles": [
                    "loaded broad brush",
                    "opaque dragging blade",
                    "sparse lifting and red transfer",
                ],
                "config": self.config,
            }
        except Exception:
            self.close()
            raise

    def _initialize(self):
        xx = ((np.arange(self.width) + 0.5) / self.width * 2 - 1) * self.aspect * self.domain
        yy = ((np.arange(self.height) + 0.5) / self.height * 2 - 1) * self.domain
        x, y = np.meshgrid(xx, yy)
        p = self.source.frame(0).positions
        d = p[1] - p[0]
        d /= max(np.linalg.norm(d), 1e-12)
        center = p.mean(axis=0)
        along = (x - center[0]) * d[0] + (y - center[1]) * d[1]
        across = -(x - center[0]) * d[1] + (y - center[1]) * d[0]
        # A thin prepared plate, aligned once to the actual initial triangle.
        footprint = (along / 0.92) ** 6 + (across / 0.38) ** 4
        edge = np.clip((1.0 - footprint) / 0.12, 0, 1)
        field = edge * edge * (3 - 2 * edge)
        paint = np.zeros((self.height, self.width, 4), dtype="f4")
        if self.config["kind"] == "nocturne":
            paint[:, :, 0] = self.config["initial_load"] * (0.72 + 0.28 * field)
        else:
            paint[:, :, 0] = self.config["initial_load"] * field
            paint[:, :, 1] = self.config["initial_load"] * 0.05 * field
        paint[:, :, 3] = 0.45 * field
        surface = np.zeros_like(paint)
        surface[:, :, 0] = (
            (paint[:, :, :3] @ np.array([0.30, 1.0, 0.45])) * self.config["height_mm"] / 1000
        )
        surface[:, :, 1] = d[0] * d[0] - d[1] * d[1]
        surface[:, :, 2] = 2 * d[0] * d[1]
        surface[:, :, 3] = 0.66
        for texture in self.paint:
            texture.write(paint.tobytes())
        for texture in self.surface:
            texture.write(surface.tobytes())

    def _segment(self, start, end, a, b):
        mid = (start + end) * 0.5
        frame = self.source.frame(mid)
        speeds = np.linalg.norm(frame.velocities, axis=1)
        pressure = contact_pressure(
            self.contact.sample(mid),
            speeds,
            contact_depth=self.config["contact_depth"],
            speed_response=self.config["speed_response"],
        )
        travel = np.maximum(b.arc_lengths - a.arc_lengths, 0)
        load_end = update_load(self.load, pressure, travel, end - start, self.config)
        loads = (self.load + load_end) * 0.5
        self.load = load_end
        directions = frame.velocities.copy()
        lengths = np.linalg.norm(directions, axis=1)
        directions /= np.maximum(lengths[:, None], 1e-15)
        directions[lengths < 1e-12] = [1, 0]
        self.shader["u_segments"].write(
            np.concatenate([a.positions, b.positions], axis=1).astype("f4").tobytes()
        )
        self.shader["u_tools"].write(
            np.stack([pressure, loads, travel, np.zeros(3)], axis=1).astype("f4").tobytes()
        )
        self.shader["u_directions"].write(directions.astype("f4").tobytes())
        self.shader["u_dt"].value = end - start
        self.paint[self.index].use(0)
        self.surface[self.index].use(1)
        self.paint[1 - self.index].bind_to_image(0, read=False, write=True)
        self.surface[1 - self.index].bind_to_image(1, read=False, write=True)
        self.shader.run(*self.groups)
        self.ctx.memory_barrier()
        self.index = 1 - self.index
        self.internal_steps += 1
        self.max_segment_travel = max(self.max_segment_travel, float(travel.max()))

    @_current
    def advance_to(self, step):
        if type(step) is not int or not self.step <= step <= self.steps:
            raise ValueError("Canonical steps must move forward within the recording")
        for canonical in range(self.step, step):
            start, end = canonical / self.steps, (canonical + 1) / self.steps
            pending = [(start, end, self.source.frame(start), self.source.frame(end))]
            while pending:
                a_time, b_time, a, b = pending.pop()
                if float(np.max(b.arc_lengths - a.arc_lengths)) > self.config["brush_width"] * 0.20:
                    if (
                        self.internal_steps + len(pending) >= MAX_INTERNAL_STEPS
                        or b_time - a_time < 1e-12
                    ):
                        raise RuntimeError("Source exceeds bounded brush sampling budget")
                    mid = (a_time + b_time) * 0.5
                    m = self.source.frame(mid)
                    pending.extend([(mid, b_time, m, b), (a_time, mid, a, m)])
                    continue
                if self.internal_steps >= MAX_INTERNAL_STEPS:
                    raise RuntimeError("Source exceeds bounded brush sampling budget")
                self._segment(a_time, b_time, a, b)
            self.step = canonical + 1

    @_current
    def snapshot(self):
        paint = (
            np.frombuffer(self.paint[self.index].read(), dtype="f4")
            .reshape(self.height, self.width, 4)
            .copy()
        )
        surface = (
            np.frombuffer(self.surface[self.index].read(), dtype="f4")
            .reshape(self.height, self.width, 4)
            .copy()
        )
        if not np.isfinite(paint).all() or not np.isfinite(surface).all():
            raise RuntimeError("Non-finite Monotype state")
        return {
            "pigment": paint[:, :, :3].copy(),
            "height": surface[:, :, 0].copy(),
            "wetness": paint[:, :, 3].copy(),
            "direction": surface[:, :, 1:3].copy(),
            "roughness": surface[:, :, 3].copy(),
            "coverage": (-np.expm1(-paint[:, :, :3].sum(axis=2) * 8)).astype("f4"),
        }

    def close(self):
        if getattr(self, "ctx", None) is not None:
            self.ctx.release()
            self.ctx = None
