"""GPU pigment transport with a fixed source clock and bounded substeps.

The prescribed flow is the curl of a smooth stream function, with a no-through-
flow boundary. It is an artistic stirring model, not a Navier--Stokes solver.
Limited MacCormack transport retains fine boundaries without negative density.
Neither this interpolation nor finite precision guarantees exact mass conservation.
"""

from __future__ import annotations

import math
from functools import wraps
from pathlib import Path

import moderngl
import numpy as np

from .optics import Material

ROOT = Path(__file__).parent
VERTEX = """#version 330
in vec2 in_position;out vec2 v_uv;uniform float u_view_domain;
void main(){v_uv=in_position*0.5/u_view_domain+0.5;gl_Position=vec4(in_position,0,1);}
"""
MAX_INTERNAL_STEPS = 500_000


def _current_context(method):
    """Bind this engine explicitly when callers interleave separate simulations."""

    @wraps(method)
    def current(self, *args, **kwargs):
        if self.ctx is None:
            raise RuntimeError("The Estuary context has been closed")
        with self.ctx:
            return method(self, *args, **kwargs)

    return current


def tool_uniforms(frame, radius):
    """Bound the force response smoothly, retaining actual source directions."""
    velocity = np.asarray(frame.velocities, dtype=np.float64)
    velocity = velocity / (1 + np.linalg.norm(velocity, axis=1)[:, None] / 24)
    positions = frame.positions
    tools = np.concatenate([positions, velocity], axis=1).astype("f4")
    pairs = []
    for pair, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        d, v = positions[b] - positions[a], velocity[b] - velocity[a]
        # The 3D denominator does not amplify crossings caused only by projection.
        spin = (d[0] * v[1] - d[1] * v[0]) / (frame.pair_distances[pair] ** 2 + radius * radius)
        spin = 20 * np.tanh(spin / 20)
        pairs.append([*((positions[a] + positions[b]) * 0.5), spin])
    return tools, np.asarray(pairs, dtype="f4")


class Engine:
    """Own a dedicated headless OpenGL context and cumulative paint textures."""

    def __init__(self, source, recipe, backend="egl"):
        self.source, self.recipe = source, recipe
        settings = recipe["simulation"]
        self.width, self.height = settings["resolution"]
        self.aspect = self.width / self.height
        self.domain = settings["domain_scale"]
        self.steps = settings["steps"]
        self.step = 0
        self.internal_steps = 0
        self.maximum_courant = 0.0
        self.ctx = moderngl.create_standalone_context(require=430, backend=backend)
        renderer = self.ctx.info["GL_RENDERER"]
        if any(word in renderer.lower() for word in ("llvmpipe", "softpipe", "software")):
            self.ctx.release()
            raise RuntimeError("A hardware GPU is required; software fallback is disabled")
        if max(self.width, self.height) > self.ctx.info["GL_MAX_TEXTURE_SIZE"]:
            raise ValueError("Simulation exceeds this GPU's texture-size limit")
        self.metadata = {
            "vendor": self.ctx.info["GL_VENDOR"],
            "renderer": renderer,
            "version": self.ctx.info["GL_VERSION"],
            "moderngl": moderngl.__version__,
            "backend": backend,
            "transport": "limited-MacCormack/RK2; kinematic stream-function flow",
            "precision": "float32 GPU state; float64 source projection",
            "cross_device_pixel_identity": False,
            "simulation_bounds": [
                [-self.aspect * self.domain, -self.domain],
                [self.aspect * self.domain, self.domain],
            ],
            "view_bounds": [[-self.aspect, -1.0], [self.aspect, 1.0]],
        }
        self.groups = ((self.width + 15) // 16, (self.height + 15) // 16)
        self.flow = self.ctx.compute_shader((ROOT / "shaders/flow.glsl").read_text())
        self.advect = self.ctx.compute_shader((ROOT / "shaders/advect.glsl").read_text())
        self.correct = self.ctx.compute_shader((ROOT / "shaders/correct.glsl").read_text())
        self.paint = [self._texture(4) for _ in range(4)]
        self.velocity = self._texture(2)
        self.maxima = self.ctx.buffer(reserve=4 * self.groups[0] * self.groups[1])
        for shader in (self.flow, self.advect, self.correct):
            shader["u_size"].value = (self.width, self.height)
            shader["u_aspect"].value = self.aspect
            shader["u_domain"].value = self.domain
        self.flow["u_radius"].value = settings["stir_radius"]
        self.flow["u_strength"].value = settings["flow_strength"]
        self.flow["u_pair_gain"].value = settings["pair_swirl"]
        self.flow["u_carrier"].value = tuple(settings["carrier_velocity"])
        self.correct["u_brush"].value = settings["brush_radius"]
        self.program = self.ctx.program(
            vertex_shader=VERTEX, fragment_shader=(ROOT / "optics.glsl").read_text()
        )
        for name, value in Material(**recipe["optics"]).uniforms().items():
            if name in self.program:
                self.program[name].value = value
        self.program["u_paint"].value = 0
        self.program["u_view_domain"].value = self.domain
        self.quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.quad, "2f", "in_position")])
        self._output = None
        self._framebuffer = None
        self._initialize()

    def _texture(self, channels):
        texture = self.ctx.texture((self.width, self.height), channels, dtype="f4")
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        return texture

    @staticmethod
    def _compact_brush(q):
        ramp = np.clip((q - 0.36) / 0.64, 0, 1)
        return 1 - ramp * ramp * (3 - 2 * ramp)

    def _initialize(self):
        settings = self.recipe["simulation"]
        state = np.zeros((self.height, self.width, 4), dtype="f4")
        if settings["initial_load"]:
            x = (
                (np.arange(self.width) + 0.5) / self.width * 2 * self.aspect - self.aspect
            ) * self.domain
            y = ((np.arange(self.height) + 0.5) / self.height * 2 - 1) * self.domain
            initial = self.source.frame(0).positions
            if settings["initial_pattern"] == "strata":
                edge = initial[1] - initial[0]
                length = float(np.linalg.norm(edge))
                direction = edge / length if length > 1e-10 else np.array([1.0, 0.0])
                center = initial.mean(axis=0)
                distance = (x[None, :] - center[0]) * -direction[1] + (
                    y[:, None] - center[1]
                ) * direction[0]
                radius = settings["load_radius"]

                def band(offset, half_width):
                    ramp = np.clip(
                        (np.abs(distance - offset) - half_width * 0.9) / (half_width * 0.1), 0, 1
                    )
                    return 1 - ramp * ramp * (3 - 2 * ramp)

                profile = settings.get("strata_profile")
                if profile is None:
                    # Keep the released initializer's arithmetic verbatim.
                    white = np.maximum.reduce(
                        [
                            band(0, radius),
                            band(-radius * 1.65, radius * 0.075),
                            band(radius * 2.15, radius * 0.045),
                        ]
                    )
                    copper = band(radius * 1.45, radius * 0.09) * (1 - white)
                else:
                    white = band(0, radius * profile["main_width_scale"])
                    fine = profile["fine_width_scale"]
                    if fine > 0:
                        white = np.maximum.reduce(
                            [
                                white,
                                band(-radius * 1.65, radius * 0.075 * fine),
                                band(radius * 2.15, radius * 0.045 * fine),
                            ]
                        )
                    copper = band(radius * 1.45, radius * 0.09 * profile["accent_width_scale"]) * (
                        1 - white
                    )
                state[:, :, 0] = settings["initial_load"] * (1 - white - copper)
                state[:, :, 1] = settings["initial_load"] * white
                state[:, :, 2] = settings["initial_load"] * copper
            else:
                for body, p in enumerate(initial):
                    r2 = (x[None, :] - p[0]) ** 2 + (y[:, None] - p[1]) ** 2
                    state[:, :, body] = (
                        settings["initial_load"]
                        * settings["pigment_weights"][body]
                        * self._compact_brush(r2 / settings["load_radius"] ** 2)
                    )
        for texture in self.paint:
            texture.write(state.tobytes())

    def _dispatch(self, shader):
        shader.run(*self.groups)
        self.ctx.memory_barrier()

    def _flow_uniforms(self, frame):
        """Descriptor hook; the default retains the original conditioning exactly."""
        return tool_uniforms(frame, self.recipe["simulation"]["stir_radius"])

    def _flow(self, fraction):
        frame = self.source.frame(fraction)
        tools, pairs = self._flow_uniforms(frame)
        self.flow["u_tools"].write(tools.tobytes())
        self.flow["u_pairs"].write(pairs.tobytes())
        self.velocity.bind_to_image(0, read=False, write=True)
        self.maxima.bind_to_storage_buffer(1)
        self._dispatch(self.flow)
        maximum = float(np.frombuffer(self.maxima.read(), dtype="f4").max())
        if not math.isfinite(maximum):
            raise FloatingPointError("Nonfinite velocity field")
        return maximum

    def _transport(self, t0, t1):
        settings = self.recipe["simulation"]
        original, forward, backward, destination = self.paint
        dt = t1 - t0
        self.velocity.use(1)
        self.advect["u_velocity"].value = 1
        self.advect["u_input"].value = 0
        for source, target, interval in ((original, forward, dt), (forward, backward, -dt)):
            source.use(0)
            target.bind_to_image(0, read=False, write=True)
            self.advect["u_dt"].value = interval
            self._dispatch(self.advect)
        for unit, (name, texture) in enumerate(
            zip(
                ("u_original", "u_forward", "u_backward", "u_velocity"),
                (original, forward, backward, self.velocity),
                strict=True,
            )
        ):
            texture.use(unit)
            self.correct[name].value = unit
        a, b = self.source.frame(t0), self.source.frame(t1)
        segments = np.concatenate([a.positions, b.positions], axis=1).astype("f4")
        travel = np.maximum(0, b.arc_lengths - a.arc_lengths)
        dose = (
            settings["deposition"]
            * np.asarray(settings["pigment_weights"])
            * travel
            / (settings["brush_radius"] * math.sqrt(math.pi))
        )
        dose *= math.exp(-settings["fade"] * (t0 + t1) * 0.5)
        self.correct["u_dt"].value = dt
        self.correct["u_segments"].write(segments.tobytes())
        self.correct["u_doses"].value = tuple(dose)
        destination.bind_to_image(0, read=False, write=True)
        self._dispatch(self.correct)
        self.paint = [destination, forward, backward, original]
        self.internal_steps += 1
        if self.internal_steps > MAX_INTERNAL_STEPS:
            raise RuntimeError("Transport work cap exceeded; reduce flow or simulation resolution")

    def _source_travel(self, start, end):
        """Source-clock subdivision hook with unchanged default arithmetic."""
        return self.source.frame(end).arc_lengths - self.source.frame(start).arc_lengths

    @_current_context
    def advance_to(self, step):
        """Advance complete canonical steps; output cadence never changes physics."""
        if type(step) is not int or not self.step <= step <= self.steps:
            raise ValueError("Source step must move forward within the declared recording")
        settings = self.recipe["simulation"]
        pixel = 2.0 * self.domain / self.height
        for index in range(self.step, step):
            start, end = index / self.steps, (index + 1) / self.steps
            travel = self._source_travel(start, end)
            pieces = max(1, math.ceil(float(np.max(travel)) / (settings["brush_radius"] * 0.4)))
            t = start
            proposal = (end - start) / pieces
            while end - t > 1e-14:
                interval = min(proposal, end - t)
                for _ in range(16):
                    actual_travel = self._source_travel(t, t + interval)
                    ratio = float(np.max(actual_travel)) / (settings["brush_radius"] * 0.4)
                    if ratio > 1.0 + 1e-10:
                        interval *= min(0.8, 0.95 / ratio)
                        continue
                    maximum = self._flow(t + interval * 0.5)
                    courant = maximum * interval / pixel
                    if courant <= 1.5 + 1e-6:
                        break
                    interval *= 1.45 / courant
                else:
                    raise RuntimeError("Unable to bound characteristic transport")
                if interval < 1e-12:
                    raise RuntimeError("Transport interval is below supported source precision")
                self.maximum_courant = max(self.maximum_courant, courant)
                self._transport(t, min(end, t + interval))
                t += interval
            self.step = index + 1

    @_current_context
    def read_state(self):
        """Return canonical bottom-to-top float32 pigment state for exact restore."""
        state = (
            np.frombuffer(self.paint[0].read(), dtype="f4")
            .reshape(self.height, self.width, 4)
            .copy()
        )
        if not np.isfinite(state).all() or np.any(state < 0):
            raise FloatingPointError("Invalid pigment concentrations")
        return state

    @_current_context
    def restore(self, state, step, *, internal_steps=0, maximum_courant=0.0):
        """Restore a validated checkpoint without lossy conversion or interpolation."""
        values = np.asarray(state)
        if (
            values.dtype != np.float32
            or values.shape != (self.height, self.width, 4)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("Invalid float32 pigment checkpoint")
        if type(step) is not int or not 0 <= step <= self.steps:
            raise ValueError("Invalid checkpoint source step")
        if type(internal_steps) is not int or not step <= internal_steps <= MAX_INTERNAL_STEPS:
            raise ValueError("Invalid cumulative transport count")
        if not math.isfinite(maximum_courant) or not 0 <= maximum_courant <= 1.500001:
            raise ValueError("Invalid cumulative characteristic bound")
        self.paint[0].write(np.ascontiguousarray(values).tobytes())
        self.step = step
        self.internal_steps = internal_steps
        self.maximum_courant = maximum_courant

    @_current_context
    def render(self, width, height):
        """Shade pigment into top-to-bottom linear sRGB without advancing time."""
        if (
            type(width) is not int
            or type(height) is not int
            or min(width, height) < 1
            or width * height > 16_777_216
            or abs(width / height - self.aspect) > 1e-8
        ):
            raise ValueError("Output dimensions must be bounded and preserve the physical aspect")
        if self._output is None or self._output.size != (width, height):
            if self._framebuffer is not None:
                self._framebuffer.release()
                self._output.release()
            self._output = self.ctx.texture((width, height), 4, dtype="f4")
            self._framebuffer = self.ctx.framebuffer([self._output])
        self._framebuffer.use()
        self.ctx.viewport = (0, 0, width, height)
        self.paint[0].use(0)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
        data = (
            np.frombuffer(self._output.read(), dtype="f4")
            .reshape(height, width, 4)[::-1, :, :3]
            .copy()
        )
        if not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1.00001):
            raise FloatingPointError("Invalid linear pigment reflectance")
        return np.minimum(data, 1)

    def close(self):
        """Release the dedicated context and its resources."""
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
