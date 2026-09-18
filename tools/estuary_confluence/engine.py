"""Confluence Fresco: transported pigment over independently retained strata.

Four or six pigment channels share the prescribed Estuary stirring field. This
is an authored thin-paint model, not a Navier--Stokes or free-surface solver.
Mobile/deposit exchange and release from underpaint conserve each pigment
locally. Limited MacCormack advection is not globally mass conservative. Added
brush pigment is a separate source, never described as conserved initial mass.

The canonical clock is independent of output cadence. Snapshots are bottom-up,
include the guard domain, and preserve actual phase concentrations. Reduced
snapshots integrate those concentrations on the GPU without changing state.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from tools.estuary_studio.fresco import phase_exchange as phase_exchange
from tools.estuary_studio.fresco import substrate_field

ROOT = Path(__file__).parent
DEFAULTS = {
    "resolution": [1024, 768],
    "steps": 3600,
    "domain_scale": 1.6,
    "flow_strength": 1.1,
    "carrier_velocity": [2.0, 0.2],
    "stir_radius": 0.22,
    "pair_swirl": 0.9,
    "brush_radius": 0.035,
    "deposition": 0.025,
    "initial_load": 0.18,
    "initial_pattern": "pools",
    "load_radius": 0.28,
    "fade": 4.0,
    "drying": 2.0,
    "wetting": 0.8,
    "granulation": 0.55,
    "shoreline_strength": 1.2,
    "bloom_strength": 1.0,
    "underpaint_strength": 0.035,
    "underpaint_release": 1.0,
    "burial_rate": 1.6,
    "mixing_rate": 1.4,
    "height_scale_mm": 1.6,
    "substrate_um": 18.0,
}


def validate_config(value):
    """Resolve a bounded, flat configuration before allocating GPU resources."""
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Confluence config must contain only documented fields")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))
    resolution = result["resolution"]
    if (
        type(resolution) not in (list, tuple)
        or len(resolution) != 2
        or any(type(v) is not int or not 32 <= v <= 8192 for v in resolution)
        or resolution[0] * resolution[1] > 40_000_000
        or not 0.2 <= resolution[0] / resolution[1] <= 5
    ):
        raise ValueError("Invalid simulation resolution")
    result["resolution"] = list(resolution)
    if type(result["steps"]) is not int or not 1 <= result["steps"] <= 40000:
        raise ValueError("steps must be an integer in [1, 40000]")
    bounds = {
        "domain_scale": (1, 2.5),
        "flow_strength": (0, 8),
        "stir_radius": (0.02, 2),
        "pair_swirl": (0, 8),
        "brush_radius": (0.002, 1),
        "deposition": (0, 100),
        "initial_load": (0, 10),
        "load_radius": (0.001, 2),
        "fade": (0, 20),
        "drying": (0, 30),
        "wetting": (0, 10),
        "granulation": (0, 1),
        "shoreline_strength": (0, 5),
        "bloom_strength": (0, 5),
        "underpaint_strength": (0, 2),
        "underpaint_release": (0, 10),
        "burial_rate": (0, 10),
        "mixing_rate": (0, 10),
        "height_scale_mm": (0, 20),
        "substrate_um": (0, 100),
    }
    for key, (low, high) in bounds.items():
        v = result[key]
        try:
            valid = type(v) in (int, float) and math.isfinite(v) and low <= v <= high
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{key} must be finite and in [{low}, {high}]")
        result[key] = float(v)
    carrier = result["carrier_velocity"]
    if (
        type(carrier) not in (list, tuple)
        or len(carrier) != 2
        or any(type(v) not in (int, float) or not -8 <= v <= 8 for v in carrier)
    ):
        raise ValueError("carrier_velocity requires two finite values in [-8, 8]")
    result["carrier_velocity"] = list(map(float, carrier))
    if result["initial_pattern"] not in ("strata", "pools"):
        raise ValueError("initial_pattern must be strata or pools")
    return result


def _palette_arrays(palette):
    """Validate numerical engine inputs without relying on a generator version."""
    count = len(palette["pigments_srgb"])
    if count not in (4, 6):
        raise ValueError("Confluence requires three or five colors plus chalk")
    arrays = {}
    for key in ("settling", "release", "specific_volumes", "granulation"):
        values = np.asarray(palette[key], dtype="f4")
        if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid palette {key}")
        arrays[key] = values
    mixtures = np.asarray(palette["body_mixtures"], dtype="f4")
    if (
        mixtures.shape != (3, count)
        or not np.isfinite(mixtures).all()
        or np.any(mixtures < 0)
        or not np.allclose(mixtures.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("body_mixtures must be three normalized pigment recipes")
    for key in ("chalk_index", "underpaint_index"):
        if type(palette[key]) is not int or not 0 <= palette[key] < count:
            raise ValueError(f"Invalid palette {key}")
    if palette["chalk_index"] == palette["underpaint_index"]:
        raise ValueError("Underpaint color cannot be chalk")
    arrays["body_mixtures"] = mixtures
    weights = np.asarray(palette["body_weights"], dtype="f4")
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.any(weights > 10)
    ):
        raise ValueError("body_weights must contain three finite nonnegative amounts")
    arrays["body_weights"] = weights
    return count, arrays


def validate_events(events):
    """Validate source-space rewetting events; no frame numbers are accepted."""
    if type(events) is not list or len(events) > 3:
        raise ValueError("At most three planned confluences are supported")
    result = copy.deepcopy(events)
    for event in result:
        if type(event) is not dict:
            raise ValueError("Event must be an object")
        for key, low, high in (
            ("fraction", 0, 1),
            ("radius", 0.005, 2),
            ("duration", 0.001, 0.5),
            ("strength", 0, 5),
        ):
            value = event.get(key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid event {key}")
        p = np.asarray(event.get("position"), dtype=np.float64)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("Event position must be a finite projected point")
    return result


def _pulse_integral(start, end, center, duration):
    """Integrated Gaussian pulse; duration is sigma in source-fraction units.

    Normalize over the complete recording so boundary encounters receive their
    declared total water exposure as well. Integration is additive under any
    adaptive transport partition, unlike point-sampled event amplitudes.
    """

    def integral(t):
        return 0.5 * (1 + math.erf((t - center) / (duration * math.sqrt(2))))

    return (integral(end) - integral(start)) / (integral(1) - integral(0))


def event_doses(events, start, end, strength=1):
    """Exact integrated event exposure independent of canonical subdivision."""
    output = np.zeros((3, 4), dtype="f4")
    for index, event in enumerate(events):
        output[index, :2] = event["position"]
        output[index, 2] = event["radius"]
        output[index, 3] = (
            event["strength"]
            * strength
            * _pulse_integral(start, end, event["fraction"], event["duration"])
        )
    return output


def reduction_factor(size, resolution):
    """Only equal integer box reductions retain the physical aspect and mean."""
    if resolution is None:
        return 1
    if (
        type(resolution) not in (list, tuple)
        or len(resolution) != 2
        or any(type(v) is not int or v < 1 for v in resolution)
    ):
        raise ValueError("Snapshot resolution must contain two positive integers")
    w, h = size
    a, b = resolution
    if w % a or h % b or w // a != h // b or a > w or b > h:
        raise ValueError("Snapshot size must divide both simulation dimensions by the same integer")
    return w // a


def substrate_seed(value):
    """Decode exact archived entropy, retaining legacy integer proof archives.

    JSON uint256 values cannot round-trip through common JavaScript consumers.
    New palettes use a hexadecimal string; both representations feed exactly
    the same integer into the existing PCG64 seed reduction.
    """
    if type(value) not in (int, str):
        raise ValueError("substrate_seed must be an integer or hexadecimal string")
    try:
        decoded = int(value, 0) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValueError("Invalid substrate_seed") from exc
    if not 0 <= decoded < 2**256:
        raise ValueError("substrate_seed must contain at most 256 unsigned bits")
    return decoded % (2**63 - 1)


class Engine:
    """Four/six-channel material simulation on a dedicated OpenGL context."""

    def __init__(self, source, config, palette, events, backend="egl"):
        from dataclasses import asdict

        from tools.estuary.engine import MAX_INTERNAL_STEPS, _current_context
        from tools.estuary.engine import Engine as Transport
        from tools.estuary.optics import Material

        self.config = settings = validate_config(config)
        self.palette = copy.deepcopy(palette)
        self.events = events = validate_events(events)
        count, arrays = _palette_arrays(palette)
        transport_keys = (
            "resolution",
            "steps",
            "domain_scale",
            "flow_strength",
            "carrier_velocity",
            "stir_radius",
            "pair_swirl",
            "brush_radius",
            "deposition",
            "initial_load",
            "initial_pattern",
            "load_radius",
            "fade",
        )
        recipe = {
            "simulation": {k: settings[k] for k in transport_keys},
            "optics": asdict(Material()),
        }
        recipe["simulation"]["pigment_weights"] = [1.0, 1.0, 1.0]

        class PaintEngine(Transport):
            def _initialize(gpu):
                # Base Engine calls this hook before the packed state exists.
                # Initialization below replaces all textures before first use.
                pass

            def __init__(gpu):
                try:
                    super().__init__(source, recipe, backend)
                    with gpu.ctx:
                        gpu.correct.release()
                        gpu.correct = gpu.ctx.compute_shader(
                            (ROOT / "shaders/correct.glsl").read_text()
                        )
                        gpu.phase = gpu.ctx.compute_shader(
                            (ROOT / "shaders/phase.glsl").read_text()
                        )
                        gpu.carrier_phase = gpu.ctx.compute_shader(
                            (ROOT / "shaders/carrier.glsl").read_text()
                        )
                        gpu.reduce = gpu.ctx.compute_shader(
                            (ROOT / "shaders/reduce.glsl").read_text()
                        )
                        for shader in (gpu.correct, gpu.phase, gpu.carrier_phase):
                            for key, value in (
                                ("u_size", (gpu.width, gpu.height)),
                                ("u_aspect", gpu.aspect),
                                ("u_domain", gpu.domain),
                            ):
                                if key in shader:
                                    shader[key].value = value
                        gpu.correct["u_brush"].value = settings["brush_radius"]
                        gpu.carrier_phase["u_brush"].value = settings["brush_radius"]
                        for key in ("drying", "wetting", "mixing_rate"):
                            gpu.carrier_phase["u_" + key].value = settings[key]
                        for key in (
                            "shoreline_strength",
                            "granulation",
                            "underpaint_release",
                            "burial_rate",
                        ):
                            gpu.phase["u_" + key].value = settings[key]
                        gpu.blocks = [gpu.paint]
                        if count > 4:
                            gpu.blocks.append([gpu._texture(4) for _ in range(4)])
                        gpu.deposits = [[gpu._texture(4) for _ in range(2)] for _ in gpu.blocks]
                        gpu.underpaints = [[gpu._texture(4) for _ in range(2)] for _ in gpu.blocks]
                        gpu.carrier = [gpu._texture(4) for _ in range(4)]
                        gpu.tooth = gpu._texture(4)
                        gpu.tooth.write(
                            substrate_field(
                                gpu.width,
                                gpu.height,
                                gpu.domain,
                                substrate_seed(palette["substrate_seed"]),
                            ).tobytes()
                        )
                        gpu._snapshot_texture = None
                        gpu._initialize_paint()
                except Exception:
                    if getattr(gpu, "ctx", None) is not None:
                        gpu.close()
                    raise

            def _initialize_paint(gpu):
                x = (
                    (np.arange(gpu.width, dtype=np.float32) + 0.5) / gpu.width * 2 * gpu.aspect
                    - gpu.aspect
                ) * gpu.domain
                y = (
                    (np.arange(gpu.height, dtype=np.float32) + 0.5) / gpu.height * 2 - 1
                ) * gpu.domain
                initial = source.frame(0).positions
                edge = initial[1] - initial[0]
                direction = edge / max(float(np.linalg.norm(edge)), 1e-10)
                if np.linalg.norm(direction) < 0.5:
                    direction = np.array([1.0, 0.0])
                center = initial.mean(axis=0)
                distance = (x[None, :] - center[0]) * -direction[1] + (
                    y[:, None] - center[1]
                ) * direction[0]
                radius = settings["load_radius"]

                def band(offset, width):
                    ramp = np.clip((np.abs(distance - offset) - width * 0.9) / (width * 0.1), 0, 1)
                    return (1 - ramp * ramp * (3 - 2 * ramp)).astype("f4")

                state = np.zeros((gpu.height, gpu.width, count), dtype="f4")
                mixtures = arrays["body_mixtures"]
                if settings["initial_pattern"] == "strata":
                    white = np.maximum.reduce(
                        [
                            band(0, radius * 0.75),
                            band(-radius * 1.65, radius * 0.075),
                            band(radius * 2.15, radius * 0.045),
                        ]
                    )
                    counter = band(-radius * 0.98, radius * 0.24) * (1 - white)
                    accent = band(radius * 1.45, radius * 0.09) * (1 - white - counter)
                    for channel in range(count):
                        state[..., channel] = (
                            mixtures[0, channel] * (1 - white - counter - accent)
                            + mixtures[1, channel] * (counter + white * 0.12)
                            + mixtures[2, channel] * accent
                        )
                    state[..., palette["chalk_index"]] += white * 0.88
                else:
                    for body, p in enumerate(initial):
                        r2 = (x[None, :] - p[0]) ** 2 + (y[:, None] - p[1]) ** 2
                        load = gpu._compact_brush(r2 / radius**2)
                        state += load[..., None] * mixtures[body] * arrays["body_weights"][body]
                state *= settings["initial_load"]
                # The quiet buried accent is spatially tied to the initial
                # triangle's strata. This is actual pigment, never an overlay.
                under = np.zeros_like(state)
                buried = band(radius * 0.12, radius * 0.68)
                if settings["initial_pattern"] == "pools":
                    # A buried accent is confined to the loaded paint. A
                    # canvas-spanning stratum beneath isolated pools would
                    # remain an unrelated straight stripe in the final image.
                    buried = np.clip(state.sum(axis=-1) / max(settings["initial_load"], 1e-9), 0, 1)
                under[..., palette["underpaint_index"]] = buried * settings["underpaint_strength"]
                zeros = np.zeros((gpu.height, gpu.width, 4), dtype="f4")
                for index, (block, deposits, underpaints) in enumerate(
                    zip(gpu.blocks, gpu.deposits, gpu.underpaints, strict=True)
                ):
                    channels = min(4, count - index * 4)
                    values = zeros.copy()
                    values[..., :channels] = state[..., index * 4 : index * 4 + channels]
                    for texture in block:
                        texture.write(values.tobytes())
                    for texture in deposits:
                        texture.write(zeros.tobytes())
                    values[..., :channels] = under[..., index * 4 : index * 4 + channels]
                    for texture in underpaints:
                        texture.write(values.tobytes())
                zeros[..., 0] = 1.0
                zeros[..., 1] = 0.0
                zeros[..., 2] = 1.0
                for texture in gpu.carrier:
                    texture.write(zeros.tobytes())

            def _advect_block(gpu, block, dt, segments, doses, signed=False):
                original, forward, backward, destination = block
                gpu.velocity.use(1)
                gpu.advect["u_velocity"].value = 1
                gpu.advect["u_input"].value = 0
                for src, target, interval in ((original, forward, dt), (forward, backward, -dt)):
                    src.use(0)
                    target.bind_to_image(0, read=False, write=True)
                    gpu.advect["u_dt"].value = interval
                    gpu._dispatch(gpu.advect)
                for unit, (name, texture) in enumerate(
                    zip(
                        ("u_original", "u_forward", "u_backward", "u_velocity"),
                        (original, forward, backward, gpu.velocity),
                        strict=True,
                    )
                ):
                    texture.use(unit)
                    gpu.correct[name].value = unit
                gpu.correct["u_dt"].value = dt
                gpu.correct["u_signed"].value = signed
                gpu.correct["u_segments"].write(segments.tobytes())
                gpu.correct["u_doses"].write(doses.tobytes())
                destination.bind_to_image(0, read=False, write=True)
                gpu._dispatch(gpu.correct)
                return [destination, forward, backward, original]

            def _transport(gpu, t0, t1):
                dt = t1 - t0
                a, b = source.frame(t0), source.frame(t1)
                segments = np.concatenate([a.positions, b.positions], axis=1).astype("f4")
                travel = np.maximum(0, b.arc_lengths - a.arc_lengths)
                fade = math.exp(-settings["fade"] * (t0 + t1) * 0.5)
                amounts = (
                    settings["deposition"]
                    * travel
                    * fade
                    * arrays["body_weights"]
                    / (settings["brush_radius"] * math.sqrt(math.pi))
                )
                doses = amounts[:, None] * arrays["body_mixtures"]
                for index, block in enumerate(gpu.blocks):
                    packed = np.zeros((3, 4), dtype="f4")
                    channels = min(4, count - index * 4)
                    packed[:, :channels] = doses[:, index * 4 : index * 4 + channels]
                    gpu.blocks[index] = gpu._advect_block(block, dt, segments, packed)
                gpu.carrier = gpu._advect_block(
                    gpu.carrier, dt, segments, np.zeros((3, 4), dtype="f4"), True
                )
                for unit, (name, texture) in enumerate(
                    (
                        ("u_input", gpu.carrier[0]),
                        ("u_tooth", gpu.tooth),
                        ("u_velocity", gpu.velocity),
                    )
                ):
                    texture.use(unit)
                    gpu.carrier_phase[name].value = unit
                gpu.carrier_phase["u_dt"].value = dt
                gpu.carrier_phase["u_time"].value = (t0 + t1) * 0.5
                gpu.carrier_phase["u_fade"].value = fade
                gpu.carrier_phase["u_segments"].write(segments.tobytes())
                gpu.carrier_phase["u_travel"].value = tuple(travel)
                gpu.carrier_phase["u_events"].write(
                    event_doses(events, t0, t1, settings["bloom_strength"]).tobytes()
                )
                gpu.carrier[3].bind_to_image(0, read=False, write=True)
                gpu._dispatch(gpu.carrier_phase)
                gpu.carrier[0], gpu.carrier[3] = gpu.carrier[3], gpu.carrier[0]
                gpu.phase["u_dt"].value = dt
                gpu.phase["u_has_other"].value = len(gpu.blocks) > 1
                for index, (block, deposits, underpaints) in enumerate(
                    zip(gpu.blocks, gpu.deposits, gpu.underpaints, strict=True)
                ):
                    other = (index + 1) % len(gpu.blocks)
                    bindings = (
                        ("u_mobile", block[0]),
                        ("u_deposit", deposits[0]),
                        ("u_underpaint", underpaints[0]),
                        ("u_carrier", gpu.carrier[0]),
                        ("u_tooth", gpu.tooth),
                        ("u_velocity", gpu.velocity),
                        ("u_mobile_other", gpu.blocks[other][0]),
                        ("u_deposit_other", gpu.deposits[other][0]),
                    )
                    for unit, (name, texture) in enumerate(bindings):
                        texture.use(unit)
                        gpu.phase[name].value = unit
                    for key, uniform in (
                        ("settling", "u_settling"),
                        ("release", "u_release"),
                        ("granulation", "u_grain"),
                    ):
                        values = np.zeros(4, dtype="f4")
                        channels = min(4, count - index * 4)
                        values[:channels] = arrays[key][index * 4 : index * 4 + channels]
                        gpu.phase[uniform].value = tuple(values)
                    block[3].bind_to_image(0, read=False, write=True)
                    deposits[1].bind_to_image(1, read=False, write=True)
                    underpaints[1].bind_to_image(2, read=False, write=True)
                    gpu._dispatch(gpu.phase)
                for block, deposits, underpaints in zip(
                    gpu.blocks, gpu.deposits, gpu.underpaints, strict=True
                ):
                    block[0], block[3] = block[3], block[0]
                    deposits.reverse()
                    underpaints.reverse()
                gpu.paint = gpu.blocks[0]
                gpu.internal_steps += 1
                if gpu.internal_steps > MAX_INTERNAL_STEPS:
                    raise RuntimeError("Confluence transport work cap exceeded")

            def _read(gpu, texture, factor):
                width, height = gpu.width // factor, gpu.height // factor
                if factor == 1:
                    data = texture.read()
                else:
                    if gpu._snapshot_texture is None or gpu._snapshot_texture.size != (
                        width,
                        height,
                    ):
                        if gpu._snapshot_texture is not None:
                            gpu._snapshot_texture.release()
                        gpu._snapshot_texture = gpu.ctx.texture((width, height), 4, dtype="f4")
                    texture.use(0)
                    gpu.reduce["u_input"].value = 0
                    gpu.reduce["u_output_size"].value = (width, height)
                    gpu.reduce["u_factor"].value = factor
                    gpu._snapshot_texture.bind_to_image(0, read=False, write=True)
                    gpu.reduce.run((width + 15) // 16, (height + 15) // 16)
                    gpu.ctx.memory_barrier()
                    data = gpu._snapshot_texture.read()
                return np.frombuffer(data, dtype="f4").reshape(height, width, 4).copy()

            @_current_context
            def snapshot(gpu, resolution=None):
                factor = reduction_factor((gpu.width, gpu.height), resolution)

                def packed(groups):
                    return np.concatenate(
                        [gpu._read(group[0], factor) for group in groups], axis=-1
                    )[..., :count]

                mobile, deposit, underpaint = (
                    packed(gpu.blocks),
                    packed(gpu.deposits),
                    packed(gpu.underpaints),
                )
                carrier, tooth = gpu._read(gpu.carrier[0], factor), gpu._read(gpu.tooth, factor)
                pigment = mobile + deposit + underpaint
                total = pigment.sum(axis=-1)
                wetness = np.clip(carrier[..., 0], 0, 1)
                mixture = np.clip(carrier[..., 1], 0, 1)
                direction = carrier[..., 2:4]
                norms = np.linalg.norm(direction, axis=-1)
                direction /= np.maximum(norms[..., None], 1e-9)
                direction[norms < 1e-9] = (1, 0)
                dry_share = np.divide(
                    (deposit + underpaint).sum(axis=-1),
                    total,
                    out=np.zeros_like(total),
                    where=total > 1e-9,
                )
                white = np.divide(
                    pigment[..., palette["chalk_index"]],
                    total,
                    out=np.zeros_like(total),
                    where=total > 1e-9,
                )
                mass_height = np.einsum(
                    "...i,i->...", deposit + underpaint + mobile * 0.22, arrays["specific_volumes"]
                )
                height = mass_height * settings["height_scale_mm"] * 0.001
                height += settings["substrate_um"] * 1e-6 * tooth[..., 3]
                roughness = np.clip(
                    0.3
                    + 0.26 * dry_share
                    + 0.11 * white
                    + 0.06 * tooth[..., 0] * dry_share
                    - 0.12 * wetness,
                    0.12,
                    0.88,
                )
                result = {
                    "pigment": pigment,
                    "mobile": mobile,
                    "deposit": deposit,
                    "underpaint": underpaint,
                    "wetness": wetness,
                    "mixing": mixture,
                    "direction": direction,
                    "height": height,
                    "roughness": roughness,
                    "coverage": -np.expm1(-total * 3),
                }
                if any(not np.isfinite(v).all() for v in result.values()):
                    raise FloatingPointError("Nonfinite Confluence material state")
                return {
                    key: np.ascontiguousarray(value, dtype="f4") for key, value in result.items()
                }

        self._gpu = PaintEngine()
        self.metadata = {
            **self._gpu.metadata,
            "model": "Confluence Fresco / authored thin pigment strata",
            "pigment_channels": count,
            "phase_exchange": (
                "exact reversible mobile/deposit exchange; conservative underpaint release"
            ),
            "mass_limitations": (
                "interpolated transport may drift; trajectories add new pigment; "
                "water events add only carrier"
            ),
            "mixing": (
                "persistent transported degree of intimate mixing; "
                "finite-rate strain/wetness response"
            ),
            "blooms": (
                "source-encounter rewetting and subsequent prescribed-flow transport; "
                "no radial force or added pigment"
            ),
            "height": "pigment specific-volume interpretation in metres, visible width 0.4 m",
            "snapshot_reduction": (
                "GPU exact integer area integration of phase concentrations before appearance"
            ),
        }

    @property
    def steps(self):
        return self._gpu.steps

    @property
    def step(self):
        return self._gpu.step

    def advance_to(self, step):
        self._gpu.advance_to(step)

    def snapshot(self, resolution=None):
        return self._gpu.snapshot(resolution)

    def close(self):
        self._gpu.close()
