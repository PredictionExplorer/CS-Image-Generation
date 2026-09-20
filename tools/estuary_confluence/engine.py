"""Confluence Fresco: transported pigment over independently retained strata.

Two, three, four, or six pigment channels share the prescribed Estuary stirring field. This
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
from bisect import bisect_right
from pathlib import Path

import numpy as np

from tools.estuary_studio.fresco import phase_exchange as phase_exchange
from tools.estuary_studio.fresco import substrate_field

from .mass_budget import RELATIVE_TOLERANCE as MASS_BUDGET_RTOL
from .mass_budget import VERSION as MASS_BUDGET_VERSION
from .mass_budget import correction_steps, validate_initial_weights
from .palette import SUPPORTED_CHROMATIC_COUNTS

ROOT = Path(__file__).parent
DIFFUSION_CFL = 0.24
MAX_DIFFUSION_SUBSTEPS = 256
MAX_DIFFUSION_STEPS = 500_000
DEFAULTS = {
    "material_model": "legacy",
    "lower_transport_scale": 0.78,
    "interlayer_exchange_rate": 0.18,
    "interlayer_min_concentration": 0.001,
    "resolution": [1024, 768],
    "steps": 3600,
    "domain_scale": 1.6,
    "flow_domain_scale": None,
    "flow_strength": 1.1,
    "carrier_velocity": [2.0, 0.2],
    "stir_radius": 0.22,
    "pair_swirl": 0.9,
    "pair_strain": 0.0,
    "brush_radius": 0.035,
    "deposition": 0.025,
    "initial_load": 0.18,
    "initial_pigment_weights": None,
    "initial_pattern": "pools",
    "initial_edge_width": 0.02,
    "load_radius": 0.28,
    "fade": 4.0,
    "drying": 2.0,
    "wetting": 0.8,
    "granulation": 0.55,
    "settling_scale": 1.0,
    "shoreline_strength": 1.2,
    "bloom_strength": 1.0,
    "underpaint_strength": 0.035,
    "underpaint_release": 1.0,
    "burial_rate": 1.6,
    "mixing_rate": 1.4,
    "diffusion_coefficient": 0.0,
    "diffusion_min_concentration": 0.001,
    "mass_budget_interval_steps": 0,
    "height_scale_mm": 1.6,
    "substrate_um": 18.0,
}


def validate_config(value):
    """Resolve a bounded, flat configuration before allocating GPU resources."""
    if type(value) is not dict or set(value) - (
        set(DEFAULTS) | {"interaction", "initial_composition", "body_influence"}
    ):
        raise ValueError("Confluence config must contain only documented fields")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))
    from .body_influence import validate_config as influence_config

    influence = influence_config(result.pop("body_influence", None))
    if influence is not None:
        result["body_influence"] = influence
    if result["material_model"] not in ("legacy", "laminate"):
        raise ValueError("material_model must be legacy or laminate")
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
    interval = result["mass_budget_interval_steps"]
    if type(interval) is not int or not 0 <= interval <= 40000:
        raise ValueError("mass_budget_interval_steps must be an integer in [0, 40000]")
    bounds = {
        "lower_transport_scale": (0.25, 1),
        "interlayer_exchange_rate": (0, 10),
        "interlayer_min_concentration": (1e-8, 1),
        "domain_scale": (1, 2.5),
        "flow_strength": (0, 8),
        "stir_radius": (0.02, 2),
        "pair_swirl": (0, 8),
        "pair_strain": (0, 4),
        "brush_radius": (0.002, 1),
        "deposition": (0, 100),
        "initial_load": (0, 10),
        "initial_edge_width": (0.001, 0.25),
        "load_radius": (0.001, 2),
        "fade": (0, 20),
        "drying": (0, 30),
        "wetting": (0, 10),
        "granulation": (0, 1),
        "settling_scale": (0, 10),
        "shoreline_strength": (0, 5),
        "bloom_strength": (0, 5),
        "underpaint_strength": (0, 2),
        "underpaint_release": (0, 10),
        "burial_rate": (0, 10),
        "mixing_rate": (0, 10),
        "diffusion_coefficient": (0, 0.01),
        "diffusion_min_concentration": (1e-8, 1),
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
    flow_domain = result["flow_domain_scale"]
    if flow_domain is None:
        flow_domain = result["domain_scale"]
    if type(flow_domain) not in (int, float) or not 1 <= flow_domain <= result["domain_scale"]:
        raise ValueError("flow_domain_scale must be within [1, domain_scale]")
    result["flow_domain_scale"] = float(flow_domain)
    if result["initial_pattern"] not in ("strata", "pools", "scattered", "engaged", "shaped"):
        raise ValueError("initial_pattern must be strata, pools, scattered, engaged or shaped")
    composition = result.pop("initial_composition", None)
    if result["initial_pattern"] == "shaped":
        from .initial_composition import validate_config as composition_config

        if composition is None:
            raise ValueError("Shaped initialization requires initial_composition")
        if result["material_model"] != "laminate":
            raise ValueError("Shaped initialization requires source-free laminate paint")
        if result["initial_pigment_weights"] is not None:
            raise ValueError("Shaped target masses already include pigment weights")
        result["initial_composition"] = composition_config(composition)
    elif composition is not None:
        raise ValueError("initial_composition is only valid with initial_pattern=shaped")
    if result["initial_pattern"] in ("scattered", "engaged"):
        from .layout import MAX_LOAD_RADIUS

        if result["underpaint_strength"] != 0:
            raise ValueError("Scattered pure pools require underpaint_strength=0")
        if result["load_radius"] > MAX_LOAD_RADIUS:
            raise ValueError(f"Scattered load_radius must not exceed {MAX_LOAD_RADIUS}")
    if interval and (
        result["initial_pattern"] not in ("scattered", "engaged", "shaped")
        or result["deposition"] != 0
        or result["settling_scale"] != 0
        or result["underpaint_strength"] != 0
    ):
        raise ValueError("Mass budgets require source-free scattered, engaged or shaped paint")
    if result["material_model"] == "laminate" and (
        result["initial_pattern"] not in ("scattered", "engaged", "shaped")
        or any(
            result[key] != 0
            for key in (
                "deposition",
                "settling_scale",
                "underpaint_strength",
                "underpaint_release",
                "burial_rate",
            )
        )
    ):
        raise ValueError("Laminate requires source-free initial paint and disabled legacy phases")
    interaction = result.pop("interaction", None)
    if interaction is not None:
        from .interaction import validate_config as interaction_config

        if result["material_model"] != "laminate":
            raise ValueError("Interaction history requires source-free laminate paint")
        result["interaction"] = interaction_config(interaction)
    result["initial_pigment_weights"] = validate_initial_weights(result["initial_pigment_weights"])
    if result["initial_pigment_weights"] is not None and (
        result["initial_pattern"] not in ("scattered", "engaged") or result["deposition"] != 0
    ):
        raise ValueError("Initial pigment weights require source-free scattered or engaged pools")
    return result


def _palette_arrays(palette):
    """Validate numerical engine inputs without relying on a generator version."""
    count = len(palette["pigments_srgb"])
    if count - 1 not in SUPPORTED_CHROMATIC_COUNTS:
        raise ValueError("Confluence requires one, two, three, or five colors plus chalk")
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


def diffusion_substeps(coefficient, dt, pixel_size):
    """Bound explicit interdiffusion in physical coordinates, not output pixels."""
    for value in (coefficient, dt, pixel_size):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Diffusion controls must be finite numbers")
    if coefficient < 0 or dt < 0 or pixel_size <= 0:
        raise ValueError("Diffusion requires nonnegative rates/time and positive cell size")
    if coefficient == 0 or dt == 0:
        return 0
    number = coefficient * dt / pixel_size**2
    if not math.isfinite(number):
        raise ValueError("Diffusion interval exceeds numerical limits")
    pieces = max(1, math.ceil(number / DIFFUSION_CFL))
    if pieces > MAX_DIFFUSION_SUBSTEPS:
        raise ValueError(
            "Diffusion interval exceeds its work cap; increase canonical steps or lower D"
        )
    return pieces


def interdiffusion_step(
    pigment, wetness, *, coefficient, dt, pixel_size, minimum_concentration=0.001
):
    """Float64 reference for one stable, fixed-thickness interdiffusion step.

    D has units of world-coordinate squared per complete source recording.
    Symmetric face fluxes exchange color fractions at fixed local film amount.
    This is an authored material model, not a molecularly calibrated diffusivity.
    """
    concentration = np.asarray(pigment, dtype=np.float64)
    wet = np.asarray(wetness, dtype=np.float64)
    if (
        concentration.ndim != 3
        or concentration.shape[:2] != wet.shape
        or not np.isfinite(concentration).all()
        or np.any(concentration < 0)
        or not np.isfinite(wet).all()
        or np.any(wet < 0)
        or np.any(wet > 1)
    ):
        raise ValueError("Invalid pigment or wetness field")
    if (
        type(minimum_concentration) not in (int, float)
        or not math.isfinite(minimum_concentration)
        or minimum_concentration <= 0
    ):
        raise ValueError("minimum_concentration must be finite and positive")
    pieces = diffusion_substeps(coefficient, dt, pixel_size)
    if pieces > 1:
        raise ValueError("Reference step exceeds the explicit stability bound")
    result = concentration.copy()
    if pieces == 0:
        return result
    amount = concentration.sum(axis=-1)
    fraction = np.divide(
        concentration,
        amount[..., None],
        out=np.zeros_like(concentration),
        where=amount[..., None] > 0,
    )
    active = amount > minimum_concentration
    scale = coefficient * dt / pixel_size**2
    for axis in (0, 1):
        lower, upper = [slice(None)] * 2, [slice(None)] * 2
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        face = (
            scale
            * np.minimum(wet[lower], wet[upper])
            * np.minimum(amount[lower], amount[upper])
            * active[lower]
            * active[upper]
        )
        flux = face[..., None] * (fraction[upper] - fraction[lower])
        result[lower] += flux
        result[upper] -= flux
    return result


def mass_budget_factors(initial_mass, current_mass):
    """Return the actual float32 channel multipliers; never invent absent paint."""
    initial = np.asarray(initial_mass, dtype="f8")
    current = np.asarray(current_mass, dtype="f8")
    if (
        initial.ndim != 1
        or current.shape != initial.shape
        or not np.isfinite(initial).all()
        or not np.isfinite(current).all()
        or np.any(initial < 0)
        or np.any(current < 0)
    ):
        raise ValueError("Mass budgets require matching finite nonnegative channel amounts")
    empty = initial == 0
    if np.any(current[empty] != 0):
        raise FloatingPointError("An initially empty pigment channel gained material")
    if np.any(current[~empty] <= 0):
        raise FloatingPointError("A pigment vanished; uniform budget restoration cannot recover it")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        factors = np.divide(initial, current, out=np.ones_like(initial), where=~empty).astype("f4")
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise FloatingPointError("Mass-budget factors exceed positive float32 limits")
    return factors


def _shaped_initial_state(layout, settings):
    """Use already-budgeted chromatic rasters, adding only the empty chalk channel."""
    from .initial_composition import rasterize

    width, height = settings["resolution"]
    chromatic = rasterize(layout, settings["resolution"], settings["domain_scale"])
    if (
        not isinstance(chromatic, np.ndarray)
        or chromatic.dtype != np.float32
        or chromatic.shape != (height, width, 3)
        or not np.isfinite(chromatic).all()
        or np.any(chromatic < 0)
        or np.any(chromatic > 1e6)
    ):
        raise ValueError("Initial composition must produce bounded native float32 pigment fields")
    actual = chromatic.sum(axis=(0, 1), dtype="f8") * (2 * settings["domain_scale"] / height) ** 2
    if not np.allclose(
        actual, settings["initial_composition"]["target_mass"], rtol=MASS_BUDGET_RTOL, atol=1e-12
    ):
        raise FloatingPointError("Initial composition does not meet its declared pigment targets")
    state = np.zeros((height, width, 4), dtype="f4")
    state[..., :3] = chromatic
    return state


class Engine:
    """Four/six-channel material simulation on a dedicated OpenGL context."""

    def __init__(self, source, config, palette, events, backend="egl"):
        from dataclasses import asdict

        from tools.estuary.engine import MAX_INTERNAL_STEPS, _current_context
        from tools.estuary.engine import Engine as Transport
        from tools.estuary.optics import Material

        self.config = settings = validate_config(config)
        from .body_influence import (
            arc_travel,
            forcing_uniforms,
            initialization_config,
            movement_segments,
            validate_event_eligibility,
        )

        influence = settings.get("body_influence")
        self.palette = copy.deepcopy(palette)
        self.events = events = validate_events(events)
        validate_event_eligibility(events, influence)
        count, arrays = _palette_arrays(palette)
        shaped = settings["initial_pattern"] == "shaped"
        if shaped and (count != 4 or palette["chalk_index"] != 3):
            raise ValueError(
                "Shaped initialization requires exactly three pigments followed by chalk"
            )
        initial_weights = validate_initial_weights(settings["initial_pigment_weights"], count - 1)
        laminate = settings["material_model"] == "laminate"
        if laminate:
            from .laminate import layer_fractions

            arrays["layer_fractions"] = layer_fractions(palette, count)
        self.layout = layout = None
        if shaped:
            from .initial_composition import plan_layout

            composition = settings["initial_composition"]
            self.layout = layout = plan_layout(
                palette["seed"],
                count - 1,
                settings["resolution"][0] / settings["resolution"][1],
                composition,
                source=source if composition["setup"] == "body-wedges" else None,
            )
        elif settings["initial_pattern"] in ("scattered", "engaged"):
            from .layout import plan_layout

            if palette["chalk_index"] != count - 1:
                raise ValueError("Scattered palettes must list chromatic pigments before chalk")
            if settings["initial_pattern"] == "engaged":
                from .participation_layout import plan_engaged_layout

                self.layout = layout = plan_engaged_layout(
                    source, count - 1, initialization_config(settings)
                )
            else:
                self.layout = layout = plan_layout(
                    palette["seed"],
                    count - 1,
                    settings["resolution"][0] / settings["resolution"][1],
                    load_radius=settings["load_radius"],
                    initial_load=settings["initial_load"],
                    edge_width=settings["initial_edge_width"],
                )
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
                gpu.interaction = None
                try:
                    super().__init__(source, recipe, backend)
                    with gpu.ctx:
                        if settings["flow_domain_scale"] != gpu.domain or settings["pair_strain"]:
                            gpu.flow.release()
                            gpu.flow = gpu.ctx.compute_shader(
                                (ROOT / "shaders/flow.glsl").read_text()
                            )
                            for key, value in {
                                "u_size": (gpu.width, gpu.height),
                                "u_aspect": gpu.aspect,
                                "u_domain": gpu.domain,
                                "u_flow_domain": settings["flow_domain_scale"],
                                "u_radius": settings["stir_radius"],
                                "u_strength": settings["flow_strength"],
                                "u_pair_gain": settings["pair_swirl"],
                                "u_pair_strain": settings["pair_strain"],
                                "u_carrier": tuple(settings["carrier_velocity"]),
                            }.items():
                                gpu.flow[key].value = value
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
                        gpu.diffusion = None
                        gpu.laminate_exchange = None
                        if laminate and settings["interlayer_exchange_rate"]:
                            gpu.laminate_exchange = gpu.ctx.compute_shader(
                                (ROOT / "shaders/laminate-exchange.glsl").read_text()
                            )
                            gpu.laminate_exchange["u_size"].value = (gpu.width, gpu.height)
                            gpu.laminate_exchange["u_has_other"].value = count > 4
                            gpu.laminate_exchange["u_minimum"].value = settings[
                                "interlayer_min_concentration"
                            ]
                        gpu.diffusion_steps = 0
                        gpu.maximum_diffusion_number = 0.0
                        gpu.mass_budget_records = []
                        gpu.mass_budget_initial = None
                        gpu.mass_budget_schedule = correction_steps(
                            gpu.steps, settings["mass_budget_interval_steps"]
                        )
                        gpu.mass_reduce = gpu.mass_scale = gpu.mass_partials = None
                        if settings["mass_budget_interval_steps"]:
                            gpu.mass_reduce = gpu.ctx.compute_shader(
                                (ROOT / "shaders/mass-reduce.glsl").read_text()
                            )
                            gpu.mass_scale = gpu.ctx.compute_shader(
                                (ROOT / "shaders/mass-scale.glsl").read_text()
                            )
                            for shader in (gpu.mass_reduce, gpu.mass_scale):
                                shader["u_size"].value = (gpu.width, gpu.height)
                                shader["u_input"].value = 0
                            gpu.mass_partials = gpu.ctx.buffer(
                                reserve=gpu.groups[0] * gpu.groups[1] * 4 * 4
                            )
                        if settings["diffusion_coefficient"] > 0:
                            gpu.diffusion = gpu.ctx.compute_shader(
                                (ROOT / "shaders/interdiffuse.glsl").read_text()
                            )
                            gpu.diffusion["u_size"].value = (gpu.width, gpu.height)
                            gpu.diffusion["u_minimum"].value = settings[
                                "diffusion_min_concentration"
                            ]
                            gpu.diffusion["u_has_other"].value = count > 4
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
                        gpu.underpaints = [
                            [gpu._texture(4) for _ in range(4 if laminate else 2)]
                            for _ in gpu.blocks
                        ]
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
                        gpu.skip_phase = (
                            settings["settling_scale"] == 0 and settings["underpaint_strength"] == 0
                        )
                        gpu._snapshot_texture = None
                        gpu._initialize_paint()
                        if settings.get("interaction") is not None:
                            from .interaction import GPUInteraction

                            gpu.interaction = GPUInteraction(
                                gpu.ctx,
                                (gpu.width, gpu.height),
                                gpu.aspect,
                                gpu.domain,
                                palette["seed"],
                                settings["interaction"],
                                settings["lower_transport_scale"],
                            )
                            gpu.interaction.initialize(
                                tuple(block[0] for block in gpu.blocks),
                                tuple(block[0] for block in gpu.underpaints),
                            )
                except Exception:
                    if getattr(gpu, "ctx", None) is not None:
                        gpu.close()
                    raise

            def close(gpu):
                if getattr(gpu, "ctx", None) is not None and gpu.interaction is not None:
                    with gpu.ctx:
                        gpu.interaction.close()
                    gpu.interaction = None
                super().close()

            def _flow(gpu, fraction):
                # The base transport retains its exact old shader and arithmetic
                # when strain is disabled. Descriptors follow the source clock,
                # including adaptive substeps, never the movie frame cadence.
                if influence is None and settings["pair_strain"]:
                    from .pair_strain import pair_strain_uniforms

                    strains = pair_strain_uniforms(source.frame(fraction), settings["stir_radius"])
                    gpu.flow["u_strains"].write(strains.tobytes())
                return super()._flow(fraction)

            def _flow_uniforms(gpu, frame):
                if influence is None:
                    return super()._flow_uniforms(frame)
                tools, pairs, strains = forcing_uniforms(
                    frame,
                    settings["stir_radius"],
                    influence,
                    include_strain=settings["pair_strain"] > 0,
                )
                if strains is not None:
                    gpu.flow["u_strains"].write(strains.tobytes())
                return tools, pairs

            def _source_travel(gpu, start, end):
                if influence is None:
                    return super()._source_travel(start, end)
                # Keep proposal and retry subdivision independent of inactive
                # paths, using the original complete-source clock.
                after, before = source.frame(end), source.frame(start)
                return arc_travel(before.arc_lengths, after.arc_lengths, influence)

            def _initialize_paint(gpu):
                if shaped:
                    state = _shaped_initial_state(layout, settings)
                else:
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
                        ramp = np.clip(
                            (np.abs(distance - offset) - width * 0.9) / (width * 0.1), 0, 1
                        )
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
                    elif settings["initial_pattern"] == "pools":
                        for body, p in enumerate(initial):
                            r2 = (x[None, :] - p[0]) ** 2 + (y[:, None] - p[1]) ** 2
                            load = gpu._compact_brush(r2 / radius**2)
                            state += load[..., None] * mixtures[body] * arrays["body_weights"][body]
                    else:
                        from .layout import pool_profile

                        for pool in layout["pools"]:
                            load = pool["load"]
                            if initial_weights is not None:
                                load *= initial_weights[pool["pigment_index"]]
                            p = pool["position"]
                            distance_to_pool = np.sqrt(
                                (x[None, :] - p[0]) ** 2 + (y[:, None] - p[1]) ** 2
                            )
                            state[..., pool["pigment_index"]] = load * pool_profile(
                                distance_to_pool, pool["radius"], pool["edge_width"]
                            )
                    if settings["initial_pattern"] not in ("scattered", "engaged"):
                        state *= settings["initial_load"]
                if settings["mass_budget_interval_steps"]:
                    gpu.mass_budget_initial = (
                        state.sum(axis=(0, 1), dtype="f8") * (2 * gpu.domain / gpu.height) ** 2
                    )
                # The quiet buried accent is spatially tied to the initial
                # triangle's strata. This is actual pigment, never an overlay.
                under = np.zeros_like(state)
                if not shaped:
                    buried = band(radius * 0.12, radius * 0.68)
                    if settings["initial_pattern"] == "pools":
                        # A buried accent is confined to the loaded paint. A
                        # canvas-spanning stratum beneath isolated pools would
                        # remain an unrelated straight stripe in the final image.
                        buried = np.clip(
                            state.sum(axis=-1) / max(settings["initial_load"], 1e-9), 0, 1
                        )
                    under[..., palette["underpaint_index"]] = (
                        buried * settings["underpaint_strength"]
                    )
                if laminate:
                    # Split the same starting amount, with no new pigment source.
                    # The palette stores upper shares; subtraction retains the
                    # total to float32 precision even for nearly pure layers.
                    upper = state * arrays["layer_fractions"]
                    under = state - upper
                    state = upper
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
                if gpu.interaction is not None:
                    # Transport history with OLD layer amounts, before any paint
                    # ping-pong buffers or local exchanges replace those inputs.
                    gpu.interaction.transport(
                        tuple(block[0] for block in gpu.blocks),
                        tuple(block[0] for block in gpu.underpaints),
                        gpu.velocity,
                        dt,
                    )
                a, b = source.frame(t0), source.frame(t1)
                if influence is None:
                    segments = np.concatenate([a.positions, b.positions], axis=1).astype("f4")
                    travel = np.maximum(0, b.arc_lengths - a.arc_lengths)
                else:
                    segments = movement_segments(a.positions, b.positions, influence)
                    travel = np.maximum(0, arc_travel(a.arc_lengths, b.arc_lengths, influence))
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
                if laminate:
                    for index, block in enumerate(gpu.underpaints):
                        gpu.underpaints[index] = gpu._advect_block(
                            block,
                            dt * settings["lower_transport_scale"],
                            segments,
                            np.zeros((3, 4), dtype="f4"),
                        )
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
                if not gpu.skip_phase:
                    gpu._exchange_phases(dt)
                if gpu.diffusion is not None:
                    gpu._interdiffuse(dt)
                if gpu.laminate_exchange is not None:
                    gpu._exchange_layers(dt)
                if gpu.interaction is not None:
                    gpu.interaction.update(
                        tuple(block[0] for block in gpu.blocks),
                        tuple(block[0] for block in gpu.underpaints),
                        gpu.carrier[0],
                        gpu.velocity,
                        dt,
                    )
                gpu.paint = gpu.blocks[0]
                gpu.internal_steps += 1
                if gpu.internal_steps > MAX_INTERNAL_STEPS:
                    raise RuntimeError("Confluence transport work cap exceeded")

            def _exchange_phases(gpu, dt):
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
                        if key == "settling" and settings["settling_scale"] != 1.0:
                            values *= settings["settling_scale"]
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

            def _interdiffuse(gpu, dt):
                coefficient = settings["diffusion_coefficient"]
                pixel_size = 2 * gpu.domain / gpu.height
                pieces = diffusion_substeps(coefficient, dt, pixel_size)
                if not pieces:
                    return
                if gpu.diffusion_steps + pieces > MAX_DIFFUSION_STEPS:
                    raise RuntimeError("Interdiffusion work cap exceeded")
                number = coefficient * (dt / pieces) / pixel_size**2
                gpu.maximum_diffusion_number = max(gpu.maximum_diffusion_number, number)
                gpu.diffusion["u_lambda"].value = number
                layers = (gpu.blocks, gpu.underpaints) if laminate else (gpu.blocks,)
                for _ in range(pieces):
                    # Both packed groups read the same old material state. Only
                    # after all writes finish may either group become current.
                    for layer in layers:
                        for index, block in enumerate(layer):
                            other = layer[(index + 1) % len(layer)][0]
                            for unit, (name, texture) in enumerate(
                                (
                                    ("u_input", block[0]),
                                    ("u_other", other),
                                    ("u_carrier", gpu.carrier[0]),
                                )
                            ):
                                texture.use(unit)
                                gpu.diffusion[name].value = unit
                            block[3].bind_to_image(0, read=False, write=True)
                            gpu._dispatch(gpu.diffusion)
                        for block in layer:
                            block[0], block[3] = block[3], block[0]
                gpu.diffusion_steps += pieces

            def _exchange_layers(gpu, dt):
                shader = gpu.laminate_exchange
                shader["u_exposure"].value = settings["interlayer_exchange_rate"] * dt
                for index, (upper, lower) in enumerate(
                    zip(gpu.blocks, gpu.underpaints, strict=True)
                ):
                    other = (index + 1) % len(gpu.blocks)
                    for unit, (name, texture) in enumerate(
                        (
                            ("u_upper", upper[0]),
                            ("u_lower", lower[0]),
                            ("u_upper_other", gpu.blocks[other][0]),
                            ("u_lower_other", gpu.underpaints[other][0]),
                            ("u_carrier", gpu.carrier[0]),
                        )
                    ):
                        texture.use(unit)
                        shader[name].value = unit
                    upper[3].bind_to_image(0, read=False, write=True)
                    lower[3].bind_to_image(1, read=False, write=True)
                    gpu._dispatch(shader)
                # Both channel packs read the same two layers before swapping.
                for layer in (gpu.blocks, gpu.underpaints):
                    for block in layer:
                        block[0], block[3] = block[3], block[0]

            def _mass_amounts(gpu):
                """Pairwise GPU block sums, then float64 accumulation on the CPU."""
                sums = []
                for index, block in enumerate(gpu.blocks):
                    total = np.zeros(4, dtype="f8")
                    layers = (block, gpu.underpaints[index]) if laminate else (block,)
                    for layer in layers:
                        layer[0].use(0)
                        gpu.mass_partials.bind_to_storage_buffer(0)
                        gpu._dispatch(gpu.mass_reduce)
                        partial = np.frombuffer(gpu.mass_partials.read(), dtype="f4").reshape(-1, 4)
                        total += partial.sum(axis=0, dtype="f8")
                    sums.append(total)
                return np.concatenate(sums)[:count] * (2 * gpu.domain / gpu.height) ** 2

            def _restore_mass_budget(gpu):
                before = gpu._mass_amounts()
                factors = mass_budget_factors(gpu.mass_budget_initial, before)
                for index, block in enumerate(gpu.blocks):
                    packed = np.ones(4, dtype="f4")
                    channels = min(4, count - index * 4)
                    packed[:channels] = factors[index * 4 : index * 4 + channels]
                    gpu.mass_scale["u_factors"].value = tuple(packed)
                    layers = (block, gpu.underpaints[index]) if laminate else (block,)
                    for layer in layers:
                        layer[0].use(0)
                        layer[3].bind_to_image(0, read=False, write=True)
                        gpu._dispatch(gpu.mass_scale)
                        layer[0], layer[3] = layer[3], layer[0]
                gpu.paint = gpu.blocks[0]
                after = gpu._mass_amounts()
                if not np.allclose(after, gpu.mass_budget_initial, rtol=MASS_BUDGET_RTOL, atol=0):
                    raise FloatingPointError("Global pigment budget could not be restored")
                gpu.mass_budget_records.append(
                    {
                        "step": gpu.step,
                        "mass_before": before.tolist(),
                        "factors": factors.astype("f8").tolist(),
                        "mass_after": after.tolist(),
                    }
                )

            def advance_to(gpu, step):
                interval = settings["mass_budget_interval_steps"]
                if not interval:
                    return super().advance_to(step)
                if gpu.ctx is None:
                    raise RuntimeError("The Estuary context has been closed")
                if type(step) is not int or not gpu.step <= step <= gpu.steps:
                    raise ValueError("Source step must move forward within the declared recording")
                while gpu.step < step:
                    next_correction = gpu.mass_budget_schedule[
                        bisect_right(gpu.mass_budget_schedule, gpu.step)
                    ]
                    checkpoint = min(step, next_correction)
                    super().advance_to(checkpoint)
                    if gpu.step == next_correction:
                        with gpu.ctx:
                            gpu._restore_mass_budget()

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
                dry_material = deposit if laminate else deposit + underpaint
                dry_share = np.divide(
                    dry_material.sum(axis=-1),
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
                    "...i,i->...",
                    (mobile + underpaint) * 0.22
                    if laminate
                    else deposit + underpaint + mobile * 0.22,
                    arrays["specific_volumes"],
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
                if gpu.interaction is not None:
                    result.update(gpu.interaction.snapshot(lambda t: gpu._read(t, factor)))
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
        if influence is not None:
            from .body_influence import eligible_pairs

            self.metadata["body_influence"] = {
                "config": copy.deepcopy(influence),
                "eligible_pairs": [list(pair) for pair in eligible_pairs(influence)],
                "initialization": "unchanged full-source layout and all initial pigment channels",
                "forcing": (
                    "zero inactive descriptors, wetting/deposition travel "
                    "and adaptive travel limits"
                ),
                "events": "eligible active pairs selected before count and refractory competition",
                "strength_normalization": "none",
                "recording": "the original complete three-body trajectories are not reintegrated",
            }
        if settings["pair_strain"]:
            from .pair_strain import VERSION as STRAIN_VERSION

            self.metadata["pair_strain"] = {
                "version": STRAIN_VERSION,
                "gain": settings["pair_strain"],
                "model": "analytic curl of pair-aligned Gaussian quadrupoles",
                "source": "signed conditioned pair extension over real three-dimensional distance",
                "limits": "prescribed incompressible flow; not a fluid pressure solver",
            }
        if settings.get("interaction") is not None:
            from .interaction import VERSION as INTERACTION_VERSION

            self.metadata["interaction"] = {
                "version": INTERACTION_VERSION,
                "seed": palette["seed"],
                "config": copy.deepcopy(settings["interaction"]),
                "state": (
                    "per-layer transported material origin, contact dose, "
                    "axial fabric and aggregate fraction"
                ),
                "transport": (
                    "limited mass-weighted MacCormack attribute reconstruction; "
                    "no feedback on baseline pigment transport"
                    if settings["interaction"].get("advection") == "maccormack"
                    else "positive mass-weighted characteristic interpolation; "
                    "no feedback on baseline pigment transport"
                ),
                "aggregation": (
                    "authored unresolved dispersed/aggregate partition of existing pigment; "
                    "not discrete particles"
                ),
                "geometry": "original pigment amounts, height and outer silhouette retained",
                "diagnostic_reduction": (
                    "area averages of intensive history descriptors; "
                    "qualified rendering uses native fields"
                ),
            }
        if layout is not None:
            self.metadata["initial_layout"] = copy.deepcopy(layout)
        if shaped:
            self.metadata["initial_composition"] = {
                "config": copy.deepcopy(settings["initial_composition"]),
                "mass_units": "world-area integral of concentration per chromatic pigment",
                "normalization": (
                    "native float32 raster; no initial_load or pigment-weight multiplier"
                ),
                "partition": "unchanged upper/lower float32 split; chalk starts at exactly zero",
                "mass_relative_tolerance": MASS_BUDGET_RTOL,
            }
        if initial_weights is not None:
            self.metadata["initial_pigment_weights"] = {
                "multipliers": initial_weights,
                "application": "base chromatic pool loads multiplied before any layer partition",
                "normalization": (
                    "none; initial and restored mass budgets include actual weighted loads"
                ),
            }
        if settings["deposition"] == 0:
            self.metadata["mass_limitations"] = (
                "interpolated transport may drift; no trajectory pigment source; "
                "water events add only carrier"
            )
        if self._gpu.skip_phase:
            self.metadata["phase_exchange"] = (
                "stationary phases remain identically zero; exact exchange dispatch is skipped "
                "because settling_scale and initial underpaint are both zero"
            )
        if laminate:
            self.metadata["model"] = "Confluence Laminate / two transported wet pigment layers"
            self.metadata["phase_exchange"] = (
                "upper and lower wet layers; positive local species-conservative contact exchange; "
                "each layer retains its local total; no stationary deposits"
            )
            self.metadata["laminate"] = {
                "version": "co-moving-laminate-v1",
                "upper_snapshot_field": "mobile",
                "lower_snapshot_field": "underpaint",
                "upper_transport_scale": 1.0,
                "lower_transport_scale": settings["lower_transport_scale"],
                "initial_upper_fractions": arrays["layer_fractions"].tolist(),
                "interlayer_exchange_rate": settings["interlayer_exchange_rate"],
                "interlayer_min_concentration": settings["interlayer_min_concentration"],
                "carrier": "shared transported wetness, mixedness and material direction",
                "calibration": (
                    "authored thin-layer model; not measured fluid or pigment properties"
                ),
            }
        if settings["flow_domain_scale"] != settings["domain_scale"]:
            extent = settings["flow_domain_scale"]
            aspect = self._gpu.aspect
            self.metadata["flow_bounds"] = [[-aspect * extent, -extent], [aspect * extent, extent]]
            self.metadata["flow_boundary"] = (
                "compact stream-function support inside the unchanged simulation guard; "
                "velocity and normal transport vanish at the active boundary"
            )
        if settings["diffusion_coefficient"] > 0:
            self.metadata["interdiffusion"] = {
                "model": "symmetric pigment-fraction exchange at fixed local film amount",
                "coefficient": settings["diffusion_coefficient"],
                "units": "world-coordinate squared per complete source recording",
                "minimum_concentration": settings["diffusion_min_concentration"],
                "boundary": "no flux between paint and air or through dry cells",
                "conservation": (
                    "local total paint and global per-pigment amounts, up to float32 rounding; "
                    "no clipping or mass normalization"
                ),
                "maximum_explicit_number": DIFFUSION_CFL,
                "maximum_substeps_per_interval": MAX_DIFFUSION_SUBSTEPS,
                "maximum_total_substeps": MAX_DIFFUSION_STEPS,
                "calibration": (
                    "authored finite-rate interdiffusion, not a measured molecular diffusivity"
                ),
            }
        if settings["mass_budget_interval_steps"]:
            self.metadata["mass_budget"] = {
                "version": MASS_BUDGET_VERSION,
                "interval_steps": settings["mass_budget_interval_steps"],
                "units": "world-area integral of pigment concentration",
                "reduction": "pairwise float32 GPU workgroup sums; float64 CPU accumulation",
                "relative_tolerance": MASS_BUDGET_RTOL,
                "model": (
                    "explicit uniform per-pigment budget restoration at canonical checkpoints "
                    "and the final step; globally budgeted, not locally conservative transport"
                ),
            }

    @property
    def steps(self):
        return self._gpu.steps

    @property
    def step(self):
        return self._gpu.step

    @property
    def diagnostics(self):
        """Actual accumulated work; captured separately from immutable inputs."""
        return {
            "canonical_steps": self._gpu.step,
            "actual_transport_substeps": self._gpu.internal_steps,
            "diffusion_substeps": self._gpu.diffusion_steps,
            "maximum_courant": self._gpu.maximum_courant,
            "maximum_diffusion_number": self._gpu.maximum_diffusion_number,
        }

    @property
    def mass_budget_report(self):
        """Return the complete correction ledger without reading or changing GPU state."""
        if not self.config["mass_budget_interval_steps"]:
            return None
        return {
            "version": MASS_BUDGET_VERSION,
            "interval_steps": self.config["mass_budget_interval_steps"],
            "initial_mass": self._gpu.mass_budget_initial.tolist(),
            "corrections": copy.deepcopy(self._gpu.mass_budget_records),
        }

    def advance_to(self, step):
        self._gpu.advance_to(step)

    def snapshot(self, resolution=None):
        return self._gpu.snapshot(resolution)

    def gpu_frame(self):
        """Borrow the current material textures until the next physical advance.

        The consumer validates the owner and source-step token before use. It
        must neither mutate these textures nor release the engine's context.
        No simulation work or CPU material readback occurs in this accessor.
        """
        if self._gpu.ctx is None:
            raise RuntimeError("The Estuary context has been closed")
        from .gpu_frame import GPUFrame

        interaction = self._gpu.interaction
        history = (
            {}
            if interaction is None
            else {
                "origin_upper": interaction.origins[0],
                "origin_lower": interaction.origins[1],
                "interaction_upper": interaction.states[0],
                "interaction_lower": interaction.states[1],
            }
        )
        return GPUFrame.capture(
            owner=self,
            context=self._gpu.ctx,
            token=(self.step, self._gpu.internal_steps),
            size=(self._gpu.width, self._gpu.height),
            pigment_count=len(self.palette["pigments_srgb"]),
            chalk_index=self.palette["chalk_index"],
            mobile=tuple(block[0] for block in self._gpu.blocks),
            deposit=tuple(group[0] for group in self._gpu.deposits),
            underpaint=tuple(group[0] for group in self._gpu.underpaints),
            carrier=self._gpu.carrier[0],
            tooth=self._gpu.tooth,
            specific_volumes=tuple(self.palette["specific_volumes"]),
            height_scale_mm=self.config["height_scale_mm"],
            substrate_um=self.config["substrate_um"],
            material_model=self.config["material_model"],
            **history,
        )

    def close(self):
        self._gpu.close()
