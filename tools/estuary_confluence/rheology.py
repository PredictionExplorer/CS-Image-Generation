"""Authored two-dimensional paint resistance with transported structural memory.

This is a quasistatic streamfunction response, not a pressure, free-surface or
three-dimensional non-Newtonian fluid solver. The analytic source velocity is
retained; only a smooth, compactly supported discrete curl is subtracted. The
response therefore does not add discrete divergence to the prescribed flow.

Structure is an intensive material property, transported with actual pigment
mass. Its bounded rebuild/breakdown equation is integrated exponentially with
rates frozen over each accepted transport substep. The screened response is
recomputed from zero for each forcing evaluation, so failed CFL proposals and
camera captures cannot advance material history.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

VERSION = "paint-rheology-v1"
FIELD_NAMES = ("structure_upper", "structure_lower")
ITERATIONS = 64
MAX_RESPONSE_SIZE = 128
DEFAULTS = {
    "version": VERSION,
    "strength": 3.0,
    "initial_structure": 0.45,
    "rebuild_rate": 1.2,
    "dry_rebuild_rate": 2.0,
    "breakdown_rate": 2.5,
    "shear_scale": 20.0,
    "response_length": 0.04,
    "minimum_concentration": 1e-5,
}
BOUNDS = {
    "strength": (0, 20),
    "initial_structure": (0, 1),
    "rebuild_rate": (0, 100),
    "dry_rebuild_rate": (0, 100),
    "breakdown_rate": (0, 100),
    "shear_scale": (1e-4, 1000),
    "response_length": (0.005, 0.5),
    "minimum_concentration": (1e-10, 1),
}
OPTIONAL_BOUNDS = {"occupancy_mass_reference": (1e-5, 1)}


def validate_config(value=None):
    """Normalize opt-in controls; zero resistance preserves the legacy path."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - (set(DEFAULTS) | set(OPTIONAL_BOUNDS)):
        raise ValueError("Rheology config must contain only documented fields")
    result = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(value)}
    if result["version"] != VERSION:
        raise ValueError(f"Rheology version must be {VERSION}")
    bounds = {**BOUNDS, **{key: limit for key, limit in OPTIONAL_BOUNDS.items() if key in result}}
    for key, (low, high) in bounds.items():
        number = result[key]
        try:
            valid = type(number) in (int, float) and math.isfinite(number) and low <= number <= high
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"Rheology {key} must be finite and in [{low}, {high}]")
        result[key] = float(number)
    return result if result["strength"] else None


def occupancy_factor(mass, reference):
    """Bounded paint-amount response, independent of the history support cutoff."""
    value = np.asarray(mass, dtype="f8")
    try:
        valid = (
            type(reference) in (int, float) and math.isfinite(reference) and 1e-5 <= reference <= 1
        )
    except OverflowError:
        valid = False
    if not np.isfinite(value).all() or np.any(value < 0) or not valid:
        raise ValueError("Occupancy requires finite nonnegative paint and a reference in [1e-5, 1]")
    return value / (value + reference)


def initialization_config(config):
    """Leave the RC1 full-source starting-layout pilot independent of resistance."""
    if "rheology" not in config:
        return config
    return {key: copy.deepcopy(value) for key, value in config.items() if key != "rheology"}


def response_plan(size, domain, config):
    """Choose cells at least as wide as the response length in both directions.

    Jacobi's infinity-norm contraction is <=4/5: each directional coefficient
    is <=1 and the diagonal is 1+a+2(rx+ry), with a>=0. In exact arithmetic,
    64 iterations reduce the initial error bound by less than 6.3e-7. Float32
    rounding is additional and is qualified separately, not called exact.
    """
    settings = validate_config(config)
    if settings is None:
        raise ValueError("A response plan requires enabled rheology")
    if (
        type(size) not in (list, tuple)
        or len(size) != 2
        or any(type(v) is not int or v < 4 for v in size)
        or type(domain) not in (int, float)
        or not math.isfinite(domain)
        or domain <= 0
    ):
        raise ValueError("Invalid rheology response extent")
    extent = (2 * domain * size[0] / size[1], 2 * domain)
    length = settings["response_length"]
    if length > min(extent) / 4:
        raise ValueError("Rheology response length requires at least four cells per axis")
    shape = [
        min(n, MAX_RESPONSE_SIZE, math.floor(span / length))
        for n, span in zip(size, extent, strict=True)
    ]
    spacing = [span / n for span, n in zip(extent, shape, strict=True)]
    coefficients = [(length / h) ** 2 for h in spacing]
    neighbor_sum = 2 * sum(coefficients)
    contraction = neighbor_sum / (1 + neighbor_sum)
    return {
        "version": VERSION,
        "resolution": shape,
        "cell_width": spacing,
        "coefficients": coefficients,
        "iterations": ITERATIONS,
        "maximum_contraction": contraction,
        "exact_arithmetic_error_factor": contraction**ITERATIONS,
        "boundary": (
            "zero correction on coarse outer cells; "
            "smooth compact flow window inset 1.5 native cells after prolongation"
        ),
        "prolongation": "positive cubic B-spline in world-aligned texture coordinates",
        "velocity": "unchanged analytic source minus native central-difference curl of correction",
    }


def evolve_structure(structure, wetness, shear, dt, config):
    """Float64 reference for the exact frozen-rate local reaction."""
    settings = validate_config(config)
    if settings is None:
        raise ValueError("Structure evolution requires enabled rheology")
    state, water, rate = np.broadcast_arrays(
        *[np.asarray(v, dtype="f8") for v in (structure, wetness, shear)]
    )
    if (
        any(not np.isfinite(v).all() for v in (state, water, rate))
        or np.any((state < 0) | (state > 1))
        or np.any((water < 0) | (water > 1))
        or np.any(rate < 0)
        or type(dt) not in (int, float)
        or not math.isfinite(dt)
        or dt < 0
    ):
        raise ValueError("Invalid structural state, wetness, shear or timestep")
    rebuild = settings["rebuild_rate"] + settings["dry_rebuild_rate"] * (1 - water)
    breakdown = settings["breakdown_rate"] * rate / (settings["shear_scale"] + rate)
    total = rebuild + breakdown
    equilibrium = np.divide(rebuild, total, out=np.zeros_like(total), where=total > 0)
    return np.clip(state + (equilibrium - state) * -np.expm1(-total * dt), 0, 1)


def screened_response(source_psi, resistance, coefficients, iterations=ITERATIONS):
    """Independent float64 Jacobi reference; a zero Dirichlet correction."""
    psi, amount = (np.asarray(v, dtype="f8") for v in (source_psi, resistance))
    weights = np.asarray(coefficients, dtype="f8")
    if (
        psi.ndim != 2
        or min(psi.shape) < 4
        or amount.shape != psi.shape
        or not np.isfinite(psi).all()
        or not np.isfinite(amount).all()
        or np.any(amount < 0)
        or weights.shape != (2,)
        or not np.isfinite(weights).all()
        or np.any((weights < 0) | (weights > 1))
        or type(iterations) is not int
        or not 0 <= iterations <= 4096
    ):
        raise ValueError("Invalid screened response fields or coefficients")
    x, y = weights
    delta = np.zeros_like(psi)
    denominator = 1 + amount[1:-1, 1:-1] + 2 * (x + y)
    rhs = (amount * psi)[1:-1, 1:-1]
    for _ in range(iterations):
        update = np.zeros_like(delta)
        update[1:-1, 1:-1] = (
            rhs
            + x * (delta[1:-1, :-2] + delta[1:-1, 2:])
            + y * (delta[:-2, 1:-1] + delta[2:, 1:-1])
        ) / denominator
        delta = update
    return delta


def discrete_curl(potential, pixel_size):
    """Native reference curl with clamp-to-edge boundary sampling."""
    value = np.asarray(potential, dtype="f8")
    if (
        value.ndim != 2
        or not np.isfinite(value).all()
        or not math.isfinite(pixel_size)
        or pixel_size <= 0
    ):
        raise ValueError("Invalid potential or native cell width")
    padded = np.pad(value, 1, mode="edge")
    return np.stack(
        (padded[2:, 1:-1] - padded[:-2, 1:-1], padded[1:-1, :-2] - padded[1:-1, 2:]), -1
    ) / (2 * pixel_size)


def contact_resistance_factor(mass, affinity, contact, *, amplitude, minimum_concentration=1e-5):
    """Reference for one response cell's bounded, paint-contact-gated trait factor.

    Arrays include all native cells from both layers. Empty paint contributes
    neither a trait nor a contact dose; no spatial noise is evaluated here.
    """
    mass, affinity, contact = (np.asarray(value, dtype="f8") for value in (mass, affinity, contact))
    if (
        mass.shape != affinity.shape
        or mass.shape != contact.shape
        or any(not np.isfinite(value).all() for value in (mass, affinity, contact))
        or np.any(mass < 0)
        or np.any(np.abs(affinity) > 1)
        or np.any((contact < 0) | (contact > 1))
        or type(amplitude) not in (int, float)
        or not math.isfinite(amplitude)
        or not 0 <= amplitude <= 0.25
        or type(minimum_concentration) not in (int, float)
        or not math.isfinite(minimum_concentration)
        or minimum_concentration <= 0
    ):
        raise ValueError("Invalid contact-gated resistance inputs")
    total = float(mass.sum())
    mean = float(np.sum(mass * affinity * contact)) / total if total > minimum_concentration else 0
    return 1 + amplitude * np.clip(mean, -1, 1)


class GPURheology:
    """Own optional GPU histories and response scratch; context is caller-bound."""

    def __init__(self, ctx, size, aspect, domain, config, flow_settings, lower_transport_scale):
        self.ctx, self.size, self.aspect, self.domain = (
            ctx,
            tuple(size),
            float(aspect),
            float(domain),
        )
        self.config = validate_config(config)
        self.plan = response_plan(list(size), domain, self.config)
        from .material_traits import validate_config as variation_config

        variation = variation_config(
            (flow_settings.get("interaction") or {}).get("material_variation")
        )
        self.trait_amplitude = None if variation is None else variation["amplitude"]
        self.lower_scale = lower_transport_scale
        self.groups = tuple((n + 15) // 16 for n in size)
        self.coarse_size = tuple(self.plan["resolution"])
        self.coarse_groups = tuple((n + 15) // 16 for n in self.coarse_size)
        self._resources = []
        self._states = []
        root = Path(__file__).parent / "shaders"
        try:

            def shader(name):
                code = (root / name).read_text()
                if name == "rheology-prepare.glsl" and self.trait_amplitude is not None:
                    code = code.replace("#version 430", "#version 430\n#define RHEOLOGY_TRAITS", 1)
                if name == "rheology-prepare.glsl" and "occupancy_mass_reference" in self.config:
                    code = code.replace(
                        "#version 430", "#version 430\n#define RHEOLOGY_OCCUPANCY_REFERENCE", 1
                    )
                code = code.replace(
                    '#include "rheology-streamfunction.glsl"',
                    (root / "rheology-streamfunction.glsl").read_text(),
                )
                return self._own(ctx.compute_shader(code))

            self.advect = shader("rheology-advect.glsl")
            self.reaction = shader("rheology-update.glsl")
            self.prepare = shader("rheology-prepare.glsl")
            self.solve = shader("rheology-solve.glsl")
            self.prolong = shader("rheology-prolong.glsl")
            self.curl = shader("rheology-curl.glsl")
            for program in (self.advect, self.reaction, self.prepare, self.prolong, self.curl):
                if "u_size" in program:
                    program["u_size"].value = self.size
                if "u_domain" in program:
                    program["u_domain"].value = self.domain
                if "u_aspect" in program:
                    program["u_aspect"].value = self.aspect
                if "u_minimum" in program:
                    program["u_minimum"].value = self.config["minimum_concentration"]
            for key in ("rebuild_rate", "dry_rebuild_rate", "breakdown_rate", "shear_scale"):
                self.reaction["u_" + key].value = self.config[key]
            self.reaction["u_lower_scale"].value = lower_transport_scale
            self.prepare["u_coarse_size"].value = self.coarse_size
            self.prepare["u_resistance_strength"].value = self.config["strength"]
            if "occupancy_mass_reference" in self.config:
                self.prepare["u_occupancy_mass_reference"].value = self.config[
                    "occupancy_mass_reference"
                ]
            if self.trait_amplitude is not None:
                self.prepare["u_trait_amplitude"].value = self.trait_amplitude
            for key, value in {
                "u_flow_domain": flow_settings["flow_domain_scale"],
                "u_radius": flow_settings["stir_radius"],
                "u_strength": flow_settings["flow_strength"],
                "u_pair_gain": flow_settings["pair_swirl"],
                "u_pair_strain": flow_settings["pair_strain"],
                "u_carrier": tuple(flow_settings["carrier_velocity"]),
            }.items():
                self.prepare[key].value = value
            self.prolong["u_flow_domain"].value = flow_settings["flow_domain_scale"]
            self.solve["u_size"].value = self.coarse_size
            self.solve["u_coefficients"].value = tuple(self.plan["coefficients"])
            initial = np.zeros((size[1], size[0], 4), dtype="f4")
            initial[..., 0] = self.config["initial_structure"]
            for _ in range(2):
                self._states.append([self._texture(size, initial.tobytes()) for _ in range(2)])
            self.guide = self._texture(self.coarse_size)
            self.delta = [self._texture(self.coarse_size) for _ in range(2)]
            self.potential = self._texture(size)
            self.source_velocity = self._own(ctx.texture(self.size, 2, dtype="f4"))
            self.source_velocity.repeat_x = self.source_velocity.repeat_y = False
        except Exception:
            self.close()
            raise

    def _own(self, resource):
        self._resources.append(resource)
        return resource

    def _texture(self, size, data=None):
        value = self._own(self.ctx.texture(tuple(size), 4, data=data, dtype="f4"))
        value.repeat_x = value.repeat_y = False
        return value

    def _dispatch(self, program, bindings=(), *, coarse=False):
        for unit, (name, texture) in enumerate(bindings):
            texture.use(unit)
            program[name].value = unit
        program.run(*(self.coarse_groups if coarse else self.groups))
        self.ctx.memory_barrier()

    @property
    def states(self):
        return tuple(layer[0] for layer in self._states)

    def initialize(self, upper, lower):
        self.transport(upper, lower, upper[0], 0.0)

    def transport(self, upper, lower, velocity, dt):
        for state, paint, scale in zip(
            self._states, (upper, lower), (1.0, self.lower_scale), strict=True
        ):
            self.advect["u_has_other"].value = len(paint) == 2
            self.advect["u_dt"].value = dt * scale
            state[1].bind_to_image(0, read=False, write=True)
            self._dispatch(
                self.advect,
                (
                    ("u_state", state[0]),
                    ("u_paint", paint[0]),
                    ("u_paint_other", paint[-1]),
                    ("u_velocity", velocity),
                ),
            )
            state.reverse()

    def update(self, upper, lower, carrier, velocity, dt):
        self.reaction["u_has_other"].value = len(upper) == 2
        self.reaction["u_dt"].value = dt
        for index, state in enumerate(self._states):
            state[1].bind_to_image(index, read=False, write=True)
        self._dispatch(
            self.reaction,
            (
                ("u_upper_state", self.states[0]),
                ("u_lower_state", self.states[1]),
                ("u_upper", upper[0]),
                ("u_lower", lower[0]),
                ("u_upper_other", upper[-1]),
                ("u_lower_other", lower[-1]),
                ("u_carrier", carrier),
                ("u_velocity", velocity),
            ),
        )
        for state in self._states:
            state.reverse()

    def apply(
        self, upper, lower, tools, pairs, strains, velocity, maxima, *, traits=None, contact=None
    ):
        """Recompute scratch response only; the source velocity is already filled."""
        self.prepare["u_tools"].write(tools.tobytes())
        self.prepare["u_pairs"].write(pairs.tobytes())
        self.prepare["u_strains"].write(strains.tobytes())
        self.prepare["u_has_other"].value = len(upper) == 2
        self.guide.bind_to_image(0, read=False, write=True)
        self.delta[0].bind_to_image(1, read=False, write=True)
        bindings = (
            ("u_upper_state", self.states[0]),
            ("u_lower_state", self.states[1]),
            ("u_upper", upper[0]),
            ("u_lower", lower[0]),
            ("u_upper_other", upper[-1]),
            ("u_lower_other", lower[-1]),
        )
        if self.trait_amplitude is not None:
            if traits is None or contact is None or len(traits) != 2 or len(contact) != 2:
                raise ValueError("Combined rheology requires both transported trait/contact layers")
            bindings += (
                ("u_upper_trait", traits[0]),
                ("u_lower_trait", traits[1]),
                ("u_upper_contact", contact[0]),
                ("u_lower_contact", contact[1]),
            )
        elif traits is not None or contact is not None:
            raise ValueError("Disabled trait coupling cannot receive material traits")
        self._dispatch(self.prepare, bindings, coarse=True)
        for _ in range(ITERATIONS):
            self.delta[1].bind_to_image(0, read=False, write=True)
            self._dispatch(
                self.solve, (("u_guide", self.guide), ("u_delta", self.delta[0])), coarse=True
            )
            self.delta.reverse()
        self.potential.bind_to_image(0, read=False, write=True)
        self._dispatch(self.prolong, (("u_delta", self.delta[0]),))
        velocity.bind_to_image(0, read=False, write=True)
        maxima.bind_to_storage_buffer(1)
        self._dispatch(
            self.curl,
            (("u_potential", self.potential), ("u_source_velocity", self.source_velocity)),
        )
        maximum = float(np.frombuffer(maxima.read(), dtype="f4").max())
        if not math.isfinite(maximum):
            raise FloatingPointError("Nonfinite rheology velocity")
        return maximum

    def snapshot(self, read_fn):
        return {
            name: read_fn(texture)[..., 0].copy()
            for name, texture in zip(FIELD_NAMES, self.states, strict=True)
        }

    def close(self):
        for resource in reversed(self._resources):
            resource.release()
        self._resources.clear()
