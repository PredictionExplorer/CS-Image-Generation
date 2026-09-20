"""Contact-generated, transported paint microstructure.

This is a bounded authored constitutive model, not measured pigment chemistry or
a discrete particle simulation. Each layer stores a material-origin first moment
and (contact dose, axial fabric x/y, aggregate fraction). Contact is possible only
in occupied overlap: separated origins also recognize same-color encounters.

Auxiliary transport defaults to positive mass-weighted semi-Lagrangian sampling.
An explicit MacCormack option limits a forward/reverse error correction to valid
old donor extrema, retaining finer material features. Neither method changes
pigment or height or guarantees locally conservative transport. The aggregate
fraction partitions each existing pigment into an
effective dispersed/aggregated state. Nucleation propensity is seed-derived in
transported material coordinates and is evaluated here, never by the renderer.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from . import material_traits
from .palette import normalize_seed

VERSION = "contact-microstructure-v1"
FIELD_NAMES = ("origin_upper", "origin_lower", "interaction_upper", "interaction_lower")
BASE_FIELDS = (
    "pigment",
    "mobile",
    "deposit",
    "underpaint",
    "wetness",
    "mixing",
    "direction",
    "height",
    "roughness",
    "coverage",
)
DEFAULTS = {
    "version": VERSION,
    "contact_rate": 5.0,
    "origin_distance": 0.025,
    "composition_threshold": 0.03,
    "minimum_concentration": 1e-5,
    "fabric_rate": 3.0,
    "fabric_relaxation": 0.12,
    "aggregation_rate": 2.0,
    "breakup_rate": 0.25,
    "nucleation_scale": 0.008,
    "nucleation_contrast": 0.7,
}
_BOUNDS = {
    "contact_rate": (0, 100),
    "origin_distance": (1e-5, 2),
    "composition_threshold": (1e-5, 0.99),
    "minimum_concentration": (1e-10, 1),
    "fabric_rate": (0, 100),
    "fabric_relaxation": (0, 100),
    "aggregation_rate": (0, 100),
    "breakup_rate": (0, 100),
    "nucleation_scale": (1e-5, 1),
    "nucleation_contrast": (0, 1),
}


def validate_config(value):
    """Return a resolved opt-in config; omission does not enable microstructure."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - (
        set(DEFAULTS) | {"advection", "material_variation"}
    ):
        raise ValueError("Interaction config must contain only documented fields")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))
    if result["version"] != VERSION:
        raise ValueError(f"Interaction version must be {VERSION}")
    # Preserve archived v1 dictionaries exactly when no new option is requested.
    # The explicit mode, when supplied, participates in recipe/archive identity.
    if "advection" in result and (
        type(result["advection"]) is not str or result["advection"] not in ("linear", "maccormack")
    ):
        raise ValueError("Interaction advection must be linear or maccormack")
    for key, (low, high) in _BOUNDS.items():
        v = result[key]
        try:
            valid = type(v) in (int, float) and math.isfinite(v) and low <= v <= high
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"Interaction {key} must be finite and in [{low}, {high}]")
        result[key] = float(v)
    variation = material_traits.validate_config(result.pop("material_variation", None))
    if variation is not None:
        result["material_variation"] = variation
    return result


def field_names(config):
    """Archive field contract; legacy names remain unchanged when traits are disabled."""
    settings = validate_config(config)
    if settings is None:
        return ()
    return FIELD_NAMES + (material_traits.FIELD_NAMES if "material_variation" in settings else ())


validate_traits = material_traits.validate_fields


def seed_key(seed):
    """Domain-separated stream; every input bit participates in the SHA-256 hash."""
    canonical = normalize_seed(seed)
    digest = hashlib.sha256(
        VERSION.encode("ascii") + b"\0nucleation\0" + int(canonical, 16).to_bytes(32, "big")
    ).digest()
    return tuple(int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4))


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def initial_origins(size, aspect, domain):
    """Bottom-up world-coordinate origins; both layers receive identical fields."""
    width, height = size
    x = ((np.arange(width, dtype="f4") + 0.5) / width * 2 - 1) * aspect * domain
    y = ((np.arange(height, dtype="f4") + 0.5) / height * 2 - 1) * domain
    origin = np.zeros((height, width, 4), dtype="f4")
    origin[..., 0], origin[..., 1] = x[None, :], y[:, None]
    return origin


def _smoothstep(low, high, x):
    t = np.clip((x - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def _lattice_hash(x, y, key):
    # uint64 intermediates explicitly mask to GLSL's unsigned 32-bit arithmetic.
    mask = np.uint64(0xFFFFFFFF)
    xx = np.asarray(x, dtype=np.int64).astype(np.uint64) & mask
    yy = np.asarray(y, dtype=np.int64).astype(np.uint64) & mask
    h = ((xx ^ np.uint64(key[0])) * np.uint64(0x9E3779B9)) & mask
    h ^= ((yy ^ np.uint64(key[1])) * np.uint64(0x85EBCA6B)) & mask
    h ^= np.uint64(key[2])
    h = ((h ^ (h >> np.uint64(16))) * np.uint64(0x7FEB352D)) & mask
    h = ((h ^ (h >> np.uint64(15))) * np.uint64(0x846CA68B)) & mask
    h ^= (h >> np.uint64(16)) ^ np.uint64(key[3])
    return (h >> np.uint64(8)).astype("f8") / 16777216.0


def nucleation_field(origins, seed, config):
    """Smooth deterministic material heterogeneity, used only in reaction rates."""
    settings = validate_config(config)
    if settings is None:
        raise ValueError("Nucleation requires an enabled interaction configuration")
    origin = np.asarray(origins, dtype="f8")
    if origin.shape[-1:] != (2,) or not np.isfinite(origin).all():
        raise ValueError("Origins require finite xy coordinates")
    p = origin / settings["nucleation_scale"]
    if np.any(np.abs(p) > (1 << 30)):
        raise ValueError("Material coordinates exceed the supported lattice range")
    cell = np.floor(p).astype(np.int64)
    f = p - cell
    f = f * f * (3 - 2 * f)
    key = seed_key(seed)
    a = _lattice_hash(cell[..., 0], cell[..., 1], key)
    b = _lattice_hash(cell[..., 0] + 1, cell[..., 1], key)
    c = _lattice_hash(cell[..., 0], cell[..., 1] + 1, key)
    d = _lattice_hash(cell[..., 0] + 1, cell[..., 1] + 1, key)
    noise = (a * (1 - f[..., 0]) + b * f[..., 0]) * (1 - f[..., 1]) + (
        c * (1 - f[..., 0]) + d * f[..., 0]
    ) * f[..., 1]
    islands = _smoothstep(0.3, 0.75, noise) ** 3
    return 1 - settings["nucleation_contrast"] + settings["nucleation_contrast"] * islands


def _paint_fields(upper, lower):
    upper, lower = (np.asarray(v, dtype="f8") for v in (upper, lower))
    if (
        upper.ndim != 3
        or lower.shape != upper.shape
        or upper.shape[-1] not in (2, 3, 4, 6)
        or any(not np.isfinite(v).all() or np.any(v < 0) for v in (upper, lower))
    ):
        raise ValueError("Paint layers require matching finite nonnegative pigment fields")
    return upper, lower


def contact_strength(upper, lower, upper_origin, lower_origin, config):
    """Wetness-independent contact gate: occupied overlap of unlike material."""
    settings = validate_config(config)
    if settings is None:
        raise ValueError("Contact requires an enabled interaction configuration")
    upper, lower = _paint_fields(upper, lower)
    origins = [np.asarray(v, dtype="f8") for v in (upper_origin, lower_origin)]
    if any(v.shape != (*upper.shape[:2], 2) or not np.isfinite(v).all() for v in origins):
        raise ValueError("Origins must match paint dimensions and contain finite xy coordinates")
    amounts = [v.sum(axis=-1) for v in (upper, lower)]
    a, b = amounts
    fa, fb = [
        np.divide(v, amount[..., None], out=np.zeros_like(v), where=amount[..., None] > 0)
        for v, amount in zip((upper, lower), amounts, strict=True)
    ]
    difference = 0.5 * np.abs(fa - fb).sum(axis=-1)
    composition = _smoothstep(settings["composition_threshold"], 1.0, difference)
    separation = np.linalg.norm(origins[0] - origins[1], axis=-1)
    origin_gate = _smoothstep(
        settings["origin_distance"] * 0.25, settings["origin_distance"], separation
    )
    balance = np.divide(2 * np.minimum(a, b), a + b, out=np.zeros_like(a), where=a + b > 0)
    return (
        np.maximum(origin_gate, composition)
        * balance
        * (a > settings["minimum_concentration"])
        * (b > settings["minimum_concentration"])
    )


def velocity_derivatives(velocity, pixel_size):
    """Return trace-free symmetric strain (xx, xy) and physical angular spin."""
    v = np.asarray(velocity, dtype="f8")
    if v.ndim != 3 or v.shape[-1] != 2 or not np.isfinite(v).all():
        raise ValueError("Velocity requires a finite H x W x 2 field")
    if type(pixel_size) not in (int, float) or not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
    padded = np.pad(v, ((1, 1), (1, 1), (0, 0)), mode="edge")
    vx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * pixel_size)
    vy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * pixel_size)
    return np.stack((0.5 * (vx[..., 0] - vy[..., 1]), 0.5 * (vx[..., 1] + vy[..., 0])), -1), (
        0.5 * (vx[..., 1] - vy[..., 0])
    )


def update_state(state, contact, wetness, strain, spin, nucleation, *, config, dt, traits=None):
    """Independent float64 reference for bounded local constitutive kinetics.

    Strain is the axial representation of the symmetric trace-free velocity
    gradient. Spin is half the curl. No velocity-direction surrogate is used.
    Exact scalar relaxation avoids Euler overshoot for arbitrarily large dt;
    fabric rotation/relaxation uses symmetric operator splitting.
    """
    settings = validate_config(config)
    if settings is None:
        raise ValueError("State updates require an enabled interaction configuration")
    dt = _nonnegative(dt, "dt")
    s = np.asarray(state, dtype="f8")
    if s.ndim != 3 or s.shape[-1] != 4 or not np.isfinite(s).all():
        raise ValueError("Microstructure requires a finite H x W x 4 state")
    shape = s.shape[:2]
    contact, wetness, spin, nucleation = (
        np.asarray(v, dtype="f8") for v in (contact, wetness, spin, nucleation)
    )
    strain = np.asarray(strain, dtype="f8")
    if any(
        v.shape != shape or not np.isfinite(v).all() for v in (contact, wetness, spin, nucleation)
    ):
        raise ValueError("Interaction scalar fields must match the state dimensions")
    if strain.shape != (*shape, 2) or not np.isfinite(strain).all():
        raise ValueError("Strain requires matching finite axial components")
    if any(np.any(v < 0) or np.any(v > 1) for v in (contact, wetness, nucleation)):
        raise ValueError("Contact, wetness and nucleation must be in [0, 1]")
    if (
        np.any(s[..., (0, 3)] < 0)
        or np.any(s[..., (0, 3)] > 1)
        or np.any(np.linalg.norm(s[..., 1:3], axis=-1) > s[..., 0] + 1e-6)
    ):
        raise ValueError("Microstructure must obey its concentration and fabric bounds")
    variation = settings.get("material_variation")
    multipliers = None
    if variation is not None:
        multipliers = material_traits.rate_multipliers(traits, contact, wetness, variation)
    elif traits is not None:
        raise ValueError("Material traits require enabled variation")
    if dt == 0:
        return s.copy()
    result = np.empty_like(s)
    exposure = settings["contact_rate"] * contact * wetness * dt
    result[..., 0] = s[..., 0] + (1 - s[..., 0]) * -np.expm1(-exposure)
    magnitude = np.linalg.norm(strain, axis=-1)
    axis = np.divide(
        strain, magnitude[..., None], out=np.zeros_like(strain), where=magnitude[..., None] > 0
    )
    magnitude = np.minimum(magnitude, 40)
    # Axial Q rotates through twice the physical angle; two half rotations
    # surround relaxation toward the fixed strain target.
    angle = spin * dt
    c, sn = np.cos(angle), np.sin(angle)

    def rotate(q):
        return np.stack((c * q[..., 0] - sn * q[..., 1], sn * q[..., 0] + c * q[..., 1]), -1)

    align = settings["fabric_rate"] * contact * wetness * magnitude
    if multipliers is not None:
        align *= multipliers[2]
    relaxation = align + settings["fabric_relaxation"] * wetness
    mix = -np.expm1(-relaxation * dt)
    target_weight = np.divide(align, relaxation, out=np.zeros_like(align), where=relaxation > 0)
    q = (
        rotate(s[..., 1:3]) * (1 - mix[..., None])
        + axis * (result[..., 0] * target_weight * mix)[..., None]
    )
    q = rotate(q)
    norm = np.linalg.norm(q, axis=-1)
    q *= np.minimum(1, np.divide(result[..., 0], norm, out=np.ones_like(norm), where=norm > 0))[
        ..., None
    ]
    result[..., 1:3] = q
    formation = settings["aggregation_rate"] * contact * wetness * nucleation
    breakup = settings["breakup_rate"] * wetness * magnitude
    if multipliers is not None:
        formation *= multipliers[0]
        breakup *= multipliers[1]
    rate = formation + breakup
    equilibrium = np.divide(formation, rate, out=np.zeros_like(formation), where=rate > 0)
    result[..., 3] = s[..., 3] + (equilibrium - s[..., 3]) * -np.expm1(-rate * dt)
    result[..., (0, 3)] = np.clip(result[..., (0, 3)], 0, 1)
    return result


def mass_weighted_sample(field, mass, positions, *, minimum_concentration=1e-5):
    """Reference convex bilinear gather in pixel-center coordinates.

    Empty neighbors contribute neither history nor a weight. Coordinates clamp
    to the guard-domain edge, matching the paint transport's texture sampler.
    This helper intentionally tests sampling independently of trajectory tracing.
    """
    return _mass_weighted_gather(field, mass, positions, minimum_concentration)[0]


def _mass_weighted_gather(field, mass, positions, minimum_concentration):
    """Sample attributes, support mass and only occupied donor extrema."""
    field, mass, positions = (np.asarray(v, dtype="f8") for v in (field, mass, positions))
    threshold = _nonnegative(minimum_concentration, "minimum_concentration")
    if (
        field.ndim != 3
        or mass.shape != field.shape[:2]
        or positions.shape[-1:] != (2,)
        or any(not np.isfinite(v).all() for v in (field, mass, positions))
        or np.any(mass < 0)
    ):
        raise ValueError("Invalid material field, mass or sampling coordinates")
    height, width = mass.shape
    p = np.clip(positions, (0, 0), (width - 1, height - 1))
    base = np.floor(p).astype(np.int64)
    f = p - base
    numerator = np.zeros((*p.shape[:-1], field.shape[-1]), dtype="f8")
    denominator = np.zeros(p.shape[:-1], dtype="f8")
    low = np.full_like(numerator, np.inf)
    high = np.full_like(numerator, -np.inf)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = np.minimum(base[..., 0] + dx, width - 1), np.minimum(base[..., 1] + dy, height - 1)
        weights = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
        m = mass[yy, xx]
        donor = (m > threshold) & (weights > 0)
        low = np.minimum(low, np.where(donor[..., None], field[yy, xx], np.inf))
        high = np.maximum(high, np.where(donor[..., None], field[yy, xx], -np.inf))
        weights *= m * (m > threshold)
        numerator += field[yy, xx] * weights[..., None]
        denominator += weights
    value = np.divide(
        numerator,
        denominator[..., None],
        out=np.zeros_like(numerator),
        where=denominator[..., None] > threshold,
    )
    occupied = denominator > threshold
    return (
        value,
        np.where(occupied, denominator, 0),
        np.where(occupied[..., None], low, 0),
        np.where(occupied[..., None], high, 0),
    )


def maccormack_transport(
    field, mass, velocity, *, pixel_size, dt, minimum_concentration=1e-5, bound_state=False
):
    """Independent float64 reference for limited mass-weighted MacCormack.

    The forward support is an interpolation guide for the backward pass, not a
    proposed update to pigment density. Componentwise extrema use only occupied
    old donors that actually contribute to the characteristic sample. Correction
    is skipped where the old target or backward support is absent. Bound-state
    mode also enforces the coupled fabric/dose invariant after component limits.
    """
    dt = _nonnegative(dt, "dt")
    field, mass, velocity = (np.asarray(v, dtype="f8") for v in (field, mass, velocity))
    if (
        field.ndim != 3
        or mass.shape != field.shape[:2]
        or velocity.shape != (*mass.shape, 2)
        or any(not np.isfinite(v).all() for v in (field, mass, velocity))
        or np.any(mass < 0)
        or type(pixel_size) not in (int, float)
        or not math.isfinite(pixel_size)
        or pixel_size <= 0
        or type(bound_state) is not bool
        or (bound_state and field.shape[-1] != 4)
    ):
        raise ValueError("Invalid MacCormack material fields, velocity or pixel size")
    threshold = _nonnegative(minimum_concentration, "minimum_concentration")
    yy, xx = np.mgrid[: mass.shape[0], : mass.shape[1]]
    centers = np.stack((xx, yy), -1).astype("f8")
    uniform_mass = np.ones_like(mass)

    def trace(duration):
        half = centers - velocity * (duration * 0.5 / pixel_size)
        middle_velocity = mass_weighted_sample(
            velocity, uniform_mass, half, minimum_concentration=0
        )
        return centers - middle_velocity * (duration / pixel_size)

    forward, support, low, high = _mass_weighted_gather(field, mass, trace(dt), threshold)
    backward, backward_support, _, _ = _mass_weighted_gather(
        forward, support, trace(-dt), threshold
    )
    eligible = (mass > threshold) & (backward_support > threshold) & (support > threshold)
    result = np.where(
        eligible[..., None], np.clip(forward + 0.5 * (field - backward), low, high), forward
    )
    if bound_state:
        result[..., (0, 3)] = np.clip(result[..., (0, 3)], 0, 1)
        norm = np.linalg.norm(result[..., 1:3], axis=-1)
        result[..., 1:3] *= np.minimum(
            1, np.divide(result[..., 0], norm, out=np.ones_like(norm), where=norm > 0)
        )[..., None]
    return result


class GPUInteraction:
    """Own optional RGBA32F history buffers; caller must bind the owning context."""

    def __init__(self, ctx, size, aspect, domain, seed, config, lower_transport_scale=1.0):
        self.ctx = ctx
        self.config = validate_config(config)
        if self.config is None:
            raise ValueError("GPUInteraction requires an enabled configuration")
        self.size, self.aspect, self.domain = tuple(size), float(aspect), float(domain)
        self.lower_transport_scale = float(lower_transport_scale)
        self.seed = normalize_seed(seed)
        self.key = seed_key(seed)
        self.groups = tuple((n + 15) // 16 for n in size)
        self._resources = []
        self._origins, self._states = [], []
        self._traits = []
        self._trait_advection = None
        self._predictor = self._corrector = None
        self._forward = self._backward = None
        root = Path(__file__).parent / "shaders"
        try:
            self.advect = self._own(
                ctx.compute_shader((root / "interaction-advect.glsl").read_text())
            )
            reaction_source = (root / "interaction-update.glsl").read_text()
            variation = self.config.get("material_variation")
            if variation is not None:
                reaction_source = reaction_source.replace(
                    "#version 430", "#version 430\n#define MATERIAL_VARIATION", 1
                )
            self.reaction = self._own(ctx.compute_shader(reaction_source))
            for shader in (self.advect, self.reaction):
                shader["u_size"].value = self.size
                shader["u_domain"].value = self.domain
                shader["u_minimum_concentration"].value = self.config["minimum_concentration"]
            self.advect["u_aspect"].value = self.aspect
            self.reaction["u_seed"].value = self.key
            self.reaction["u_lower_scale"].value = self.lower_transport_scale
            for key in _BOUNDS:
                self.reaction["u_" + key].value = self.config[key]
            origins = initial_origins(size, aspect, domain).tobytes()
            zeros = np.zeros((size[1], size[0], 4), dtype="f4").tobytes()
            for _ in range(2):
                self._origins.append([self._texture(origins) for _ in range(2)])
                self._states.append([self._texture(zeros) for _ in range(2)])
            if variation is not None:
                self.reaction["u_trait_amplitude"].value = variation["amplitude"]
                self._trait_advection = self._own(
                    ctx.compute_shader((root / "material-traits-transport.glsl").read_text())
                )
                for name, value in (
                    ("u_size", self.size),
                    ("u_aspect", self.aspect),
                    ("u_domain", self.domain),
                    ("u_minimum_concentration", self.config["minimum_concentration"]),
                ):
                    self._trait_advection[name].value = value
                initial = np.zeros((size[1], size[0], 4), dtype="f4")
                initial[..., :2] = material_traits.initial_traits(
                    initial_origins(size, aspect, domain)[..., :2], seed, variation
                )
                for _ in range(2):
                    self._traits.append([self._texture(initial.tobytes()) for _ in range(2)])
            if self.config.get("advection", "linear") == "maccormack":
                self._predictor = self._own(
                    ctx.compute_shader((root / "interaction-predict.glsl").read_text())
                )
                self._corrector = self._own(
                    ctx.compute_shader((root / "interaction-correct.glsl").read_text())
                )
                for shader in (self._predictor, self._corrector):
                    shader["u_size"].value = self.size
                    shader["u_aspect"].value = self.aspect
                    shader["u_domain"].value = self.domain
                    shader["u_minimum_concentration"].value = self.config["minimum_concentration"]
                # Shared scratch is safe: each layer finishes all three passes
                # before the next layer starts, reducing the added memory by 2x.
                self._forward = tuple(self._texture(zeros) for _ in range(2))
                self._backward = tuple(self._texture(zeros) for _ in range(2))
        except Exception:
            self.close()
            raise

    def _own(self, resource):
        self._resources.append(resource)
        return resource

    def _texture(self, data):
        texture = self._own(self.ctx.texture(self.size, 4, data=data, dtype="f4"))
        texture.repeat_x = texture.repeat_y = False
        return texture

    @property
    def origins(self):
        return tuple(layer[0] for layer in self._origins)

    @property
    def states(self):
        return tuple(layer[0] for layer in self._states)

    @property
    def traits(self):
        return tuple(layer[0] for layer in self._traits) if self._traits else None

    def _dispatch(self, shader, bindings):
        for unit, (name, texture) in enumerate(bindings):
            texture.use(unit)
            shader[name].value = unit
        shader.run(*self.groups)
        self.ctx.memory_barrier()

    def transport(self, upper, lower, velocity, dt):
        """Call before paint transport, passing its OLD concentration textures."""
        dt = _nonnegative(dt, "dt")
        if dt == 0:
            return
        self._transport(upper, lower, velocity, dt)

    def initialize(self, upper, lower):
        """Mask initial world origins to real paint before any transport occurs.

        A zero-duration gather uses the first paint pack as a harmless velocity
        binding: no velocity contributes when dt is zero. No old paint is held.
        """
        self._transport(upper, lower, upper[0], 0.0)

    def _transport(self, upper, lower, velocity, dt):
        for index, (paint, scale) in enumerate(
            zip((upper, lower), (1, self.lower_transport_scale), strict=True)
        ):
            if len(paint) not in (1, 2):
                raise ValueError("Interaction transport requires one or two RGBA pigment packs")
            origin, state = self._origins[index], self._states[index]
            if self._traits:
                self._transport_traits(self._traits[index], paint, velocity, dt * scale)
            if self._predictor is not None and dt != 0:
                self._transport_maccormack(origin, state, paint, velocity, dt * scale)
                origin.reverse()
                state.reverse()
                continue
            self.advect["u_has_other"].value = len(paint) == 2
            self.advect["u_dt"].value = dt * scale
            origin[1].bind_to_image(0, read=False, write=True)
            state[1].bind_to_image(1, read=False, write=True)
            self._dispatch(
                self.advect,
                (
                    ("u_origin", origin[0]),
                    ("u_state", state[0]),
                    ("u_paint", paint[0]),
                    ("u_paint_other", paint[-1]),
                    ("u_velocity", velocity),
                ),
            )
            origin.reverse()
            state.reverse()

    def _transport_traits(self, traits, paint, velocity, dt):
        """Transport actual initialized attributes; origins never supply their values.

        MacCormack temporarily borrows history scratch buffers. All trait passes
        finish before history transport starts, so no additional scratch is owned.
        """
        shader = self._trait_advection
        shader["u_has_other"].value = len(paint) == 2
        corrected = self._predictor is not None and dt != 0
        forward = self._forward[0] if corrected else traits[1]
        backward = self._backward[0] if corrected else traits[0]
        passes = ((0, traits[0], forward, dt),)
        if corrected:
            passes += ((1, forward, backward, -dt), (2, traits[0], traits[1], dt))
        for phase, source, target, duration in passes:
            shader["u_phase"].value = phase
            shader["u_keep_support"].value = corrected and phase != 2
            shader["u_dt"].value = duration
            target.bind_to_image(0, read=False, write=True)
            self._dispatch(
                shader,
                (
                    ("u_field", source),
                    ("u_forward", forward),
                    ("u_backward", backward),
                    ("u_paint", paint[0]),
                    ("u_paint_other", paint[-1]),
                    ("u_velocity", velocity),
                ),
            )
        traits.reverse()

    def _transport_maccormack(self, origin, state, paint, velocity, dt):
        """Three passes on attributes only; OLD actual pigment is read-only."""
        predictor = self._predictor
        predictor["u_has_other"].value = len(paint) == 2
        for source, target, duration, use_guide in (
            ((origin[0], state[0]), self._forward, dt, False),
            (self._forward, self._backward, -dt, True),
        ):
            predictor["u_dt"].value = duration
            predictor["u_use_mass_guide"].value = use_guide
            target[0].bind_to_image(0, read=False, write=True)
            target[1].bind_to_image(1, read=False, write=True)
            self._dispatch(
                predictor,
                (
                    ("u_origin", source[0]),
                    ("u_state", source[1]),
                    ("u_paint", paint[0]),
                    ("u_paint_other", paint[-1]),
                    ("u_velocity", velocity),
                ),
            )
        corrector = self._corrector
        corrector["u_has_other"].value = len(paint) == 2
        corrector["u_dt"].value = dt
        origin[1].bind_to_image(0, read=False, write=True)
        state[1].bind_to_image(1, read=False, write=True)
        self._dispatch(
            corrector,
            (
                ("u_origin", origin[0]),
                ("u_state", state[0]),
                ("u_forward_origin", self._forward[0]),
                ("u_forward_state", self._forward[1]),
                ("u_backward_origin", self._backward[0]),
                ("u_backward_state", self._backward[1]),
                ("u_paint", paint[0]),
                ("u_paint_other", paint[-1]),
                ("u_velocity", velocity),
            ),
        )

    def update(self, upper, lower, carrier, velocity, dt):
        """React after all paint and wetness updates; never write pigment textures."""
        dt = _nonnegative(dt, "dt")
        if dt == 0:
            return
        if len(upper) not in (1, 2) or len(lower) != len(upper):
            raise ValueError("Interaction update requires matching pigment packs")
        self.reaction["u_dt"].value = dt
        self.reaction["u_has_other"].value = len(upper) == 2
        for index, state in enumerate(self._states):
            state[1].bind_to_image(index, read=False, write=True)
        trait_bindings = ()
        if self._traits:
            for index, layer in enumerate(self._traits):
                layer[1].bind_to_image(index + 2, read=False, write=True)
            trait_bindings = (
                ("u_upper_traits", self.traits[0]),
                ("u_lower_traits", self.traits[1]),
            )
        self._dispatch(
            self.reaction,
            (
                ("u_upper_origin", self.origins[0]),
                ("u_lower_origin", self.origins[1]),
                ("u_upper_state", self.states[0]),
                ("u_lower_state", self.states[1]),
                ("u_upper", upper[0]),
                ("u_lower", lower[0]),
                ("u_upper_other", upper[-1]),
                ("u_lower_other", lower[-1]),
                ("u_carrier", carrier),
                ("u_velocity", velocity),
                *trait_bindings,
            ),
        )
        for state in self._states:
            state.reverse()
        for layer in self._traits:
            layer.reverse()

    def snapshot(self, read_fn):
        """Read actual material state through the engine's reduction/read helper."""
        return dict(
            zip(
                field_names(self.config),
                (
                    *(read_fn(texture)[..., :2].copy() for texture in self.origins),
                    *(read_fn(texture) for texture in self.states),
                    *(read_fn(texture)[..., :2].copy() for texture in (self.traits or ())),
                ),
                strict=True,
            )
        )

    def close(self):
        """Idempotently release only resources owned by this optional module."""
        for resource in reversed(self._resources):
            resource.release()
        self._resources.clear()
