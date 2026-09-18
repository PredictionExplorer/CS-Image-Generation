"""Tidal Fresco: advected wet pigment, stationary sediment, and drying.

This is an authored thin-layer material model under Estuary's prescribed
incompressible stirring field, not a free-surface or three-dimensional fluid.
Only mobile pigment and its carrier are transported. Reversible mobile/sediment
exchange is locally conservative; the limited MacCormack transport inherited
from Estuary is not globally mass-conservative. Snapshots retain both phases
for numerical accounting; added source pigment must be distinguished from drift.

The three channels are blue stain, opaque lime, and granular iron oxide. Rates
are per complete source recording. Height is an authored specific-volume
interpretation in metres for a 0.4 m-wide visible painting, never luminance.
All snapshots are bottom-up and include the complete guard domain.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

DEFAULTS = {
    "resolution": [1024, 768],
    "steps": 3600,
    "domain_scale": 1.6,
    "flow_strength": 1.1,
    "carrier_velocity": [2.0, 0.2],
    "stir_radius": 0.22,
    "pair_swirl": 0.9,
    "brush_radius": 0.035,
    "deposition": 0.005,
    "pigment_weights": [0.7, 1.0, 0.08],
    "initial_load": 0.6,
    "initial_pattern": "strata",
    "load_radius": 0.28,
    "fade": 4.0,
    "drying": 3.2,
    "settling": [0.25, 2.2, 3.8],
    "remobilization": [0.6, 0.24, 0.08],
    "wetting": 0.8,
    "granulation": 0.55,
    "bank_strength": 1.2,
    "height_scale_mm": 1.6,
    "substrate_um": 18.0,
    "seed": 17,
}


def validate_config(value):
    """Resolve documented flat controls, rejecting unknown/nonfinite values."""
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Fresco config must be an object with known fields")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))

    def number(key, low, high):
        v = result[key]
        if type(v) not in (int, float):
            raise ValueError(f"{key} must be a number")
        try:
            v = float(v)
        except OverflowError as exc:
            raise ValueError(f"{key} must be finite") from exc
        if not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"{key} must be in [{low}, {high}]")
        result[key] = v

    for key, low, high in (("steps", 1, 40000), ("seed", 0, 2**31 - 1)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"Invalid {key}")
    resolution = result["resolution"]
    if (
        type(resolution) not in (list, tuple)
        or len(resolution) != 2
        or any(type(v) is not int or not 32 <= v <= 8192 for v in resolution)
        or resolution[0] * resolution[1] > 40_000_000
        or not 0.2 <= resolution[0] / resolution[1] <= 5
    ):
        raise ValueError("Invalid Fresco resolution")
    result["resolution"] = list(resolution)
    for key, low, high in (
        ("domain_scale", 1, 2.5),
        ("flow_strength", 0, 8),
        ("stir_radius", 0.02, 2),
        ("pair_swirl", 0, 8),
        ("brush_radius", 0.002, 1),
        ("deposition", 0, 100),
        ("initial_load", 0, 10),
        ("load_radius", 0.001, 2),
        ("fade", 0, 20),
        ("drying", 0, 30),
        ("wetting", 0, 10),
        ("granulation", 0, 1),
        ("bank_strength", 0, 5),
        ("height_scale_mm", 0, 20),
        ("substrate_um", 0, 100),
    ):
        number(key, low, high)
    for key, length, low, high in (
        ("carrier_velocity", 2, -8, 8),
        ("pigment_weights", 3, 0, 10),
        ("settling", 3, 0, 30),
        ("remobilization", 3, 0, 30),
    ):
        v = result[key]
        if type(v) not in (list, tuple) or len(v) != length:
            raise ValueError(f"Invalid {key}")
        checked = []
        for item in v:
            if type(item) not in (int, float):
                raise ValueError(f"Invalid {key}")
            try:
                item = float(item)
            except OverflowError as exc:
                raise ValueError(f"Invalid {key}") from exc
            if not math.isfinite(item) or not low <= item <= high:
                raise ValueError(f"Invalid {key}")
            checked.append(item)
        result[key] = checked
    if result["initial_pattern"] not in ("strata", "pools"):
        raise ValueError("initial_pattern must be strata or pools")
    return result


def phase_exchange(mobile, deposited, settling, release, dt):
    """Exact reversible two-compartment exchange for fixed nonnegative rates.

    This CPU reference and the GPU kernel use the same exponential solution.
    It preserves mobile + deposited channel-by-channel for arbitrary dt.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in (mobile, deposited, settling, release)]
    if (
        type(dt) not in (float, int)
        or not math.isfinite(dt)
        or dt < 0
        or any(not np.isfinite(v).all() or np.any(v < 0) for v in arrays)
    ):
        raise ValueError("Phase exchange requires finite nonnegative inputs")
    m, d, s, r = np.broadcast_arrays(*arrays)
    rate = s + r
    factor = np.divide(
        -np.expm1(-rate * dt), rate, out=np.full_like(rate, float(dt)), where=rate > 0
    )
    transfer = (s * m - r * d) * factor
    return m - transfer, d + transfer


def substrate_field(width, height, domain, seed):
    """Deterministic, fixed-coordinate mineral tooth and absorption variations.

    Frequencies are tied to the physical painting, not output pixels or time.
    The finite modes provide a quiet substrate; only sediment sees the grain.
    """
    x = ((np.arange(width, dtype=np.float64) + 0.5) / width - 0.5) * domain
    y = ((np.arange(height, dtype=np.float64) + 0.5) / height - 0.5) * domain * height / width
    x, y = x[None, :], y[:, None]
    phase = (seed % 997) * 0.01721
    broad = (
        np.sin(2 * math.pi * (2.1 * x + 0.7 * y) + phase)
        + 0.5 * np.sin(2 * math.pi * (0.9 * x - 3.2 * y) + 2.1)
    ) / 1.5
    # Incommensurate directions avoid a woven/grid appearance. Analytic modes
    # retain fixed physical locations at every simulation/output resolution.
    rng = np.random.Generator(np.random.PCG64(seed))
    tooth = np.zeros((height, width), dtype=np.float64)
    nyquist = 0.5 * width / domain
    for _ in range(31):
        frequency = rng.uniform(180, 720)
        angle = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(0, 2 * math.pi)
        kx, ky = frequency * math.cos(angle), frequency * math.sin(angle)
        # Resolve the *same* physical modes on each grid, integrating over a
        # texel and tapering before Nyquist. Unresolved grains become their
        # average, not enlarged blobs or crawling checkerboard aliases.
        footprint = domain / width
        attenuation = np.sinc(kx * footprint) * np.sinc(ky * footprint)
        attenuation *= np.clip((nyquist - max(abs(kx), abs(ky))) / (nyquist * 0.25), 0, 1)
        if attenuation > 0:
            tooth += attenuation * np.sin(2 * math.pi * (kx * x + ky * y) + offset)
    tooth = np.tanh(tooth / 4.0)
    result = np.empty((height, width, 4), dtype="f4")
    result[..., 0] = np.clip(0.5 + 0.5 * tooth, 0, 1)
    result[..., 1] = 0.8 + 0.15 * broad
    result[..., 2] = 0.5 + 0.5 * broad
    result[..., 3] = 0.5 + 0.25 * broad + 0.25 * tooth
    return result


class Fresco:
    """Dedicated GPU context with fixed-clock, reversible paint deposition.

    A small adapter subclass reuses the existing tested flow, adaptive transport,
    and source traversal. The original Estuary code remains unchanged.
    """

    def __init__(self, source, config, backend="egl"):
        from dataclasses import asdict

        from tools.estuary.engine import Engine, _current_context
        from tools.estuary.optics import Material

        self.config = validate_config(config)
        settings = self.config
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
            "pigment_weights",
            "initial_load",
            "initial_pattern",
            "load_radius",
            "fade",
        )
        recipe = {
            "simulation": {k: settings[k] for k in transport_keys},
            "optics": asdict(Material()),
        }

        class WetEngine(Engine):
            def __init__(gpu, source, recipe, backend):
                super().__init__(source, recipe, backend)
                try:
                    with gpu.ctx:
                        gpu.phase = gpu.ctx.compute_shader(
                            (Path(__file__).parent / "shaders/fresco_phase.glsl").read_text()
                        )
                        gpu.sediment = [gpu._texture(4), gpu._texture(4)]
                        gpu.directions = [gpu._texture(4), gpu._texture(4)]
                        zero = np.zeros((gpu.height, gpu.width, 4), dtype="f4")
                        for texture in gpu.sediment + gpu.directions:
                            texture.write(zero.tobytes())
                        gpu.substrate_values = substrate_field(
                            gpu.width, gpu.height, gpu.domain, settings["seed"]
                        )
                        gpu.substrate = gpu._texture(4)
                        gpu.substrate.write(gpu.substrate_values.tobytes())
                        state = (
                            np.frombuffer(gpu.paint[0].read(), dtype="f4")
                            .reshape(gpu.height, gpu.width, 4)
                            .copy()
                        )
                        state[..., 3] = 0.96 + 0.04 * gpu.substrate_values[..., 2]
                        gpu.paint[0].write(state.tobytes())
                        gpu.phase["u_size"].value = (gpu.width, gpu.height)
                        gpu.phase["u_aspect"].value = gpu.aspect
                        gpu.phase["u_domain"].value = gpu.domain
                        for key in (
                            "drying",
                            "wetting",
                            "granulation",
                            "bank_strength",
                            "brush_radius",
                        ):
                            gpu.phase["u_" + key].value = settings[key]
                        gpu.phase["u_settling"].value = tuple(settings["settling"])
                        gpu.phase["u_release"].value = tuple(settings["remobilization"])
                except Exception:
                    gpu.close()
                    raise

            def _transport(gpu, t0, t1):
                super()._transport(t0, t1)
                mobile, _, _, destination = gpu.paint
                for unit, (name, texture) in enumerate(
                    (
                        ("u_mobile", mobile),
                        ("u_sediment", gpu.sediment[0]),
                        ("u_tooth", gpu.substrate),
                        ("u_velocity", gpu.velocity),
                        ("u_direction", gpu.directions[0]),
                    )
                ):
                    texture.use(unit)
                    gpu.phase[name].value = unit
                a, b = source.frame(t0), source.frame(t1)
                segments = np.concatenate([a.positions, b.positions], axis=1).astype("f4")
                travel = np.maximum(0, b.arc_lengths - a.arc_lengths)
                gpu.phase["u_segments"].write(segments.tobytes())
                gpu.phase["u_travel"].value = tuple(travel)
                gpu.phase["u_fade"].value = math.exp(-settings["fade"] * (t0 + t1) * 0.5)
                gpu.phase["u_dt"].value = t1 - t0
                gpu.phase["u_time"].value = (t0 + t1) * 0.5
                destination.bind_to_image(0, read=False, write=True)
                gpu.sediment[1].bind_to_image(1, read=False, write=True)
                gpu.directions[1].bind_to_image(2, read=False, write=True)
                gpu._dispatch(gpu.phase)
                gpu.paint[0], gpu.paint[3] = destination, mobile
                gpu.sediment.reverse()
                gpu.directions.reverse()

            @_current_context
            def snapshot(gpu):
                mobile = (
                    np.frombuffer(gpu.paint[0].read(), dtype="f4")
                    .reshape(gpu.height, gpu.width, 4)
                    .copy()
                )
                sediment = (
                    np.frombuffer(gpu.sediment[0].read(), dtype="f4")
                    .reshape(gpu.height, gpu.width, 4)
                    .copy()
                )
                direction = (
                    np.frombuffer(gpu.directions[0].read(), dtype="f4")
                    .reshape(gpu.height, gpu.width, 4)[..., :2]
                    .copy()
                )
                pigment = mobile[..., :3] + sediment[..., :3]
                wetness = np.clip(mobile[..., 3], 0, 1)
                total = pigment.sum(axis=-1)
                coverage = -np.expm1(-total * 3)
                deposited = sediment[..., :3].sum(axis=-1)
                dry_share = np.divide(
                    deposited, total, out=np.zeros_like(total), where=total > 1e-9
                )
                white = np.divide(
                    pigment[..., 1], total, out=np.zeros_like(total), where=total > 1e-9
                )
                # Lime banks have far more volume than blue stains. Grain only
                # lifts deposited paint; uncovered ground remains nearly flat.
                mass_height = np.einsum("...i,i->...", sediment[..., :3], [0.05, 1.0, 0.38])
                mass_height += np.einsum("...i,i->...", mobile[..., :3], [0.025, 0.22, 0.1])
                height = mass_height * settings["height_scale_mm"] * 0.001
                height += settings["substrate_um"] * 1e-6 * gpu.substrate_values[..., 3]
                roughness = np.clip(
                    0.3
                    + 0.26 * dry_share
                    + 0.11 * white
                    + 0.06 * gpu.substrate_values[..., 0] * dry_share
                    - 0.12 * wetness,
                    0.12,
                    0.88,
                )
                norms = np.linalg.norm(direction, axis=-1)
                direction /= np.maximum(norms[..., None], 1e-9)
                direction[norms < 1e-9] = (1, 0)
                result = {
                    "pigment": pigment,
                    "mobile": mobile[..., :3],
                    "deposited": sediment[..., :3],
                    "height": height,
                    "wetness": wetness,
                    "direction": direction,
                    "roughness": roughness,
                    "coverage": coverage,
                }
                if any(not np.isfinite(v).all() for v in result.values()):
                    raise FloatingPointError("Nonfinite Fresco state")
                return {k: np.ascontiguousarray(v, dtype="f4") for k, v in result.items()}

        self._gpu = WetEngine(source, recipe, backend)
        self.metadata = {
            **self._gpu.metadata,
            "model": "Tidal Fresco: mobile pigment, stationary sediment, drying and rewetting",
            "phase_exchange": "exact reversible two-compartment exponential; locally conservative",
            "mass_limitations": (
                "MacCormack interpolation is not globally conservative; source adds pigment"
            ),
            "height": "authored pigment specific-volume map in metres; visible width 0.4 m",
            "substrate": (
                "fixed deterministic analytic tooth; deposition selective; not frame noise"
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

    def snapshot(self):
        return self._gpu.snapshot()

    def close(self):
        self._gpu.close()
