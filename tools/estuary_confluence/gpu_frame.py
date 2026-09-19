"""Read-only, step-bound views of an Engine's native GPU material textures.

A view borrows resources; it never owns or releases the context or textures.
The owner is weakly referenced, and every submitted view must still match the
canonical step and internal transport counter at which it was captured.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any

from .palette import SUPPORTED_CHROMATIC_COUNTS


@dataclass(frozen=True)
class GPUFrame:
    context: Any
    token: tuple[int, int]
    size: tuple[int, int]
    pigment_count: int
    mobile: tuple[Any, ...]
    deposit: tuple[Any, ...]
    underpaint: tuple[Any, ...]
    carrier: Any
    tooth: Any
    specific_volumes: tuple[float, ...]
    height_scale_mm: float
    substrate_um: float
    chalk_index: int
    _owner: weakref.ReferenceType
    material_model: str = "legacy"
    origin_upper: Any = None
    origin_lower: Any = None
    interaction_upper: Any = None
    interaction_lower: Any = None

    @classmethod
    def capture(
        cls,
        *,
        owner,
        context,
        token,
        size,
        pigment_count,
        mobile,
        deposit,
        underpaint,
        carrier,
        tooth,
        specific_volumes,
        height_scale_mm,
        substrate_um,
        chalk_index=None,
        material_model="legacy",
        origin_upper=None,
        origin_lower=None,
        interaction_upper=None,
        interaction_lower=None,
    ):
        frame = cls(
            context,
            tuple(token),
            tuple(size),
            pigment_count,
            tuple(mobile),
            tuple(deposit),
            tuple(underpaint),
            carrier,
            tooth,
            tuple(specific_volumes),
            float(height_scale_mm),
            float(substrate_um),
            pigment_count - 1 if chalk_index is None else chalk_index,
            weakref.ref(owner),
            material_model,
            origin_upper,
            origin_lower,
            interaction_upper,
            interaction_lower,
        )
        frame.validate()
        return frame

    @property
    def step(self):
        return self.token[0]

    @property
    def has_interaction(self):
        """Whether this view carries the atomic material-history extension."""
        return self.interaction_upper is not None

    def owner_alive(self):
        owner = self._owner()
        return owner is not None and getattr(owner._gpu, "ctx", None) is self.context

    def require_owner(self):
        owner = self._owner()
        if owner is None or getattr(owner._gpu, "ctx", None) is not self.context:
            raise RuntimeError(
                "The borrowed simulation context has closed or its owner was released"
            )
        return owner

    def validate(self):
        owner = self.require_owner()
        current = (owner.step, owner._gpu.internal_steps)
        if self.token != current:
            raise RuntimeError("GPU frame has expired after simulation advance")
        if (
            len(self.token) != 2
            or any(type(value) is not int or value < 0 for value in self.token)
            or len(self.size) != 2
            or any(type(value) is not int or value < 4 for value in self.size)
            or type(self.pigment_count) is not int
            or self.pigment_count - 1 not in SUPPORTED_CHROMATIC_COUNTS
        ):
            raise ValueError("Invalid GPU material dimensions, channels or source token")
        groups = (self.pigment_count + 3) // 4
        for phase in (self.mobile, self.deposit, self.underpaint):
            if len(phase) != groups:
                raise ValueError("GPU phase packing differs from its pigment count")
        for texture in (*self.mobile, *self.deposit, *self.underpaint, self.carrier, self.tooth):
            if (
                texture.ctx is not self.context
                or texture.size != self.size
                or texture.components != 4
                or texture.dtype != "f4"
            ):
                raise ValueError("Native material views require matching RGBA32F textures")
        extension = (
            self.origin_upper,
            self.origin_lower,
            self.interaction_upper,
            self.interaction_lower,
        )
        if any(texture is not None for texture in extension):
            if any(texture is None for texture in extension) or self.material_model != "laminate":
                raise ValueError("Interaction material views require all four laminate fields")
            for texture in extension:
                if (
                    texture.ctx is not self.context
                    or texture.size != self.size
                    or texture.components != 4
                    or texture.dtype != "f4"
                ):
                    raise ValueError("Interaction material views require matching RGBA32F textures")
        if (
            self.material_model not in ("legacy", "laminate")
            or type(self.chalk_index) is not int
            or not 0 <= self.chalk_index < self.pigment_count
            or len(self.specific_volumes) != self.pigment_count
            or any(not math.isfinite(value) or value < 0 for value in self.specific_volumes)
            or not math.isfinite(self.height_scale_mm)
            or not 0 <= self.height_scale_mm <= 20
            or not math.isfinite(self.substrate_um)
            or not 0 <= self.substrate_um <= 100
        ):
            raise ValueError("Invalid native geometry coefficients")
        return self
