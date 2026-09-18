"""Physically lit Principled approximations for layered Estuary pigment.

The color image stores linear RGB reflectance, without illumination. The separate
fraction image stores the blue, ivory, and vermilion pigment fractions; those
fractions drive material properties independently of the displayed color.
Neither the palette nor these parameters represent measured spectral pigments.
The optional clear-glaze model blends an opaque pigment BSDF and a separately
tinted glass BSDF with complementary weights. It is an authored, energy-
conserving surface mixture, not measured pigment transmission or volume optics.

Mesh-local coordinates and Blender scene units must be metres. The optional
200-micrometre substrate grain affects shading normals only; it never displaces
the geometry or contributes to the shape of the painting.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

MODEL = "principled-rgb-pigment-approximation-v1"
GLAZE_MODEL = "pigment-clear-glass-bsdf-mixture-v1"
MICRO_GRAIN_METRES = 200.0e-6
_DEFAULTS = {
    "blue_roughness": 0.4,
    "ivory_roughness": 0.32,
    "vermilion_roughness": 0.25,
    "ior": 1.47,
    "blue_transmission": 0.12,
    "ivory_transmission": 0.0,
    "subsurface_mm": 0.15,
    "metallic_accent": 0.0,
    "coat": 0.04,
    "micro_bump_um": 0.0,
    "transmission_model": "pigment",
    "glass_tint": (0.75, 0.87, 1.0),
    "glass_roughness": 0.08,
}
_BOUNDS = {
    "blue_roughness": (0.08, 1.0),
    "ivory_roughness": (0.08, 1.0),
    "vermilion_roughness": (0.08, 1.0),
    "ior": (1.0, 2.5),
    "blue_transmission": (0.0, 0.95),
    "ivory_transmission": (0.0, 0.3),
    "subsurface_mm": (0.0, 1.0),
    "metallic_accent": (0.0, 0.4),
    "coat": (0.0, 0.25),
    "micro_bump_um": (0.0, 30.0),
    "glass_roughness": (0.0, 1.0),
}


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve defaults and reject unknown keys or invalid numeric parameters."""
    if not isinstance(config, Mapping):
        raise ValueError("Material configuration must be a mapping")
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown material parameters: {sorted(unknown, key=str)}")
    result = dict(_DEFAULTS)
    result["glass_tint"] = list(result["glass_tint"])
    for key, value in config.items():
        if key == "transmission_model":
            if not isinstance(value, str) or value not in ("pigment", "clear-glaze"):
                raise ValueError("transmission_model must be pigment or clear-glaze")
            result[key] = value
            continue
        if key == "glass_tint":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, Real)
                    or not math.isfinite(v)
                    or not 0 <= v <= 1
                    for v in value
                )
            ):
                raise ValueError("glass_tint must contain three finite linear RGB values in [0, 1]")
            result[key] = [float(v) for v in value]
            continue
        lo, hi = _BOUNDS[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise ValueError(f"{key} must be a finite number in [{lo}, {hi}]")
        result[key] = float(value)
    return result


def property_reference(fractions: Sequence[float], config: Mapping[str, Any]) -> dict[str, float]:
    """Scalar reference for the pigment-property graph, independent of Blender.

    Empty or sub-epsilon density falls back continuously toward the blue ground.
    The geometric surface remains fully opaque in alpha; transmission describes
    dielectric scattering, not transparent holes in the painting.
    """
    settings = validate_config(config)
    if len(fractions) != 3 or any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0
        for v in fractions
    ):
        raise ValueError("Pigment fractions must contain three finite nonnegative numbers")
    total = max(sum(fractions), 1e-8)
    ivory = fractions[1] / total
    vermilion = fractions[2] / total
    blue = max(1.0 - ivory - vermilion, 0.0)
    return {
        "roughness": blue * settings["blue_roughness"]
        + ivory * settings["ivory_roughness"]
        + vermilion * settings["vermilion_roughness"],
        "transmission": blue * settings["blue_transmission"]
        + ivory * settings["ivory_transmission"],
        "subsurface_weight": ivory if settings["subsurface_mm"] > 0 else 0.0,
        "subsurface_scale_metres": settings["subsurface_mm"] / 1000.0,
        "metallic": vermilion * settings["metallic_accent"],
        "coat": settings["coat"],
        "micro_bump_distance_metres": settings["micro_bump_um"] / 1e6,
    }


def build_material(
    bpy: Any, images: Mapping[str, Any], config: Mapping[str, Any], *, layer_index: int = 0
) -> Any:
    """Create a Blender 4.5 material from linear color and pigment fraction images.

    ``images`` must provide ``color`` and ``fractions``. Both are linear numeric
    fields, so their image color spaces are explicitly set to Non-Color. Their
    existing UV coordinates use Blender's active UV map. The caller must supply
    closed, finite-thickness geometry when using transmission or subsurface.
    """
    settings = validate_config(config)
    if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
        raise ValueError("layer_index must be a nonnegative integer")
    if not isinstance(images, Mapping) or not {"color", "fractions"} <= images.keys():
        raise ValueError("Material images must include color and fractions")
    if any(images[key] is None for key in ("color", "fractions")):
        raise ValueError("Material images must not be None")
    material = bpy.data.materials.new(f"Estuary pigment layer {layer_index:02d}")
    material.use_nodes = True
    clear_glaze = settings["transmission_model"] == "clear-glaze"
    material["estuary_material_model"] = GLAZE_MODEL if clear_glaze else MODEL
    material["estuary_material_config"] = json.dumps(settings, sort_keys=True)
    material["estuary_layer_index"] = layer_index
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (980, 200)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (620, 200)
    bsdf.label = "Pigment-weighted dielectric paint"
    bsdf.subsurface_method = "RANDOM_WALK"
    bsdf.inputs["IOR"].default_value = settings["ior"]
    bsdf.inputs["Alpha"].default_value = 1.0
    bsdf.inputs["Coat Weight"].default_value = settings["coat"]
    bsdf.inputs["Coat Roughness"].default_value = 0.28
    bsdf.inputs["Coat IOR"].default_value = settings["ior"]
    bsdf.inputs["Subsurface Scale"].default_value = settings["subsurface_mm"] / 1000.0
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.85, 0.65)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    def texture(key: str, location: tuple[int, int]) -> Any:
        node = nodes.new("ShaderNodeTexImage")
        node.image = images[key]
        node.image.colorspace_settings.name = "Non-Color"
        node.interpolation = "Linear"
        node.extension = "EXTEND"
        node.label = f"Linear {key} field"
        node.location = location
        return node

    color = texture("color", (290, 650))
    fraction = texture("fractions", (-1050, 250))
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    separate.location = (-830, 250)
    links.new(fraction.outputs["Color"], separate.inputs["Color"])
    operation_count = 0

    def operation(kind: str, *values: Any) -> Any:
        nonlocal operation_count
        node = nodes.new("ShaderNodeMath")
        node.operation = kind
        node.location = (-580 + 210 * (operation_count // 7), 200 - 135 * (operation_count % 7))
        operation_count += 1
        for index, value in enumerate(values):
            if isinstance(value, Real):
                node.inputs[index].default_value = float(value)
            else:
                links.new(value, node.inputs[index])
        return node.outputs[0]

    raw = [operation("MAXIMUM", separate.outputs[name], 0.0) for name in ("Red", "Green", "Blue")]
    total = operation("MAXIMUM", operation("ADD", operation("ADD", raw[0], raw[1]), raw[2]), 1e-8)
    ivory = operation("DIVIDE", raw[1], total)
    vermilion = operation("DIVIDE", raw[2], total)
    blue = operation("MAXIMUM", operation("SUBTRACT", 1.0, operation("ADD", ivory, vermilion)), 0.0)
    roughness = operation(
        "ADD",
        operation(
            "ADD",
            operation("MULTIPLY", blue, settings["blue_roughness"]),
            operation("MULTIPLY", ivory, settings["ivory_roughness"]),
        ),
        operation("MULTIPLY", vermilion, settings["vermilion_roughness"]),
    )
    links.new(roughness, bsdf.inputs["Roughness"])
    transmission = operation(
        "ADD",
        operation("MULTIPLY", blue, settings["blue_transmission"]),
        operation("MULTIPLY", ivory, settings["ivory_transmission"]),
    )
    glass = None
    if clear_glaze:
        # Mix Shader uses (1-factor)*opaque + factor*glass. Keep the pigment
        # branch opaque so transmission is never applied twice.
        bsdf.inputs["Transmission Weight"].default_value = 0.0
        glass = nodes.new("ShaderNodeBsdfGlass")
        glass.location = (630, -420)
        glass.label = "Linear-tint clear glaze; no pigment-color multiplication"
        glass.distribution = "GGX"
        glass.inputs["Color"].default_value = (*settings["glass_tint"], 1.0)
        glass.inputs["Roughness"].default_value = settings["glass_roughness"]
        glass.inputs["IOR"].default_value = settings["ior"]
        mixture = nodes.new("ShaderNodeMixShader")
        mixture.location = (1010, 200)
        mixture.label = "Complementary opaque-pigment / clear-glass weights"
        links.new(transmission, mixture.inputs[0])
        links.new(bsdf.outputs["BSDF"], mixture.inputs[1])
        links.new(glass.outputs["BSDF"], mixture.inputs[2])
        links.new(mixture.outputs[0], output.inputs["Surface"])
        output.location = (1230, 200)
    else:
        links.new(transmission, bsdf.inputs["Transmission Weight"])
    if settings["subsurface_mm"] > 0:
        links.new(ivory, bsdf.inputs["Subsurface Weight"])
    else:
        bsdf.inputs["Subsurface Weight"].default_value = 0.0
    if settings["metallic_accent"] > 0:
        links.new(
            operation("MULTIPLY", vermilion, settings["metallic_accent"]), bsdf.inputs["Metallic"]
        )
    else:
        bsdf.inputs["Metallic"].default_value = 0.0
    if settings["micro_bump_um"] > 0:
        coordinate = nodes.new("ShaderNodeTexCoord")
        coordinate.location = (-400, -1050)
        noise = nodes.new("ShaderNodeTexNoise")
        noise.location = (-160, -1050)
        noise.noise_dimensions = "3D"
        noise.inputs["Scale"].default_value = 1.0 / MICRO_GRAIN_METRES
        noise.inputs["Detail"].default_value = 2.0
        noise.inputs["Roughness"].default_value = 0.55
        links.new(coordinate.outputs["Object"], noise.inputs["Vector"])
        bump = nodes.new("ShaderNodeBump")
        bump.location = (150, -1000)
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = settings["micro_bump_um"] / 1e6
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Coat Normal"])
        if glass is not None:
            links.new(bump.outputs["Normal"], glass.inputs["Normal"])
    return material
