"""Material bounds, units, pigment isolation, and optional native Blender wiring."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest

from tools.estuary_depth.materials import (
    GLAZE_MODEL,
    MICRO_GRAIN_METRES,
    MODEL,
    build_material,
    property_reference,
    validate_config,
)


class MaterialConfigTests(unittest.TestCase):
    def test_defaults_are_resolved_without_mutating_input(self):
        config = {"ior": 1.5}
        result = validate_config(config)
        self.assertEqual(config, {"ior": 1.5})
        self.assertEqual(result["ior"], 1.5)
        self.assertEqual(result["micro_bump_um"], 0)
        result["coat"] = 0.2
        self.assertEqual(validate_config({})["coat"], 0.04)

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({"blue_roughnes": 0.4})
        with self.assertRaises(ValueError):
            validate_config(None)

    def test_nonfinite_and_non_numeric_settings_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, "0.4", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"blue_roughness": value})

    def test_physical_controls_enforce_explicit_bounds(self):
        invalid = {
            "blue_roughness": 0.079,
            "ivory_roughness": 1.001,
            "vermilion_roughness": -0.1,
            "ior": 0.99,
            "blue_transmission": 0.951,
            "ivory_transmission": 0.301,
            "subsurface_mm": 1.001,
            "metallic_accent": 0.401,
            "coat": 0.251,
            "micro_bump_um": 30.001,
            "glass_roughness": 1.001,
        }
        for key, value in invalid.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({key: value})

    def test_transmission_model_tint_and_glass_roughness_are_validated(self):
        for settings in (
            {"transmission_model": "glass"},
            {"transmission_model": True},
            {"transmission_model": ["clear-glaze"]},
            {"glass_tint": [1, 1]},
            {"glass_tint": [1, 1, 1.001]},
            {"glass_tint": [1, 1, math.nan]},
            {"glass_tint": [1, 1, True]},
            {"glass_tint": "white"},
            {"glass_roughness": -0.001},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_config(settings)
        settings = validate_config({"transmission_model": "clear-glaze", "blue_transmission": 0.95})
        self.assertEqual(settings["glass_tint"], [0.75, 0.87, 1.0])
        self.assertEqual(settings["glass_roughness"], 0.08)
        self.assertEqual(validate_config({})["transmission_model"], "pigment")

    def test_glass_tint_configuration_does_not_alias_caller_or_defaults(self):
        original = {"glass_tint": [0.7, 0.8, 1.0]}
        first = validate_config(original)
        first["glass_tint"][0] = 0
        self.assertEqual(original["glass_tint"][0], 0.7)
        defaults = validate_config({})
        defaults["glass_tint"][0] = 0
        self.assertEqual(validate_config({})["glass_tint"][0], 0.75)

    def test_units_are_metres(self):
        result = property_reference([0, 1, 0], {"subsurface_mm": 0.6, "micro_bump_um": 12})
        self.assertAlmostEqual(result["subsurface_scale_metres"], 0.0006)
        self.assertAlmostEqual(result["micro_bump_distance_metres"], 0.000012)
        self.assertAlmostEqual(MICRO_GRAIN_METRES, 0.0002)


class PigmentPropertyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "blue_transmission": 0.5,
            "ivory_transmission": 0.1,
            "metallic_accent": 0.3,
            "subsurface_mm": 0.4,
        }

    def test_each_pigment_has_its_own_material_response(self):
        blue = property_reference([1, 0, 0], self.config)
        ivory = property_reference([0, 1, 0], self.config)
        vermilion = property_reference([0, 0, 1], self.config)
        self.assertEqual(blue["roughness"], 0.4)
        self.assertEqual(blue["transmission"], 0.5)
        self.assertEqual(blue["subsurface_weight"], 0)
        self.assertEqual(blue["metallic"], 0)
        self.assertEqual(ivory["roughness"], 0.32)
        self.assertEqual(ivory["transmission"], 0.1)
        self.assertEqual(ivory["subsurface_weight"], 1)
        self.assertEqual(ivory["metallic"], 0)
        self.assertEqual(vermilion["roughness"], 0.25)
        self.assertEqual(vermilion["transmission"], 0)
        self.assertEqual(vermilion["subsurface_weight"], 0)
        self.assertEqual(vermilion["metallic"], 0.3)

    def test_mixed_properties_use_normalized_pigment_fractions(self):
        mixed = property_reference([2, 3, 5], self.config)
        self.assertAlmostEqual(mixed["roughness"], 0.2 * 0.4 + 0.3 * 0.32 + 0.5 * 0.25)
        self.assertAlmostEqual(mixed["transmission"], 0.2 * 0.5 + 0.3 * 0.1)
        self.assertAlmostEqual(mixed["subsurface_weight"], 0.3)
        self.assertAlmostEqual(mixed["metallic"], 0.15)

    def test_fraction_magnitude_does_not_change_material(self):
        first = property_reference([2, 3, 5], self.config)
        second = property_reference([0.2, 0.3, 0.5], self.config)
        for key in first:
            self.assertAlmostEqual(first[key], second[key])

    def test_empty_fraction_field_has_ground_response(self):
        self.assertEqual(
            property_reference([0, 0, 0], self.config),
            property_reference([1, 0, 0], self.config),
        )

    def test_disabled_effects_are_exactly_zero(self):
        off = {
            "blue_transmission": 0,
            "ivory_transmission": 0,
            "subsurface_mm": 0,
            "metallic_accent": 0,
            "coat": 0,
            "micro_bump_um": 0,
        }
        for fractions in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            result = property_reference(fractions, off)
            for key in result.keys() - {"roughness"}:
                self.assertEqual(result[key], 0)

    def test_invalid_fraction_values_are_rejected(self):
        for fractions in ([1, 0], [1, -1, 0], [1, math.nan, 0], [1, math.inf, 0], [True, 0, 0]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                property_reference(fractions, {})

    def test_glaze_mixture_weights_stay_bounded_and_preserve_vermilion_opacity(self):
        settings = {
            "transmission_model": "clear-glaze",
            "blue_transmission": 0.95,
            "ivory_transmission": 0.3,
        }
        self.assertEqual(property_reference([1, 0, 0], settings)["transmission"], 0.95)
        self.assertEqual(property_reference([0, 1, 0], settings)["transmission"], 0.3)
        self.assertEqual(property_reference([0, 0, 1], settings)["transmission"], 0)
        for fractions in ([2, 3, 5], [0, 0, 0], [0.99, 0.001, 0.009], [1e-12, 1e-12, 1e-12]):
            amount = property_reference(fractions, settings)["transmission"]
            self.assertGreaterEqual(amount, 0)
            self.assertLessEqual(amount, 0.95)
            self.assertAlmostEqual((1 - amount) + amount, 1)


@unittest.skipUnless(importlib.util.find_spec("bpy") is not None, "requires native Blender Python")
class BlenderMaterialTests(unittest.TestCase):
    def setUp(self):
        import bpy

        self.bpy = bpy
        self.images = {
            key: bpy.data.images.new(f"Test Estuary {key}", width=1, height=1, float_buffer=True)
            for key in ("color", "fractions")
        }
        self.materials = []

    def tearDown(self):
        for material in self.materials:
            self.bpy.data.materials.remove(material)
        for image in self.images.values():
            self.bpy.data.images.remove(image)

    def build(self, settings):
        material = build_material(self.bpy, self.images, settings, layer_index=2)
        self.materials.append(material)
        return material, next(
            node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
        )

    def test_default_graph_is_lit_opaque_and_preserves_linear_color_input(self):
        material, shader = self.build({})
        self.assertEqual(material["estuary_material_model"], MODEL)
        self.assertEqual(material["estuary_layer_index"], 2)
        self.assertEqual(json.loads(material["estuary_material_config"])["ior"], 1.47)
        self.assertEqual(shader.inputs["Alpha"].default_value, 1)
        self.assertEqual(shader.inputs["Emission Strength"].default_value, 0)
        self.assertEqual(shader.inputs["Base Color"].links[0].from_node.image, self.images["color"])
        self.assertTrue(
            all(image.colorspace_settings.name == "Non-Color" for image in self.images.values())
        )
        self.assertEqual(len(shader.inputs["Normal"].links), 0)
        self.assertFalse(
            any(
                node.type in {"BUMP", "DISPLACEMENT", "BSDF_GLASS", "MIX_SHADER"}
                for node in material.node_tree.nodes
            )
        )
        self.assertEqual(len(shader.inputs["Transmission Weight"].links), 1)

    def test_optional_effects_have_physical_units_and_disappear_when_disabled(self):
        material, shader = self.build(
            {"micro_bump_um": 12, "subsurface_mm": 0.6, "metallic_accent": 0.2}
        )
        bump = next(node for node in material.node_tree.nodes if node.type == "BUMP")
        self.assertAlmostEqual(bump.inputs["Distance"].default_value, 12e-6)
        self.assertAlmostEqual(shader.inputs["Subsurface Scale"].default_value, 0.0006)
        self.assertEqual(shader.inputs["Normal"].links[0].from_node, bump)
        self.assertEqual(len(shader.inputs["Metallic"].links), 1)
        self.assertFalse(any(node.type == "DISPLACEMENT" for node in material.node_tree.nodes))
        _, disabled = self.build({"subsurface_mm": 0, "metallic_accent": 0, "micro_bump_um": 0})
        self.assertEqual(len(disabled.inputs["Subsurface Weight"].links), 0)
        self.assertEqual(disabled.inputs["Subsurface Weight"].default_value, 0)
        self.assertEqual(len(disabled.inputs["Metallic"].links), 0)
        self.assertEqual(disabled.inputs["Metallic"].default_value, 0)

    def test_clear_glaze_uses_separate_glass_with_complementary_surface_weights(self):
        material, pigment = self.build(
            {"transmission_model": "clear-glaze", "blue_transmission": 0.95}
        )
        nodes = material.node_tree.nodes
        glass = next(node for node in nodes if node.type == "BSDF_GLASS")
        mixture = next(node for node in nodes if node.type == "MIX_SHADER")
        output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
        self.assertEqual(material["estuary_material_model"], GLAZE_MODEL)
        self.assertEqual(pigment.inputs["Transmission Weight"].default_value, 0)
        self.assertEqual(len(pigment.inputs["Transmission Weight"].links), 0)
        self.assertEqual(
            pigment.inputs["Base Color"].links[0].from_node.image, self.images["color"]
        )
        self.assertEqual(mixture.inputs[1].links[0].from_node, pigment)
        self.assertEqual(mixture.inputs[2].links[0].from_node, glass)
        self.assertEqual(len(mixture.inputs[0].links), 1)
        self.assertEqual(len(output.inputs["Surface"].links), 1)
        self.assertEqual(output.inputs["Surface"].links[0].from_node, mixture)
        self.assertEqual(glass.distribution, "GGX")
        self.assertAlmostEqual(glass.inputs["Roughness"].default_value, 0.08)
        self.assertAlmostEqual(glass.inputs["IOR"].default_value, 1.47)
        for actual, expected in zip(
            glass.inputs["Color"].default_value, [0.75, 0.87, 1, 1], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertFalse(any(node.type == "ADD_SHADER" for node in nodes))

    def test_clear_glaze_and_pigment_share_the_same_optional_surface_normal(self):
        material, pigment = self.build({"transmission_model": "clear-glaze", "micro_bump_um": 2})
        glass = next(node for node in material.node_tree.nodes if node.type == "BSDF_GLASS")
        self.assertEqual(
            glass.inputs["Normal"].links[0].from_node, pigment.inputs["Normal"].links[0].from_node
        )


if __name__ == "__main__":
    unittest.main()
