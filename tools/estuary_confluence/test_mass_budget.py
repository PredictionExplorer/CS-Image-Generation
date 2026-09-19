"""Certified global amounts are tied to initial geometry and actual final paint."""

import copy
import os
import unittest

import numpy as np

from .mass_budget import (
    VERSION,
    correction_steps,
    initial_pool_mass,
    pigment_mass,
    validate_initial_weights,
    validate_report,
)


class MassBudgetTests(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "count": 3,
            "pools": [
                {
                    "pigment_index": i,
                    "position": [x, 0],
                    "radius": 0.18,
                    "edge_width": 0.02,
                    "load": 0.18,
                }
                for i, x in enumerate((-0.5, 0, 0.5))
            ],
        }
        self.recipe = {
            "chromatic_count": 3,
            "simulation": {
                "resolution": [128, 96],
                "domain_scale": 1.6,
                "steps": 25,
                "mass_budget_interval_steps": 12,
                "initial_pattern": "scattered",
                "deposition": 0,
                "settling_scale": 0,
                "underpaint_strength": 0,
            },
        }
        target = initial_pool_mass(self.layout, [128, 96], 1.6)
        self.report = {
            "version": VERSION,
            "interval_steps": 12,
            "initial_mass": target.tolist(),
            "corrections": [
                {
                    "step": step,
                    "mass_before": (target * 2).tolist(),
                    "factors": [0.5, 0.5, 0.5, 1.0],
                    "mass_after": target.tolist(),
                }
                for step in (12, 24, 25)
            ],
        }
        area = 4 * 1.6**2 * 128 / 96
        paint = np.broadcast_to(target / area, (96, 128, 4)).astype("f4").copy()
        self.fields = {
            "pigment": paint,
            "deposit": np.zeros_like(paint),
            "underpaint": np.zeros_like(paint),
        }

    def verify(self, report=None, recipe=None, fields=None, layout=None):
        validate_report(
            self.report if report is None else report,
            self.recipe if recipe is None else recipe,
            self.fields if fields is None else fields,
            layout=self.layout if layout is None else layout,
        )

    def test_schedule_is_canonical_and_always_includes_final(self):
        self.assertEqual(correction_steps(25, 12), [12, 24, 25])
        self.assertEqual(correction_steps(24, 12), [12, 24])
        self.assertEqual(correction_steps(10, 12), [10])
        self.assertEqual(correction_steps(25, 0), [])
        for steps, interval in ((0, 12), (10, -1), (10, True)):
            with self.assertRaises(ValueError):
                correction_steps(steps, interval)

    def test_world_area_integral_is_independent_of_sampling_density(self):
        p = np.ones((48, 64, 4), dtype="f4")
        expected = np.full(4, 4 * 1.6**2 * 64 / 48)
        np.testing.assert_allclose(pigment_mass(p, 1.6), expected)
        np.testing.assert_allclose(pigment_mass(np.ones((96, 128, 4)), 1.6), expected)

    def test_consistent_report_and_float32_final_are_accepted(self):
        self.verify()

    def test_weighted_loads_define_the_initial_and_final_budget(self):
        weights = [2.2, 0.9, 0.5]
        original = initial_pool_mass(self.layout, [128, 96], 1.6)
        weighted = initial_pool_mass(self.layout, [128, 96], 1.6, weights=weights)
        np.testing.assert_allclose(weighted, original * [*weights, 1], rtol=1e-7)
        recipe = copy.deepcopy(self.recipe)
        recipe["simulation"]["initial_pigment_weights"] = weights
        report = copy.deepcopy(self.report)
        report["initial_mass"] = weighted.tolist()
        for row in report["corrections"]:
            row["mass_before"] = (weighted * 2).tolist()
            row["mass_after"] = weighted.tolist()
        area = 4 * 1.6**2 * 128 / 96
        paint = np.broadcast_to(weighted / area, (96, 128, 4)).astype("f4").copy()
        fields = {**self.fields, "pigment": paint}
        self.verify(recipe=recipe, report=report, fields=fields)
        recipe["simulation"]["initial_pigment_weights"][0] = 2.1
        with self.assertRaisesRegex(ValueError, "regenerated starting pools"):
            self.verify(recipe=recipe, report=report, fields=fields)

    def test_initial_weights_are_bounded_explicit_and_match_chromatic_count(self):
        from .engine import Engine, validate_config
        from .palette import generate_palette

        controls = {"initial_pattern": "scattered", "underpaint_strength": 0, "deposition": 0}
        self.assertIsNone(validate_config({})["initial_pigment_weights"])
        for invalid in (
            [],
            [1, 1],
            [1] * 4,
            [1] * 6,
            [1, 0, 1],
            [1, 6, 1],
            [True, 1, 1],
            [float("nan"), 1, 1],
            [float("inf"), 1, 1],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config({**controls, "initial_pigment_weights": invalid})
        for unsupported in ({"initial_pattern": "pools"}, {"deposition": 0.01}):
            with self.assertRaisesRegex(ValueError, "source-free"):
                validate_config({**controls, "initial_pigment_weights": [1, 1, 1], **unsupported})
        with self.assertRaisesRegex(ValueError, "per chromatic pigment"):
            # Count validation precedes source planning and GPU allocation.
            Engine(
                object(),
                {**controls, "initial_pigment_weights": [1, 1, 1]},
                generate_palette("0xbc53af1cd380", 5),
                [],
            )
        values = [2.2, 0.9, 0.5]
        checked = validate_initial_weights(values, 3)
        checked[0] = 4
        self.assertEqual(values[0], 2.2)

    def test_wrong_steps_factors_or_corrected_amounts_are_rejected(self):
        for key, value in (("step", 11), ("factors", [0.6] * 4), ("mass_after", [1] * 4)):
            report = copy.deepcopy(self.report)
            report["corrections"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(report=report)
        report["corrections"] = report["corrections"][:-1]
        with self.assertRaises(ValueError):
            self.verify(report=report)

    def test_matching_report_cannot_certify_a_different_native_paint_amount(self):
        fields = {**self.fields, "pigment": self.fields["pigment"] * 1.01}
        with self.assertRaisesRegex(ValueError, "Native final paint"):
            self.verify(fields=fields)

    def test_initial_budget_is_regenerated_from_actual_pool_geometry(self):
        layout = copy.deepcopy(self.layout)
        layout["pools"][0]["radius"] *= 1.1
        with self.assertRaisesRegex(ValueError, "regenerated starting pools"):
            self.verify(layout=layout)

    def test_stationary_phases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stationary pigment"):
            self.verify(fields={**self.fields, "deposit": self.fields["pigment"]})

    def test_laminate_budget_certifies_both_moving_layers(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["simulation"].update(material_model="laminate", underpaint_release=0, burial_rate=0)
        fields = {
            **self.fields,
            "mobile": self.fields["pigment"] * 0.75,
            "underpaint": self.fields["pigment"] * 0.25,
        }
        self.verify(recipe=recipe, fields=fields)
        with self.assertRaisesRegex(ValueError, "both moving layers"):
            self.verify(recipe=recipe, fields={**fields, "mobile": fields["mobile"] * 0.9})
        with self.assertRaisesRegex(ValueError, "both moving layers"):
            self.verify(recipe=recipe, fields={**fields, "deposit": fields["pigment"]})
        with self.assertRaisesRegex(ValueError, "both moving layers"):
            self.verify(
                recipe=recipe,
                fields={
                    **fields,
                    "mobile": -fields["pigment"],
                    "underpaint": fields["pigment"] * 2,
                },
            )

    def test_empty_channels_cannot_be_created_then_erased(self):
        report = copy.deepcopy(self.report)
        report["corrections"][0]["mass_before"][-1] = 1
        report["corrections"][0]["factors"][-1] = 0
        with self.assertRaisesRegex(ValueError, "preserve empty channels"):
            self.verify(report=report)

    def test_tiny_reported_amount_cannot_reclassify_an_initially_empty_channel(self):
        report = copy.deepcopy(self.report)
        report["initial_mass"][-1] = 1e-13
        with self.assertRaisesRegex(ValueError, "regenerated starting pools"):
            self.verify(report=report)

    def test_unsupported_source_or_material_phases_cannot_be_certified(self):
        for key, value in (
            ("deposition", 1),
            ("settling_scale", 1),
            ("underpaint_strength", 1),
            ("initial_pattern", "strata"),
        ):
            recipe = copy.deepcopy(self.recipe)
            recipe["simulation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "source-free"):
                self.verify(recipe=recipe)

    def test_disabled_restoration_has_no_report(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["simulation"]["mass_budget_interval_steps"] = 0
        validate_report(None, recipe, self.fields, layout=None)
        with self.assertRaises(ValueError):
            self.verify(recipe=recipe)

    def test_malformed_reports_are_rejected(self):
        for key, value in (
            ("version", "unknown"),
            ("initial_mass", [float("nan")] * 4),
            ("initial_mass", [True] * 4),
            ("initial_mass", [10**1000] * 4),
            ("initial_mass", [1] * 3),
            ("interval_steps", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(report={**self.report, key: value})


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class WeightedInitialBudgetGPUTests(unittest.TestCase):
    def test_weighted_initial_state_and_both_layer_budgets(self):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette

        for count in (3, 5):
            weights = [2.2, 0.9, 0.5, 0.6, 0.4][:count]
            config = {
                "resolution": [96, 72],
                "steps": 40,
                "initial_pattern": "scattered",
                "material_model": "laminate",
                "initial_pigment_weights": weights,
                "deposition": 0,
                "settling_scale": 0,
                "underpaint_strength": 0,
                "underpaint_release": 0,
                "burial_rate": 0,
                "carrier_velocity": [0, 0],
                "flow_strength": 0.8,
                "mass_budget_interval_steps": 7,
                "diffusion_coefficient": 0.00002,
            }
            palette = generate_palette("0xbc53af1cd380", count, mode="composed")
            weighted = Engine(SourceFixture(), config, palette, [])
            self.addCleanup(weighted.close)
            unweighted = Engine(
                SourceFixture(), {**config, "initial_pigment_weights": None}, palette, []
            )
            self.addCleanup(unweighted.close)
            initial, baseline = weighted.snapshot(), unweighted.snapshot()
            np.testing.assert_allclose(
                initial["pigment"], baseline["pigment"] * [*weights, 1], rtol=3e-7, atol=1e-8
            )
            target = initial_pool_mass(
                weighted.layout,
                config["resolution"],
                weighted.config["domain_scale"],
                weights=weights,
            )
            np.testing.assert_allclose(
                pigment_mass(initial["pigment"], weighted.config["domain_scale"]), target, rtol=1e-7
            )
            weighted.advance_to(weighted.steps)
            validate_report(
                weighted.mass_budget_report,
                {"chromatic_count": count, "simulation": weighted.config},
                weighted.snapshot(),
                layout=weighted.layout,
            )
            weighted.close()
            unweighted.close()


if __name__ == "__main__":
    unittest.main()
