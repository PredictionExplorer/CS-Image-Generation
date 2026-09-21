"""Frozen legacy raster parity and explicit component pigment budgets."""

import copy
import hashlib
import unittest

import numpy as np

from .initial_composition import plan_layout
from .initial_composition import rasterize as composition_raster
from .mass_budget import pigment_mass
from .test_initial_composition import CONFIG, InitialSource
from .vector_raster import rasterize

FROZEN = {
    "random-circles": "d2510757ad6a5e36d5ba8332391d2446ca4846275a2ce4b267e2546245d44145",
    "random-ribbons": "4b8e0bdeecb90af44c078ecc02d0fa3c79680a8cc1c9173fa2262a0ced4ecc18",
    "random-crescents": "1bdf3721332adbd431e05488c435b1f83739c937f35441923e1c75ab2469becd",
    "facing-shores": "d1340167ac07dbea673766dae2f73421352f6b0708a332e78b92e216adf69cba",
    "scattered-commas": "3b0307dc06a10a1a243d9ad50db107b9263624f9a068611449cfac787e23f903",
    "body-wedges": "9010a1c5a86b15cca294f852a8b675b378e4f35b468003aaacd2bc195fd11bc0",
}


class VectorRasterTests(unittest.TestCase):
    def test_original_six_setup_rasters_match_pre_extraction_bytes(self):
        for setup, expected in FROZEN.items():
            layout = plan_layout(
                "0x808861c25b6c",
                3,
                4 / 3,
                {**CONFIG, "setup": setup},
                InitialSource() if setup == "body-wedges" else None,
            )
            with self.subTest(setup=setup):
                self.assertEqual(
                    hashlib.sha256(
                        composition_raster(layout, (512, 384), 1.6).tobytes()
                    ).hexdigest(),
                    expected,
                )

    @staticmethod
    def primitives():
        return [
            {
                "kind": "stroke",
                "points": [[x, y]],
                "radii": [radius],
                "pigment_index": pigment,
                "target_mass": mass,
            }
            for x, y, radius, pigment, mass in (
                (-0.6, 0, 0.2, 0, 0.021),
                (0.6, 0, 0.09, 0, 0.009),
                (-0.4, 0.6, 0.13, 1, 0.012),
                (0.4, 0.6, 0.16, 2, 0.006),
            )
        ]

    def render(self, parts, **kwargs):
        return rasterize(
            parts,
            (512, 384),
            1.6,
            aspect=4 / 3,
            target_mass=[0.03, 0.012, 0.006],
            edge_width_world=0.002,
            **kwargs,
        )

    def test_split_budgets_are_realized_despite_unequal_component_areas(self):
        parts = self.primitives()
        saved = copy.deepcopy(parts)
        field = self.render(parts)
        cell_area = (3.2 / 384) ** 2
        np.testing.assert_allclose(pigment_mass(field, 1.6), [0.03, 0.012, 0.006], rtol=2e-7)
        np.testing.assert_allclose(
            [
                field[:, :256, 0].sum(dtype="f8") * cell_area,
                field[:, 256:, 0].sum(dtype="f8") * cell_area,
            ],
            [0.021, 0.009],
            rtol=1e-7,
            atol=1e-12,
        )
        self.assertEqual(parts, saved)

    def test_partial_or_inconsistent_component_budgets_are_rejected(self):
        partial = self.primitives()
        del partial[0]["target_mass"]
        with self.assertRaisesRegex(ValueError, "every vector"):
            self.render(partial)
        wrong = self.primitives()
        wrong[1]["target_mass"] *= 2
        with self.assertRaisesRegex(ValueError, "total pigment"):
            self.render(wrong)

    def test_peak_guard_rejects_without_clipping_or_renormalizing(self):
        field = self.render(self.primitives())
        peak = field.max(axis=(0, 1)).astype("f8")
        np.testing.assert_array_equal(
            self.render(self.primitives(), peak_limits=peak * 1.01), field
        )
        with self.assertRaisesRegex(ValueError, "peak concentration"):
            self.render(self.primitives(), peak_limits=peak * 0.9)


if __name__ == "__main__":
    unittest.main()
