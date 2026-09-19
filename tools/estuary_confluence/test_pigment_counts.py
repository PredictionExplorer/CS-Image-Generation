"""One- and two-pigment studies keep genuine, deterministic material prefixes."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from .assessment import assess
from .backgrounds import generate_background
from .layout import plan_layout
from .mass_budget import initial_pool_mass, validate_report
from .palette import generate_palette, validate_palette
from .participation_layout import plan_engaged_layout
from .spectral import build_spectral_material, reflectance
from .test_participation_layout import Orbit, config


class PigmentCountTests(unittest.TestCase):
    def test_all_palette_modes_retain_exact_material_prefix_and_complete_seed(self):
        keys = (
            "pigments_srgb",
            "pigment_names",
            "pigment_roles",
            "pigment_ids",
            "scattering",
            "settling",
            "release",
            "specific_volumes",
            "granulation",
        )
        for mode in ("curated", "harmonic", "random", "composed"):
            for seed in (17, 17 | (1 << 255)):
                full = generate_palette(seed, 5, mode=mode)
                for count in (1, 2, 3):
                    with self.subTest(mode=mode, seed=seed, count=count):
                        palette = generate_palette(seed, count, mode=mode)
                        self.assertEqual(palette, generate_palette(hex(seed), count, mode=mode))
                        self.assertEqual(validate_palette(json.loads(json.dumps(palette))), palette)
                        for key in (*keys, *(("layer_fractions",) if mode == "composed" else ())):
                            self.assertEqual(palette[key], [*full[key][:count], full[key][-1]])
                        self.assertEqual(palette["quality"], full["quality"])
                        self.assertEqual(palette["substrate_seed"], full["substrate_seed"])
                        self.assertEqual(palette["chalk_index"], count)
                        self.assertLess(palette["underpaint_index"], count)
                        mixtures = np.asarray(palette["body_mixtures"])
                        self.assertEqual(mixtures.shape, (3, count + 1))
                        np.testing.assert_allclose(mixtures.sum(axis=1), 1, atol=2e-16)
                        self.assertTrue(np.all(mixtures >= 0))
                        self.assertEqual(
                            list(np.argmax(mixtures, axis=1)), [i % count for i in range(3)]
                        )
                        np.testing.assert_array_equal(
                            mixtures[:, -1], np.asarray(full["body_mixtures"])[:, -1]
                        )
                        self.assertEqual(
                            generate_background("palette-night", palette)["ground_srgb"],
                            generate_background("palette-night", full)["ground_srgb"],
                        )

    def test_released_three_and_five_palettes_keep_exact_identity(self):
        fixtures = {
            ("curated", 3): "7373b90009207fa66870aa5502188af704c8376f33882ac454fcad3885f448f9",
            ("curated", 5): "8a1a12ccbf917fed4ec29fc66c5df25553e7495ec9992a1adcfd83e9f5e6e078",
            ("harmonic", 3): "a866c87e726d8983c5bd960af52bbf1fbb9c3b30ff8e2f371ef8636e2012cfad",
            ("harmonic", 5): "646e8c9ed2f8c4c42f5c3d9dca72d7c9ec49717dcaa4827240e2f3018dc0efe8",
            ("random", 3): "212b8ab21033cc116d3fa44165be46e6560ad0f5d362b3a6cd8e627c4f0aa27a",
            ("random", 5): "8af6396751054ea22beb2ddff4ff4e0c2ed8745ae6429843f4ba08f728648e46",
            ("composed", 3): "62696b9263c664a745fbf93622cc00b77dfd68c1a75dedd8bac4febc43417790",
            ("composed", 5): "d4087964da64e443e03a84fa94d424d78adb409c7ad93315fd88d877072f10cb",
        }
        for (mode, count), identity in fixtures.items():
            self.assertEqual(
                generate_palette("0x808861c25b6c", count, mode=mode)["identity_sha256"], identity
            )

    def test_pools_and_weighted_mass_are_exact_prefixes_without_extra_chalk(self):
        for seed in (0, 2**256 - 1):
            full = plan_layout(seed, 5, 4 / 3)
            weights = [2.2, 0.9, 0.5, 0.6, 0.4]
            reference = initial_pool_mass(full, [128, 96], 1.6, weights=weights)
            for count in (1, 2):
                smaller = plan_layout(seed, count, 4 / 3)
                self.assertEqual(smaller["pools"], full["pools"][:count])
                observed = initial_pool_mass(smaller, [128, 96], 1.6, weights=weights[:count])
                np.testing.assert_array_equal(observed, [*reference[:count], 0])
                self.assertTrue(np.all(observed[:count] > 0))

    def test_engaged_prefix_keeps_selection_and_labels_pilot_scope_honestly(self):
        with (
            patch("tools.estuary_confluence.participation_layout.PILOT_STEPS", 96),
            patch("tools.estuary_confluence.participation_layout.CANDIDATE_LAYOUTS", 6),
        ):
            full = plan_engaged_layout(Orbit(), 5, config())
            for count in (1, 2):
                smaller = plan_engaged_layout(Orbit(), count, config())
                self.assertEqual(smaller["pools"], full["pools"][:count])
                self.assertEqual(smaller["pilot"], full["pilot"])
                self.assertEqual(len(smaller["pilot"]["selection"]["prefix_three"]["per_pool"]), 3)
                self.assertEqual(len(smaller["pilot"]["selection"]["all_five"]["per_pool"]), 5)

    def test_one_pigment_has_tonal_depth_without_a_second_material(self):
        palette = generate_palette("0x808861c25b6c", 1, mode="composed")
        samples = np.array([[0, 0], [0.045, 0], [0.18, 0], [0.7, 0]])
        colors = reflectance(samples, build_spectral_material(palette), layer_scale=12)
        self.assertEqual(colors.shape, (4, 3))
        self.assertTrue(np.isfinite(colors).all())
        self.assertTrue(np.all((colors >= 0) & (colors <= 1)))
        luminance = colors @ [0.2126, 0.7152, 0.0722]
        self.assertTrue(np.all(np.diff(luminance) < 0))
        self.assertGreater(float(luminance[1] - luminance[-1]), 0.1)

    def test_single_pigment_contact_is_zero_and_two_pigment_contact_is_measurable(self):
        for count in (1, 2):
            material = np.zeros((12, 12, count + 1), dtype="f4")
            material[..., :count] = 1 / count
            report = assess(material, count, 1)
            self.assertEqual(len(report["pigments"]), count)
            self.assertEqual(report["minimum_contact_mass_fraction"], count - 1)
            self.assertEqual(report["shared_painted_area_fraction"], count - 1)
            for pigment in report["pigments"]:
                self.assertEqual(pigment["dominant_area_fraction"], 1 / count)

    def test_night_ground_never_uses_unused_chalk_as_a_chromatic_anchor(self):
        for count in (1, 2):
            palette = {
                "seed": "0x12",
                "chromatic_count": count,
                "pigments_srgb": [[0.5, 0.5, 0.5]] * count + [[1, 0, 0]],
            }
            background = generate_background("palette-night", palette)
            self.assertTrue(background["method"]["neutral_fallback"])
            self.assertLess(background["method"]["anchor_pigment_index"], count)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class PigmentCountGPUTests(unittest.TestCase):
    def test_fewer_pigments_survive_native_layered_render_and_budget_verification(self):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .surface import Surface

        for count in (1, 2):
            palette = generate_palette("0x808861c25b6c", count, mode="composed")
            engine = Engine(
                SourceFixture(),
                {
                    "resolution": [96, 72],
                    "steps": 40,
                    "initial_pattern": "scattered",
                    "initial_pigment_weights": [2.2, 0.9][:count],
                    "material_model": "laminate",
                    "lower_transport_scale": 0.82,
                    "interlayer_exchange_rate": 0.55,
                    "carrier_velocity": [0, 0],
                    "deposition": 0,
                    "settling_scale": 0,
                    "underpaint_strength": 0,
                    "underpaint_release": 0,
                    "burial_rate": 0,
                    "substrate_um": 0,
                    "mass_budget_interval_steps": 7,
                    "diffusion_coefficient": 0.00002,
                },
                palette,
                [],
            )
            self.addCleanup(engine.close)
            engine.advance_to(engine.steps)
            fields = engine.snapshot()
            self.assertEqual(fields["pigment"].shape, (72, 96, count + 1))
            for phase in ("pigment", "mobile", "deposit", "underpaint"):
                np.testing.assert_array_equal(fields[phase][..., -1], 0)
            self.assertGreater(float(fields["underpaint"].sum()), 0)
            validate_report(
                engine.mass_budget_report,
                {"chromatic_count": count, "simulation": engine.config},
                fields,
                layout=engine.layout,
            )
            for optics in ("rgb", "spectral"):
                surface = Surface(
                    {"optics_model": optics, "finish": "glazed"},
                    palette,
                    gpu_frame=engine.gpu_frame(),
                )
                self.addCleanup(surface.close)
                cpu = surface.render(fields, (128, 96))
                gpu = surface.render_gpu(engine.gpu_frame(), (128, 96))
                self.assertTrue(np.isfinite(gpu).all())
                np.testing.assert_allclose(gpu, cpu, atol=2e-6, rtol=2e-5)
                surface.close()
            engine.close()


if __name__ == "__main__":
    unittest.main()
