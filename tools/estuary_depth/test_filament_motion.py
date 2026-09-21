"""Paired-photo fidelity, complete source cadence and shared lighting simulations."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.estuary.recipe import validate_recipe
from tools.estuary.run import exposure_plan, frame_plan
from tools.estuary_depth import filament_motion as motion
from tools.estuary_depth.filament_gallery import LIGHTING
from tools.estuary_depth.filament_studies import (
    REFERENCE_SEED,
    VARIANTS,
    make_depth_recipe,
    make_paint_recipe,
)
from tools.estuary_depth.render import camera_pose, motion_angles
from tools.estuary_depth.render import recipe as validate_depth_recipe


class FilamentMotionTests(unittest.TestCase):
    def test_catalog_covers_all_fifteen_existing_options_without_changing_their_looks(self):
        self.assertEqual(set(motion.OPTIONS), set(VARIANTS) | set(LIGHTING))
        self.assertEqual(len(motion.OPTIONS), 15)
        for option, spec in motion.OPTIONS.items():
            with self.subTest(option=option):
                if option in VARIANTS:
                    expected = make_depth_recipe(REFERENCE_SEED, VARIANTS[option].label)
                    self.assertEqual(spec.material_variant, option)
                else:
                    _, label, parameter, value = LIGHTING[option]
                    expected = make_depth_recipe(REFERENCE_SEED, label)
                    if parameter == "relief_mm":
                        expected[parameter] = value
                    else:
                        expected["lighting"][parameter] = value
                    self.assertEqual(spec.material_variant, "control")
                self.assertEqual(motion.make_photo_recipe(REFERENCE_SEED, option), expected)

    def test_formation_keeps_paint_physics_and_samples_both_source_endpoints(self):
        for option, spec in motion.OPTIONS.items():
            with self.subTest(option=option):
                original = make_paint_recipe(REFERENCE_SEED, spec.material_variant)
                result = motion.make_formation_recipe(REFERENCE_SEED, option)
                self.assertEqual(result, validate_recipe(result))
                for key in ("simulation", "projection", "optics"):
                    self.assertEqual(result[key], original[key])
                self.assertEqual(result["render"]["resolution"], [1920, 1440])
                self.assertEqual(result["render"]["fps"], 24)
                self.assertEqual(result["render"]["frames"], 721)
                self.assertEqual(result["render"]["temporal_samples"], 4)
                steps = frame_plan(result["simulation"]["steps"], result["render"]["frames"])
                self.assertEqual(steps, list(range(0, 7201, 10)))
                exposures = exposure_plan(steps, result["render"]["temporal_samples"])
                self.assertEqual(exposures[0], [0])
                self.assertEqual(exposures[-1], [7197, 7198, 7199, 7200])
        control = motion.make_formation_recipe(REFERENCE_SEED, "control")
        for option in LIGHTING:
            self.assertEqual(motion.make_formation_recipe(REFERENCE_SEED, option), control)

    def test_camera_finishes_at_the_photo_pose_and_all_other_scene_controls_match(self):
        for option in motion.OPTIONS:
            with self.subTest(option=option):
                photo = motion.make_photo_recipe(REFERENCE_SEED, option)
                movie = motion.make_motion_recipe(REFERENCE_SEED, option)
                self.assertEqual(movie, validate_depth_recipe(movie))
                for key in photo.keys() - {"camera", "render"}:
                    self.assertEqual(movie[key], photo[key])
                self.assertEqual(movie["render"]["resolution"], [1920, 1440])
                self.assertEqual(movie["render"]["samples"], 32)
                self.assertEqual(movie["render"]["seed"], photo["render"]["seed"])
                start = motion_angles(movie["camera"], 0, motion.MOTION_FRAMES)
                end = motion_angles(movie["camera"], motion.MOTION_FRAMES - 1, motion.MOTION_FRAMES)
                expected = motion_angles(photo["camera"], 0, 1)
                self.assertEqual(end, expected)
                self.assertTrue(0 < np.linalg.norm(np.asarray(end) - start) < 6)
                target = [*photo["camera"]["target"], 0.001]
                np.testing.assert_array_equal(
                    camera_pose(*end, target), camera_pose(*expected, target)
                )

    def test_explicit_master_preserves_the_existing_broad_pool_master_recipe(self):
        result = motion.make_photo_recipe(REFERENCE_SEED, "three-broad-pools", master=True)
        self.assertEqual(
            result, make_depth_recipe(REFERENCE_SEED, "Three broad pools", proof=False)
        )
        self.assertEqual(result["render"]["resolution"], [3840, 2880])
        self.assertEqual(result["render"]["samples"], 256)

    def test_seed_determinism_and_returned_mutations_do_not_change_other_recipes(self):
        for function in (
            motion.make_formation_recipe,
            motion.make_photo_recipe,
            motion.make_motion_recipe,
        ):
            first = function(17, "control")
            self.assertEqual(first, function("0X0011", "control"))
            self.assertNotEqual(first, function(17 + (1 << 240), "control"))
            before = copy.deepcopy(first)
            first["render"]["resolution"][0] = 128
            self.assertEqual(before, function(17, "control"))

    def test_unknown_options_and_non_boolean_quality_flags_are_rejected(self):
        for function in (
            motion.make_formation_recipe,
            motion.make_photo_recipe,
            motion.make_motion_recipe,
        ):
            for option in ("Control", "light-unknown", "../control", True, None, []):
                with (
                    self.subTest(function=function.__name__, option=option),
                    self.assertRaises(ValueError),
                ):
                    function(0, option)
        for value in (0, 1, "false", None):
            with self.subTest(master=value), self.assertRaises(ValueError):
                motion.make_photo_recipe(0, "control", master=value)


if __name__ == "__main__":
    unittest.main()
