"""Small schedule and provenance checks; no image rendering or decoding."""

import copy
import unittest
from typing import Any

from align_comparison import build_plan


def fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    physics = {
        "initial_condition_f64_bits": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "integrator": "production-yoshida4-native-f64-v1",
        "integration_dt": 0.001,
        "warmup_steps": 101,
        "recording_steps": 101,
        "sample_stride": 1,
    }
    common = {
        "seed": "0xab",
        "body_ids": ["A", "B", "C"],
        "coordinate_system": "normalized_xy_top_left",
        "width": 1280,
        "height": 828,
    }
    checkpoints = [10, 30, 50, 70, 100]
    normal = {
        "seed": "0xab",
        "complete": True,
        "source_sample_count": 101,
        "source_first_step": 0,
        "source_last_step": 100,
        "source_dt": 0.001,
        "source_orbit_provenance": physics,
        "frame_count": 5,
        "frame_checkpoint_indices": checkpoints,
        "body_markers": {
            **common,
            "fps": 60,
            "frames": [
                {
                    "frame": frame,
                    "source_index": checkpoint,
                    "source_fraction": checkpoint / 100,
                    "bodies": [[frame / 10, 0.1], [0.2, 0.3], [0.4, 0.5]],
                }
                for frame, checkpoint in enumerate(checkpoints)
            ],
        },
    }
    silk = {
        **common,
        "fps": 30,
        "frames": [
            {
                "frame": frame,
                "source_fraction": fraction,
                "bodies": [[0.5, 0.5]] * 3,
            }
            for frame, fraction in enumerate([0.0, 0.2, 0.5, 0.75, 1.0])
        ],
    }
    info = {
        "frames": 5,
        "fps": 30,
        "recipe": {
            "seed": "0xab",
            "source_samples": 101,
            "source_dt": 0.001,
            "source_provenance": copy.deepcopy(physics),
            "first_visible_source_fraction": 0.0,
            "last_visible_source_fraction": 1.0,
        },
    }
    return normal, silk, info


class AlignmentTests(unittest.TestCase):
    def test_nearest_checkpoint_tie_and_missing_zero_preserve_actual_markers(self) -> None:
        normal, silk, info = fixtures()
        markers, mapping, summary = build_plan(normal, silk, info)
        self.assertEqual([item["original_normal_frame"] for item in mapping], [0, 0, 2, 3, 4])
        self.assertEqual(markers["fps"], 30)
        self.assertEqual(markers["frames"][1]["source_fraction"], 0.1)
        self.assertEqual(markers["frames"][1]["target_source_fraction"], 0.2)
        self.assertEqual(
            markers["frames"][1]["bodies"], normal["body_markers"]["frames"][0]["bodies"]
        )
        self.assertEqual(summary["first"]["source_index_error"], 10.0)
        self.assertEqual(summary["last"]["source_index_error"], 0.0)
        self.assertEqual(summary["max_abs_source_time_error_seconds"], 0.01)

    def test_shared_seed_alone_does_not_allow_different_source_physics(self) -> None:
        normal, silk, info = fixtures()
        info["recipe"]["source_provenance"]["initial_condition_f64_bits"][0][0] = 999
        with self.assertRaisesRegex(ValueError, "initial_condition_f64_bits"):
            build_plan(normal, silk, info)
        normal, silk, info = fixtures()
        info["recipe"]["source_samples"] = 102
        with self.assertRaisesRegex(ValueError, "sample counts"):
            build_plan(normal, silk, info)

    def test_identity_order_and_checkpoint_mismatch_are_rejected(self) -> None:
        normal, silk, info = fixtures()
        silk["body_ids"] = ["B", "A", "C"]
        with self.assertRaisesRegex(ValueError, "body order"):
            build_plan(normal, silk, info)
        normal, silk, info = fixtures()
        normal["body_markers"]["frames"][2]["source_index"] = 51
        with self.assertRaisesRegex(ValueError, "checkpoints disagree"):
            build_plan(normal, silk, info)


if __name__ == "__main__":
    unittest.main()
