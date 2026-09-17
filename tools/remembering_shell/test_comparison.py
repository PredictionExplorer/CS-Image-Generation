"""Timing, provenance, publication and real decoder checks for comparisons."""

import copy
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import comparison as c


def fixture():
    plan = c.alignment()
    orbit = {
        "seed": "0xabcd",
        "sha256": "file-hash",
        "samples_sha256": "sample-hash",
        "provenance": {"sample_stride": 1},
        "dt": 0.001,
        "count": 1_000_000,
    }
    normal = {
        "complete": True,
        "seed": orbit["seed"],
        "source_sample_count": orbit["count"],
        "source_first_step": 0,
        "source_last_step": 999999,
        "frame_checkpoint_indices": plan["normal_checkpoints"],
        "fps": 60,
        "frame_count": 1802,
        "source_orbit_provenance": orbit["provenance"],
        "source_samples_sha256": orbit["samples_sha256"],
        "source_dt": orbit["dt"],
    }
    frames = [
        {
            "index": x["frame"],
            "geometry_index": x["frame"],
            "phase": "excavation",
            "source_fraction": x["source_fraction"],
        }
        for x in plan["frames"]
    ]
    timeline = [
        {"kind": "excavation", "render_frame": i, "slot": i, "time_seconds": i / 30}
        for i in range(901)
    ]
    request = {
        "inputs": {"orbit": {"sha256": orbit["sha256"]}},
        "frames": frames,
        "encoding_timeline": timeline,
        "fps": 30,
    }
    identity = hashlib.sha256(c.encoded(request)).hexdigest()
    film = {
        "complete": True,
        "request_sha256": identity,
        "frames": [{**frame, "complete": True} for frame in frames],
        "encoding_timeline": copy.deepcopy(timeline),
    }
    movie = {"full_decode_verified": True, "request_sha256": identity, "fps": 30, "frames": 901}
    return film, request, movie, normal, orbit, plan


class AlignmentTests(unittest.TestCase):
    def test_complete_pair_endpoints(self):
        plan = c.alignment()
        self.assertEqual(len(plan["normal_checkpoints"]), 1802)
        self.assertEqual(len(plan["frames"]), 901)
        self.assertEqual(
            plan["frames"][0],
            {"frame": 0, "normal_frame": 1, "checkpoint": 1110, "source_fraction": 1110 / 999999},
        )
        self.assertEqual(
            plan["frames"][-1],
            {"frame": 900, "normal_frame": 1801, "checkpoint": 999999, "source_fraction": 1},
        )
        self.assertEqual(plan["duration_seconds"], 1802 / 60)
        self.assertTrue(
            all(
                a["checkpoint"] < b["checkpoint"]
                for a, b in zip(plan["frames"], plan["frames"][1:], strict=False)
            )
        )

    def test_invalid_or_unpairable_schedule(self):
        for samples in (True, 1, 1.5, 100_000_001, 1800):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                c.alignment(samples)

    def test_normal_requires_actual_original_checkpoints(self):
        _, _, _, normal, _, plan = fixture()
        c.validate_normal(normal, plan)
        for key, value in [
            ("complete", False),
            ("source_first_step", 555),
            ("fps", 30),
            ("source_last_step", 999555),
            ("frame_count", 1800),
            ("frame_checkpoint_indices", list(range(1802))),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_normal({**normal, key: value}, plan)

    def test_shell_requires_complete_bound_provenance(self):
        values = fixture()
        c.validate_shell(*values)
        mutations = [
            (0, "complete", False),
            (0, "request_sha256", "stale"),
            (2, "full_decode_verified", False),
            (2, "frames", 900),
            (3, "source_samples_sha256", "different"),
            (3, "source_dt", 0.002),
            (4, "seed", "0xeeee"),
            (4, "sha256", "other-orbit"),
        ]
        for index, key, value in mutations:
            altered = copy.deepcopy(values)
            altered[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_shell(*altered)

    def test_shell_rejects_holds_or_linear_clock(self):
        for mutation in ("hold", "time", "missing", "incomplete"):
            altered = copy.deepcopy(fixture())
            film = altered[0]
            if mutation == "hold":
                film["encoding_timeline"][0]["kind"] = "start_hold"
            elif mutation == "time":
                film["frames"][0]["source_fraction"] = 0
            elif mutation == "missing":
                film["frames"].pop()
            else:
                film["frames"][10]["complete"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.validate_shell(*altered)


class ArchiveTests(unittest.TestCase):
    def test_published_media_is_immutable_even_before_movies_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "source.png", root / "shell.png"
            source.write_bytes(b"first rendered image")
            record = {"sha256": c.digest(source), "bytes": source.stat().st_size}
            c.publish_copy(source, target, record)
            c.publish_copy(source, target, record)
            c.write(root / "stills.json", {"artifacts": {"shell.png": record}})
            source.write_bytes(b"a different image")
            changed = {"sha256": c.digest(source), "bytes": source.stat().st_size}
            with self.assertRaises(ValueError):
                c.publish_copy(source, target, changed)
            self.assertEqual(target.read_bytes(), b"first rendered image")
            target.unlink()
            with self.assertRaises(ValueError):
                c.publish_copy(source, target, changed)

    def test_orbit_inspection_binds_exact_raw_sample_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.orbit"
            header = {
                "seed": "0xabcd",
                "dt": 0.001,
                "masses": [1, 2, 3],
                "count": 2,
                "provenance": {"sample_stride": 1},
            }
            payload = struct.pack("<18d", *range(18))
            raw = json.dumps(header).encode()
            path.write_bytes(b"CSORBIT1" + struct.pack("<Q", len(raw)) + raw + payload)
            observed = c.inspect_orbit(path)
            expected = hashlib.sha256(struct.pack("<4d", 0.001, 1, 2, 3) + payload).hexdigest()
            self.assertEqual(observed["samples_sha256"], expected)
            self.assertEqual(observed["sha256"], c.digest(path))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                c.inspect_orbit(path)

    def test_collection_retains_pending_seeds_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c.update_gallery(root, ["0xaaaa", "0xbbbb"])
            c.update_gallery(root, completed="0xaaaa")
            data = c.read(root / "collection.json")
            self.assertEqual(data["seeds"]["0xbbbb"]["status"], "pending")
            self.assertEqual(data["seeds"]["0xaaaa"]["status"], "ready")
            with self.assertRaises(ValueError):
                c.update_gallery(root, ["../invalid"])

    def test_artifact_hash_detects_same_length_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "artifact"
            path.write_bytes(b"original")
            record = {"bytes": 8, "sha256": c.digest(path)}
            c.checked_artifact(path, record)
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                c.checked_artifact(path, record)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class EncoderTests(unittest.TestCase):
    def test_actual_pair_selection_and_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "source.rgb"
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
            raw.write_bytes(b"".join(bytes(color) * 16 * 16 for color in colors))
            original = root / "original.mp4"
            c.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "16x16",
                    "-r",
                    "60",
                    "-i",
                    str(raw),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "scale=out_color_matrix=bt709:out_range=tv",
                    "-color_primaries",
                    "smpte432",
                    "-color_trc",
                    "iec61966-2-1",
                    "-colorspace",
                    "bt709",
                    "-color_range",
                    "tv",
                    str(original),
                ]
            )
            c.verify_video(original, "ffmpeg", "ffprobe", [16, 16], 4, 60)
            result, _ = c.transcode(
                [original],
                c.NORMAL_PAIR_FILTER,
                root / "paired.mp4",
                "ffmpeg",
                2,
            )
            receipt = c.verify_video(result, "ffmpeg", "ffprobe", [16, 16], 2, 30)
            self.assertTrue(receipt["full_decode_verified"])
            self.assertEqual(receipt["probe"]["color_primaries"], "bt709")
            pixels = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(result),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            self.assertLess(pixels[0], 5)
            self.assertGreater(pixels[1], 245)
            self.assertTrue(all(value > 245 for value in pixels[768:771]))
            with self.assertRaises(ValueError):
                c.verify_video(result, "ffmpeg", "ffprobe", [16, 16], 3, 30)


if __name__ == "__main__":
    unittest.main()
