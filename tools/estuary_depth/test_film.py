"""Exact editorial timing, matching completed paintings and real codec verification."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.estuary.run import artifact
from tools.estuary_depth import film


class FilmTests(unittest.TestCase):
    def test_complete_source_precedes_dissolve_with_exact_rational_frame_count(self):
        plan = film.timeline(901, 30, 193, 24)
        self.assertEqual(plan["formation_frames"], 721)
        self.assertEqual(plan["crossfade_start_frame"], 721)
        self.assertEqual(plan["endpoint_hold_frames"], plan["crossfade_frames"])
        self.assertEqual(plan["output_frames"], 914)
        self.assertEqual(plan["duration"], "457/12")
        self.assertEqual(plan["original_durations"], ["901/30", "193/24"])

    def test_timing_rejects_invalid_or_insufficient_recordings(self):
        for values in [
            (0, 30, 193, 24),
            (901, True, 193, 24),
            (901, 30, 24, 24),
            (901, 30, 100000, 24),
            (901.0, 30, 193, 24),
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                film.timeline(*values)

    def test_probe_requires_exact_frame_count_duration_and_resolution(self):
        info = {
            "nb_read_frames": "914",
            "avg_frame_rate": "24/1",
            "duration_ts": 467968,
            "time_base": "1/12288",
            "width": 1920,
            "height": 1440,
            "color_primaries": "bt709",
            "color_transfer": "iec61966-2-1",
            "color_space": "bt709",
            "color_range": "tv",
        }
        film.verify_timing(info, 914, 24, [1920, 1440])
        for key, value in [
            ("nb_read_frames", "913"),
            ("avg_frame_rate", "30/1"),
            ("duration_ts", 467969),
            ("width", 1280),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                film.verify_timing({**info, key: value}, 914, 24, [1920, 1440])

    def test_command_pads_completed_formation_before_transition_and_bounds_threads(self):
        args = film.command("ffmpeg", Path("/tmp/edit"), film.timeline(901, 30, 193, 24))
        graph = args[args.index("-filter_complex") + 1]
        self.assertIn("trim=end_frame=745", graph)
        self.assertIn("offset=30.041666666667", graph)
        self.assertIn("trim=end_frame=914", graph)
        self.assertEqual(args[args.index("-filter_complex_threads") + 1], "1")
        self.assertTrue(
            all(args[i + 1] == "4" for i, value in enumerate(args) if value == "-threads")
        )

    def test_wrong_seed_or_different_paint_of_same_orbit_cannot_be_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formation, orbit = root / "formation", root / "experiment" / "case"
            formation.mkdir()
            orbit.mkdir(parents=True)
            bundle_dir = orbit.parent / "inputs/bundle"
            bundle_dir.mkdir(parents=True)
            (formation / "film.mp4").write_bytes(b"formation movie")
            movie_artifact = artifact(formation / "film.mp4", formation)
            movie = {
                "artifact": movie_artifact,
                "frames": 901,
                "fps": 30,
                "full_decode_verified": True,
            }
            film.write(formation / "movie.json", movie)
            records = {
                "film.mp4": movie_artifact,
                "movie.json": artifact(formation / "movie.json", formation),
                "final-state.npy": {"sha256": "final paint"},
            }
            recipe = {"render": {"frames": 901, "fps": 30}}
            request = {"source": {"sha256": "orbit hash", "seed": "0x1234"}}
            for name in ("request", "receipt"):
                film.write(formation / f"{name}.json", request)
            film.write(orbit.parent / "experiment-request.json", {"fixture": True})
            receipt = {"source_fraction": 1.0, "source": request["source"]}
            film.write(orbit / "receipt.json", receipt)
            film.write(
                orbit / "request.json",
                {"motion": {"frames": 193, "fps": 24, "source_fraction": 1.0}},
            )
            film.write(
                orbit / "experiment-result.json",
                {
                    "movie": {
                        "sha256": "movie hash",
                        "bytes": 123,
                        "frames": 193,
                        "fps": 24,
                        "full_decode_verified": True,
                    }
                },
            )
            bundle = {
                "request": {
                    "inputs": {
                        "render_identity": "render identity",
                        "artifacts": {"final-state.npy": {"sha256": "final paint"}},
                    }
                }
            }
            film.write(bundle_dir / "manifest.json", bundle)
            with (
                patch.object(
                    film,
                    "verified_run",
                    return_value=(request, "render identity", recipe, None, records),
                ),
                patch.object(film, "finished", return_value=True),
            ):
                self.assertEqual(film.validate_inputs(formation, orbit)[0]["seed"], "0x1234")
                film.write(
                    orbit / "receipt.json",
                    {**receipt, "source": {"sha256": "orbit hash", "seed": "other"}},
                )
                with self.assertRaisesRegex(ValueError, "seed differs"):
                    film.validate_inputs(formation, orbit)
                film.write(orbit / "receipt.json", receipt)
                bundle["request"]["inputs"]["artifacts"]["final-state.npy"]["sha256"] = (
                    "different paint"
                )
                film.write(bundle_dir / "manifest.json", bundle)
                with self.assertRaisesRegex(ValueError, "different completed painting"):
                    film.validate_inputs(formation, orbit)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg and ffprobe"
    )
    def test_actual_codec_edit_reaches_both_endpoints_and_reuses_verified_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formation, orbit, output = root / "formation", root / "orbit", root / "edit"
            formation.mkdir()
            orbit.mkdir()
            inputs = {}
            for name, folder, color, frames, fps in [
                ("formation", formation, "red", 31, 30),
                ("orbit", orbit, "blue", 25, 24),
            ]:
                path = folder / "film.mp4"
                generator = f"color={color}:size=64x48:rate={fps}"
                if name == "formation":
                    generator += ",drawbox=color=lime:t=fill:enable='eq(n,30)'"
                film.capture(
                    [
                        shutil.which("ffmpeg"),
                        "-nostdin",
                        "-y",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        generator,
                        "-frames:v",
                        frames,
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-pix_fmt",
                        "yuv420p",
                        path,
                    ]
                )
                inputs[name] = {
                    "path": str(path),
                    "sha256": film.digest(path),
                    "bytes": path.stat().st_size,
                    "frames": frames,
                    "fps": fps,
                }
            source = {
                "seed": "0x1234",
                "sha256": "source",
                "render_identity": "render",
                "final_state_sha256": "state",
            }
            with patch.object(
                film,
                "validate_inputs",
                return_value=(source, inputs, film.timeline(31, 30, 25, 24)),
            ):
                result = film.compose(formation, orbit, output)
                self.assertEqual(result["movie"]["frames"], 50)
                self.assertTrue(result["movie"]["full_decode_verified"])
                decoded = subprocess.run(
                    [
                        shutil.which("ffmpeg"),
                        "-v",
                        "error",
                        "-threads",
                        "2",
                        "-i",
                        str(output / "film.mp4"),
                        "-filter_threads",
                        "1",
                        "-vf",
                        "scale=1:1",
                        "-pix_fmt",
                        "rgb24",
                        "-f",
                        "rawvideo",
                        "-",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                ).stdout
                colors = [decoded[index : index + 3] for index in range(0, len(decoded), 3)]
                self.assertEqual(len(colors), 50)
                self.assertGreater(colors[0][0], 240)  # Original red formation begins intact.
                self.assertGreater(
                    colors[24][1], 240
                )  # Last native green source frame is retained.
                self.assertGreater(colors[25][1], 240)  # Dissolve starts with that completed frame.
                self.assertTrue(80 < colors[37][1] < 180 and 80 < colors[37][2] < 180)
                self.assertGreater(colors[-1][2], 240)  # Orbit finishes fully blue.
                self.assertEqual(film.compose(formation, orbit, output), result)
                self.assertEqual(
                    result["identity_sha256"],
                    hashlib.sha256(film.encoded(film.read(output / "request.json"))).hexdigest(),
                )
                with (output / "film.mp4").open("ab") as stream:
                    stream.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "changed or missing"):
                    film.verify_complete(output)


if __name__ == "__main__":
    unittest.main()
