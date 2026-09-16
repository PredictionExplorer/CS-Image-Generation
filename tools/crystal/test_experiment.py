"""Small deterministic archive tests; no actual renderer is executed."""

import copy
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "crystal_experiment", Path(__file__).with_name("experiment.py")
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def base_recipe():
    return {
        "kind": "crystal",
        "frames": 1802,
        "fps": 60,
        "temporal_samples": 4,
        "shutter_fraction": 0.5,
        "prelude_fraction": 0.0,
        "camera": {"orthographic_height": 7.2},
        "render": {"width": 4, "height": 4, "aa": 2},
        "crystal": {
            "freeze_source_fraction": None,
            "polarizer_sweep_degrees": 0.0,
            "optics": {"retardance_scale_nm": 3000.0, "polarizer_degrees": 0.0},
            "field": {
                "memory_fraction": 0.02,
                "memory_weight": 0.28,
                "memory_samples": 12,
                "load_softness": 0.55,
            },
            "absorption": {"x": 0.1, "y": 0.07, "z": 0.05},
            "surface_strength": 0.018,
            "edge_strength": 0.14,
            "edge_thickness": 0.65,
        },
    }


def canonical_recipe(recipe):
    """Simulate real serde additions/omissions, not an unchanged input echo."""
    result = copy.deepcopy(recipe)
    crystal = result["crystal"]
    crystal.setdefault("surface_depth", 0.65)
    crystal["field"].setdefault("preconditioner", "incomplete_cholesky")
    optics = crystal["optics"]
    if not optics.get("spectral_peaks"):
        optics.pop("spectral_peaks", None)
    if optics.get("spectral_floor", 1) == 1:
        optics.pop("spectral_floor", None)
    for inactive in ["calligraphy", "loom", "aurora", "light", "engraving", "eclipse"]:
        result.pop(inactive, None)
    return result


def fake_process(command, log):
    """Write synthetic receipt/PNG fixtures for archive validation, not artwork."""

    def argument(name):
        return command[command.index(name) + 1]

    output = Path(argument("--output"))
    recipe = canonical_recipe(experiment.finite_json(Path(argument("--config"))))
    if "resolve-crystal" in command:
        experiment.write_json(output, recipe)
        log.write_text("Synthetic canonical-resolution process only.\n")
        return 0.01
    output.mkdir(exist_ok=True)
    frame = int(argument("--frame"))
    width, height = recipe["render"]["width"], recipe["render"]["height"]

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = zlib.compress((b"\0" + b"\x20\x30\x40" * width) * height)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )
    image = output / f"frame_{frame:06}.png"
    image.write_bytes(png)
    experiment.write_json(
        output / "render.json",
        {
            "complete": True,
            "rendered_frames": [frame],
            "config": recipe,
            "orbit_sha256": experiment.digest(Path(argument("--orbit"))),
            "executable_sha256": experiment.digest(Path(command[0])),
        },
    )
    experiment.write_json(
        image.with_name(image.name + ".json"),
        {
            "source_fraction": frame / (recipe["frames"] - 1),
            "png_sha256": experiment.digest(image),
            "crystal_samples": [{} for _ in range(recipe["temporal_samples"])],
            "seconds": 0.125,
        },
    )
    log.write_text("Synthetic test process only.\n")
    return 0.25


class ExperimentTests(unittest.TestCase):
    def arguments(self, directory):
        base = directory / "base.json"
        base.write_text(json.dumps(base_recipe()))
        orbit = directory / "orbit"
        orbit.write_bytes(b"test orbit placeholder")
        return experiment.parser().parse_args(
            [
                "--executable",
                sys.executable,
                "--orbit",
                str(orbit),
                "--base-recipe",
                str(base),
                "--output",
                str(directory / "proofs"),
                "--frames",
                "0,1801",
                "--variants",
                "A",
                "E",
            ]
        )

    def test_matrix_is_exact_and_base_stays_unchanged(self):
        base = base_recipe()
        original = copy.deepcopy(base)
        candidates = experiment.make_candidates(base, list("ABCDEF"), {})
        self.assertEqual(base, original)
        self.assertEqual(
            [c["recipe"]["crystal"]["optics"]["retardance_scale_nm"] for c in candidates],
            [12000, 18000, 30000, 18000, 18000, 18000],
        )
        self.assertEqual(candidates[3]["recipe"]["crystal"]["optics"]["polarizer_degrees"], 22.5)
        self.assertEqual(candidates[4]["recipe"]["crystal"]["field"]["memory_samples"], 24)
        self.assertEqual(candidates[5]["recipe"]["crystal"]["field"]["load_softness"], 0.9)
        self.assertEqual(candidates[5]["recipe"]["camera"]["orthographic_height"], 6.4)
        self.assertEqual(candidates[0]["recipe"]["crystal"]["field"]["memory_weight"], 0.28)

    def test_no_implicit_quality_changes_and_explicit_overrides(self):
        base = base_recipe()
        candidate = experiment.make_candidates(base, ["A"], {})[0]["recipe"]
        self.assertEqual(candidate["render"], base["render"])
        self.assertEqual(candidate["temporal_samples"], 4)
        override = experiment.make_candidates(
            base, ["A"], {"render.width": 12, "temporal_samples": 2}
        )[0]["recipe"]
        self.assertEqual(override["render"]["width"], 12)
        self.assertEqual(override["temporal_samples"], 2)

    def test_complete_archive_resume_and_conflicting_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            with mock.patch.object(experiment, "run_process", side_effect=fake_process) as runner:
                experiment.experiment(args)
                self.assertEqual(runner.call_count, 6)
            output = args.output
            summary = experiment.finite_json(output / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertTrue(all(p["render_seconds"] == 0.125 for p in summary["proofs"]))
            self.assertFalse((output / ".experiment.lock").exists())
            with self.assertRaises(ValueError):
                experiment.experiment(args)
            args.resume = True
            with mock.patch.object(experiment, "run_process", side_effect=fake_process):
                experiment.experiment(args)
            args.orbit.write_bytes(b"changed source")
            with self.assertRaises(ValueError):
                experiment.experiment(args)

    def test_failed_job_preserves_evidence_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))

            def failed_renderer(command, log):
                if "resolve-crystal" in command:
                    return fake_process(command, log)
                raise RuntimeError("bad <renderer>")

            with (
                mock.patch.object(experiment, "run_process", side_effect=failed_renderer),
                self.assertRaises(RuntimeError),
            ):
                experiment.experiment(args)
            summary = experiment.finite_json(args.output / "summary.json")
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["proofs"][0]["status"], "failed")
            self.assertFalse((args.output / ".experiment.lock").exists())
            page = (args.output / "index.html").read_text()
            self.assertIn("bad &lt;renderer&gt;", page)
            self.assertNotIn("bad <renderer>", page)
            self.assertIn("Development proofs.", page)

    def test_invalid_inputs_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.frames = ["1802"]
            with self.assertRaises(ValueError):
                experiment.experiment(args)
            self.assertFalse(args.output.exists())
            args.frames = ["0"]
            args.width = 16
            with self.assertRaises(ValueError):
                experiment.experiment(args)
            self.assertFalse(args.output.exists())

    def test_incomplete_receipts_and_corrupt_images_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.frames, args.variants = ["0"], ["A"]

            def bad_process(command, log):
                duration = fake_process(command, log)
                if "resolve-crystal" in command:
                    return duration
                output = Path(command[command.index("--output") + 1])
                (output / "frame_000000.png").write_bytes(b"not a png")
                return duration

            with (
                mock.patch.object(experiment, "run_process", side_effect=bad_process),
                self.assertRaises(ValueError),
            ):
                experiment.experiment(args)
            self.assertFalse(experiment.finite_json(args.output / "summary.json")["complete"])

    def test_resolved_defaults_are_archived_rendered_and_required_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.frames, args.variants = ["0"], ["A"]
            requested = base_recipe()
            requested["crystal"]["optics"].update(spectral_peaks=[], spectral_floor=1.0)
            requested["calligraphy"] = {"history_fraction": 0.2}
            args.base_recipe.write_bytes(experiment.encoded(requested))
            with mock.patch.object(experiment, "run_process", side_effect=fake_process) as runner:
                experiment.experiment(args)
            self.assertIn("resolve-crystal", runner.call_args_list[0].args[0])
            archived = experiment.finite_json(args.output / "A" / "recipe.json")
            self.assertNotIn("spectral_peaks", archived["crystal"]["optics"])
            self.assertNotIn("spectral_floor", archived["crystal"]["optics"])
            self.assertNotIn("calligraphy", archived)
            self.assertEqual(archived["crystal"]["surface_depth"], 0.65)
            self.assertEqual(archived["crystal"]["field"]["preconditioner"], "incomplete_cholesky")
            self.assertEqual(experiment.finite_json(args.output / "base-recipe.json"), requested)
            summary = experiment.finite_json(args.output / "summary.json")
            self.assertTrue(summary["complete"])
            proof_manifest = experiment.finite_json(args.output / summary["proofs"][0]["manifest"])
            self.assertEqual(proof_manifest["config"], archived)
            args.resume = True
            with mock.patch.object(experiment, "run_process", side_effect=fake_process):
                experiment.experiment(args)

            def changed_resolution(command, log):
                duration = fake_process(command, log)
                if "resolve-crystal" in command:
                    path = Path(command[command.index("--output") + 1])
                    value = experiment.finite_json(path)
                    value["crystal"]["surface_depth"] = 0.75
                    experiment.write_json(path, value)
                return duration

            with (
                mock.patch.object(experiment, "run_process", side_effect=changed_resolution),
                self.assertRaisesRegex(ValueError, "identical resolved"),
            ):
                experiment.experiment(args)


if __name__ == "__main__":
    unittest.main()
