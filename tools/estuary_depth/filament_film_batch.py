"""Durable, receipt-bound image and full-film batches for the fine-fold studies.

Each material is simulated once and shared by its photographic options. A case
becomes complete only after its still, camera film and full-formation edit verify.
Failed attempts are preserved, while certified paint checkpoints can resume.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.estuary.run import checked_artifact, completed, exposure_plan, frame_plan
from tools.estuary_confluence.run import runtime_identity
from tools.estuary_depth import film
from tools.estuary_depth.experiment import finished, verify_render
from tools.estuary_depth.filament_batch import (
    SPECIFIC_VOLUMES,
    _runtime_contract,
    _verify_paint_preparation,
    _verify_photo_runtime,
)
from tools.estuary_depth.filament_motion import (
    FPS,
    MOTION_FRAMES,
    OPTIONS,
    make_formation_recipe,
    make_motion_recipe,
    make_photo_recipe,
)
from tools.estuary_depth.prepare import build_bundle, verified_run
from tools.estuary_depth.render import camera_pose, motion_angles
from tools.estuary_studio.common import artifact, encoded, read, require, write

ROOT = Path(__file__).resolve().parents[2]
VERSION = "fine-fold-film-study-v1"
BUNDLE_RESOLUTION = (4096, 3072)
MESH_RESOLUTION = (1536, 1152)
STAGE_TIMEOUT = 3 * 3600
# A stage wrapper first stops its own detached Blender/encoder with an 8-second
# grace. Give that cleanup time to finish before killing the wrapper itself.
WRAPPER_STOP_GRACE = 20
PATTERN_STUDY_FAMILY = "pattern-studies-v1"


@dataclass(frozen=True)
class StudyCatalog:
    options: Mapping[str, Any]
    formation_recipe: Callable[..., dict]
    photo_recipe: Callable[..., dict]
    motion_recipe: Callable[..., dict]
    runtime_extensions: tuple[str, ...] = ()
    reference_option: str = "control"
    default_option: str = "three-broad-pools"


def study_catalog(study_family=None):
    """Select an explicit versioned family without changing the legacy default."""
    if study_family is None:
        return StudyCatalog(OPTIONS, make_formation_recipe, make_photo_recipe, make_motion_recipe)
    require(study_family == PATTERN_STUDY_FAMILY, "Unknown study family")
    from tools.estuary_depth import pattern_studies

    return StudyCatalog(
        pattern_studies.OPTIONS,
        pattern_studies.make_formation_recipe,
        pattern_studies.make_photo_recipe,
        pattern_studies.make_motion_recipe,
        pattern_studies.PAINT_RUNTIME_EXTENSIONS,
        pattern_studies.REFERENCE_OPTION,
        pattern_studies.DEFAULT_OPTION,
    )


def resolve_catalog(plan):
    if "study_family" in plan:
        require(type(plan["study_family"]) is str, "Study family must be explicit when present")
    catalog = study_catalog(plan.get("study_family"))
    require(
        plan.get("paint_runtime_extensions", []) == list(catalog.runtime_extensions),
        "Paint runtime extensions differ from the study family",
    )
    require(
        "study_family" in plan or "paint_runtime_extensions" not in plan,
        "Legacy plans do not declare runtime extensions",
    )
    return catalog


def _sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _fingerprint(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")


def _sources(cohort, kind, source_root=None):
    if kind == "generated":
        from tools.estuary_depth.filament_cohort import verify_cohort

        return verify_cohort(cohort, source_root=source_root)
    require(kind == "legacy", "Unknown source cohort")
    require(
        cohort == read(ROOT / "tools/estuary_confluence/recipes/ten-seeds.json"),
        "Legacy source cohort differs from the archived collection",
    )
    rows = {row["seed"]: row for row in cohort["sources"]}
    if source_root is not None:
        for seed, row in rows.items():
            require(
                artifact(Path(source_root) / f"{seed}.orbit")["sha256"] == row["sha256"],
                "Source recording differs from its cohort",
            )
    return rows


def _tool(tool):
    return {key: tool[key] for key in ("path", "sha256")}


def validate_plan(plan):
    """Validate the complete archived contract without original live source files."""
    require(plan.get("version") == VERSION, "Unknown film study plan")
    require(
        plan.get("identity_sha256")
        == _sha({k: v for k, v in plan.items() if k != "identity_sha256"}),
        "Film plan identity differs",
    )
    catalog = resolve_catalog(plan)
    options = catalog.options
    _runtime_contract(plan)
    _fingerprint(plan["runtime"]["estuary_depth"]["film.py"])
    rows = _sources(plan["cohort"], plan["cohort_kind"])
    require(
        plan["bundle_resolution"] == list(BUNDLE_RESOLUTION)
        and plan["mesh_resolution"] == list(MESH_RESOLUTION),
        "Film material preparation dimensions differ",
    )
    require(set(plan["tools"]) == {"blender", "ffmpeg", "ffprobe"}, "Incomplete film tools")
    for tool in plan["tools"].values():
        require(set(tool) == {"path", "sha256", "bytes"}, "Invalid tool record")
        require(Path(tool["path"]).is_absolute(), "Tool path must be absolute")
        require(type(tool["bytes"]) is int and tool["bytes"] > 0, "Invalid tool size")
        _fingerprint(tool["sha256"])
    cases, materials = plan["cases"], plan["materials"]
    require(
        type(cases) is list and 1 <= len(cases) <= len(rows) * len(options), "Invalid film cases"
    )
    require(type(materials) is dict and bool(materials), "Missing film materials")
    require(len({case["id"] for case in cases}) == len(cases), "Duplicate film cases")
    references = {}
    for case in cases:
        seed, option = case["seed"], case["option"]
        require(seed in rows and type(option) is str and option in options, "Unknown film case")
        material_id = f"{seed}-{options[option].material_variant}"
        require(
            case["id"] == f"{seed}-{option}" and case["material_id"] == material_id,
            "Film case identifier differs from its controls",
        )
        require(type(case["master"]) is bool, "Master flag must be boolean")
        require(
            case["photo_recipe"] == catalog.photo_recipe(seed, option, master=case["master"])
            and case["motion_recipe"] == catalog.motion_recipe(seed, option),
            "Film case recipe differs from its canonical controls",
        )
        reference = case["reference"]
        if reference is not None:
            require(
                type(reference) is dict
                and {"final_state_sha256", "master"} <= reference.keys()
                and not reference.keys()
                - {"final_state_sha256", "master", "prior_publication_sha256"}
                and reference["master"] is case["master"],
                "Invalid prior study reference",
            )
            _fingerprint(reference["final_state_sha256"])
            if "prior_publication_sha256" in reference:
                _fingerprint(reference["prior_publication_sha256"])
            require(
                references.setdefault(material_id, reference["final_state_sha256"])
                == reference["final_state_sha256"],
                "Shared material has conflicting reference states",
            )
    require(set(materials) == {case["material_id"] for case in cases}, "Unused or missing material")
    for key, material in materials.items():
        seed, variant = material["seed"], material["variant"]
        require(
            seed in rows
            and key == f"{seed}-{variant}"
            and variant in options
            and options[variant].material_variant == variant,
            "Unknown material study",
        )
        require(
            Path(material["source"]).is_absolute()
            and material["source_sha256"] == rows[seed]["sha256"],
            "Material source differs from its cohort",
        )
        require(
            material["recipe"] == catalog.formation_recipe(seed, variant),
            "Formation recipe differs",
        )
        require(
            material["expected_final_state_sha256"] == references.get(key),
            "Material reference differs from its cases",
        )
    return {case["id"]: case for case in cases}


def make_plan(
    cohort,
    *,
    source_root,
    blender,
    ffmpeg,
    ffprobe,
    cohort_kind="generated",
    seeds=None,
    options=None,
    selections=None,
    case_ids=None,
    references=None,
    study_family=None,
):
    """Freeze a cross-product or explicit case selection; references constrain backfills."""
    catalog = study_catalog(study_family)
    cohort = copy.deepcopy(cohort)
    rows = _sources(cohort, cohort_kind, source_root)
    references = copy.deepcopy(references or {})
    if case_ids is None and references and seeds is None and options is None and selections is None:
        case_ids = list(references)
    if case_ids is not None:
        require(
            selections is None and seeds is None and options is None, "Use one case selection mode"
        )
        selections = [case_id.split("-", 1) for case_id in case_ids]
    if selections is None:
        seeds = list(rows) if seeds is None else list(seeds)
        options = list(catalog.options) if options is None else list(options)
        selections = [(seed, option) for seed in seeds for option in options]
    else:
        require(seeds is None and options is None, "Use selections or seed/option lists")
    materials, cases = {}, []
    for seed, option in selections:
        require(seed in rows and option in catalog.options, "Unknown film selection")
        spec, case_id = catalog.options[option], f"{seed}-{option}"
        material_id = f"{seed}-{spec.material_variant}"
        reference = references.get(case_id)
        master = reference.get("master", False) if reference else False
        if reference is not None:
            reference.setdefault("master", False)
        material = materials.setdefault(
            material_id,
            {
                "seed": seed,
                "variant": spec.material_variant,
                "source": str(
                    (Path(source_root) / rows[seed].get("path", f"{seed}.orbit")).resolve(
                        strict=True
                    )
                ),
                "source_sha256": rows[seed]["sha256"],
                "recipe": catalog.formation_recipe(seed, spec.material_variant),
                "expected_final_state_sha256": None,
            },
        )
        if reference is not None:
            current = material["expected_final_state_sha256"]
            require(
                current in (None, reference["final_state_sha256"]), "Conflicting reference states"
            )
            material["expected_final_state_sha256"] = reference["final_state_sha256"]
        cases.append(
            {
                "id": case_id,
                "seed": seed,
                "option": option,
                "material_id": material_id,
                "master": master,
                "reference": reference,
                "photo_recipe": catalog.photo_recipe(seed, option, master=master),
                "motion_recipe": catalog.motion_recipe(seed, option),
            }
        )
    require(set(references) <= {case["id"] for case in cases}, "Unused prior study references")
    plan = {
        "version": VERSION,
        "cohort_kind": cohort_kind,
        "cohort": cohort,
        "runtime": runtime_identity(),
        "materials": materials,
        "cases": cases,
        "tools": {
            name: {"path": str(Path(path).resolve(strict=True)), **artifact(Path(path))}
            for name, path in (("blender", blender), ("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        },
        "bundle_resolution": list(BUNDLE_RESOLUTION),
        "mesh_resolution": list(MESH_RESOLUTION),
    }
    if study_family is not None:
        plan.update(
            study_family=study_family, paint_runtime_extensions=list(catalog.runtime_extensions)
        )
    plan["identity_sha256"] = _sha(plan)
    validate_plan(plan)
    return plan


def _paths(root, case):
    folder = root / "cases" / case["id"]
    material = root / "materials" / case["material_id"]
    return {
        "plan": root / "plan.json",
        "material": material,
        "paint": material / "paint",
        "bundle": material / "bundle",
        "photograph": folder / "photographs/00-painting",
        "motion": folder / "motion/00-painting",
        "film": folder / "film",
    }


def _verify_material(plan, material, folder):
    paint = folder / "paint"
    request, identity, recipe, _state, artifacts = verified_run(paint)
    require(completed(paint, identity) and request["mode"] == "film", "Incomplete formation film")
    require(
        recipe == material["recipe"]
        and request["source"]["seed"] == material["seed"]
        and request["source"]["sha256"] == material["source_sha256"],
        "Formation source or recipe differs",
    )
    require(
        request["tools"] == {name: _tool(plan["tools"][name]) for name in ("ffmpeg", "ffprobe")},
        "Formation encoder differs from the plan",
    )
    steps = frame_plan(recipe["simulation"]["steps"], recipe["render"]["frames"])
    require(
        request["frame_steps"] == steps
        and request["exposure_steps"] == exposure_plan(steps, recipe["render"]["temporal_samples"]),
        "Formation does not follow its complete canonical clock",
    )
    checked_artifact(paint, artifacts["initial.png"])
    final = artifacts["final-state.npy"]["sha256"]
    require(
        material["expected_final_state_sha256"] in (None, final), "Backfill final paint differs"
    )
    movie = read(checked_artifact(paint, artifacts["movie.json"]))
    require(
        movie["full_decode_verified"] is True
        and movie["artifact"] == artifacts["film.mp4"]
        and movie["frames"] == recipe["render"]["frames"]
        and movie["fps"] == FPS
        and movie["resolution"] == recipe["render"]["resolution"],
        "Formation movie controls differ",
    )
    bundle = read(folder / "bundle/manifest.json")
    require(
        bundle.get("complete") is True and bundle["identity_sha256"] == _sha(bundle["request"]),
        "Incomplete material bundle",
    )
    require(
        bundle["request"]["inputs"]["render_identity"] == identity
        and bundle["request"]["inputs"]["artifacts"]["final-state.npy"]
        == artifacts["final-state.npy"]
        and bundle["source"] == bundle["request"]["source"] == request["source"],
        "Prepared material differs from the formation",
    )
    _verify_paint_preparation(plan, request, artifacts, bundle)
    domain = recipe["simulation"]["domain_scale"]
    width, height = recipe["simulation"]["resolution"]
    aspect = width / height
    geometry = {
        "domain_scale": domain,
        "view_aspect": aspect,
        "coordinates": {
            "row_order": "bottom-to-top",
            "uv_origin": "bottom-left",
            "view_bounds": [[-aspect, -1], [aspect, 1]],
            "domain_bounds": [[-aspect * domain, -domain], [aspect * domain, domain]],
            "resampling": "exact pixel-area averages over the full guard domain",
        },
    }
    require(
        bundle["request"]["geometry"] == geometry
        and all(bundle[key] == value for key, value in geometry.items()),
        "Prepared geometry differs from the source canvas",
    )
    checked_artifact(folder / "bundle", bundle["bundle"])
    return request, identity, artifacts, bundle


def validate_camera_ledger(records, camera, frames):
    """Verify every declared pose without accessing live source or bundle files.

    The first matrix supplies only the fixed target's height. Its orientation,
    horizontal target, all subsequent poses and camera timing are independently
    reconstructed from the canonical camera controls. This does not independently
    remeasure the target's height from paint geometry.
    """
    require(type(frames) is int and frames >= 1, "Invalid camera frame count")
    require(type(records) is list and len(records) == frames, "Camera ledger length differs")
    target = None
    for index, pose in enumerate(records):
        expected = motion_angles(camera, index, frames)
        require(
            type(pose) is dict and type(pose.get("frame")) is int and pose["frame"] == index,
            "Camera frame order differs from its canonical path",
        )
        require(
            all(
                type(pose.get(key)) in (int, float)
                and math.isfinite(pose[key])
                # Covers only floating-point evaluation differences, not a
                # meaningful edit to the authored camera movement.
                and math.isclose(pose[key], angle, rel_tol=0, abs_tol=1e-10)
                for key, angle in zip(("tilt_degrees", "azimuth_degrees"), expected, strict=True)
            ),
            "Camera pose differs from its canonical path or paired photograph",
        )
        matrix = pose.get("matrix_world")
        require(
            type(matrix) is list
            and len(matrix) == 4
            and all(
                type(row) is list
                and len(row) == 4
                and all(type(value) in (int, float) and math.isfinite(value) for value in row)
                for row in matrix
            ),
            "Camera matrix must be finite 4 by 4 values",
        )
        if target is None:
            target = [*camera["target"], matrix[2][3] - 0.8 * math.cos(math.radians(expected[0]))]
        require(
            np.allclose(matrix, camera_pose(*expected, target), rtol=0, atol=1e-10),
            "Camera matrix differs from the canonical orientation or fixed target",
        )


def _verify_photograph(plan, case, folder, bundle, *, motion):
    experiment = read(folder.parent / "experiment-request.json")
    require(
        experiment["blender"] == _tool(plan["tools"]["blender"]), "Blender differs from the plan"
    )
    require(finished(folder, _sha(experiment)), "Photograph experiment is incomplete")
    expected_frames = MOTION_FRAMES if motion else 1
    request, receipt = read(folder / "request.json"), verify_render(folder)
    _verify_photo_runtime(plan, request)
    require(
        request["recipe"] == case["motion_recipe" if motion else "photo_recipe"]
        and request["bundle_sha256"] == bundle["bundle"]["sha256"]
        and request["bundle_manifest_sha256"] == _sha(bundle)
        and request["motion"]["frames"] == receipt["motion_frames"] == expected_frames
        and request["motion"]["fps"] == FPS
        and receipt["source_fraction"] == 1.0
        and receipt["source"] == bundle["source"]
        and receipt["history_fractions"] == [1.0],
        "Photograph controls or source differ",
    )
    if motion:
        require(
            experiment["encoders"]
            == {name: _tool(plan["tools"][name]) for name in ("ffmpeg", "ffprobe")},
            "Motion encoders differ from the plan",
        )
        movie = read(folder / "experiment-result.json")["movie"]
        require(
            movie
            and movie["full_decode_verified"] is True
            and movie["frames"] == MOTION_FRAMES
            and movie["fps"] == FPS
            and movie["resolution"] == case["motion_recipe"]["render"]["resolution"],
            "Incomplete motion movie",
        )
        require(
            case["motion_recipe"]["camera"]["orbit_end"]
            == [
                case["photo_recipe"]["camera"]["tilt_degrees"],
                case["photo_recipe"]["camera"]["azimuth_degrees"],
            ],
            "Motion does not finish at the paired photograph",
        )
    validate_camera_ledger(
        read(folder / "camera.json"),
        case["motion_recipe" if motion else "photo_recipe"]["camera"],
        expected_frames,
    )
    return receipt


def _inspect_case(folder, plan, case):
    paths = _paths(folder.parent.parent, case)
    material = plan["materials"][case["material_id"]]
    request, paint_id, artifacts, bundle = _verify_material(plan, material, paths["material"])
    photo = _verify_photograph(plan, case, paths["photograph"], bundle, motion=False)
    motion = _verify_photograph(plan, case, paths["motion"], bundle, motion=True)
    require(
        np.allclose(
            read(paths["motion"] / "camera.json")[-1]["matrix_world"],
            read(paths["photograph"] / "camera.json")[0]["matrix_world"],
            rtol=0,
            atol=1e-10,
        ),
        "Motion does not finish at the paired photograph matrix",
    )
    edited = film.verify_complete(paths["film"])
    source, inputs, timeline = film.validate_inputs(paths["paint"], paths["motion"])
    edit_request = read(paths["film"] / "request.json")
    require(
        edit_request["source"] == source
        and edit_request["inputs"] == inputs
        and edit_request["timeline"] == timeline
        and edit_request["editor_sha256"] == plan["runtime"]["estuary_depth"]["film.py"]
        and edit_request["tools"]
        == {name: _tool(plan["tools"][name]) for name in ("ffmpeg", "ffprobe")},
        "Edited film differs from the paired study",
    )
    record = {
        "version": VERSION,
        "complete": True,
        "id": case["id"],
        "seed": case["seed"],
        "option": case["option"],
        "material_variant": material["variant"],
        "material_id": case["material_id"],
        "master": case["master"],
        "source_sha256": request["source"]["sha256"],
        "final_state_sha256": artifacts["final-state.npy"]["sha256"],
        "plan_identity_sha256": plan["identity_sha256"],
        "paint_identity_sha256": paint_id,
        "photo_identity_sha256": photo["identity_sha256"],
        "motion_identity_sha256": motion["identity_sha256"],
        "film_identity_sha256": edited["identity_sha256"],
        "movie": edited["movie"],
        "timeline": timeline,
    }
    if "study_family" in plan:
        record["study_family"] = plan["study_family"]
    return record, {key: str(value.resolve()) for key, value in paths.items()}


def verify_case(folder):
    """Verify every source artifact before exposing a paired image and full film."""
    folder = Path(folder).resolve()
    plan = read(folder.parent.parent / "plan.json")
    cases = validate_plan(plan)
    require(folder.name in cases, "Case is absent from its film plan")
    record, paths = _inspect_case(folder, plan, cases[folder.name])
    require(read(folder / "study.json") == record, "Film study certification differs")
    return record, paths


def _check_tools(plan):
    for tool in plan["tools"].values():
        require(
            artifact(Path(tool["path"])) == {key: tool[key] for key in ("sha256", "bytes")},
            "Pinned executable changed",
        )


def _write_exact(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, "Archived stage input changed")
    else:
        write(path, value)


def _preserve_attempt(path):
    path.rename(path.with_name(f"{path.name}.incomplete-{time.time_ns()}"))


class _OwnedStage:
    """Serialize cancellation and the worker's error cleanup for one wrapper."""

    def __init__(self, child):
        self.child, self.lock, self.stopped = child, threading.Lock(), False

    def stop(self):
        with self.lock:
            if self.stopped:
                return
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=WRAPPER_STOP_GRACE)
            except subprocess.TimeoutExpired:
                pass
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
                self.stopped = True


class _Processes:
    def __init__(self):
        self.lock, self.children, self.cancelled = threading.Lock(), set(), False

    def run(self, command, log):
        with log.open("ab") as stream:
            with self.lock:
                require(not self.cancelled, "Film batch interrupted")
                stream.write(encoded([str(value) for value in command]))
                stream.flush()
                child = subprocess.Popen(
                    [str(value) for value in command],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                owned = _OwnedStage(child)
                self.children.add(owned)
            try:
                result = child.wait(timeout=STAGE_TIMEOUT)
                require(result == 0, f"Film stage exited with status {result}; inspect {log}")
            except BaseException:
                owned.stop()
                raise
            finally:
                with self.lock:
                    self.children.discard(owned)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            children = list(self.children)
        for owned in children:
            owned.stop()


def _ensure_material(root, plan, key, processes, progress):
    material = plan["materials"][key]
    folder = root / "materials" / key
    folder.mkdir(parents=True, exist_ok=True)
    _write_exact(folder / "recipe.json", material["recipe"])
    paint = folder / "paint"
    if paint.exists() and not (paint / "request.json").exists():
        _preserve_attempt(paint)
    if paint.exists():
        request = read(paint / "request.json")
        require(
            request["recipe"] == material["recipe"]
            and request["source"]["sha256"] == material["source_sha256"]
            and request["code"] == _runtime_contract(plan)["paint"],
            "Existing paint attempt belongs to different inputs",
        )
        if (
            not (paint / "receipt.json").exists()
            or read(paint / "receipt.json").get("complete") is not True
        ) and not (paint / "checkpoint.json").exists():
            _preserve_attempt(paint)
    progress("materials", key, "formation")
    command = [
        sys.executable,
        "-m",
        "tools.estuary.run",
        "--orbit",
        material["source"],
        "--recipe",
        folder / "recipe.json",
        "--output",
        paint,
        "--checkpoint-retention",
        "2",
        "--ffmpeg",
        plan["tools"]["ffmpeg"]["path"],
        "--ffprobe",
        plan["tools"]["ffprobe"]["path"],
    ]
    if paint.exists():
        command.append("--resume")
    processes.run(command, folder / "paint.log")
    progress("materials", key, "preparation")
    bundle = folder / "bundle"
    if bundle.exists() and (
        not (bundle / "manifest.json").exists()
        or read(bundle / "manifest.json").get("complete") is not True
    ):
        _preserve_attempt(bundle)
    build_bundle(
        paint,
        bundle,
        resolution=tuple(plan["bundle_resolution"]),
        mesh_resolution=tuple(plan["mesh_resolution"]),
        specific_volumes=SPECIFIC_VOLUMES,
    )
    _verify_material(plan, material, folder)
    progress("materials", key, "verified")


def _ensure_case(root, plan, case, processes, progress):
    folder = root / "cases" / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "study.json").exists():
        verify_case(folder)
        return
    paths = _paths(root, case)
    for stage, recipe_key, frames in (
        ("photographs", "photo_recipe", 1),
        ("motion", "motion_recipe", MOTION_FRAMES),
    ):
        progress("cases", case["id"], stage)
        recipes = folder / f"{stage}-recipes"
        _write_exact(recipes / "00-painting.json", case[recipe_key])
        target = folder / stage
        if target.exists() and not (target / "experiment-request.json").exists():
            _preserve_attempt(target)
        _check_tools(plan)
        processes.run(
            [
                sys.executable,
                "-m",
                "tools.estuary_depth.experiment",
                "--blender",
                plan["tools"]["blender"]["path"],
                "--render-script",
                ROOT / "tools/estuary_depth/render.py",
                "--bundle",
                paths["bundle"],
                "--recipes",
                recipes,
                "--output",
                target,
                "--workers",
                "1",
                "--motion-frames",
                str(frames),
                "--fps",
                str(FPS),
                "--ffmpeg",
                plan["tools"]["ffmpeg"]["path"],
                "--ffprobe",
                plan["tools"]["ffprobe"]["path"],
            ],
            folder / f"{stage}.log",
        )
        _check_tools(plan)
    progress("cases", case["id"], "editing")
    edited = paths["film"]
    if edited.exists() and (
        not (edited / "receipt.json").exists()
        or read(edited / "receipt.json").get("complete") is not True
    ):
        _preserve_attempt(edited)
    processes.run(
        [
            sys.executable,
            "-m",
            "tools.estuary_depth.film",
            "--formation-run",
            paths["paint"],
            "--orbit-case",
            paths["motion"],
            "--output",
            edited,
            "--ffmpeg",
            plan["tools"]["ffmpeg"]["path"],
            "--ffprobe",
            plan["tools"]["ffprobe"]["path"],
        ],
        folder / "film.log",
    )
    record, _ = _inspect_case(folder, plan, case)
    write(folder / "study.json", record)


def execute_plan(output, plan, *, workers=2, on_case_complete=None):
    """Resume owned stages; publish only fully verified cases through a serial callback."""
    require(type(workers) is int and 1 <= workers <= 2, "Use one or two GPU workers")
    validate_plan(plan)
    require(plan["runtime"] == runtime_identity(), "Code differs from the frozen film plan")
    _check_tools(plan)
    for material in plan["materials"].values():
        require(
            artifact(Path(material["source"]))["sha256"] == material["source_sha256"],
            "Source changed",
        )
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / "plan.json").exists():
            require(
                not any(p.name != ".batch.lock" for p in root.iterdir()), "Unowned output preserved"
            )
        _write_exact(root / "plan.json", plan)
        return _execute(root, plan, workers, on_case_complete)


def _execute(root, plan, workers, callback):
    started, processes, events = time.monotonic(), _Processes(), queue.Queue()
    mutex = threading.Lock()
    status = {
        "plan_identity_sha256": plan["identity_sha256"],
        "complete": False,
        "materials": {},
        "cases": {case["id"]: {"stage": "pending"} for case in plan["cases"]},
    }

    def progress(kind, key, stage, error=None):
        with mutex:
            value = {"stage": stage, "updated_unix": time.time()}
            if error is not None:
                value["error"] = str(error) or type(error).__name__
            status[kind][key] = value
            write(root / "progress.json", status)
            with (root / "progress.jsonl").open("a") as log:
                log.write(json.dumps({"kind": kind, "id": key, **value}) + "\n")

    # Case order is explicit in the plan. JSON object key sorting must not
    # silently replace an option-first study order with seed-first scheduling.
    groups = {}
    for case in plan["cases"]:
        groups.setdefault(case["material_id"], []).append(case)

    def render_case(case):
        try:
            _ensure_case(root, plan, case, processes, progress)
            progress("cases", case["id"], "verified")
            events.put(root / "cases" / case["id"])
        except Exception as error:
            progress("cases", case["id"], "failed", error)

    def group(key):
        cases = groups[key]
        try:
            _ensure_material(root, plan, key, processes, progress)
        except Exception as error:
            progress("materials", key, "failed", error)
            for case in cases:
                progress("cases", case["id"], "failed", error)
            return False
        render_case(cases[0])
        return True

    pool = ThreadPoolExecutor(max_workers=workers)
    primary = {pool.submit(group, key): key for key in groups}
    pending, ready = set(primary), set()
    secondary_submitted = False
    publication_errors = []
    try:
        while pending or not events.empty():
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                if future.result() and future in primary:
                    ready.add(primary[future])
            if not pending and not secondary_submitted:
                # Every material gets its first comparison film before one
                # Control material occupies a worker for four lighting films.
                pending = {
                    pool.submit(render_case, case)
                    for case in plan["cases"]
                    if case["material_id"] in ready and case is not groups[case["material_id"]][0]
                }
                secondary_submitted = True
            while not events.empty():
                folder = events.get_nowait()
                if callback is not None:
                    try:
                        callback(folder)
                    except Exception as error:
                        publication_errors.append({"case": folder.name, "error": str(error)})
    except BaseException:
        processes.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    require(runtime_identity() == plan["runtime"], "Code changed during film execution")
    _check_tools(plan)
    failed = {key: value for key, value in status["cases"].items() if value["stage"] != "verified"}
    status.update(
        complete=not failed and not publication_errors,
        seconds=time.monotonic() - started,
        publication_errors=publication_errors,
    )
    write(root / "progress.json", status)
    write(root / "results.json", status)
    require(status["complete"], "Some film stages failed; inspect preserved logs and progress")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    execute_plan(args.output, read(args.plan), workers=args.workers)


if __name__ == "__main__":

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Film batch interrupted; certified checkpoints remain available")

    signal.signal(signal.SIGTERM, interrupted)
    main()
