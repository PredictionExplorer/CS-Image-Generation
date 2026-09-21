"""Incremental, portable review of certified fine-fold photograph/film pairs.

Assets and entry records are immutable. The sole authoritative comparison.json
is replaced atomically after each entry has been copied and checked. In-progress
source cases never supply a film link or inflate the completed-pair count.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from tools.estuary.run import exposure_plan, frame_plan
from tools.estuary_confluence.review_page import Presentation
from tools.estuary_confluence.review_page import document as shared_document
from tools.estuary_studio.common import artifact, encoded, read, require, write
from tools.estuary_studio.gallery import _copy_verified

from . import film
from .filament_batch import _verify_paint_preparation, _verify_photo_runtime
from .filament_cohort import _validate_plan as validate_cohort_plan
from .filament_film_batch import VERSION as BATCH_VERSION
from .filament_film_batch import validate_camera_ledger, validate_plan, verify_case
from .filament_gallery import _bundle, _fingerprint, _image, _matches, _paint_artifacts, _relative
from .filament_motion import (
    FILM_RESOLUTION,
    FORMATION_FRAMES,
    FPS,
    MOTION_FRAMES,
    OPTIONS,
    make_formation_recipe,
    make_motion_recipe,
    make_photo_recipe,
)

VERSION = "filament-film-review-v1"
ENTRY_VERSION = "filament-film-entry-v1"
PREVIEW_VERSION = "filament-preview-v1"
PREVIEW_RESOLUTION = (640, 480)
PRESENTATION = Presentation(
    version=VERSION,
    eyebrow="Three bodies / Fine folds in motion",
    intro=(
        "New trajectories appear first, followed by earlier reference seeds. Each completed "
        "pair includes the finished photograph, the full paint formation, and a short camera "
        "view ending at the photograph's pose."
    ),
    legend=(
        "Only verified image and film pairs are shown. Pending work is listed in the progress "
        "count. Gold outlines mark subjective visual picks."
    ),
    default_variant="three-broad-pools",
    reference_variant="control",
    film_only_selection=True,
)
RECORD_KEYS = {
    "study",
    "paint_request",
    "paint_receipt",
    "paint_recipe",
    "paint_movie",
    "bundle",
    *{
        f"{kind}_{name}"
        for kind in ("photo", "motion")
        for name in (
            "request",
            "receipt",
            "recipe",
            "camera",
            "experiment",
            "result",
            "input_recipe",
        )
    },
    "film_request",
    "film_receipt",
    "film_probe",
    "film_command",
    "film_decode",
}


def _sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _description(path, root):
    return {"path": str(path.relative_to(root)), **artifact(path)}


def _checked(root, record, *, folder, suffix, hash_file=True):
    require(
        type(record) is dict and set(record) == {"path", "sha256", "bytes"},
        "Invalid portable artifact descriptor",
    )
    info, name = _fingerprint(record), _relative(record["path"])
    require(name == f"{folder}/{info['sha256']}{suffix}", "Content address differs from its hash")
    if hash_file:
        return _matches(root, name, info)
    path = (root / name).resolve(strict=True)
    require(
        path.is_relative_to(root.resolve())
        and path.is_file()
        and path.stat().st_size == info["bytes"],
        "Immutable media is missing or changed size",
    )
    return path


def _copy(root, source, *, folder="records", expected=None):
    source = Path(source)
    info = artifact(source) if expected is None else _fingerprint(expected)
    suffix = source.suffix.lower()
    require(suffix in {".json", ".txt", ".png", ".mp4"}, "Unsupported publication artifact")
    target = root / folder / (info["sha256"] + suffix)
    _copy_verified(source, target, info)
    return {"path": str(target.relative_to(root)), **info}


def _preview_pixels(source):
    """Version-one preview: display RGB8, full frame, Pillow Lanczos at 640x480."""
    with Image.open(source) as image:
        require(
            image.format == "PNG"
            and image.width * PREVIEW_RESOLUTION[1] == image.height * PREVIEW_RESOLUTION[0],
            "Preview source must be a 4:3 PNG",
        )
        return image.convert("RGB").resize(PREVIEW_RESOLUTION, Image.Resampling.LANCZOS)


def _make_preview(root, source):
    path = _checked(root, source, folder="assets", suffix=".png")
    image = _preview_pixels(path)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=9, optimize=False)
    payload = buffer.getvalue()
    info = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    target = root / "assets" / (info["sha256"] + ".png")
    if target.exists():
        require(artifact(target) == info, "Immutable preview asset changed")
    else:
        partial = target.with_suffix(".png.partial")
        partial.write_bytes(payload)
        partial.replace(target)
    return {
        "version": PREVIEW_VERSION,
        "source_image_sha256": source["sha256"],
        "resolution": list(PREVIEW_RESOLUTION),
        "artifact": {"path": str(target.relative_to(root)), **info},
    }


def _verify_preview(root, entry, *, compare_pixels):
    preview = entry["preview"]
    require(
        type(preview) is dict
        and set(preview) == {"version", "source_image_sha256", "resolution", "artifact"}
        and preview["version"] == PREVIEW_VERSION
        and preview["source_image_sha256"] == entry["media"]["image"]["sha256"]
        and preview["resolution"] == list(PREVIEW_RESOLUTION),
        "Preview version, dimensions or source image differ",
    )
    path = _checked(root, preview["artifact"], folder="assets", suffix=".png")
    with Image.open(path) as image:
        require(
            image.format == "PNG" and image.mode == "RGB" and image.size == PREVIEW_RESOLUTION,
            "Preview pixels have invalid dimensions or mode",
        )
        if compare_pixels:
            expected = _preview_pixels(root / entry["media"]["image"]["path"])
            require(
                np.array_equal(np.asarray(image), np.asarray(expected)),
                "Preview pixels differ from their source image",
            )
        else:
            image.verify()
    return preview["artifact"]["path"]


def _json(root, descriptor):
    return read(_checked(root, descriptor, folder="records", suffix=".json"))


def _equal_hash(root, descriptor, expected):
    require(
        _fingerprint(descriptor) == _fingerprint(expected), "Copied record differs from its receipt"
    )
    return root / descriptor["path"]


def _tools(plan, *names):
    return {name: {key: plan["tools"][name][key] for key in ("path", "sha256")} for name in names}


def document(title):
    """Use the shared viewer, adding progress and readable full-width seed labels."""
    page = shared_document(title, presentation=PRESENTATION)
    additions = """
      <p id="pair-progress" class="note" role="status" aria-live="polite"></p>
      <button id="refresh-pairs" type="button">Check for completed films</button>
"""
    anchor = '      <div id="content" hidden>'
    require(page.count(anchor) == 1, "Shared review content anchor changed")
    page = page.replace(anchor, additions + anchor)
    script = """
      const shortSeed = seed => seed.length > 24
        ? seed.slice(0, 10) + "…" + seed.slice(-8) : seed;
      function showPairProgress(result) {
        const p = result.progress;
        if (!p || !Number.isInteger(p.expected_pairs) || !Number.isInteger(p.ready_pairs))
          throw new Error("Invalid paired-film progress.");
        $("pair-progress").textContent = p.ready_pairs + " of " + p.expected_pairs +
          " image / film pairs ready · " + p.new_seed_count + " new seeds + " +
          p.reference_seed_count + " reference seeds" +
          (p.preparing_source_count ? " · New-source cohort is still being prepared" : "") +
          (p.awaiting_control_case_ids.length
            ? " · Some pairs await their Control comparison" : "");
        $("refresh-pairs").hidden = p.complete;
      }
      $("refresh-pairs").addEventListener("click", () => window.location.reload());
"""
    page = page.replace(
        '      fetch("comparison.json")',
        script + '      fetch("comparison.json", {cache: "no-store"})',
    )
    before = "        .then((result) => {\n          if ("
    after = """        .then((result) => {
          showPairProgress(result);
          if (result.version === presentation.version && Array.isArray(result.rows) &&
              result.rows.length === 0 && result.progress.visible_pairs === 0) {
            $("status").textContent = "No verified Control and treatment pairs are ready yet.";
            return;
          }
          if ("""
    require(page.count(before) == 1, "Shared review loading contract changed")
    page = page.replace(before, after)
    for variable in ("index", "i"):
        before = f'{variable} + 1 + " · " + seed'
        require(page.count(before) == 1, "Shared seed label contract changed")
        page = page.replace(before, f'{variable} + 1 + " · " + shortSeed(seed)')
    page = page.replace("option.value = seed;", "option.value = seed; option.title = seed;")
    page = page.replace("o.value = seed;", "o.value = seed; o.title = seed;")
    return page


def _catalog(root, plan_descriptors, pending_cohorts=()):
    plans, cases, order, seed_kind = {}, {}, [], {}
    for descriptor in plan_descriptors:
        plan = _json(root, descriptor)
        indexed = validate_plan(plan)
        identity = plan["identity_sha256"]
        require(identity not in plans, "Repeated batch plan")
        plans[identity] = (plan, descriptor)
        for case_id, case in indexed.items():
            require(case_id not in cases, "Case appears in more than one batch plan")
            cases[case_id] = (plan, case)
            seed = case["seed"]
            kind = plan["cohort_kind"]
            require(
                seed not in seed_kind or seed_kind[seed] == kind,
                "Seed has conflicting cohort roles",
            )
            if seed not in seed_kind:
                order.append(seed)
            seed_kind[seed] = kind
    require(cases, "Register at least one planned paired study")
    expected_ids = list(cases)
    prepared_cohorts = {
        plan["cohort"]["plan"]["identity_sha256"]
        for plan, _ in plans.values()
        if plan["cohort_kind"] == "generated"
    }
    preparing = set()
    cohort_ids = set()
    for descriptor in pending_cohorts:
        cohort = validate_cohort_plan(_json(root, descriptor))
        identity = cohort["identity_sha256"]
        require(identity not in cohort_ids, "Repeated pending cohort")
        cohort_ids.add(identity)
        for seed in cohort["seeds"]:
            require(
                seed not in seed_kind or seed_kind[seed] == "generated",
                "Pending fresh seed overlaps a reference seed",
            )
            if seed not in seed_kind:
                order.append(seed)
            seed_kind[seed] = "generated"
            if identity not in prepared_cohorts:
                preparing.add(seed)
            for option in OPTIONS:
                case_id = f"{seed}-{option}"
                if case_id not in expected_ids:
                    expected_ids.append(case_id)
    require(
        all(f"{seed}-control" in expected_ids for seed in order), "Every planned seed needs Control"
    )
    order.sort(key=lambda seed: seed_kind[seed] != "generated")
    option_order = list(OPTIONS)
    expected_ids.sort(
        key=lambda key: (
            order.index(key.split("-", 1)[0]),
            option_order.index(key.split("-", 1)[1]),
        )
    )
    return plans, cases, order, seed_kind, expected_ids, preparing


def _depth(root, descriptors, records, kind, plan, case, bundle):
    request, receipt, experiment, result = (
        records[f"{kind}_{key}"] for key in ("request", "receipt", "experiment", "result")
    )
    expected = case["photo_recipe" if kind == "photo" else "motion_recipe"]
    frames = 1 if kind == "photo" else MOTION_FRAMES
    require(
        receipt.get("complete") is True and receipt["identity_sha256"] == _sha(request),
        "Incomplete or mismatched depth receipt",
    )
    require(
        request["recipe"]
        == records[f"{kind}_recipe"]
        == records[f"{kind}_input_recipe"]
        == expected,
        "Depth recipe differs from the canonical paired study",
    )
    _verify_photo_runtime(plan, request)
    require(
        request["bundle_sha256"] == bundle["bundle"]["sha256"]
        and request["bundle_manifest_sha256"] == descriptors["bundle"]["sha256"]
        and receipt["source"] == bundle["source"]
        and receipt["source_fraction"] == 1.0
        and receipt["history_fractions"] == [1.0],
        "Depth source or completed paint differs",
    )
    require(
        type(request["motion"]["frames"]) is int
        and request["motion"]["frames"] == frames
        and type(receipt["motion_frames"]) is int
        and receipt["motion_frames"] == frames
        and request["motion"]["fps"] == FPS
        and request["motion"]["source_fraction"] == 1.0
        and request["motion"]["semantics"] == "frozen completed painting; camera only",
        "Depth motion controls differ",
    )
    artifacts = receipt["artifacts"]
    require(
        {"render.png", "recipe.json", "camera.json", "scene.blend", "render.exr"}
        <= artifacts.keys(),
        "Depth archive lacks required provenance",
    )
    for name, info in artifacts.items():
        _relative(name)
        _fingerprint(info)
    for name in ("recipe", "camera"):
        _equal_hash(root, descriptors[f"{kind}_{name}"], artifacts[name + ".json"])
    require(
        result.get("complete") is True
        and result["case"] == "00-painting"
        and result["identity_sha256"] == _sha(experiment)
        and result["render_receipt_sha256"] == descriptors[f"{kind}_receipt"]["sha256"],
        "Depth experiment completion differs",
    )
    require(
        experiment["blender"] == _tools(plan, "blender")["blender"]
        and experiment["motion_frames"] == frames
        and experiment["fps"] == FPS,
        "Depth experiment tools or cadence differ",
    )
    files = experiment["files"]
    require(
        files["bundle/bundle.npz"]["sha256"] == request["bundle_sha256"]
        and files["bundle/manifest.json"]["sha256"] == request["bundle_manifest_sha256"]
        and files["recipes/00-painting.json"]["sha256"]
        == descriptors[f"{kind}_input_recipe"]["sha256"]
        and request["renderer"]
        == {name: files[name]["sha256"] for name in ("render.py", "materials.py")},
        "Depth experiment input binding differs",
    )
    camera = records[f"{kind}_camera"]
    validate_camera_ledger(camera, expected["camera"], frames)
    if kind == "motion":
        require(
            experiment["encoders"] == _tools(plan, "ffmpeg", "ffprobe"), "Motion encoders differ"
        )
        require(
            {name for name in artifacts if name.startswith("frames/")}
            == {f"frames/{i:06d}.png" for i in range(MOTION_FRAMES)},
            "Motion frame ledger is incomplete",
        )
        movie = result["movie"]
        require(
            movie
            and movie["full_decode_verified"] is True
            and movie["frames"] == MOTION_FRAMES
            and movie["fps"] == FPS
            and movie["resolution"] == list(FILM_RESOLUTION)
            and movie["duration_seconds"] == MOTION_FRAMES / FPS,
            "Motion movie differs",
        )
        require(
            [camera[-1][key] for key in ("tilt_degrees", "azimuth_degrees")]
            == [case["photo_recipe"]["camera"][key] for key in ("tilt_degrees", "azimuth_degrees")],
            "Motion does not end at the paired photograph",
        )
    else:
        require(result.get("movie") is None, "Photograph case must remain a single still")
    return request, receipt, result


def _verify_entry(root, descriptor, catalog, *, hash_media=True):
    entry_path = _checked(root, descriptor, folder="entries", suffix=".json")
    entry = read(entry_path)
    require(
        set(entry)
        in (
            {"version", "id", "plan", "records", "media"},
            {"version", "id", "plan", "records", "media", "preview"},
        )
        and entry["version"] == ENTRY_VERSION
        and entry["id"] in catalog,
        "Unknown paired entry",
    )
    require(
        set(entry["records"]) == RECORD_KEYS
        and set(entry["media"]) == {"initial", "image", "film"},
        "Incomplete paired entry",
    )
    plan, case = catalog[entry["id"]]
    require(_json(root, entry["plan"]) == plan, "Entry uses a different registered plan")
    records = {
        name: _json(root, value)
        for name, value in entry["records"].items()
        if name != "film_decode"
    }
    decode_path = _checked(root, entry["records"]["film_decode"], folder="records", suffix=".txt")
    study, request, receipt, bundle = (
        records[key] for key in ("study", "paint_request", "paint_receipt", "bundle")
    )
    material = plan["materials"][case["material_id"]]
    identity, source = _sha(request), request["source"]
    recipe = make_formation_recipe(case["seed"], case["option"])
    require(
        request["recipe"] == records["paint_recipe"] == material["recipe"] == recipe,
        "Formation recipe differs from its canonical option",
    )
    require(
        request["mode"] == receipt["mode"] == "film"
        and receipt.get("complete") is True
        and receipt["identity_sha256"] == identity
        and receipt["source"] == source
        and source["seed"] == case["seed"]
        and source["sha256"] == material["source_sha256"],
        "Formation certification differs",
    )
    require(
        type(source["samples"]) is int
        and source["samples"] > 1
        and type(source["source_first_step"]) is int
        and source["source_first_step"] == 0
        and type(source["source_last_step"]) is int
        and source["source_last_step"] == source["samples"] - 1
        and receipt["final_step"] == recipe["simulation"]["steps"]
        and receipt["source_fraction"] == 1.0,
        "Formation omits part of the source",
    )
    frames = frame_plan(recipe["simulation"]["steps"], FORMATION_FRAMES)
    require(
        request["frame_steps"] == frames and request["exposure_steps"] == exposure_plan(frames, 4),
        "Formation frame schedule differs",
    )
    require(request["tools"] == _tools(plan, "ffmpeg", "ffprobe"), "Formation encoders differ")
    artifacts = _paint_artifacts(receipt)
    require({"film.mp4", "movie.json"} <= artifacts.keys(), "Formation movie is missing")
    for name, key in (("paint_recipe", "recipe.json"), ("paint_movie", "movie.json")):
        _equal_hash(root, entry["records"][name], artifacts[key])
    require(
        artifacts["inputs/source.orbit"]["sha256"] == source["sha256"], "Source artifact differs"
    )
    movie = records["paint_movie"]
    require(
        movie["full_decode_verified"] is True
        and movie["artifact"] == artifacts["film.mp4"]
        and movie["frames"] == FORMATION_FRAMES
        and movie["fps"] == FPS
        and movie["resolution"] == recipe["render"]["resolution"]
        and movie["duration_seconds"] == FORMATION_FRAMES / FPS,
        "Formation movie receipt differs",
    )
    inputs = _bundle(bundle)
    require(
        bundle["source"] == source
        and inputs["render_identity"] == identity
        and inputs["request_sha256"] == entry["records"]["paint_request"]["sha256"]
        and inputs["receipt_sha256"] == entry["records"]["paint_receipt"]["sha256"],
        "Material bundle belongs to different formation",
    )
    for name in ("final-state.npy", "linear.npy", "recipe.json", "inputs/source.orbit"):
        require(inputs["artifacts"][name] == artifacts[name], "Prepared input artifact differs")
    _verify_paint_preparation(plan, request, artifacts, bundle)
    final_hash = artifacts["final-state.npy"]["sha256"]
    require(
        material["expected_final_state_sha256"] in (None, final_hash),
        "Backfill final state differs",
    )
    require(
        case["photo_recipe"]
        == make_photo_recipe(case["seed"], case["option"], master=case["master"])
        and case["motion_recipe"] == make_motion_recipe(case["seed"], case["option"]),
        "Pair recipes differ",
    )
    photo_request, photo, _ = _depth(root, entry["records"], records, "photo", plan, case, bundle)
    motion_request, motion, motion_result = _depth(
        root, entry["records"], records, "motion", plan, case, bundle
    )
    require(
        np.allclose(
            records["motion_camera"][-1]["matrix_world"],
            records["photo_camera"][0]["matrix_world"],
            rtol=0,
            atol=1e-12,
        ),
        "Final film camera matrix differs from the paired photograph",
    )
    edit, edited = records["film_request"], records["film_receipt"]
    expected_source = {
        "seed": case["seed"],
        "sha256": source["sha256"],
        "render_identity": identity,
        "final_state_sha256": final_hash,
    }
    expected_timeline = film.timeline(FORMATION_FRAMES, FPS, MOTION_FRAMES, FPS)
    require(
        edited.get("complete") is True
        and edited["identity_sha256"] == _sha(edit)
        and edit["source"] == edited["source"] == expected_source
        and edit["timeline"] == expected_timeline
        and edit["tools"] == _tools(plan, "ffmpeg", "ffprobe")
        and edit["editor_sha256"] == plan["runtime"]["estuary_depth"]["film.py"],
        "Edited film source, timeline or runtime differs",
    )
    edit_artifacts = edited["artifacts"]
    require(
        {
            "film.mp4",
            "request.json",
            "command.json",
            "probe.json",
            "decode.txt",
            "inputs/formation.mp4",
            "inputs/orbit.mp4",
        }
        <= edit_artifacts.keys(),
        "Edited film provenance is incomplete",
    )
    for name, value in edit_artifacts.items():
        _relative(name)
        _fingerprint(value)
    for name in ("request", "command", "probe", "decode"):
        _equal_hash(
            root,
            entry["records"]["film_" + name],
            edit_artifacts[name + (".txt" if name == "decode" else ".json")],
        )
    for kind, record_name, film_artifact, count in (
        ("formation", "paint", artifacts["film.mp4"], FORMATION_FRAMES),
        ("orbit", "motion", motion_result["movie"], MOTION_FRAMES),
    ):
        supplied = edit["inputs"][kind]
        require(
            _fingerprint(supplied)
            == _fingerprint(film_artifact)
            == _fingerprint(edit_artifacts[f"inputs/{kind}.mp4"])
            and supplied["frames"] == count
            and supplied["fps"] == FPS
            and supplied["request_sha256"] == entry["records"][record_name + "_request"]["sha256"]
            and supplied["receipt_sha256"] == entry["records"][record_name + "_receipt"]["sha256"],
            "Film edit inputs differ from certified formation/motion",
        )
    require(
        edit["inputs"]["orbit"]["experiment_identity"] == _sha(records["motion_experiment"]),
        "Film orbit experiment differs",
    )
    output_movie = edited["movie"]
    require(
        output_movie["path"] == "film.mp4"
        and output_movie["full_decode_verified"] is True
        and output_movie["frames"] == expected_timeline["output_frames"]
        and output_movie["fps"] == FPS
        and output_movie["resolution"] == list(FILM_RESOLUTION)
        and output_movie["duration_seconds"] == float(Fraction(expected_timeline["duration"]))
        and _fingerprint(output_movie) == _fingerprint(edit_artifacts["film.mp4"]),
        "Edited movie metadata differs",
    )
    film.verify_timing(records["film_probe"], output_movie["frames"], FPS, list(FILM_RESOLUTION))
    progress = decode_path.read_text()
    counts = [int(line[6:]) for line in progress.splitlines() if line.startswith("frame=")]
    require(
        counts and counts[-1] == output_movie["frames"] and "progress=end" in progress,
        "Movie decode proof is incomplete",
    )
    command = records["film_command"]
    require(
        type(command) is list
        and command
        and Path(command[-1]).name == "film.partial.mp4"
        and command
        == film.command(
            plan["tools"]["ffmpeg"]["path"], Path(command[-1]).parent, expected_timeline
        ),
        "Film edit command differs from its timeline",
    )
    expected_study = {
        "version": BATCH_VERSION,
        "complete": True,
        "id": case["id"],
        "seed": case["seed"],
        "option": case["option"],
        "material_variant": material["variant"],
        "material_id": case["material_id"],
        "master": case["master"],
        "source_sha256": source["sha256"],
        "final_state_sha256": final_hash,
        "plan_identity_sha256": plan["identity_sha256"],
        "paint_identity_sha256": identity,
        "photo_identity_sha256": photo["identity_sha256"],
        "motion_identity_sha256": motion["identity_sha256"],
        "film_identity_sha256": edited["identity_sha256"],
        "movie": output_movie,
        "timeline": expected_timeline,
    }
    require(study == expected_study, "Paired study record differs from its copied certificates")
    for key, expected in (
        ("initial", artifacts["initial.png"]),
        ("image", photo["artifacts"]["render.png"]),
        ("film", output_movie),
    ):
        media = entry["media"][key]
        require(_fingerprint(media) == _fingerprint(expected), "Paired media hash differs")
        _checked(
            root,
            media,
            folder="assets",
            suffix=".mp4" if key == "film" else ".png",
            hash_file=hash_media or key != "film",
        )
    _image(
        root,
        entry["media"]["initial"]["path"],
        artifacts["initial.png"],
        recipe["render"]["resolution"],
    )
    _image(
        root,
        entry["media"]["image"]["path"],
        photo["artifacts"]["render.png"],
        case["photo_recipe"]["render"]["resolution"],
    )
    spec = OPTIONS[case["option"]]
    row = {
        "seed": case["seed"],
        "variant": case["option"],
        "label": spec.label,
        "description": spec.description,
        "features": {},
        **{key: value["path"] for key, value in entry["media"].items()},
        "preview": entry["media"]["image"]["path"],
        "resolution": case["photo_recipe"]["render"]["resolution"],
        "initial_resolution": recipe["render"]["resolution"],
        "film_resolution": list(FILM_RESOLUTION),
        "film_fps": FPS,
        "film_frames": output_movie["frames"],
        "source_sha256": source["sha256"],
        "physical_state_sha256": final_hash,
        "request": entry["records"]["photo_request"]["path"],
        "settings": {
            "seed": case["seed"],
            "paint": recipe,
            "photograph": photo_request["recipe"],
            "camera_film": motion_request["recipe"],
            "timeline": expected_timeline,
            "prior_reference": case["reference"],
            "note": (
                "Full source formation precedes the camera view; the final camera pose matches "
                "the photograph, while image/video sampling differs."
            ),
        },
    }
    if "preview" in entry:
        row["preview"] = _verify_preview(root, entry, compare_pixels=hash_media)
        row["preview_resolution"] = list(PREVIEW_RESOLUTION)
    return entry, row


def _manifest(root, provenance, title, picks, *, hash_media):
    require(
        set(provenance) == {"plans", "entries", "pending_cohorts"}, "Unknown publication provenance"
    )
    plans, catalog, seed_order, seed_kind, expected_ids, preparing = _catalog(
        root, provenance["plans"], provenance["pending_cohorts"]
    )
    rows, indexed = {}, {}
    for descriptor in provenance["entries"]:
        entry, row = _verify_entry(root, descriptor, catalog, hash_media=hash_media)
        require(entry["id"] not in rows, "Repeated paired entry")
        require(
            entry["plan"] == plans[catalog[entry["id"]][0]["identity_sha256"]][1],
            "Entry plan descriptor differs",
        )
        rows[entry["id"]], indexed[entry["id"]] = row, entry
    controls = {row["seed"] for row in rows.values() if row["variant"] == "control"}
    visible = [row for row in rows.values() if row["seed"] in controls]
    option_order = list(OPTIONS)
    visible.sort(
        key=lambda row: (seed_order.index(row["seed"]), option_order.index(row["variant"]))
    )
    require(type(picks) is list, "Visual picks must be a list")
    seen = set()
    for pick in picks:
        require(
            type(pick) is dict
            and set(pick) == {"seed", "variant", "note"}
            and type(pick["seed"]) is str
            and type(pick["variant"]) is str,
            "Invalid visual pick",
        )
        key = f"{pick['seed']}-{pick['variant']}"
        require(
            key in rows
            and rows[key]["seed"] in controls
            and key not in seen
            and type(pick["note"]) is str
            and 0 < len(pick["note"].strip()) <= 700,
            "Visual pick is not a displayed verified pair",
        )
        seen.add(key)
    progress = {
        "expected_pairs": len(expected_ids),
        "registered_pairs": len(catalog),
        "ready_pairs": len(rows),
        "visible_pairs": len(visible),
        "complete": len(rows) == len(expected_ids),
        "pending_case_ids": [key for key in expected_ids if key not in rows],
        "preparing_source_count": len(preparing),
        "awaiting_control_case_ids": [
            key for key, row in rows.items() if row["seed"] not in controls
        ],
        "new_seed_count": sum(seed_kind[s] == "generated" for s in seed_order),
        "reference_seed_count": sum(seed_kind[s] == "legacy" for s in seed_order),
    }
    data = {
        "version": VERSION,
        "title": title,
        "seeds": [s for s in seed_order if s in controls],
        "variants": [v for v in option_order if any(r["variant"] == v for r in visible)],
        "rows": visible,
        "picks": picks,
        "progress": progress,
        "provenance": provenance,
    }
    data["identity_sha256"] = _sha(data)
    return data, indexed


def verify_review(output):
    root = Path(output).resolve()
    actual = read(root / "comparison.json")
    require(actual.get("version") == VERSION, "Unknown paired-film review")
    expected, _ = _manifest(
        root, actual["provenance"], actual["title"], actual["picks"], hash_media=True
    )
    require(actual == expected, "Review rows, progress or identity differ from certified entries")
    require((root / "index.html").read_text() == document(actual["title"]), "Paired viewer differs")
    return actual


def _case_records(folder, context):
    paint, bundle, photo, motion, edit = (
        Path(context[key]) for key in ("paint", "bundle", "photograph", "motion", "film")
    )
    result = {
        "study": folder / "study.json",
        "bundle": bundle / "manifest.json",
        **{
            f"paint_{name}": paint / (name + ".json")
            for name in ("request", "receipt", "recipe", "movie")
        },
        **{
            f"film_{name}": edit / (name + (".txt" if name == "decode" else ".json"))
            for name in ("request", "receipt", "probe", "command", "decode")
        },
    }
    for kind, path in (("photo", photo), ("motion", motion)):
        result.update(
            {
                f"{kind}_{name}": path / (name + ".json")
                for name in ("request", "receipt", "recipe", "camera")
            }
        )
        result.update(
            {
                f"{kind}_experiment": path.parent / "experiment-request.json",
                f"{kind}_result": path / "experiment-result.json",
                f"{kind}_input_recipe": path.parent / "inputs/recipes/00-painting.json",
            }
        )
    return result


def publish_review(
    output,
    batch_roots,
    *,
    picks=None,
    pending_cohorts=None,
    title="The Estuary · Fine folds in motion",
):
    root = Path(output).resolve()
    batches = [Path(path).resolve(strict=True) for path in batch_roots]
    require(batches and len(set(batches)) == len(batches), "Register distinct source batches")
    require(
        all(not root.is_relative_to(path) for path in batches), "Publish outside source archives"
    )
    page = document(title)
    owner = {"version": VERSION, "title": title}
    if root.exists():
        require(
            (root / ".review-owner.json").is_file() and read(root / ".review-owner.json") == owner,
            "Refuse to overwrite an unrelated output",
        )
    else:
        root.mkdir(parents=True)
        write(root / ".review-owner.json", owner)
    with (root / ".publication.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(root / "comparison.json") if (root / "comparison.json").is_file() else None
        entries, indexed = [], {}
        if previous is not None:
            require(
                previous.get("version") == VERSION and previous["title"] == title,
                "Existing output belongs to a different review",
            )
            expected, indexed = _manifest(
                root, previous["provenance"], title, previous["picks"], hash_media=False
            )
            require(
                previous == expected and (root / "index.html").read_text() == page,
                "Existing review changed",
            )
            entries = list(previous["provenance"]["entries"])
        plan_descriptors = []
        batch_plans = []
        for batch in batches:
            plan = read(batch / "plan.json")
            cases = validate_plan(plan)
            plan_descriptors.append(_copy(root, batch / "plan.json"))
            batch_plans.append((batch, plan, cases, plan_descriptors[-1]))
        if previous is not None:
            require(
                all(old in plan_descriptors for old in previous["provenance"]["plans"]),
                "Previously registered batch plans cannot change or disappear",
            )
        pending = (
            list(previous["provenance"]["pending_cohorts"])
            if pending_cohorts is None and previous
            else [_copy(root, Path(path).resolve(strict=True)) for path in (pending_cohorts or [])]
        )
        if previous is not None:
            require(
                all(old in pending for old in previous["provenance"]["pending_cohorts"]),
                "Pending cohort plans cannot change or disappear",
            )
        _, catalog, _, _, _, _ = _catalog(root, plan_descriptors, pending)
        for batch, plan, cases, plan_descriptor in batch_plans:
            for case_id in cases:
                folder = batch / "cases" / case_id
                certificate = folder / "study.json"
                if case_id in indexed:
                    if certificate.exists():
                        require(
                            artifact(certificate)
                            == _fingerprint(indexed[case_id]["records"]["study"]),
                            "Previously certified source case changed",
                        )
                    continue
                if not certificate.is_file() or read(certificate).get("complete") is not True:
                    continue
                study, context = verify_case(folder)
                require(
                    study["id"] == case_id and read(Path(context["plan"])) == plan,
                    "Certified case differs from its registered batch",
                )
                records = {
                    name: _copy(root, source)
                    for name, source in _case_records(folder, context).items()
                }
                paint_receipt = read(Path(context["paint"]) / "receipt.json")
                paint_artifacts = _paint_artifacts(paint_receipt)
                photo_receipt = read(Path(context["photograph"]) / "receipt.json")
                entry = {
                    "version": ENTRY_VERSION,
                    "id": case_id,
                    "plan": plan_descriptor,
                    "records": records,
                    "media": {
                        "initial": _copy(
                            root,
                            Path(context["paint"]) / "initial.png",
                            folder="assets",
                            expected=paint_artifacts["initial.png"],
                        ),
                        "image": _copy(
                            root,
                            Path(context["photograph"]) / "render.png",
                            folder="assets",
                            expected=photo_receipt["artifacts"]["render.png"],
                        ),
                        "film": _copy(
                            root,
                            Path(context["film"]) / "film.mp4",
                            folder="assets",
                            expected=study["movie"],
                        ),
                    },
                }
                entry["preview"] = _make_preview(root, entry["media"]["image"])
                target = root / "entries" / (_sha(entry) + ".json")
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    require(read(target) == entry, "Immutable entry changed")
                else:
                    write(target, entry)
                descriptor = _description(target, root)
                _verify_entry(root, descriptor, catalog)
                entries.append(descriptor)
        data, _ = _manifest(
            root,
            {"plans": plan_descriptors, "entries": entries, "pending_cohorts": pending},
            title,
            previous["picks"] if picks is None and previous else ([] if picks is None else picks),
            hash_media=False,
        )
        if (root / "index.html").exists():
            require((root / "index.html").read_text() == page, "Paired viewer changed")
        else:
            partial = root / "index.html.partial"
            partial.write_text(page)
            partial.replace(root / "index.html")
        # All referenced immutable entries/media exist before this single commit.
        write(root / "comparison.json", data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=Path, nargs="+")
    parser.add_argument("--picks", type=Path)
    parser.add_argument("--pending-cohort", type=Path, action="append")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        data = verify_review(args.output)
    else:
        require(args.batches, "Supply registered paired-film batches")
        data = publish_review(
            args.output,
            args.batches,
            picks=read(args.picks) if args.picks else None,
            pending_cohorts=args.pending_cohort,
        )
    print(
        json.dumps(
            {
                "verified": True,
                **{
                    key: value
                    for key, value in data["progress"].items()
                    if not isinstance(value, list)
                },
            }
        )
    )


if __name__ == "__main__":
    main()
