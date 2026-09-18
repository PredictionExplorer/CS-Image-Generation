#!/usr/bin/env python3
"""Combine verified stills and camera films without changing their render archives.

The catalog's ordered studies select completed cases from any experiment. A
separate motion case must describe the same seed and original Estuary render.
Only presentation assets are copied; their hash-named paths remain immutable so
an updated catalog can be published without replacing a previously served image.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary_depth.experiment import checked, digest, encoded, finished, preserve, read, write
from tools.estuary_depth.gallery import DOCUMENT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_file(value):
    require(isinstance(value, str) and Path(value).is_absolute(), "Catalog paths must be absolute")
    return Path(value).resolve(strict=True)


def verified_case(value):
    folder = absolute_file(value)
    require(folder.is_dir(), "Study must name a completed case directory")
    experiment = read(folder.parent / "experiment-request.json")
    identity = hashlib.sha256(encoded(experiment)).hexdigest()
    require(finished(folder, identity), f"Study is incomplete: {folder}")
    request = read(folder / "request.json")
    receipt = read(folder / "receipt.json")
    result = read(folder / "experiment-result.json")
    manifest = read(folder.parent / "inputs/bundle/manifest.json")
    require(
        manifest.get("complete") is True
        and manifest.get("identity_sha256")
        == hashlib.sha256(encoded(manifest["request"])).hexdigest(),
        "Bundle manifest is incomplete or inconsistent",
    )
    require(
        receipt.get("source") == manifest["source"] and receipt.get("source_fraction") == 1.0,
        "Study does not certify the completed bundle source",
    )
    seed = manifest["source"]["seed"]
    painting = manifest["request"]["inputs"]["render_identity"]
    require(
        isinstance(seed, str) and bool(seed) and isinstance(painting, str) and bool(painting),
        "Missing source identity",
    )
    return {
        "folder": folder,
        "request": request,
        "receipt": receipt,
        "result": result,
        "seed": seed,
        "render_identity": painting,
        "experiment_identity": identity,
    }


def verified_motion(case):
    movie = case["result"].get("movie")
    motion = case["request"].get("motion", {})
    require(
        isinstance(movie, dict) and movie.get("full_decode_verified") is True,
        "Motion case lacks a fully decoded movie receipt",
    )
    frames, fps = movie.get("frames"), movie.get("fps")
    require(
        type(frames) is int and frames > 1 and type(fps) is int and 1 <= fps <= 60,
        "Motion frame count or cadence is invalid",
    )
    require(
        motion.get("frames") == case["receipt"].get("motion_frames") == frames
        and motion.get("fps") == fps
        and motion.get("source_fraction") == 1.0
        and movie.get("resolution") == case["request"]["recipe"]["render"]["resolution"],
        "Movie and renderer frame receipts disagree",
    )
    require(
        motion.get("semantics") == "frozen completed painting; camera only",
        "Only a camera orbit of the frozen painting can be paired here",
    )
    names = {name for name in case["receipt"]["artifacts"] if name.startswith("frames/")}
    require(
        names == {f"frames/{index:06d}.png" for index in range(frames)},
        "Motion archive has missing or unexpected certified frames",
    )
    return case["folder"] / "film.mp4"


def verified_formation(value, still):
    from tools.estuary_depth.film import verify_complete

    folder = absolute_file(value)
    require(folder.is_dir(), "Formation film must name a completed film directory")
    receipt = verify_complete(folder)
    source = receipt["source"]
    require(
        (source["seed"], source["render_identity"], source["sha256"])
        == (still["seed"], still["render_identity"], still["receipt"]["source"]["sha256"]),
        "Formation film must use the same completed painting and source as the still",
    )
    return {"folder": folder, "receipt": receipt}


def media(source, folder, prefix, expected=None):
    if expected is not None:
        checked(source, expected)
    sha = digest(source)
    target = folder / f"{prefix}-{sha[:16]}{source.suffix.lower()}"
    preserve(target, source)
    if expected is not None:
        checked(target, expected)
    return target, {"sha256": sha, "bytes": target.stat().st_size}


def curate(catalog_path, output):
    catalog = read(Path(catalog_path))
    require(
        isinstance(catalog, dict) and not catalog.keys() - {"studies", "baselines"},
        "Catalog must contain studies and optional baselines",
    )
    selected, references = catalog.get("studies"), catalog.get("baselines", {})
    require(
        isinstance(selected, list) and 1 <= len(selected) <= 128,
        "Select one through 128 completed studies",
    )
    require(
        isinstance(references, dict) and all(isinstance(seed, str) for seed in references),
        "Baselines must map source seeds to image paths",
    )
    prepared = []
    for entry in selected:
        require(
            isinstance(entry, dict)
            and set(entry) <= {"still", "motion", "formation", "name"}
            and "still" in entry,
            "Each study requires a still and optional motion/formation/name",
        )
        still = verified_case(entry["still"])
        motion = verified_case(entry["motion"]) if entry.get("motion") else None
        if motion is not None:
            require(
                (motion["seed"], motion["render_identity"])
                == (still["seed"], still["render_identity"]),
                "Still and motion must use the same seed and original completed painting",
            )
            verified_motion(motion)
        label = entry.get("name", still["request"]["recipe"]["name"])
        require(
            isinstance(label, str) and 0 < len(label.strip()) <= 160,
            "Study name must be a short nonempty string",
        )
        formation = (
            verified_formation(entry["formation"], still) if entry.get("formation") else None
        )
        if formation is not None and motion is not None:
            edit_request = read(formation["folder"] / "request.json")
            expected_orbit = {key: motion["result"]["movie"][key] for key in ("sha256", "bytes")}
            require(
                {key: edit_request["inputs"]["orbit"][key] for key in expected_orbit}
                == formation["receipt"]["artifacts"]["inputs/orbit.mp4"]
                == expected_orbit,
                "Formation edit must contain the exact paired camera orbit",
            )
        prepared.append((still, motion, formation, label))
    baseline_paths = {seed: absolute_file(value) for seed, value in references.items()}
    require(
        all(path.is_file() for path in baseline_paths.values()), "Baseline must be an image file"
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".curation.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assets, baselines, studies, provenance = {}, {}, [], []
        if baseline_paths:
            (output / "baselines").mkdir(exist_ok=True)
        for seed, source in baseline_paths.items():
            target, record = media(source, output / "baselines", "painting")
            baselines[seed] = str(target.relative_to(output))
            assets[baselines[seed]] = record
        for index, (still, motion, formation, label) in enumerate(prepared, 1):
            identifier = f"study-{index:03d}"
            folder = output / identifier
            folder.mkdir(exist_ok=True)
            image, record = media(
                still["folder"] / "render.png",
                folder,
                "render",
                still["receipt"]["artifacts"]["render.png"],
            )
            image_name = str(image.relative_to(output))
            assets[image_name] = record
            movie_name = None
            if motion is not None:
                movie, record = media(
                    motion["folder"] / "film.mp4", folder, "orbit", motion["result"]["movie"]
                )
                movie_name = str(movie.relative_to(output))
                assets[movie_name] = record
            formation_name = None
            if formation is not None:
                movie, record = media(
                    formation["folder"] / "film.mp4",
                    folder,
                    "formation",
                    formation["receipt"]["movie"],
                )
                formation_name = str(movie.relative_to(output))
                assets[formation_name] = record
            studies.append(
                {
                    "id": identifier,
                    "name": label,
                    "seed": still["seed"],
                    "group": still["request"]["recipe"]["family"],
                    "image": image_name,
                    "film": movie_name,
                    "formation": formation_name,
                    "baseline": baselines.get(still["seed"]),
                    "resolution": still["request"]["recipe"]["render"]["resolution"],
                    "render_identity": still["render_identity"],
                }
            )
            provenance.append(
                {
                    "still": str(still["folder"]),
                    "motion": None if motion is None else str(motion["folder"]),
                    "formation": None if formation is None else str(formation["folder"]),
                    "formation_receipt_sha256": None
                    if formation is None
                    else digest(formation["folder"] / "receipt.json"),
                    "source_render_identity": still["render_identity"],
                    "still_experiment_identity": still["experiment_identity"],
                    "still_result_sha256": digest(still["folder"] / "experiment-result.json"),
                    "motion_result_sha256": None
                    if motion is None
                    else digest(motion["folder"] / "experiment-result.json"),
                }
            )
        # Copy validation is complete before switching the user-facing catalog.
        write(
            output / "curation.json",
            {
                "schema_version": 1,
                "complete": True,
                "catalog": catalog,
                "provenance": provenance,
                "assets": assets,
            },
        )
        write(
            output / "collection.json",
            {
                "schema_version": 1,
                "studies": studies,
                "history": "Each study uses its own verified completed painting",
            },
        )
        temporary = output / "index.html.partial"
        temporary.write_text(DOCUMENT)
        temporary.replace(output / "index.html")
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(curate(args.catalog, args.output))
