"""Frozen full-trajectory paint and Cycles photography studies of fine folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.estuary.run import RUNTIME_FILES, checked_artifact, completed
from tools.estuary_confluence.run import runtime_identity
from tools.estuary_depth.experiment import finished, verify_render
from tools.estuary_depth.filament_studies import VARIANTS, make_depth_recipe, make_paint_recipe
from tools.estuary_depth.prepare import build_bundle, verified_run
from tools.estuary_studio.common import artifact, encoded, read, require, write

ROOT = Path(__file__).resolve().parents[2]
VERSION = "fine-fold-study-v1"
SPECIFIC_VOLUMES = (0.25, 1.0, 0.5)


def _runtime_contract(plan):
    """Select renderer dependencies from the archived plan, never today's files."""
    runtime = plan.get("runtime")
    require(type(runtime) is dict, "Study plan lacks its frozen runtime")
    paint, depth = runtime.get("estuary"), runtime.get("estuary_depth")
    require(type(paint) is dict and type(depth) is dict, "Incomplete frozen study runtime")
    names = set(RUNTIME_FILES) | {
        name for name in paint if name.startswith("shaders/") and name.endswith(".glsl")
    }
    require(names <= paint.keys(), "Frozen paint runtime is incomplete")
    require(
        {"prepare.py", "render.py", "materials.py"} <= depth.keys(),
        "Frozen photograph runtime is incomplete",
    )
    hashes = [
        *(paint[name] for name in names),
        *(depth[name] for name in ("prepare.py", "render.py", "materials.py")),
    ]
    require(
        all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes),
        "Invalid frozen runtime fingerprint",
    )
    return {
        "paint": {name: paint[name] for name in sorted(names)},
        "prepare": depth["prepare.py"],
        "optics": paint["optics.py"],
        "photo": {name: depth[name] for name in ("render.py", "materials.py")},
    }


def _verify_paint_preparation(plan, request, artifacts, bundle):
    contract = _runtime_contract(plan)
    require(request.get("code") == contract["paint"], "Paint runtime differs from its frozen plan")
    for name, sha in contract["paint"].items():
        require(
            artifacts.get(f"inputs/code/{name}", {}).get("sha256") == sha,
            "Archived paint code differs from its frozen plan",
        )
    preparation = bundle["request"]
    require(
        preparation.get("prepare_sha256") == contract["prepare"]
        and preparation.get("optics_sha256") == contract["optics"],
        "Material preparation runtime differs from its frozen plan",
    )
    require(
        preparation["parameters"]
        == {
            "resolution": plan["bundle_resolution"],
            "mesh_resolution": plan["mesh_resolution"],
            "history_fractions": [],
            "specific_volumes": list(SPECIFIC_VOLUMES),
        },
        "Material preparation parameters differ from the study contract",
    )


def _verify_photo_runtime(plan, request):
    require(
        request.get("renderer") == _runtime_contract(plan)["photo"],
        "Photograph runtime differs from its frozen plan",
    )


def _validated_plan(plan):
    """Validate archived controls without requiring the original runtime or files."""
    require(
        plan.get("version") == VERSION and type(plan.get("proof")) is bool,
        "Unknown study plan",
    )
    require(
        plan.get("identity_sha256")
        == hashlib.sha256(
            encoded({k: v for k, v in plan.items() if k != "identity_sha256"})
        ).hexdigest(),
        "Plan identity differs",
    )
    _runtime_contract(plan)
    sources = {
        row["seed"]: row["sha256"]
        for row in read(ROOT / "tools/estuary_confluence/recipes/ten-seeds.json")["sources"]
    }
    cases = plan["cases"]
    require(
        type(cases) is list and 1 <= len(cases) <= len(sources) * len(VARIANTS), "Invalid cases"
    )
    require(len({case["id"] for case in cases}) == len(cases), "Duplicate study cases")
    for case in cases:
        seed, variant = case["seed"], case["variant"]
        require(seed in sources and variant in VARIANTS, "Unknown study seed or variant")
        require(case["id"] == f"{seed}-{variant}", "Case id differs from its controls")
        require(case["source_sha256"] == sources[seed], "Source differs from the archived cohort")
        require(
            case["paint_recipe"] == make_paint_recipe(seed, variant)
            and case["depth_recipe"]
            == make_depth_recipe(seed, VARIANTS[variant].label, proof=plan["proof"]),
            "Planned recipes differ from the declared study",
        )
    return {case["id"]: case for case in cases}


def _check_blender(plan):
    pinned = plan["blender"]
    require(
        artifact(Path(pinned["path"])) == {key: pinned[key] for key in ("sha256", "bytes")},
        "Blender binary differs from the frozen study",
    )


def verify_case(folder):
    """Bind the source, initial view, material bundle and photograph together."""
    folder = Path(folder)
    record = read(folder / "study.json")
    require(record.get("version") == VERSION and record.get("complete") is True, "Incomplete study")
    seed, variant, proof = record["seed"], record["variant"], record["proof"]
    require(variant in VARIANTS and type(proof) is bool, "Unknown study controls")
    plan = read(folder.parent.parent / "plan.json")
    cases = _validated_plan(plan)
    require(folder.name in cases, "Study is absent from its archived plan")
    case = cases[folder.name]
    require(
        record["plan_identity_sha256"] == plan["identity_sha256"]
        and (seed, variant, proof) == (case["seed"], case["variant"], plan["proof"])
        and record["source_sha256"] == case["source_sha256"],
        "Study controls or source differ from its archived plan",
    )
    paint = folder / "paint"
    request, identity, recipe, _state, artifacts = verified_run(paint)
    require(completed(paint, identity), "Paint archive is incomplete")
    require(request["source"]["seed"] == seed, "Study seed differs from the recording")
    require(
        request["source"]["sha256"] == record["source_sha256"], "Study source recording differs"
    )
    require(
        recipe == make_paint_recipe(seed, variant), "Study recipe differs from its declared variant"
    )
    require("initial.png" in artifacts, "Missing actual starting-paint image")
    checked_artifact(paint, artifacts["initial.png"])
    bundle = read(folder / "bundle/manifest.json")
    require(
        bundle.get("complete") is True
        and bundle["identity_sha256"] == hashlib.sha256(encoded(bundle["request"])).hexdigest(),
        "Incomplete or incorrectly identified material bundle",
    )
    require(
        bundle["request"]["inputs"]["render_identity"] == identity
        and bundle["request"]["inputs"]["artifacts"]["final-state.npy"]
        == artifacts["final-state.npy"]
        and bundle["source"] == request["source"],
        "Photograph bundle belongs to different paint",
    )
    _verify_paint_preparation(plan, request, artifacts, bundle)
    checked_artifact(folder / "bundle", bundle["bundle"])
    photo = folder / "photographs/00-painting"
    experiment = read(photo.parent / "experiment-request.json")
    require(
        experiment["blender"] == {key: plan["blender"][key] for key in ("path", "sha256")},
        "Photograph used a different Blender binary",
    )
    require(
        finished(photo, hashlib.sha256(encoded(experiment)).hexdigest()),
        "Photograph experiment is incomplete",
    )
    receipt = verify_render(photo)
    shot = read(photo / "request.json")
    _verify_photo_runtime(plan, shot)
    require(
        shot["recipe"] == make_depth_recipe(seed, VARIANTS[variant].label, proof=proof)
        and shot["bundle_sha256"] == bundle["bundle"]["sha256"]
        and shot["bundle_manifest_sha256"] == artifact(folder / "bundle/manifest.json")["sha256"],
        "Photograph settings or material identity differ",
    )
    require(
        record["paint_identity_sha256"] == identity
        and record["photo_identity_sha256"] == receipt["identity_sha256"],
        "Study identities differ",
    )
    return record, request, receipt


def make_plan(seeds, variants, *, source_root, blender, proof=True):
    require(type(proof) is bool, "proof must be boolean")
    require(seeds and len(set(seeds)) == len(seeds), "Choose distinct seeds")
    require(
        variants and len(set(variants)) == len(variants) and all(v in VARIANTS for v in variants),
        "Choose distinct study variants",
    )
    cohort = read(ROOT / "tools/estuary_confluence/recipes/ten-seeds.json")
    sources = {row["seed"]: row for row in cohort["sources"]}
    cases = []
    for seed in seeds:
        require(seed in sources, "Choose an archived trajectory seed")
        for variant in variants:
            source = Path(source_root).resolve() / f"{seed}.orbit"
            require(
                artifact(source)["sha256"] == sources[seed]["sha256"], "Source recording differs"
            )
            cases.append(
                {
                    "id": f"{seed}-{variant}",
                    "seed": seed,
                    "variant": variant,
                    "source": str(source),
                    "source_sha256": sources[seed]["sha256"],
                    "paint_recipe": make_paint_recipe(seed, variant),
                    "depth_recipe": make_depth_recipe(seed, VARIANTS[variant].label, proof=proof),
                }
            )
    plan = {
        "version": VERSION,
        "runtime": runtime_identity(),
        "proof": proof,
        "cases": cases,
        "blender": {"path": str(Path(blender).resolve(strict=True)), **artifact(Path(blender))},
        "bundle_resolution": [4096, 3072],
        "mesh_resolution": [1536, 1152],
        "contract": (
            "Complete source recording; pigment transport at each declared native grid "
            "(6144 x 4608 baseline, explicitly labeled 2048 x 1536 resolution probes); "
            "three pigments; "
            "independently varied initial geometry or current. Cycles photographs the actual "
            "final concentrations. Initial images show the actual step-zero pigment from above."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def execute_plan(output, plan, *, workers=2):
    require(type(workers) is int and 1 <= workers <= 2, "Use one or two GPU workers")
    _validated_plan(plan)
    require(plan["runtime"] == runtime_identity(), "Code differs from the frozen study")
    _check_blender(plan)
    for case in plan["cases"]:
        require(artifact(Path(case["source"]))["sha256"] == case["source_sha256"], "Source changed")
    output = Path(output).resolve()
    require(not output.exists(), "Use a new immutable batch directory")
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    started = time.monotonic()

    def run(case):
        folder = output / "cases" / case["id"]
        folder.mkdir(parents=True)
        write(folder / "paint-recipe.json", case["paint_recipe"])
        recipes = folder / "depth-recipes"
        recipes.mkdir()
        write(recipes / "00-painting.json", case["depth_recipe"])
        with (folder / "paint.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.estuary.run",
                    "--orbit",
                    case["source"],
                    "--recipe",
                    str(folder / "paint-recipe.json"),
                    "--output",
                    str(folder / "paint"),
                    "--still-only",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        bundle = build_bundle(
            folder / "paint",
            folder / "bundle",
            resolution=tuple(plan["bundle_resolution"]),
            mesh_resolution=tuple(plan["mesh_resolution"]),
            specific_volumes=SPECIFIC_VOLUMES,
        )
        require(bundle["source"]["sha256"] == case["source_sha256"], "Prepared source differs")
        _check_blender(plan)
        with (folder / "photograph.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.estuary_depth.experiment",
                    "--blender",
                    plan["blender"]["path"],
                    "--render-script",
                    str(ROOT / "tools/estuary_depth/render.py"),
                    "--bundle",
                    str(folder / "bundle"),
                    "--recipes",
                    str(recipes),
                    "--output",
                    str(folder / "photographs"),
                    "--workers",
                    "1",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        _check_blender(plan)
        paint_request = read(folder / "paint/request.json")
        photo_receipt = verify_render(folder / "photographs/00-painting")
        record = {
            "version": VERSION,
            "complete": True,
            "seed": case["seed"],
            "variant": case["variant"],
            "proof": plan["proof"],
            "source_sha256": case["source_sha256"],
            "paint_identity_sha256": hashlib.sha256(encoded(paint_request)).hexdigest(),
            "photo_identity_sha256": photo_receipt["identity_sha256"],
            "plan_identity_sha256": plan["identity_sha256"],
        }
        write(folder / "study.json", record)
        verify_case(folder)
        return {"id": case["id"], "path": str(folder), "status": "verified"}

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, case): case for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as error:
                result = {"id": case["id"], "status": "failed", "error": str(error)}
                failures.append(result)
            with (output / "progress.jsonl").open("a") as log:
                log.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
    require(runtime_identity() == plan["runtime"], "Code changed during study execution")
    report = {
        "complete": not failures,
        "cases": results,
        "failures": failures,
        "seconds": time.monotonic() - started,
        "plan_identity_sha256": plan["identity_sha256"],
    }
    write(output / "results.json", report)
    require(not failures, "Some studies failed; inspect the preserved logs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", default=["0xbc53af1cd380"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    execute_plan(
        args.output,
        make_plan(
            args.seeds,
            args.variants,
            source_root=args.source_root,
            blender=args.blender,
            proof=not args.final,
        ),
    )


if __name__ == "__main__":
    main()
