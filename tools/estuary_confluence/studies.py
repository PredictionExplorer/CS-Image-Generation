"""Bounded, reproducible color-count and silhouette experiment batches.

Every case uses a complete, hash-pinned recording. The plan freezes each recipe
before rendering, and only independently verified archives enter the result list.
Screening resolution is explicit; it is never presented as native 4K material.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.estuary_studio.common import digest, encoded, read, require, write

from .blend_study import PRESETS as APPEARANCES
from .run import ROOT, runtime_identity, validate_recipe, verify_run

COUNTS = (1, 2, 3, 5)
FLOW_STUDIES = {
    "original": ("Original currents", {}),
    "quiet": ("Quiet currents", {"pair_swirl": 0.15, "stir_radius": 0.18, "flow_strength": 1.3}),
    "fold": ("Tidal folds", {"pair_swirl": 0.15, "pair_strain": 0.7}),
    "open": ("Open currents", {"pair_swirl": 0.0, "pair_strain": 1.4, "stir_radius": 0.28}),
    "ribbon": (
        "Calligraphic currents",
        {
            "pair_swirl": 0.12,
            "pair_strain": 0.55,
            "stir_radius": 0.16,
            "flow_strength": 1.4,
            "load_radius": 0.18,
            "initial_load": 0.28,
        },
    ),
}


def study_recipe(count, flow, *, width=1024, film=False, appearance="ordered"):
    """Change explicit artistic variables; retain the canonical full-source clock."""
    require(type(count) is int and count in COUNTS, "Unknown starting pigment count")
    require(flow in FLOW_STUDIES, "Unknown flow study")
    require(type(width) is int and width in (1024, 2048, 4096), "Use a qualified material size")
    require(type(film) is bool, "film must be boolean")
    require(appearance in ("ordered", "thin-glaze", "ink"), "Unknown study appearance")
    label, overrides = FLOW_STUDIES[flow]
    recipe = copy.deepcopy(read(ROOT / "recipes/layered-five.json"))
    size = [width, width * 3 // 4]
    recipe.update(
        name=f"{count} pigment{'s' if count != 1 else ''} · {label}", chromatic_count=count
    )
    recipe["simulation"].update(resolution=size, **overrides)
    if appearance != "ordered":
        appearance_label, optical_controls = APPEARANCES[appearance]
        recipe["surface"].update(optical_controls)
        recipe["name"] += f" · {appearance_label}"
    recipe["simulation"]["initial_pigment_weights"] = recipe["simulation"][
        "initial_pigment_weights"
    ][:count]
    recipe["assessment"]["resolution"] = [512, 384]
    # Both optical interpretations share one physical history and exact palette.
    recipe["looks"] = ["layered", "homogeneous"] if count > 1 else ["layered"]
    recipe["render"].update(
        capture_resolution=size,
        resolution=[min(width, 1440), min(width, 1440) * 3 // 4],
        still_resolution=size if width < 4096 else [3840, 2880],
        formation_frames=721,
        hold_frames=72,
        orbit_frames=145,
        frame_supersampling=2 if film else 1,
    )
    return validate_recipe(recipe)


def make_plan(seeds, variants, *, width=1024, film=False, source_root=None, appearance="ordered"):
    cohort = read(ROOT / "recipes/ten-seeds.json")
    available = {source["seed"]: source for source in cohort["sources"]}
    require(bool(seeds) and len(set(seeds)) == len(seeds), "Choose distinct seeds")
    require(all(seed in available for seed in seeds), "Seed is outside the pinned cohort")
    require(bool(variants) and len(set(variants)) == len(variants), "Choose distinct variants")
    root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    for count, flow in variants:
        recipe = study_recipe(count, flow, width=width, film=film, appearance=appearance)
        for seed in seeds:
            cases.append(
                {
                    "id": f"{seed}-{count}-{flow}",
                    "seed": seed,
                    "source": str(root / f"{seed}.orbit"),
                    "source_sha256": available[seed]["sha256"],
                    "recipe": recipe,
                    "mode": "film" if film else "still",
                }
            )
    plan = {
        "schema_version": 1,
        "runtime": runtime_identity(),
        "cases": cases,
        "contract": (
            "Complete recordings; common pigment/pool prefixes; "
            "fewer pigments remove pools and mass, without renormalization. "
            "Optical views share material. Different flows re-plan source-aware starting locations."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def execute_plan(output, plan, *, workers=2):
    """Run at most two independent GPU jobs, keeping failed archives inspectable."""
    require(type(workers) is int and 1 <= workers <= 2, "Use one or two GPU workers")
    require(
        plan["runtime"] == runtime_identity(), "Runtime differs from the frozen experiment plan"
    )
    expected = hashlib.sha256(
        encoded({k: v for k, v in plan.items() if k != "identity_sha256"})
    ).hexdigest()
    require(plan["identity_sha256"] == expected, "Experiment plan identity differs")
    output = Path(output).resolve()
    require(not output.exists(), "Use a new experiment directory; existing results are immutable")
    for source, sha in {(case["source"], case["source_sha256"]) for case in plan["cases"]}:
        require(digest(Path(source)) == sha, "Recording differs from the pinned cohort")
    output.mkdir(parents=True)
    (output / "recipes").mkdir()
    (output / "logs").mkdir()
    write(output / "plan.json", plan)
    started = time.monotonic()

    def run(case):
        recipe = output / "recipes" / f"{case['id']}.json"
        write(recipe, case["recipe"])
        destination = output / "cases" / case["id"]
        command = [
            sys.executable,
            "-m",
            "tools.estuary_confluence.run",
            "--source",
            case["source"],
            "--recipe",
            str(recipe),
            "--output",
            str(destination),
        ]
        if case["mode"] == "still":
            command.append("--still-only")
        then = time.monotonic()
        with (output / "logs" / f"{case['id']}.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                cwd=ROOT.parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=7200,
            )
        request, receipt = verify_run(destination)
        expected_recipe = copy.deepcopy(case["recipe"])
        if "background" in expected_recipe:
            from .backgrounds import generate_background
            from .palette import generate_palette

            palette = generate_palette(
                case["seed"],
                expected_recipe["chromatic_count"],
                mode=expected_recipe["palette_mode"],
            )
            expected_recipe["surface"]["ground_srgb"] = generate_background(
                expected_recipe["background"], palette
            )["ground_srgb"]
        require(request["code"] == plan["runtime"], "Rendered code differs from the frozen plan")
        require(
            request["recipe"] == expected_recipe, "Rendered recipe differs from the frozen plan"
        )
        require(request["mode"] == case["mode"], "Rendered output mode differs")
        if case.get("accepted_base_material_sha256") is not None:
            require(
                receipt.get("base_material_sha256") == case["accepted_base_material_sha256"],
                "Interaction experiment changed the accepted base painting",
            )
        require(request["source"]["sha256"] == case["source_sha256"], "Rendered recording differs")
        require(
            receipt["source_fraction"] == 1 and receipt["complete"], "Incomplete source traversal"
        )
        return {
            "id": case["id"],
            "seed": case["seed"],
            "path": str(destination),
            "identity_sha256": receipt["identity_sha256"],
            "physical_state_sha256": receipt["physical_state_sha256"],
            "seconds": time.monotonic() - then,
        }

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run, case): case for case in plan["cases"]}
        for future in as_completed(pending):
            case = pending[future]
            try:
                result = future.result()
                results.append(result)
                event = {"status": "verified", **result}
            except Exception as error:
                event = {"status": "failed", "id": case["id"], "error": str(error)}
                failures.append(event)
            with (output / "progress.jsonl").open("a") as log:
                log.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
    report = {
        "plan_identity_sha256": plan["identity_sha256"],
        "complete": not failures,
        "cases": results,
        "failures": failures,
        "seconds": time.monotonic() - started,
    }
    write(output / "results.json", report)
    require(not failures, "Some studies failed; inspect logs and preserved archives")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--counts", nargs="+", type=int, default=list(COUNTS))
    parser.add_argument("--flows", nargs="+", choices=FLOW_STUDIES, default=["original"])
    parser.add_argument("--width", type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument("--film", action="store_true")
    parser.add_argument("--appearance", choices=("ordered", "thin-glaze", "ink"), default="ordered")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds or [s["seed"] for s in read(ROOT / "recipes/ten-seeds.json")["sources"]]
    plan = make_plan(
        seeds,
        [(count, flow) for flow in args.flows for count in args.counts],
        width=args.width,
        film=args.film,
        source_root=args.source_root,
        appearance=args.appearance,
    )
    if args.execute:
        execute_plan(args.output, plan)
    else:
        require(not args.output.exists(), "Plan destination already exists")
        write(args.output, plan)


if __name__ == "__main__":
    main()
