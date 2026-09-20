"""Matched RC1 studies with each individual body and each pair stirring paint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tools.estuary_studio.common import encoded, read, require

from .body_marker_studies import RELEASE, released_recipe
from .run import ROOT, runtime_identity, validate_recipe
from .studies import execute_plan

INPUTS = ROOT / "releases/composition-rc1-inputs-v1.json"
VARIANTS = {
    "body-1": (0,),
    "body-2": (1,),
    "body-3": (2,),
    "bodies-1-2": (0, 1),
    "bodies-1-3": (0, 2),
    "bodies-2-3": (1, 2),
    "all-bodies": (0, 1, 2),
}
LABELS = {
    "body-1": "Body 1 only",
    "body-2": "Body 2 only",
    "body-3": "Body 3 only",
    "bodies-1-2": "Bodies 1 + 2",
    "bodies-1-3": "Bodies 1 + 3",
    "bodies-2-3": "Bodies 2 + 3",
    "all-bodies": "All 3 bodies",
}
REDUCED_VARIANTS = tuple(key for key in VARIANTS if key != "all-bodies")


def references():
    """Reuse the pinned original RC1 amounts, not a later study's paint loading."""
    inputs = read(INPUTS)
    require(
        inputs["reference_tag"] == "RC1"
        and inputs["reference_release_sha256"] == hashlib.sha256(RELEASE.read_bytes()).hexdigest(),
        "Body-influence material references differ from RC1",
    )
    return inputs


def influence_recipe(seed, variant):
    require(variant in VARIANTS, "Unknown body-influence selection")
    rows = [row for row in references()["cases"] if row["seed"] == seed]
    require(len(rows) == 1, "Choose a seed from the RC1 collection")
    reference = rows[0]
    accepted, recipe = released_recipe(seed)
    require(
        accepted["recipe_sha256"] == reference["rc1_recipe_sha256"]
        and accepted["source_sha256"] == reference["source_sha256"],
        "Body-influence reference recipe or trajectory differs",
    )
    recipe["name"] = LABELS[variant]
    recipe["looks"] = ["silk-grain"]
    recipe["simulation"]["body_influence"] = {
        "version": "body-influence-v1",
        "bodies": list(VARIANTS[variant]),
    }
    return accepted, reference, validate_recipe(recipe)


def make_plan(seeds=None, variants=None, *, film=True, source_root=None):
    inputs = references()
    if seeds is None:
        seeds = [row["seed"] for row in inputs["cases"]]
    if variants is None:
        variants = list(REDUCED_VARIANTS)
    require(
        type(seeds) in (list, tuple) and seeds and len(seeds) == len(set(seeds)),
        "Choose distinct RC1 seeds",
    )
    require(
        type(variants) in (list, tuple)
        and variants
        and len(variants) == len(set(variants))
        and all(variant in VARIANTS for variant in variants),
        "Choose distinct body-influence selections",
    )
    require(type(film) is bool, "film must be boolean")
    cohort = read(ROOT / "recipes/ten-seeds.json")
    source_root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    for seed in seeds:
        for variant in variants:
            accepted, reference, recipe = influence_recipe(seed, variant)
            case = {
                "id": f"{seed}-{variant}",
                "seed": seed,
                "source": str(source_root / f"{seed}.orbit"),
                "source_sha256": reference["source_sha256"],
                "recipe": recipe,
                "mode": "film" if film else "still",
                "reference_initial_mass": [*reference["target_mass"], 0.0],
                "reference_layout_artifact": reference["layout_artifact"],
                "reference_palette_identity_sha256": reference["palette_identity_sha256"],
            }
            if variant == "all-bodies":
                case["accepted_base_material_sha256"] = accepted["base_material_sha256"]
                case["accepted_physical_state_sha256"] = accepted["physical_state_sha256"]
            cases.append(case)
    plan = {
        "schema_version": 1,
        "runtime": runtime_identity(),
        "cases": cases,
        "reference_tag": "RC1",
        "reference_inputs_sha256": hashlib.sha256(INPUTS.read_bytes()).hexdigest(),
        "contract": (
            "All three recorded gravitational trajectories and the RC1 initial paint, palette, "
            "amounts, surface, camera and complete film timeline stay fixed. Only the selected "
            "bodies stir, deposit or rewet paint; pairs require both bodies. Eligible encounter "
            "pairs are selected before event ranking and suppression. Active strength is not "
            "renormalized, and adaptive travel bounds use only active bodies. All-three controls "
            "must reproduce the original complete RC1 material history exactly."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--still-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute_plan(
        args.output,
        make_plan(
            args.seeds,
            args.variants,
            film=not args.still_only,
            source_root=args.source_root,
        ),
        workers=2,
    )


if __name__ == "__main__":
    main()
