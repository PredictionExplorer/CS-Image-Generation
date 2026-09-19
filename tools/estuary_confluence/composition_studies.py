"""Matched full-trajectory studies of six seeded starting-paint compositions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tools.estuary_studio.common import encoded, read, require

from .body_marker_studies import RELEASE, released_recipe
from .run import ROOT, runtime_identity, validate_recipe
from .studies import execute_plan

REFERENCES = ROOT / "releases/composition-rc1-inputs-v1.json"
SETUPS = {
    "random-circles": "Random circles",
    "random-ribbons": "Tapered ribbons",
    "random-crescents": "Open crescents",
    "facing-shores": "Facing shores",
    "scattered-commas": "Scattered commas",
    "body-wedges": "Body-centered wedges",
}


def references():
    record = read(REFERENCES)
    require(
        record["reference_release_sha256"] == hashlib.sha256(RELEASE.read_bytes()).hexdigest(),
        "Composition material references differ from RC1",
    )
    return record


def composition_recipe(seed, setup, *, body_markers=False):
    from .initial_composition import VERSION

    require(setup in SETUPS, "Unknown initial composition")
    require(type(body_markers) is bool, "body_markers must be boolean")
    rows = [row for row in references()["cases"] if row["seed"] == seed]
    require(len(rows) == 1, "Choose a seed from the RC1 collection")
    reference = rows[0]
    accepted, recipe = released_recipe(seed)
    require(
        accepted["recipe_sha256"] == reference["rc1_recipe_sha256"]
        and accepted["source_sha256"] == reference["source_sha256"],
        "Composition reference recipe or trajectory differs",
    )
    recipe["name"] = SETUPS[setup]
    recipe["looks"] = ["silk-grain"]
    recipe["simulation"].update(
        initial_pattern="shaped",
        initial_pigment_weights=None,
        initial_composition={
            "version": VERSION,
            "setup": setup,
            "target_mass": reference["target_mass"],
            "reference_radii": reference["reference_radii"],
        },
    )
    if body_markers:
        recipe["render"]["body_markers"] = True
    return reference, validate_recipe(recipe)


def make_plan(seeds=None, setups=None, *, film=True, source_root=None, body_markers=False):
    inputs = references()
    if seeds is None:
        seeds = [row["seed"] for row in inputs["cases"]]
    if setups is None:
        setups = list(SETUPS)
    require(
        type(seeds) in (list, tuple) and seeds and len(seeds) == len(set(seeds)),
        "Choose distinct RC1 seeds",
    )
    require(
        type(setups) in (list, tuple)
        and setups
        and len(setups) == len(set(setups))
        and all(setup in SETUPS for setup in setups),
        "Choose distinct composition setups",
    )
    require(type(film) is bool, "film must be boolean")
    cohort = read(ROOT / "recipes/ten-seeds.json")
    source_root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    # Seed-major order yields a complete six-way comparison early in a batch.
    for seed in seeds:
        for setup in setups:
            reference, recipe = composition_recipe(seed, setup, body_markers=body_markers)
            cases.append(
                {
                    "id": f"{seed}-{setup}",
                    "seed": seed,
                    "source": str(source_root / f"{seed}.orbit"),
                    "source_sha256": reference["source_sha256"],
                    "recipe": recipe,
                    "mode": "film" if film else "still",
                    "reference_initial_mass": [*reference["target_mass"], 0.0],
                }
            )
    plan = {
        "schema_version": 1,
        "runtime": runtime_identity(),
        "cases": cases,
        "reference_tag": "RC1",
        "reference_inputs_sha256": hashlib.sha256(REFERENCES.read_bytes()).hexdigest(),
        "contract": (
            "Same source, palette, per-pigment initial amounts, flow, material evolution, "
            "lighting, camera and complete film timeline as RC1. Only starting paint geometry "
            "and placement change. New material hashes are expected. Every initial and final "
            "pigment budget must match the pinned RC1 amounts within float32 tolerance. "
            "Only three actual pigments enter the new layouts; no future-trajectory pilot."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--setups", nargs="+", choices=SETUPS)
    parser.add_argument("--still-only", action="store_true")
    parser.add_argument("--body-markers", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute_plan(
        args.output,
        make_plan(
            args.seeds,
            args.setups,
            film=not args.still_only,
            body_markers=args.body_markers,
        ),
        workers=2,
    )


if __name__ == "__main__":
    main()
