"""Render the RC1 ten-seed paintings with optional body-position diagnostics.

Every original recipe is reconstructed and checked against the release record
before adding markers. Full material identity, including interaction history,
must match RC1 after the complete trajectory is rendered again.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tools.estuary_studio.common import encoded, read, require

from .interaction_studies import texture_recipe
from .run import ROOT, runtime_identity, validate_recipe
from .studies import execute_plan

RELEASE = ROOT / "releases/contact-textures-v1.json"


def released_recipe(seed):
    release = read(RELEASE)
    matches = [case for case in release["cases"] if case["seed"] == seed]
    require(len(matches) == 1, "Choose a seed from the RC1 collection")
    case, selection = matches[0], release["selection"]
    recipe = texture_recipe(
        seed,
        case["chromatic_count"],
        selection["profile"],
        width=selection["material_resolution"][0],
        film=True,
        appearance=selection["appearance"],
        looks=selection["looks"],
    )
    require(
        hashlib.sha256(encoded(recipe)).hexdigest() == case["recipe_sha256"],
        "RC1 recipe has changed; use its pinned renderer to reconstruct this study",
    )
    return case, recipe


def make_plan(seeds=None, *, film=True, source_root=None):
    release = read(RELEASE)
    if seeds is None:
        seeds = [case["seed"] for case in release["cases"]]
    require(
        type(seeds) in (list, tuple) and seeds and len(seeds) == len(set(seeds)),
        "Choose distinct RC1 seeds",
    )
    require(type(film) is bool, "film must be boolean")
    cohort = read(ROOT / "recipes/ten-seeds.json")
    source_root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    for seed in seeds:
        accepted, recipe = released_recipe(seed)
        recipe["name"] = "3 pigments · Body positions"
        recipe["looks"] = ["silk-grain"]
        recipe["render"]["body_markers"] = True
        cases.append(
            {
                "id": f"{seed}-body-markers",
                "seed": seed,
                "source": str(source_root / f"{seed}.orbit"),
                "source_sha256": accepted["source_sha256"],
                "recipe": validate_recipe(recipe),
                "mode": "film" if film else "still",
                "accepted_base_material_sha256": accepted["base_material_sha256"],
                "accepted_physical_state_sha256": accepted["physical_state_sha256"],
            }
        )
    plan = {
        "schema_version": 1,
        "runtime": runtime_identity(),
        "cases": cases,
        "reference_tag": "RC1",
        "reference_release_sha256": hashlib.sha256(RELEASE.read_bytes()).hexdigest(),
        "contract": (
            "The complete RC1 source, paint and interaction history are unchanged. "
            "Red X markers show instantaneous body positions projected onto the canvas plane. "
            "Markers are annotations applied after shading; hold and camera frames freeze "
            "the final source positions. "
            "Every completed case must match the full RC1 material hash."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--still-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute_plan(args.output, make_plan(args.seeds, film=not args.still_only), workers=2)


if __name__ == "__main__":
    main()
