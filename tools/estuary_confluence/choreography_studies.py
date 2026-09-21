"""Matched RC1 studies of trajectory-informed starting paint.

Only the initial geometry and explicitly labeled layer allocations change.
Every study retains the full orbit, original palette, native resolution and
the amount of each pigment. Pilot measurements guide placement, not beauty.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from tools.estuary_studio.common import encoded, read, require

from .body_marker_studies import RELEASE, released_recipe
from .composition_studies import REFERENCES, references
from .run import ROOT, runtime_identity, validate_recipe
from .studies import execute_plan

VERSION = "choreography-study-v1"
DEFAULT_SEEDS = ("0xb7f327f9f722", "0x808861c25b6c", "0xceddf97909f39cc2")


@dataclass(frozen=True)
class StudySpec:
    label: str
    description: str
    setup: str
    mobility_bias: tuple[float, float, float] | None = None


VARIANTS = {
    "active-pools": StudySpec(
        "Active pools", "Three pools selected together for movement and contact.", "active-pools"
    ),
    "compact-pools": StudySpec(
        "Compact pools", "Smaller footprints with the same paint amounts.", "compact-pools"
    ),
    "broad-pools": StudySpec(
        "Broad pools", "Wider, shallower deposits with the same paint amounts.", "broad-pools"
    ),
    "unequal-pools": StudySpec(
        "Unequal pools",
        "A small, medium and broad pool create an unequal opening gesture.",
        "unequal-pools",
    ),
    "stretch-ovals": StudySpec(
        "Stretch ovals",
        "Elongated deposits oriented around the measured currents.",
        "stretch-ovals",
    ),
    "cross-strokes": StudySpec(
        "Crossing strokes", "Tapered strokes placed across active currents.", "cross-strokes"
    ),
    "facing-banks": StudySpec(
        "Facing banks",
        "Two elongated banks and an upstream deposit share an active corridor.",
        "facing-banks",
    ),
    "split-lobes": StudySpec(
        "Split lobes",
        "The dominant pigment starts in two substantial deposits, both checked for motion.",
        "split-lobes",
    ),
    "long-ribbons": StudySpec(
        "Long ribbons",
        "Longer tapered deposits span a wider part of the active currents.",
        "long-ribbons",
    ),
    "swept-crescents": StudySpec(
        "Swept crescents",
        "Open curved deposits introduce empty space inside each starting gesture.",
        "swept-crescents",
    ),
    "ovals-balanced-layers": StudySpec(
        "Ovals · balanced layers",
        "The same ovals; 12% of the first pigment shifts to the slower layer and 12% "
        "of the second to the faster layer. This changes starting layer allocation, not viscosity.",
        "stretch-ovals",
        (-0.12, 0.12, 0.0),
    ),
    "ovals-accent-layers": StudySpec(
        "Ovals · accent mobility",
        "The same ovals; 12% of the first pigment shifts to the slower layer and 12% "
        "of the third to the faster layer. This changes starting layer allocation, not viscosity.",
        "stretch-ovals",
        (-0.12, 0.0, 0.12),
    ),
}


def choreography_recipe(seed, variant):
    from .choreography import VERSION as layout_version

    require(type(variant) is str and variant in VARIANTS, "Unknown choreography study")
    rows = [row for row in references()["cases"] if row["seed"] == seed]
    require(len(rows) == 1, "Choose a seed from the RC1 collection")
    reference, spec = rows[0], VARIANTS[variant]
    accepted, recipe = released_recipe(seed)
    require(
        accepted["recipe_sha256"] == reference["rc1_recipe_sha256"]
        and accepted["source_sha256"] == reference["source_sha256"],
        "Choreography reference differs from RC1",
    )
    recipe["name"] = spec.label
    recipe["looks"] = ["silk-grain"]
    recipe["launch_assessment"] = {}
    recipe["simulation"].update(
        initial_pattern="choreographed",
        initial_pigment_weights=None,
        initial_choreography={
            "version": layout_version,
            "setup": spec.setup,
            "target_mass": reference["target_mass"],
            "reference_radii": reference["reference_radii"],
            **({"mobility_bias": list(spec.mobility_bias)} if spec.mobility_bias else {}),
        },
    )
    return reference, validate_recipe(recipe)


def identify_recipe(seed, recipe):
    """Match complete canonical controls, including resolved seeded ground color."""
    from .backgrounds import generate_background
    from .palette import generate_palette

    matches = []
    for variant in ("rc1", *VARIANTS):
        _, expected = (
            released_recipe(seed) if variant == "rc1" else choreography_recipe(seed, variant)
        )
        if "background" in expected:
            palette = generate_palette(
                seed, expected["chromatic_count"], mode=expected["palette_mode"]
            )
            expected["surface"]["ground_srgb"] = generate_background(
                expected["background"], palette
            )["ground_srgb"]
        if encoded(recipe) == encoded(expected):
            matches.append(variant)
    require(len(matches) == 1, "Archived recipe does not match an exact choreography study")
    return matches[0]


def make_plan(seeds=None, variants=None, *, film=False, source_root=None):
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    variants = tuple(VARIANTS) if variants is None else variants
    require(
        type(seeds) in (list, tuple)
        and bool(seeds)
        and all(type(seed) is str for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "Choose distinct RC1 seeds",
    )
    require(
        type(variants) in (list, tuple)
        and bool(variants)
        and all(type(v) is str and v in VARIANTS for v in variants)
        and len(set(variants)) == len(variants),
        "Choose distinct choreography studies",
    )
    require(type(film) is bool, "film must be boolean")
    cohort = read(ROOT / "recipes/ten-seeds.json")
    source_root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    for seed in seeds:
        for variant in variants:
            reference, recipe = choreography_recipe(seed, variant)
            cases.append(
                {
                    "id": f"{seed}-{variant}",
                    "seed": seed,
                    "source": str(source_root / f"{seed}.orbit"),
                    "source_sha256": reference["source_sha256"],
                    "recipe": recipe,
                    "mode": "film" if film else "still",
                    "reference_initial_mass": [*reference["target_mass"], 0.0],
                    "reference_palette_identity_sha256": reference["palette_identity_sha256"],
                    "study": {
                        "version": VERSION,
                        "variant": variant,
                        "label": VARIANTS[variant].label,
                    },
                }
            )
    plan = {
        "schema_version": 1,
        "version": VERSION,
        "runtime": runtime_identity(),
        "cases": cases,
        "reference_tag": "RC1",
        "reference_inputs_sha256": hashlib.sha256(REFERENCES.read_bytes()).hexdigest(),
        "reference_release_sha256": hashlib.sha256(RELEASE.read_bytes()).hexdigest(),
        "contract": (
            "Complete three-body trajectories, original RC1 palettes and per-pigment amounts. "
            "Native2048 paint and final images, original light, camera and film timeline. "
            "Only initial paint geometry and declared layer allocation change. The source-derived "
            "pilot checks each component; its predictions do not guarantee final participation "
            "or artistic merit. Layer allocation is not a viscosity model."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", nargs="+")
    seeds.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--film", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = [row["seed"] for row in references()["cases"]] if args.all_seeds else args.seeds
    execute_plan(
        args.output,
        make_plan(selected, args.variants, film=args.film, source_root=args.source_root),
        workers=2,
    )


if __name__ == "__main__":
    main()
