"""Matched all-three-body RC1 paint studies with a complete four-factor matrix.

The default study renders full-trajectory stills for three contrasting seeds.
All cases retain the same starting paint. Optical changes preserve physical
history; transported traits may change interaction history, and resistance may
change paint motion. The recorded gravitational trajectories remain fixed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from tools.estuary_studio.common import encoded, read, require

from .body_influence_studies import INPUTS, references
from .body_marker_studies import RELEASE, released_recipe
from .run import ROOT, runtime_identity, validate_recipe
from .studies import execute_plan

VERSION = "paint-material-study-v1"
DEFAULT_SEEDS = ("0xb7f327f9f722", "0x808861c25b6c", "0xceddf97909f39cc2")
FEATURES = ("fuller", "relief", "traits", "resistance")
FEATURE_LABELS = {
    "fuller": "Fuller paint",
    "relief": "Directional relief",
    "traits": "Seeded material",
    "resistance": "Paint resistance",
}
FEATURE_DESCRIPTIONS = {
    "fuller": "10% more displayed paint height; material loading stays fixed.",
    "relief": "Directional contact relief redistributes existing displayed thickness.",
    "traits": "Seeded properties vary wet-contact response and optional structural resistance.",
    "resistance": "Transported structure changes how paint responds to the original currents.",
}


@dataclass(frozen=True)
class StudySpec:
    label: str
    description: str
    features: tuple[str, ...] = ()
    height_scale: float | None = None
    relief_strength: float | None = None
    trait_amplitude: float | None = None
    resistance_strength: float | None = None
    roughness_bias: float | None = None


def _factorial_specs():
    result = {}
    for count in range(len(FEATURES) + 1):
        for selected in combinations(FEATURES, count):
            name = "-".join(selected) or "rc1"
            result[name] = StudySpec(
                label=" + ".join(FEATURE_LABELS[feature] for feature in selected)
                or "RC1 reference",
                description=" ".join(FEATURE_DESCRIPTIONS[feature] for feature in selected)
                or "The preserved all-three-body RC1 recipe, without new material controls.",
                features=selected,
                height_scale=66.0 if "fuller" in selected else None,
                relief_strength=0.6 if "relief" in selected else None,
                trait_amplitude=0.25 if "traits" in selected else None,
                resistance_strength=3.0 if "resistance" in selected else None,
            )
    return result


FACTORIAL = _factorial_specs()
STRENGTHS = {
    "height-63": StudySpec(
        "Gentler paint height",
        "5% more displayed paint height; material loading stays fixed.",
        ("fuller",),
        height_scale=63.0,
    ),
    "relief-mild": StudySpec(
        "Gentler contact relief",
        "Directional relief strength 0.3, with the same contact history.",
        ("relief",),
        relief_strength=0.3,
    ),
    "traits-subtle": StudySpec(
        "Subtle seeded material",
        "Material variation amplitude 0.10; identical seeded property maps.",
        ("traits",),
        trait_amplitude=0.10,
    ),
    "resistance-light": StudySpec(
        "Lighter resistance",
        "Structural paint resistance strength 1; unchanged source orbits.",
        ("resistance",),
        resistance_strength=1.0,
    ),
    "resistance-strong": StudySpec(
        "Stronger resistance",
        "Structural paint resistance strength 6; unchanged source orbits.",
        ("resistance",),
        resistance_strength=6.0,
    ),
    "finish-matte": StudySpec(
        "Softer matte finish",
        "Surface roughness bias +0.04; unchanged material and color coefficients.",
        roughness_bias=0.04,
    ),
    "finish-satin": StudySpec(
        "Clearer satin finish",
        "Surface roughness bias -0.04; unchanged material and color coefficients.",
        roughness_bias=-0.04,
    ),
}
VARIANTS = {**FACTORIAL, **STRENGTHS}
SUITES = {"factorial": tuple(FACTORIAL), "strengths": tuple(STRENGTHS), "all": tuple(VARIANTS)}


def _reference_recipe(seed):
    rows = [row for row in references()["cases"] if row["seed"] == seed]
    require(len(rows) == 1, "Choose a seed from the RC1 collection")
    reference = rows[0]
    accepted, recipe = released_recipe(seed)
    require(
        accepted["recipe_sha256"] == reference["rc1_recipe_sha256"]
        and accepted["source_sha256"] == reference["source_sha256"],
        "Paint-material reference recipe or source differs from RC1",
    )
    require(
        recipe["surface"]["height_scale"] == 60
        and recipe["simulation"]["height_scale_mm"] == 1.6
        and recipe["simulation"]["resolution"] == [2048, 1536]
        and recipe["render"]["still_resolution"] == [2048, 1536]
        and recipe["surface"]["roughness_bias"] == 0
        and "body_influence" not in recipe["simulation"]
        and "body_markers" not in recipe["render"],
        "Paint-material studies require the original native2048 all-three-body RC1 controls",
    )
    return accepted, reference, recipe


def _apply_variant(original, variant):
    recipe = copy.deepcopy(original)
    if variant == "rc1":
        return validate_recipe(recipe)
    spec = VARIANTS[variant]
    recipe["name"] = spec.label
    recipe["looks"] = ["silk-grain"]
    if spec.height_scale is not None:
        recipe["surface"]["height_scale"] = spec.height_scale
    if spec.relief_strength is not None:
        recipe["surface"]["interaction"]["directional_relief"] = {
            "version": "directional-contact-relief-v1",
            "strength": spec.relief_strength,
            "anisotropy": 0.8,
        }
    if spec.trait_amplitude is not None:
        recipe["simulation"]["interaction"]["material_variation"] = {
            "version": "paint-material-variation-v1",
            "amplitude": spec.trait_amplitude,
            "coarse_scale": 0.16,
            "fine_scale": 0.035,
            "fine_fraction": 0.25,
        }
    if spec.resistance_strength is not None:
        recipe["simulation"]["rheology"] = {
            "version": "paint-rheology-v1",
            "strength": spec.resistance_strength,
        }
    if spec.roughness_bias is not None:
        recipe["surface"]["roughness_bias"] = spec.roughness_bias
    return validate_recipe(recipe)


def material_recipe(seed, variant):
    """Reconstruct and hash-check RC1 before applying only the declared controls."""
    require(type(variant) is str and variant in VARIANTS, "Unknown paint-material study")
    accepted, reference, recipe = _reference_recipe(seed)
    return accepted, reference, _apply_variant(recipe, variant)


def recipe_catalog(seed):
    """Exact archived recipes, including the runner's deterministic background resolution.

    Resolution is applied only to the trusted expected recipes. A supplied
    archived recipe with a different ground color must still fail identification.
    """
    from .backgrounds import generate_background
    from .palette import generate_palette

    _, _, original = _reference_recipe(seed)
    catalog = {}
    for variant in VARIANTS:
        recipe = _apply_variant(original, variant)
        if "background" in recipe:
            palette = generate_palette(seed, recipe["chromatic_count"], mode=recipe["palette_mode"])
            recipe["surface"]["ground_srgb"] = generate_background(recipe["background"], palette)[
                "ground_srgb"
            ]
        catalog[variant] = recipe
    return catalog


def identify_recipe(seed, recipe):
    """Identify a preset by all canonical recipe controls, never by its label."""
    supplied = encoded(recipe)
    matches = [
        variant
        for variant, expected in recipe_catalog(seed).items()
        if encoded(expected) == supplied
    ]
    require(len(matches) == 1, "Archived recipe does not match an exact paint-material study")
    return matches[0]


def _study_metadata(variant, recipe, accepted):
    spec = VARIANTS[variant]
    return {
        "version": VERSION,
        "variant": variant,
        "suite": "factorial" if variant in FACTORIAL else "strengths",
        "label": spec.label,
        "description": spec.description,
        "features": {feature: feature in spec.features for feature in FEATURES},
        "parameters": {
            "display_height_scale": recipe["surface"]["height_scale"],
            "physical_height_scale_mm": recipe["simulation"]["height_scale_mm"],
            "directional_relief": copy.deepcopy(
                recipe["surface"]["interaction"].get("directional_relief")
            ),
            "material_variation": copy.deepcopy(
                recipe["simulation"]["interaction"].get("material_variation")
            ),
            "rheology": copy.deepcopy(recipe["simulation"].get("rheology")),
            "roughness_bias": recipe["surface"]["roughness_bias"],
        },
        "reference_case_id": accepted["id"],
        "reference_recipe_sha256": accepted["recipe_sha256"],
        "comparison_look": "silk-grain",
        "all_three_bodies": True,
        "material_identity": (
            "Paint history may differ; original orbits and starting design are fixed."
            if spec.resistance_strength is not None
            else "Original ten material fields must match RC1; interaction history may differ."
            if spec.trait_amplitude is not None
            else "Complete material history must match RC1; only display controls may differ."
        ),
    }


def make_plan(seeds=None, variants=None, *, suite="factorial", film=False, source_root=None):
    """Default: 16 combinations x 3 seeds, full-source stills at native2048."""
    require(type(suite) is str and suite in SUITES, "Unknown paint-material study suite")
    require(variants is None or suite == "factorial", "Choose a suite or explicit variants")
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    variants = SUITES[suite] if variants is None else variants
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
        and all(type(variant) is str and variant in VARIANTS for variant in variants)
        and len(set(variants)) == len(variants),
        "Choose distinct paint-material studies",
    )
    require(type(film) is bool, "film must be boolean")
    cohort = read(ROOT / "recipes/ten-seeds.json")
    source_root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    for seed in seeds:
        for variant in variants:
            accepted, reference, recipe = material_recipe(seed, variant)
            spec = VARIANTS[variant]
            case = {
                "id": f"{seed}-{variant}",
                "seed": seed,
                "source": str(source_root / f"{seed}.orbit"),
                "source_sha256": reference["source_sha256"],
                "recipe": recipe,
                "mode": "film" if film else "still",
                "reference_initial_mass": [*reference["target_mass"], 0.0],
                "reference_layout_artifact": copy.deepcopy(reference["layout_artifact"]),
                "reference_palette_identity_sha256": reference["palette_identity_sha256"],
                "study": _study_metadata(variant, recipe, accepted),
            }
            if spec.resistance_strength is None:
                case["accepted_base_material_sha256"] = accepted["base_material_sha256"]
                if spec.trait_amplitude is None:
                    case["accepted_physical_state_sha256"] = accepted["physical_state_sha256"]
            cases.append(case)
    plan = {
        "schema_version": 1,
        "version": VERSION,
        "runtime": runtime_identity(),
        "cases": cases,
        "reference_tag": "RC1",
        "reference_inputs_sha256": hashlib.sha256(INPUTS.read_bytes()).hexdigest(),
        "reference_release_sha256": hashlib.sha256(RELEASE.read_bytes()).hexdigest(),
        "feature_order": list(FEATURES),
        "contract": (
            "Complete original three-body trajectories, fixed RC1 initial layout, palette "
            "and pigment amounts. "
            "Native2048 material and final images; unchanged camera, light and film timeline. "
            "Displayed height and directional relief affect appearance only; transported traits "
            "affect wet-contact kinetics; structural resistance may change paint motion. "
            "No stationary noise, "
            "body masking, extra pigments or position markers. Optical-only cases require full RC1 "
            "material identity; traits without resistance require unchanged original ten "
            "material fields."
        ),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", nargs="+")
    seeds.add_argument("--all-seeds", action="store_true")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--suite", choices=SUITES, default="factorial")
    variants.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--film", action="store_true", help="Render complete films in addition to stills"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected_seeds = (
        [row["seed"] for row in references()["cases"]] if args.all_seeds else args.seeds
    )
    execute_plan(
        args.output,
        make_plan(
            selected_seeds,
            args.variants,
            suite=args.suite,
            film=args.film,
            source_root=args.source_root,
        ),
        workers=2,
    )


if __name__ == "__main__":
    main()
