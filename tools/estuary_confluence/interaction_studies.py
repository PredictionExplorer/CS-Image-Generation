"""Matched contact-texture studies anchored to the accepted Color & Form release.

Every view shares one complete trajectory and one material history. The baseline
recipe is first reconstructed and hash-checked against the preserved release.
Renderer controls then expose control, satin fabric, and satin plus aggregation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

from tools.estuary_studio.common import encoded, read, require

from .backgrounds import generate_background
from .interaction import validate_config as interaction_config
from .palette import generate_palette
from .run import ROOT, runtime_identity, validate_recipe
from .studies import execute_plan, study_recipe

PROFILES = {
    "gentle": ("Gentle contact", {}),
    "worked": (
        "Worked pigment",
        {
            "contact_rate": 8,
            "fabric_rate": 6,
            "aggregation_rate": 10,
            "breakup_rate": 0.05,
            "nucleation_scale": 0.006,
            "nucleation_contrast": 0.95,
        },
    ),
    "mineral": (
        "Mineral contact",
        {
            "contact_rate": 16,
            "fabric_rate": 15,
            "aggregation_rate": 20,
            "breakup_rate": 0.01,
            "nucleation_scale": 0.006,
            "nucleation_contrast": 1.0,
        },
    ),
}
PROFILES.update(
    {
        "woven": ("Woven contact", {**PROFILES["worked"][1], "advection": "maccormack"}),
        "mineral-sharp": (
            "Fine mineral contact",
            {**PROFILES["mineral"][1], "advection": "maccormack"},
        ),
        "encounter": (
            "Encounter grain",
            {
                "advection": "maccormack",
                "contact_rate": 16,
                "origin_distance": 0.12,
                "composition_threshold": 0.15,
                "fabric_rate": 12,
                "aggregation_rate": 40,
                "breakup_rate": 0.02,
                "nucleation_scale": 0.008,
                "nucleation_contrast": 1.0,
            },
        ),
    }
)
LOOKS = ["silk-grain", "silk", "control"]


def released_case(seed, count):
    release = read(ROOT / "releases/color-and-form-v1.json")
    matches = [c for c in release["cases"] if c["seed"] == seed and c["chromatic_count"] == count]
    require(len(matches) == 1, "Choose a seed/count from the accepted release")
    case = matches[0]
    recipe = study_recipe(count, case["flow"], width=2048, film=True)
    recipe["name"] = f"{count} pigment{'s' if count != 1 else ''} · Selected currents"
    palette = generate_palette(seed, count, mode=recipe["palette_mode"])
    recipe["surface"]["ground_srgb"] = generate_background(recipe["background"], palette)[
        "ground_srgb"
    ]
    require(
        hashlib.sha256(encoded(recipe)).hexdigest() == case["recipe_sha256"],
        "Accepted recipe has changed; select its pinned source release",
    )
    return copy.deepcopy(case), recipe


def texture_recipe(seed, count, profile, *, width=2048, film=False, appearance=None, looks=None):
    require(profile in PROFILES, "Unknown texture profile")
    require(type(width) is int and width in (1024, 2048, 4096), "Use a qualified material width")
    require(type(film) is bool, "film must be boolean")
    _, recipe = released_case(seed, count)
    label, controls = PROFILES[profile]
    recipe["name"] = f"{count} pigment{'s' if count != 1 else ''} · {label}"
    recipe["looks"] = list(LOOKS if looks is None else looks)
    recipe["simulation"]["interaction"] = interaction_config(controls)
    recipe["simulation"]["resolution"] = [width, width * 3 // 4]
    recipe["surface"]["interaction"] = {"silk_strength": 1.0, "grain_strength": 1.0}
    # The new marks come exclusively from transported interaction history.
    require(recipe["surface"]["grain_um"] == 0, "Stationary support grain must remain off")
    size = list(recipe["simulation"]["resolution"])
    recipe["render"].update(
        capture_resolution=size,
        resolution=[min(width, 1440), min(width, 1440) * 3 // 4],
        still_resolution=size if width < 4096 else [3840, 2880],
        frame_supersampling=2 if film else 1,
    )
    if appearance is not None:
        from .interaction_appearance import presentation

        palette = generate_palette(seed, count, mode=recipe["palette_mode"])
        view = presentation(
            {
                "recipe": recipe,
                "palette": palette,
                "background": generate_background(recipe["background"], palette),
            },
            appearance,
        )
        recipe["name"] += f" · {view['name']}"
        recipe["surface"] = view["surface"]
        recipe["render"]["still_tilt_degrees"] = view["camera"]["tilt_degrees"]
        recipe["render"]["azimuth_end"] = view["camera"]["azimuth_degrees"]
        recipe["render"]["orbit_tilt_degrees"] = max(
            recipe["render"]["orbit_tilt_degrees"], view["camera"]["tilt_degrees"]
        )
    return validate_recipe(recipe)


def make_plan(
    pairs, profiles, *, width=2048, film=False, source_root=None, appearance=None, looks=None
):
    require(pairs and len(set(pairs)) == len(pairs), "Choose distinct seed/count pairs")
    require(profiles and len(set(profiles)) == len(profiles), "Choose distinct texture profiles")
    cohort_path = ROOT / "recipes/ten-seeds.json"
    cohort = read(cohort_path)
    source_root = Path(source_root or Path(cohort["archive_root"]) / "orbits").resolve()
    cases = []
    for profile in profiles:
        for seed, count in pairs:
            baseline, _ = released_case(seed, count)
            cases.append(
                {
                    "id": f"{seed}-{count}-{profile}" + (f"-{appearance}" if appearance else ""),
                    "seed": seed,
                    "source": str(source_root / f"{seed}.orbit"),
                    "source_sha256": baseline["source_sha256"],
                    "recipe": texture_recipe(
                        seed,
                        count,
                        profile,
                        width=width,
                        film=film,
                        appearance=appearance,
                        looks=looks,
                    ),
                    "mode": "film" if film else "still",
                    "accepted_base_material_sha256": baseline["physical_state_sha256"]
                    if width == 2048
                    else None,
                }
            )
    plan = {
        "schema_version": 1,
        "runtime": runtime_identity(),
        "cases": cases,
        "contract": (
            "Complete source; optional contact/fabric/aggregation state; "
            "unchanged archived base pigment and geometry. Optional displayed packing relief "
            "reconstructs bounded volume-conserving native-grid thickness from aggregation. "
            "Selected surface views share full physical history. "
            "Native2048 cases compare base-state "
            "identity to accepted release; other resolutions are separate qualification studies."
        ),
        "accepted_release_sha256": hashlib.sha256(
            (ROOT / "releases/color-and-form-v1.json").read_bytes()
        ).hexdigest(),
    }
    plan["identity_sha256"] = hashlib.sha256(encoded(plan)).hexdigest()
    return plan


def main():
    from .interaction_appearance import PRESETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--counts", nargs="+", type=int, choices=(1, 2, 3), default=[3])
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["gentle"])
    parser.add_argument("--width", type=int, choices=(1024, 2048, 4096), default=2048)
    parser.add_argument("--film", action="store_true")
    parser.add_argument("--appearance", choices=PRESETS)
    parser.add_argument("--looks", nargs="+", choices=LOOKS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(
        [(seed, count) for count in args.counts for seed in args.seeds],
        args.profiles,
        width=args.width,
        film=args.film,
        appearance=args.appearance,
        looks=args.looks,
    )
    # Native4096 history plus three retained surfaces is qualified one job at a
    # time on the16GiB experiment GPU. Two workers remain bounded at2048.
    execute_plan(args.output, plan, workers=1 if args.width == 4096 else 2)


if __name__ == "__main__":
    main()
