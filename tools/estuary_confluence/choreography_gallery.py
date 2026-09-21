"""Portable, provenance-checked RC1 comparisons for choreographed paint starts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.gallery import _copy_verified

from .choreography_studies import REFERENCES, RELEASE, VARIANTS, identify_recipe
from .composition_gallery import _copy_gallery, _initial_mass, _record, _reference_context
from .gallery import build_gallery, verify_gallery
from .mass_budget import RELATIVE_TOLERANCE
from .review_page import Presentation
from .review_page import document as review_document

VERSION = "choreography-review-v1"
PRESENTATION = Presentation(
    version=VERSION,
    eyebrow="One dance / Deliberate beginnings",
    intro=(
        "Original RC1 color and paint, with starting deposits arranged around the currents. "
        "Compare the opening arrangement, final painting and complete films for the same seed."
    ),
    legend=(
        "Every color keeps its original amount. Placement is guided by the full trajectory. "
        "Layer studies change the starting allocation between slower and faster paint layers. "
        "Gold outlines mark subjective visual picks."
    ),
    default_variant="stretch-ovals",
    film_only_selection=True,
)
DOCUMENTS = (
    "index.html",
    "comparison.json",
    "composition-inputs.json",
    "reference-release.json",
    "reference/curation.json",
    "studies/curation.json",
)


def document(title):
    return review_document(title, presentation=PRESENTATION)


def _shared_recipe(recipe):
    result = copy.deepcopy(recipe)
    for key in ("name", "looks", "launch_assessment"):
        result.pop(key, None)
    for key in ("initial_pattern", "initial_pigment_weights", "initial_choreography"):
        result["simulation"].pop(key, None)
    return result


def _row(study, request, origin, prefix, variant):
    from .laminate import layer_fractions

    spec = VARIANTS.get(variant)
    layout = request["layout"]
    fractions = layout.get(
        "effective_layer_fractions", layer_fractions(request["palette"], 4).tolist()
    )[:3]
    return {
        "seed": study["seed"],
        "variant": variant,
        "label": spec.label if spec else "RC1 reference",
        "description": spec.description if spec else "The preserved RC1 composition.",
        "features": {},
        **{
            key: f"{prefix}/{study[key]}" if study.get(key) else None
            for key in ("image", "preview", "initial", "film")
        },
        "resolution": study["resolution"],
        "initial_resolution": request["recipe"]["render"]["resolution"],
        **{
            key: study[key]
            for key in (
                "film_resolution",
                "film_fps",
                "film_frames",
                "physical_state_sha256",
                "source_sha256",
                "palette_identity_sha256",
            )
        },
        "request": f"{prefix}/{origin['request']}",
        "settings": {
            "starting_paint": request["recipe"]["simulation"].get(
                "initial_choreography", {"setup": "Preserved RC1"}
            ),
            "placement_diagnostics": layout.get("pilot"),
            "initial_layer_allocation": {
                "pigment_order": "First, second, third pigment in the saved palette",
                "fraction_in_faster_layer": fractions,
                "fraction_in_slower_layer": [1 - value for value in fractions],
            },
            "caution": (
                "Pilot measurements predict motion and contact; they are not an artistic score "
                "or a guarantee of final engagement."
            ),
        },
    }


def manifest(output, *, title, picks):
    from .choreography import effective_layer_fractions, validate_layout
    from .laminate import layer_fractions

    output = Path(output)
    seeds, references = _reference_context(output)
    collection, provenance = verify_gallery(output / "studies")
    available = {study["seed"] for study in collection["studies"]}
    seeds = [seed for seed in seeds if seed in available]
    controls = {ref[0]["seed"]: ref for ref in references}
    rows, indexed = [], set()
    for _inputs, study, request, _receipt, origin, _mass in references:
        if study["seed"] not in available:
            continue
        rows.append(_row(study, request, origin, "reference", "rc1"))
        indexed.add((study["seed"], "rc1"))
    for study in collection["studies"]:
        require(study["group"] == "silk-grain", "Choreography review requires the RC1 finish")
        request, _receipt, origin = _record(output / "studies", provenance, study)
        seed = study["seed"]
        require(seed in controls, "Unknown choreography seed")
        inputs, baseline, original, _, _, reference_mass = controls[seed]
        variant = identify_recipe(seed, request["recipe"])
        layout = validate_layout(request["layout"])
        config = request["recipe"]["simulation"]["initial_choreography"]
        require(
            layout["config"] == config
            and layout["seed"] == seed
            and layout["source_sha256"] == request["source"]["sha256"]
            and encoded(layout["source_projection"]) == encoded(request["source"]["projection"])
            and layout["aspect"]
            == request["recipe"]["simulation"]["resolution"][0]
            / request["recipe"]["simulation"]["resolution"][1]
            and layout["baseline_layer_fractions"]
            == layer_fractions(request["palette"], 4).tolist()
            and layout["effective_layer_fractions"]
            == effective_layer_fractions(request["palette"], config).tolist(),
            "Starting paint metadata differs from its recipe, palette or source",
        )
        require(variant != "rc1" and (seed, variant) not in indexed, "Repeated choreography study")
        indexed.add((seed, variant))
        require(
            encoded(_shared_recipe(request["recipe"]))
            == encoded(_shared_recipe(original["recipe"])),
            "Study changes controls beyond starting paint",
        )
        for field in ("source", "palette", "events", "spectral", "background"):
            require(
                encoded(request.get(field)) == encoded(original.get(field)),
                f"Study {field} differs from RC1",
            )
        require(
            encoded(request["surface_configs"]["silk-grain"])
            == encoded(original["surface_configs"]["silk-grain"]),
            "Study camera, light or finish differs",
        )
        require(
            study["source_sha256"] == inputs["source_sha256"]
            and study["palette_identity_sha256"] == inputs["palette_identity_sha256"],
            "Study source or palette differs from pinned RC1",
        )
        actual_mass = _initial_mass(output / "studies", origin, study)
        require(
            np.allclose(actual_mass, reference_mass, rtol=RELATIVE_TOLERANCE, atol=0),
            "Study pigment amounts differ",
        )
        if study["film"]:
            for key in ("film_frames", "film_fps", "film_resolution"):
                require(study[key] == baseline[key], f"Study {key} differs from RC1")
        rows.append(_row(study, request, origin, "studies", variant))
    require(len(rows) > len(seeds), "Select completed choreography experiments")
    require(type(picks) is list, "Visual picks must be a list")
    picked = set()
    for pick in picks:
        require(
            type(pick) is dict and set(pick) == {"seed", "variant", "note"}, "Invalid visual pick"
        )
        require(type(pick["seed"]) is str and type(pick["variant"]) is str, "Invalid pick identity")
        key = pick["seed"], pick["variant"]
        require(key in indexed and key not in picked, "Unknown or repeated visual pick")
        require(
            type(pick["note"]) is str and 0 < len(pick["note"].strip()) <= 700,
            "Explain each visual pick",
        )
        picked.add(key)
    return {
        "version": VERSION,
        "title": title,
        "seeds": seeds,
        "variants": [v for v in ("rc1", *VARIANTS) if any(r["variant"] == v for r in rows)],
        "rows": rows,
        "picks": picks,
    }


def build_review(
    output, cases, reference_gallery, *, picks=None, title="The Estuary · Deliberate beginnings"
):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new immutable choreography review directory")
    cases = [Path(case).resolve(strict=True) for case in cases]
    reference_gallery = Path(reference_gallery).resolve(strict=True)
    require(
        cases and all(not output.is_relative_to(case) for case in (*cases, reference_gallery)),
        "Publish outside immutable source archives",
    )
    build_gallery(output / "studies", cases, title=title, allow_stills=True, layout="studies")
    _copy_gallery(reference_gallery, output / "reference")
    for source, name in (
        (REFERENCES, "composition-inputs.json"),
        (RELEASE, "reference-release.json"),
    ):
        _copy_verified(source, output / name, artifact(source))
    data = manifest(output, title=title, picks=[] if picks is None else picks)
    write(output / "comparison.json", data)
    (output / "index.html").write_text(document(title))
    write(
        output / "publication.json",
        {
            "version": VERSION,
            "artifacts": {name: artifact(output / name) for name in DOCUMENTS},
            "comparison_sha256": hashlib.sha256(encoded(data)).hexdigest(),
        },
    )
    return verify_review(output)


def verify_review(output):
    output = Path(output)
    record, data = read(output / "publication.json"), read(output / "comparison.json")
    require(
        record.get("version") == VERSION and set(record.get("artifacts", {})) == set(DOCUMENTS),
        "Incomplete choreography publication",
    )
    for name, info in record["artifacts"].items():
        checked(output, name, info)
    expected = manifest(output, title=data["title"], picks=data["picks"])
    require(
        encoded(expected) == encoded(data)
        and hashlib.sha256(encoded(data)).hexdigest() == record["comparison_sha256"],
        "Comparison differs from source records",
    )
    require(
        (output / "index.html").read_text() == document(data["title"]),
        "Review page differs from its qualified template",
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+")
    parser.add_argument("--reference-gallery", type=Path)
    parser.add_argument("--picks", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        data = verify_review(args.output)
    else:
        require(
            args.cases and args.reference_gallery,
            "Supply completed studies and the preserved RC1 gallery",
        )
        data = build_review(
            args.output,
            args.cases,
            args.reference_gallery,
            picks=read(args.picks) if args.picks else [],
        )
    print(
        json.dumps(
            {
                "verified": True,
                "paintings": len(data["rows"]),
                "seeds": len(data["seeds"]),
                "films": sum(bool(r["film"]) for r in data["rows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
