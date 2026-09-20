"""Portable, verified comparison of all-three-body paint material studies."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.gallery import _copy_verified

from .composition_gallery import _initial_mass, _record
from .gallery import build_gallery, verify_gallery
from .paint_material_studies import FEATURES, INPUTS, RELEASE, VARIANTS, identify_recipe
from .palette import validate_palette

VERSION = "paint-material-review-v1"
TEMPLATE = Path(__file__).with_suffix(".html")
DOCUMENTS = (
    "comparison.json",
    "index.html",
    "rc1-inputs.json",
    "reference-release.json",
    "studies/curation.json",
)


def _references(output):
    inputs, release = read(output / "rc1-inputs.json"), read(output / "reference-release.json")
    require(
        inputs.get("schema_version") == 1
        and inputs.get("version") == "composition-rc1-inputs-v1"
        and inputs.get("reference_tag") == "RC1"
        and inputs.get("reference_release_sha256")
        == artifact(output / "reference-release.json")["sha256"],
        "Material reference inputs are not bound to RC1",
    )
    bound = {row["seed"]: row for row in inputs["cases"]}
    accepted = {row["seed"]: row for row in release["cases"]}
    require(
        len(bound) == len(inputs["cases"])
        and len(accepted) == len(release["cases"])
        and bool(bound)
        and set(bound) == set(accepted),
        "RC1 reference cohort differs",
    )
    for seed, row in bound.items():
        original = accepted[seed]
        require(
            row["rc1_case_id"] == original["id"]
            and row["source_sha256"] == original["source_sha256"]
            and row["rc1_recipe_sha256"] == original["recipe_sha256"],
            "RC1 reference recipe or source differs",
        )
    return bound, accepted


def manifest(output, *, title, picks):
    """Re-derive every selection from canonical recipes and portable provenance."""
    output = Path(output)
    references, accepted = _references(output)
    root = output / "studies"
    collection, provenance = verify_gallery(root)
    rows, indexed, seeds, requests, histories = [], set(), [], {}, {}
    for study in collection["studies"]:
        if study["group"] != "silk-grain":
            continue
        request, receipt, origin = _record(root, provenance, study)
        seed = study["seed"]
        require(seed in references, "Material study seed is outside the RC1 cohort")
        reference, baseline = references[seed], accepted[seed]
        variant = identify_recipe(seed, request["recipe"])
        key = seed, variant
        require(key not in indexed, "Repeated seed and material setup")
        indexed.add(key)
        requests[key] = request
        if seed not in seeds:
            seeds.append(seed)
        spec = VARIANTS[variant]
        validate_palette(request["palette"])
        require(
            study["source_sha256"] == reference["source_sha256"]
            and study["palette_identity_sha256"] == reference["palette_identity_sha256"],
            "Material study source or palette differs from pinned RC1",
        )
        require(
            receipt["artifacts"].get("layout.json") == reference["layout_artifact"],
            "Material study initial layout differs from pinned RC1",
        )
        require(
            _initial_mass(root, origin, study).tolist() == [*reference["target_mass"], 0.0],
            "Material study initial pigment amounts differ from pinned RC1",
        )
        if spec.resistance_strength is None:
            require(
                receipt["base_material_sha256"] == baseline["base_material_sha256"],
                "Material study changed the original RC1 material fields",
            )
            if spec.trait_amplitude is None:
                require(
                    receipt["physical_state_sha256"] == baseline["physical_state_sha256"],
                    "Optical study changed the complete RC1 material history",
                )
        if variant == "rc1":
            original_view = next(view for view in baseline["views"] if view["look"] == "silk-grain")
            expected_image = {key: original_view["image"][key] for key in ("bytes", "sha256")}
            require(
                receipt["artifacts"]["silk-grain/poster.png"] == expected_image,
                "RC1 reference painting differs from its preserved release",
            )
        require(
            "body_influence" not in request["recipe"]["simulation"],
            "Material studies require all three bodies",
        )
        history_key = seed, encoded(request["recipe"]["simulation"])
        previous_history = histories.setdefault(history_key, receipt["physical_state_sha256"])
        require(
            receipt["physical_state_sha256"] == previous_history,
            "Identical simulation settings produced different material histories",
        )
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "label": spec.label,
                "description": spec.description,
                "features": {feature: feature in spec.features for feature in FEATURES},
                "image": "studies/" + study["image"],
                "preview": "studies/" + study["preview"],
                "initial": "studies/" + study["initial"],
                "film": "studies/" + study["film"] if study["film"] else None,
                "resolution": study["resolution"],
                "initial_resolution": request["recipe"]["render"]["resolution"],
                "film_resolution": study["film_resolution"],
                "film_fps": study["film_fps"],
                "film_frames": study["film_frames"],
                "request": "studies/" + origin["request"],
                "physical_state_sha256": study["physical_state_sha256"],
                "base_material_sha256": receipt["base_material_sha256"],
                "source_sha256": study["source_sha256"],
                "palette_identity_sha256": study["palette_identity_sha256"],
                "surface": request["surface_configs"]["silk-grain"],
                "material_variation": request["recipe"]["simulation"]["interaction"].get(
                    "material_variation"
                ),
                "rheology": request["recipe"]["simulation"].get("rheology"),
            }
        )
    require(bool(rows), "Select at least one completed material study")
    for seed in seeds:
        require((seed, "rc1") in indexed, "Every seed requires its RC1 control")
        baseline = requests[(seed, "rc1")]
        for (case_seed, _), request in requests.items():
            if case_seed == seed:
                for field in ("source", "palette", "layout", "events", "spectral", "background"):
                    require(
                        encoded(request.get(field)) == encoded(baseline.get(field)),
                        f"Within-seed material {field} differs from its RC1 control",
                    )
        cohort = [r for r in rows if r["seed"] == seed]
        require(
            len({r["source_sha256"] for r in cohort}) == 1
            and len({r["palette_identity_sha256"] for r in cohort}) == 1,
            "Within-seed studies must share their complete trajectory and palette",
        )
    require(type(picks) is list, "Visual picks must be a list")
    pick_keys = set()
    for pick in picks:
        require(
            type(pick) is dict and set(pick) == {"seed", "variant", "note"}, "Invalid visual pick"
        )
        require(
            type(pick["seed"]) is str and type(pick["variant"]) is str,
            "Invalid visual pick identity",
        )
        key = pick["seed"], pick["variant"]
        require(key in indexed and key not in pick_keys, "Unknown or repeated visual pick")
        require(
            type(pick["note"]) is str and bool(pick["note"].strip()) and len(pick["note"]) <= 700,
            "Each pick requires a concise artistic judgment",
        )
        pick_keys.add(key)
    return {
        "version": VERSION,
        "title": title,
        "seeds": seeds,
        "variants": [v for v in VARIANTS if any(r["variant"] == v for r in rows)],
        "rows": rows,
        "picks": picks,
    }


def document(title):
    require(type(title) is str and 0 < len(title) <= 120, "Use a short gallery title")
    source = TEMPLATE.read_text()
    require(source.count("__TITLE__") == 2, "Gallery title placeholders differ")
    return source.replace("__TITLE__", html.escape(title))


def build_review(output, cases, *, picks=None, title="The Estuary · Paint material studies"):
    page = document(title)
    output = Path(output).resolve()
    require(not output.exists(), "Use a new immutable material review directory")
    build_gallery(output / "studies", cases, title=title, allow_stills=True, layout="studies")
    for source, name in ((INPUTS, "rc1-inputs.json"), (RELEASE, "reference-release.json")):
        _copy_verified(source, output / name, artifact(source))
    data = manifest(output, title=title, picks=[] if picks is None else picks)
    write(output / "comparison.json", data)
    (output / "index.html").write_text(page)
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
    require(record.get("version") == VERSION, "Unknown material publication version")
    require(
        type(record.get("artifacts")) is dict and set(record["artifacts"]) == set(DOCUMENTS),
        "Material publication provenance is incomplete",
    )
    for name, info in record["artifacts"].items():
        checked(output, name, info)
    expected = manifest(output, title=data["title"], picks=data["picks"])
    require(
        encoded(expected) == encoded(data)
        and hashlib.sha256(encoded(data)).hexdigest() == record["comparison_sha256"],
        "Material comparison differs from its verified provenance",
    )
    require(
        (output / "index.html").read_text() == document(data["title"]),
        "Material comparison page differs from the qualified template",
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+")
    parser.add_argument("--picks", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        data = verify_review(args.output)
    else:
        require(args.cases is not None, "Supply completed cases")
        data = build_review(args.output, args.cases, picks=read(args.picks) if args.picks else [])
    print(
        json.dumps({"verified": True, "paintings": len(data["rows"]), "seeds": len(data["seeds"])})
    )


if __name__ == "__main__":
    main()
