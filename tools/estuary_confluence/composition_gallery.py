"""Portable same-seed comparisons of verified starting-composition studies.

The ordinary gallery publisher remains responsible for verifying and copying
artwork. This page adds only a comparison manifest over two verified portable
galleries. RC1 is retained intact for provenance; only its silk-grain views are
eligible reference panels. Different compositions have different material
histories and are never advertised as alternate views of one physical state.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import html
from pathlib import Path

import numpy as np

from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.gallery import _copy_verified

from .gallery import build_gallery, verify_gallery
from .mass_budget import RELATIVE_TOLERANCE
from .palette import normalize_seed

ROOT = Path(__file__).parent
VERSION = "composition-comparison-v1"
INPUTS = ROOT / "releases/composition-rc1-inputs-v1.json"
REFERENCE_RELEASE = ROOT / "releases/contact-textures-v1.json"
TEMPLATE = ROOT / "composition_gallery.html"
TITLE_TOKEN = "__COMPOSITION_TITLE__"
SETUPS = {
    "random-circles": "Random circles",
    "random-ribbons": "Tapered ribbons",
    "random-crescents": "Open crescents",
    "facing-shores": "Facing shores",
    "scattered-commas": "Scattered commas",
    "body-wedges": "Body-centered wedges",
}


def document(title):
    require(type(title) is str and 0 < len(title) <= 120, "Use a short comparison title")
    template = TEMPLATE.read_text(encoding="utf-8")
    require(template.count(TITLE_TOKEN) == 2, "Comparison title placeholders differ")
    return template.replace(TITLE_TOKEN, html.escape(title))


def _same(actual, expected, message):
    require(encoded(actual) == encoded(expected), message)


def _record(gallery, curation, study):
    source = next((s for s in curation["sources"] if s["id"] == study["case_id"]), None)
    require(source is not None, "Comparison study has no portable source record")
    return (
        read(gallery / source["request"]),
        read(gallery / source["receipt"]),
        source,
    )


def _comparison_recipe(recipe):
    """Remove only declared initial-composition and display-guide differences."""
    result = copy.deepcopy(recipe)
    result.pop("name", None)
    result.pop("looks", None)
    for key in ("initial_pattern", "initial_composition", "initial_pigment_weights"):
        result["simulation"].pop(key, None)
    result["render"].pop("body_markers", None)
    return result


def _initial_mass(folder, source, study):
    path = study.get("mass_budget_record")
    require(
        path is not None and path == source.get("mass_budget"), "Missing initial paint accounting"
    )
    value = np.asarray(read(folder / path).get("initial_mass"), dtype="f8")
    require(
        value.shape == (4,)
        and np.isfinite(value).all()
        and np.all(value[:3] > 0)
        and value[3] == 0,
        "Composition comparison requires three loaded pigments and no added chalk",
    )
    return value


def _view(study, prefix, setup, label, request, amounts):
    def media(name):
        value = study.get(name)
        require(type(value) is str and bool(value), f"Comparison is missing {name}")
        return f"{prefix}/{value}"

    return {
        "seed": study["seed"],
        "setup": setup,
        "label": label,
        "study_id": study["id"],
        "case_id": study["case_id"],
        "image": media("image"),
        "initial": media("initial"),
        "preview": media("preview"),
        "film": media("film"),
        "image_resolution": study["resolution"],
        "initial_resolution": request["recipe"]["render"]["resolution"],
        "film_resolution": study["film_resolution"],
        "film_frames": study["film_frames"],
        "film_fps": study["film_fps"],
        "film_seconds": study["film_seconds"],
        "source_sha256": study["source_sha256"],
        "palette_identity_sha256": study["palette_identity_sha256"],
        "physical_state_sha256": study["physical_state_sha256"],
        "initial_pigment_mass": amounts.tolist(),
        "body_markers": bool(study.get("body_markers")),
        "palette_record": media("palette_record"),
        "mass_budget_record": media("mass_budget_record"),
        **(
            {"body_marker_record": media("body_marker_record")} if study.get("body_markers") else {}
        ),
    }


def comparison_manifest(output, *, title):
    """Derive all UI claims from independently verified portable records."""
    output = Path(output)
    inputs, release = (
        read(output / "composition-inputs.json"),
        read(output / "reference-release.json"),
    )
    require(
        inputs.get("schema_version") == 1
        and inputs.get("version") == "composition-rc1-inputs-v1"
        and inputs.get("reference_tag") == "RC1"
        and inputs.get("reference_release_sha256")
        == artifact(output / "reference-release.json")["sha256"],
        "Composition inputs are not bound to the preserved RC1 release",
    )
    input_rows = inputs.get("cases")
    require(type(input_rows) is list and bool(input_rows), "Composition inputs have no seeds")
    seeds = [normalize_seed(row["seed"]) for row in input_rows]
    require(len(seeds) == len(set(seeds)), "Composition inputs repeat a seed")
    released = {row["seed"]: row for row in release["cases"]}
    require(
        set(released) == set(seeds) and len(released) == len(release["cases"]), "RC1 cohort differs"
    )
    reference_root, studies_root = output / "reference", output / "studies"
    references, reference_provenance = verify_gallery(reference_root)
    studies, study_provenance = verify_gallery(studies_root)
    selected_reference = [s for s in references["studies"] if s["group"] == "silk-grain"]
    require(
        len(selected_reference) == len(seeds)
        and {s["seed"] for s in selected_reference} == set(seeds),
        "Expected exactly one RC1 contact-finish reference for every seed",
    )
    require(
        len(studies["studies"]) == len(seeds) * len(SETUPS)
        and all(s["group"] == "silk-grain" for s in studies["studies"]),
        "Expected six complete composition films per seed",
    )
    rows = []
    new_index = {}
    for study in studies["studies"]:
        request, receipt, origin = _record(studies_root, study_provenance, study)
        spec = request["recipe"]["simulation"].get("initial_composition")
        require(type(spec) is dict and spec.get("setup") in SETUPS, "Unknown starting composition")
        key = (study["seed"], spec["setup"])
        require(key not in new_index, "Duplicate composition for one seed")
        new_index[key] = study, request, receipt, origin
    for inputs_row in input_rows:
        seed = normalize_seed(inputs_row["seed"])
        ref = next(s for s in selected_reference if s["seed"] == seed)
        reference, receipt, origin = _record(reference_root, reference_provenance, ref)
        accepted = released[seed]
        require(
            inputs_row["rc1_case_id"] == accepted["id"]
            and inputs_row["rc1_recipe_sha256"]
            == accepted["recipe_sha256"]
            == hashlib.sha256(encoded(reference["recipe"])).hexdigest()
            and ref["identity_sha256"] == accepted["request_identity_sha256"]
            and ref["physical_state_sha256"] == accepted["physical_state_sha256"]
            and ref["source_sha256"] == inputs_row["source_sha256"] == accepted["source_sha256"]
            and ref["palette_identity_sha256"] == inputs_row["palette_identity_sha256"],
            "Reference source, palette or material differs from RC1",
        )
        require(ref["chromatic_count"] == 3, "Reference must use three pigments")
        accepted_view = next(
            (view for view in accepted["views"] if view["look"] == "silk-grain"), None
        )
        require(accepted_view is not None, "RC1 has no contact-finish view")
        require(
            receipt["artifacts"]["silk-grain/poster.png"]["sha256"]
            == accepted_view["image"]["sha256"]
            and receipt["artifacts"]["silk-grain/film.mp4"]["sha256"]
            == accepted_view["movie"]["sha256"],
            "RC1 reference media differ from the saved release",
        )
        require(
            receipt["artifacts"]["mass-budget.json"] == inputs_row["mass_budget_artifact"]
            and receipt["artifacts"]["layout.json"] == inputs_row["layout_artifact"],
            "Initial composition inputs refer to different RC1 records",
        )
        target = np.asarray(inputs_row["target_mass"], dtype="f8")
        require(
            target.shape == (3,) and np.isfinite(target).all() and np.all(target > 0),
            "Invalid target mass",
        )
        radii = [pool["radius"] for pool in reference["layout"]["pools"]]
        _same(radii, inputs_row["reference_radii"], "Reference pool radii differ")
        ref_mass = _initial_mass(reference_root, origin, ref)
        _same(ref_mass[:3].tolist(), inputs_row["target_mass"], "RC1 starting paint amounts differ")
        rows.append(_view(ref, "reference", "rc1", "RC1 · saved composition", reference, ref_mass))
        for setup, label in SETUPS.items():
            require((seed, setup) in new_index, "A seed is missing a composition")
            study, request, _, source = new_index[(seed, setup)]
            recipe, spec = request["recipe"], request["recipe"]["simulation"]["initial_composition"]
            require(
                recipe["simulation"]["initial_pattern"] == "shaped"
                and recipe["simulation"].get("initial_pigment_weights") is None
                and spec.get("version") == "initial-composition-v1"
                and recipe["looks"] == ["silk-grain"],
                "Composition recipe does not use the declared source-free shape model",
            )
            _same(
                spec["target_mass"], inputs_row["target_mass"], "Composition target amounts differ"
            )
            _same(
                spec["reference_radii"],
                inputs_row["reference_radii"],
                "Composition reference radii differ",
            )
            layout = request.get("layout")
            require(
                type(layout) is dict
                and layout.get("setup") == setup
                and layout.get("version") == spec["version"]
                and layout.get("seed") == seed
                and layout.get("count") == 3,
                "Composition label differs from its archived starting layout",
            )
            _same(layout.get("target_mass"), spec["target_mass"], "Layout paint amounts differ")
            _same(
                layout.get("reference_radii"),
                spec["reference_radii"],
                "Layout reference radii differ",
            )
            _same(
                _comparison_recipe(recipe),
                _comparison_recipe(reference["recipe"]),
                "Composition changes controls beyond starting paint",
            )
            for name in ("source", "palette", "events", "spectral", "background", "frames"):
                _same(
                    request.get(name),
                    reference.get(name),
                    f"Composition {name} differs from its reference",
                )
            _same(
                request["surface_configs"]["silk-grain"],
                reference["surface_configs"]["silk-grain"],
                "Composition camera or lighting differs",
            )
            amounts = _initial_mass(studies_root, source, study)
            require(
                np.allclose(amounts[:3], target, rtol=RELATIVE_TOLERANCE, atol=0),
                "Composition starting pigment amounts differ beyond float32 tolerance",
            )
            require(
                study["film"] is not None and ref["film"] is not None,
                "Every comparison needs complete films",
            )
            for key in ("film_frames", "film_fps", "film_seconds", "film_resolution", "resolution"):
                _same(study[key], ref[key], f"Comparison {key} differs")
            rows.append(_view(study, "studies", setup, label, request, amounts))
    return {
        "schema_version": 1,
        "version": VERSION,
        "title": title,
        "seeds": seeds,
        "setups": [
            {"id": "rc1", "label": "RC1 · saved composition"},
            *({"id": key, "label": value} for key, value in SETUPS.items()),
        ],
        "studies": rows,
        "comparison": (
            "Same seed, trajectory, palette, initial pigment amounts, camera and timeline; "
            "different starting compositions and material histories"
        ),
        "mass_relative_tolerance": RELATIVE_TOLERANCE,
    }


def _copy_gallery(source, output):
    """Copy the already verified portable publication without rewriting records."""
    _, curation = verify_gallery(source)
    for name, info in {
        **curation["artifacts"],
        "curation.json": artifact(source / "curation.json"),
    }.items():
        _copy_verified(source / name, output / name, info)
    verify_gallery(output)


def build_comparison(
    output,
    cases,
    reference_gallery,
    *,
    title="The Estuary · Shape & placement",
    inputs=INPUTS,
    reference_release=REFERENCE_RELEASE,
):
    page = document(title)
    output, reference_gallery = Path(output).resolve(), Path(reference_gallery).resolve(strict=True)
    cases = [Path(case).resolve(strict=True) for case in cases]
    require(not output.exists(), "Use a new comparison publication directory")
    require(not output.is_relative_to(reference_gallery), "Publish outside the RC1 gallery")
    require(
        bool(cases) and all(output != case and not output.is_relative_to(case) for case in cases),
        "Publish outside immutable composition archives",
    )
    output.mkdir(parents=True)
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        build_gallery(
            output / "studies",
            cases,
            title="Composition studies · Source records",
            layout="studies",
        )
        _copy_gallery(reference_gallery, output / "reference")
        for source, name in (
            (Path(inputs), "composition-inputs.json"),
            (Path(reference_release), "reference-release.json"),
        ):
            _copy_verified(source, output / name, artifact(source))
        manifest = comparison_manifest(output, title=title)
        write(output / "comparison.json", manifest)
        (output / "index.html").write_text(page, encoding="utf-8")
        write(
            output / "comparison-verification.json",
            {
                "schema_version": 1,
                "version": VERSION,
                "complete": True,
                "title": title,
                "artifacts": {
                    name: artifact(output / name)
                    for name in (
                        "index.html",
                        "comparison.json",
                        "composition-inputs.json",
                        "reference-release.json",
                        "studies/curation.json",
                        "reference/curation.json",
                    )
                },
            },
        )
        verify_comparison(output)
    return output / "index.html"


def verify_comparison(output):
    """Re-derive every comparison claim, including after portable media are moved."""
    output = Path(output)
    proof = read(output / "comparison-verification.json")
    require(
        proof.get("schema_version") == 1
        and proof.get("version") == VERSION
        and proof.get("complete") is True,
        "Incomplete composition publication",
    )
    required = {
        "index.html",
        "comparison.json",
        "composition-inputs.json",
        "reference-release.json",
        "studies/curation.json",
        "reference/curation.json",
    }
    require(set(proof["artifacts"]) == required, "Incomplete comparison provenance")
    for name, info in proof["artifacts"].items():
        checked(output, name, info)
    expected = comparison_manifest(output, title=proof["title"])
    _same(
        read(output / "comparison.json"),
        expected,
        "Comparison manifest differs from its verified source records",
    )
    return expected, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-gallery", type=Path, required=True)
    parser.add_argument("--title", default="The Estuary · Shape & placement")
    args = parser.parse_args()
    print(build_comparison(args.output, args.cases, args.reference_gallery, title=args.title))


if __name__ == "__main__":
    main()
