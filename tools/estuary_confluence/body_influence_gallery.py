"""Verified RC1 comparisons that change only which bodies influence the paint.

All three recorded trajectories and the starting painting remain fixed. Six
proper subsets are compared with the preserved RC1 all-body view. Derived pair
events are allowed to change and must satisfy the archived eligible-pair
contract; they are not mistaken for body-indexed event slots.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
from itertools import combinations
from pathlib import Path

from tools.estuary_studio.common import artifact, checked, read, require, write
from tools.estuary_studio.gallery import _copy_verified

from .body_influence import validate_config as influence_config
from .body_influence import validate_event_eligibility
from .composition_gallery import (
    INPUTS,
    REFERENCE_RELEASE,
    ComparisonPresentation,
    _copy_gallery,
    _initial_mass,
    _record,
    _reference_context,
    _same,
    _view,
    document,
)
from .gallery import build_gallery, verify_gallery

VERSION = "body-influence-comparison-v1"


def _selection_name(bodies):
    numbers = [str(body + 1) for body in bodies]
    return (
        ("body-" + numbers[0], "Body " + numbers[0] + " only")
        if len(numbers) == 1
        else ("bodies-" + "-".join(numbers), "Bodies " + " + ".join(numbers))
    )


SUBSETS = tuple(group for size in (1, 2) for group in combinations(range(3), size))
SETUPS = {_selection_name(bodies)[0]: bodies for bodies in SUBSETS}
LABELS = {_selection_name(bodies)[0]: _selection_name(bodies)[1] for bodies in SUBSETS}
PRESENTATION = ComparisonPresentation(
    manifest_version=VERSION,
    default_setup="body-1",
    eyebrow="Same orbits / Different paint influence",
    intro=(
        "All three bodies keep their original recorded orbits. All starting paint colors "
        "remain: a single selected body can move all three pigments. Selection changes "
        "only who influences the paint, without boosting active strength. The starting "
        "paint, camera and duration stay fixed."
    ),
    selector_label="paint influence",
    browse_title="All body selections for this seed",
    browse_note=(
        "Choose two body selections to compare. Switching a selection pauses both films "
        "at the same point."
    ),
    initial_status="Identical starting paint · selected bodies change its development",
    final_status="Final paintings · unchanged orbits, selected paint influence",
    ready_status="Final paintings · choose which bodies influence the paint",
)


def _comparison_recipe(recipe):
    result = copy.deepcopy(recipe)
    result.pop("name", None)
    result.pop("looks", None)
    result["simulation"].pop("body_influence", None)
    return result


def _selection(request, receipt):
    from .run import body_influence_metadata

    supplied = request["recipe"]["simulation"].get("body_influence")
    config = influence_config(supplied)
    require(config is not None, "New influence cases must select a proper body subset")
    _same(supplied, config, "Body influence selection is not canonical")
    bodies = tuple(config["bodies"])
    require(bodies in SUBSETS, "Unknown body influence subset")
    validate_event_eligibility(request["events"], config)
    metadata = body_influence_metadata(request["recipe"], request["source"], request["events"])
    require(metadata is not None, "Missing body influence provenance")
    _same(request.get("body_influence"), metadata, "Body influence request metadata differs")
    _same(receipt.get("body_influence"), metadata, "Body influence receipt metadata differs")
    return _selection_name(bodies)[0], bodies, metadata


def comparison_manifest(output, *, title):
    """Re-derive selector labels and comparison guarantees from bound archives."""
    output = Path(output)
    seeds, references = _reference_context(output, inputs_name="rc1-inputs.json")
    studies_root = output / "studies"
    studies, provenance = verify_gallery(studies_root)
    require(
        len(studies["studies"]) == len(seeds) * len(SETUPS)
        and all(study["group"] == "silk-grain" for study in studies["studies"]),
        "Expected six complete body-influence films per seed",
    )
    indexed = {}
    for study in studies["studies"]:
        request, receipt, origin = _record(studies_root, provenance, study)
        setup, bodies, metadata = _selection(request, receipt)
        key = study["seed"], setup
        require(key not in indexed, "Duplicate body selection for one seed")
        indexed[key] = study, request, receipt, origin, bodies, metadata
    rows = []
    for _inputs, reference, original, original_receipt, _origin, initial_mass in references:
        seed = reference["seed"]
        require(
            "body_influence" not in original["recipe"]["simulation"]
            and "body_influence" not in original
            and "body_influence" not in original_receipt,
            "The saved RC1 reference must retain the original all-body path",
        )
        rows.append(
            {
                **_view(reference, "reference", "rc1", "All 3 · RC1", original, initial_mass),
                "active_body_indices": [0, 1, 2],
            }
        )
        for setup in SETUPS:
            require((seed, setup) in indexed, "A seed is missing a body selection")
            study, request, receipt, origin, bodies, metadata = indexed[(seed, setup)]
            require(
                request["recipe"]["looks"] == ["silk-grain"], "Use the matched RC1 contact finish"
            )
            _same(
                _comparison_recipe(request["recipe"]),
                _comparison_recipe(original["recipe"]),
                "Influence experiment changes controls beyond the body selection",
            )
            for name in ("source", "layout", "palette", "spectral", "background", "frames"):
                _same(request.get(name), original.get(name), f"Influence {name} differs from RC1")
            _same(
                request["surface_configs"]["silk-grain"],
                original["surface_configs"]["silk-grain"],
                "Influence camera or lighting differs from RC1",
            )
            _same(
                receipt["artifacts"]["layout.json"],
                original_receipt["artifacts"]["layout.json"],
                "Influence initial layout artifact differs from RC1",
            )
            _same(
                receipt["artifacts"]["silk-grain/initial.png"],
                original_receipt["artifacts"]["silk-grain/initial.png"],
                "Influence starting-paint image differs from RC1",
            )
            amounts = _initial_mass(studies_root, origin, study)
            _same(
                amounts.tolist(),
                initial_mass.tolist(),
                "Influence initial pigment amounts differ from RC1",
            )
            require(
                study["film"] is not None and reference["film"] is not None,
                "Every influence comparison needs complete films",
            )
            for key in ("film_frames", "film_fps", "film_seconds", "film_resolution", "resolution"):
                _same(study[key], reference[key], f"Influence comparison {key} differs")
            rows.append(
                {
                    **_view(study, "studies", setup, LABELS[setup], request, amounts),
                    "active_body_indices": list(bodies),
                    "body_influence": metadata,
                    "influence_record": f"studies/{origin['request']}",
                }
            )
    return {
        "schema_version": 1,
        "version": VERSION,
        "title": title,
        "seeds": seeds,
        "setups": [
            {"id": "rc1", "label": "All 3 · RC1"},
            *({"id": setup, "label": LABELS[setup]} for setup in SETUPS),
        ],
        "studies": rows,
        "comparison": (
            "All three recorded trajectories and the RC1 initial painting remain unchanged. "
            "Only the selected bodies and their eligible pair events influence the paint. "
            "Palette, pigment amounts, camera, lighting and timeline are matched."
        ),
        "body_indexing": "zero-based source indices; displayed labels add one",
    }


def build_comparison(
    output,
    cases,
    reference_gallery,
    *,
    title="The Estuary · Body influence",
    inputs=INPUTS,
    reference_release=REFERENCE_RELEASE,
):
    page = document(title, presentation=PRESENTATION)
    output, reference_gallery = Path(output).resolve(), Path(reference_gallery).resolve(strict=True)
    cases = [Path(case).resolve(strict=True) for case in cases]
    require(not output.exists(), "Use a new body-influence comparison directory")
    require(not output.is_relative_to(reference_gallery), "Publish outside the RC1 gallery")
    require(
        bool(cases) and all(output != case and not output.is_relative_to(case) for case in cases),
        "Publish outside immutable influence archives",
    )
    output.mkdir(parents=True)
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        build_gallery(
            output / "studies", cases, title="Body influence · Source records", layout="studies"
        )
        _copy_gallery(reference_gallery, output / "reference")
        for source, name in (
            (Path(inputs), "rc1-inputs.json"),
            (Path(reference_release), "reference-release.json"),
        ):
            _copy_verified(source, output / name, artifact(source))
        write(output / "comparison.json", comparison_manifest(output, title=title))
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
                        "rc1-inputs.json",
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
    output = Path(output)
    proof = read(output / "comparison-verification.json")
    require(
        proof.get("schema_version") == 1
        and proof.get("version") == VERSION
        and proof.get("complete") is True,
        "Incomplete body-influence publication",
    )
    require(
        set(proof["artifacts"])
        == {
            "index.html",
            "comparison.json",
            "rc1-inputs.json",
            "reference-release.json",
            "studies/curation.json",
            "reference/curation.json",
        },
        "Incomplete body-influence provenance",
    )
    for name, info in proof["artifacts"].items():
        checked(output, name, info)
    expected = comparison_manifest(output, title=proof["title"])
    _same(
        read(output / "comparison.json"),
        expected,
        "Body-influence manifest differs from its verified source records",
    )
    return expected, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-gallery", type=Path, required=True)
    parser.add_argument("--title", default="The Estuary · Body influence")
    args = parser.parse_args()
    print(build_comparison(args.output, args.cases, args.reference_gallery, title=args.title))


if __name__ == "__main__":
    main()
