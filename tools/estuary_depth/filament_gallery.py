"""Portable, receipt-bound comparisons of complete fine-fold paint studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from tools.estuary_confluence.review_page import Presentation, document
from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.gallery import _copy_verified

from .experiment import finished, verify_render
from .filament_batch import VERSION as STUDY_VERSION
from .filament_batch import _validated_plan, verify_case
from .filament_studies import REFERENCE_SEED, VARIANTS, make_depth_recipe, make_paint_recipe

VERSION = "filament-review-v1"
PRESENTATION = Presentation(
    version=VERSION,
    eyebrow="Three bodies / The folded tide",
    intro=(
        "Compare the original folds with changes to starting paint and current. "
        "Starting images show the actual paint before the complete trajectory; "
        "the final photographs show its completed relief."
    ),
    legend=(
        "Every material study follows the full recording. Lighting studies reuse exactly "
        "the Control material. Gold outlines mark subjective visual picks."
    ),
    default_variant="fine-bands",
    reference_variant="control",
    film_only_selection=True,
)
LIGHTING = {
    "light-flat": ("00-flat", "Flat pigment", "relief_mm", 0),
    "light-shallow": ("01-shallow", "Shallow folds", "relief_mm", 6),
    "light-deep": ("03-deep", "Deeper folds", "relief_mm", 24),
    "light-raking": ("04-raking", "Lower light", "elevation_degrees", 25),
}
MATERIAL_RECORDS = {
    "batch_plan": "../../plan.json",
    "study": "study.json",
    "paint_request": "paint/request.json",
    "paint_receipt": "paint/receipt.json",
    "paint_recipe": "paint/recipe.json",
    "bundle": "bundle/manifest.json",
    "photo_request": "photographs/00-painting/request.json",
    "photo_receipt": "photographs/00-painting/receipt.json",
    "photo_recipe": "photographs/00-painting/recipe.json",
    "photo_camera": "photographs/00-painting/camera.json",
    "photo_experiment": "photographs/experiment-request.json",
    "photo_result": "photographs/00-painting/experiment-result.json",
    "photo_input_recipe": "photographs/inputs/recipes/00-painting.json",
}
LIGHT_RECORD_KEYS = {
    "bundle",
    "photo_request",
    "photo_receipt",
    "photo_recipe",
    "photo_camera",
    "experiment",
    "result",
    "photo_input_recipe",
}


def _sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _fingerprint(value):
    require(type(value) is dict, "Invalid artifact record")
    require(
        type(value.get("sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        and type(value.get("bytes")) is int
        and value["bytes"] > 0,
        "Invalid artifact hash or size",
    )
    return {key: value[key] for key in ("sha256", "bytes")}


def _relative(value):
    require(type(value) is str and value and "\\" not in value, "Use relative publication paths")
    path = Path(value)
    require(
        not path.is_absolute() and ".." not in path.parts and str(path) == value,
        "Publication path escapes its directory",
    )
    return value


def _record(root, entry, name):
    return read(root / _relative(entry["records"][name]))


def _matches(root, path, expected):
    return checked(root, _relative(path), _fingerprint(expected))


def _image(root, path, expected, resolution):
    file = _matches(root, path, expected)
    require(path == f"assets/{expected['sha256']}.png", "Image address differs from its content")
    with Image.open(file) as image:
        require(
            image.format == "PNG" and list(image.size) == resolution,
            "Published image dimensions differ from its recipe",
        )
        image.verify()


def _paint_artifacts(receipt):
    rows = receipt.get("artifacts")
    require(type(rows) is list, "Missing paint artifact records")
    result = {}
    for row in rows:
        _fingerprint(row)
        name = _relative(row.get("path"))
        require(name not in result, "Repeated paint artifact")
        result[name] = row
    require(
        {"initial.png", "recipe.json", "final-state.npy", "linear.npy", "inputs/source.orbit"}
        <= result.keys(),
        "Missing required paint provenance",
    )
    return result


def _bundle(bundle):
    require(
        bundle.get("complete") is True
        and bundle.get("schema_version") == 1
        and bundle.get("identity_sha256") == _sha(bundle["request"]),
        "Incomplete or inconsistent material bundle",
    )
    _fingerprint(bundle["bundle"])
    require(bundle["source"] == bundle["request"]["source"], "Bundle source differs")
    require(
        bundle["history"]["source_fractions"] == [1.0]
        and bundle["request"]["parameters"]["history_fractions"] == []
        and bundle["request"]["inputs"]["history"] == [],
        "Photograph requires the complete final material state",
    )
    for key, value in bundle["request"]["geometry"].items():
        require(bundle[key] == value, "Bundle geometry metadata differs")
    return bundle["request"]["inputs"]


def _photo(root, entry, bundle, expected):
    request, receipt = (_record(root, entry, k) for k in ("photo_request", "photo_receipt"))
    require(
        receipt.get("complete") is True and receipt.get("identity_sha256") == _sha(request),
        "Incomplete photograph or request identity differs",
    )
    require(
        request["recipe"] == expected and _record(root, entry, "photo_recipe") == expected,
        "Photograph differs from the canonical study recipe",
    )
    require(
        request["bundle_sha256"] == bundle["bundle"]["sha256"]
        and request["bundle_manifest_sha256"]
        == artifact(root / entry["records"]["bundle"])["sha256"],
        "Photograph uses a different material bundle",
    )
    require(
        receipt["source"] == bundle["source"]
        and receipt["history_fractions"] == [1.0]
        and receipt["motion_frames"] == 1
        and type(request["motion"]["frames"]) is int
        and request["motion"]["frames"] == 1
        and request["motion"]["source_fraction"] == 1.0,
        "Photograph source interval differs",
    )
    artifacts = receipt["artifacts"]
    require(
        {"render.png", "recipe.json", "camera.json", "scene.blend"} <= artifacts.keys()
        and any(name.endswith(".exr") for name in artifacts),
        "Missing photograph provenance",
    )
    for name in artifacts:
        _relative(name)
        _fingerprint(artifacts[name])
    for key, name in (("photo_recipe", "recipe.json"), ("photo_camera", "camera.json")):
        _matches(root, entry["records"][key], artifacts[name])
    _image(root, entry["image"], artifacts["render.png"], expected["render"]["resolution"])
    return request, receipt


def _row(seed, variant, label, description, entry, paint, photograph, final_hash):
    return {
        "seed": seed,
        "variant": variant,
        "label": label,
        "description": description,
        "features": {},
        "image": entry["image"],
        "preview": entry["image"],
        "initial": entry["initial"],
        "film": None,
        "film_resolution": None,
        "film_fps": None,
        "film_frames": 0,
        "resolution": photograph["render"]["resolution"],
        "initial_resolution": paint["recipe"]["render"]["resolution"],
        "source_sha256": paint["source"]["sha256"],
        "physical_state_sha256": final_hash,
        "request": entry["records"]["photo_request"],
        "settings": {
            "paint": paint["recipe"],
            "photograph": photograph,
            "source_interval": "Complete original recording; final source fraction 1.0",
            "relief": "Authored relief from final pigment concentrations, not simulated 3D paint",
        },
    }


def _experiment(root, entry, case_id, request, *, plan=None):
    experiment_key, result_key = (
        ("photo_experiment", "photo_result") if plan is not None else ("experiment", "result")
    )
    experiment, result = (_record(root, entry, key) for key in (experiment_key, result_key))
    require(
        result.get("complete") is True
        and result["case"] == case_id
        and result["identity_sha256"] == _sha(experiment)
        and result["render_receipt_sha256"]
        == artifact(root / entry["records"]["photo_receipt"])["sha256"],
        "Incomplete or inconsistent photograph experiment",
    )
    files = experiment["files"]
    require(
        files["bundle/bundle.npz"]["sha256"] == request["bundle_sha256"]
        and files["bundle/manifest.json"]["sha256"] == request["bundle_manifest_sha256"]
        and request["renderer"]
        == {key: files[key]["sha256"] for key in ("render.py", "materials.py")}
        and experiment["motion_frames"] == request["motion"]["frames"]
        and experiment["fps"] == request["motion"]["fps"],
        "Photograph experiment input binding differs",
    )
    require(
        _record(root, entry, "photo_input_recipe") == request["recipe"]
        and artifact(root / entry["records"]["photo_input_recipe"])["sha256"]
        == files[f"recipes/{case_id}.json"]["sha256"],
        "Photograph differs from its archived experiment recipe",
    )
    if plan is not None:
        require(
            experiment["blender"] == {key: plan["blender"][key] for key in ("path", "sha256")},
            "Photograph Blender binary differs from its study plan",
        )


def _material(root, entry):
    require(
        set(entry) == {"kind", "records", "image", "initial"}
        and set(entry["records"]) == set(MATERIAL_RECORDS),
        "Invalid material entry",
    )
    study, request, receipt, bundle = (
        _record(root, entry, key) for key in ("study", "paint_request", "paint_receipt", "bundle")
    )
    require(
        study.get("version") == STUDY_VERSION and study.get("complete") is True,
        "Incomplete filament study",
    )
    seed, variant, proof = study["seed"], study["variant"], study["proof"]
    require(
        type(seed) is str
        and re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]{0,63})", seed)
        and type(variant) is str
        and variant in VARIANTS
        and type(proof) is bool,
        "Invalid study seed or variant",
    )
    plan = _record(root, entry, "batch_plan")
    planned = _validated_plan(plan)
    case_id = f"{seed}-{variant}"
    require(
        case_id in planned
        and study["plan_identity_sha256"] == plan["identity_sha256"]
        and proof == plan["proof"]
        and study["source_sha256"] == planned[case_id]["source_sha256"],
        "Study differs from its archived batch plan",
    )
    expected = make_paint_recipe(seed, variant)
    require(
        request["recipe"] == expected and _record(root, entry, "paint_recipe") == expected,
        "Paint differs from the canonical study recipe",
    )
    identity, source = _sha(request), request["source"]
    require(
        receipt.get("complete") is True
        and receipt.get("identity_sha256") == identity
        and study["paint_identity_sha256"] == identity,
        "Paint request identity differs",
    )
    require(
        receipt["source"] == source
        and source["seed"] == seed
        and study["source_sha256"] == source["sha256"],
        "Paint source differs",
    )
    require(
        type(source["samples"]) is int
        and source["samples"] > 1
        and type(source["source_first_step"]) is int
        and source["source_first_step"] == 0
        and type(source["source_last_step"]) is int
        and source["source_last_step"] == source["samples"] - 1
        and receipt["final_step"] == expected["simulation"]["steps"]
        and receipt["source_fraction"] == 1.0,
        "Paint does not cover the full source interval",
    )
    artifacts = _paint_artifacts(receipt)
    require(
        artifacts["inputs/source.orbit"]["sha256"] == source["sha256"],
        "Recorded source artifact differs",
    )
    _matches(root, entry["records"]["paint_recipe"], artifacts["recipe.json"])
    _image(root, entry["initial"], artifacts["initial.png"], expected["render"]["resolution"])
    inputs = _bundle(bundle)
    require(
        inputs["render_identity"] == identity
        and inputs["request_sha256"] == artifact(root / entry["records"]["paint_request"])["sha256"]
        and inputs["receipt_sha256"] == artifact(root / entry["records"]["paint_receipt"])["sha256"]
        and bundle["source"] == source,
        "Bundle belongs to different paint records",
    )
    for name in ("final-state.npy", "linear.npy", "recipe.json", "inputs/source.orbit"):
        require(inputs["artifacts"][name] == artifacts[name], "Bundle input artifact differs")
    require(
        bundle["domain_scale"] == expected["simulation"]["domain_scale"]
        and bundle["view_aspect"]
        == expected["simulation"]["resolution"][0] / expected["simulation"]["resolution"][1],
        "Material bundle uses a different paint domain",
    )
    photograph = make_depth_recipe(seed, VARIANTS[variant].label, proof=proof)
    photo_request, photo_receipt = _photo(root, entry, bundle, photograph)
    _experiment(root, entry, "00-painting", photo_request, plan=plan)
    require(
        study["photo_identity_sha256"] == photo_receipt["identity_sha256"],
        "Study photograph identity differs",
    )
    row = _row(
        seed,
        variant,
        VARIANTS[variant].label,
        VARIANTS[variant].description,
        entry,
        request,
        photograph,
        artifacts["final-state.npy"]["sha256"],
    )
    return row, {"entry": entry, "request": request, "bundle": bundle, "artifacts": artifacts}


def _lighting(root, entry, control):
    require(
        set(entry) == {"kind", "variant", "records", "image", "initial"}
        and set(entry["records"]) == LIGHT_RECORD_KEYS
        and entry["variant"] in LIGHTING,
        "Invalid lighting entry",
    )
    variant = entry["variant"]
    case_id, label, parameter, value = LIGHTING[variant]
    bundle = _record(root, entry, "bundle")
    inputs = _bundle(bundle)
    original = control["bundle"]
    require(
        bundle["bundle"] == original["bundle"]
        and bundle["source"] == original["source"]
        and bundle["request"]["optics_sha256"] == original["request"]["optics_sha256"]
        and bundle["request"]["geometry"] == original["request"]["geometry"]
        and inputs["artifacts"]["final-state.npy"] == control["artifacts"]["final-state.npy"],
        "Lighting material/source/optics differs from Control",
    )
    require(
        entry["initial"] == control["entry"]["initial"],
        "Lighting starting view differs from Control",
    )
    expected = make_depth_recipe(REFERENCE_SEED, label, proof=True)
    if parameter == "relief_mm":
        expected[parameter] = value
    else:
        expected["lighting"][parameter] = value
    request, _receipt = _photo(root, entry, bundle, expected)
    _experiment(root, entry, case_id, request)
    return _row(
        REFERENCE_SEED,
        variant,
        label,
        "The same Control paint, photographed with "
        + (f"{value} mm authored relief." if parameter == "relief_mm" else "a lower key light."),
        entry,
        control["request"],
        expected,
        control["artifacts"]["final-state.npy"]["sha256"],
    )


def _manifest(root, entries, title, picks):
    rows, controls, sources = [], {}, {}
    for entry in entries:
        require(
            type(entry) is dict and entry.get("kind") in {"material", "lighting"},
            "Unknown gallery entry",
        )
        if entry["kind"] == "lighting":
            continue
        row, context = _material(root, entry)
        rows.append(row)
        seed = row["seed"]
        require(
            seed not in sources or sources[seed] == context["request"]["source"],
            "Same-seed studies use different source projections",
        )
        sources[seed] = context["request"]["source"]
        if row["variant"] == "control":
            controls[seed] = context
    require(rows and set(controls) == set(sources), "Every seed requires its Control baseline")
    for entry in entries:
        if entry["kind"] == "lighting":
            require(REFERENCE_SEED in controls, "Lighting comparison requires the BC53 Control")
            rows.append(_lighting(root, entry, controls[REFERENCE_SEED]))
    keys = [(row["seed"], row["variant"]) for row in rows]
    require(len(set(keys)) == len(keys), "Repeated gallery seed/variant")
    seeds = list(sources)
    order = list(VARIANTS) + list(LIGHTING)
    rows.sort(key=lambda row: (seeds.index(row["seed"]), order.index(row["variant"])))
    require(type(picks) is list, "Visual picks must be a list")
    seen = set()
    for pick in picks:
        require(
            type(pick) is dict and set(pick) == {"seed", "variant", "note"}, "Invalid visual pick"
        )
        require(
            type(pick["seed"]) is str and type(pick["variant"]) is str,
            "Invalid visual pick identity",
        )
        key = pick["seed"], pick["variant"]
        require(
            key in keys
            and key not in seen
            and type(pick["note"]) is str
            and 0 < len(pick["note"].strip()) <= 700,
            "Unknown, repeated or unexplained visual pick",
        )
        seen.add(key)
    return {
        "version": VERSION,
        "title": title,
        "seeds": seeds,
        "variants": [v for v in order if any(row["variant"] == v for row in rows)],
        "rows": rows,
        "picks": picks,
    }


def _copy_records(output, folder, mapping, prefix):
    records = {}
    for key, name in mapping.items():
        source = folder / name
        target = f"records/{prefix}/{key.replace('_', '-')}.json"
        _copy_verified(source, output / target, artifact(source))
        records[key] = target
    return records


def _copy_image(output, source, expected):
    info = _fingerprint(expected)
    target = f"assets/{info['sha256']}.png"
    _copy_verified(source, output / target, info)
    return target


def _documents(entries):
    return {"index.html", "comparison.json"} | {
        _relative(path)
        for entry in entries
        for path in (*entry["records"].values(), entry["image"], entry["initial"])
    }


def build_review(
    output, cases, *, picks=None, title="The Estuary · Fine folds", lighting_root=None
):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new immutable filament gallery directory")
    cases = [Path(case).resolve(strict=True) for case in cases]
    require(cases and len(set(cases)) == len(cases), "Supply distinct completed studies")
    lighting_root = Path(lighting_root).resolve(strict=True) if lighting_root is not None else None
    roots = cases + ([lighting_root] if lighting_root else [])
    archive_roots = set(roots)
    for root in roots:
        archive_roots.update(
            p
            for p in root.parents
            if any(
                (p / name).is_file()
                for name in ("study.json", "plan.json", "experiment-request.json")
            )
        )
    require(
        all(not output.is_relative_to(root) for root in archive_roots),
        "Publish outside immutable source archives",
    )
    page = document(title, presentation=PRESENTATION)
    verified = [(case, *verify_case(case)) for case in cases]
    require(
        len({(study["seed"], study["variant"]) for _, study, _, _ in verified}) == len(verified),
        "Repeated source study",
    )
    if lighting_root:
        experiment = read(lighting_root / "experiment-request.json")
        for case_id, _, _, _ in LIGHTING.values():
            require(
                finished(lighting_root / case_id, _sha(experiment)), "Incomplete lighting study"
            )
            verify_render(lighting_root / case_id)
    output.mkdir(parents=True)
    entries = []
    for case, study, _, photo_receipt in verified:
        prefix = study["seed"] + "-" + study["variant"]
        records = _copy_records(output, case, MATERIAL_RECORDS, prefix)
        paint_artifacts = _paint_artifacts(read(case / "paint/receipt.json"))
        entries.append(
            {
                "kind": "material",
                "records": records,
                "image": _copy_image(
                    output,
                    case / "photographs/00-painting/render.png",
                    photo_receipt["artifacts"]["render.png"],
                ),
                "initial": _copy_image(
                    output, case / "paint/initial.png", paint_artifacts["initial.png"]
                ),
            }
        )
    if lighting_root:
        controls = [
            entry
            for entry in entries
            if _record(output, entry, "study")["seed"] == REFERENCE_SEED
            and _record(output, entry, "study")["variant"] == "control"
        ]
        require(len(controls) == 1, "Lighting studies require exactly one BC53 Control")
        for variant, (case_id, _, _, _) in LIGHTING.items():
            mapping = {
                "bundle": "inputs/bundle/manifest.json",
                "experiment": "experiment-request.json",
                "result": f"{case_id}/experiment-result.json",
                "photo_input_recipe": f"inputs/recipes/{case_id}.json",
                **{
                    f"photo_{name}": f"{case_id}/{name}.json"
                    for name in ("request", "receipt", "recipe", "camera")
                },
            }
            records = _copy_records(output, lighting_root, mapping, REFERENCE_SEED + "-" + variant)
            receipt = read(lighting_root / case_id / "receipt.json")
            entries.append(
                {
                    "kind": "lighting",
                    "variant": variant,
                    "records": records,
                    "image": _copy_image(
                        output,
                        lighting_root / case_id / "render.png",
                        receipt["artifacts"]["render.png"],
                    ),
                    "initial": controls[0]["initial"],
                }
            )
    data = _manifest(output, entries, title, [] if picks is None else picks)
    write(output / "comparison.json", data)
    (output / "index.html").write_text(page)
    write(
        output / "publication.json",
        {
            "version": VERSION,
            "entries": entries,
            "artifacts": {name: artifact(output / name) for name in sorted(_documents(entries))},
            "comparison_sha256": _sha(data),
        },
    )
    return verify_review(output)


def verify_review(output):
    root = Path(output).resolve()
    publication, data = read(root / "publication.json"), read(root / "comparison.json")
    require(
        type(publication) is dict
        and set(publication) == {"version", "entries", "artifacts", "comparison_sha256"}
        and publication["version"] == VERSION
        and type(publication["entries"]) is list
        and publication["entries"],
        "Incomplete filament publication",
    )
    require(
        set(publication["artifacts"]) == _documents(publication["entries"]),
        "Publication artifact list differs",
    )
    for name, info in publication["artifacts"].items():
        _matches(root, name, info)
    expected = _manifest(root, publication["entries"], data["title"], data["picks"])
    require(
        encoded(data) == encoded(expected) and publication["comparison_sha256"] == _sha(data),
        "Comparison differs from its copied records",
    )
    require(
        (root / "index.html").read_text() == document(data["title"], presentation=PRESENTATION),
        "Review page differs from the shared template",
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+")
    parser.add_argument("--lighting-root", type=Path)
    parser.add_argument("--picks", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        data = verify_review(args.output)
    else:
        require(args.cases, "Supply completed filament cases")
        data = build_review(
            args.output,
            args.cases,
            picks=read(args.picks) if args.picks else [],
            lighting_root=args.lighting_root,
        )
    print(
        json.dumps({"verified": True, "paintings": len(data["rows"]), "seeds": len(data["seeds"])})
    )


if __name__ == "__main__":
    main()
