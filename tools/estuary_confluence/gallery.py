#!/usr/bin/env python3
"""Publish complete Confluence films, paintings, palettes, and fair comparisons.

One physical archive can supply several optical views. Published media are content
addressed; copied request/receipt/palette/event records retain their association
after the original experiment directories move. Comparison relationships are
derived from the archived source, pigment count, and physical-state identity.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
from pathlib import Path

import numpy as np
from PIL import Image

from tools.estuary.optics import linear_to_srgb, srgb_to_linear
from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.gallery import FILM_CAPTION, _copy_verified, _json_artifact

from .palette import SUPPORTED_CHROMATIC_COUNTS, normalize_seed
from .run import INTERACTION_LOOKS, interaction_metadata, surface_configs, verify_run

EARLIER_CAPTION = "Earlier version · same trajectory"
PUBLICATION_VERSION = 2

GROUPS = {
    "layered": "Layered",
    "homogeneous": "Blended",
    "control": "Control",
    "silk": "Satin seams",
    "silk-grain": "Satin + grain",
}
INTERACTION_PUBLIC_FIELDS = frozenset({"interaction_version", "base_material_sha256"})
BODY_MARKER_PUBLIC_FIELDS = frozenset({"body_markers", "body_marker_record"})
TEMPLATE = Path(__file__).with_suffix(".html")
TITLE_TOKEN = "__CONFLUENCE_TITLE_HTML__"


def thumbnail_pixels(path):
    """Derive a bounded review preview from the portable poster in linear light.

    Pillow decodes RGB16 as RGB8 here; this affects only the review preview.
    Full-resolution RGB16 originals remain unchanged and downloadable.
    """
    with Image.open(path) as source:
        width, height = source.size
        size = tuple(max(1, round(n * min(1, 960 / max(width, height)))) for n in source.size)
        linear = srgb_to_linear(np.asarray(source.convert("RGB"), dtype="f4") / 255)
    channels = [
        np.asarray(Image.fromarray(linear[..., i].astype("f4")).resize(size, Image.Resampling.BOX))
        for i in range(3)
    ]
    return np.rint(linear_to_srgb(np.stack(channels, axis=-1)) * 255).astype("u1")


def document(title, layout="classic"):
    """Fill only the escaped title in Confluence's independent HTML template."""
    require(type(title) is str and 0 < len(title) <= 120, "Gallery title must be short text")
    require(layout in ("classic", "films", "studies"), "Unknown gallery layout")
    template = (
        TEMPLATE if layout == "classic" else TEMPLATE.with_name("film_review.html")
    ).read_text(encoding="utf-8")
    require(template.count(TITLE_TOKEN) == 2, "Gallery title placeholders differ")
    return template.replace(TITLE_TOKEN, html.escape(title))


def _film_metadata(request, receipt, look):
    names = ("film_caption", "film_resolution", "film_frames", "film_fps", "film_seconds")
    if request["mode"] == "still":
        return dict.fromkeys(names)
    render, movie = request["recipe"]["render"], receipt["looks"][look]["movie"]
    frames = render["formation_frames"] + render["hold_frames"] + render["orbit_frames"] - 1
    require(
        movie["full_decode_verified"] is True,
        "Film needs complete decode evidence",
    )
    require(
        movie["frames"] == frames
        and movie["fps"] == render["fps"]
        and movie["resolution"] == render["resolution"],
        "Film metadata differs from its complete timeline",
    )
    return {
        "film_caption": (
            FILM_CAPTION if render["orbit_frames"] > 1 else "Complete formation of the painting"
        ),
        "film_resolution": movie["resolution"],
        "film_frames": movie["frames"],
        "film_fps": movie["fps"],
        "film_seconds": movie["frames"] / movie["fps"],
    }


def _study_metadata(request, receipt, look):
    recipe = request["recipe"]
    render = recipe["render"]
    metadata = {
        "optics_model": recipe.get("surface", {}).get("optics_model", "rgb"),
        "initial_pattern": recipe["simulation"].get("initial_pattern", "pools"),
        "formation_seconds": (
            (render["formation_frames"] - 1) / render["fps"] if request["mode"] == "film" else None
        ),
        "image_balance": receipt["looks"][look].get("image_balance"),
        "solver_diagnostics": receipt.get("solver_diagnostics"),
    }
    if "material_model" in recipe["simulation"]:
        metadata["material_model"] = recipe["simulation"]["material_model"]
    metadata.update(_interaction_metadata(request, receipt))
    metadata.update(_body_marker_metadata(request, receipt))
    return metadata


def _interaction_metadata(request, receipt):
    """Bind portable interaction claims to their verified source provenance.

    Publishing verifies the complete physical archive first. Portable review
    retains its request/receipt association, without claiming to recompute the
    raw material hash from image files.
    """
    recipe = request["recipe"]
    if recipe["simulation"].get("interaction") is None:
        require(
            "interaction" not in request
            and "interaction" not in receipt
            and "base_material_sha256" not in receipt,
            "Disabled interaction cannot publish microstructure metadata",
        )
        return {}
    expected = interaction_metadata(recipe, request["source"]["seed"])
    require(
        request.get("interaction") == expected and receipt.get("interaction") == expected,
        "Published interaction version, seed or settings differ from its material",
    )
    require(
        request.get("surface_configs") == surface_configs(recipe),
        "Published interaction optical views differ from their recipe",
    )
    base_hash = receipt.get("base_material_sha256")
    require(
        type(base_hash) is str
        and len(base_hash) == 64
        and all(character in "0123456789abcdef" for character in base_hash),
        "Published interaction needs its original-material identity",
    )
    return {"interaction_version": expected["version"], "base_material_sha256": base_hash}


def _body_marker_metadata(request, receipt):
    """Publish position-guide claims only when both immutable records bind them."""
    from .run import body_marker_metadata

    expected = body_marker_metadata(request["recipe"], request["source"])
    if expected is None:
        require(
            "body_markers" not in request
            and "body_markers" not in receipt
            and "body-markers.json" not in receipt["artifacts"],
            "Unannotated material cannot publish body-marker metadata",
        )
        return {}
    require(
        request.get("body_markers") == expected and receipt.get("body_markers") == expected,
        "Published body-marker metadata differs from its case",
    )
    require(
        "body-markers.json" in receipt["artifacts"],
        "Annotated material requires its bound body-position record",
    )
    return {"body_markers": expected}


def _require_view_layout(recipe, layout):
    require(
        not recipe.get("render", {}).get("body_markers") or layout in ("films", "studies"),
        "Body-position guides require a film review or studies gallery layout",
    )
    require(
        layout == "studies" or not INTERACTION_LOOKS.intersection(recipe["looks"]),
        "Interaction comparison views require the studies gallery layout",
    )
    require(all(look in GROUPS for look in recipe["looks"]), "Unknown optical view")
    if INTERACTION_LOOKS.intersection(recipe["looks"]):
        # Do not infer named comparison controls from a label or a color count.
        surface_configs(recipe)


def _study_key(study):
    return (
        study["seed"],
        study["chromatic_count"],
        study.get("palette_mode", "curated"),
        study["group"],
    )


def _experiment_metadata(request, look):
    """Bind a named experiment collection to its archived recipe and optical view."""
    recipe = request["recipe"]
    name = recipe["name"]
    require(type(name) is str and 0 < len(name) <= 100, "Experiment needs a short recipe name")
    key = {
        "name": name,
        "chromatic_count": recipe["chromatic_count"],
        "palette_mode": recipe.get("palette_mode", "curated"),
        "look": look,
    }
    return {
        "experiment_label": name,
        "collection_label": f"{name} · {GROUPS[look]}",
        "collection_key": hashlib.sha256(encoded(key)).hexdigest(),
    }


def _experiment_comparisons(studies):
    """Pair only optical interpretations of the very same physical experiment.

    Changed flow, count, or loading is a separate experiment. Matching a seed
    never justifies calling those cases the same material history.
    """
    views, collections, sources = {}, set(), {}
    for study in studies:
        key = (study["seed"], study["collection_key"])
        require(key not in collections, "Ambiguous duplicate seed within an experiment collection")
        collections.add(key)
        view_key = (study["case_id"], study["group"])
        require(view_key not in views, "Duplicate optical experiment view")
        views[view_key] = study
        sources.setdefault(study["seed"], study["source_sha256"])
        require(
            sources[study["seed"]] == study["source_sha256"],
            "Experiment seed refers to different source trajectories",
        )
    result = {}
    for study in studies:
        look = study["group"]
        if look == "control":
            other = "silk-grain" if (study["case_id"], "silk-grain") in views else "silk"
        elif look in ("silk", "silk-grain"):
            other = "control"
        else:
            other = "homogeneous" if look == "layered" else "layered"
        target = views.get((study["case_id"], other))
        if target is not None:
            require(
                target["physical_state_sha256"] == study["physical_state_sha256"],
                "Optical experiment views use different physical states",
            )
        result[study["id"]] = (
            (target["id"], target["image"], f"{GROUPS[other]} · same material history")
            if target is not None
            else (None, None, None)
        )
    return result


def _verify_earlier(output, files, record):
    """Validate a portable earlier image against its immutable case records."""
    require(
        all(record[key] in files for key in ("request", "receipt", "image")),
        "Earlier version is missing its provenance or image",
    )
    request = read(output / record["request"])
    receipt = read(output / record["receipt"])
    identity = hashlib.sha256(encoded(request)).hexdigest()
    source, recipe = request["source"], request["recipe"]
    look = record["group"]
    require(
        identity == record["identity_sha256"] == receipt["identity_sha256"]
        and receipt["complete"] is True
        and record["case_id"] == identity[:16],
        "Earlier version case identity differs",
    )
    require(
        record["seed"] == normalize_seed(source["seed"])
        and record["source_sha256"] == source["sha256"]
        and receipt["source"] == source
        and record["chromatic_count"] == recipe["chromatic_count"]
        and record["palette_mode"] == recipe.get("palette_mode", "curated")
        and look in recipe["looks"],
        "Earlier version source or study association differs",
    )
    require(
        receipt["source_fraction"] == 1.0
        and receipt["final_step"] == recipe["simulation"]["steps"]
        and files[record["image"]] == receipt["artifacts"][f"{look}/poster.png"],
        "Earlier version image or completed trajectory differs",
    )
    return record


def _name(palette):
    label = {
        "harmonic": "Color harmony",
        "random": "Independent colors",
        "composed": "Composed colors",
    }.get(palette.get("mode"), palette["family"].replace("-", " ").title())
    count = palette["chromatic_count"]
    return f"{label} · {count} color{'s' if count != 1 else ''}"


def _swatches(palette, *, include_chalk=True):
    count = palette["chromatic_count"] + 1
    require(
        len(palette["pigments_srgb"])
        == len(palette["pigment_names"])
        == len(palette["pigment_roles"])
        == count,
        "Palette swatch dimensions differ",
    )
    result = []
    for name, role, color in zip(
        palette["pigment_names"], palette["pigment_roles"], palette["pigments_srgb"], strict=True
    ):
        require(
            type(name) is str
            and type(role) is str
            and len(color) == 3
            and all(type(v) in (int, float) and 0 <= v <= 1 for v in color),
            "Invalid palette swatch",
        )
        result.append({"name": name, "role": role, "rgba": [*color, 1.0]})
    return result if include_chalk else [s for s in result if s["role"] != "chalk"]


def _comparison_inputs(first, second, *, count_change):
    """Check the archived experiment inputs behind a controlled comparison.

    Legacy count comparisons retain their old caption and do not invent pool
    geometry. New scattered comparisons must preserve the common pool prefix.
    Palette comparisons may change colors but preserve every material control.
    """
    layouts = [request.get("layout") for request in (first, second)]
    if count_change and layouts == [None, None]:
        return False
    require(
        (layouts[0] is None) == (layouts[1] is None),
        "Comparison mixes scattered and legacy initial conditions",
    )
    for key in ("simulation", "projection", "surface"):
        require(
            first["recipe"].get(key) == second["recipe"].get(key),
            f"Comparison {key} controls differ",
        )
    for key in ("events", "code"):
        require(first.get(key) == second.get(key), f"Comparison {key} differ")
    if count_change:
        counts = [request["recipe"]["chromatic_count"] for request in (first, second)]
        require(sorted(counts) == [3, 5], "Count comparison must use three and five colors")
        require(
            all(
                request["recipe"]["simulation"].get("deposition") == 0
                for request in (first, second)
            ),
            "Starting-pool comparison requires zero continuing deposition",
        )
        require(
            {key: value for key, value in layouts[0].items() if key not in ("count", "pools")}
            == {key: value for key, value in layouts[1].items() if key not in ("count", "pools")},
            "Comparison starting layout controls differ",
        )
        require(
            layouts[0]["pools"][:3] == layouts[1]["pools"][:3],
            "Comparison first three starting pools differ",
        )
    else:
        require(layouts[0] == layouts[1], "Palette comparison starting layouts differ")
    a, b = first["palette"], second["palette"]
    for key in (
        "substrate_srgb",
        "substrate_seed",
        "body_weights",
        "underpaint_index",
        "version",
        "physical_version",
    ):
        require(a.get(key) == b.get(key), f"Comparison palette {key} differs")
    if layouts[0] is not None:
        require(a["substrate_srgb"] == [1, 1, 1], "Scattered comparison requires a white ground")
    for key in ("scattering", "settling", "release", "specific_volumes", "granulation"):
        values_a, values_b = a[key], b[key]
        if count_change:
            values_a, values_b = [*values_a[:3], values_a[-1]], [*values_b[:3], values_b[-1]]
        require(values_a == values_b, f"Comparison pigment {key} differs")
    if count_change:
        require(
            a["pigments_srgb"][:3] == b["pigments_srgb"][:3]
            and a["pigments_srgb"][-1] == b["pigments_srgb"][-1],
            "Comparison primary pigment colors differ",
        )
    else:
        require(a["body_mixtures"] == b["body_mixtures"], "Palette comparison body mixtures differ")
    return layouts[0] is not None


def _comparisons(studies, case_requests):
    """Derive unambiguous same-source comparisons and reject false pairings."""
    indexed, source_by_seed = {}, {}
    for study in studies:
        key = (
            study["seed"],
            study.get("palette_mode", "curated"),
            study["chromatic_count"],
            study["group"],
        )
        require(key not in indexed, "Ambiguous duplicate seed, pigment count, and optical view")
        indexed[key] = study
        source_by_seed.setdefault(study["seed"], study["source_sha256"])
        require(
            source_by_seed[study["seed"]] == study["source_sha256"],
            "Comparison seed refers to different source trajectories",
        )
    result = {}
    for study in studies:
        seed, count, look = study["seed"], study["chromatic_count"], study["group"]
        mode = study.get("palette_mode", "curated")
        if mode == "composed" or (
            mode == "harmonic" and (seed, "composed", count, look) in indexed
        ):
            target_mode = "harmonic" if mode == "composed" else "composed"
            target = indexed.get((seed, target_mode, count, look))
            caption = "Earlier painting" if mode == "composed" else "New painting"
            caption += " · same trajectory, different colors and material"
            if target:
                a = case_requests[study["case_id"]]["recipe"]
                b = case_requests[target["case_id"]]["recipe"]
                require(
                    a.get("projection") == b.get("projection")
                    and a["simulation"]["resolution"][0] * b["simulation"]["resolution"][1]
                    == b["simulation"]["resolution"][0] * a["simulation"]["resolution"][1],
                    "Material comparison uses a different trajectory projection",
                )
        elif mode == "random":
            target = indexed.get((seed, "harmonic", count, look))
            caption = "Seeded harmony · same starting pools and motion"
            if target:
                require(
                    study["physical_state_sha256"] == target["physical_state_sha256"],
                    "Palette comparison uses different physical states",
                )
                _comparison_inputs(
                    case_requests[study["case_id"]],
                    case_requests[target["case_id"]],
                    count_change=False,
                )
        elif count == 5 and look == "layered":
            target = indexed.get((seed, mode, 3, "layered"))
            caption = "Three colors · layered"
            if target and _comparison_inputs(
                case_requests[study["case_id"]], case_requests[target["case_id"]], count_change=True
            ):
                caption = "Three colors · three starting pools"
        else:
            target = indexed.get(
                (seed, mode, count, "homogeneous" if look == "layered" else "layered")
            )
            caption = ("Blended" if look == "layered" else "Layered") + " · same material history"
            if target:
                require(
                    study["physical_state_sha256"] == target["physical_state_sha256"],
                    "Optical comparison uses different physical states",
                )
            if target is None and mode == "harmonic" and count == 3:
                target = indexed.get((seed, mode, 5, look))
                caption = "Five colors · layered"
                if target and _comparison_inputs(
                    case_requests[study["case_id"]],
                    case_requests[target["case_id"]],
                    count_change=True,
                ):
                    caption = "Five colors · two additional starting pools"
        result[study["id"]] = (
            (target["id"], target["image"], caption) if target else (None, None, None)
        )
    return result


def build_gallery(
    output,
    cases,
    *,
    title="Confluence Fresco",
    allow_stills=False,
    earlier_gallery=None,
    layout="classic",
):
    """Publish all selected complete cases; each seed retains its CLI order."""
    require(type(title) is str and 0 < len(title) <= 120, "Gallery title must be short text")
    require(type(allow_stills) is bool, "allow_stills must be boolean")
    require(
        layout != "studies" or earlier_gallery is None,
        "Named experiments use their own optical comparisons, without an earlier gallery",
    )
    page = document(title, layout)
    cases = [Path(case).resolve(strict=True) for case in cases]
    require(bool(cases), "Choose at least one completed confluence")
    records, identities, prefixes, seeds = [], set(), set(), []
    for case in cases:
        request, receipt = verify_run(case)
        _require_view_layout(request["recipe"], layout)
        require(
            layout != "films"
            or (
                request["recipe"]["chromatic_count"] == 5
                and request["recipe"]["looks"] == ["layered"]
            ),
            "Film review requires five-color paintings with the layered optical view",
        )
        identity = receipt["identity_sha256"]
        require(identity not in identities, "A physical case was selected more than once")
        require(identity[:16] not in prefixes, "Truncated case identity collision")
        identities.add(identity)
        prefixes.add(identity[:16])
        require(allow_stills or request["mode"] == "film", "Every selected painting needs its film")
        seed = normalize_seed(request["source"]["seed"])
        if seed not in seeds:
            seeds.append(seed)
        records.append((case, request, receipt))
    earlier_collection = earlier_curation = None
    if earlier_gallery is not None:
        earlier_gallery = Path(earlier_gallery).resolve(strict=True)
        earlier_collection, earlier_curation = verify_gallery(earlier_gallery)
    output = Path(output).resolve()
    require(
        earlier_gallery is None
        or (output != earlier_gallery and not output.is_relative_to(earlier_gallery)),
        "Publish outside the earlier gallery",
    )
    require(
        all(output != case and not output.is_relative_to(case) for case in cases),
        "Publish outside immutable source archives",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        files, studies, origins, earlier_sources = {}, [], [], []

        def copy(source, name, expected):
            _copy_verified(source, output / name, expected)
            files[name] = expected
            return name

        for case, request, receipt in records:
            identity, palette = receipt["identity_sha256"], request["palette"]
            case_id, seed = identity[:16], normalize_seed(request["source"]["seed"])
            paths = {}
            for name, value in (
                ("request", request),
                ("receipt", receipt),
                ("palette", palette),
                ("events", request["events"]),
            ):
                paths[name] = copy(
                    case / f"{name}.json", f"records/{case_id}/{name}.json", _json_artifact(value)
                )
            if request.get("layout") is not None:
                paths["layout"] = copy(
                    case / "layout.json",
                    f"records/{case_id}/layout.json",
                    _json_artifact(request["layout"]),
                )
            if request.get("spectral") is not None:
                expected = _json_artifact(request["spectral"])
                require(
                    receipt["artifacts"].get("spectral.json") == expected,
                    "Spectral material is not bound to its case receipt",
                )
                paths["spectral"] = copy(
                    case / "spectral.json", f"records/{case_id}/spectral.json", expected
                )
            if "background" in request["recipe"]:
                expected = _json_artifact(request["background"])
                require(
                    receipt["artifacts"].get("background.json") == expected,
                    "Seeded background is not bound to its case receipt",
                )
                paths["background"] = copy(
                    case / "background.json", f"records/{case_id}/background.json", expected
                )
            if "assessment.json" in receipt["artifacts"]:
                paths["assessment"] = copy(
                    case / "assessment.json",
                    f"records/{case_id}/assessment.json",
                    receipt["artifacts"]["assessment.json"],
                )
            if request["recipe"]["simulation"].get("mass_budget_interval_steps", 0) > 0:
                require(
                    "mass-budget.json" in receipt["artifacts"],
                    "Enabled mass restoration requires a bound mass-budget report",
                )
                paths["mass_budget"] = copy(
                    case / "mass-budget.json",
                    f"records/{case_id}/mass-budget.json",
                    receipt["artifacts"]["mass-budget.json"],
                )
            marker_metadata = _body_marker_metadata(request, receipt)
            if marker_metadata:
                from .run import validate_body_marker_records

                marker_records = read(case / "body-markers.json")
                validate_body_marker_records(request, receipt, marker_records)
                paths["body_markers"] = copy(
                    case / "body-markers.json",
                    f"records/{case_id}/body-markers.json",
                    receipt["artifacts"]["body-markers.json"],
                )
            origins.append(
                {
                    "id": case_id,
                    "identity_sha256": identity,
                    "case": str(case),
                    "source_sha256": request["source"]["sha256"],
                    **paths,
                }
            )
            for look in request["recipe"]["looks"]:
                require(look in GROUPS, "Unknown optical view")
                film_metadata = _film_metadata(request, receipt, look)
                poster_info = receipt["artifacts"][f"{look}/poster.png"]
                poster = copy(
                    case / look / "poster.png",
                    f"assets/{poster_info['sha256']}/poster.png",
                    poster_info,
                )
                film = None
                initial = None
                if request.get("layout") is not None:
                    info = receipt["artifacts"][f"{look}/initial.png"]
                    initial = copy(
                        case / look / "initial.png", f"assets/{info['sha256']}/initial.png", info
                    )
                if request["mode"] == "film":
                    info = receipt["artifacts"][f"{look}/film.mp4"]
                    film = copy(case / look / "film.mp4", f"assets/{info['sha256']}/film.mp4", info)
                studies.append(
                    {
                        "id": f"{case_id}-{look}",
                        "case_id": case_id,
                        "identity_sha256": identity,
                        "physical_state_sha256": receipt["looks"][look]["physical_state_sha256"],
                        "name": _name(palette),
                        "group": look,
                        "seed": seed,
                        "chromatic_count": palette["chromatic_count"],
                        "palette_mode": request["recipe"].get("palette_mode", "curated"),
                        "image": poster,
                        "initial": initial,
                        "film": film,
                        "source_sha256": request["source"]["sha256"],
                        "palette_identity_sha256": palette["identity_sha256"],
                        "palette_record": paths["palette"],
                        "layout_record": paths.get("layout"),
                        "spectral_record": paths.get("spectral"),
                        "assessment_record": paths.get("assessment"),
                        "mass_budget_record": paths.get("mass_budget"),
                        "background_record": paths.get("background"),
                        **_study_metadata(request, receipt, look),
                        "swatches": _swatches(palette, include_chalk=initial is None),
                        **(
                            {"body_marker_record": paths["body_markers"]} if marker_metadata else {}
                        ),
                        "resolution": request["recipe"]["render"]["still_resolution"],
                        **film_metadata,
                    }
                )
                if layout == "studies":
                    studies[-1].update(_experiment_metadata(request, look))
                if layout in ("films", "studies"):
                    preview = output / ".preview.png"
                    Image.fromarray(thumbnail_pixels(output / poster)).save(preview)
                    info = artifact(preview)
                    studies[-1]["preview"] = copy(
                        preview, f"assets/{info['sha256']}/preview.png", info
                    )
                    preview.unlink()
        if layout != "studies":
            rank = {
                (count, look): index
                for index, (count, look) in enumerate(
                    (count, look)
                    for look in ("layered", "homogeneous")
                    for count in SUPPORTED_CHROMATIC_COUNTS
                )
            }
            studies.sort(
                key=lambda study: (
                    seeds.index(study["seed"]),
                    study["palette_mode"] == "random",
                    rank[(study["chromatic_count"], study["group"])],
                )
            )
        case_requests = {
            receipt["identity_sha256"][:16]: request for _, request, receipt in records
        }
        comparisons = (
            _experiment_comparisons(studies)
            if layout == "studies"
            else _comparisons(studies, case_requests)
        )
        for study in studies:
            study["comparison_id"], study["baseline"], study["comparison_caption"] = comparisons[
                study["id"]
            ]
        earlier_index = {}
        if earlier_collection is not None:
            for candidate in earlier_collection["studies"]:
                key = _study_key(candidate)
                require(key not in earlier_index, "Ambiguous earlier version match")
                earlier_index[key] = candidate
            earlier_origins = {origin["id"]: origin for origin in earlier_curation["sources"]}
        for study in studies:
            study["earlier_id"] = study["earlier_image"] = study["earlier_caption"] = None
            candidate = earlier_index.get(_study_key(study))
            if candidate is None:
                continue
            require(
                study["source_sha256"] == candidate["source_sha256"],
                "Earlier version uses a different source trajectory",
            )
            origin = earlier_origins[candidate["case_id"]]
            image_info = earlier_curation["artifacts"][candidate["image"]]
            image_name = copy(
                earlier_gallery / candidate["image"],
                f"assets/{image_info['sha256']}/earlier.png",
                image_info,
            )
            earlier_id = candidate["id"]
            entry = {
                "id": earlier_id,
                "case_id": candidate["case_id"],
                "identity_sha256": candidate["identity_sha256"],
                "seed": candidate["seed"],
                "chromatic_count": candidate["chromatic_count"],
                "palette_mode": candidate.get("palette_mode", "curated"),
                "group": candidate["group"],
                "source_sha256": candidate["source_sha256"],
                "image": image_name,
            }
            for key in ("request", "receipt"):
                entry[key] = copy(
                    earlier_gallery / origin[key],
                    f"earlier/{candidate['case_id']}/{key}.json",
                    earlier_curation["artifacts"][origin[key]],
                )
            earlier_sources.append(entry)
            study["earlier_id"], study["earlier_image"], study["earlier_caption"] = (
                earlier_id,
                image_name,
                EARLIER_CAPTION,
            )
        collection = {
            "schema_version": 1,
            "publication_version": PUBLICATION_VERSION,
            "title": title,
            "studies": studies,
            "history": "Complete trajectories; finished material stays fixed during camera motion",
        }
        if layout == "studies":
            collection["layout"] = "studies"
        write(output / "collection.json", collection)
        partial = output / "index.html.partial"
        partial.write_text(page)
        partial.replace(output / "index.html")
        for name in ("collection.json", "index.html"):
            files[name] = artifact(output / name)
        write(
            output / "curation.json",
            {
                "schema_version": 1,
                "complete": True,
                "title": title,
                "sources": origins,
                "earlier_sources": earlier_sources,
                "artifacts": files,
                **({"layout": "studies"} if layout == "studies" else {}),
            },
        )
        verify_gallery(output)
    return output / "index.html"


def verify_gallery(output):
    """Verify portable publication against each archived request and receipt."""
    output = Path(output)
    curation = read(output / "curation.json")
    require(
        curation.get("schema_version") == 1 and curation.get("complete") is True,
        "Incomplete gallery curation",
    )
    files = curation["artifacts"]
    require({"collection.json", "index.html"} <= files.keys(), "Missing gallery documents")
    for name, info in files.items():
        checked(output, name, info)
    collection = read(output / "collection.json")
    require(
        collection.get("schema_version") == 1 and collection["title"] == curation["title"],
        "Gallery collection differs from its curation",
    )
    require(
        collection.get("publication_version", 1) in (1, PUBLICATION_VERSION),
        "Unsupported gallery publication",
    )
    layout = collection.get("layout")
    require(
        layout in (None, "studies") and layout == curation.get("layout"),
        "Gallery experiment layout differs from its curation",
    )
    require(
        layout != "studies" or not curation.get("earlier_sources"), "Unbound earlier experiment"
    )
    origins = {record["id"]: record for record in curation["sources"]}
    studies = {study["id"]: study for study in collection["studies"]}
    require(
        len(origins) == len(curation["sources"]) and len(studies) == len(collection["studies"]),
        "Duplicate source or optical-view identity",
    )
    expected_ids, case_requests = set(), {}
    for case_id, origin in origins.items():
        require(
            all(origin[key] in files for key in ("request", "receipt", "palette", "events")),
            "Source design records are not archived",
        )
        request, receipt = read(output / origin["request"]), read(output / origin["receipt"])
        case_requests[case_id] = request
        identity = hashlib.sha256(encoded(request)).hexdigest()
        require(
            identity == receipt["identity_sha256"] == origin["identity_sha256"]
            and case_id == identity[:16]
            and receipt["complete"] is True,
            "Source case identity differs",
        )
        palette, recipe, source = request["palette"], request["recipe"], request["source"]
        _require_view_layout(recipe, layout)
        require(
            read(output / origin["palette"]) == palette
            and read(output / origin["events"]) == request["events"],
            "Source design differs",
        )
        if request.get("layout") is not None:
            require(
                origin.get("layout") in files
                and read(output / origin["layout"]) == request["layout"],
                "Published starting pools differ",
            )
        optics_model = recipe.get("surface", {}).get("optics_model", "rgb")
        require(optics_model in ("rgb", "spectral"), "Unknown published optical model")
        if optics_model == "spectral":
            from .spectral import validate_spectral_material

            require(
                origin.get("spectral") in files and request.get("spectral") is not None,
                "Published spectral material is missing",
            )
            require(
                read(output / origin["spectral"]) == request["spectral"]
                and files[origin["spectral"]] == receipt["artifacts"]["spectral.json"],
                "Published spectral material differs",
            )
            validate_spectral_material(request["spectral"])
            require(
                request["spectral"]["palette_identity_sha256"] == palette["identity_sha256"],
                "Published spectrum belongs to another palette",
            )
        else:
            require(
                origin.get("spectral") is None and request.get("spectral") is None,
                "RGB view cannot advertise spectral material",
            )
        if "background" in recipe:
            from .backgrounds import validate_background

            require(origin.get("background") in files, "Published seeded background is missing")
            background = validate_background(read(output / origin["background"]), palette)
            require(
                background == request.get("background")
                and background["name"] == recipe["background"]
                and background["ground_srgb"] == recipe["surface"]["ground_srgb"]
                and files[origin["background"]] == receipt["artifacts"]["background.json"],
                "Published seeded background differs",
            )
        else:
            require(origin.get("background") is None, "Unbound seeded background")
        if "assessment.json" in receipt["artifacts"]:
            require(
                origin.get("assessment") in files
                and files[origin["assessment"]] == receipt["artifacts"]["assessment.json"],
                "Published assessment differs from its case",
            )
            report = read(output / origin["assessment"])
            require(
                report["settings"] == recipe.get("assessment"),
                "Published assessment settings differ",
            )
        else:
            require(origin.get("assessment") is None, "Unbound assessment report")
        if recipe["simulation"].get("mass_budget_interval_steps", 0) > 0:
            require(
                "mass-budget.json" in receipt["artifacts"]
                and origin.get("mass_budget") in files
                and files[origin["mass_budget"]] == receipt["artifacts"]["mass-budget.json"],
                "Published mass-budget report differs from its case",
            )
        else:
            require(origin.get("mass_budget") is None, "Unbound mass-budget report")
        marker_metadata = _body_marker_metadata(request, receipt)
        if marker_metadata:
            from .run import validate_body_marker_records

            require(
                origin.get("body_markers") in files
                and files[origin["body_markers"]] == receipt["artifacts"]["body-markers.json"],
                "Published body-position record differs from its case",
            )
            validate_body_marker_records(request, receipt, read(output / origin["body_markers"]))
        else:
            require("body_markers" not in origin, "Unbound body-position record")
        require(
            receipt["source"] == source
            and origin["source_sha256"] == source["sha256"]
            and receipt["source_fraction"] == 1
            and receipt["final_step"] == recipe["simulation"]["steps"],
            "Published case does not represent its complete source",
        )
        for look in recipe["looks"]:
            study_id = f"{case_id}-{look}"
            expected_ids.add(study_id)
            require(study_id in studies, "Missing selected optical view")
            study = studies[study_id]
            require(
                study["case_id"] == case_id
                and study["identity_sha256"] == identity
                and study["physical_state_sha256"]
                == receipt["physical_state_sha256"]
                == receipt["looks"][look]["physical_state_sha256"],
                "Physical-state association differs",
            )
            require(
                study["name"] == _name(palette)
                and study["group"] == look
                and study["seed"] == normalize_seed(source["seed"])
                and study["source_sha256"] == source["sha256"]
                and study["chromatic_count"]
                == recipe["chromatic_count"]
                == palette["chromatic_count"]
                and study["palette_identity_sha256"] == palette["identity_sha256"]
                and study["palette_record"] == origin["palette"]
                and study.get("palette_mode", "curated") == recipe.get("palette_mode", "curated")
                and study.get("layout_record") == origin.get("layout")
                and study.get("background_record") == origin.get("background")
                and study["swatches"]
                == _swatches(palette, include_chalk=request.get("layout") is None),
                "Painting caption or palette association differs",
            )
            metadata = _study_metadata(request, receipt, look)
            expected_marker_fields = BODY_MARKER_PUBLIC_FIELDS if marker_metadata else set()
            require(
                BODY_MARKER_PUBLIC_FIELDS.intersection(study) == expected_marker_fields,
                "Unbound or missing published body-marker metadata",
            )
            if marker_metadata:
                require(
                    study["body_marker_record"] == origin["body_markers"]
                    and study["body_markers"] == marker_metadata["body_markers"],
                    "Published body-marker records belong to another case",
                )
            require(
                INTERACTION_PUBLIC_FIELDS.intersection(study)
                == INTERACTION_PUBLIC_FIELDS.intersection(metadata),
                "Unbound or missing published interaction metadata",
            )
            experiment_metadata = _experiment_metadata(request, look) if layout == "studies" else {}
            if layout == "studies":
                require(
                    all(study.get(key) == value for key, value in experiment_metadata.items()),
                    "Published experiment label or collection differs from its recipe",
                )
            else:
                require(
                    not {"experiment_label", "collection_label", "collection_key"} & study.keys(),
                    "Unbound experiment collection metadata",
                )
            if collection.get("publication_version", 1) >= 2 or any(
                key in study for key in metadata
            ):
                require(
                    all(study.get(key) == value for key, value in metadata.items()),
                    "Published optics, process duration, or diagnostic metadata differs",
                )
                require(
                    study.get("spectral_record") == origin.get("spectral")
                    and study.get("assessment_record") == origin.get("assessment")
                    and study.get("mass_budget_record") == origin.get("mass_budget"),
                    "Published study records belong to another case",
                )
            if request.get("layout") is not None:
                require(
                    study.get("initial") in files
                    and files[study["initial"]] == receipt["artifacts"][f"{look}/initial.png"],
                    "Published starting-pool image differs",
                )
            else:
                require(study.get("initial") is None, "Unbound starting-pool image")
            require(
                study["resolution"] == recipe["render"]["still_resolution"]
                and study["image"] in files
                and files[study["image"]] == receipt["artifacts"][f"{look}/poster.png"],
                "Published painting image differs",
            )
            if "preview" in study:
                require(study["preview"] in files, "Unbound review preview")
                with Image.open(output / study["preview"]) as preview:
                    require(
                        np.array_equal(
                            np.asarray(preview), thumbnail_pixels(output / study["image"])
                        ),
                        "Review preview does not depict its bound painting",
                    )
            if request["mode"] == "film":
                require(
                    study["film"] in files
                    and files[study["film"]] == receipt["artifacts"][f"{look}/film.mp4"]
                    and all(
                        study[key] == value
                        for key, value in _film_metadata(request, receipt, look).items()
                    ),
                    "Published film or formation caption differs",
                )
            else:
                require(
                    request["mode"] == "still"
                    and study["film"] is None
                    and all(study[key] is None for key in _film_metadata(request, receipt, look)),
                    "A still view cannot advertise a film",
                )
    require(expected_ids == set(studies), "Gallery contains unbound optical views")
    comparisons = (
        _experiment_comparisons(collection["studies"])
        if layout == "studies"
        else _comparisons(collection["studies"], case_requests)
    )
    for study in collection["studies"]:
        require(
            (study["comparison_id"], study["baseline"], study["comparison_caption"])
            == comparisons[study["id"]],
            "Comparison source or material-history association differs",
        )
    earlier_records = curation.get("earlier_sources", [])
    earlier = {}
    for record in earlier_records:
        require(record["id"] not in earlier, "Duplicate earlier version provenance")
        earlier[record["id"]] = _verify_earlier(output, files, record)
    referenced = set()
    for study in collection["studies"]:
        earlier_id = study.get("earlier_id")
        if earlier_id is None:
            require(
                study.get("earlier_image") is None and study.get("earlier_caption") is None,
                "Unbound earlier version image",
            )
            continue
        require(earlier_id in earlier, "Earlier version provenance is missing")
        record = earlier[earlier_id]
        require(
            _study_key(study) == _study_key(record)
            and study["source_sha256"] == record["source_sha256"]
            and study.get("earlier_image") == record["image"]
            and study.get("earlier_caption") == EARLIER_CAPTION,
            "Earlier version comparison association differs",
        )
        referenced.add(earlier_id)
    require(referenced == set(earlier), "Unreferenced earlier version provenance")
    return collection, curation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", required=True, nargs="+", type=Path)
    parser.add_argument("--title", default="Confluence Fresco")
    parser.add_argument("--allow-stills", action="store_true")
    parser.add_argument("--earlier-gallery", type=Path)
    parser.add_argument("--layout", choices=("classic", "films", "studies"), default="classic")
    args = parser.parse_args()
    print(
        build_gallery(
            args.output,
            args.cases,
            title=args.title,
            allow_stills=args.allow_stills,
            earlier_gallery=args.earlier_gallery,
            layout=args.layout,
        )
    )


if __name__ == "__main__":
    main()
