#!/usr/bin/env python3
"""Publish a curated, self-contained gallery of verified three-body paintings.

Media are copied under content hashes. Original request and receipt documents
travel with each work, while curation.json binds every published file. All cases
are validated before any gallery is published; experiments remain untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary_depth.gallery import DOCUMENT as DEPTH_DOCUMENT
from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.run import verified_run

FILM_CAPTION = "Complete formation, followed by a camera orbit"


def document(title):
    """Keep the established comparison interface with truthful film captions."""
    escaped = html.escape(title)
    result = DEPTH_DOCUMENT.replace(
        "<title>Estuary / Into depth</title>", f"<title>{escaped}</title>"
    )
    replacements = [
        (
            '<p class="eyebrow">Estuary / Into depth</p>',
            '<p class="eyebrow">Three-body painting / Material studies</p>',
        ),
        (
            "<h1>Paint, given a body.</h1>",
            f"<h1>{escaped}</h1>",
        ),
        (
            "<p>Paintings, held at their completed moment.\n"
            "Relief, translucent layers and raking light reveal their depth.</p>",
            "<p>Three-body motion becomes washes, gestures and reflected light.\n"
            "Each film follows the complete trajectory, then explores the finished painting.</p>",
        ),
        (
            'aria-label="Choose a depth study"',
            'aria-label="Choose a painting"',
        ),
        (
            'alt="A rendered Estuary depth study"',
            'alt="A three-body painting"',
        ),
        (
            '<p class="comparisonLabel">Original painting</p>',
            '<p class="comparisonLabel">Earlier Estuary · same trajectory</p>',
        ),
        (
            'id="motion" hidden>Camera orbit</button>',
            'id="motion" hidden>Play film</button>',
        ),
        (
            '<button id="formation" hidden>Formation and orbit</button>\n',
            "",
        ),
        (
            '<a id="download" download>Download the image</a>',
            '<a id="download" download>Download the image</a>\n'
            '<a id="downloadFilm" download hidden>Download the film</a>',
        ),
        (
            '<p id="motionCaption" hidden>Camera orbit of the completed painting</p>',
            f'<p id="motionCaption" hidden>{FILM_CAPTION}</p>',
        ),
        (
            "const families={relief:'Relief',layered:'Buried layers',hybrid:'Relief and glass'};",
            "const families={fresco:'Tidal Fresco',monotype:'Three-Body Monotype',"
            "nocturne:'Nocturne'};",
        ),
        (
            "$('formation').hidden=!study.formation;",
            "$('downloadFilm').hidden=!study.film;\n"
            "if(study.film){$('downloadFilm').href=study.film}"
            "else{$('downloadFilm').removeAttribute('href')}",
        ),
        (
            "$('motionCaption').textContent=kind==='film'?"
            "'Camera orbit of the completed painting':\n"
            "'Formation of the painting, followed by its camera orbit';",
            f"$('motionCaption').textContent='{FILM_CAPTION}';",
        ),
        (
            "$('formation').onclick=()=>playMotion('formation');\n",
            "",
        ),
        (
            "<h2>The studies</h2>",
            "<h2>The paintings</h2>",
        ),
        (
            '<footer id="source">These studies explore material and light over complete\n'
            "recorded trajectories.</footer>",
            '<footer id="source">Each painting follows a complete recorded three-body trajectory. '
            "The finished material stays fixed during the camera movement. "
            '<a href="http://127.0.0.1:8785/">Earlier studies</a></footer>',
        ),
    ]
    for before, after in replacements:
        require(
            before in result, "The established gallery template changed; review its integration"
        )
        result = result.replace(before, after)
    return result


def _copy_verified(source, target, expected):
    """Preserve existing content and reject changes occurring during a copy."""
    source, target = Path(source), Path(target)
    if target.exists():
        require(artifact(target) == expected, f"Published media changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".copy-", delete=False) as stream:
        partial = Path(stream.name)
    try:
        shutil.copyfile(source, partial)
        require(artifact(partial) == expected, f"Source changed while copying: {source}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def _json_artifact(value):
    data = encoded(value)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def baseline_arguments(values, seeds):
    """A bare reference is unambiguous only for a collection with one seed."""
    baselines = {}
    for value in values or []:
        if "=" in value:
            seed, path = value.split("=", 1)
            seed = seed.lower()
            require(re.fullmatch(r"0x[0-9a-f]+", seed) is not None, "Invalid baseline seed")
        else:
            require(len(seeds) == 1, "A reference for multiple seeds needs seed=PATH")
            seed, path = next(iter(seeds)), value
        require(seed in seeds and seed not in baselines, "Unknown or duplicate baseline seed")
        baselines[seed] = Path(path)
    return baselines


def build_gallery(output, cases, *, baselines=None, title="Three paintings", allow_stills=False):
    """Publish selected, complete case archives; order is the curator's order."""
    require(isinstance(title, str) and 0 < len(title) <= 120, "Gallery title must be short text")
    require(type(allow_stills) is bool, "allow_stills must be boolean")
    cases = [Path(case).resolve(strict=True) for case in cases]
    require(bool(cases), "Choose at least one completed painting")
    records, identities = [], set()
    for case in cases:
        request, receipt = verified_run(case)
        identity = receipt["identity_sha256"]
        require(identity not in identities, "A painting was selected more than once")
        identities.add(identity)
        require(allow_stills or request["mode"] == "film", "Each selected painting needs its film")
        records.append((case, request, receipt))
    seeds = {request["source"]["seed"].lower() for _, request, _ in records}
    if isinstance(baselines, (list, tuple)):
        baselines = baseline_arguments(baselines, seeds)
    else:
        baselines = {} if baselines is None else dict(baselines)
    require(set(baselines) <= seeds, "A baseline must belong to a selected seed")
    references = {seed: (Path(path), artifact(Path(path))) for seed, path in baselines.items()}
    output = Path(output).resolve()
    require(
        all(output != case and not output.is_relative_to(case) for case in cases),
        "Publish outside the immutable source case archives",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        files, studies, origins, baseline_map = {}, [], [], {}

        def copy(source, name, expected):
            _copy_verified(source, output / name, expected)
            files[name] = expected
            return name

        for seed, (path, info) in references.items():
            require(path.suffix.lower() == ".png", "Original painting references must be PNG files")
            baseline_map[seed] = copy(path, f"assets/{info['sha256']}/original.png", info)
        for case, request, receipt in records:
            identity = receipt["identity_sha256"]
            study_id = identity[:16]
            require(
                all(s["id"] != study_id for s in studies), "Truncated painting identity collision"
            )
            recipe = request["recipe"]
            seed = request["source"]["seed"].lower()
            poster_info = receipt["artifacts"]["poster.png"]
            poster = copy(
                case / "poster.png", f"assets/{poster_info['sha256']}/poster.png", poster_info
            )
            movie = None
            if request["mode"] == "film":
                movie_info = receipt["artifacts"]["film.mp4"]
                movie = copy(
                    case / "film.mp4", f"assets/{movie_info['sha256']}/film.mp4", movie_info
                )
            request_name = copy(
                case / "request.json",
                f"records/{study_id}/request.json",
                _json_artifact(request),
            )
            receipt_name = copy(
                case / "receipt.json",
                f"records/{study_id}/receipt.json",
                _json_artifact(receipt),
            )
            studies.append(
                {
                    "id": study_id,
                    "identity_sha256": identity,
                    "name": recipe["name"],
                    "group": recipe["family"],
                    "dynamics": recipe["dynamics"],
                    "seed": seed,
                    "image": poster,
                    "film": movie,
                    "baseline": baseline_map.get(seed),
                    "resolution": recipe["render"]["still_resolution"],
                    "film_caption": FILM_CAPTION if movie else None,
                }
            )
            origins.append(
                {
                    "id": study_id,
                    "identity_sha256": identity,
                    "case": str(case),
                    "request": request_name,
                    "receipt": receipt_name,
                    "source_sha256": request["source"]["sha256"],
                }
            )
        collection = {
            "schema_version": 1,
            "title": title,
            "studies": studies,
            "history": (
                "Complete trajectories; finished material remains fixed during the camera orbit"
            ),
        }
        write(output / "collection.json", collection)
        partial = output / "index.html.partial"
        partial.write_text(document(title))
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
                "baselines": baseline_map,
                "artifacts": files,
            },
        )
        verify_gallery(output)
    return output / "index.html"


def verify_gallery(output):
    """Audit a published copy without requiring the original case directories."""
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
    require(collection.get("schema_version") == 1, "Unsupported gallery collection")
    require(collection["title"] == curation["title"], "Gallery title differs from its curation")
    sources = {record["id"]: record for record in curation["sources"]}
    require(
        len(sources) == len(curation["sources"]) == len(collection["studies"]),
        "Gallery selection differs from its curation",
    )
    require(
        set(sources) == {s["id"] for s in collection["studies"]},
        "Gallery selection has duplicate or missing paintings",
    )
    for study in collection["studies"]:
        origin = sources[study["id"]]
        require(
            origin["request"] in files and origin["receipt"] in files,
            "Source records are not archived",
        )
        request = read(output / origin["request"])
        receipt = read(output / origin["receipt"])
        identity = hashlib.sha256(encoded(request)).hexdigest()
        require(
            identity
            == receipt["identity_sha256"]
            == origin["identity_sha256"]
            == study["identity_sha256"]
            and receipt["complete"] is True,
            "Source painting identity differs",
        )
        recipe, source = request["recipe"], request["source"]
        require(
            receipt["source_fraction"] == 1.0
            and receipt["final_step"] == recipe["simulation"]["steps"],
            "Published painting does not represent the complete source",
        )
        require(
            study["name"] == recipe["name"]
            and study["group"] == recipe["family"]
            and study["dynamics"] == recipe["dynamics"]
            and study["seed"] == source["seed"].lower()
            and origin["source_sha256"] == source["sha256"] == receipt["source"]["sha256"],
            "Painting caption or source association differs",
        )
        require(
            study["resolution"] == recipe["render"]["still_resolution"],
            "Published image dimensions differ from the source recipe",
        )
        require(
            files[study["image"]] == receipt["artifacts"]["poster.png"],
            "Published painting image differs",
        )
        require(
            study["baseline"] == curation["baselines"].get(study["seed"]),
            "Original painting reference belongs to a different seed",
        )
        if study["baseline"]:
            require(study["baseline"] in files, "Original painting reference is not archived")
        if request["mode"] == "film":
            require(
                study["film"] in files
                and files[study["film"]] == receipt["artifacts"]["film.mp4"]
                and study["film_caption"] == FILM_CAPTION
                and receipt["movie"]["full_decode_verified"] is True,
                "Published film or its caption differs",
            )
        else:
            require(study["film"] is None, "A still study cannot advertise a film")
    return collection, curation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", action="append", default=[], metavar="[SEED=]PNG")
    parser.add_argument("--title", default="Three paintings")
    parser.add_argument("--allow-stills", action="store_true")
    args = parser.parse_args()
    print(
        build_gallery(
            args.output,
            args.cases,
            baselines=args.baseline,
            title=args.title,
            allow_stills=args.allow_stills,
        )
    )


if __name__ == "__main__":
    main()
