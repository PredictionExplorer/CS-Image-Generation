"""Publish portable appearance comparisons over immutable physical paintings.

Full posters, compact previews and provenance travel with the gallery. Large
linear rasters and simulation arrays remain in their original archives. Films
are optional and attach only to the exact material, palette, surface and camera
they depict. White comparisons always come from the same physical parent.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
from pathlib import Path

from tools.estuary_studio.common import artifact, checked, encoded, read, require, write
from tools.estuary_studio.gallery import _copy_verified, _json_artifact

from .appearance import VERSION as APPEARANCE_VERSION
from .appearance import verify_study
from .backgrounds import validate_background
from .palette import normalize_seed
from .run import frame_plan, verify_run

VERSION = "confluence-appearance-gallery-v1"
TEMPLATE = Path(__file__).with_name("appearance.html")
TITLE_TOKEN = "__APPEARANCE_GALLERY_TITLE__"


def document(title):
    require(type(title) is str and 0 < len(title) <= 120, "Gallery title must be short text")
    template = TEMPLATE.read_text(encoding="utf-8")
    require(template.count(TITLE_TOKEN) == 2, "Gallery title placeholders differ")
    return template.replace(TITLE_TOKEN, html.escape(title))


def _identity(request, receipt):
    identity = hashlib.sha256(encoded(request)).hexdigest()
    require(
        receipt.get("complete") is True and receipt.get("identity_sha256") == identity,
        "Incomplete or changed source archive",
    )
    return identity


def _appearance_binding(request, receipt, parent, physical):
    identity = _identity(request, receipt)
    parent_identity = _identity(parent, physical)
    require(request["version"] == APPEARANCE_VERSION, "Unsupported appearance archive")
    require(
        request["parent_identity_sha256"] == parent_identity
        and request["physical_state_sha256"] == physical["physical_state_sha256"]
        and physical["source_fraction"] == 1
        and physical["final_step"] == parent["recipe"]["simulation"]["steps"],
        "Appearance parent does not match the completed physical painting",
    )
    require(
        normalize_seed(request["seed"]) == normalize_seed(parent["source"]["seed"])
        and request["source_sha256"] == parent["source"]["sha256"]
        and request["chromatic_count"]
        == parent["recipe"]["chromatic_count"]
        == parent["palette"]["chromatic_count"]
        and normalize_seed(parent["palette"]["seed"]) == normalize_seed(parent["source"]["seed"]),
        "Appearance seed or color count differs from its parent",
    )
    ids = [look["id"] for look in request["looks"]]
    require(
        ids == request["presentations"]
        and len(ids) == len(set(ids))
        and set(ids) == set(receipt["looks"]),
        "Appearance presentation set differs",
    )
    for look in request["looks"]:
        validate_background(look["background"], parent["palette"])
    return identity


def _film_matches(parent, physical, look, request, receipt, film_look):
    render = request["recipe"]["render"]
    camera = {
        "tilt_degrees": render["still_tilt_degrees"],
        "azimuth_degrees": render["azimuth_end"],
    }
    return (
        request["mode"] == "film"
        and receipt.get("physical_state_sha256") == physical["physical_state_sha256"]
        and request["source"]["sha256"] == parent["source"]["sha256"]
        and normalize_seed(request["source"]["seed"]) == normalize_seed(parent["source"]["seed"])
        and request["palette"] == parent["palette"]
        and request.get("spectral") == parent.get("spectral")
        and request["recipe"]["simulation"] == parent["recipe"]["simulation"]
        and request["recipe"].get("projection") == parent["recipe"].get("projection")
        and request["events"] == parent["events"]
        and request["surface_configs"].get(film_look) == look["surface"]
        and camera == look["camera"]
    )


def _film_metadata(request, receipt, look):
    require(
        request["mode"] == "film"
        and receipt["source_fraction"] == 1
        and receipt["final_step"] == request["recipe"]["simulation"]["steps"],
        "A selected film must contain the complete trajectory",
    )
    require(request["frames"] == frame_plan(request["recipe"]), "Film timeline differs")
    movie = receipt["looks"][look]["movie"]
    render = request["recipe"]["render"]
    require(
        movie["full_decode_verified"] is True
        and movie["frames"] == len(request["frames"])
        and movie["fps"] == render["fps"]
        and movie["resolution"] == render["resolution"]
        and receipt["looks"][look]["physical_state_sha256"] == receipt["physical_state_sha256"],
        "Film lacks matching material or complete decode evidence",
    )
    return {
        "frames": movie["frames"],
        "fps": movie["fps"],
        "resolution": movie["resolution"],
        "seconds": movie["frames"] / movie["fps"],
        "caption": "Complete formation, followed by a camera orbit"
        if render["orbit_frames"] > 1
        else "Complete formation of the painting",
    }


def build_gallery(output, studies, *, films=None, title="A different light."):
    """Publish verified studies; seed order follows the supplied physical parents."""
    document(title)  # Validate the template before publishing any files.
    studies = [Path(path).resolve(strict=True) for path in studies]
    require(bool(studies), "Choose at least one completed appearance study")
    records, seed_counts, identities = [], set(), set()
    for path in studies:
        request, receipt = verify_study(path)
        parent, physical = read(path / "parent-request.json"), read(path / "parent-receipt.json")
        identity = _appearance_binding(request, receipt, parent, physical)
        key = (normalize_seed(request["seed"]), request["chromatic_count"])
        require(key not in seed_counts, "Choose one appearance archive per seed and color count")
        require(identity[:16] not in identities, "Duplicate or ambiguous appearance identity")
        seed_counts.add(key)
        identities.add(identity[:16])
        records.append((path, request, receipt, parent, physical))
    film_records, film_ids = [], set()
    for path in films or []:
        path = Path(path).resolve(strict=True)
        request, receipt = verify_run(path)
        identity = _identity(request, receipt)
        require(request["mode"] == "film", "Optional films must be completed film archives")
        require(identity[:16] not in film_ids, "A film was selected more than once")
        film_ids.add(identity[:16])
        film_records.append((path, request, receipt))
    matches = {}
    for _, request, _, parent, physical in records:
        for look in request["looks"]:
            found = [
                (path, film, receipt, film_look)
                for path, film, receipt in film_records
                for film_look in film["recipe"]["looks"]
                if _film_matches(parent, physical, look, film, receipt, film_look)
            ]
            require(len(found) <= 1, "Several films match one presentation; select one film")
            if found:
                _film_metadata(found[0][1], found[0][2], found[0][3])
            matches[(request["parent_identity_sha256"], look["id"])] = found[0] if found else None
    output = Path(output).resolve()
    inputs = [*studies, *(path for path, _, _ in film_records)]
    require(
        all(output != path and not output.is_relative_to(path) for path in inputs),
        "Publish outside the immutable input archives",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        files, parents, origins, published_films = {}, [], [], {}

        def copy(source, name, expected):
            _copy_verified(source, output / name, expected)
            files[name] = expected
            return name

        def publish_film(match):
            path, request, receipt, look = match
            identity = receipt["identity_sha256"]
            film_id = identity[:16]
            if film_id not in published_films:
                source = {"id": film_id, "identity_sha256": identity, "archive": str(path)}
                for kind, value in (("request", request), ("receipt", receipt)):
                    source[kind] = copy(
                        path / f"{kind}.json",
                        f"records/films/{film_id}/{kind}.json",
                        _json_artifact(value),
                    )
                published_films[film_id] = source
            info = receipt["artifacts"][f"{look}/film.mp4"]
            return {
                "source_id": film_id,
                "look": look,
                "src": copy(path / look / "film.mp4", f"assets/{info['sha256']}/film.mp4", info),
                **_film_metadata(request, receipt, look),
            }

        for path, request, receipt, parent, physical in records:
            study_id = receipt["identity_sha256"][:16]
            origin = {
                "id": study_id,
                "identity_sha256": receipt["identity_sha256"],
                "archive": str(path),
            }
            for kind, value, filename in (
                ("request", request, "request.json"),
                ("receipt", receipt, "receipt.json"),
                ("parent_request", parent, "parent-request.json"),
                ("parent_receipt", physical, "parent-receipt.json"),
            ):
                origin[kind] = copy(
                    path / filename, f"records/studies/{study_id}/{filename}", _json_artifact(value)
                )
            origins.append(origin)
            entry = {
                "id": study_id,
                "identity_sha256": receipt["identity_sha256"],
                "parent_identity_sha256": request["parent_identity_sha256"],
                "physical_state_sha256": request["physical_state_sha256"],
                "source_sha256": request["source_sha256"],
                "seed": normalize_seed(request["seed"]),
                "chromatic_count": request["chromatic_count"],
                "records": {
                    key: origin[key]
                    for key in ("request", "receipt", "parent_request", "parent_receipt")
                },
                "looks": [],
            }
            for look in request["looks"]:
                key = look["id"]
                view = {
                    "id": key,
                    "name": look["name"],
                    "resolution": request["resolution"],
                    "ground_srgb": look["background"]["ground_srgb"],
                    "film": None,
                }
                for field, filename in (("image", "poster.png"), ("preview", "preview.png")):
                    info = receipt["artifacts"][f"{key}/{filename}"]
                    view[field] = copy(
                        path / key / filename, f"assets/{info['sha256']}/{filename}", info
                    )
                info = receipt["artifacts"][f"{key}/background.json"]
                view["background_record"] = copy(
                    path / key / "background.json",
                    f"records/studies/{study_id}/{key}/background.json",
                    info,
                )
                match = matches[(request["parent_identity_sha256"], key)]
                if match is not None:
                    view["film"] = publish_film(match)
                entry["looks"].append(view)
            parents.append(entry)
        used = set(published_films)
        excluded = [
            {
                "identity_sha256": receipt["identity_sha256"],
                "reason": "No exact material, palette, surface and camera match",
            }
            for _, _, receipt in film_records
            if receipt["identity_sha256"][:16] not in used
        ]
        collection = {
            "version": VERSION,
            "title": title,
            "default_presentation": "palette-night",
            "parents": parents,
        }
        write(output / "collection.json", collection)
        partial = output / "index.html.partial"
        partial.write_text(document(title), encoding="utf-8")
        partial.replace(output / "index.html")
        for name in ("collection.json", "index.html"):
            files[name] = artifact(output / name)
        write(
            output / "curation.json",
            {
                "version": VERSION,
                "complete": True,
                "title": title,
                "sources": origins,
                "films": list(published_films.values()),
                "excluded_films": excluded,
                "artifacts": files,
            },
        )
        verify_gallery(output)
    return output / "index.html"


def verify_gallery(output):
    """Validate the portable gallery without native arrays or original folders."""
    output = Path(output)
    curation = read(output / "curation.json")
    require(
        curation.get("version") == VERSION and curation.get("complete") is True,
        "Incomplete appearance gallery",
    )
    files = curation["artifacts"]
    require({"collection.json", "index.html"} <= files.keys(), "Gallery documents are missing")
    for name, info in files.items():
        checked(output, name, info)
    collection = read(output / "collection.json")
    require(
        collection["version"] == VERSION and collection["title"] == curation["title"],
        "Appearance collection differs",
    )
    origins = {source["id"]: source for source in curation["sources"]}
    parents = {parent["id"]: parent for parent in collection["parents"]}
    require(
        len(origins) == len(curation["sources"]) == len(parents) == len(collection["parents"])
        and set(origins) == set(parents),
        "Appearance parent selection differs",
    )
    film_sources = {}
    for origin in curation["films"]:
        require(origin["id"] not in film_sources, "Duplicate published film source")
        require(
            origin["request"] in files and origin["receipt"] in files, "Film records are missing"
        )
        request, receipt = read(output / origin["request"]), read(output / origin["receipt"])
        identity = _identity(request, receipt)
        require(
            identity == origin["identity_sha256"] and origin["id"] == identity[:16],
            "Film identity differs",
        )
        film_sources[origin["id"]] = (request, receipt)
    used_films, seed_counts = set(), set()
    for parent_id, entry in parents.items():
        origin = origins[parent_id]
        require(
            all(
                origin[key] in files
                for key in ("request", "receipt", "parent_request", "parent_receipt")
            ),
            "Study records are missing",
        )
        request, receipt, parent, physical = [
            read(output / origin[key])
            for key in ("request", "receipt", "parent_request", "parent_receipt")
        ]
        identity = _appearance_binding(request, receipt, parent, physical)
        require(
            identity == entry["identity_sha256"] == origin["identity_sha256"]
            and parent_id == identity[:16],
            "Appearance identity differs",
        )
        key = (normalize_seed(request["seed"]), request["chromatic_count"])
        require(key not in seed_counts, "Duplicate published seed and color count")
        seed_counts.add(key)
        require(
            entry["seed"] == key[0]
            and entry["chromatic_count"] == key[1]
            and entry["source_sha256"] == request["source_sha256"]
            and entry["parent_identity_sha256"] == request["parent_identity_sha256"]
            and entry["physical_state_sha256"] == request["physical_state_sha256"]
            and entry["records"]
            == {
                key: origin[key]
                for key in ("request", "receipt", "parent_request", "parent_receipt")
            },
            "Appearance caption or physical association differs",
        )
        require(
            [look["id"] for look in entry["looks"]] == request["presentations"],
            "Published presentation selection differs",
        )
        for view, look in zip(entry["looks"], request["looks"], strict=True):
            name = look["id"]
            require(
                view["name"] == look["name"]
                and view["resolution"] == request["resolution"]
                and view["ground_srgb"] == look["background"]["ground_srgb"],
                "Presentation caption or ground differs",
            )
            for field, filename in (
                ("image", "poster.png"),
                ("preview", "preview.png"),
                ("background_record", "background.json"),
            ):
                require(
                    view[field] in files
                    and files[view[field]] == receipt["artifacts"][f"{name}/{filename}"],
                    "Published presentation media differs",
                )
            require(
                read(output / view["background_record"]) == look["background"],
                "Published background record differs",
            )
            movie = view["film"]
            if movie is not None:
                require(movie["source_id"] in film_sources, "Unbound presentation film")
                film, film_receipt = film_sources[movie["source_id"]]
                require(
                    _film_matches(parent, physical, look, film, film_receipt, movie["look"]),
                    "Film does not match this presentation and physical parent",
                )
                expected = _film_metadata(film, film_receipt, movie["look"])
                require(
                    all(movie[key] == value for key, value in expected.items())
                    and movie["src"] in files
                    and files[movie["src"]]
                    == film_receipt["artifacts"][f"{movie['look']}/film.mp4"],
                    "Published film media or timeline differs",
                )
                used_films.add(movie["source_id"])
    require(used_films == set(film_sources), "Published film has no matching presentation")
    return collection, curation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--studies", required=True, nargs="+", type=Path)
    parser.add_argument("--films", nargs="*", type=Path)
    parser.add_argument("--title", default="A different light.")
    args = parser.parse_args()
    print(build_gallery(args.output, args.studies, films=args.films, title=args.title))


if __name__ == "__main__":
    main()
