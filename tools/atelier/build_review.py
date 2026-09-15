#!/usr/bin/env python3
"""Build a portable local review gallery for completed orbital art films."""

import argparse
import hashlib
import json
from pathlib import Path

STUDIES = [
    (
        "01-calligraphy",
        "Tidal Calligraphy",
        "Sweeping translucent bands and fine fibers follow the bodies' recorded paths.",
    ),
    (
        "02-loom",
        "Gravity Loom",
        "Pairwise relationships become open arches and fans of luminous thread.",
    ),
    (
        "03-aurora",
        "Aurora Veils",
        "Broad transparent curtains reveal overlapping layers of moving light.",
    ),
    (
        "04-light",
        "Light Cast by Gravity",
        "The motion becomes a choreography of projected pools, arcs, and seams of light.",
    ),
    (
        "05-engraving",
        "Orbital Engraving",
        "Fine line families form slowly changing patterns with intricate detail.",
    ),
    (
        "06-eclipse",
        "Eclipse Garden",
        "Dark forms shape crescents, openings, and luminous negative space.",
    ),
]


def has_media(path: Path) -> bool:
    """Only advertise present, nonempty files."""
    return path.is_file() and path.stat().st_size > 0


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def is_complete(directory: Path, kind: str) -> bool:
    """Require verified full-source films, their matching bytes, and a poster."""
    required = [directory / name for name in ["web.mp4", "master.mp4", "poster.png"]]
    if not all(has_media(path) for path in required):
        return False
    try:
        verification = json.loads((directory / "verification.json").read_text())
        if (
            verification.get("complete") is not True
            or verification.get("kind") != kind
            or verification.get("source_endpoints") != [0.0, 1.0]
        ):
            return False
        verified = {
            item["sha256"]: item
            for item in verification.get("movies", [])
            if item.get("all_frames_decoded_without_errors") is True
        }
        for path in required[:2]:
            record = verified.get(file_hash(path))
            if record is None or record.get("bytes") != path.stat().st_size:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def build(root: Path, *, include_development: bool = False) -> None:
    """Write the gallery beside its movies; no external service is required."""
    records = []
    for index, (slug, title, description) in enumerate(STUDIES, 1):
        directory = root / slug
        complete = is_complete(directory, slug.split("-", 1)[1])
        preview = include_development and all(
            has_media(directory / name) for name in ["web.mp4", "poster.png"]
        )
        available = {
            key: f"{slug}/{name}" if has_media(directory / name) else None
            for key, name in [
                ("video", "web.mp4"),
                ("master", "master.mp4"),
                ("poster", "poster.png"),
                ("recipe", "recipe.json"),
            ]
        }
        records.append(
            {
                "id": slug,
                "number": f"{index:02d}",
                "title": title,
                "description": description,
                "ready": complete or preview,
                "complete": complete,
                "development": preview and not complete,
                **available,
            }
        )
    if has_media(root / "reference" / "normal.mp4"):
        records.append(
            {
                "id": "reference-normal",
                "number": "Reference",
                "title": "Original accumulation reference",
                "description": "The original accumulation follows the same recorded movement "
                "through a different view combining position and velocity.",
                "ready": True,
                "complete": False,
                "development": False,
                "reference": True,
                "video": "reference/normal.mp4",
                "master": None,
                "poster": "reference/poster.png"
                if has_media(root / "reference" / "poster.png")
                else None,
                "recipe": "reference/normal.json"
                if has_media(root / "reference" / "normal.json")
                else None,
            }
        )
    template = Path(__file__).with_name("review.html").read_text()
    data = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(template.replace("/*STUDY_DATA*/", data))
    (root / "collection.json").write_text(json.dumps(records, indent=2) + "\n")
    print(root / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--include-development",
        action="store_true",
        help="Enable clearly labelled previews in a private development gallery",
    )
    args = parser.parse_args()
    build(args.directory.resolve(), include_development=args.include_development)
