"""Shared accessible still/film comparison page with explicit presentation copy."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tools.estuary_studio.common import require

TEMPLATE = Path(__file__).with_name("paint_material_gallery.html")


@dataclass(frozen=True)
class Presentation:
    version: str
    eyebrow: str
    intro: str
    legend: str
    default_variant: str
    film_only_selection: bool = False


MATERIALS = Presentation(
    version="paint-material-review-v1",
    eyebrow="Three bodies / Paint with memory",
    intro=(
        "All three bodies, the same starting colors, and the complete trajectory. "
        "Compare RC1 with any treatment; choose a card below to change the right painting."
    ),
    legend=(
        "F = fuller paint · D = directional relief · S = seeded properties · "
        "R = resistance and memory. Gold outlines mark my subjective visual picks."
    ),
    default_variant="gentle-worked-paint",
)


def document(title, *, presentation=MATERIALS):
    require(type(title) is str and 0 < len(title) <= 120, "Use a short gallery title")
    require(type(presentation) is Presentation, "Use explicit review presentation")
    for key, value in vars(presentation).items():
        if key == "film_only_selection":
            require(type(value) is bool, "film_only_selection must be a boolean")
        else:
            require(type(value) is str and 0 < len(value) <= 1000, f"Invalid presentation {key}")
    for key in ("version", "default_variant"):
        require(
            re.fullmatch(r"[a-z][a-z0-9-]{0,79}", getattr(presentation, key)),
            "Invalid review identifier",
        )
    settings = json.dumps(
        {
            "version": presentation.version,
            "default_variant": presentation.default_variant,
            "film_only_selection": presentation.film_only_selection,
        }
    )
    substitutions = {
        "__TITLE__": html.escape(title),
        "__EYEBROW__": html.escape(presentation.eyebrow),
        "__INTRO__": html.escape(presentation.intro),
        "__LEGEND__": html.escape(presentation.legend),
        "__REVIEW_SETTINGS__": settings.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026"),
    }
    template = TEMPLATE.read_text()
    for token in substitutions:
        require(
            template.count(token) == (2 if token == "__TITLE__" else 1),
            "Review placeholders differ",
        )
    return re.sub("|".join(map(re.escape, substitutions)), lambda m: substitutions[m[0]], template)
