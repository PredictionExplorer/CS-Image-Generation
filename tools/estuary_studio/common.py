"""Small archive and numeric contracts shared by the studio's offline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SURFACE_FIELDS = ("pigment", "height", "wetness", "direction", "roughness", "coverage")


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    value = json.loads(Path(path).read_text())
    encoded(value)
    return value


def write(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encoded(value))
    partial.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def artifact(path):
    path = Path(path)
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def checked(root, name, expected):
    root = Path(root).resolve()
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or artifact(path) != expected:
        raise ValueError(f"Archived file changed or escapes its directory: {name}")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_fields(fields, size):
    """Validate actual simulation fields before they can become an artwork."""
    w, h = size
    shapes = {
        "pigment": (h, w, 3),
        "height": (h, w),
        "wetness": (h, w),
        "direction": (h, w, 2),
        "roughness": (h, w),
        "coverage": (h, w),
    }
    for key, shape in shapes.items():
        require(key in fields, f"Missing material field: {key}")
        value = fields[key]
        require(
            value.dtype == np.float32 and value.shape == shape, f"Unexpected field layout: {key}"
        )
        require(np.isfinite(value).all(), f"Nonfinite material field: {key}")
        if key == "direction":
            require((np.linalg.norm(value, axis=2) <= 1.001).all(), "Invalid axial direction")
        else:
            require((value >= 0).all(), f"Negative material field: {key}")
            if key in ("wetness", "roughness", "coverage"):
                require((value <= 1.00001).all(), f"Material fraction outside [0, 1]: {key}")
    return {
        key: {
            "minimum": float(value.min()),
            "maximum": float(value.max()),
            "mean": float(value.mean(dtype=np.float64)),
        }
        for key, value in fields.items()
        if key in shapes
    }


def runtime_identity():
    """Bind every shipped studio module/shader and the unchanged Estuary runtime."""
    from tools.estuary.run import code_identity

    folder = Path(__file__).parent
    return {
        "studio": {
            str(p.relative_to(folder)): digest(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".glsl") and not p.name.startswith("test_")
        },
        "estuary": code_identity(),
        "depth": {
            name: digest(folder.parent / "estuary_depth" / name)
            for name in ("__init__.py", "gallery.py", "experiment.py")
        },
    }


def verify_code(folder, identity):
    """Bind copied runtime files to the request, independently of its receipt."""
    folder = Path(folder).resolve()
    # Initial studio archives contain the complete painting runtime but predate
    # bundling the optional gallery template. Both archive generations remain readable.
    require(
        set(identity) in ({"studio", "estuary"}, {"studio", "estuary", "depth"}),
        "Invalid runtime code identity",
    )
    packages = {"studio": "estuary_studio", "estuary": "estuary", "depth": "estuary_depth"}
    for group, package in packages.items():
        if group not in identity:
            continue
        root = folder / "tools" / package
        for name, sha in identity[group].items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root) and path.is_file() and digest(path) == sha,
                f"Archived runtime differs from request: {package}/{name}",
            )
