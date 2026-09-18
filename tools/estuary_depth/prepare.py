"""Prepare authored 3D relief from verified, actual Estuary pigment concentrations.

Height never uses RGB. All prepared maps use bottom-up rows and the full original
guard domain. Historical layers require archived hashes; filenames alone are not
evidence of an earlier material state. This is a new relief interpretation of a
2D painting, not a claim that the original solver simulated a 3D liquid.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.optics import Material, reflectance
from tools.estuary.recipe import validate_recipe
from tools.estuary.run import artifact, checked_artifact, digest, encoded, read_json, write_json
from tools.estuary.source import Source


def require(condition, message):
    if not condition:
        raise ValueError(message)


def map_resolution(value, maximum):
    require(
        isinstance(value, (list, tuple)) and len(value) == 2, "Resolution needs width and height"
    )
    require(all(type(n) is int and 2 <= n <= 4096 for n in value), "Invalid map resolution")
    require(math.prod(value) <= maximum, "Map pixel budget exceeded")
    return tuple(value)


def checked_array(path, shape, *, upper=None):
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(value.dtype == np.float32 and value.shape == shape, f"Invalid array layout: {path}")
    for start in range(0, shape[0], 64):
        block = value[start : start + 64]
        require(np.isfinite(block).all() and (block >= 0).all(), f"Invalid concentrations: {path}")
        require(upper is None or (block <= upper).all(), f"Invalid linear reflectance: {path}")
    return value


def area_resample(array, size, domain=1.0):
    """Average exact source-pixel overlap, retaining bottom-up row orientation.

    Source pixels are piecewise constant concentration cells. The camera sees
    the central 1/domain portion of each axis; no RGB-dependent fitting occurs.
    """
    height, width, channels = array.shape
    out_width, out_height = size
    require(math.isfinite(domain) and domain >= 1, "Invalid crop domain")

    def weights(count, target):
        span = count / domain
        edges = (count - span) * 0.5 + np.arange(target + 1) * (span / target)
        first = np.floor(edges[:-1]).astype(int)
        indices = first[:, None] + np.arange(math.ceil(span / target) + 1)
        overlap = np.maximum(
            0, np.minimum(edges[1:, None], indices + 1) - np.maximum(edges[:-1, None], indices)
        ) / (span / target)
        return np.clip(indices, 0, count - 1), overlap

    xi, xw = weights(width, out_width)
    yi, yw = weights(height, out_height)
    result = np.empty((out_height, out_width, channels), dtype=np.float32)
    # One output row bounds temporary memory even when the source is much larger.
    for row in range(out_height):
        source = np.asarray(array[yi[row]], dtype=np.float64)
        horizontal = np.zeros((len(yi[row]), out_width, channels))
        for index in range(xi.shape[1]):
            horizontal += source[:, xi[:, index]] * xw[None, :, index, None]
        result[row] = np.einsum("i,ijk->jk", yw[row], horizontal)
    return result


def material_maps(state, size, mesh_size, domain, material, volumes):
    density = area_resample(state[..., :3], size, domain)
    total = density.sum(axis=2, keepdims=True, dtype=np.float64)
    fractions = np.divide(density, total, out=np.zeros_like(density), where=total > 0)
    color = np.empty_like(density)
    for start in range(0, len(density), 64):
        color[start : start + 64] = reflectance(density[start : start + 64], material)
    geometry_density = area_resample(state[..., :3], mesh_size, domain)
    height = np.einsum("...i,i->...", geometry_density, np.asarray(volumes, dtype=np.float64))
    require(
        np.isfinite(height).all() and (height <= np.finfo(np.float32).max).all(),
        "Authored height exceeds finite range",
    )
    height = height.astype(np.float32)
    return {"color_linear": color, "density": density, "fractions": fractions, "height": height}


def verified_run(folder):
    request, receipt = read_json(folder / "request.json"), read_json(folder / "receipt.json")
    identity = hashlib.sha256(encoded(request)).hexdigest()
    require(receipt.get("complete") is True, "Estuary run is incomplete")
    require(receipt.get("identity_sha256") == identity, "Render request identity differs")
    recipe = validate_recipe(request["recipe"])
    require(
        recipe == request["recipe"], "Render recipe does not include its exact resolved controls"
    )
    require(receipt.get("source") == request["source"], "Receipt source identity differs")
    require(
        receipt.get("final_step") == recipe["simulation"]["steps"]
        and receipt.get("source_fraction") == 1.0,
        "Render does not cover the complete source",
    )
    records = {item["path"]: item for item in receipt["artifacts"]}
    require(len(records) == len(receipt["artifacts"]), "Duplicate archived artifact names")
    required = ("final-state.npy", "linear.npy", "recipe.json", "inputs/source.orbit")
    require(all(name in records for name in required), "Required source artifacts are missing")
    paths = {name: checked_artifact(folder, records[name]) for name in required}
    require(read_json(paths["recipe.json"]) == recipe, "Archived recipe differs")
    sw, sh = recipe["simulation"]["resolution"]
    rw, rh = recipe["render"]["resolution"]
    state = checked_array(paths["final-state.npy"], (sh, sw, 4))
    checked_array(paths["linear.npy"], (rh, rw, 3), upper=1)
    source = Source.read(paths["inputs/source.orbit"], aspect=sw / sh, **recipe["projection"])
    for key in ("sha256", "samples_sha256", "samples", "dt", "masses", "seed", "provenance"):
        require(source.metadata[key] == request["source"][key], f"Source {key} differs")
    return request, identity, recipe, state, records


def history_records(path, request, identity, final_record):
    """A deterministic replay must reproduce the archived final NPY file exactly."""
    if path is None:
        return [], None
    path = Path(path).resolve(strict=True)
    ledger = read_json(path)
    require(
        ledger.get("schema_version") == 1 and ledger.get("identity_sha256") == identity,
        "History belongs to another render",
    )
    require(
        ledger.get("source_sha256") == request["source"]["sha256"]
        and ledger.get("code") == request["code"],
        "History source or code identity differs",
    )
    replay_final = ledger["final_state"]
    checked_artifact(path.parent, replay_final)
    require(replay_final["sha256"] == final_record["sha256"], "Replay final state differs")
    steps = request["recipe"]["simulation"]["steps"]
    found, seen = [], set()
    for entry in ledger["checkpoints"]:
        step = entry["step"]
        require(
            type(step) is int and 0 < step < steps and step not in seen,
            "Invalid or duplicate history step",
        )
        require(entry.get("source_fraction") == step / steps, "History source fraction differs")
        checked_artifact(path.parent, entry["state"])
        seen.add(step)
        found.append(entry)
    return sorted(found, key=lambda entry: entry["step"]), path


def build_bundle(
    run_path,
    out,
    resolution=(2048, 1536),
    mesh_resolution=(768, 576),
    history_fractions=(),
    specific_volumes=(0.25, 1.0, 0.5),
    history_manifest=None,
):
    """Write bundle.npz and manifest.json, or verify an identical existing bundle."""
    folder, output = Path(run_path).resolve(strict=True), Path(out).resolve()
    size = map_resolution(resolution, 16_777_216)
    mesh_size = map_resolution(mesh_resolution, 4_194_304)
    volumes = np.asarray(specific_volumes, dtype=np.float64)
    require(
        volumes.shape == (3,)
        and np.isfinite(volumes).all()
        and (volumes > 0).all()
        and (volumes <= 10).all(),
        "Invalid specific volumes",
    )
    targets = np.asarray(history_fractions, dtype=np.float64)
    require(
        targets.ndim == 1
        and len(targets) <= 6
        and np.isfinite(targets).all()
        and (targets > 0).all()
        and (targets < 1).all(),
        "Invalid history fractions",
    )
    require(np.all(np.diff(targets) > 0), "History fractions must increase strictly")
    request, identity, recipe, state, records = verified_run(folder)
    sw, sh = recipe["simulation"]["resolution"]
    require(all(w * sh == h * sw for w, h in (size, mesh_size)), "Map aspect must match the source")
    candidates, ledger_path = history_records(
        history_manifest, request, identity, records["final-state.npy"]
    )
    require(
        len(targets) == 0 or candidates, "Requested historical layers need a verified replay ledger"
    )
    steps = recipe["simulation"]["steps"]
    available = [item for item in candidates if 0 < item["step"] < steps]
    selected = {}
    for target in targets:
        if available:
            item = min(available, key=lambda entry: abs(entry["step"] / steps - target))
            selected[item["step"]] = item
    inputs = {
        "render_identity": identity,
        "request_sha256": digest(folder / "request.json"),
        "receipt_sha256": digest(folder / "receipt.json"),
        "artifacts": {
            name: records[name]
            for name in ("final-state.npy", "linear.npy", "recipe.json", "inputs/source.orbit")
        },
        "history": list(selected.values()),
        "history_ledger_sha256": None if ledger_path is None else digest(ledger_path),
    }
    params = {
        "resolution": list(size),
        "mesh_resolution": list(mesh_size),
        "history_fractions": targets.tolist(),
        "specific_volumes": volumes.tolist(),
    }
    domain = recipe["simulation"]["domain_scale"]
    geometry = {
        "domain_scale": domain,
        "view_aspect": sw / sh,
        "coordinates": {
            "row_order": "bottom-to-top",
            "uv_origin": "bottom-left",
            "view_bounds": [[-sw / sh, -1], [sw / sh, 1]],
            "domain_bounds": [[-sw / sh * domain, -domain], [sw / sh * domain, domain]],
            "resampling": "exact pixel-area averages over the full guard domain",
        },
    }
    prepare_request = {
        "schema_version": 1,
        "inputs": inputs,
        "parameters": params,
        "geometry": geometry,
        "source": request["source"],
        "prepare_sha256": digest(Path(__file__)),
        "optics_sha256": digest(Path(reflectance.__code__.co_filename)),
    }
    bundle_identity = hashlib.sha256(encoded(prepare_request)).hexdigest()
    if output.exists():
        manifest = read_json(output / "manifest.json")
        require(
            manifest.get("complete") is True and manifest.get("identity_sha256") == bundle_identity,
            "Existing bundle identity differs or is incomplete",
        )
        require(manifest.get("request") == prepare_request, "Stored preparation request differs")
        require(
            all(manifest.get(key) == value for key, value in geometry.items()),
            "Bundle mapping metadata differs from its preparation identity",
        )
        require(manifest.get("source") == request["source"], "Bundle source metadata differs")
        checked_artifact(output, manifest["bundle"])
        return manifest
    output.mkdir(parents=True)
    material = Material(**recipe["optics"])
    maps = material_maps(state, size, mesh_size, 1.0, material, volumes)
    histories, times = [], []
    for step, record in sorted(selected.items()):
        old = checked_array(checked_artifact(ledger_path.parent, record["state"]), state.shape)
        histories.append(material_maps(old, size, mesh_size, 1.0, material, volumes))
        times.append(step / steps)
    histories.append(maps)
    times.append(1.0)
    for key in ("color_linear", "fractions", "height"):
        maps[f"history_{key}"] = np.stack([item[key] for item in histories])
    maps["history_times"] = np.asarray(times, dtype=np.float64)
    partial = output / "bundle.partial.npz"
    np.savez(partial, **maps)
    partial.replace(output / "bundle.npz")
    # Revalidate core identities after preparation before certifying the result.
    for record in inputs["artifacts"].values():
        checked_artifact(folder, record)
    for record in inputs["history"]:
        checked_artifact(ledger_path.parent, record["state"])
    if ledger_path is not None:
        history_records(ledger_path, request, identity, records["final-state.npy"])
    require(
        ledger_path is None or digest(ledger_path) == inputs["history_ledger_sha256"],
        "History ledger changed during preparation",
    )
    require(
        digest(folder / "request.json") == inputs["request_sha256"]
        and digest(folder / "receipt.json") == inputs["receipt_sha256"],
        "Render metadata changed",
    )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "identity_sha256": bundle_identity,
        "request": prepare_request,
        "source": request["source"],
        "bundle": artifact(output / "bundle.npz", output),
        **geometry,
        "height": {
            "formula": "sum(concentration * specific_volume)",
            "units": "authored relative volume per canvas area; no RGB contribution",
            "physically_simulated_3d": False,
        },
        "history": {
            "source_fractions": times,
            "older_history_available": bool(selected),
            "interpretation": "frozen authentic 2D states, not simulated 3D layers",
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--mesh-width", type=int, default=768)
    parser.add_argument("--mesh-height", type=int, default=576)
    args = parser.parse_args()
    print(
        encoded(
            build_bundle(
                args.run,
                args.output,
                (args.width, args.height),
                (args.mesh_width, args.mesh_height),
                (0.35, 0.65) if args.history else (),
                history_manifest=args.history,
            )
        ).decode()
    )
