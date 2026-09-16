#!/usr/bin/env python3
"""Render and archive explicitly labelled Polarized Crystal development proofs.

All jobs run sequentially and receive the complete --workers budget. The base
recipe's resolution and sampling remain unchanged unless explicitly overridden.
Use a new output directory, or --resume with the exact same inputs and settings.
"""

import argparse
import copy
import hashlib
import html
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

VARIANTS = {
    "A": (
        "Clear tension",
        "Lower color scale reveals broad stress regions",
        {
            "crystal.optics.retardance_scale_nm": 12000.0,
        },
    ),
    "B": (
        "Broad opal",
        "Fewer, broader interference bands",
        {
            "crystal.optics.retardance_scale_nm": 18000.0,
        },
    ),
    "C": (
        "Jewel interior",
        "Stronger color variation inside the stress field",
        {
            "crystal.optics.retardance_scale_nm": 30000.0,
        },
    ),
    "D": (
        "Dark seams",
        "Rotated extinction seams reveal another structure",
        {
            "crystal.optics.polarizer_degrees": 22.5,
        },
    ),
    "E": (
        "Afterimage",
        "A brief source-derived memory softens the release",
        {
            "crystal.field.memory_fraction": 0.045,
            "crystal.field.memory_weight": 0.55,
            "crystal.field.memory_samples": 24,
        },
    ),
    "F": (
        "Polished specimen",
        "Clearer material, quieter reflections, closer framing",
        {
            "crystal.field.load_softness": 0.9,
            "crystal.absorption": {"x": 0.035, "y": 0.035, "z": 0.035},
            "crystal.surface_strength": 0.008,
            "crystal.edge_strength": 0.09,
            "crystal.edge_thickness": 0.85,
            "camera.orthographic_height": 6.4,
        },
    ),
}


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def preserve_json(path, value):
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing archived input differs: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def finite_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number in {path}: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    # Reject overflowed exponents too, such as JSON's syntactically valid 1e999.
    encoded(value)
    return value


def set_value(recipe, key, value):
    components = key.split(".")
    target = recipe
    for component in components[:-1]:
        target = target[component]
    if components[-1] not in target:
        raise ValueError(f"Base recipe must explicitly include {key}")
    target[components[-1]] = copy.deepcopy(value)


def make_candidates(base, selections, overrides):
    if base.get("kind") != "crystal" or not isinstance(base.get("crystal"), dict):
        raise ValueError("Base recipe must be a fully specified crystal recipe")
    if (
        base["crystal"].get("freeze_source_fraction") is not None
        or base["crystal"].get("polarizer_sweep_degrees", 0) != 0
    ):
        raise ValueError("Motion comparisons require an unfrozen source and fixed polarizers")
    if base.get("prelude_fraction") != 0:
        raise ValueError("Crystal development requires prelude_fraction=0")
    candidates = []
    for key in selections:
        name, caption, changes = VARIANTS[key]
        recipe = copy.deepcopy(base)
        for path, value in {
            "crystal.optics.retardance_scale_nm": 18000.0,
            "crystal.optics.polarizer_degrees": 0.0,
            **changes,
            **overrides,
        }.items():
            set_value(recipe, path, value)
        candidates.append({"id": key, "name": name, "caption": caption, "recipe": recipe})
    return candidates


def run_process(command, log):
    """Keep logs and stop the direct renderer cleanly on interruption."""
    started = time.monotonic()
    with log.open("ab") as stream:
        stream.write(("\nCommand: " + json.dumps(command) + "\n").encode())
        stream.flush()
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT
        )
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Renderer exited with status {code}; see {log}")
    return time.monotonic() - started


def read_base(args):
    if args.base_recipe:
        return finite_json(args.base_recipe.resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="crystal-default-") as temporary:
        folder = Path(temporary)
        recipe = folder / "base.json"
        run_process(
            [str(args.executable), "config", "--preset", "crystal", "--output", str(recipe)],
            folder / "config.log",
        )
        return finite_json(recipe)


def resolve_recipe(executable, recipe):
    """Ask the same renderer to resolve defaults and serialization omissions."""
    with tempfile.TemporaryDirectory(prefix="crystal-resolve-") as temporary:
        folder = Path(temporary)
        requested = folder / "requested.json"
        resolved = folder / "resolved.json"
        log = folder / "resolve.log"
        requested.write_bytes(encoded(recipe))
        try:
            run_process(
                [
                    str(executable),
                    "resolve-crystal",
                    "--config",
                    str(requested),
                    "--output",
                    str(resolved),
                ],
                log,
            )
        except RuntimeError as error:
            # The temporary directory is removed on exit. Keep the renderer's
            # useful error text instead of referring the user to a vanished log.
            detail = (
                log.read_text(encoding="utf-8", errors="replace")[-4000:] if log.exists() else ""
            )
            raise RuntimeError(f"Crystal recipe resolution failed: {error}\n{detail}") from error
        canonical = finite_json(resolved)
        if not isinstance(canonical, dict) or canonical.get("kind") != "crystal":
            raise ValueError("Renderer did not resolve a canonical crystal recipe")
        return canonical


def png_dimensions(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Missing PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def verify_proof(output, proof, request, candidate):
    image = output / proof["image"]
    receipt = finite_json(output / proof["receipt"])
    manifest = finite_json(output / proof["manifest"])
    recipe = candidate["recipe"]
    if manifest.get("complete") is not True or manifest.get("rendered_frames") != [proof["frame"]]:
        raise ValueError("Proof manifest is incomplete or contains different frames")
    if (
        manifest.get("config") != recipe
        or manifest.get("orbit_sha256") != request["orbit"]["sha256"]
        or manifest.get("executable_sha256") != request["executable"]["sha256"]
    ):
        raise ValueError("Proof manifest does not match the archived recipe, source, and renderer")
    if png_dimensions(image) != (
        recipe["render"]["width"],
        recipe["render"]["height"],
    ) or receipt.get("png_sha256") != digest(image):
        raise ValueError("Proof PNG dimensions or receipt hash do not match")
    samples = receipt.get("crystal_samples")
    if not isinstance(samples, list) or len(samples) != recipe["temporal_samples"]:
        raise ValueError("Proof receipt is missing complete crystal exposure diagnostics")
    if receipt.get("source_fraction") != proof["frame"] / (recipe["frames"] - 1):
        raise ValueError("Proof receipt has a different nominal source time")
    seconds = receipt.get("seconds")
    if type(seconds) not in (int, float) or seconds < 0:
        raise ValueError("Proof receipt has invalid rendering duration")
    return {
        "png_sha256": receipt["png_sha256"],
        "receipt_sha256": digest(output / proof["receipt"]),
        "render_seconds": seconds,
    }


def gallery(output, request, summary):
    esc = html.escape
    panels = []
    for candidate in request["candidates"]:
        recipe = candidate["recipe"]
        crystal = recipe["crystal"]
        optics, field = crystal["optics"], crystal["field"]
        settings = (
            f"Color scale {optics['retardance_scale_nm']:g} nm per unit stress-thickness · "
            f"polarizer {optics['polarizer_degrees']:g}° · memory {field['memory_weight']:.0%} "
            f"over {field['memory_fraction']:g} of the recording "
            f"({field['memory_samples']} samples) · "
            f"camera height {recipe['camera']['orthographic_height']:g}"
        )
        figures = []
        for proof in summary["proofs"]:
            if proof["variant"] != candidate["id"]:
                continue
            fraction = proof["frame"] / (recipe["frames"] - 1)
            label = f"Frame {proof['frame']} · source {fraction:.6f} · {proof['status']}"
            body = f"<p class='pending'>{esc(proof.get('error', proof['status']))}</p>"
            if proof["status"] == "complete":
                url = esc(quote(proof["image"], safe="/"), quote=True)
                alt = esc(candidate["name"] + " — " + label, quote=True)
                body = f"<a href='{url}'><img loading='lazy' src='{url}' alt='{alt}'></a>"
            figures.append(f"<figure>{body}<figcaption>{esc(label)}</figcaption></figure>")
        dimensions = recipe["render"]
        sampling = (
            f"{dimensions['width']} x {dimensions['height']} · "
            f"AA {dimensions['aa']} x {dimensions['aa']} · "
            f"{recipe['temporal_samples']} shutter samples"
        )
        recipe_url = esc(quote(f"{candidate['id']}/recipe.json", safe="/"), quote=True)
        panels.append(
            f"<section><h2>{esc(candidate['id'] + ' — ' + candidate['name'])}</h2>"
            f"<p>{esc(candidate['caption'])}</p><p class='settings'>{esc(settings)}<br>"
            f"{esc(sampling)} · <a href='{recipe_url}'>Exact recipe</a></p>"
            f"<div class='proofs'>{''.join(figures)}</div></section>"
        )
    page = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polarized Crystal — development proofs</title><style>
body{margin:0 auto;max-width:1500px;padding:36px 24px;background:#0d0e11;
color:#e3e4e9;font:16px/1.55 system-ui,sans-serif}
h1,h2{font-weight:500}h2{margin-bottom:4px}a{color:#b9cfff}
section{border-top:1px solid #333641;margin-top:32px;padding-top:18px}
.settings,figcaption{color:#a8abb7;font-size:13px}
.proofs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:18px}
figure{margin:0}img{width:100%;display:block;background:#050506}
.pending{min-height:180px;background:#161820;display:grid;place-items:center}
figcaption{padding:8px 0}.notice{max-width:850px;color:#c4c7d1}
</style></head><body><h1>Polarized Crystal</h1><p class="notice">
<strong>Development proofs.</strong>
These isolated stills compare material and polarization choices.
They do not establish full-film quality, temporal stability, or numerical convergence.
Open an image to inspect its native size.</p>"""
        + "".join(panels)
        + "</body></html>\n"
    )
    temporary = output / "index.html.partial"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output / "index.html")


def experiment(args):
    args.executable = args.executable.resolve(strict=True)
    args.orbit = args.orbit.resolve(strict=True)
    if (
        not args.executable.is_file()
        or not os.access(args.executable, os.X_OK)
        or not args.orbit.is_file()
    ):
        raise ValueError("Renderer must be executable and orbit must be a regular file")
    executable_sha256 = digest(args.executable)
    base = read_base(args)
    frames = list(dict.fromkeys(int(item) for group in args.frames for item in group.split(",")))
    selections = [item.upper() for group in args.variants for item in group.split(",")]
    if (
        len(selections) != len(set(selections))
        or not selections
        or any(item not in VARIANTS for item in selections)
    ):
        raise ValueError("Variants must be distinct letters A through F")
    film_frames = base.get("frames")
    if (
        type(film_frames) is not int
        or film_frames < 2
        or not frames
        or any(frame < 0 or frame >= film_frames for frame in frames)
    ):
        raise ValueError("Proof frames must lie inside the base recipe's full source timeline")
    if args.workers < 1 or (args.width is None) != (args.height is None):
        raise ValueError("Workers must be positive; override width and height together")
    overrides = {}
    for name, key, maximum in [
        ("width", "render.width", 16384),
        ("height", "render.height", 16384),
        ("aa", "render.aa", 8),
        ("temporal_samples", "temporal_samples", 4096),
    ]:
        value = getattr(args, name)
        if value is not None:
            if not 1 <= value <= maximum:
                raise ValueError(f"Invalid {name}: expected 1 through {maximum}")
            overrides[key] = value
    candidates = make_candidates(base, selections, overrides)
    for candidate in candidates:
        recipe = candidate["recipe"]
        for value, maximum in [
            (recipe["render"]["width"], 16384),
            (recipe["render"]["height"], 16384),
            (recipe["render"]["aa"], 8),
            (recipe["temporal_samples"], 4096),
        ]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Recipe must contain valid explicit dimensions and sampling")
    request = {
        "schema_version": 1,
        "development": True,
        "artifact_kind": "crystal-development-proofs",
        "helper_sha256": digest(Path(__file__)),
        "executable": {"path": str(args.executable), "sha256": executable_sha256},
        "orbit": {"path": str(args.orbit), "sha256": digest(args.orbit)},
        "base_recipe_sha256": hashlib.sha256(encoded(base)).hexdigest(),
        "workers": args.workers,
        "proof_frames": frames,
    }
    output = args.output.resolve()
    previous = None
    if output.exists():
        if not args.resume or not output.is_dir() or not (output / "request.json").is_file():
            raise ValueError(
                "Output already exists; use a new directory or --resume with identical inputs"
            )
        previous = finite_json(output / "request.json")
        if not isinstance(previous, dict) or any(
            previous.get(key) != value for key, value in request.items()
        ):
            raise ValueError("Resume requires the same base recipe, source, renderer, and settings")
    elif args.resume:
        raise ValueError("--resume requires an existing matching experiment")
    # Resolve after all experiment overrides, before creating the archive. The
    # resulting exact recipe is used for rendering, comparison, and resumption;
    # the separately archived base still records what the user requested.
    for candidate in candidates:
        candidate["recipe"] = resolve_recipe(args.executable, candidate["recipe"])
    if digest(args.executable) != executable_sha256:
        raise ValueError("Renderer changed while the crystal recipes were being resolved")
    request["candidates"] = candidates
    if previous is not None and previous != request:
        raise ValueError("Resume requires identical resolved crystal recipes")
    if previous is None:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / ".experiment.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        preserve_json(output / "request.json", request)
        preserve_json(output / "base-recipe.json", base)
        summary = {"schema_version": 1, "development": True, "complete": False, "proofs": []}
        for candidate in candidates:
            folder = output / candidate["id"]
            folder.mkdir(exist_ok=True)
            preserve_json(folder / "recipe.json", candidate["recipe"])
            for frame in frames:
                directory = f"{candidate['id']}/frame-{frame:06}"
                image = f"{directory}/frame_{frame:06}.png"
                summary["proofs"].append(
                    {
                        "variant": candidate["id"],
                        "frame": frame,
                        "status": "pending",
                        "recipe": f"{candidate['id']}/recipe.json",
                        "directory": directory,
                        "image": image,
                        "receipt": image + ".json",
                        "manifest": directory + "/render.json",
                        "log": f"{candidate['id']}/frame-{frame:06}.log",
                    }
                )

        def publish():
            write_json(output / "summary.json", summary)
            gallery(output, request, summary)

        publish()
        for proof in summary["proofs"]:
            candidate = next(item for item in candidates if item["id"] == proof["variant"])
            command = [
                str(args.executable),
                "--threads",
                str(args.workers),
                "render",
                "--orbit",
                str(args.orbit),
                "--config",
                str(output / proof["recipe"]),
                "--output",
                str(output / proof["directory"]),
                "--frame",
                str(proof["frame"]),
            ]
            proof.update(status="rendering", command=command)
            publish()
            print(f"Development proof {proof['variant']}, frame {proof['frame']}", flush=True)
            started = time.monotonic()
            try:
                proof["process_seconds"] = run_process(command, output / proof["log"])
                proof.update(verify_proof(output, proof, request, candidate))
                proof["status"] = "complete"
            except BaseException as exc:
                proof.update(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=str(exc) or type(exc).__name__,
                    process_seconds=time.monotonic() - started,
                )
                publish()
                raise
            publish()
        summary["complete"] = True
        publish()
        print(f"Development gallery: {output / 'index.html'}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--executable", type=Path, required=True)
    result.add_argument("--orbit", type=Path, required=True)
    result.add_argument(
        "--base-recipe",
        type=Path,
        help="Complete crystal recipe; omitted: obtain it with config --preset crystal",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument(
        "--frames",
        nargs="+",
        default=["900", "1670"],
        help="Source frame indices, separated by spaces or commas",
    )
    result.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), help="Subset of A B C D E F"
    )
    result.add_argument("--width", type=int)
    result.add_argument("--height", type=int)
    result.add_argument("--aa", type=int)
    result.add_argument("--temporal-samples", type=int)
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        experiment(parser().parse_args())
    except KeyboardInterrupt:
        print("Interrupted; preserved development outputs can be resumed.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
