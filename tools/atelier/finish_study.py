#!/usr/bin/env python3
"""Finish one already-designed study after its independently rendered ranges complete."""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import time
from pathlib import Path

from verify_film import digest, verify


def complete(path: Path) -> bool:
    """Only a published complete render manifest releases a range for assembly."""
    manifest = path / "render.json"
    return manifest.is_file() and json.loads(manifest.read_text()).get("complete") is True


def run_logged(command: list[str], log: Path) -> None:
    """Keep detailed encoder output beside the study."""
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def finish(args: argparse.Namespace) -> None:
    """Assemble verified pixels, encode both films, validate, and publish the poster."""
    root = args.output.resolve()
    chunks = [path.resolve() for path in args.input]
    executable = str(args.executable.resolve())
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "finish-logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    previous = None
    while not all(complete(path) for path in chunks):
        if not args.wait:
            raise ValueError("One or more source ranges are incomplete; use --wait to wait")
        counts = tuple(sum(1 for _ in path.glob("frame_*.png")) for path in chunks)
        if counts != previous:
            print(f"Rendered frames by range: {counts}", flush=True)
            previous = counts
        if time.monotonic() - started > args.timeout_hours * 3600:
            raise TimeoutError("Timed out waiting for the source ranges")
        time.sleep(15)
    frames = root / "final-frames"
    command = [executable, "assemble", "--output", str(frames)]
    for path in chunks:
        command.extend(["--input", str(path)])
    print("Assembling verified full-source frames", flush=True)
    run_logged(command, logs / "assemble.log")
    manifest = json.loads((frames / "render.json").read_text())

    def encode(hq: bool) -> Path:
        name = "master" if hq else "web"
        path = root / (name + ".mp4")
        metadata = path.with_suffix(".mp4.json")
        if path.exists():
            if not metadata.is_file():
                raise ValueError(f"Existing movie has no provenance: {path}")
            record = json.loads(metadata.read_text())
            if record.get("source") != manifest or record.get("sha256") != digest(path):
                raise ValueError(f"Existing movie has different provenance: {path}")
            return path
        cmd = [
            executable,
            "encode",
            "--input",
            str(frames),
            "--output",
            str(path),
            "--encoder-threads",
            str(args.encoder_threads),
        ]
        if hq:
            cmd.append("--hq")
        run_logged(cmd, logs / (name + "-encode.log"))
        return path

    print("Encoding the browser film and 10-bit master", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        movies = list(pool.map(encode, [False, True]))
    print("Decoding every frame and verifying both films", flush=True)
    verify(frames, movies, root / "verification.json")
    recipe = manifest["config"]
    (root / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    shutil.copy2(frames / "render.json", root / "render.json")
    shutil.copy2(frames / "assembly.json", root / "assembly.json")
    poster = frames / f"frame_{args.poster_frame:06d}.png"
    if not poster.is_file():
        raise ValueError("Requested poster frame is absent")
    shutil.copy2(poster, root / "poster.png")
    print(f"Finished study: {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--poster-frame", type=int, default=1670)
    parser.add_argument("--encoder-threads", type=int, default=16)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()
    if args.encoder_threads < 1 or args.poster_frame < 0 or args.timeout_hours <= 0:
        parser.error("Invalid thread count, poster index, or wait timeout")
    finish(args)
