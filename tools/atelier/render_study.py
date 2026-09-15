#!/usr/bin/env python3
"""Render one curated study in independent ranges, then finish both full films."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def render(args: argparse.Namespace) -> None:
    """Keep every full-quality range resumable and preserve its render log."""
    recipe = json.loads(args.config.read_text())
    frames = recipe.get("frames", 1802)
    if not isinstance(frames, int) or frames < 2:
        raise ValueError("The recipe must specify at least two frames")
    parallel = min(args.parallel_ranges, args.chunks)
    if args.chunks > frames or args.workers < parallel:
        raise ValueError("Each chunk needs a frame and each active range needs a worker")
    if args.poster_frame >= frames:
        raise ValueError("Poster frame falls outside the full film")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "render-logs"
    logs.mkdir(exist_ok=True)
    request = output / "render-request.json"
    if request.exists():
        if json.loads(request.read_text()) != recipe:
            raise ValueError("This study folder was started with a different recipe")
    else:
        with request.open("x") as stream:
            stream.write(json.dumps(recipe, indent=2) + "\n")
    executable = str(args.executable.resolve(strict=True))
    orbit = str(args.orbit.resolve(strict=True))
    config = str(request)
    worker_count = args.workers // parallel
    chunks = [output / f"final-chunk-{i}" for i in range(args.chunks)]

    def run(index: int) -> None:
        start = frames * index // args.chunks
        end = frames * (index + 1) // args.chunks
        command = [
            executable,
            "--threads",
            str(worker_count),
            "render",
            "--orbit",
            orbit,
            "--config",
            config,
            "--output",
            str(chunks[index]),
            "--start",
            str(start),
            "--end",
            str(end),
        ]
        print(f"Range {index}: frames {start} through {end - 1}", flush=True)
        with (logs / f"range-{index}.log").open("a") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"Range {index} complete", flush=True)

    failures = []
    stride = (args.chunks + parallel - 1) // parallel
    order = sorted(range(args.chunks), key=lambda index: (index % stride, index // stride))
    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        pending = {pool.submit(run, i): i for i in order}
        for future in concurrent.futures.as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                failures.append((pending[future], str(exc)))
    if failures:
        raise RuntimeError(f"Render ranges failed; completed ranges are resumable: {failures}")
    command = [
        sys.executable,
        str(Path(__file__).with_name("finish_study.py")),
        "--output",
        str(output),
        "--executable",
        executable,
        "--poster-frame",
        str(args.poster_frame),
        "--encoder-threads",
        str(args.encoder_threads),
    ]
    for chunk in chunks:
        command.extend(["--input", str(chunk)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--parallel-ranges", type=int, default=4)
    parser.add_argument("--workers", type=int, default=112)
    parser.add_argument("--encoder-threads", type=int, default=16)
    parser.add_argument("--poster-frame", type=int, default=900)
    args = parser.parse_args()
    if (
        min(args.chunks, args.parallel_ranges, args.workers, args.encoder_threads) < 1
        or args.poster_frame < 0
    ):
        parser.error("Counts must be positive and the poster index nonnegative")
    render(args)
