#!/usr/bin/env python3
"""Run a finite collection with at most three independently resumable GPU films."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.gallery import build_gallery
from tools.estuary.recipe import read_recipe
from tools.estuary.run import code_identity, completed, digest, encoded, read_json, write_json

REPO = Path(__file__).resolve().parents[2]


def child_command(args, seed):
    folder = args.output / seed
    command = [
        sys.executable,
        "-m",
        "tools.estuary.run",
        "--orbit",
        str(args.sources / f"{seed}.orbit"),
        "--recipe",
        str(args.recipe),
        "--output",
        str(folder),
        "--backend",
        "egl",
    ]
    for name in ("ffmpeg", "ffprobe"):
        if getattr(args, name) is not None:
            command += ["--" + name, str(getattr(args, name))]
    return command + (["--resume"] if folder.exists() else [])


def descendants(pids):
    """Snapshot only descendants of owned runners, including detached encoders."""
    listing = subprocess.run(
        ["ps", "-eo", "pid=,ppid="], check=True, capture_output=True, text=True
    )
    pairs = [tuple(map(int, line.split())) for line in listing.stdout.splitlines()]
    owned = set(pids)
    while True:
        found = {pid for pid, parent in pairs if parent in owned} - owned
        if not found:
            return owned - set(pids)
        owned.update(found)


def birth_identity(pid):
    """Prevent a reused PID from being mistaken for a previously owned encoder."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2 :].split()[19]
    except FileNotFoundError:
        info = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, check=False
        )
        return info.stdout.strip() or None


def stop_children(active):
    children = [value[0] for value in active.values()]
    owned = set()
    # A failed process-list query must never prevent signaling the owned runners.
    with contextlib.suppress(OSError, subprocess.CalledProcessError, ValueError):
        owned = descendants([child.pid for child in children]) if children else set()
    births = {}
    for pid in owned:
        with contextlib.suppress(OSError, ValueError):
            births[pid] = birth_identity(pid)
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + 8
    while any(child.poll() is None for child in children) and time.monotonic() < deadline:
        time.sleep(0.05)
    for child in children:
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    for pid, birth in births.items():
        if birth is not None and birth_identity(pid) == birth:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    for _, stream in active.values():
        stream.close()


def request_for(args):
    recipe = read_recipe(args.recipe)
    return {
        "schema_version": 1,
        "seeds": args.seeds,
        "recipe": {
            "path": str(args.recipe),
            "sha256": digest(args.recipe),
            "resolved": recipe,
        },
        "sources": {
            seed: {
                "path": str(args.sources / f"{seed}.orbit"),
                "sha256": digest(args.sources / f"{seed}.orbit"),
            }
            for seed in args.seeds
        },
        "runtime_code": code_identity(recipe),
        "tools": {
            name: None
            if getattr(args, name) is None
            else {"path": str(getattr(args, name)), "sha256": digest(getattr(args, name))}
            for name in ("ffmpeg", "ffprobe")
        },
    }


def verify_seed(folder, seed, request):
    actual = read_json(folder / "request.json")
    identity = hashlib.sha256(encoded(actual)).hexdigest()
    if (
        actual.get("mode") != "film"
        or actual.get("code") != request["runtime_code"]
        or actual.get("recipe") != request["recipe"]["resolved"]
        or actual.get("source", {}).get("sha256") != request["sources"][seed]["sha256"]
        or actual.get("source", {}).get("seed") != seed
        or not completed(folder, identity)
    ):
        raise ValueError(f"Render receipt does not certify this complete batch source: {seed}")
    movie = read_json(folder / "movie.json")
    render = request["recipe"]["resolved"]["render"]
    if (
        movie.get("full_decode_verified") is not True
        or movie.get("frames") != render["frames"]
        or movie.get("fps") != render["fps"]
        or movie.get("resolution") != render["resolution"]
    ):
        raise ValueError(f"Movie receipt has incomplete or different media: {seed}")


def execute(args, request):
    status = {
        "schema_version": 1,
        "complete": False,
        "status": "running",
        "workers": args.workers,
        "controller_pid": os.getpid(),
        "seeds": {seed: {"status": "pending", "pid": None} for seed in args.seeds},
    }
    active, ready, queue = {}, set(), deque(args.seeds)

    def update(seed=None, **values):
        target = status if seed is None else status["seeds"][seed]
        target.update(values)
        status["updated_unix"] = time.time()
        write_json(args.output / "status.json", status)

    update()
    try:
        while queue or active:
            while queue and len(active) < args.workers:
                seed = queue.popleft()
                stream = None
                try:
                    if request_for(args) != request:
                        raise ValueError("Batch input or renderer runtime changed")
                    command = child_command(args, seed)
                    stream = (args.output / "logs" / f"{seed}.log").open("ab")
                    stream.write(b"\nCommand: " + encoded(command))
                    stream.flush()
                    child = subprocess.Popen(
                        command,
                        cwd=REPO,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[seed] = (child, stream)
                    update(seed, status="running", pid=child.pid, command=command)
                except Exception as error:
                    if stream is not None:
                        stream.close()
                    update(seed, status="failed", error=str(error), pid=None)
            for seed, (child, stream) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del active[seed]
                try:
                    if code:
                        raise RuntimeError(
                            f"Renderer exited with status {code}; see logs/{seed}.log"
                        )
                    verify_seed(args.output / seed, seed, request)
                    ready.add(seed)
                    build_gallery(
                        [args.output / item for item in args.seeds if item in ready],
                        args.output / "gallery",
                    )
                    update(seed, status="complete", pid=None, exit_code=0)
                except Exception as error:
                    ready.discard(seed)
                    update(seed, status="failed", error=str(error), pid=None, exit_code=code)
            if active:
                time.sleep(0.25)
        success = len(ready) == len(args.seeds)
        update(complete=success, status="complete" if success else "failed", controller_pid=None)
        if not success:
            raise RuntimeError(
                "Some films failed; successful seeds were published, inspect status.json"
            )
    except BaseException as error:
        update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error) or type(error).__name__,
            complete=False,
        )
        stop_children(active)
        for seed in active:
            update(seed, status="interrupted", pid=None)
        update(controller_pid=None)
        raise


def run(args):
    if type(args.workers) is not int or not 1 <= args.workers <= 3:
        raise ValueError("Workers must be between one and three")
    if (
        not 1 <= len(args.seeds) <= 12
        or len(set(args.seeds)) != len(args.seeds)
        or any(not re.fullmatch(r"0x[0-9a-fA-F]{2,64}", seed) for seed in args.seeds)
    ):
        raise ValueError("Use one through twelve distinct hexadecimal seeds")
    for name in ("sources", "recipe", "ffmpeg", "ffprobe"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".batch.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = request_for(args)
        archived = args.output / "batch-request.json"
        if archived.exists() and read_json(archived) != request:
            raise ValueError("Batch archive belongs to different inputs or renderer runtime")
        write_json(archived, request)
        (args.output / "logs").mkdir(exist_ok=True)
        execute(args, request)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "recipe", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--seeds", nargs="+", required=True)
    result.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    result.add_argument("--ffmpeg", type=Path)
    result.add_argument("--ffprobe", type=Path)
    return result


def interrupted(_signal, _frame):
    raise KeyboardInterrupt("Batch interrupted; completed images and checkpoints remain archived")


def main():
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run(parser().parse_args())
        return 0
    except KeyboardInterrupt as error:
        print(error, file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Batch failed: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
