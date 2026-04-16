"""Shared helpers for the Python runner scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GENERATOR_CANDIDATES: list[str] = [
    "./target/release/three_body_problem",
    "./three_body_problem",
]


def fmt_duration(seconds: float) -> str:
    """Format an elapsed-seconds value as a compact human string (e.g. '3m42s')."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def check_ffmpeg() -> None:
    """Exit with a clear message if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH", file=sys.stderr)
        print("The Rust generator requires ffmpeg for MP4 encoding.", file=sys.stderr)
        print("Install it:  brew install ffmpeg  /  apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def _source_mtime(root: Path) -> float:
    """Latest modification time under `root` across Cargo.toml/Cargo.lock/src/build.rs."""
    latest = 0.0
    for relative in ("Cargo.toml", "Cargo.lock", "build.rs"):
        candidate = root / relative
        if candidate.is_file():
            latest = max(latest, candidate.stat().st_mtime)
    src_dir = root / "src"
    if src_dir.is_dir():
        for path in src_dir.rglob("*.rs"):
            try:
                latest = max(latest, path.stat().st_mtime)
            except FileNotFoundError:
                continue
    return latest


def ensure_release_build(binary: str | Path, *, repo_root: Path | None = None) -> Path:
    """Make sure `./target/release/three_body_problem` is freshly built.

    If the binary is missing or older than any Rust source file we automatically
    invoke ``cargo build --release`` before returning.  This guarantees that
    batch runners always pick up the latest pipeline changes without the caller
    having to remember to rebuild.
    """
    binary_path = Path(binary)
    root = repo_root or Path.cwd()
    needs_build = True
    if binary_path.is_file():
        bin_mtime = binary_path.stat().st_mtime
        if bin_mtime >= _source_mtime(root):
            needs_build = False

    if needs_build:
        print(f"Building release binary -> {binary_path}", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=root,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if proc.returncode != 0:
            print("cargo build --release failed", file=sys.stderr)
            sys.exit(proc.returncode)
        print(f"  build complete in {fmt_duration(time.monotonic() - t0)}", flush=True)

    return resolve_binary(binary_path)


def resolve_binary(path: str | Path) -> Path:
    """Validate that *path* exists and is executable, or exit."""
    p = Path(path)
    if not p.is_file():
        print(f"Error: binary not found at {p}", file=sys.stderr)
        print("Build it first:  cargo build --release", file=sys.stderr)
        sys.exit(1)
    if not os.access(p, os.X_OK):
        print(f"Error: {p} is not executable", file=sys.stderr)
        sys.exit(1)
    return p
