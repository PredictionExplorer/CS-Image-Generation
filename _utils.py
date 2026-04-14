"""Shared helpers for the Python runner scripts."""

from __future__ import annotations

import os
import shutil
import sys
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
