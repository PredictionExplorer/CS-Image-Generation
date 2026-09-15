#!/usr/bin/env python3
"""Collect the six verified study packages, rebuild their gallery, then exit."""

import argparse
import contextlib
import fcntl
import io
import json
import math
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from build_review import STUDIES, build, is_complete

SSH_TARGET = "user@100.76.88.48"
REMOTE_ROOT = "/home/user/tidal-silk/six-studies-b7"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3] / "CS-Image-Generation/output/six-art-studies-b7"
)
PACKAGE_FILES = (
    "web.mp4",
    "master.mp4",
    "web.mp4.json",
    "master.mp4.json",
    "poster.png",
    "recipe.json",
    "render.json",
    "assembly.json",
    "verification.json",
)
SLUGS = tuple(slug for slug, _title, _description in STUDIES)

# Sent to Python over SSH as one shell-quoted argument. Remote operations are read-only.
REMOTE_SCRIPT = r"""
import json
import sys
import tarfile
from pathlib import Path

request = json.loads(sys.argv[1])
root = Path(request["root"])
files = request["files"]

def readiness(slug):
    directory = root / slug
    try:
        present = [name for name in files if (directory / name).is_file()
                   and not (directory / name).is_symlink()
                   and (directory / name).stat().st_size > 0]
        if len(present) != len(files):
            reason = f"Waiting for package files ({len(present)}/{len(files)})"
            return {"ready": False, "reason": reason}
        record = json.loads((directory / "verification.json").read_text())
        if (not isinstance(record, dict) or record.get("complete") is not True
                or record.get("kind") != slug.split("-", 1)[1]
                or record.get("source_endpoints") != [0.0, 1.0]):
            return {"ready": False, "reason": "Waiting for complete verification"}
        movies = record.get("movies", [])
        for name in ("web.mp4", "master.mp4"):
            matching = [movie for movie in movies if isinstance(movie, dict)
                        and movie.get("file") == name]
            if len(matching) != 1:
                return {"ready": False, "reason": "Waiting for both verified movies"}
            movie = matching[0]
            digest = movie.get("sha256", "")
            if (movie.get("all_frames_decoded_without_errors") is not True
                    or movie.get("bytes") != (directory / name).stat().st_size
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                return {"ready": False, "reason": "Waiting for matching movie verification"}
        return {"ready": True, "reason": "Verified package ready"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"ready": False, "reason": "Waiting for readable verification"}

if len(sys.argv) == 2:
    print(json.dumps({slug: readiness(slug) for slug in request["slugs"]}))
else:
    slug = sys.argv[2]
    if slug not in request["slugs"] or not readiness(slug)["ready"]:
        sys.exit("The requested package is not complete")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name in files:
            archive.add(root / slug / name, arcname=name, recursive=False)
"""


def ssh_command(slug: str | None = None) -> list[str]:
    """Quote the entire remote Python invocation; never interpolate a shell command."""
    request = json.dumps({"root": REMOTE_ROOT, "files": PACKAGE_FILES, "slugs": SLUGS})
    remote = ["python3", "-c", REMOTE_SCRIPT, request]
    if slug is not None:
        if slug not in SLUGS:
            raise ValueError("Unknown study package")
        remote.append(slug)
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "--",
        SSH_TARGET,
        shlex.join(remote),
    ]


def remaining(deadline: float, limit: float) -> float:
    """Bound each network call and sleep by the collection's finite deadline."""
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Collection wait timed out")
    return min(seconds, limit)


def local_complete(directory: Path, slug: str) -> bool:
    """Use the gallery's movie hashes, including protection against filesystem races."""
    try:
        return is_complete(directory, slug.split("-", 1)[1])
    except OSError:
        return False


def remote_status(deadline: float) -> dict:
    result = subprocess.run(
        ssh_command(),
        capture_output=True,
        text=True,
        check=True,
        timeout=remaining(deadline, 60),
    )
    status = json.loads(result.stdout)
    if not isinstance(status, dict) or set(status) != set(SLUGS):
        raise ValueError("Remote status did not contain the six known studies")
    for item in status.values():
        if (
            not isinstance(item, dict)
            or type(item.get("ready")) is not bool
            or not isinstance(item.get("reason"), str)
        ):
            raise ValueError("Remote status has an invalid package record")
    return status


def unpack_package(archive_path: Path, stage: Path) -> None:
    """Copy only nine regular files; reject archive paths, links and duplicate entries."""
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        if len(members) != len(PACKAGE_FILES) or {m.name for m in members} != set(PACKAGE_FILES):
            raise ValueError("Transfer did not contain exactly the expected package files")
        for member in members:
            if not member.isfile() or member.size <= 0:
                raise ValueError("Transfer contains an empty file or non-file entry")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Transfer contains an unreadable file")
            with source, (stage / member.name).open("xb") as target:
                shutil.copyfileobj(source, target)
            if (stage / member.name).stat().st_size != member.size:
                raise ValueError("Transfer contains a truncated file")


def collect_package(root: Path, slug: str, deadline: float) -> None:
    """Validate a private staging directory before publishing it beside the gallery."""
    destination = root / slug
    with tempfile.TemporaryDirectory(prefix=f".{slug}.incoming-", dir=root) as temporary:
        workspace = Path(temporary)
        archive = workspace / "package.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ssh_command(slug),
                stdout=stream,
                stderr=subprocess.PIPE,
                check=True,
                timeout=remaining(deadline, 1800),
            )
        stage = workspace / "package"
        stage.mkdir()
        unpack_package(archive, stage)
        if not local_complete(stage, slug):
            raise ValueError("Downloaded package failed the gallery's hash verification")
        # Another process may have completed the destination while SSH was transferring.
        if local_complete(destination, slug):
            return
        backup = None
        if destination.exists() or destination.is_symlink():
            backup = root / f".{slug}.before-finish-{uuid.uuid4().hex}"
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup is not None:
                backup.rename(destination)
            raise


class ChangedStates:
    """Keep a background build log quiet while package states remain unchanged."""

    def __init__(self) -> None:
        self.previous: dict[str, str] = {}

    def report(self, key: str, message: str) -> None:
        if self.previous.get(key) == message:
            return
        self.previous[key] = message
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"{stamp} {key}: {message}", flush=True)


def failure_message(error: Exception) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return "Network operation timed out; will retry"
    if isinstance(error, subprocess.CalledProcessError):
        detail = error.stderr or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        detail = " ".join(detail.split())[-300:]
        return f"Remote operation failed ({error.returncode}): {detail}"
    return str(error)


def rebuild(root: Path) -> None:
    # build() validates the advertised movie bytes; browser notes are not in these files.
    with contextlib.redirect_stdout(io.StringIO()):
        build(root)


def collect(args: argparse.Namespace) -> int:
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_hours * 3600
    log = ChangedStates()
    completed: set[str] = set()
    gallery_built = False
    # An advisory lock prevents two instances from publishing over one another.
    with (root / ".finish-collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "A collection finisher already holds this output directory"
            ) from error
        while True:
            before = set(completed)
            for slug in SLUGS:
                if slug not in completed and local_complete(root / slug, slug):
                    completed.add(slug)
                    log.report(slug, "Complete locally; preserved")
            if len(completed) < len(SLUGS):
                try:
                    status = remote_status(deadline)
                    log.report("remote", "Available")
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    status = {}
                    log.report("remote", failure_message(error))
                for slug in SLUGS:
                    if slug in completed or slug not in status:
                        continue
                    if not status[slug]["ready"]:
                        log.report(slug, status[slug]["reason"])
                        continue
                    try:
                        collect_package(root, slug, deadline)
                        completed.add(slug)
                        log.report(slug, "Copied and verified complete")
                    except (
                        OSError,
                        ValueError,
                        tarfile.TarError,
                        subprocess.SubprocessError,
                    ) as error:
                        log.report(slug, failure_message(error))
            if completed != before or not gallery_built:
                rebuild(root)
                gallery_built = True
                log.report(
                    "gallery", f"Rebuilt with {len(completed)} of {len(SLUGS)} films complete"
                )
            if len(completed) == len(SLUGS):
                # Recheck once at completion in case other tooling changed an earlier package.
                completed = {slug for slug in SLUGS if local_complete(root / slug, slug)}
                if len(completed) == len(SLUGS):
                    log.report("collection", "All six films pass verification; finished")
                    return 0
                gallery_built = False
                log.report(
                    "collection", "A local package changed; waiting for a verified replacement"
                )
            if args.once:
                log.report(
                    "collection", f"One pass complete; {len(SLUGS) - len(completed)} films pending"
                )
                return 1
            try:
                time.sleep(remaining(deadline, args.poll_seconds))
            except TimeoutError:
                log.report("collection", f"Timed out; {len(SLUGS) - len(completed)} films pending")
                return 1


def positive_finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-hours", type=positive_finite, default=48.0)
    parser.add_argument("--poll-seconds", type=positive_finite, default=60.0)
    parser.add_argument(
        "--once", action="store_true", help="Run one pass; exit 1 if films are pending"
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_hours * 3600):
        parser.error("Wait timeout is too large")
    return args


def main() -> int:
    args = parse_args()
    try:
        return collect(args)
    except KeyboardInterrupt:
        print(
            "Collection finisher interrupted; verified local packages remain available", flush=True
        )
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Collection finisher: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
