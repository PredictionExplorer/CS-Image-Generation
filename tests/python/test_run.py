"""Tests for run.py: planning, the ember failure ledger, generation outcomes, uploads, stale ember
editions and main().

Standard library only, and no network. The generator, ssh and scp are fakes on PATH: tiny
launcher scripts that call the fake_* functions of this module. The fake ssh runs the remote
command with /bin/sh on this machine and the fake scp copies files locally, so the "remote" asset
directory is a temporary directory and every remote command run.py builds really runs.

Run from the repository root:

    python -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import posixpath
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run  # noqa: E402

# ---------------------------------------------------------------------------
# Seeds and package contents
# ---------------------------------------------------------------------------

SEED_A = "a" * 64
SEED_B = "b" * 64
SEED_C = "c" * 64
SEED_D = "d" * 64

# The nft_traits.json masses of every fake package, as JSON text (FAKE_GEN_MASSES_<seed>
# overrides them for one seed).
DEFAULT_MASSES = "[150.25, 200.5, 180.125]"

# The nft_traits.json view fields of every fake package (run.VIEW_IDENTITY_FIELDS), each as JSON
# text: the generator's real field names, with the values of docs/fixtures/nft_traits.example.json.
DEFAULT_VIEW: dict[str, str | None] = {
    "stack_label": '"orbit_ribbons+harmonic_weave@0.29"',
    "projection": '"position"',
    "symmetry": '"none"',
    "mode": '"elliptical"',
    "scale": "1.1646093650188862",
    "arc_fraction": "0.8249768504515819",
    "orbit_eccentricity": "0.40457605398750857",
}
# Where the generator writes each of them in nft_traits.json.
VIEW_FIELD_PATHS = {
    "stack_label": "generation.structure.stack_label",
    "projection": "generation.projection",
    "symmetry": "generation.symmetry",
    "mode": "generation.drift.mode",
    "scale": "generation.drift.scale",
    "arc_fraction": "generation.drift.arc_fraction",
    "orbit_eccentricity": "generation.drift.orbit_eccentricity",
}

# The ember look the fake generator renders (FAKE_GEN_EMBER_ALGORITHM overrides it), an older
# one, which live packages of the earlier look record, and one this generator does not know yet.
CURRENT_ALGORITHM = "ember-v3"
STALE_ALGORITHM = "ember-v2"
NEWER_ALGORITHM = "ember-v4"

# The ember files and manifest roles of the current look, in the generator's order, and those of
# an ember-v2 package as the release before ember-v3 published it (six files, five roles, a
# layout-3 certificate). Spelled out, never derived from run.py's tuples: the first pins those
# tuples, and the second is what the current run.py finds on the asset host after the deploy.
EMBER_V3_FILES = (
    "images/source/ember.png",
    "images/web/ember_full.webp",
    "images/web/ember_preview.webp",
    "videos/web/ember.mp4",
    "videos/web/ember_slow.mp4",
    "videos/hq/ember.mp4",
    "metadata/ember.json",
)
# The order in which an ember-mode backfill stages them and then swaps them in: the media, the
# merged manifest, the certificate last.
EMBER_UPLOAD_ORDER = (*EMBER_V3_FILES[:-1], "metadata/assets.json", "metadata/ember.json")
EMBER_V3_ROLES = (
    "ember_source_master",
    "ember_web_full",
    "ember_web_preview",
    "ember_web",
    "ember_slow_web",
    "ember_hq",
)
EMBER_V2_MEDIA_ROLES = {
    "images/source/ember.png": "ember_source_master",
    "images/web/ember_full.webp": "ember_web_full",
    "images/web/ember_preview.webp": "ember_web_preview",
    "videos/web/ember.mp4": "ember_web",
    "videos/hq/ember.mp4": "ember_hq",
}
EMBER_V2_FILES = (*EMBER_V2_MEDIA_ROLES, "metadata/ember.json")
EMBER_V2_CERTIFICATE = """{
  "schema_version": 3,
  "edition": "ember",
  "algorithm": "ember-v2",
  "contract": "the live render"
}
"""

CORE_MEDIA_ROLES = {
    "images/source/master.png": "source_master",
    "images/web/full.webp": "web_full",
    "images/web/preview.webp": "web_preview",
    "videos/web/main.mp4": "main_web",
    "videos/web/spectral_sweep.mp4": "spectral_sweep_web",
    "videos/hq/main.mp4": "main_hq",
    "videos/hq/spectral_sweep.mp4": "spectral_sweep_hq",
}
# What a truncated or corrupted live metadata file holds in these tests.
CORRUPT_JSON = '{"schema_version": 2, "assets": ['

EMBER_MEDIA_ROLES = dict(zip(run.EMBER_MEDIA_FILES, run.EMBER_MANIFEST_ROLES, strict=True))
SPECTRAL_FILES = tuple(f"spectral/{b:02d}_{380 + 5 * b}nm.png" for b in range(64))


def manifest_entry(path: str, role: str, tag: str) -> dict[str, object]:
    """One fake metadata/assets.json entry; `tag` marks which render produced it."""
    return {
        "path": path,
        "kind": "video" if path.endswith(".mp4") else "image",
        "role": role,
        "format": path.rsplit(".", 1)[1] if "." in path else "png",
        "duration_seconds": 30.033333333333335,
        "bytes": 12,
        "sha256": f"{tag}:{path}",
    }


def view_members(**changes: str | None) -> str:
    """The view fields of a fake nft_traits.json, as the JSON text of "generation" members:
    DEFAULT_VIEW with `changes` (JSON text per field; None leaves that field out)."""
    fields = {**DEFAULT_VIEW, **changes}

    def members(*names: str) -> list[str]:
        return [f'"{name}": {fields[name]}' for name in names if fields[name] is not None]

    structure = ['"primary": "Orbit Ribbons"', *members("stack_label")]
    drift = [
        '"enabled": true',
        *members("mode", "scale", "arc_fraction", "orbit_eccentricity"),
        '"randomized": true',
    ]
    return ", ".join(
        [
            f'"structure": {{{", ".join(structure)}}}',
            *members("projection", "symmetry"),
            f'"drift": {{{", ".join(drift)}}}',
        ]
    )


def traits_text(seed: str, tag: str, masses: str = DEFAULT_MASSES, view: str | None = None) -> str:
    """A fake metadata/nft_traits.json: the orbit fields (`masses` as JSON text) and the view
    fields (`view` as view_members() text, the default view if None)."""
    return (
        f'{{"seed": "0x{seed}", "pipeline_version": "1.1.0", "tag": "{tag}", '
        f'"simulation": {{"masses": {masses}, "dt": 0.001}}, '
        f'"generation": {{"borda": {{"selected_index": 7, "retry_count": 1}}, '
        f"{view_members() if view is None else view}}}}}"
    )


def certificate_text(algorithm: str, tag: str) -> str:
    """A fake metadata/ember.json, laid out like the generator's (serde_json's pretty printer):
    the top-level key is the line `  "algorithm": "<id>",`. The nested "algorithm" is a decoy that
    must never be read as the certificate's."""
    certificate = {
        "schema_version": 4,
        "edition": "ember",
        "algorithm": algorithm,
        "contract": f"the {tag} render",
        "config": {"look": {"algorithm": "ember-v0"}},
    }
    return json.dumps(certificate, indent=2) + "\n"


def write_package(
    package: Path,
    seed: str,
    *,
    tag: str,
    ember_files: bool,
    ember_manifest: bool,
    masses: str = DEFAULT_MASSES,
    view: str | None = None,
    extra_entries: Sequence[dict[str, object]] = (),
    algorithm: str = CURRENT_ALGORITHM,
) -> None:
    """Write a fake package: every core file, 64 spectral bins, metadata (traits_text() of
    `masses` and `view`), optionally the ember edition's files (its certificate records
    `algorithm`) and manifest entries. Every file's content names `tag`."""
    for path in (*CORE_MEDIA_ROLES, *SPECTRAL_FILES):
        target = package / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{path} {tag}\n", encoding="utf-8")
    metadata = package / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "generation.json").write_text(
        json.dumps({"seed": f"0x{seed}", "tag": tag}), encoding="utf-8"
    )
    (metadata / "nft_traits.json").write_text(
        traits_text(seed, tag, masses, view), encoding="utf-8"
    )
    entries = [manifest_entry(path, role, tag) for path, role in CORE_MEDIA_ROLES.items()]
    entries.append(manifest_entry("spectral/", "spectral_gallery", tag))
    entries.extend(extra_entries)
    if ember_manifest:
        entries.extend(manifest_entry(p, r, tag) for p, r in EMBER_MEDIA_ROLES.items())
    manifest = {"schema_version": 2, "generated_at": f"{tag}-time", "assets": entries}
    (metadata / "assets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if ember_files:
        for path in run.EMBER_MEDIA_FILES:
            target = package / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{path} {tag}\n", encoding="utf-8")
        (package / run.EMBER_CERTIFICATE).write_text(
            certificate_text(algorithm, tag), encoding="utf-8"
        )


def remote_listing(seed: str, files: Sequence[str]) -> set[str]:
    """The list_remote_files() paths of a remote package holding `files` and every bin."""
    return {f"0x{seed}/{path}" for path in (*files, *SPECTRAL_FILES)}


# ---------------------------------------------------------------------------
# Fakes (run as scripts on PATH through tiny launchers)
# ---------------------------------------------------------------------------


def _log_call(entry: list[object]) -> None:
    """Append one call to the FAKE_LOG file (JSON lines)."""
    log_path = os.environ.get("FAKE_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def _probe_lock(path: str) -> list[bool]:
    """[whether the lock at `path` is free, whether this process holds a descriptor of it]."""
    lock_stat = Path(path).stat()
    inherited = False
    for fd in range(256):
        try:
            fd_stat = os.fstat(fd)
        except OSError:
            continue
        if (fd_stat.st_dev, fd_stat.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
            inherited = True
    probe = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = True
    except BlockingIOError:
        free = False
    finally:
        os.close(probe)
    return [free, inherited]


def fake_generator(argv: list[str]) -> int:
    """three_body_problem: writes output/<name>/ as FAKE_GEN_MODE[_<seed>] says.

    Modes: complete (exit 0), core_only (exit 3), stray (exit 3 leaving an ember file), stale
    (exit 0 without the ember edition), no_ember_files (exit 0, ember entries but no ember
    files), fail (exit 1), sleep (30 s), signal (killed by SIGTERM). With FAKE_GEN_STALE=1 its
    --help does not list --no-ember. FAKE_GEN_MASSES_<seed> and FAKE_GEN_VIEW_<seed> give the
    masses and the view (view_members() text) of that seed's nft_traits.json, in place of
    DEFAULT_MASSES and DEFAULT_VIEW. FAKE_GEN_CORRUPT names a file it overwrites with invalid
    JSON while rendering (a live file that changes during the render). FAKE_GEN_LOCK_PROBE names
    run.py's lock file: each render logs whether it is free and whether the generator holds it.
    FAKE_GEN_EMBER_ALGORITHM (default CURRENT_ALGORITHM) is what --ember-algorithm prints and what
    its certificates record; empty, or with FAKE_GEN_STALE=1, the binary predates that flag and
    its argument parser rejects it (exit 2, like clap).
    """
    algorithm = os.environ.get("FAKE_GEN_EMBER_ALGORITHM", CURRENT_ALGORITHM)
    if "--help" in argv:
        print("Usage: three_body_problem [OPTIONS]\n      --image-only")
        if os.environ.get("FAKE_GEN_STALE") != "1":
            print("      --no-ember   Skip the ember edition")
        return 0
    if "--ember-algorithm" in argv:
        if not algorithm or os.environ.get("FAKE_GEN_STALE") == "1":
            print("error: unexpected argument '--ember-algorithm' found", file=sys.stderr)
            return 2
        print(algorithm)
        return 0
    seed = argv[argv.index("--seed") + 1].removeprefix("0x")
    name = argv[argv.index("--output") + 1]
    mode = os.environ.get(f"FAKE_GEN_MODE_{seed}", os.environ.get("FAKE_GEN_MODE", "complete"))
    _log_call(["generate", seed])
    lock_probe = os.environ.get("FAKE_GEN_LOCK_PROBE")
    if lock_probe:
        _log_call(["lock_probe", *_probe_lock(lock_probe)])
    corrupt = os.environ.get("FAKE_GEN_CORRUPT")
    if corrupt:
        Path(corrupt).write_text(CORRUPT_JSON, encoding="utf-8")
    if mode == "fail":
        return 1
    if mode == "sleep":
        time.sleep(30)
        return 0
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        return 0
    package = Path("output") / name
    write_package(
        package,
        seed,
        tag=os.environ.get("FAKE_GEN_TAG", "new"),
        ember_files=mode == "complete",
        ember_manifest=mode in ("complete", "no_ember_files"),
        masses=os.environ.get(f"FAKE_GEN_MASSES_{seed}", DEFAULT_MASSES),
        view=os.environ.get(f"FAKE_GEN_VIEW_{seed}"),
        algorithm=algorithm,
    )
    if mode == "stray":
        (package / run.EMBER_MEDIA_FILES[0]).write_text("partial", encoding="utf-8")
    return run.GENERATOR_EXIT_EMBER_FAILED if mode in ("core_only", "stray") else 0


def fake_ssh(argv: list[str]) -> int:
    """ssh: runs the remote command locally with /bin/sh.

    Exits 255 like a failed connection with FAKE_SSH_FAIL=1, or for a command that starts with
    FAKE_SSH_FAIL_ON. A command with an `&&` step that contains FAKE_SSH_CUT_AFTER is cut off
    like a connection lost in the middle of it: the steps up to and including that one run, the
    rest do not, and ssh exits 255.
    """
    index = 0
    while index < len(argv) and argv[index].startswith("-"):
        index += 2 if argv[index] in ("-o", "-l", "-p", "-i", "-F") else 1
    command = " ".join(argv[index + 1 :])
    _log_call(["ssh", command])
    fail_on = os.environ.get("FAKE_SSH_FAIL_ON")
    if os.environ.get("FAKE_SSH_FAIL") == "1" or (fail_on and command.startswith(fail_on)):
        print("ssh: connect to host fake port 22: Connection refused", file=sys.stderr)
        return 255
    cut_after = os.environ.get("FAKE_SSH_CUT_AFTER")
    if cut_after and cut_after in command:
        steps = command.split(" && ")
        done = next(index for index, step in enumerate(steps) if cut_after in step) + 1
        subprocess.run(["/bin/sh", "-c", " && ".join(steps[:done])], check=False)
        print("ssh: Connection to fake closed by remote host.", file=sys.stderr)
        return 255
    return subprocess.run(["/bin/sh", "-c", command], check=False).returncode


def _copy_file(source: Path, target: Path, fail_on: str | None) -> bool:
    """Copy one file like scp; False, leaving a truncated copy, if its path contains `fail_on`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if fail_on and fail_on in str(source):
        data = source.read_bytes()
        target.write_bytes(data[: len(data) // 2])
        return False
    shutil.copyfile(source, target)
    return True


def fake_scp(argv: list[str]) -> int:
    """scp: copies the sources to the local path named by the user@host:path target.

    The target is a directory to copy into or, for one file, the file to write. Fails like scp on
    a missing destination or a directory source without -r. A transfer that reaches a file whose
    path contains FAKE_SCP_FAIL_ON is interrupted: the files before it are copied, that one is
    left truncated, and scp fails.
    """
    paths: list[str] = []
    recursive = False
    index = 0
    while index < len(argv):
        if argv[index] in ("-o", "-P", "-i", "-F", "-l"):
            index += 2
            continue
        if argv[index].startswith("-"):
            recursive = recursive or argv[index] == "-r"
            index += 1
            continue
        paths.append(argv[index])
        index += 1
    *sources, target = paths
    destination = Path(target.split(":", 1)[1])
    _log_call(["scp", sources, str(destination)])
    fail_on = os.environ.get("FAKE_SCP_FAIL_ON")
    if destination.is_dir():
        copies = [(Path(source), destination / Path(source).name) for source in sources]
    elif len(sources) == 1 and not target.endswith("/") and destination.parent.is_dir():
        copies = [(Path(sources[0]), destination)]
    else:
        print(f"scp: {destination}: No such file or directory", file=sys.stderr)
        return 1
    for source, copy in copies:
        if not source.is_dir():
            files = {source: copy}
        elif recursive:
            files = {
                p: copy / p.relative_to(source) for p in sorted(source.rglob("*")) if p.is_file()
            }
        else:
            print(f"scp: {source}: not a regular file", file=sys.stderr)
            return 1
        for file, file_copy in files.items():
            if not _copy_file(file, file_copy, fail_on):
                print("scp: Connection closed", file=sys.stderr)
                return 1
    return 0


def fake_find(argv: list[str]) -> int:
    """find: always fails, like find on an unreadable directory."""
    _log_call(["find", argv])
    print("find: .: Permission denied", file=sys.stderr)
    return 1


FAKES = {
    "three_body_problem": fake_generator,
    "ssh": fake_ssh,
    "scp": fake_scp,
    "find": fake_find,
}


def install_fake(bin_dir: Path, name: str) -> Path:
    """Write an executable launcher for FAKES[name] into `bin_dir`."""
    launcher = bin_dir / name
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import test_run\n"
        f"sys.exit(test_run.FAKES[{name!r}](sys.argv[1:]))\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def setUpModule() -> None:
    """Keep run.py's log records out of the test output (assertLogs still sees them)."""
    run.log.addHandler(logging.NullHandler())
    run.log.propagate = False


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------


class SyncTestCase(unittest.TestCase):
    """A temporary working directory and "remote" directory, and the fakes on PATH."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.remote = self.root / "remote" / "assets"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls_file = self.root / "calls.jsonl"
        self.generator = str(install_fake(self.bin, "three_body_problem"))
        for name in ("ssh", "scp"):
            install_fake(self.bin, name)

        previous_cwd = Path.cwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, previous_cwd)

        env = mock.patch.dict(
            os.environ,
            {"PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"},
        )
        env.start()
        self.addCleanup(env.stop)
        for key in list(os.environ):
            if key.startswith(("FAKE_", "COSMICSIG_")) or key == "SSH_OPTS_EXTRA":
                del os.environ[key]
        os.environ["FAKE_LOG"] = str(self.calls_file)

        for patcher in (
            mock.patch.object(run, "shutdown_requested", False),
            mock.patch("time.sleep"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    # --- the fake remote ---------------------------------------------------

    def remote_package(
        self,
        seed: str,
        *,
        ember: bool = False,
        masses: str = DEFAULT_MASSES,
        extra_entries: Sequence[dict[str, object]] = (),
        algorithm: str = CURRENT_ALGORITHM,
    ) -> Path:
        """A live package on the fake remote, tagged "live" (its certificate records
        `algorithm`)."""
        package = self.remote / f"0x{seed}"
        write_package(
            package,
            seed,
            tag="live",
            ember_files=ember,
            ember_manifest=ember,
            masses=masses,
            extra_entries=extra_entries,
            algorithm=algorithm,
        )
        return package

    def remote_v2_package(
        self, seed: str, *, extra_entries: Sequence[dict[str, object]] = ()
    ) -> Path:
        """A live package holding an ember-v2 edition, tagged "live": the EMBER_V2_* literals
        (its manifest lists `extra_entries` after the five ember-v2 entries)."""
        ember = [manifest_entry(path, role, "live") for path, role in EMBER_V2_MEDIA_ROLES.items()]
        package = self.remote_package(seed, extra_entries=[*ember, *extra_entries])
        for path in EMBER_V2_MEDIA_ROLES:
            (package / path).write_text(f"{path} live\n", encoding="utf-8")
        (package / "metadata/ember.json").write_text(EMBER_V2_CERTIFICATE, encoding="utf-8")
        return package

    def package_files(self, seed: str) -> dict[str, str]:
        """The remote package of `seed`: {path relative to the package: content}."""
        prefix = f"0x{seed}/"
        return {
            path.removeprefix(prefix): content
            for path, content in self.remote_snapshot().items()
            if path.startswith(prefix)
        }

    def ember_upload_commands(self, seed: str, retired: Sequence[str] = ()) -> list[str]:
        """The two ssh commands of an ember-mode upload into the package of `seed`, whose live
        edition has the files `retired` that the current look does not: the one before the
        transfers (leftover .part files go, the directories are made), and the one after them,
        which swaps the staged edition in (the old certificate and `retired` go, then every
        staged file is renamed into place, in EMBER_UPLOAD_ORDER)."""
        package = f"{self.remote}/0x{seed}"
        parts = " ".join(f"{package}/{path}.part" for path in EMBER_UPLOAD_ORDER)
        directories = sorted(
            {posixpath.dirname(f"{package}/{path}") for path in EMBER_UPLOAD_ORDER}
        )
        removed = " ".join(f"{package}/{path}" for path in ("metadata/ember.json", *retired))
        renames = [
            f"mv -f -- {package}/{path}.part {package}/{path}" for path in EMBER_UPLOAD_ORDER
        ]
        return [
            f"rm -f -- {parts} && mkdir -p -- {' '.join(directories)}",
            " && ".join([f"rm -f -- {removed}", *renames]),
        ]

    def changes(self) -> list[str]:
        """The remote commands that changed the fake remote (every one but the reads)."""
        return [c for c in self.ssh_commands() if not c.startswith(("cat -- ", "cd "))]

    def staged(self) -> list[str]:
        """Per scp call, its destination relative to the fake remote."""
        return [
            str(Path(str(destination)).relative_to(self.remote))
            for _sources, destination in self.calls("scp")
        ]

    def listing(self) -> set[str]:
        """run.list_remote_files() of the fake remote, which must succeed."""
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        return remote

    def remote_snapshot(self) -> dict[str, str]:
        """Every remote file (relative path) and its content."""
        if not self.remote.is_dir():
            return {}
        return {
            str(path.relative_to(self.remote)): path.read_text(encoding="utf-8")
            for path in sorted(self.remote.rglob("*"))
            if path.is_file()
        }

    def remote_part_files(self) -> list[str]:
        """The staged (not yet renamed) files on the fake remote."""
        return [path for path in self.remote_snapshot() if path.endswith(run.PART_SUFFIX)]

    def call_log(self) -> list[list[object]]:
        """Every logged call of the fakes, in order: [kind, *details]."""
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def forget_calls(self) -> None:
        """Start the call log afresh."""
        self.calls_file.unlink(missing_ok=True)

    def calls(self, kind: str) -> list[list[object]]:
        """The logged calls of one fake ("generate", "ssh", "scp", "find"), in order."""
        return [entry[1:] for entry in self.call_log() if entry[0] == kind]

    def ssh_commands(self) -> list[str]:
        """The remote commands run through the fake ssh, in order."""
        return [str(call[0]) for call in self.calls("ssh")]

    def uploaded(self) -> list[list[str]]:
        """Per scp call, the uploaded paths relative to the local package directory."""
        batches: list[list[str]] = []
        for sources, _destination in self.calls("scp"):
            assert isinstance(sources, list)
            batches.append([str(Path(s)).split("/0x", 1)[1].split("/", 1)[-1] for s in sources])
        return batches

    def generated(self) -> list[object]:
        """The seeds the fake generator ran for, in order."""
        return [call[0] for call in self.calls("generate")]

    def process(
        self,
        seed: str,
        *,
        backfill: run.BackfillMode | None = None,
        ember_capable: bool = True,
    ) -> run.Outcome:
        """run.process_seed() against the fake remote with the fake generator."""
        return run.process_seed(
            seed,
            [self.generator],
            "fakehost",
            "fakeuser",
            str(self.remote),
            60,
            False,
            backfill=backfill,
            ember_capable=ember_capable,
        )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlanningTests(unittest.TestCase):
    """find_missing_seeds, plan_seed_queue and given_up_seeds."""

    def test_find_missing_seeds_splits_urgent_and_backfill_in_api_order(self) -> None:
        remote = (
            remote_listing(SEED_A, run.REQUIRED_PACKAGE_FILES)  # complete
            | remote_listing(SEED_B, run.CORE_PACKAGE_FILES)  # predates the ember edition
            | remote_listing(SEED_C, run.CORE_PACKAGE_FILES[1:])  # lacks master.png
            | remote_listing(SEED_D, [*run.CORE_PACKAGE_FILES, run.EMBER_MEDIA_FILES[0]])
        )
        new_mint = "e" * 64
        urgent, backfill = run.find_missing_seeds(
            [new_mint, SEED_D, SEED_C, SEED_B, SEED_A], remote
        )
        self.assertEqual(urgent, [new_mint, SEED_C])
        self.assertEqual(backfill, [SEED_D, SEED_B])

    def test_a_package_missing_spectral_bins_is_urgent(self) -> None:
        remote = {f"0x{SEED_A}/{path}" for path in run.CORE_PACKAGE_FILES}
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([SEED_A], []))

    def test_urgent_seeds_come_first_and_the_backfill_is_capped(self) -> None:
        urgent, backfill = ["u1", "u2"], ["b1", "b2", "b3"]
        empty = run.BackfillLedger()
        self.assertEqual(run.plan_seed_queue(urgent, backfill, 1, empty), ["u1", "u2", "b1"])
        self.assertEqual(run.plan_seed_queue(urgent, backfill, 0, empty), ["u1", "u2"])
        self.assertEqual(run.plan_seed_queue([], backfill, 5, empty), backfill)

    def test_failed_backfill_seeds_go_last_fewest_failed_runs_first(self) -> None:
        ledger = run.BackfillLedger(ember_failures={"b1": 2}, other_failures={"b2": 1})
        self.assertEqual(run.plan_seed_queue([], ["b1", "b2", "b3"], 3, ledger), ["b3", "b2", "b1"])

    def test_a_seed_that_always_fails_cannot_starve_the_others(self) -> None:
        # b1 crashed five times (never an ember attempt); b2's ember edition failed once.
        ledger = run.BackfillLedger(ember_failures={"b2": 1}, other_failures={"b1": 5})
        self.assertEqual(run.plan_seed_queue([], ["b1", "b2"], 1, ledger), ["b2"])
        self.assertEqual(run.given_up_seeds(["b1", "b2"], ledger, 3), [])

    def test_seeds_at_the_attempt_cap_are_excluded(self) -> None:
        ledger = run.BackfillLedger(ember_failures={"b1": 3, "b2": 2})
        self.assertEqual(run.plan_seed_queue(["u1"], ["b1", "b2"], 5, ledger), ["u1", "b2"])
        self.assertEqual(run.given_up_seeds(["b1", "b2"], ledger, 3), ["b1"])
        self.assertEqual(run.plan_seed_queue([], ["b1", "b2"], 5, ledger, 4), ["b2", "b1"])
        self.assertEqual(run.given_up_seeds(["b1", "b2"], ledger, 4), [])

    def test_a_package_of_the_previous_look_lacks_only_the_slow_film(self) -> None:
        remote = remote_listing(SEED_A, [*run.CORE_PACKAGE_FILES, *EMBER_V2_FILES])
        self.assertEqual(
            run.missing_remote_package_parts(SEED_A, remote), ["videos/web/ember_slow.mp4"]
        )
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([], [SEED_A]))

    def test_an_identity_mismatch_gives_a_seed_up_at_once(self) -> None:
        ledger = run.BackfillLedger(ember_failures={"b1": 1}, identity_mismatches={"b1"})
        self.assertEqual(run.plan_seed_queue(["u1"], ["b1", "b2"], 5, ledger), ["u1", "b2"])
        self.assertEqual(run.given_up_seeds(["b1", "b2"], ledger, 3), ["b1"])
        # A higher attempt cap does not bring it back: this binary regenerates the same package.
        self.assertEqual(run.given_up_seeds(["b1", "b2"], ledger, 100), ["b1"])

    def test_other_failures_never_give_a_seed_up(self) -> None:
        ledger = run.BackfillLedger(other_failures={"b1": 10})
        self.assertEqual(run.plan_seed_queue([], ["b1"], 1, ledger), ["b1"])
        self.assertEqual(run.given_up_seeds(["b1"], ledger, 3), [])


# ---------------------------------------------------------------------------
# The ember failure ledger
# ---------------------------------------------------------------------------


class LedgerTests(unittest.TestCase):
    """backfill_failures.json: load, save, reset on a generator change, corrupt files."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "backfill_failures.json"
        binary = self.dir / "three_body_problem"
        binary.write_bytes(b"generator v1")
        self.binary = binary
        identity = run.GeneratorIdentity.of(str(binary))
        assert identity is not None
        self.identity = identity

    def load(self, identity: run.GeneratorIdentity | None = None) -> run.BackfillLedger:
        return run.load_backfill_ledger(identity or self.identity, self.path)

    def test_counts_round_trip_for_the_same_generator(self) -> None:
        ledger = run.BackfillLedger({SEED_B: 2, SEED_A: 1}, {SEED_C: 4})
        run.save_backfill_ledger(ledger, self.identity, self.path)
        self.assertEqual(self.load(), ledger)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["generator"]["path"], str(self.binary.resolve()))
        self.assertEqual(list(data["ember_failures"]), [SEED_A, SEED_B])
        self.assertEqual(data["other_failures"], {SEED_C: 4})
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_identity_mismatches_round_trip_and_older_files_are_read(self) -> None:
        ledger = run.BackfillLedger({SEED_A: 1, SEED_B: 1}, {}, {SEED_B, SEED_A})
        run.save_backfill_ledger(ledger, self.identity, self.path)
        self.assertEqual(self.load(), ledger)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["identity_mismatches"], [SEED_A, SEED_B])
        self.assertNotIn("orbit_mismatches", data)

        del data["identity_mismatches"]  # a file written before the field existed
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load(), run.BackfillLedger({SEED_A: 1, SEED_B: 1}))

        # A file written while the check compared the orbit only: its list is read under its
        # earlier name, unless the file has the current one too.
        data["orbit_mismatches"] = [SEED_B]
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load().identity_mismatches, {SEED_B})
        data["identity_mismatches"] = [SEED_A]
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load().identity_mismatches, {SEED_A})
        del data["orbit_mismatches"]

        data["identity_mismatches"] = [SEED_A, 7, None]  # malformed entries are dropped
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load().identity_mismatches, {SEED_A})
        data["identity_mismatches"] = {"not": "a list"}
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load().identity_mismatches, set())

    def test_counts_reset_when_the_generator_changes(self) -> None:
        run.save_backfill_ledger(
            run.BackfillLedger({SEED_A: 3}, {SEED_B: 1}), self.identity, self.path
        )
        self.binary.write_bytes(b"generator v2, rebuilt")
        rebuilt = run.GeneratorIdentity.of(str(self.binary))
        self.assertNotEqual(rebuilt, self.identity)
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.load(rebuilt), run.BackfillLedger())
        self.assertIn("counts reset", "\n".join(logs.output))

        self.binary.write_bytes(b"generator v1")  # the same size, touched later
        touched = self.identity.mtime_ns + 1_000_000_000
        os.utime(self.binary, ns=(touched, touched))
        self.assertNotEqual(run.GeneratorIdentity.of(str(self.binary)), self.identity)

    def test_a_missing_ledger_is_empty(self) -> None:
        self.assertEqual(self.load(), run.BackfillLedger())

    def test_a_corrupt_or_foreign_ledger_is_empty(self) -> None:
        generator = self.identity.to_json()
        for text in (
            "{not json",
            "[1, 2]",
            json.dumps({SEED_A: 2}),
            json.dumps({"generator": generator, "failures": {SEED_A: 2}}),
            json.dumps({"generator": generator, "ember_failures": {}, "other_failures": []}),
        ):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertLogs(run.log, level="WARNING"):
                    self.assertEqual(self.load(), run.BackfillLedger())

    def test_malformed_counts_are_dropped(self) -> None:
        data = {
            "generator": self.identity.to_json(),
            "ember_failures": {SEED_A: 2, SEED_B: 0, SEED_C: True, SEED_D: "3", "e": -1},
            "other_failures": {SEED_B: 1.5, SEED_C: 1},
        }
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load(), run.BackfillLedger({SEED_A: 2}, {SEED_C: 1}))

    def test_record(self) -> None:
        ledger = run.BackfillLedger({SEED_A: 1})
        self.assertTrue(ledger.record(SEED_A, run.Outcome.CORE_ONLY, backfill=True))
        self.assertTrue(ledger.record(SEED_A, run.Outcome.EMBER_FAILED, backfill=True))
        self.assertTrue(ledger.record(SEED_A, run.Outcome.FAILED, backfill=True))
        self.assertEqual(ledger, run.BackfillLedger({SEED_A: 3}, {SEED_A: 1}))
        self.assertEqual(ledger.failed_runs(SEED_A), 4)

        # An urgent seed: exit 3 counts (it becomes a backfill seed), a failure does not.
        self.assertTrue(ledger.record(SEED_B, run.Outcome.CORE_ONLY, backfill=False))
        self.assertFalse(ledger.record(SEED_C, run.Outcome.FAILED, backfill=False))
        self.assertEqual(ledger, run.BackfillLedger({SEED_A: 3, SEED_B: 1}, {SEED_A: 1}))

        for outcome in (run.Outcome.EMBER_FAILED, run.Outcome.FAILED):
            with self.assertLogs(run.log, level="INFO"):
                self.assertFalse(ledger.record(SEED_A, outcome, backfill=True, interrupted=True))
        self.assertEqual(ledger.failed_runs(SEED_A), 4)

        self.assertTrue(ledger.record(SEED_A, run.Outcome.COMPLETE, backfill=True))
        self.assertFalse(ledger.record(SEED_C, run.Outcome.COMPLETE, backfill=True))
        self.assertEqual(ledger, run.BackfillLedger({SEED_B: 1}))

        # An identity mismatch counts one attempt and gives the seed up; COMPLETE forgets it.
        self.assertTrue(ledger.record(SEED_C, run.Outcome.IDENTITY_MISMATCH, backfill=True))
        self.assertEqual(ledger, run.BackfillLedger({SEED_B: 1, SEED_C: 1}, {}, {SEED_C}))
        self.assertTrue(ledger.given_up(SEED_C, 3))
        with self.assertLogs(run.log, level="INFO"):
            self.assertFalse(
                ledger.record(
                    SEED_D, run.Outcome.IDENTITY_MISMATCH, backfill=True, interrupted=True
                )
            )
        self.assertFalse(ledger.given_up(SEED_D, 3))
        self.assertTrue(ledger.record(SEED_C, run.Outcome.COMPLETE, backfill=True))
        self.assertEqual(ledger, run.BackfillLedger({SEED_B: 1}))

    def test_retain_keeps_only_the_waiting_seeds(self) -> None:
        ledger = run.BackfillLedger(
            {SEED_A: 1, SEED_B: 2}, {SEED_B: 1, SEED_C: 3}, {SEED_A, SEED_B}
        )
        ledger.retain({SEED_B})
        self.assertEqual(ledger, run.BackfillLedger({SEED_B: 2}, {SEED_B: 1}, {SEED_B}))


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


class GeneratorTests(SyncTestCase):
    """generate() exit statuses and the ember capability probe."""

    def generate(self, mode: str, timeout: int = 60) -> run.Outcome:
        os.environ["FAKE_GEN_MODE"] = mode
        return run.generate([self.generator], SEED_A, timeout)

    def test_exit_statuses(self) -> None:
        self.assertIs(self.generate("complete"), run.Outcome.COMPLETE)
        with self.assertLogs(run.log, level="WARNING"):
            self.assertIs(self.generate("core_only"), run.Outcome.CORE_ONLY)
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.generate("fail"), run.Outcome.FAILED)
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertIs(self.generate("signal"), run.Outcome.FAILED)
        self.assertIn("killed by signal", "\n".join(logs.output))

    def test_timeout(self) -> None:
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertIs(self.generate("sleep", timeout=1), run.Outcome.FAILED)
        self.assertIn("TIMEOUT", "\n".join(logs.output))

    def test_capability_probe(self) -> None:
        self.assertTrue(run.generator_supports_ember([self.generator]))
        os.environ["FAKE_GEN_STALE"] = "1"
        self.assertFalse(run.generator_supports_ember([self.generator]))
        with self.assertLogs(run.log, level="ERROR"):
            self.assertFalse(run.generator_supports_ember([str(self.root / "missing")]))

    def ember_algorithm(self, printed: str | None = None) -> str | None:
        """run.generator_ember_algorithm() of the fake generator, which prints `printed`."""
        if printed is not None:
            os.environ["FAKE_GEN_EMBER_ALGORITHM"] = printed
        return run.generator_ember_algorithm([self.generator])

    def test_ember_algorithm_probe(self) -> None:
        self.assertEqual(self.ember_algorithm(), CURRENT_ALGORITHM)
        self.assertEqual(self.ember_algorithm("ember-v10"), "ember-v10")
        self.assertEqual(self.ember_algorithm("  ember-v4\n"), "ember-v4")

    def test_ember_algorithm_probe_accepts_nothing_but_one_id(self) -> None:
        for printed in ("Ember v2", "ember-v2 (sumi)", "ember-v", "EMBER-V2", "ember-v2\nember-v3"):
            with self.subTest(printed=printed), self.assertLogs(run.log, "WARNING") as logs:
                self.assertIsNone(self.ember_algorithm(printed))
            self.assertIn("not an ember algorithm id", "\n".join(logs.output))
            self.assertIn("stale ember editions cannot be detected", "\n".join(logs.output))

    def test_a_generator_that_predates_the_ember_algorithm_probe(self) -> None:
        # The argument parser of an older binary rejects the flag with status 2.
        with self.assertLogs(run.log, level="WARNING") as logs:
            self.assertIsNone(self.ember_algorithm(""))
        self.assertIn("exited with rc=2 (error: unexpected argument", "\n".join(logs.output))
        self.assertIn("so no live ember edition is withdrawn", "\n".join(logs.output))
        with self.assertLogs(run.log, level="WARNING") as logs:
            self.assertIsNone(run.generator_ember_algorithm([str(self.root / "missing")]))
        self.assertIn("could not be run", "\n".join(logs.output))


# ---------------------------------------------------------------------------
# Remote listing
# ---------------------------------------------------------------------------


class ListRemoteFilesTests(SyncTestCase):
    """list_remote_files: listing, a missing directory, and failures."""

    def list(self) -> set[str] | None:
        return run.list_remote_files("fakehost", "fakeuser", str(self.remote))

    def test_lists_package_files(self) -> None:
        self.remote_package(SEED_A, ember=True)
        (self.remote / "stray-top-level-file").write_text("x", encoding="utf-8")
        files = self.list()
        assert files is not None
        self.assertEqual(files, remote_listing(SEED_A, run.REQUIRED_PACKAGE_FILES))

    def test_a_missing_remote_directory_is_a_failure(self) -> None:
        # An empty listing would make every published package look new and replace it in full.
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIsNone(self.list())

    def test_ssh_failure_is_none(self) -> None:
        os.environ["FAKE_SSH_FAIL"] = "1"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIsNone(self.list())

    def test_find_failure_is_none(self) -> None:
        self.remote.mkdir(parents=True)
        install_fake(self.bin, "find")
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertIsNone(self.list())
        self.assertIn("Permission denied", "\n".join(logs.output))


# ---------------------------------------------------------------------------
# One seed, end to end
# ---------------------------------------------------------------------------


class ProcessSeedTests(SyncTestCase):
    """process_seed against the fake remote."""

    def assert_certificate_last(self) -> None:
        batches = self.uploaded()
        self.assertEqual(batches[-1], [run.EMBER_CERTIFICATE])
        for batch in batches[:-1]:
            self.assertNotIn(run.EMBER_CERTIFICATE, batch)

    def test_new_mint_uploads_the_whole_package_media_first_certificate_last(self) -> None:
        self.assertIs(self.process(SEED_A), run.Outcome.COMPLETE)
        prepare, *renames = self.ssh_commands()
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        self.assertEqual(remote, remote_listing(SEED_A, run.REQUIRED_PACKAGE_FILES))
        self.assertEqual(
            self.uploaded(),
            [
                ["images", "spectral", "videos"],
                ["metadata/generation.json"],
                ["metadata/nft_traits.json"],
                [run.ASSET_MANIFEST],
                [run.EMBER_CERTIFICATE],
            ],
        )
        # The manifest and certificate are deleted first; the metadata files and then the
        # certificate are renamed into place, the manifest after the other metadata files.
        package = f"{self.remote}/0x{SEED_A}"
        self.assertTrue(
            prepare.startswith(
                f"rm -f -- {package}/{run.ASSET_MANIFEST} {package}/{run.EMBER_CERTIFICATE} && "
                "mkdir -p -- "
            ),
            prepare,
        )

        def rename(*names: str) -> str:
            return " && ".join(
                f"mv -f -- {package}/metadata/{name}.part {package}/metadata/{name}"
                for name in names
            )

        self.assertEqual(
            renames,
            [rename("generation.json", "nft_traits.json", "assets.json"), rename("ember.json")],
        )
        self.assertEqual(self.remote_part_files(), [])
        self.assertFalse((run.LOCAL_OUTPUT_DIR / f"0x{SEED_A}").exists())

    def test_core_only_mint_deletes_stale_remote_ember_files_before_uploading(self) -> None:
        package = self.remote_package(SEED_A, ember=True)
        (package / run.NFT_TRAITS).unlink()  # urgent: a core file is missing
        os.environ["FAKE_GEN_MODE"] = "stray"
        with self.assertLogs(run.log, level="WARNING"):
            self.assertIs(self.process(SEED_A), run.Outcome.CORE_ONLY)
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        self.assertEqual(remote, remote_listing(SEED_A, run.CORE_PACKAGE_FILES))
        manifest = json.loads((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"))
        self.assertFalse(any(run.is_ember_asset(entry) for entry in manifest["assets"]))
        # The first remote call after the render removes every ember file, before any scp.
        calls = self.call_log()
        kinds = [entry[0] for entry in calls]
        self.assertEqual(kinds[:2], ["generate", "ssh"])
        removal = str(calls[1][1])
        self.assertTrue(removal.startswith("rm -f -- "), removal)
        for path in run.EMBER_PACKAGE_FILES:
            self.assertIn(f"0x{SEED_A}/{path}", removal)
        self.assertLess(1, kinds.index("scp"))
        self.assertEqual(self.uploaded()[-1], [run.ASSET_MANIFEST])

    def test_core_only_mint_uploads_nothing_if_the_remote_cleanup_fails(self) -> None:
        os.environ["FAKE_GEN_MODE"] = "core_only"
        os.environ["FAKE_SSH_FAIL"] = "1"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A), run.Outcome.FAILED)
        self.assertEqual(self.calls("scp"), [])
        self.assertEqual(self.remote_snapshot(), {})

    def test_ember_backfill_uploads_only_the_ember_edition_and_a_merged_manifest(self) -> None:
        stale_entry = manifest_entry("videos/web/ember.mp4", "ember_web", "stale")
        package = self.remote_package(SEED_A, extra_entries=[stale_entry])
        live = self.remote_snapshot()
        live_manifest = json.loads((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"))

        outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)

        self.assertIs(outcome, run.Outcome.COMPLETE)
        remote = self.remote_snapshot()
        for path, content in live.items():
            if not path.endswith(run.ASSET_MANIFEST):
                self.assertEqual(remote[path], content, path)
        for path in run.EMBER_MEDIA_FILES:
            self.assertEqual(remote[f"0x{SEED_A}/{path}"], f"{path} new\n")
        self.assertEqual(
            remote[f"0x{SEED_A}/{run.EMBER_CERTIFICATE}"],
            certificate_text(CURRENT_ALGORITHM, "new"),
        )

        merged = json.loads((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"))
        kept = [entry for entry in live_manifest["assets"] if not run.is_ember_asset(entry)]
        self.assertEqual(merged["schema_version"], live_manifest["schema_version"])
        self.assertEqual(merged["generated_at"], "new-time")
        self.assertEqual(merged["assets"][: len(kept)], kept)
        ember = merged["assets"][len(kept) :]
        self.assertEqual([entry["role"] for entry in ember], list(run.EMBER_MANIFEST_ROLES))
        self.assertTrue(all(entry["sha256"].startswith("new:") for entry in ember))

        # Every file is staged as <name>.part, one transfer each, in the generator's order, then
        # the manifest and the certificate; nothing lands under a real name by scp.
        self.assertEqual(self.uploaded(), [[path] for path in EMBER_UPLOAD_ORDER])
        self.assertEqual(self.staged(), [f"0x{SEED_A}/{path}.part" for path in EMBER_UPLOAD_ORDER])
        # Two ssh calls change the package: one before the transfers, which deletes no live
        # file, and the swap after them (the old certificate goes, the certificate lands last).
        self.assertEqual(self.changes(), self.ember_upload_commands(SEED_A))
        self.assertEqual(self.remote_part_files(), [])
        # The live package was read before the render and again before the upload.
        reads = [command for command in self.ssh_commands() if command.startswith("cat -- ")]
        self.assertEqual(len(reads), 4)
        self.assertEqual(self.call_log()[2][0], "generate")

    def test_a_failed_ember_transfer_leaves_the_live_package_as_it_was(self) -> None:
        package = self.remote_package(SEED_A)
        certificate = package / run.EMBER_CERTIFICATE
        certificate.write_text("from an earlier, interrupted upload", encoding="utf-8")
        leftover = package / "videos/web/ember.mp4.part"
        leftover.write_text("left by an earlier attempt", encoding="utf-8")
        live = self.remote_snapshot()
        os.environ["FAKE_SCP_FAIL_ON"] = "videos/hq"
        with self.assertLogs(run.log, level="ERROR"):
            outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)
        self.assertIs(outcome, run.Outcome.FAILED)
        # Five media had been staged when the sixth failed: none is under its real name, the
        # certificate and the manifest are untouched, and no staged file is left behind.
        self.assertEqual(len(self.calls("scp")), 5 + 2)
        leftover_path = str(leftover.relative_to(self.remote))
        self.assertEqual(
            self.remote_snapshot(), {p: c for p, c in live.items() if p != leftover_path}
        )
        self.assertEqual(self.remote_part_files(), [])
        prepare, discard = self.changes()
        self.assertEqual(prepare, self.ember_upload_commands(SEED_A)[0])
        self.assertEqual(discard, prepare.split(" && ")[0])
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([], [SEED_A]))

    def test_ember_backfill_with_a_different_orbit_uploads_nothing(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        os.environ[f"FAKE_GEN_MASSES_{SEED_A}"] = "[150.25, 200.5, 180.12500000000003]"
        with self.assertLogs(run.log, level="ERROR") as logs:
            outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)
        self.assertIs(outcome, run.Outcome.IDENTITY_MISMATCH)
        self.assertEqual(self.calls("scp"), [])
        self.assertEqual(self.remote_snapshot(), live)
        message = "\n".join(logs.output)
        self.assertIn(f"0x{SEED_A}", message)
        self.assertIn("DIFFERENT ORBIT OR VIEW", message)
        self.assertIn(f"IDENTITY MISMATCH  seed=0x{SEED_A}", message)
        # Only the differing field is named, with both values.
        self.assertIn(
            "(simulation.masses: live [150.25, 200.5, 180.125], "
            "regenerated [150.25, 200.5, 180.12500000000003])",
            message,
        )

    def test_ember_backfill_with_a_different_view_uploads_nothing(self) -> None:
        # The same orbit shown another way: the ember bodies would not follow the published art.
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        cases: tuple[tuple[str, str | None, str], ...] = (
            (
                "projection",
                '"phase_portrait"',
                'generation.projection: live "position", regenerated "phase_portrait"',
            ),
            (
                "scale",
                "1.1646093650188864",
                "generation.drift.scale: live 1.1646093650188862, regenerated 1.1646093650188864",
            ),
            # A regenerated file without the field cannot show that it matches.
            (
                "symmetry",
                None,
                'generation.symmetry: live "none", regenerated <missing>',
            ),
        )
        for field, value, difference in cases:
            with self.subTest(field=field, value=value):
                os.environ[f"FAKE_GEN_VIEW_{SEED_A}"] = view_members(**{field: value})
                with self.assertLogs(run.log, level="ERROR") as logs:
                    outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)
                self.assertIs(outcome, run.Outcome.IDENTITY_MISMATCH)
                message = "\n".join(logs.output)
                self.assertIn(f"DIFFERENT ORBIT OR VIEW than the live one ({difference})", message)
                self.assertIn(f"IDENTITY MISMATCH  seed=0x{SEED_A}", message)
        self.assertEqual(self.calls("scp"), [])
        self.assertEqual(self.remote_snapshot(), live)

    def test_replacing_an_edition_in_place_deletes_the_files_the_current_look_dropped(self) -> None:
        # An edition of an older look with one file more than the current one, and a manifest
        # whose other ember entries name files that are not this package's ember files.
        teaser = "videos/hq/ember_teaser.mp4"
        entries = [
            manifest_entry(teaser, "ember_teaser_hq", "live"),
            manifest_entry("images/source/master.png", "ember_decoy", "live"),
            manifest_entry(f"../0x{SEED_B}/images/source/ember.png", "ember_decoy", "live"),
        ]
        package = self.remote_v2_package(SEED_A, extra_entries=entries)
        (package / teaser).write_text("teaser live\n", encoding="utf-8")
        self.remote_v2_package(SEED_B)
        before_a, before_b = self.package_files(SEED_A), self.package_files(SEED_B)

        with self.assertLogs(run.log, level="INFO") as logs:
            outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)

        self.assertIs(outcome, run.Outcome.COMPLETE)
        text = "\n".join(logs.output)
        self.assertIn(
            f"has 1 files that the current edition does not, which are deleted with it: {teaser}",
            text,
        )
        self.assertIn("so those are left alone: 'images/source/master.png', ", text)
        # The teaser went with the old certificate, in the swap: nothing went before it.
        self.assertEqual(self.changes(), self.ember_upload_commands(SEED_A, retired=[teaser]))
        after = self.package_files(SEED_A)
        self.assertEqual(
            sorted(after), sorted([*run.CORE_PACKAGE_FILES, *SPECTRAL_FILES, *EMBER_V3_FILES])
        )
        # Nothing but the ember edition changed: not the main art the decoy named, not SEED_B.
        unchanged = set(before_a) - {*EMBER_V2_FILES, teaser, run.ASSET_MANIFEST}
        self.assertEqual({p: after[p] for p in unchanged}, {p: before_a[p] for p in unchanged})
        self.assertEqual(self.package_files(SEED_B), before_b)
        manifest = json.loads(after[run.ASSET_MANIFEST])
        ember = [entry for entry in manifest["assets"] if run.is_ember_asset(entry)]
        self.assertEqual([entry["role"] for entry in ember], list(EMBER_V3_ROLES))

    def test_backfill_whose_ember_edition_fails_uploads_nothing(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        for mode in ("core_only", "no_ember_files"):
            os.environ["FAKE_GEN_MODE"] = mode
            for backfill_mode in run.BackfillMode:
                with self.subTest(mode=mode, backfill_mode=backfill_mode):
                    with self.assertLogs(run.log, level="WARNING"):
                        outcome = self.process(SEED_A, backfill=backfill_mode)
                    self.assertIs(outcome, run.Outcome.EMBER_FAILED)
        self.assertEqual(self.calls("scp"), [])
        # Only the ember mode's check of the live package, before each render, reached the host.
        self.assertTrue(all(c.startswith("cat -- ") for c in self.ssh_commands()))
        self.assertEqual(len(self.ssh_commands()), 2 * 2)
        self.assertEqual(self.remote_snapshot(), live)

    def test_full_backfill_replaces_the_whole_package(self) -> None:
        self.remote_package(SEED_A)
        self.assertIs(self.process(SEED_A, backfill=run.BackfillMode.FULL), run.Outcome.COMPLETE)
        remote = self.remote_snapshot()
        self.assertEqual(
            remote[f"0x{SEED_A}/images/source/master.png"], "images/source/master.png new\n"
        )
        self.assertIn('"tag": "new"', remote[f"0x{SEED_A}/{run.NFT_TRAITS}"])
        self.assert_certificate_last()

    def test_an_interrupted_upload_leaves_the_package_incomplete(self) -> None:
        cases = (
            ("ember.json", "backfill"),
            ("metadata/assets.json", "urgent"),
            ("metadata/generation.json", "urgent"),
            ("spectral", "urgent"),
        )
        for fail_on, remaining in cases:
            with self.subTest(fail_on=fail_on):
                os.environ["FAKE_SCP_FAIL_ON"] = fail_on
                with self.assertLogs(run.log, level="ERROR"):
                    self.assertIs(self.process(SEED_B), run.Outcome.FAILED)
                remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
                assert remote is not None
                urgent, backfill = run.find_missing_seeds([SEED_B], remote)
                self.assertEqual(urgent if remaining == "urgent" else backfill, [SEED_B])
                self.assertFalse((run.LOCAL_OUTPUT_DIR / f"0x{SEED_B}").exists())
                # No metadata file is ever left truncated under its real name.
                for path, content in self.remote_snapshot().items():
                    if "/metadata/" in path and not path.endswith(run.PART_SUFFIX):
                        json.loads(content)
                shutil.rmtree(self.remote, ignore_errors=True)

    def test_an_interrupted_urgent_upload_stays_urgent_until_it_is_repaired(self) -> None:
        # A live package that lost its HQ video: urgent, although its metadata is all there.
        package = self.remote_package(SEED_A, ember=True)
        (package / "videos/hq/main.mp4").unlink()
        os.environ["FAKE_SCP_FAIL_ON"] = "videos/hq/main.mp4"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A), run.Outcome.FAILED)
        self.assertTrue((package / "videos/hq/main.mp4").is_file())  # truncated by the failure
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([SEED_A], []))

        del os.environ["FAKE_SCP_FAIL_ON"]
        self.assertIs(self.process(SEED_A), run.Outcome.COMPLETE)
        main_hq = (package / "videos/hq/main.mp4").read_text(encoding="utf-8")
        self.assertEqual(main_hq, "videos/hq/main.mp4 new\n")
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        self.assertEqual(remote, remote_listing(SEED_A, run.REQUIRED_PACKAGE_FILES))

    def test_an_interrupted_full_backfill_is_regenerated_in_full(self) -> None:
        self.remote_package(SEED_A)
        os.environ["FAKE_SCP_FAIL_ON"] = "videos/web/main.mp4"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A, backfill=run.BackfillMode.FULL), run.Outcome.FAILED)
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        # Urgent: whatever the backfill mode, the next run replaces the half-replaced media.
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([SEED_A], []))

    def test_an_interrupted_ember_manifest_upload_never_truncates_the_live_manifest(self) -> None:
        package = self.remote_package(SEED_A)
        live_manifest = (package / run.ASSET_MANIFEST).read_text(encoding="utf-8")
        live = self.remote_snapshot()
        os.environ["FAKE_SCP_FAIL_ON"] = run.ASSET_MANIFEST
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A, backfill=run.BackfillMode.EMBER), run.Outcome.FAILED)
        self.assertEqual((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"), live_manifest)
        # Every medium had been staged; none was swapped in, and the staged files are gone.
        self.assertEqual(self.remote_snapshot(), live)
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([], [SEED_A]))

        del os.environ["FAKE_SCP_FAIL_ON"]
        self.assertIs(self.process(SEED_A, backfill=run.BackfillMode.EMBER), run.Outcome.COMPLETE)
        self.assertEqual(self.remote_part_files(), [])
        merged = json.loads((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"))
        roles = [entry["role"] for entry in merged["assets"]]
        self.assertEqual(roles[-len(run.EMBER_MANIFEST_ROLES) :], list(run.EMBER_MANIFEST_ROLES))

    def test_an_unusable_live_package_is_neither_rendered_nor_touched(self) -> None:
        borda_less = json.dumps(
            {"simulation": {"masses": [1.5, 2.5, 3.5]}, "generation": {"selector": "borda"}}
        )
        # Every orbit field, but a view field is gone: the live view cannot be known.
        drift_scale_less = traits_text(SEED_A, "live", view=view_members(scale=None))
        cases = (
            (run.ASSET_MANIFEST, CORRUPT_JSON, "assets.json is not valid JSON"),
            (run.ASSET_MANIFEST, '{"schema_version": 2}', "has no assets list"),
            (run.NFT_TRAITS, CORRUPT_JSON, "nft_traits.json is not valid JSON"),
            (run.NFT_TRAITS, borda_less, "lacks generation.borda.selected_index"),
            (run.NFT_TRAITS, drift_scale_less, "nft_traits.json lacks generation.drift.scale."),
        )
        for path, content, problem in cases:
            with self.subTest(path=path, content=content):
                shutil.rmtree(self.remote, ignore_errors=True)
                self.remote_package(SEED_A)
                (self.remote / f"0x{SEED_A}" / path).write_text(content, encoding="utf-8")
                live = self.remote_snapshot()
                with self.assertLogs(run.log, level="ERROR") as logs:
                    outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)
                self.assertIs(outcome, run.Outcome.FAILED)
                self.assertIn(problem, "\n".join(logs.output))
                self.assertIn("repaired on the asset host", "\n".join(logs.output))
                self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.calls("scp"), [])

    def test_an_unreadable_live_package_is_not_rendered(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        os.environ["FAKE_SSH_FAIL_ON"] = "cat "
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A, backfill=run.BackfillMode.EMBER), run.Outcome.FAILED)
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.calls("scp"), [])
        self.assertEqual(self.remote_snapshot(), live)

    def test_a_live_package_that_breaks_during_the_render_is_not_merged(self) -> None:
        package = self.remote_package(SEED_A)
        os.environ["FAKE_GEN_CORRUPT"] = str(package / run.ASSET_MANIFEST)
        with self.assertLogs(run.log, level="ERROR") as logs:
            outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)
        self.assertIs(outcome, run.Outcome.FAILED)
        self.assertIn("cannot be used", "\n".join(logs.output))
        self.assertEqual(self.generated(), [SEED_A])
        self.assertEqual(self.calls("scp"), [])
        self.assertEqual((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"), CORRUPT_JSON)

    def test_regenerated_metadata_that_cannot_show_its_orbit_is_an_ember_failure(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        os.environ[f"FAKE_GEN_MASSES_{SEED_A}"] = "[1.0, 2.0"  # breaks the local nft_traits.json
        with self.assertLogs(run.log, level="ERROR") as logs:
            outcome = self.process(SEED_A, backfill=run.BackfillMode.EMBER)
        self.assertIs(outcome, run.Outcome.EMBER_FAILED)
        self.assertIn(
            "regenerated metadata/nft_traits.json is not valid JSON", "\n".join(logs.output)
        )
        self.assertEqual(self.calls("scp"), [])
        self.assertEqual(self.remote_snapshot(), live)

    def test_no_upload_retry_starts_after_a_shutdown_request(self) -> None:
        source = self.root / "file.bin"
        source.write_bytes(b"payload")
        self.remote.mkdir(parents=True)
        os.environ["FAKE_SCP_FAIL_ON"] = "file.bin"
        target = f"fakeuser@fakehost:{self.remote}/"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertFalse(run.scp_transfer([source], target, "test", retries=3))
        self.assertEqual(len(self.calls("scp")), 3)
        with mock.patch.object(run, "shutdown_requested", True), self.assertLogs(run.log, "ERROR"):
            self.assertFalse(run.scp_transfer([source], target, "test", retries=3))
        self.assertEqual(len(self.calls("scp")), 4)

    def test_a_generator_that_predates_the_ember_edition_uploads_the_core_package(self) -> None:
        os.environ["FAKE_GEN_MODE"] = "stale"
        with self.assertLogs(run.log, level="WARNING"):
            outcome = self.process(SEED_A, ember_capable=False)
        self.assertIs(outcome, run.Outcome.CORE_ONLY)
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        self.assertEqual(remote, remote_listing(SEED_A, run.CORE_PACKAGE_FILES))

    def test_the_same_package_fails_validation_with_an_ember_capable_generator(self) -> None:
        os.environ["FAKE_GEN_MODE"] = "stale"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A), run.Outcome.EMBER_FAILED)
        self.assertEqual(self.calls("scp"), [])

    def test_a_failed_generator_uploads_nothing_and_leaves_no_local_files(self) -> None:
        os.environ["FAKE_GEN_MODE"] = "fail"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIs(self.process(SEED_A), run.Outcome.FAILED)
        self.assertEqual(self.calls("ssh"), [])
        self.assertFalse((run.LOCAL_OUTPUT_DIR / f"0x{SEED_A}").exists())


# ---------------------------------------------------------------------------
# Stale ember editions
# ---------------------------------------------------------------------------

SEED_E = "e" * 64
SEED_F = "f" * 64


class StaleEmberTests(SyncTestCase):
    """Reading the live certificates, choosing the stale editions, and retiring them: withdrawn,
    or kept online for the backfill to replace."""

    def ember_paths(self, *seeds: str) -> set[str]:
        """The remote listing's paths of every ember file of `seeds`."""
        return {f"0x{seed}/{path}" for seed in seeds for path in run.EMBER_PACKAGE_FILES}

    def retire(
        self, seeds: list[str], listing: set[str], *, keep: bool = False, dry_run: bool = False
    ) -> tuple[set[str], bool]:
        """run.retire_stale_ember_editions() against the fake remote, planned from `listing`."""
        return run.retire_stale_ember_editions(
            seeds,
            listing,
            CURRENT_ALGORITHM,
            "fakehost",
            "fakeuser",
            str(self.remote),
            keep=keep,
            dry_run=dry_run,
        )

    def retire_all(
        self, seeds: list[str], *, keep: bool = False, dry_run: bool = False
    ) -> tuple[set[str], bool, set[str]]:
        """retire() from the fake remote's own listing: its listing, its status, and the listing
        it started from. The call log starts after that first listing."""
        listing = self.listing()
        self.forget_calls()
        remaining, ok = self.retire(seeds, listing, keep=keep, dry_run=dry_run)
        return remaining, ok, listing

    def withdraw(self, seed: str) -> bool:
        """run.withdraw_ember_edition() of `seed` against the fake remote."""
        return run.withdraw_ember_edition(seed, "fakehost", "fakeuser", str(self.remote))

    def test_every_certificate_is_read_in_one_ssh_call(self) -> None:
        self.remote_package(SEED_A, ember=True)
        self.remote_package(SEED_B, ember=True, algorithm=STALE_ALGORITHM)
        self.remote_package(SEED_C, ember=True)
        self.remote_package(SEED_D)  # no ember edition, so no certificate
        self.remote_package(SEED_E, ember=True, algorithm="ember-2")
        self.remote_package(SEED_F, ember=True)
        certificate = self.remote / f"0x{SEED_C}" / run.EMBER_CERTIFICATE
        certificate.write_text('{"schema_version": 2, "algorithm": "ember-v1"}', encoding="utf-8")
        duplicated = self.remote / f"0x{SEED_F}" / run.EMBER_CERTIFICATE
        duplicated.write_text(
            '{\n  "algorithm": "ember-v1",\n  "algorithm": "ember-v2"\n}\n', encoding="utf-8"
        )
        algorithms = run.list_remote_ember_algorithms("fakehost", "fakeuser", str(self.remote))
        self.assertEqual(
            algorithms,
            {
                SEED_A: CURRENT_ALGORITHM,
                SEED_B: STALE_ALGORITHM,
                SEED_C: None,  # another layout (compact JSON): unreadable
                SEED_E: None,  # not an ember algorithm id
                SEED_F: None,  # two top-level ids
            },
        )
        self.assertEqual(len(self.ssh_commands()), 1)

    def test_a_failed_certificate_listing_is_none(self) -> None:
        with self.assertLogs(run.log, level="ERROR"):  # the directory is missing
            self.assertIsNone(run.list_remote_ember_algorithms("fakehost", "fakeuser", "/missing"))
        os.environ["FAKE_SSH_FAIL"] = "1"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertIsNone(
                run.list_remote_ember_algorithms("fakehost", "fakeuser", str(self.remote))
            )

    def test_only_older_readable_editions_of_listed_complete_packages_are_stale(self) -> None:
        unlisted, urgent, backfill = "1" * 64, "2" * 64, "3" * 64
        live: dict[str, str | None] = {
            SEED_A: CURRENT_ALGORITHM,
            SEED_B: STALE_ALGORITHM,
            SEED_C: None,
            SEED_D: NEWER_ALGORITHM,
            unlisted: STALE_ALGORITHM,
            urgent: STALE_ALGORITHM,
            backfill: "ember-v0",
        }
        remote = remote_listing(urgent, run.REQUIRED_PACKAGE_FILES[1:])  # lacks master.png
        remote |= remote_listing(backfill, [*run.CORE_PACKAGE_FILES, run.EMBER_CERTIFICATE])
        for seed in (SEED_A, SEED_B, SEED_C, SEED_D, unlisted):
            remote |= remote_listing(seed, run.REQUIRED_PACKAGE_FILES)
        seeds = [backfill, SEED_A, SEED_B, SEED_C, SEED_D, urgent]
        with self.assertLogs(run.log, level="WARNING") as logs:
            stale = run.stale_ember_editions(seeds, live, CURRENT_ALGORITHM, remote)
        self.assertEqual(list(stale.items()), [(backfill, "ember-v0"), (SEED_B, STALE_ALGORITHM)])
        warnings = "\n".join(logs.output)
        self.assertIn(
            f"0x{SEED_C}: the algorithm of its live metadata/ember.json cannot be read", warnings
        )
        self.assertIn(f"newer than this generator's {CURRENT_ALGORITHM}", warnings)
        self.assertIn(f"rolled back?): 0x{SEED_D}", warnings)
        self.assertIn("not in the seed list, so nothing would render them again", warnings)
        self.assertIn(f"obsolete): 0x{unlisted}", warnings)
        self.assertNotIn(urgent, warnings)

    def test_algorithms_compare_by_number(self) -> None:
        remote = remote_listing(SEED_A, run.REQUIRED_PACKAGE_FILES)
        stale = run.stale_ember_editions([SEED_A], {SEED_A: "ember-v9"}, "ember-v10", remote)
        self.assertEqual(stale, {SEED_A: "ember-v9"})
        with self.assertLogs(run.log, level="WARNING"):
            self.assertEqual(
                run.stale_ember_editions([SEED_A], {SEED_A: "ember-v10"}, "ember-v9", remote), {}
            )

    def test_the_certificate_goes_first_then_the_manifest_entries_then_the_media(self) -> None:
        poster: dict[str, object] = {
            "role": "poster",
            "title": "Kōzo, été",
            "duration_seconds": 30.033333333333335,
            "bytes": 12345678901234567890123,
        }
        package = self.remote_package(
            SEED_B, ember=True, algorithm=STALE_ALGORITHM, extra_entries=[poster]
        )
        live = self.remote_snapshot()
        live_manifest = json.loads((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"))

        self.assertTrue(self.withdraw(SEED_B))

        remote = self.remote_snapshot()
        manifest_path = f"0x{SEED_B}/{run.ASSET_MANIFEST}"
        kept = {
            path: content
            for path, content in live.items()
            if path not in self.ember_paths(SEED_B) and path != manifest_path
        }
        self.assertEqual({p: c for p, c in remote.items() if p != manifest_path}, kept)
        manifest = json.loads(remote[manifest_path])
        # The same fields in the same order, each unchanged; only the ember entries are gone.
        self.assertEqual(list(manifest), list(live_manifest))
        for field, value in live_manifest.items():
            if field != "assets":
                self.assertEqual(manifest[field], value, field)
        self.assertEqual(
            manifest["assets"], [e for e in live_manifest["assets"] if not run.is_ember_asset(e)]
        )
        self.assertEqual(manifest["assets"][-1], poster)
        self.assertEqual(len(manifest["assets"]), len(CORE_MEDIA_ROLES) + 2)

        remote_package = f"{self.remote}/0x{SEED_B}"
        manifest_target = f"{remote_package}/{run.ASSET_MANIFEST}"
        self.assertEqual(
            self.call_log(),
            [
                ["ssh", f"cat -- {manifest_target}"],
                [
                    "ssh",
                    f"rm -f -- {remote_package}/{run.EMBER_CERTIFICATE} && "
                    f"mkdir -p -- {remote_package}/metadata",
                ],
                ["scp", [f"output/0x{SEED_B}/{run.ASSET_MANIFEST}"], f"{manifest_target}.part"],
                ["ssh", f"mv -f -- {manifest_target}.part {manifest_target}"],
                [
                    "ssh",
                    "rm -f -- "
                    + " ".join(f"{remote_package}/{path}" for path in run.EMBER_PACKAGE_FILES),
                ],
            ],
        )
        self.assertEqual(self.remote_part_files(), [])
        self.assertFalse((run.LOCAL_OUTPUT_DIR / f"0x{SEED_B}").exists())
        self.assertEqual(run.find_missing_seeds([SEED_B], self.listing()), ([], [SEED_B]))

    def test_a_withdrawal_takes_the_files_the_current_look_dropped_too(self) -> None:
        teaser = "videos/hq/ember_teaser.mp4"
        entries = [
            manifest_entry(teaser, "ember_teaser_hq", "live"),
            manifest_entry("images/source/master.png", "ember_decoy", "live"),
        ]
        package = self.remote_v2_package(SEED_A, extra_entries=entries)
        (package / teaser).write_text("teaser live\n", encoding="utf-8")
        before = self.package_files(SEED_A)

        with self.assertLogs(run.log, level="WARNING") as logs:
            self.assertTrue(self.withdraw(SEED_A))

        self.assertIn("left alone: 'images/source/master.png'", "\n".join(logs.output))
        after = self.package_files(SEED_A)
        self.assertEqual(sorted(after), sorted([*run.CORE_PACKAGE_FILES, *SPECTRAL_FILES]))
        kept = set(after) - {run.ASSET_MANIFEST}
        self.assertEqual({p: after[p] for p in kept}, {p: before[p] for p in kept})
        manifest = json.loads(after[run.ASSET_MANIFEST])
        self.assertFalse(any(run.is_ember_asset(entry) for entry in manifest["assets"]))
        # The media, the teaser included, go last: after the manifest that no longer lists them.
        commands = self.ssh_commands()
        self.assertTrue(commands[-2].startswith("mv -f -- "), commands[-2])
        self.assertTrue(commands[-1].startswith("rm -f -- "), commands[-1])
        self.assertTrue(commands[-1].endswith(f" {self.remote}/0x{SEED_A}/{teaser}"), commands[-1])
        self.assertNotIn("master.png", commands[-1])

    def test_a_withdrawal_is_idempotent(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        self.assertTrue(self.withdraw(SEED_A))
        withdrawn = self.remote_snapshot()
        self.assertTrue(self.withdraw(SEED_A))
        self.assertEqual(self.remote_snapshot(), withdrawn)
        # A package without a certificate is never considered again: not even ssh is needed.
        remaining, ok, listing = self.retire_all([SEED_A])
        self.assertTrue(ok)
        self.assertEqual(remaining, listing)
        self.assertEqual(self.call_log(), [])
        self.assertEqual(self.remote_snapshot(), withdrawn)

    def test_stale_editions_are_withdrawn_and_planned_as_backfill_seeds(self) -> None:
        self.remote_package(SEED_A, ember=True)
        self.remote_package(SEED_B, ember=True, algorithm=STALE_ALGORITHM)
        self.remote_package(SEED_C, ember=True, algorithm="ember-v0")
        a_before = {p: c for p, c in self.remote_snapshot().items() if SEED_A in p}
        with self.assertLogs(run.log, level="INFO") as logs:
            remaining, ok, _listing = self.retire_all([SEED_A, SEED_B, SEED_C])
        self.assertTrue(ok)
        self.assertEqual(remaining, self.listing())  # the listing a new run would see
        self.assertEqual(
            run.find_missing_seeds([SEED_A, SEED_B, SEED_C], remaining), ([], [SEED_B, SEED_C])
        )
        self.assertEqual({p: c for p, c in self.remote_snapshot().items() if SEED_A in p}, a_before)
        text = "\n".join(logs.output)
        self.assertIn(f"WITHDRAWN  seed=0x{SEED_B}  its {STALE_ALGORITHM} ember edition", text)
        self.assertIn(f"WITHDRAWN  seed=0x{SEED_C}  its ember-v0 ember edition", text)
        self.assertIn(
            f"Withdrew 2 stale ember editions (ember-v0, {STALE_ALGORITHM} -> {CURRENT_ALGORITHM})",
            text,
        )

    def test_kept_editions_stay_online_and_are_planned_as_backfill_seeds(self) -> None:
        self.remote_package(SEED_A, ember=True)
        self.remote_package(SEED_B, ember=True, algorithm=STALE_ALGORITHM)  # every current file
        self.remote_v2_package(SEED_C)  # lacks the slow film as well
        live = self.remote_snapshot()
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run), self.assertLogs(run.log, level="INFO") as logs:
                remaining, ok, listing = self.retire_all(
                    [SEED_A, SEED_B, SEED_C], keep=True, dry_run=dry_run
                )
            self.assertTrue(ok)
            # Nothing changed on the asset host: only the certificates were read.
            self.assertEqual(self.remote_snapshot(), live)
            self.assertEqual(len(self.ssh_commands()), 1)
            self.assertEqual(self.calls("scp"), [])
            # The plan counts both older editions as backfill seeds, whatever files they hold.
            self.assertEqual(remaining, listing - self.ember_paths(SEED_B, SEED_C))
            self.assertEqual(
                run.find_missing_seeds([SEED_A, SEED_B, SEED_C], remaining),
                ([], [SEED_B, SEED_C]),
            )
            text = "\n".join(logs.output)
            self.assertIn(
                f"Kept 2 stale ember editions online ({STALE_ALGORITHM} -> {CURRENT_ALGORITHM}): "
                "the ember backfill replaces each in place",
                text,
            )
            self.assertNotIn("WITHDRAWN", text)
            self.assertNotIn("Withdrew", text)

    def test_a_failed_withdrawal_does_not_stop_the_others(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        self.remote_package(SEED_B, ember=True, algorithm=STALE_ALGORITHM)
        os.environ["FAKE_SSH_FAIL_ON"] = f"cat -- {self.remote}/0x{SEED_A}/"
        with self.assertLogs(run.log, level="INFO") as logs:
            remaining, ok, _listing = self.retire_all([SEED_A, SEED_B])
        self.assertFalse(ok)
        self.assertEqual(remaining, self.listing())
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B], remaining), ([], [SEED_B]))
        self.assertIn(
            f"1 stale ember editions could not be withdrawn (see above; a later run retries each, "
            f"or renders it again if its certificate is already gone): 0x{SEED_A}",
            "\n".join(logs.output),
        )

    def test_an_interrupted_withdrawal_already_reads_as_a_backfill_seed(self) -> None:
        package = self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        live_manifest = (package / run.ASSET_MANIFEST).read_text(encoding="utf-8")
        os.environ["FAKE_SCP_FAIL_ON"] = run.ASSET_MANIFEST
        with self.assertLogs(run.log, level="ERROR"):
            self.assertFalse(self.withdraw(SEED_A))
        self.assertFalse((package / run.EMBER_CERTIFICATE).exists())
        self.assertEqual((package / run.ASSET_MANIFEST).read_text(encoding="utf-8"), live_manifest)
        self.assertEqual(run.find_missing_seeds([SEED_A], self.listing()), ([], [SEED_A]))

        # The backfill replaces whatever the withdrawal left: media, manifest entries, certificate.
        del os.environ["FAKE_SCP_FAIL_ON"]
        self.assertIs(self.process(SEED_A, backfill=run.BackfillMode.EMBER), run.Outcome.COMPLETE)
        remote = self.remote_snapshot()
        for path in run.EMBER_MEDIA_FILES:
            self.assertEqual(remote[f"0x{SEED_A}/{path}"], f"{path} new\n")
        certificate = json.loads(remote[f"0x{SEED_A}/{run.EMBER_CERTIFICATE}"])
        self.assertEqual(certificate["algorithm"], CURRENT_ALGORITHM)
        manifest = json.loads(remote[f"0x{SEED_A}/{run.ASSET_MANIFEST}"])
        ember = [entry for entry in manifest["assets"] if run.is_ember_asset(entry)]
        self.assertEqual([entry["role"] for entry in ember], list(run.EMBER_MANIFEST_ROLES))
        self.assertTrue(all(str(entry["sha256"]).startswith("new:") for entry in ember))

    def test_an_unusable_live_manifest_leaves_the_package_untouched(self) -> None:
        package = self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        (package / run.ASSET_MANIFEST).write_text(CORRUPT_JSON, encoding="utf-8")
        live = self.remote_snapshot()
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertFalse(self.withdraw(SEED_A))
        self.assertIn("repaired on the asset host", "\n".join(logs.output))
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.calls("scp"), [])

    def test_a_dry_run_changes_nothing(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        live = self.remote_snapshot()
        with self.assertLogs(run.log, level="INFO") as logs:
            remaining, ok, listing = self.retire_all([SEED_A], dry_run=True)
        self.assertTrue(ok)
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(len(self.ssh_commands()), 1)  # the certificates were read, nothing else
        self.assertEqual(self.calls("scp"), [])
        # The plan a real run would make: the package waits for the backfill.
        self.assertEqual(remaining, listing - self.ember_paths(SEED_A))
        text = "\n".join(logs.output)
        self.assertIn(
            f"DRY-RUN  would withdraw the {STALE_ALGORITHM} ember edition of 0x{SEED_A}", text
        )
        self.assertIn("DRY-RUN  would withdraw 1 stale ember editions", text)

    def test_nothing_is_retired_when_the_certificates_cannot_be_read(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        live = self.remote_snapshot()
        listing = self.listing()
        os.environ["FAKE_SSH_FAIL"] = "1"
        for keep, consequence in ((False, "withdrawn"), (True, "replaced")):
            with self.subTest(keep=keep), self.assertLogs(run.log, level="ERROR") as logs:
                remaining, ok = self.retire([SEED_A], listing, keep=keep)
            self.assertFalse(ok)
            self.assertEqual(remaining, listing)
            self.assertEqual(self.remote_snapshot(), live)
            self.assertIn(f"none is {consequence} this run", "\n".join(logs.output))


# ---------------------------------------------------------------------------
# Manifest merge, orbit check, timeouts
# ---------------------------------------------------------------------------


class EmberBackfillHelperTests(unittest.TestCase):
    """The ember file and role lists, merge_ember_manifest, retired_ember_files,
    identity_differences and scp_timeout."""

    def test_the_ember_files_and_roles_of_the_current_look(self) -> None:
        # The generator lists its ember outputs in exactly this order (app::EMBER_OUTPUT_PATHS; a
        # Rust unit test compares that with run.py's tuple), and writes these manifest roles.
        self.assertEqual(run.EMBER_PACKAGE_FILES, EMBER_V3_FILES)
        self.assertEqual(run.EMBER_MANIFEST_ROLES, EMBER_V3_ROLES)
        self.assertEqual(run.EMBER_MEDIA_FILES, EMBER_V3_FILES[:-1])
        self.assertEqual(run.EMBER_CERTIFICATE, EMBER_V3_FILES[-1])
        self.assertEqual(run.REQUIRED_PACKAGE_FILES[-len(EMBER_V3_FILES) :], EMBER_V3_FILES)
        # The look before it had every file but the slow film, and every role but its role.
        self.assertEqual(set(EMBER_V3_FILES) - set(EMBER_V2_FILES), {"videos/web/ember_slow.mp4"})
        self.assertEqual(
            set(EMBER_V3_ROLES) - set(EMBER_V2_MEDIA_ROLES.values()), {"ember_slow_web"}
        )
        self.assertLessEqual(set(EMBER_V2_FILES), set(EMBER_V3_FILES))

    def test_only_ember_files_are_named_ember(self) -> None:
        # retired_ember_files() relies on it: a file it may delete is never a core file.
        def named_ember(path: str) -> bool:
            return posixpath.basename(path).startswith(run.EMBER_FILE_PREFIX)

        self.assertTrue(all(named_ember(path) for path in run.EMBER_PACKAGE_FILES))
        self.assertFalse(any(named_ember(path) for path in run.CORE_PACKAGE_FILES))
        self.assertFalse(any(named_ember(path) for path in SPECTRAL_FILES))

    def test_retired_ember_files_are_ember_files_of_the_package_only(self) -> None:
        def entry(path: object, role: str = "ember_retired") -> dict[str, object]:
            return {"role": role, "path": path}

        current = [entry(path, role) for path, role in EMBER_MEDIA_ROLES.items()]
        with self.assertNoLogs(run.log, level="INFO"):
            self.assertEqual(run.retired_ember_files({"assets": current}, SEED_A), [])
            v2 = [entry(path, role) for path, role in EMBER_V2_MEDIA_ROLES.items()]
            self.assertEqual(run.retired_ember_files({"assets": v2}, SEED_A), [])

        refused: list[object] = [
            "images/source/master.png",  # a core file under an ember role
            "videos/web/main.mp4",
            "spectral/00_380nm.png",
            "../ember.png",  # outside the package
            f"../0x{SEED_B}/videos/hq/ember.mp4",
            "videos/../../ember.png",
            "/tmp/ember.png",
            "videos//ember_old.mp4",  # not normalised
            "./ember_old.png",
            "videos/web/ember_frames/",
            "",
            7,
            None,
        ]
        live = {
            "assets": [
                *current,
                entry("videos/hq/ember_slow.mp4"),
                entry("ember_notes.txt"),
                entry("videos/hq/ember_slow.mp4", "ember_slow_hq"),  # listed twice
                *(entry(path) for path in refused),
                {"role": "ember_without_a_path"},
                entry("videos/hq/ember_other.mp4", "main_hq"),  # not an ember entry
            ]
        }
        with self.assertLogs(run.log, level="INFO") as logs:
            retired = run.retired_ember_files(live, SEED_A)
        self.assertEqual(retired, ["ember_notes.txt", "videos/hq/ember_slow.mp4"])
        self.assertEqual([record.levelname for record in logs.records], ["WARNING", "INFO"])
        warning, info = (record.getMessage() for record in logs.records)
        self.assertTrue(warning.startswith(f"0x{SEED_A}: the live metadata/assets.json lists"))
        self.assertTrue(
            warning.endswith("left alone: " + ", ".join(repr(path) for path in refused)), warning
        )
        self.assertTrue(info.endswith(": ember_notes.txt, videos/hq/ember_slow.mp4"), info)

        malformed: list[object] = [[], {"assets": {}}, {"assets": [{"path": "no role"}]}]
        for manifest in malformed:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                run.retired_ember_files(manifest, SEED_A)

    def test_merge_keeps_live_entries_and_replaces_ember_ones(self) -> None:
        source: dict[str, object] = {"role": "source_master", "sha256": "live"}
        main_web: dict[str, object] = {"role": "main_web", "duration_seconds": 30.033333333333335}
        stale_ember: dict[str, object] = {"role": "ember_web", "sha256": "stale"}
        new_ember = [{"role": role, "sha256": "new"} for role in run.EMBER_MANIFEST_ROLES]
        live = {
            "schema_version": 2,
            "generated_at": "then",
            "extra": {"kept": True},
            "assets": [source, stale_ember, main_web],
        }
        local = {
            "schema_version": 2,
            "generated_at": "now",
            "assets": [{"role": "source_master", "sha256": "regenerated"}, *new_ember],
        }
        merged = run.merge_ember_manifest(live, local)
        self.assertEqual(merged["schema_version"], 2)
        self.assertEqual(merged["generated_at"], "now")
        self.assertEqual(merged["extra"], {"kept": True})
        self.assertEqual(merged["assets"], [source, main_web, *new_ember])

    def test_merge_rejects_malformed_manifests(self) -> None:
        ember = [{"role": role} for role in run.EMBER_MANIFEST_ROLES]
        full = {"generated_at": "now", "assets": ember}
        cases: list[tuple[object, object]] = [
            ([], full),
            ({"assets": {}}, full),
            ({"assets": [{"path": "no role"}]}, full),
            ({"assets": []}, {"assets": ember}),
            ({"assets": []}, {"generated_at": "now", "assets": ember[:-1]}),
        ]
        for live, local in cases:
            with self.subTest(live=live, local=local), self.assertRaises(ValueError):
                run.merge_ember_manifest(live, local)

    def test_without_ember_entries_keeps_everything_else_verbatim(self) -> None:
        source: dict[str, object] = {"role": "source_master", "sha256": "live"}
        ember: dict[str, object] = {"role": "ember_web", "sha256": "stale"}
        main_web: dict[str, object] = {"role": "main_web", "duration_seconds": 30.033333333333335}
        live = {
            "schema_version": 2,
            "generated_at": "then",
            "assets": [source, ember, main_web],
            "extra": {"kept": True},
        }
        stripped = run.without_ember_entries(live)
        self.assertEqual(list(stripped), list(live))
        self.assertEqual(stripped, {**live, "assets": [source, main_web]})
        self.assertEqual(live["assets"], [source, ember, main_web])  # the input is unchanged
        malformed: list[object] = [[], {"assets": {}}, {"assets": [{"path": "no role"}]}]
        for manifest in malformed:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                run.without_ember_entries(manifest)

    def test_ember_algorithm_numbers(self) -> None:
        self.assertEqual(run.ember_algorithm_number("ember-v2"), 2)
        self.assertEqual(run.ember_algorithm_number("ember-v10"), 10)
        # "٣" is an Arabic-Indic three: a digit to `\d`, but not an ember algorithm number.
        for invalid in ("ember-v", "ember-2", "ember-v2 ", "Ember-v2", "ember-v٣"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run.ember_algorithm_number(invalid)

    def test_the_identity_fields_are_the_orbit_and_the_view(self) -> None:
        # The real paths of the generator's nft_traits.json (src/nft_traits.rs, and
        # docs/nft_traits.schema.json, which requires every one of them).
        self.assertEqual(
            [".".join(field) for field in run.IDENTITY_FIELDS],
            [
                "simulation.masses",
                "generation.borda.selected_index",
                "generation.borda.retry_count",
                "generation.structure.stack_label",
                "generation.projection",
                "generation.symmetry",
                "generation.drift.mode",
                "generation.drift.scale",
                "generation.drift.arc_fraction",
                "generation.drift.orbit_eccentricity",
            ],
        )
        self.assertEqual(run.IDENTITY_FIELDS, run.ORBIT_IDENTITY_FIELDS + run.VIEW_IDENTITY_FIELDS)
        example = REPO_ROOT / "docs" / "fixtures" / "nft_traits.example.json"
        published = run.parse_json_exact(example.read_text(encoding="utf-8"))
        self.assertEqual(run.identity_differences(published, published), [])
        run.parse_live_package(example.read_text(encoding="utf-8"), '{"assets": []}')

    def test_identity_differences_compare_exact_values(self) -> None:
        def traits(
            masses: str = "[1.1, 2.5, 3.0]", index: int = 7, retries: int = 1, view: str = ""
        ) -> object:
            return run.parse_json_exact(
                f'{{"simulation": {{"masses": {masses}}}, '
                f'"generation": {{"borda": {{"selected_index": {index}, '
                f'"retry_count": {retries}}}, {view or view_members()}}}}}'
            )

        base = traits()
        self.assertEqual(run.identity_differences(base, traits()), [])
        self.assertEqual(
            run.identity_differences(base, traits("[1.1000000000000001, 2.5, 3.0]")),
            ["simulation.masses: live [1.1, 2.5, 3.0], regenerated [1.1000000000000001, 2.5, 3.0]"],
        )
        differences = run.identity_differences(base, traits(index=8, retries=2))
        self.assertEqual(
            [difference.split(":")[0] for difference in differences],
            ["generation.borda.selected_index", "generation.borda.retry_count"],
        )

        # Each view field alone is a difference, named with both values.
        changes = {
            "stack_label": (
                '"triangle_web"',
                'generation.structure.stack_label: live "orbit_ribbons+harmonic_weave@0.29"',
            ),
            "projection": ('"hodograph"', 'generation.projection: live "position"'),
            "symmetry": ('"rot4"', 'generation.symmetry: live "none", regenerated "rot4"'),
            "mode": ('"linear"', 'generation.drift.mode: live "elliptical", regenerated "linear"'),
            "scale": ("1.1646093650188864", "generation.drift.scale: live 1.1646093650188862"),
            "arc_fraction": ("0.82497685045158", "generation.drift.arc_fraction: live 0.8249"),
            "orbit_eccentricity": ("0.0", "generation.drift.orbit_eccentricity: live 0.4045"),
        }
        self.assertEqual(len(changes), len(run.VIEW_IDENTITY_FIELDS))
        for field, (value, difference) in changes.items():
            with self.subTest(field=field):
                changed = traits(view=view_members(**{field: value}))
                (found,) = run.identity_differences(base, changed)
                self.assertTrue(found.startswith(difference), found)
                self.assertTrue(found.endswith(f"regenerated {value}"), found)
                # A field the regenerated file lacks is a difference too.
                without = traits(view=view_members(**{field: None}))
                (lacking,) = run.identity_differences(base, without)
                self.assertTrue(lacking.endswith("regenerated <missing>"), lacking)
        # The drift fields that do not shape the view are not compared.
        text = traits_text(SEED_A, "live")
        other = text.replace('"enabled": true', '"enabled": false')
        other = other.replace('"randomized": true', '"randomized": false')
        self.assertEqual(other.count("false"), 2)
        self.assertEqual(
            run.identity_differences(run.parse_json_exact(text), run.parse_json_exact(other)), []
        )
        self.assertEqual(len(run.identity_differences({}, {})), len(run.IDENTITY_FIELDS))

    def test_a_live_package_must_show_its_orbit_and_view(self) -> None:
        manifest = '{"assets": []}'
        run.parse_live_package(traits_text(SEED_A, "live"), manifest)
        self.assertEqual(
            list(VIEW_FIELD_PATHS.values()),
            [".".join(field) for field in run.VIEW_IDENTITY_FIELDS],
        )
        for field, path in VIEW_FIELD_PATHS.items():
            with self.subTest(field=field), self.assertRaises(ValueError) as raised:
                run.parse_live_package(
                    traits_text(SEED_A, "live", view=view_members(**{field: None})), manifest
                )
            self.assertEqual(str(raised.exception), f"metadata/nft_traits.json lacks {path}")

    def test_scp_timeout_scales_with_size(self) -> None:
        floor_bytes = run.SCP_MIN_TIMEOUT * run.SCP_MIN_BYTES_PER_SECOND
        self.assertEqual(run.scp_timeout(0), run.SCP_MIN_TIMEOUT)
        self.assertEqual(run.scp_timeout(floor_bytes), run.SCP_MIN_TIMEOUT)
        self.assertEqual(run.scp_timeout(floor_bytes + 1), run.SCP_MIN_TIMEOUT + 1)
        self.assertEqual(run.scp_timeout(2_000_000_001), 2001)


# ---------------------------------------------------------------------------
# main() and --preflight
# ---------------------------------------------------------------------------


class MainTests(SyncTestCase):
    """main() end to end, with the seed list patched."""

    def setUp(self) -> None:
        super().setUp()
        # A deployment creates the remote asset directory before the first run (README setup);
        # a missing one fails the listing (ListRemoteFilesTests).
        self.remote.mkdir(parents=True)

    def main(self, seeds: list[str], *extra: str) -> int:
        argv = [
            "--ssh-host",
            "fakehost",
            "--ssh-user",
            "fakeuser",
            "--remote-dir",
            str(self.remote),
            "--generator",
            self.generator,
            *extra,
        ]
        with (
            mock.patch.object(run, "resolve_token_seeds", return_value=(seeds, "test")),
            mock.patch.object(run, "setup_logging"),
            mock.patch.object(run, "install_signal_handlers"),
        ):
            return run.main(argv)

    def identity(self) -> run.GeneratorIdentity:
        identity = run.GeneratorIdentity.of(self.generator)
        assert identity is not None
        return identity

    def ledger(self) -> run.BackfillLedger:
        """The ledger main() saved, for the fake generator."""
        return run.load_backfill_ledger(self.identity())

    def test_a_failed_remote_listing_stops_the_run(self) -> None:
        os.environ["FAKE_SSH_FAIL"] = "1"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertEqual(self.main([SEED_A, SEED_B]), 1)
        self.assertEqual(self.generated(), [])

    def test_new_mints_first_then_one_backfill_seed(self) -> None:
        self.remote_package(SEED_A)
        self.remote_package(SEED_B)
        self.assertEqual(self.main([SEED_A, SEED_B, SEED_C]), 0)
        self.assertEqual(self.generated(), [SEED_C, SEED_A])
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B, SEED_C], remote), ([], [SEED_B]))

    def test_a_different_orbit_gives_the_seed_up_at_once(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        os.environ[f"FAKE_GEN_MASSES_{SEED_A}"] = "[1.0, 2.0, 3.0]"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertEqual(self.main([SEED_A]), 1)
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.ledger(), run.BackfillLedger({SEED_A: 1}, {}, {SEED_A}))

        # No second render with this binary, whatever the attempt cap; a WARNING every run.
        for extra in ((), ("--max-backfill-attempts", "100")):
            with self.assertLogs(run.log, level="WARNING") as logs:
                self.assertEqual(self.main([SEED_A], *extra), 0)
            self.assertIn(
                "regenerates a different orbit or view than the live package (rebuild the "
                "generator, or delete the seed from identity_mismatches in backfill_failures.json",
                "\n".join(logs.output),
            )
        self.assertEqual(self.generated(), [SEED_A])

        # A rebuilt generator tries it again (and here renders the live orbit this time).
        del os.environ[f"FAKE_GEN_MASSES_{SEED_A}"]
        stat = Path(self.generator).stat()
        os.utime(self.generator, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.main([SEED_A]), 0)
        self.assertEqual(self.generated(), [SEED_A, SEED_A])
        self.assertEqual(self.ledger(), run.BackfillLedger())

    def test_a_different_view_gives_the_seed_up_at_once(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        os.environ[f"FAKE_GEN_VIEW_{SEED_A}"] = view_members(mode='"linear"')
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertEqual(self.main([SEED_A]), 1)
        self.assertIn(
            'generation.drift.mode: live "elliptical", regenerated "linear"', "\n".join(logs.output)
        )
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.ledger(), run.BackfillLedger({SEED_A: 1}, {}, {SEED_A}))
        saved = json.loads(run.BACKFILL_FAILURES.read_text(encoding="utf-8"))
        self.assertEqual(saved["identity_mismatches"], [SEED_A])

        # No second render with this binary: the seed is listed as given up on every run.
        with self.assertLogs(run.log, level="WARNING") as logs:
            self.assertEqual(self.main([SEED_A]), 0)
        self.assertIn("regenerates a different orbit or view", "\n".join(logs.output))
        self.assertEqual(self.generated(), [SEED_A])

    def test_a_given_up_seed_is_skipped_and_logged_every_run(self) -> None:
        self.remote_package(SEED_A)
        run.save_backfill_ledger(run.BackfillLedger({SEED_A: 3}), self.identity())
        for _ in range(2):
            with self.assertLogs(run.log, level="WARNING") as logs:
                self.assertEqual(self.main([SEED_A]), 0)
            self.assertIn("ember backfill given up after 3 attempts", "\n".join(logs.output))
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.main([SEED_A], "--max-backfill-attempts", "4"), 0)
        self.assertEqual(self.generated(), [SEED_A])
        self.assertEqual(self.ledger(), run.BackfillLedger())

    def test_a_rebuilt_generator_retries_given_up_seeds(self) -> None:
        self.remote_package(SEED_A)
        run.save_backfill_ledger(run.BackfillLedger({SEED_A: 3}), self.identity())
        stat = Path(self.generator).stat()
        os.utime(self.generator, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.main([SEED_A]), 0)
        self.assertEqual(self.generated(), [SEED_A])

    def test_a_backfill_seed_whose_ember_edition_fails_counts_and_gives_up(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        os.environ["FAKE_GEN_MODE"] = "core_only"
        for attempt in range(1, 4):
            with self.assertLogs(run.log, level="WARNING"):
                self.assertEqual(self.main([SEED_A]), 1)
            self.assertEqual(self.ledger(), run.BackfillLedger({SEED_A: attempt}))
        with self.assertLogs(run.log, level="WARNING"):
            self.assertEqual(self.main([SEED_A]), 0)
        self.assertEqual(self.generated(), [SEED_A] * 3)
        self.assertEqual(self.remote_snapshot(), live)

    def test_a_killed_generator_is_not_an_ember_attempt(self) -> None:
        self.remote_package(SEED_A)
        os.environ["FAKE_GEN_MODE"] = "signal"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertEqual(self.main([SEED_A]), 1)
        self.assertEqual(self.ledger(), run.BackfillLedger(other_failures={SEED_A: 1}))

    def test_a_backfill_seed_that_always_fails_does_not_stall_the_backfill(self) -> None:
        self.remote_package(SEED_A)
        self.remote_package(SEED_B)
        os.environ[f"FAKE_GEN_MODE_{SEED_A}"] = "fail"
        with self.assertLogs(run.log, level="ERROR"):
            self.assertEqual(self.main([SEED_A, SEED_B]), 1)
        self.assertEqual(self.ledger(), run.BackfillLedger(other_failures={SEED_A: 1}))
        self.assertEqual(self.main([SEED_A, SEED_B]), 0)  # SEED_B is next
        with self.assertLogs(run.log, level="ERROR"):
            self.assertEqual(self.main([SEED_A, SEED_B]), 1)  # the backlog is SEED_A alone
        self.assertEqual(self.generated(), [SEED_A, SEED_B, SEED_A])
        self.assertEqual(self.ledger(), run.BackfillLedger(other_failures={SEED_A: 2}))
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B], remote), ([], [SEED_A]))

    def test_an_urgent_seed_without_its_ember_edition_carries_its_attempt(self) -> None:
        os.environ["FAKE_GEN_MODE"] = "core_only"
        with self.assertLogs(run.log, level="WARNING"):
            self.assertEqual(self.main([SEED_A]), 0)  # the new mint got its core package
        self.assertEqual(self.ledger(), run.BackfillLedger({SEED_A: 1}))
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A], remote), ([], [SEED_A]))

        with self.assertLogs(run.log, level="WARNING"):
            self.assertEqual(self.main([SEED_A]), 1)  # its first backfill run: attempt 2
        self.assertEqual(self.ledger(), run.BackfillLedger({SEED_A: 2}))

        os.environ["FAKE_GEN_MODE"] = "complete"
        self.assertEqual(self.main([SEED_A]), 0)
        self.assertEqual(self.ledger(), run.BackfillLedger())
        self.assertEqual(self.generated(), [SEED_A] * 3)
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        self.assertEqual(remote, remote_listing(SEED_A, run.REQUIRED_PACKAGE_FILES))

    def test_an_unusable_live_package_costs_no_render_and_is_never_given_up(self) -> None:
        package = self.remote_package(SEED_A)
        (package / run.ASSET_MANIFEST).write_text(CORRUPT_JSON, encoding="utf-8")
        live = self.remote_snapshot()
        for _ in range(4):
            with self.assertLogs(run.log, level="ERROR"):
                self.assertEqual(self.main([SEED_A]), 1)
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.ledger(), run.BackfillLedger(other_failures={SEED_A: 4}))
        self.assertEqual(self.remote_snapshot(), live)

    def test_a_stale_generator_uploads_core_packages_and_pauses_the_backfill(self) -> None:
        self.remote_package(SEED_A)
        os.environ["FAKE_GEN_STALE"] = "1"
        os.environ["FAKE_GEN_MODE"] = "stale"
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B]), 0)
        self.assertIn("rebuild the generator", "\n".join(logs.output))
        self.assertEqual(self.generated(), [SEED_B])
        remote = run.list_remote_files("fakehost", "fakeuser", str(self.remote))
        assert remote is not None
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B], remote), ([], [SEED_A, SEED_B]))
        self.assertFalse(run.BACKFILL_FAILURES.exists())  # no ember attempt was made

    def ember_edition(self, seed: str) -> dict[str, str]:
        """The remote ember files of `seed` and its manifest: {path: content}."""
        paths = {f"0x{seed}/{path}" for path in (*run.EMBER_PACKAGE_FILES, run.ASSET_MANIFEST)}
        return {path: text for path, text in self.remote_snapshot().items() if path in paths}

    def test_a_stale_ember_edition_is_withdrawn_and_rendered_again(self) -> None:
        a_package = self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        b_package = self.remote_package(SEED_B, ember=True, algorithm=STALE_ALGORITHM)
        ember_or_manifest = [*run.EMBER_PACKAGE_FILES, run.ASSET_MANIFEST]
        core = {
            path: content
            for path, content in self.remote_snapshot().items()
            if not path.endswith(tuple(ember_or_manifest))
        }

        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], "--max-backfill", "1"), 0)
        text = "\n".join(logs.output)
        self.assertIn(
            f"Withdrew 2 stale ember editions ({STALE_ALGORITHM} -> {CURRENT_ALGORITHM})", text
        )
        self.assertIn("2 missing only the ember edition", text)
        # Both editions went before the plan; the backfill rendered one of them again.
        self.assertEqual(self.generated(), [SEED_A])
        a_certificate = json.loads((a_package / run.EMBER_CERTIFICATE).read_text(encoding="utf-8"))
        self.assertEqual(a_certificate["algorithm"], CURRENT_ALGORITHM)
        self.assertEqual(
            (a_package / run.EMBER_MEDIA_FILES[0]).read_text(encoding="utf-8"),
            f"{run.EMBER_MEDIA_FILES[0]} new\n",
        )
        # SEED_B has no ember edition until a later run: no certificate, media or manifest entry.
        self.assertEqual(self.ember_edition(SEED_B).keys(), {f"0x{SEED_B}/{run.ASSET_MANIFEST}"})
        b_manifest = json.loads((b_package / run.ASSET_MANIFEST).read_text(encoding="utf-8"))
        self.assertFalse(any(run.is_ember_asset(entry) for entry in b_manifest["assets"]))
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B], self.listing()), ([], [SEED_B]))
        # The core packages are the published ones, byte for byte.
        self.assertEqual({path: self.remote_snapshot()[path] for path in core}, core)

        # The next run renders SEED_B; the fresh edition of SEED_A is never withdrawn again.
        a_edition = self.ember_edition(SEED_A)
        self.assertEqual(self.main([SEED_A, SEED_B], "--max-backfill", "1"), 0)
        self.assertEqual(self.generated(), [SEED_A, SEED_B])
        self.assertEqual(self.ember_edition(SEED_A), a_edition)
        b_certificate = json.loads((b_package / run.EMBER_CERTIFICATE).read_text(encoding="utf-8"))
        self.assertEqual(b_certificate["algorithm"], CURRENT_ALGORITHM)

        # Every token shows the current look: a third run has nothing to do.
        live = self.remote_snapshot()
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], "--max-backfill", "1"), 0)
        self.assertIn("complete asset packages on remote. Nothing to do.", "\n".join(logs.output))
        self.assertEqual(self.generated(), [SEED_A, SEED_B])
        self.assertEqual(self.remote_snapshot(), live)

    def assert_current_edition(self, seed: str, published: dict[str, str]) -> None:
        """The remote package of `seed` holds the current look's ember edition, rendered by the
        fake generator ("new"), and is otherwise `published`, its files before the run, byte for
        byte: the ember files, roles and certificate are compared with the EMBER_V3_* literals."""
        package = self.package_files(seed)
        self.assertEqual(
            sorted(package), sorted([*run.CORE_PACKAGE_FILES, *SPECTRAL_FILES, *EMBER_V3_FILES])
        )
        for path in EMBER_V3_FILES[:-1]:
            self.assertEqual(package[path], f"{path} new\n")
        certificate = json.loads(package["metadata/ember.json"])
        self.assertEqual(
            (certificate["algorithm"], certificate["schema_version"], certificate["contract"]),
            ("ember-v3", 4, "the new render"),
        )
        untouched = set(package) - {*EMBER_V3_FILES, "metadata/assets.json"}
        self.assertEqual(
            {path: package[path] for path in untouched},
            {path: published[path] for path in untouched},
        )
        # The manifest: the published entries of everything else, verbatim and in order, then
        # the six entries of the new edition. No entry of the old edition is left.
        manifest = json.loads(package["metadata/assets.json"])
        published_manifest = json.loads(published["metadata/assets.json"])
        kept = [entry for entry in published_manifest["assets"] if not run.is_ember_asset(entry)]
        self.assertEqual(manifest["assets"][: len(kept)], kept)
        ember = manifest["assets"][len(kept) :]
        self.assertEqual([entry["role"] for entry in ember], list(EMBER_V3_ROLES))
        self.assertEqual([entry["path"] for entry in ember], list(EMBER_V3_FILES[:-1]))
        self.assertTrue(all(entry["sha256"].startswith("new:") for entry in ember))

    def assert_no_ember_edition(self, seed: str, published: dict[str, str]) -> None:
        """The remote package of `seed` is `published` without its ember edition: no ember file,
        no ember manifest entry, and every other file byte for byte."""
        package = self.package_files(seed)
        self.assertEqual(sorted(package), sorted([*run.CORE_PACKAGE_FILES, *SPECTRAL_FILES]))
        untouched = set(package) - {"metadata/assets.json"}
        self.assertEqual(
            {path: package[path] for path in untouched},
            {path: published[path] for path in untouched},
        )
        manifest = json.loads(package["metadata/assets.json"])
        published_manifest = json.loads(published["metadata/assets.json"])
        kept = [entry for entry in published_manifest["assets"] if not run.is_ember_asset(entry)]
        self.assertEqual(manifest, {**published_manifest, "assets": kept})

    def test_an_ember_v2_package_is_withdrawn_at_once_and_rendered_again(self) -> None:
        self.remote_v2_package(SEED_A)
        self.remote_v2_package(SEED_B)
        published_a, published_b = self.package_files(SEED_A), self.package_files(SEED_B)
        self.assertEqual(
            sorted(published_a), sorted([*run.CORE_PACKAGE_FILES, *SPECTRAL_FILES, *EMBER_V2_FILES])
        )

        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], "--max-backfill", "1"), 0)
        text = "\n".join(logs.output)
        for seed in (SEED_A, SEED_B):
            self.assertIn(
                f"WITHDRAWN  seed=0x{seed}  its ember-v2 ember edition is off the asset host", text
            )
        self.assertIn("Withdrew 2 stale ember editions (ember-v2 -> ember-v3)", text)
        self.assertNotIn("Kept", text)
        # Both editions went before the plan: six files and five manifest entries each. The
        # backfill rendered one of them again, with the seven files and six roles of ember-v3.
        self.assertEqual(self.generated(), [SEED_A])
        self.assert_current_edition(SEED_A, published_a)
        self.assert_no_ember_edition(SEED_B, published_b)
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B], self.listing()), ([], [SEED_B]))

        # The next run renders SEED_B; the fresh edition of SEED_A is not touched again.
        after_first = self.package_files(SEED_A)
        self.assertEqual(self.main([SEED_A, SEED_B], "--max-backfill", "1"), 0)
        self.assertEqual(self.generated(), [SEED_A, SEED_B])
        self.assert_current_edition(SEED_B, published_b)
        self.assertEqual(self.package_files(SEED_A), after_first)
        self.assertEqual(run.find_missing_seeds([SEED_A, SEED_B], self.listing()), ([], []))

    def test_a_kept_ember_v2_package_stays_online_until_it_is_replaced_in_place(self) -> None:
        self.remote_v2_package(SEED_A)
        self.remote_v2_package(SEED_B)
        published_a, published_b = self.package_files(SEED_A), self.package_files(SEED_B)
        keep = ("--keep-stale-ember", "--max-backfill", "1")

        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], *keep), 0)
        text = "\n".join(logs.output)
        self.assertIn(
            "Kept 2 stale ember editions online (ember-v2 -> ember-v3): the ember backfill "
            "replaces each in place",
            text,
        )
        self.assertNotIn("WITHDRAWN", text)
        self.assertNotIn("Withdrew", text)
        # Nothing was withdrawn: SEED_B is the published ember-v2 package, byte for byte.
        self.assertEqual(self.package_files(SEED_B), published_b)
        # SEED_A was replaced in place: its five old media replaced, the slow film added, the
        # manifest's five ember entries replaced by six, a layout-4 ember-v3 certificate.
        self.assertEqual(self.generated(), [SEED_A])
        self.assert_current_edition(SEED_A, published_a)
        # Every file was staged beside the live edition, which stayed whole until one ssh call
        # swapped the new one in; the only live file ever deleted is the old certificate, there.
        self.assertEqual(self.uploaded(), [[path] for path in EMBER_UPLOAD_ORDER])
        self.assertEqual(self.staged(), [f"0x{SEED_A}/{path}.part" for path in EMBER_UPLOAD_ORDER])
        self.assertEqual(self.changes(), self.ember_upload_commands(SEED_A))

        # The next run replaces SEED_B; the fresh edition of SEED_A is not rendered again.
        after_first = self.package_files(SEED_A)
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], *keep), 0)
        self.assertIn("Kept 1 stale ember editions online", "\n".join(logs.output))
        self.assertEqual(self.generated(), [SEED_A, SEED_B])
        self.assert_current_edition(SEED_B, published_b)
        self.assertEqual(self.package_files(SEED_A), after_first)

        # Every token shows the current look: a third run has nothing to keep or to do.
        live = self.remote_snapshot()
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], *keep), 0)
        text = "\n".join(logs.output)
        self.assertIn("complete asset packages on remote. Nothing to do.", text)
        self.assertNotIn("Kept", text)
        self.assertEqual(self.generated(), [SEED_A, SEED_B])
        self.assertEqual(self.remote_snapshot(), live)

    def test_a_kept_edition_survives_a_failed_upload_untouched(self) -> None:
        package = self.remote_v2_package(SEED_A)
        published = self.package_files(SEED_A)
        swap = f"rm -f -- {package}/metadata/ember.json"
        failures = (
            # The slow film cannot be written (a lost connection, a full asset host).
            ("FAKE_SCP_FAIL_ON", "videos/web/ember_slow.mp4"),
            # Every file is staged, but the ssh call that swaps them in never starts.
            ("FAKE_SSH_FAIL_ON", swap),
        )
        for attempt, (knob, value) in enumerate(failures, 1):
            with self.subTest(knob=knob):
                os.environ[knob] = value
                with self.assertLogs(run.log, level="ERROR") as logs:
                    self.assertEqual(self.main([SEED_A], "--keep-stale-ember"), 1)
                del os.environ[knob]
                self.assertIn("UPLOAD FAILED", "\n".join(logs.output))
                # The published ember-v2 edition is online as it was, byte for byte: its six
                # files, its manifest, and no staged file beside them.
                self.assertEqual(self.package_files(SEED_A), published)
                self.assertEqual(self.generated(), [SEED_A] * attempt)
                # Not an ember attempt: the seed only moves back in the queue.
                self.assertEqual(
                    self.ledger(), run.BackfillLedger(other_failures={SEED_A: attempt})
                )

        # The next run replaces it.
        self.assertEqual(self.main([SEED_A], "--keep-stale-ember"), 0)
        self.assert_current_edition(SEED_A, published)
        self.assertEqual(self.ledger(), run.BackfillLedger())

    def test_a_cut_off_swap_leaves_a_backfill_seed_that_a_later_run_finishes(self) -> None:
        package = self.remote_v2_package(SEED_A)
        published = self.package_files(SEED_A)
        # The connection is lost in the middle of the swap, after the film was renamed into
        # place: the certificate is gone, four media are new, the rest of the package is old.
        os.environ["FAKE_SSH_CUT_AFTER"] = f"mv -f -- {package}/videos/web/ember.mp4.part "
        with self.assertLogs(run.log, level="ERROR"):
            self.assertEqual(self.main([SEED_A], "--keep-stale-ember"), 1)
        del os.environ["FAKE_SSH_CUT_AFTER"]

        swapped = EMBER_V3_FILES[:4]
        expected = {path: text for path, text in published.items() if path != "metadata/ember.json"}
        expected.update({path: f"{path} new\n" for path in swapped})
        # No file is truncated or half new: each medium is the old one or the new one, the
        # manifest is the published one, and the staged files that were not swapped in are gone.
        self.assertEqual(self.package_files(SEED_A), expected)
        self.assertEqual(run.find_missing_seeds([SEED_A], self.listing()), ([], [SEED_A]))

        # Without a certificate the package is a backfill seed, switch or no switch: the next
        # run renders it again and repeats the whole upload.
        self.assertEqual(self.main([SEED_A]), 0)
        self.assert_current_edition(SEED_A, published)
        self.assertEqual(self.generated(), [SEED_A, SEED_A])

    def test_a_kept_edition_survives_a_failed_or_mismatched_render(self) -> None:
        self.remote_v2_package(SEED_A)
        self.remote_v2_package(SEED_B)
        live = self.remote_snapshot()
        os.environ[f"FAKE_GEN_MODE_{SEED_A}"] = "core_only"
        os.environ[f"FAKE_GEN_VIEW_{SEED_B}"] = view_members(symmetry='"mirror_x"')
        for _ in range(2):
            with self.assertLogs(run.log, level="WARNING"):
                self.assertEqual(self.main([SEED_A, SEED_B], "--keep-stale-ember"), 1)
        # Each render failed, so nothing was uploaded or deleted: both ember-v2 editions are
        # online as published, and the mismatched one is given up without losing its edition.
        self.assertEqual(self.generated(), [SEED_A, SEED_B])
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.ledger(), run.BackfillLedger({SEED_A: 1, SEED_B: 1}, {}, {SEED_B}))

    def test_the_switch_and_a_paused_backfill_hold_every_edition_as_it_is(self) -> None:
        self.remote_v2_package(SEED_A)
        self.remote_package(SEED_B)  # no ember edition yet
        live = self.remote_snapshot()
        for _ in range(2):
            self.assertEqual(
                self.main([SEED_A, SEED_B], "--keep-stale-ember", "--max-backfill", "0"), 0
            )
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.calls("scp"), [])

        # Without the switch a paused backfill renders nothing either, but the stale edition is
        # still withdrawn at once.
        published = self.package_files(SEED_A)
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A, SEED_B], "--max-backfill", "0"), 0)
        self.assertIn(
            "Withdrew 1 stale ember editions (ember-v2 -> ember-v3)", "\n".join(logs.output)
        )
        self.assertEqual(self.generated(), [])
        self.assert_no_ember_edition(SEED_A, published)

    def test_the_keep_stale_ember_switch_replaces_every_older_edition(self) -> None:
        # An edition of an older look that has every file of the current one (a look change
        # that adds no file): the switch plans it for replacement by its algorithm alone.
        package = self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        published = self.package_files(SEED_A)
        os.environ[run.ENV_KEEP_STALE_EMBER] = "yes"
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A]), 0)
        self.assertIn("Kept 1 stale ember editions online", "\n".join(logs.output))
        self.assertEqual(self.generated(), [SEED_A])
        self.assert_current_edition(SEED_A, published)
        # Replaced by the staged swap, not withdrawn: no live medium was ever deleted.
        self.assertEqual(self.changes(), self.ember_upload_commands(SEED_A))

        # The command line overrides the environment: `--keep-stale-ember no` withdraws.
        (package / run.EMBER_CERTIFICATE).write_text(
            certificate_text(STALE_ALGORITHM, "live"), encoding="utf-8"
        )
        published = self.package_files(SEED_A)
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(
                self.main([SEED_A], "--keep-stale-ember", "no", "--max-backfill", "0"), 0
            )
        self.assertIn("Withdrew 1 stale ember editions", "\n".join(logs.output))
        self.assert_no_ember_edition(SEED_A, published)
        self.assertEqual(self.generated(), [SEED_A])

    def test_a_dry_run_withdraws_nothing_and_shows_the_plan(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        live = self.remote_snapshot()
        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertEqual(self.main([SEED_A], "--dry-run"), 0)
        text = "\n".join(logs.output)
        self.assertIn(
            f"DRY-RUN  would withdraw the {STALE_ALGORITHM} ember edition of 0x{SEED_A}", text
        )
        self.assertIn(f"DRY-RUN  would regenerate 0x{SEED_A}", text)
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.calls("scp"), [])

    def test_a_generator_without_the_ember_algorithm_probe_withdraws_nothing(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        live = self.remote_snapshot()
        os.environ["FAKE_GEN_EMBER_ALGORITHM"] = ""  # a binary that predates the flag
        with self.assertLogs(run.log, level="WARNING") as logs:
            self.assertEqual(self.main([SEED_A]), 0)
        self.assertIn("stale ember editions cannot be detected", "\n".join(logs.output))
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.generated(), [])

    def test_unreadable_certificates_withdraw_nothing_and_fail_the_run(self) -> None:
        self.remote_package(SEED_A, ember=True, algorithm=STALE_ALGORITHM)
        live = self.remote_snapshot()
        os.environ["FAKE_SSH_FAIL_ON"] = f"cd {shlex.quote(str(self.remote))} || exit 1; for "
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertEqual(self.main([SEED_A]), 1)
        self.assertIn("none is withdrawn this run", "\n".join(logs.output))
        self.assertEqual(self.remote_snapshot(), live)
        self.assertEqual(self.generated(), [])

    def test_the_keep_stale_ember_setting(self) -> None:
        self.assertFalse(run.parse_args([]).keep_stale_ember)
        self.assertTrue(run.parse_args(["--keep-stale-ember"]).keep_stale_ember)
        for value, expected in (
            ("yes", True),
            ("TRUE", True),
            (" on ", True),
            ("1", True),
            ("no", False),
            ("off", False),
            ("0", False),
            ("", False),
        ):
            with self.subTest(value=value):
                os.environ[run.ENV_KEEP_STALE_EMBER] = value
                self.assertIs(run.parse_args([]).keep_stale_ember, expected)
        os.environ[run.ENV_KEEP_STALE_EMBER] = "maybe"
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            run.parse_args([])
        self.assertEqual(raised.exception.code, 2)
        del os.environ[run.ENV_KEEP_STALE_EMBER]
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            run.parse_args(["--keep-stale-ember", "2"])
        self.assertEqual(raised.exception.code, 2)

    def test_the_seed_timeout_stays_well_below_the_run_ceiling(self) -> None:
        unit = REPO_ROOT / "ops" / "systemd" / "cosmicsig-sync.service"
        ceiling = [
            int(line.split("=", 1)[1])
            for line in unit.read_text(encoding="utf-8").splitlines()
            if line.startswith("TimeoutStartSec=")
        ]
        self.assertEqual(len(ceiling), 1)
        self.assertLessEqual(run.DEFAULT_TIMEOUT, ceiling[0] // 2)

    def test_dry_run_generates_and_uploads_nothing(self) -> None:
        self.remote_package(SEED_A)
        self.assertEqual(self.main([SEED_A, SEED_B], "--dry-run"), 0)
        self.assertEqual(self.generated(), [])
        self.assertEqual(self.calls("scp"), [])
        self.assertFalse(run.BACKFILL_FAILURES.exists())

    def hold_run_lock(self) -> None:
        """Hold run.py's single-instance lock like another run would (pid 4242)."""
        holder = os.open(run.RUN_LOCK, os.O_RDWR | os.O_CREAT)
        self.addCleanup(os.close, holder)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(holder, b"4242\n")

    def test_a_held_run_lock_stops_the_run_before_it_does_anything(self) -> None:
        self.remote_package(SEED_A)
        live = self.remote_snapshot()
        self.hold_run_lock()
        for extra in ((), ("--dry-run",)):
            with self.subTest(extra=extra):
                with (
                    mock.patch.object(run, "resolve_generator") as resolve_generator,
                    self.assertLogs(run.log, level="ERROR") as logs,
                ):
                    self.assertEqual(self.main([SEED_A, SEED_B], *extra), 1)
                resolve_generator.assert_not_called()
                self.assertIn(f"{self.work / 'run.lock'} (pid 4242)", "\n".join(logs.output))
        self.assertEqual(self.call_log(), [])  # no listing, render or upload
        self.assertEqual(self.remote_snapshot(), live)
        self.assertFalse(run.LOCAL_OUTPUT_DIR.exists())

    def test_help_does_not_need_the_run_lock(self) -> None:
        self.hold_run_lock()
        with mock.patch("sys.stdout"), self.assertRaises(SystemExit) as raised:
            self.main([], "--help")
        self.assertEqual(raised.exception.code, 0)

    def test_the_run_lock_is_held_for_the_run_and_released_at_exit(self) -> None:
        os.environ["FAKE_GEN_LOCK_PROBE"] = str(self.work / "run.lock")
        self.assertEqual(self.main([SEED_A]), 0)
        # During the render the lock was taken, and the generator (a child) held no descriptor
        # of it: the lock is run.py's alone.
        self.assertEqual(self.calls("lock_probe"), [[False, False]])
        self.assertEqual(run.run_lock_holder(), str(os.getpid()))
        fd = run.acquire_run_lock()  # released at exit
        self.assertIsNotNone(fd)
        assert fd is not None
        os.close(fd)

        with (
            mock.patch.object(run, "sync", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError),
        ):
            self.main([SEED_A])
        fd = run.acquire_run_lock()  # released when the run fails, too
        self.assertIsNotNone(fd)
        assert fd is not None
        os.close(fd)

    def test_invalid_backfill_settings_are_rejected(self) -> None:
        for extra in (["--backfill-mode", "partial"], ["--max-backfill-attempts", "0"]):
            with self.subTest(extra=extra), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as raised:
                    run.parse_args(extra)
                self.assertEqual(raised.exception.code, 2)
        os.environ[run.ENV_BACKFILL_MODE] = "FULL"
        os.environ[run.ENV_MAX_BACKFILL_ATTEMPTS] = "5"
        args = run.parse_args([])
        self.assertIs(args.backfill_mode, run.BackfillMode.FULL)
        self.assertEqual(args.max_backfill_attempts, 5)

    def test_preflight_fails_for_a_stale_generator(self) -> None:
        def preflight() -> bool:
            with (
                mock.patch.object(run, "eth_call_uint256", return_value=5),
                mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            ):
                return run.preflight(
                    "fakehost",
                    "fakeuser",
                    str(self.remote),
                    "",
                    "https://rpc.invalid",
                    run.DEFAULT_NFT_CONTRACT,
                    self.generator,
                )

        with self.assertLogs(run.log, level="INFO") as logs:
            self.assertTrue(preflight())
        self.assertIn(
            f"ember edition supported, renders {CURRENT_ALGORITHM}", "\n".join(logs.output)
        )
        os.environ["FAKE_GEN_STALE"] = "1"
        with self.assertLogs(run.log, level="ERROR") as logs:
            self.assertFalse(preflight())
        self.assertIn("predates the ember edition", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
