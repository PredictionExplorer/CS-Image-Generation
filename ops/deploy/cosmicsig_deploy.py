#!/usr/bin/env python3
"""Continuous deployment of GitHub ``main`` to the CosmicSignature generator host.

A systemd user timer (cosmicsig-deploy.timer) runs ``cosmicsig_deploy.py run`` every two minutes.
One tick fetches ``origin/main``; if it names a commit that is not deployed yet and whose CI
passed on GitHub, the tick builds and tests that commit natively in a staging worktree, waits
until no sync run (run.py) is active, and then switches the production checkout to it: a
fast-forward, an atomic install of the tested generator binary, the systemd user units rendered
from ops/systemd/, and a smoke test. Any failure during the switch undoes all of it. Finally it
restarts the sync timer and starts a sync run, which regenerates whatever the new version needs.

Safety invariants (docs/deployment.md explains each one):

- The production checkout only ever moves forward along ``main`` (``git merge --ff-only``), and
  only from a clean tracked tree on branch ``main``; a rewritten ``main`` is refused. Untracked
  production state (.env, output/, logs, ledgers) is never touched: a commit that would write a
  path where an untracked file lives is refused.
- A commit is deployed only if GitHub reports a completed, successful ``CI passed`` check run from
  GitHub Actions for exactly that commit, and only after ``cargo build``, ``cargo test`` and the
  Python unit tests passed for it on this host.
- A sync run is never interrupted: the switch waits (without an upper bound) for the sync service
  to finish and then holds run.lock, the lock run.py itself takes, so a run.py started by hand is
  waited for too.
- A pause (``pause``, or ``rollback``, which pauses first) stops every switch that has not
  started changing the checkout yet, even one whose build was already running.
- A failed build, test or switch is final for that commit until ``retry`` or a new commit; a CI
  failure is re-checked every 15 minutes (a re-run on GitHub can turn it green).
- A switch whose undo failed, or that died half way (killed, a power loss), pauses auto-deploy
  and disables the sync timer: a human puts the checkout back before anything runs on it.
- A no-op tick is quiet and cheap: one ``git fetch``, no GitHub API call, nothing written.

Subcommands: run, status, install, pause, resume, retry, rollback (``--help`` for each).
Standard library only; Python 3.10+.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import enum
import fcntl
import hashlib
import http.client
import json
import logging
import os
import pwd
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TextIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNC_SERVICE = "cosmicsig-sync.service"
SYNC_TIMER = "cosmicsig-sync.timer"
DEPLOY_SERVICE = "cosmicsig-deploy.service"
DEPLOY_TIMER = "cosmicsig-deploy.timer"
# The unit templates in ops/systemd/, installed as user units under the same names.
UNIT_NAMES = (SYNC_SERVICE, SYNC_TIMER, DEPLOY_SERVICE, DEPLOY_TIMER)

# Paths inside the production checkout (and inside every commit that is deployed).
UNIT_TEMPLATE_DIR = "ops/systemd"
AGENT_PATH = "ops/deploy/cosmicsig_deploy.py"
SYNC_SCRIPT = "run.py"
BINARY_NAME = "three_body_problem"
BINARY_PATH = f"target/release/{BINARY_NAME}"
PREVIOUS_SUFFIX = ".previous"
# run.py's single-instance lock (in its working directory, the checkout).
RUN_LOCK = "run.lock"
# The branch the checkout must be on, and the remote-tracking ref of GitHub main.
BRANCH = "main"
ORIGIN_MAIN = "refs/remotes/origin/main"
FETCH_REFSPEC = f"+refs/heads/{BRANCH}:{ORIGIN_MAIN}"
# Files every deployed commit must contain: without them the next tick could not render the
# units or start the agent, and auto-deploy would stop for good.
REQUIRED_TREE_PATHS = (
    *(f"{UNIT_TEMPLATE_DIR}/{name}" for name in UNIT_NAMES),
    AGENT_PATH,
    SYNC_SCRIPT,
)
# Unit template placeholders.
PLACEHOLDER_REPO = "@REPO@"
PLACEHOLDER_PYTHON = "@PYTHON@"
_LEFTOVER_PLACEHOLDER_RE = re.compile(r"@[A-Z][A-Z_]*@")
# Paths substituted into unit files must not need systemd quoting or specifier escaping.
_UNIT_SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+-]*$")

# The legacy SYSTEM units that ops/server/bootstrap-root.sh removes. While they exist, two
# schedulers would start the sync, so the agent refuses to run.
LEGACY_UNIT_DIR = Path("/etc/systemd/system")
LEGACY_UNITS = (SYNC_SERVICE, SYNC_TIMER)

# Environment.
ENV_REPO = "COSMICSIG_DEPLOY_REPO"
ENV_GITHUB_API = "COSMICSIG_DEPLOY_GITHUB_API"
ENV_GITHUB_REPO = "COSMICSIG_DEPLOY_GITHUB_REPO"
ENV_GITHUB_TOKEN = "COSMICSIG_DEPLOY_GITHUB_TOKEN"
ENV_GITHUB_TOKEN_FALLBACK = "GITHUB_TOKEN"
# Only the agent itself talks to the API, so these are removed from the environment of every
# command it runs: the token stays out of child environments and their logs. That is no
# barrier against the code the host builds and tests, though: build scripts, proc macros and
# the staged tests run as this same user and can read the env file (or this process's
# /proc/PID/environ) directly. Hence the docs prescribe a fine-grained token with read-only
# access to public repositories only, and an expiry date.
TOKEN_VARIABLES = (ENV_GITHUB_TOKEN, ENV_GITHUB_TOKEN_FALLBACK)
ENV_PYTHON = "COSMICSIG_DEPLOY_PYTHON"
DEFAULT_GITHUB_API = "https://api.github.com"
SYSTEM_PYTHON = Path("/usr/bin/python3")
# Checks the rendered unit files during a switch, where it is installed (systemd ships it).
SYSTEMD_ANALYZE = "systemd-analyze"

# The CI gate: the aggregate job of .github/workflows/ci.yml (job id ci-passed).
CI_CHECK_NAME = "CI passed"
CI_APP_SLUG = "github-actions"
CI_RECHECK_INTERVAL = datetime.timedelta(minutes=15)
CI_STALL_WARNING = datetime.timedelta(minutes=45)
"""How long after a commit was committed (for a pull request merge: merged) the agent waits
for its `CI passed` result quietly. A full CI run takes 15 to 25 minutes, more when macOS
rebuilds FFmpeg. After that, every tick that still finds no result logs a WARNING and records
it for `status`: the commit may have no Actions run at all (GitHub skips the push run when the
commit message contains `[skip ci]` or a similar instruction), or its run may hang."""
API_TIMEOUT = 30.0
USER_AGENT = "cosmicsig-deploy/1 (+https://github.com/PredictionExplorer/CS-Image-Generation)"

# Failure reasons recorded in state.json. Every reason but CI is final for its commit until
# `retry` or a new commit: CI can turn green by a re-run on GitHub, so it is re-checked.
REASON_CI = "ci"
REASON_BUILD = "build"
REASON_TESTS = "tests"
REASON_SWITCH = "switch"
REASON_ROLLBACK = "rollback"
MAX_FAILED_SHAS = 20

# Command timeouts (seconds). Builds run niced beside an hour-long render that uses every core,
# so their limits are generous.
GIT_TIMEOUT = 300.0
FETCH_TIMEOUT = 600.0
SYSTEMCTL_TIMEOUT = 120.0
SMOKE_TIMEOUT = 120.0
BUILD_TIMEOUT = 3 * 3600.0
TEST_TIMEOUT = 3 * 3600.0
PYTHON_TEST_TIMEOUT = 3600.0
NICE = ("nice", "-n", "10")

# Waiting for the sync run and for locks.
SYNC_POLL_SECONDS = 30.0
SYNC_PROGRESS_SECONDS = 600.0
RUN_LOCK_TIMEOUT = 3600.0
LOCK_POLL_SECONDS = 1.0
LOCK_PROGRESS_SECONDS = 60.0
# systemd ActiveState values during which the sync service is (still) running.
BUSY_STATES = frozenset({"active", "activating", "deactivating", "reloading", "refreshing"})
KNOWN_ACTIVE_STATES = BUSY_STATES | {"inactive", "failed", "maintenance"}

# Releases kept in $DATA_DIR/releases (the deployed and previous ones are always kept too).
KEEP_RELEASES = 5
RELEASE_MANIFEST = "release.json"

log = logging.getLogger("cosmicsig.deploy")

# How often a wait looks at the stop request (the latency of `systemctl stop` during a wait).
STOP_POLL_SECONDS = 0.1


class _StopRequest:
    """A stop request (SIGTERM, SIGINT or SIGHUP), recorded by the signal handler.

    Deliberately lock-free instead of a threading.Event: Event.set() takes the lock that
    Event.wait() holds for a few bytecodes, and a Python signal handler runs in the main thread
    between bytecodes, so a signal arriving at that moment would deadlock the agent (until
    systemd's SIGKILL, which leaves the sync timer stopped). Storing an attribute is atomic.
    """

    def __init__(self) -> None:
        self._requested = False

    def set(self) -> None:
        """Request a stop (safe in a signal handler)."""
        self._requested = True

    def clear(self) -> None:
        """Forget the request."""
        self._requested = False

    def is_set(self) -> bool:
        """True once a stop was requested."""
        return self._requested

    def wait(self, timeout: float) -> bool:
        """Sleep up to `timeout` seconds; True as soon as (within STOP_POLL_SECONDS) a stop is
        requested, False if the time ran out first."""
        deadline = time.monotonic() + timeout
        while not self._requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, STOP_POLL_SECONDS))
        return True


# Set by SIGTERM/SIGINT/SIGHUP. Waits and builds stop at the next poll; the switch itself (seconds)
# never looks at it, so a stop request cannot leave the checkout half-switched.
_stop = _StopRequest()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeployError(Exception):
    """A failure that ends the current command (logged as an ERROR, exit status 1)."""


class CommandFailed(DeployError):
    """An external command failed, timed out or could not be started."""


class BuildFailed(Exception):
    """The build or a test suite failed for a commit: final for that commit."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage} failed: {detail}")
        self.stage = stage
        self.detail = detail


class SwitchFailed(Exception):
    """The switch failed and was rolled back completely: final for that commit."""


class RollbackFailed(Exception):
    """Undoing a failed switch failed too. Auto-deploy is paused and a human must intervene."""


class Abandoned(Exception):
    """The switch was given up before anything changed (paused, or run.lock stayed busy)."""

    def __init__(self, message: str, *, paused: bool) -> None:
        super().__init__(message)
        self.paused = paused


class Interrupted(Exception):
    """SIGTERM, SIGINT or SIGHUP arrived while waiting or building; nothing was switched."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime.datetime:
    """The current time, timezone-aware UTC."""
    return datetime.datetime.now(datetime.timezone.utc)


def isoformat(moment: datetime.datetime) -> str:
    """An ISO 8601 timestamp to the second (parseable by datetime.fromisoformat on 3.10)."""
    return moment.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(text: str | None) -> datetime.datetime | None:
    """The time an isoformat() string names, or None if absent or malformed."""
    if not text:
        return None
    try:
        moment = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def fmt_duration(seconds: float) -> str:
    """A compact duration such as 42s, 3m07s or 1h02m03s."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def short(sha: str | None) -> str:
    """An abbreviated commit id for log lines (12 hex digits)."""
    return sha[:12] if sha else "-"


_CREDENTIALS_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.IGNORECASE)


def redact(text: str) -> str:
    """`text` with any user:password@ part of a URL masked (git errors may quote the URL)."""
    return _CREDENTIALS_RE.sub(r"\g<scheme>***@", text)


def file_sha256(path: Path) -> str | None:
    """The SHA-256 of a file's bytes, or None if it does not exist."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return None
    return digest.hexdigest()


def _fsync_dir(directory: Path) -> None:
    """Make a rename in `directory` durable (best effort: not every filesystem supports it)."""
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Replace `path` with `data` atomically: a reader sees the old or the new file, never part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.chmod(mode)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, data: object) -> None:
    """Replace `path` with `data` as indented JSON, atomically."""
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def copy_file_atomic(source: Path, dest: Path, *, mode: int) -> None:
    """Copy `source` over `dest` atomically, keeping its modification time.

    Keeping the mtime matters for the generator binary: run.py identifies the binary by path,
    size and mtime (its backfill ledger resets when the identity changes), so reinstalling the
    same release restores the same identity.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as src:
            shutil.copyfileobj(src, out, 1 << 20)
            out.flush()
            os.fsync(out.fileno())
        shutil.copystat(source, tmp)
        tmp.chmod(mode)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(dest.parent)


def _as_str(value: object) -> str | None:
    """`value` if it is a string, else None."""
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# sd-daemon(3) priority prefixes: journald files a stdout/stderr line that starts with <N> under
# syslog priority N, so `journalctl -p warning` shows exactly the WARNING and ERROR lines.
_JOURNAL_PRIORITY = {
    logging.DEBUG: 7,
    logging.INFO: 6,
    logging.WARNING: 4,
    logging.ERROR: 3,
    logging.CRITICAL: 2,
}


class _Formatter(logging.Formatter):
    """Plain `LEVEL message` lines (journald adds the timestamps), no colours.

    Under journald every line of a record also gets the record's priority prefix.
    """

    def __init__(self, journal: bool) -> None:
        super().__init__("%(levelname)s %(message)s")
        self._journal = journal

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self._journal:
            return text
        priority = _JOURNAL_PRIORITY.get(record.levelno, 6)
        return "\n".join(f"<{priority}>{line}" for line in text.splitlines())


def stream_is_journal(stream: TextIO) -> bool:
    """True if `stream` is connected to journald (systemd sets JOURNAL_STREAM=dev:inode)."""
    expected = os.environ.get("JOURNAL_STREAM", "")
    if not expected:
        return False
    try:
        stat = os.fstat(stream.fileno())
    except (OSError, ValueError):
        return False
    return expected == f"{stat.st_dev}:{stat.st_ino}"


def setup_logging(verbose: bool) -> None:
    """Log to stderr: INFO and up, or DEBUG and up with --verbose (idempotent)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter(stream_is_journal(sys.stderr)))
    log.handlers[:] = [handler]
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def stop_signals() -> Iterator[None]:
    """Turn SIGTERM, SIGINT and SIGHUP into a stop request (`_stop`) for the duration of a command.

    `systemctl stop` sends SIGTERM to the agent only (the deploy service uses KillMode=mixed), so
    a git command in flight is never killed by it: waits and builds end at their next poll, and
    the switch, which takes seconds, always completes or rolls back first.

    SIGHUP matters for `rollback`, which an operator runs over SSH and which can wait hours for
    a sync run: when the session drops, the terminal's hangup would otherwise kill the agent at
    once, skipping the code that puts the stopped sync timer back.

    A signal the agent was started with ignored stays ignored: that is the caller's choice, as
    with `nohup` (SIGHUP) or a background job of a script (SIGINT). A rollback run under nohup
    must survive the hangup it was protected from. (systemd resets every handler for a service.)
    """

    def request_stop(_signum: int, _frame: types.FrameType | None) -> None:
        _stop.set()

    previous = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        if signal.getsignal(sig) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _stop.clear()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _xdg_dir(variable: str, fallback: str) -> Path:
    """An XDG base directory: $variable if it is an absolute path, else ~/<fallback>."""
    value = os.environ.get(variable, "")
    if value and Path(value).is_absolute():
        return Path(value)
    return Path.home() / fallback


def default_repo() -> Path:
    """The production checkout: $COSMICSIG_DEPLOY_REPO, else the checkout holding this script."""
    override = os.environ.get(ENV_REPO, "")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class Paths:
    """Every location the agent reads or writes."""

    repo: Path
    """The production checkout (run.py's working directory)."""
    state_dir: Path
    """$XDG_STATE_HOME/cosmicsig-deploy: state.json, the pause flag and the locks."""
    data_dir: Path
    """$XDG_DATA_HOME/cosmicsig-deploy: the staging worktree, cargo target dir and releases."""
    unit_dir: Path
    """$XDG_CONFIG_HOME/systemd/user: where systemd --user reads the installed units."""

    @classmethod
    def from_env(cls) -> Paths:
        """The paths for this user and environment."""
        config = _xdg_dir("XDG_CONFIG_HOME", ".config")
        return cls(
            repo=default_repo(),
            state_dir=_xdg_dir("XDG_STATE_HOME", ".local/state") / "cosmicsig-deploy",
            data_dir=_xdg_dir("XDG_DATA_HOME", ".local/share") / "cosmicsig-deploy",
            unit_dir=config / "systemd" / "user",
        )

    @property
    def state_file(self) -> Path:
        """The deployment record."""
        return self.state_dir / "state.json"

    @property
    def state_lock(self) -> Path:
        """Serializes read-modify-write updates of state.json (held for milliseconds)."""
        return self.state_dir / "state.lock"

    @property
    def deploy_lock(self) -> Path:
        """Held for a whole tick, install or rollback: one of them at a time."""
        return self.state_dir / "deploy.lock"

    @property
    def pause_file(self) -> Path:
        """Exists while auto-deploy is paused; holds the reason."""
        return self.state_dir / "paused"

    @property
    def stage(self) -> Path:
        """The staging worktree where commits are built and tested."""
        return self.data_dir / "stage"

    @property
    def cargo_target_dir(self) -> Path:
        """The persistent CARGO_TARGET_DIR (incremental builds across deploys)."""
        return self.data_dir / "target"

    @property
    def releases(self) -> Path:
        """Built and tested binaries: releases/<sha>/three_body_problem."""
        return self.data_dir / "releases"

    @property
    def installed_binary(self) -> Path:
        """The generator run.py uses."""
        return self.repo / BINARY_PATH


def python_executable() -> str:
    """The interpreter the units run run.py and the agent with, also used for staged tests.

    $COSMICSIG_DEPLOY_PYTHON, else /usr/bin/python3 (the distribution's, which unattended
    upgrades keep patched), else the interpreter running this script.
    """
    override = os.environ.get(ENV_PYTHON, "")
    if override:
        return override
    if SYSTEM_PYTHON.is_file() and os.access(SYSTEM_PYTHON, os.X_OK):
        return str(SYSTEM_PYTHON)
    return sys.executable


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Completed:
    """A finished command. In streaming mode `stdout` holds the tail of the merged output."""

    returncode: int
    stdout: str
    stderr: str


def _output_tail(text: str, lines: int = 8) -> str:
    """The last `lines` non-empty lines of `text`, joined with ' | ' for a one-line message."""
    kept = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(kept[-lines:])


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Stop a streamed command and everything it started: SIGTERM, then SIGKILL."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def run_command(
    argv: Sequence[str],
    *,
    label: str,
    timeout: float,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    stream: bool = False,
) -> Completed:
    """Run an external command: the one place the agent starts processes.

    Every command gets a timeout, stdin from /dev/null (nothing can prompt), an environment
    without the GitHub token (TOKEN_VARIABLES), and a DEBUG log line. Captured output is
    returned; with `stream`, stdout and stderr are merged and logged
    line by line at INFO as they arrive (so a build shows up in the journal live), the command
    runs in its own process group so a timeout or stop request ends everything it started, and
    a stop request raises Interrupted.

    Raises CommandFailed if the command cannot be started or times out, or (with `check`) if it
    exits non-zero.
    """
    argv = list(argv)
    child_env = dict(os.environ if env is None else env)
    for variable in TOKEN_VARIABLES:
        child_env.pop(variable, None)
    env = child_env
    log.debug("[%s] $ %s", label, redact(shlex.join(argv)))
    if stream:
        completed = _run_streaming(argv, label=label, timeout=timeout, cwd=cwd, env=env)
    else:
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandFailed(
                f"{label}: `{redact(shlex.join(argv))}` timed out after {fmt_duration(timeout)}"
            ) from exc
        except OSError as exc:
            raise CommandFailed(f"{label}: cannot run `{argv[0]}`: {exc}") from exc
        completed = Completed(proc.returncode, proc.stdout, proc.stderr)
        if completed.stdout.strip():
            log.debug("[%s] stdout: %s", label, redact(completed.stdout.rstrip()[-2000:]))
        if completed.stderr.strip():
            log.debug("[%s] stderr: %s", label, redact(completed.stderr.rstrip()[-2000:]))
    if check and completed.returncode != 0:
        tail = _output_tail(completed.stderr) or _output_tail(completed.stdout)
        raise CommandFailed(
            f"{label}: `{redact(shlex.join(argv))}` exited with status {completed.returncode}"
            + (f": {redact(tail)}" if tail else "")
        )
    return completed


def _run_streaming(
    argv: list[str],
    *,
    label: str,
    timeout: float,
    cwd: Path | None,
    env: dict[str, str],
) -> Completed:
    """run_command()'s streaming mode (see there)."""
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise CommandFailed(f"{label}: cannot run `{argv[0]}`: {exc}") from exc
    tail: deque[str] = deque(maxlen=40)

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            tail.append(line)
            log.info("[%s] %s", label, redact(line))

    reader = threading.Thread(target=pump, name=f"{label} output", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                returncode = proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if _stop.is_set():
                    _kill_group(proc)
                    raise Interrupted(f"{label} stopped by a signal") from None
                if time.monotonic() >= deadline:
                    _kill_group(proc)
                    raise CommandFailed(
                        f"{label}: `{redact(shlex.join(argv))}` timed out after "
                        f"{fmt_duration(timeout)}"
                    ) from None
    finally:
        reader.join(timeout=10)
        # A reader still blocked here means a killed command left a descendant holding the
        # pipe; the daemon thread and the pipe then end with the agent.
        if proc.stdout is not None and not reader.is_alive():
            proc.stdout.close()
    return Completed(returncode, "\n".join(tail), "")


def _c_locale_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment with messages in the C locale, so their text can be matched."""
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    if extra:
        env.update(extra)
    return env


def git_env() -> dict[str, str]:
    """The environment for git: never prompts, no optional locks, messages in English."""
    env = _c_locale_env({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"})
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    return env


# Options for every git command of the agent: no repository hooks run (the checkout's
# .git/hooks are not part of a deploy, and a post-merge or post-checkout hook must not run
# unattended), and an automatic gc runs in the foreground: a detached one would outlive the
# tick and be killed when the service stops.
GIT_OPTIONS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "gc.autoDetach=false",
    "-c",
    "maintenance.autoDetach=false",
)


def _git_subcommand(args: Sequence[str]) -> str:
    """The subcommand in git arguments, skipping leading options such as `-c NAME=VALUE`."""
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 2 if args[index] == "-c" else 1
    return args[index] if index < len(args) else ""


def git(repo: Path, *args: str, timeout: float = GIT_TIMEOUT, check: bool = True) -> Completed:
    """Run `git -C repo args...` (with GIT_OPTIONS)."""
    subcommand = _git_subcommand(args)
    return run_command(
        ["git", *GIT_OPTIONS, "-C", str(repo), *args],
        label=f"git {subcommand}" if subcommand else "git",
        timeout=timeout,
        env=git_env(),
        check=check,
    )


def git_out(repo: Path, *args: str, timeout: float = GIT_TIMEOUT) -> str:
    """The stripped stdout of a git command that must succeed."""
    return git(repo, *args, timeout=timeout).stdout.strip()


def resolve_commit(repo: Path, ref: str) -> str | None:
    """The full commit id `ref` names in `repo`, or None if it names no commit."""
    result = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def commit_subject(repo: Path, sha: str) -> str:
    """The first line of a commit's message ('' if it cannot be read)."""
    result = git(repo, "log", "-1", "--format=%s", sha, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def commit_time(repo: Path, sha: str) -> datetime.datetime | None:
    """When a commit was committed (for a pull request merge: when it was merged), or None."""
    result = git(repo, "log", "-1", "--format=%ct", sha, check=False)
    seconds = result.stdout.strip()
    if result.returncode != 0 or not seconds.isdigit():
        return None
    return datetime.datetime.fromtimestamp(int(seconds), datetime.timezone.utc)


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """True if `ancestor` is `descendant` or one of its ancestors."""
    result = git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode in (0, 1):
        return result.returncode == 0
    raise CommandFailed(
        f"git merge-base --is-ancestor failed (status {result.returncode}): "
        f"{_output_tail(result.stderr)}"
    )


def systemctl(*args: str, check: bool = True) -> Completed:
    """Run `systemctl --user args...` (the agent never touches the system manager)."""
    return run_command(
        ["systemctl", "--user", *args],
        label="systemctl",
        timeout=SYSTEMCTL_TIMEOUT,
        env=_c_locale_env(),
        check=check,
    )


def unit_properties(unit: str) -> dict[str, str]:
    """LoadState, ActiveState, SubState and UnitFileState of a user unit."""
    out = systemctl("show", "--property=LoadState,ActiveState,SubState,UnitFileState", unit).stdout
    props: dict[str, str] = {}
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            props[key.strip()] = value.strip()
    return props


def unit_active_state(unit: str) -> str:
    """`systemctl --user is-active unit`: active, activating, inactive, failed, ...

    Raises DeployError for anything else (for example when the user manager is unreachable), so
    an unknown state is never mistaken for an idle sync service.
    """
    result = systemctl("is-active", unit, check=False)
    state = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if state not in KNOWN_ACTIVE_STATES:
        raise DeployError(
            f"cannot tell whether {unit} is running (systemctl --user is-active printed "
            f"{state!r}, status {result.returncode}: {_output_tail(result.stderr)})"
        )
    return state


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------


def _try_flock(fd: int) -> bool:
    """Take an exclusive flock on `fd` without blocking; False if another process holds it."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _open_lock_file(path: Path) -> int:
    """Open (creating it) a lock file. Python opens it non-inheritable: no child ever holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)


def raise_if_paused(paths: Paths) -> None:
    """Raise Abandoned if auto-deploy is paused (a switch checks it before changing anything)."""
    pause = read_pause(paths)
    if pause is not None:
        raise Abandoned(f"auto-deploy was paused ({pause.reason})", paused=True)


def interruptible_sleep(seconds: float, *, abandon_if_paused: Paths | None = None) -> None:
    """Sleep, but raise Interrupted on a stop request, or Abandoned once auto-deploy is paused."""
    if _stop.wait(seconds):
        raise Interrupted("stopped by a signal")
    if abandon_if_paused is not None:
        raise_if_paused(abandon_if_paused)


@contextlib.contextmanager
def held_lock(path: Path, *, wait: bool, what: str) -> Iterator[bool]:
    """Hold the flock on `path` for the block; yields False if it is busy and not `wait`.

    With `wait`, polls until the lock is free (INFO once a minute), raising Interrupted on a stop
    request.
    """
    fd = _open_lock_file(path)
    try:
        acquired = _try_flock(fd)
        if not acquired and wait:
            started = time.monotonic()
            next_progress = started
            while not acquired:
                if time.monotonic() >= next_progress:
                    log.info(
                        "waiting for %s to finish (%s is held, %s so far)",
                        what,
                        path,
                        fmt_duration(time.monotonic() - started),
                    )
                    next_progress = time.monotonic() + LOCK_PROGRESS_SECONDS
                interruptible_sleep(LOCK_POLL_SECONDS)
                acquired = _try_flock(fd)
        yield acquired
    finally:
        os.close(fd)


def _lock_holder(path: Path) -> str:
    """The pid a lock file's holder wrote into it, for messages ('unknown' if none)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "unknown"
    return text if text.isdigit() else "unknown"


def wait_for_run_lock(paths: Paths, *, abandon_on_pause: bool) -> int:
    """Take run.py's own lock (REPO/run.lock), waiting up to RUN_LOCK_TIMEOUT; the lock's fd.

    systemd already told us the sync service is idle, so a holder is a run.py started by hand;
    it is waited for like a service run. Raises Abandoned when the wait times out (the next tick
    tries again) or auto-deploy is paused, Interrupted on a stop request.
    """
    path = paths.repo / RUN_LOCK
    fd = _open_lock_file(path)
    try:
        started = time.monotonic()
        announced = False
        while not _try_flock(fd):
            waited = time.monotonic() - started
            if waited >= RUN_LOCK_TIMEOUT:
                raise Abandoned(
                    f"{path} is still held (pid {_lock_holder(path)}) after "
                    f"{fmt_duration(waited)}: a run.py started outside systemd is still running",
                    paused=False,
                )
            if not announced:
                log.info(
                    "waiting for a run.py started outside systemd to finish (%s held by pid %s)",
                    path,
                    _lock_holder(path),
                )
                announced = True
            interruptible_sleep(
                LOCK_POLL_SECONDS, abandon_if_paused=paths if abandon_on_pause else None
            )
        # The pid is only for messages: a full disk must not stop the switch.
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
    except BaseException:
        os.close(fd)
        raise
    return fd


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Failure:
    """Why a commit was not deployed."""

    reason: str
    """ci, build, tests, switch or rollback."""
    detail: str
    at: str
    """When it was first recorded (isoformat)."""
    checked_at: str | None = None
    """CI failures only: when GitHub was last asked (re-checked every CI_RECHECK_INTERVAL)."""
    seq: int = 0
    """The record's place in the order failures were recorded: State.record_failure numbers
    each record above every other record. 0 marks a record written by a version that did not
    number them (see recording_order)."""

    @property
    def final(self) -> bool:
        """True unless the failure is CI's, which a re-run on GitHub can turn green."""
        return self.reason != REASON_CI

    def to_json(self) -> dict[str, object]:
        """The state.json form."""
        data: dict[str, object] = {
            "reason": self.reason,
            "detail": self.detail,
            "at": self.at,
            "seq": self.seq,
        }
        if self.checked_at is not None:
            data["checked_at"] = self.checked_at
        return data

    @classmethod
    def from_json(cls, data: object) -> Failure | None:
        """Parse a state.json failure record; None if malformed (a malformed seq reads as 0)."""
        if not isinstance(data, dict):
            return None
        reason, detail, at = _as_str(data.get("reason")), data.get("detail"), data.get("at")
        if reason is None:
            return None
        seq = data.get("seq")
        return cls(
            reason=reason,
            detail=_as_str(detail) or "",
            at=_as_str(at) or "",
            checked_at=_as_str(data.get("checked_at")),
            seq=seq if isinstance(seq, int) and not isinstance(seq, bool) and seq > 0 else 0,
        )


def recording_order(item: tuple[str, Failure]) -> tuple[int, str, str]:
    """Sort key of a failed_shas entry that puts the oldest record first.

    The record's number decides. Records without one (0) sort before every numbered record, by
    `at` and then commit id: `at` is the only evidence of their order left, but it is written
    to the second, so records of the same second fall back to the commit id. (`at` is
    isoformat() in UTC, so its text sorts by time.)
    """
    sha, failure = item
    return failure.seq, failure.at, sha


@dataclasses.dataclass
class SwitchRecord:
    """state.json's record of a switch that is changing the checkout.

    Written just before the checkout moves, and cleared in the state.json update that records
    the outcome: the deploy, the rollback, or the failure whose switch was undone. After a
    failed undo it stays, so the tick after `resume` checks that the checkout is back where the
    switch started. Any other record that is still there when the next tick or rollback starts
    means the process that switched died half way (SIGKILL, the OOM killer, a power loss): see
    check_interrupted_switch().
    """

    target: str
    """The commit the switch moves to."""
    old_head: str
    """The commit it started from: the only safe undo point."""
    old_binary_sha256: str | None
    """SHA-256 of the generator binary before the switch (for the human who looks)."""
    mode: str
    """SwitchMode's value: fast-forward (a deploy) or reset (a rollback)."""
    started_at: str

    def to_json(self) -> dict[str, object]:
        """The state.json form."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: object) -> SwitchRecord | None:
        """Parse a state.json switch record; None if absent or malformed."""
        if not isinstance(data, dict):
            return None
        target, old_head = _as_str(data.get("target")), _as_str(data.get("old_head"))
        if not target or not old_head:
            return None
        return cls(
            target=target,
            old_head=old_head,
            old_binary_sha256=_as_str(data.get("old_binary_sha256")),
            mode=_as_str(data.get("mode")) or "",
            started_at=_as_str(data.get("started_at")) or "",
        )


@dataclasses.dataclass
class State:
    """$STATE_DIR/state.json: what is deployed and what failed.

    No file means nothing was ever deployed by the agent: the first tick goes through every step
    (build, tests, switch) even if the checkout is already at origin/main.
    """

    deployed_sha: str | None = None
    deployed_subject: str | None = None
    deployed_at: str | None = None
    binary_sha256: str | None = None
    """SHA-256 of the installed generator binary (a no-op tick checks it)."""
    previous_sha: str | None = None
    """What `rollback` switches back to: the commit deployed before this one (None after the
    first deploy, whose predecessor the agent never built or tested, and after a rollback)."""
    previous_binary_sha256: str | None = None
    rolled_back_from: str | None = None
    failed_shas: dict[str, Failure] = dataclasses.field(default_factory=dict)
    """The MAX_FAILED_SHAS most recently recorded failures, oldest first.

    state.json's keys are sorted, so this order does not survive a save: each record's `seq`
    carries it. Versions before `seq` read a state.json that has it (they ignore the field) but
    drop it when they write the file, as the agent of an older commit does after a rollback;
    this version then orders those records by `at` (recording_order).
    """
    last_error: str | None = None
    last_error_at: str | None = None
    sync_timer_restart_pending: bool = False
    """True while the sync timer could not be enabled again after a switch: every tick that
    is not paused retries until it succeeds (restart_pending_sync_timer)."""
    switch_in_progress: SwitchRecord | None = None
    """Set while a switch changes the checkout (see SwitchRecord)."""

    _STRING_FIELDS = (
        "deployed_sha",
        "deployed_subject",
        "deployed_at",
        "binary_sha256",
        "previous_sha",
        "previous_binary_sha256",
        "rolled_back_from",
        "last_error",
        "last_error_at",
    )

    def to_json(self) -> dict[str, object]:
        """The state.json form."""
        data: dict[str, object] = {"schema_version": 1}
        for name in self._STRING_FIELDS:
            data[name] = getattr(self, name)
        data["failed_shas"] = {sha: f.to_json() for sha, f in self.failed_shas.items()}
        data["sync_timer_restart_pending"] = self.sync_timer_restart_pending
        data["switch_in_progress"] = (
            self.switch_in_progress.to_json() if self.switch_in_progress else None
        )
        return data

    @classmethod
    def from_json(cls, data: dict[str, object]) -> State:
        """Parse state.json (unknown or malformed fields are ignored)."""
        state = cls()
        for name in cls._STRING_FIELDS:
            setattr(state, name, _as_str(data.get(name)))
        failed = data.get("failed_shas")
        if isinstance(failed, dict):
            records: list[tuple[str, Failure]] = []
            for sha, record in failed.items():
                failure = Failure.from_json(record)
                if isinstance(sha, str) and failure is not None:
                    records.append((sha, failure))
            state.failed_shas = dict(sorted(records, key=recording_order))
        state.sync_timer_restart_pending = data.get("sync_timer_restart_pending") is True
        state.switch_in_progress = SwitchRecord.from_json(data.get("switch_in_progress"))
        return state

    def record_failure(self, sha: str, failure: Failure) -> None:
        """Remember a failed commit as the most recent record (renumbering `failure`), and
        forget the oldest records beyond MAX_FAILED_SHAS."""
        self.failed_shas.pop(sha, None)
        newest = max((known.seq for known in self.failed_shas.values()), default=0)
        self.failed_shas[sha] = dataclasses.replace(failure, seq=newest + 1)
        while len(self.failed_shas) > MAX_FAILED_SHAS:
            oldest, _ = min(self.failed_shas.items(), key=recording_order)
            del self.failed_shas[oldest]


def load_state(path: Path) -> State:
    """Read state.json; an absent or unreadable file is an empty state (logged if unreadable)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return State()
    except OSError as exc:
        log.warning("cannot read %s (%s): treating it as empty", path, exc)
        return State()
    try:
        data = json.loads(text)
    except ValueError:
        log.warning("%s is not valid JSON: treating it as empty", path)
        return State()
    if not isinstance(data, dict):
        log.warning("%s has an unknown layout: treating it as empty", path)
        return State()
    return State.from_json(data)


@contextlib.contextmanager
def state_transaction(paths: Paths) -> Iterator[State]:
    """Read, modify and atomically write state.json under the short-lived state lock.

    Every update re-reads the file, so commands that run while a tick builds (retry, pause) and
    the tick itself never overwrite each other's changes. Nothing is written if the block raises.
    """
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    fd = _open_lock_file(paths.state_lock)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        state = load_state(paths.state_file)
        yield state
        atomic_write_json(paths.state_file, state.to_json())
    finally:
        os.close(fd)


def save_failure(
    paths: Paths, sha: str, reason: str, detail: str, *, switch_ended: bool = False
) -> Failure:
    """Record a failed commit in state.json; the record.

    With `switch_ended` (the failed switch was undone), the same update drops the switch
    record, so no crash can leave that record behind without the failure (which would make the
    next tick try the commit again).
    """
    now = isoformat(utcnow())
    failure = Failure(reason, detail, now, now if reason == REASON_CI else None)
    with state_transaction(paths) as state:
        state.record_failure(sha, failure)
        if switch_ended:
            state.switch_in_progress = None
        recorded = state.failed_shas[sha]
    return recorded


def set_sync_timer_restart_pending(paths: Paths, pending: bool) -> None:
    """Record whether the sync timer still has to be enabled again (writes only on a change).

    Never raises: it runs right after a switch, whose outcome must still be recorded. A write
    that fails (a full disk) is logged; the flag then stays as it was.
    """
    try:
        if load_state(paths.state_file).sync_timer_restart_pending == pending:
            return
        with state_transaction(paths) as state:
            state.sync_timer_restart_pending = pending
    except OSError as exc:
        log.error("cannot update %s's restart flag in %s: %s", SYNC_TIMER, paths.state_file, exc)


@dataclasses.dataclass(frozen=True)
class Pause:
    """The pause flag's content."""

    reason: str
    at: str | None


def read_pause(paths: Paths) -> Pause | None:
    """The pause record, or None if auto-deploy is not paused.

    The flag's existence is what pauses: an unreadable flag still pauses.
    """
    try:
        text = paths.pause_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return Pause(f"(pause flag unreadable: {exc})", None)
    try:
        data = json.loads(text)
    except ValueError:
        return Pause(text.strip() or "(no reason given)", None)
    if not isinstance(data, dict):
        return Pause("(no reason given)", None)
    return Pause(_as_str(data.get("reason")) or "(no reason given)", _as_str(data.get("at")))


def write_pause(paths: Paths, reason: str) -> None:
    """Pause auto-deploy (atomically; replaces an earlier reason)."""
    atomic_write_json(paths.pause_file, {"reason": reason, "at": isoformat(utcnow())})


# ---------------------------------------------------------------------------
# GitHub CI gate
# ---------------------------------------------------------------------------

_GITHUB_REMOTE_RE = re.compile(
    r"^(?:https?://(?:[^@/]+@)?github\.com/"
    r"|(?:ssh://)?git@github\.com[:/]"
    r"|git://github\.com/)"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_ARG_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")


def parse_github_slug(url: str) -> str | None:
    """OWNER/REPO from a github.com remote URL (https, ssh or scp-like form); None otherwise."""
    match = _GITHUB_REMOTE_RE.match(url.strip())
    if match is None:
        return None
    return f"{match.group('owner')}/{match.group('name')}"


def github_slug(repo: Path) -> str:
    """OWNER/REPO for the GitHub API: $COSMICSIG_DEPLOY_GITHUB_REPO, else origin's URL."""
    override = os.environ.get(ENV_GITHUB_REPO, "").strip()
    if override:
        if not _SLUG_RE.match(override):
            raise DeployError(f"{ENV_GITHUB_REPO}={override!r} is not of the form OWNER/REPO")
        return override
    url = git(repo, "config", "--get", "remote.origin.url", check=False).stdout.strip()
    slug = parse_github_slug(url)
    if slug is None:
        raise DeployError(
            f"cannot tell the GitHub repository from origin's URL {redact(url)!r}; "
            f"set {ENV_GITHUB_REPO}=OWNER/REPO"
        )
    return slug


def github_token_source() -> tuple[str, str] | None:
    """The optional API token and the variable that holds it (a token raises the rate limit; a
    public repository needs none)."""
    for variable in (ENV_GITHUB_TOKEN, ENV_GITHUB_TOKEN_FALLBACK):
        value = os.environ.get(variable, "").strip()
        if value:
            return variable, value
    return None


def github_token() -> str | None:
    """The optional API token, if one is set."""
    source = github_token_source()
    return source[1] if source else None


class CiVerdict(enum.Enum):
    """What GitHub says about a commit's `CI passed` check."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"
    """The API could not be asked (network, HTTP error, rate limit): never deploy, retry later."""


@dataclasses.dataclass(frozen=True)
class CiStatus:
    """A CI verdict and a one-line explanation."""

    verdict: CiVerdict
    detail: str
    token_error: str | None = None
    """Set when GitHub rejected the configured token; the verdict is then the answer to an
    anonymous request. A problem for `status` to show, although the tick carries on."""


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """A one-line description of an API error response, naming rate limiting explicitly."""
    headers = exc.headers
    remaining = headers.get("x-ratelimit-remaining") if headers else None
    retry_after = headers.get("retry-after") if headers else None
    if exc.code in (403, 429) and (remaining == "0" or retry_after):
        reset = headers.get("x-ratelimit-reset") if headers else None
        when = ""
        if reset and reset.isdigit():
            moment = datetime.datetime.fromtimestamp(int(reset), datetime.timezone.utc)
            when = f"; it resets at {isoformat(moment)}"
        elif retry_after:
            when = f"; retry after {retry_after}s"
        hint = "" if github_token() else f" (set {ENV_GITHUB_TOKEN} to raise the limit)"
        return f"GitHub API rate limit exceeded (HTTP {exc.code}{when}){hint}"
    return f"GitHub API returned HTTP {exc.code} {exc.reason}"


def _check_run_id(run: dict[str, object]) -> int:
    """A check run's id (newer runs have larger ids); -1 if missing."""
    value = run.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def ci_status_from_check_runs(data: object, sha: str) -> CiStatus:
    """The verdict for `sha` from a GET .../commits/{sha}/check-runs response body.

    Only `CI passed` runs of the GitHub Actions app for exactly `sha` count (another app could
    create a check run with the same name). The newest one decides.
    """
    runs = data.get("check_runs") if isinstance(data, dict) else None
    if not isinstance(runs, list):
        return CiStatus(CiVerdict.UNAVAILABLE, "unexpected GitHub API response (no check_runs)")
    candidates: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        app = run.get("app")
        app_slug = app.get("slug") if isinstance(app, dict) else None
        head_sha = run.get("head_sha", sha)
        if app_slug == CI_APP_SLUG and run.get("name") == CI_CHECK_NAME and head_sha == sha:
            candidates.append(run)
    if not candidates:
        return CiStatus(CiVerdict.PENDING, f"no {CI_CHECK_NAME!r} check run yet")
    newest = max(candidates, key=_check_run_id)
    status = _as_str(newest.get("status")) or "unknown"
    conclusion = _as_str(newest.get("conclusion"))
    url = _as_str(newest.get("html_url"))
    where = f" ({url})" if url else ""
    if status != "completed":
        return CiStatus(CiVerdict.PENDING, f"{CI_CHECK_NAME!r} is {status}{where}")
    if conclusion == "success":
        return CiStatus(CiVerdict.PASSED, f"{CI_CHECK_NAME!r} succeeded{where}")
    return CiStatus(CiVerdict.FAILED, f"{CI_CHECK_NAME!r} concluded {conclusion}{where}")


def fetch_ci_status(slug: str, sha: str) -> CiStatus:
    """Ask GitHub for the `CI passed` check of `sha` (never raises: errors are UNAVAILABLE).

    GitHub answers every request that carries an expired or revoked token with HTTP 401, even
    for a public repository; it never falls back to anonymous access. Left alone, a stale token
    would block every deploy behind a WARNING. So a 401 to a request with a token is logged as
    an ERROR naming the variable, the request is repeated once without the token, and that
    answer decides (with token_error set, so the tick records the problem).
    """
    base = os.environ.get(ENV_GITHUB_API, "").strip() or DEFAULT_GITHUB_API
    query = urllib.parse.urlencode(
        {"check_name": CI_CHECK_NAME, "filter": "latest", "per_page": "100"},
        quote_via=urllib.parse.quote,
    )
    url = f"{base.rstrip('/')}/repos/{slug}/commits/{sha}/check-runs?{query}"
    source = github_token_source()
    status, http_code = _request_ci_status(url, sha, source[1] if source else None)
    if http_code != 401 or source is None:
        return status
    problem = (
        f"{source[0]} was rejected by GitHub (HTTP 401): the token expired or was revoked; "
        "replace or remove it (docs/deployment.md, Rate limits and the optional token)"
    )
    log.error("%s; asking again without it", problem)
    status, _ = _request_ci_status(url, sha, None)
    return dataclasses.replace(status, token_error=problem)


def _request_ci_status(url: str, sha: str, token: str | None) -> tuple[CiStatus, int | None]:
    """fetch_ci_status()'s request: the status, and the HTTP error code if GitHub sent one."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        },
    )
    if token:
        # Unredirected: urllib would otherwise repeat the token to any host a redirect names.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    log.debug("GET %s%s", url, " (with the token)" if token else "")
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = _describe_http_error(exc)
        exc.close()
        return CiStatus(CiVerdict.UNAVAILABLE, detail), exc.code
    except (OSError, http.client.HTTPException) as exc:
        return CiStatus(CiVerdict.UNAVAILABLE, f"GitHub API request failed: {exc}"), None
    try:
        data: object = json.loads(body)
    except ValueError:
        return CiStatus(CiVerdict.UNAVAILABLE, "GitHub API response is not JSON"), None
    return ci_status_from_check_runs(data, sha), None


# ---------------------------------------------------------------------------
# Checkout checks
# ---------------------------------------------------------------------------


def legacy_units_present() -> list[Path]:
    """The legacy system unit files that are still installed."""
    return [LEGACY_UNIT_DIR / name for name in LEGACY_UNITS if (LEGACY_UNIT_DIR / name).exists()]


def require_no_legacy_units() -> None:
    """Refuse to act while the legacy system units could start a second sync scheduler."""
    present = legacy_units_present()
    if present:
        raise DeployError(
            "the legacy system units "
            + ", ".join(str(path) for path in present)
            + " are still installed: run `sudo ops/server/bootstrap-root.sh` first "
            "(docs/deployment.md, First-time setup)"
        )


class SwitchMode(enum.Enum):
    """How the checkout moves to the target commit."""

    FAST_FORWARD = "fast-forward"
    """Deploy: `git merge --ff-only`; the target must descend from HEAD."""
    RESET = "reset"
    """Rollback: `git reset --hard` to an ancestor of HEAD."""


def verify_checkout(repo: Path, target: str, mode: SwitchMode) -> str:
    """Refuse unless the checkout can move to `target` safely; HEAD's commit id.

    The tracked tree must be clean (so `git reset --hard` in a rollback can lose nothing), the
    checkout must be on branch main, and HEAD must be an ancestor of `target` (for a rollback,
    `target` an ancestor of HEAD). A main rewritten on GitHub fails the last check: a human must
    resolve it, the agent never discards commits.
    """
    # Not stripped: each porcelain line starts with two status columns and a space.
    status = git(repo, "status", "--porcelain", "--untracked-files=no").stdout
    if status.strip():
        changed = ", ".join(line[3:] for line in status.splitlines()[:5] if len(line) > 3)
        raise DeployError(
            f"the checkout {repo} has uncommitted changes to tracked files ({changed}); "
            "refusing to deploy until they are committed or reverted"
        )
    branch = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if branch != BRANCH:
        raise DeployError(
            f"the checkout {repo} is on {branch or 'a detached HEAD'}, not {BRANCH}; "
            f"refusing to deploy (run `git -C {repo} switch {BRANCH}`)"
        )
    head = git_out(repo, "rev-parse", "--verify", "HEAD")
    if mode is SwitchMode.FAST_FORWARD and not is_ancestor(repo, head, target):
        raise DeployError(
            f"origin/{BRANCH} ({short(target)}) does not descend from the checkout's HEAD "
            f"({short(head)}): {BRANCH} was rewritten, or the checkout has local commits. "
            "Refusing to deploy; a human must reconcile the checkout "
            "(docs/deployment.md, Troubleshooting)"
        )
    if mode is SwitchMode.RESET and not is_ancestor(repo, target, head):
        raise DeployError(
            f"{short(target)} is not an ancestor of the checkout's HEAD ({short(head)}); "
            "refusing to roll back"
        )
    return head


def missing_tree_paths(repo: Path, sha: str) -> list[str]:
    """REQUIRED_TREE_PATHS that commit `sha` lacks."""
    listed = git_out(repo, "ls-tree", "-r", "--name-only", "-z", sha, "--", *REQUIRED_TREE_PATHS)
    present = set(listed.split("\0"))
    return [path for path in REQUIRED_TREE_PATHS if path not in present]


def _lexists(path: Path) -> bool:
    """True if anything (a dangling symlink included) exists at `path`."""
    try:
        path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _blocks_directory(path: Path) -> bool:
    """True if something that is not a real directory exists at `path`."""
    return _lexists(path) and (path.is_symlink() or not path.is_dir())


def untracked_collisions(repo: Path, old: str, new: str) -> list[str]:
    """Paths the move from `old` to `new` adds that are blocked by untracked files.

    A path is blocked if something already exists there, or if one of its parent directories is
    a file or a symlink. git would overwrite an ignored file (for example .env or a ledger)
    without asking, so such a switch is refused instead.

    What `old` tracks is not in the way, though: the tracked tree is clean, so it is exactly
    `old`'s content, and git moves it aside itself. That covers a tracked file that becomes a
    directory (`foo` -> `foo/bar`: the parent `foo` is tracked) and a tracked directory that
    becomes a file (`foo/bar` -> `foo`: the existing `foo` is a tracked directory, which only
    blocks if untracked files, ignored ones included, live in it). Anything else that exists
    at an added path is untracked production state.
    """
    listed = git(
        repo, "diff", "--name-only", "--no-renames", "-z", "--diff-filter=A", old, new
    ).stdout
    added = [relative for relative in listed.split("\0") if relative]
    if not added:
        return []
    # Every path `old` tracks: its files and the directories (trees) that hold them.
    tracked = set(git(repo, "ls-tree", "-r", "-t", "--name-only", "-z", old).stdout.split("\0"))
    collisions: list[str] = []
    for relative in added:
        # parents[:-1] leaves out ".", the checkout itself. A tracked parent is git's to replace.
        parent_blocked = any(
            _blocks_directory(repo / parent)
            for parent in Path(relative).parents[:-1]
            if parent.as_posix() not in tracked
        )
        # Something exists at the path itself: untracked state, unless it is a tracked
        # directory with nothing untracked in it.
        path_blocked = _lexists(repo / relative) and (
            relative not in tracked or _untracked_files_in(repo, relative)
        )
        if parent_blocked or path_blocked:
            collisions.append(relative)
    return collisions


def _untracked_files_in(repo: Path, directory: str) -> bool:
    """True if untracked files (ignored ones included) exist in `directory` of the checkout."""
    listed = git(repo, "ls-files", "--others", "-z", "--", f":(literal){directory}").stdout
    return bool(listed.strip("\0"))


# ---------------------------------------------------------------------------
# Build and test in the staging worktree
# ---------------------------------------------------------------------------


def _common_git_dir(path: Path) -> Path | None:
    """The shared .git directory of the checkout or worktree at `path` (None if not one)."""
    result = git(path, "rev-parse", "--path-format=absolute", "--git-common-dir", check=False)
    text = result.stdout.strip()
    return Path(text).resolve() if result.returncode == 0 and text else None


def prepare_stage(paths: Paths, target: str) -> Path:
    """Check `target` out, clean, in the staging worktree (a worktree of the checkout).

    The worktree shares the checkout's object store, so nothing is downloaded twice, but has
    its own working tree: building and testing never touch production files.
    """
    repo, stage = paths.repo, paths.stage
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "prune")
    repo_git = _common_git_dir(repo)
    if (stage / ".git").is_file() and repo_git is not None and _common_git_dir(stage) == repo_git:
        git(
            stage,
            "-c",
            "advice.detachedHead=false",
            "checkout",
            "--quiet",
            "--detach",
            "--force",
            target,
        )
        git(stage, "clean", "-ffdxq")
    else:
        if stage.is_symlink() or stage.is_file():
            stage.unlink()
        elif stage.exists():
            shutil.rmtree(stage)
        # Prune again now that the broken stage is gone: a registration whose directory still
        # existed was kept by the first prune, and `worktree add` refuses a registered path.
        git(repo, "worktree", "prune")
        git(repo, "worktree", "add", "--quiet", "--detach", str(stage), target)
    head = git_out(stage, "rev-parse", "--verify", "HEAD")
    if head != target:
        raise DeployError(f"the staging worktree is at {short(head)}, not {short(target)}")
    return stage


def build_env(paths: Paths) -> dict[str, str]:
    """The environment of the build and test commands.

    CI=true makes the FFmpeg-dependent Rust tests fail instead of skipping when FFmpeg is
    missing. The persistent CARGO_TARGET_DIR keeps builds incremental across deploys.
    """
    env = dict(os.environ)
    env.update(
        CARGO_TARGET_DIR=str(paths.cargo_target_dir),
        CI="true",
        CARGO_TERM_COLOR="never",
        CARGO_TERM_PROGRESS_WHEN="never",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def verified_release(paths: Paths, sha: str) -> Path | None:
    """The binary of an intact, already built and tested release of `sha`, if there is one."""
    release = paths.releases / sha
    binary = release / BINARY_NAME
    try:
        manifest = json.loads((release / RELEASE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("sha") != sha:
        return None
    expected = manifest.get("sha256")
    if not isinstance(expected, str) or file_sha256(binary) != expected:
        return None
    return binary


def publish_release(paths: Paths, sha: str, binary: Path, *, source: str) -> Path:
    """Store a tested binary as releases/<sha>/ (atomically); the stored binary."""
    paths.releases.mkdir(parents=True, exist_ok=True)
    tmp = paths.releases / f".{sha}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    copy_file_atomic(binary, tmp / BINARY_NAME, mode=0o755)
    manifest = {
        "sha": sha,
        "sha256": file_sha256(tmp / BINARY_NAME),
        "built_at": isoformat(utcnow()),
        "source": source,
    }
    atomic_write_json(tmp / RELEASE_MANIFEST, manifest)
    final = paths.releases / sha
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final / BINARY_NAME


def build_release(paths: Paths, target: str) -> Path:
    """Build and test `target` natively; the tested binary in releases/<target>/.

    Runs cargo build, cargo test and the Python unit tests in the staging worktree (niced, with
    timeouts, output streamed to the journal). A release that was already built and tested is
    reused. Raises BuildFailed (final for the commit), CommandFailed (staging problems, retried
    next tick) or Interrupted.
    """
    existing = verified_release(paths, target)
    if existing is not None:
        log.info("reusing the built and tested release %s", existing.parent)
        return existing
    stage = prepare_stage(paths, target)
    env = build_env(paths)
    python = python_executable()

    def step(reason: str, label: str, argv: list[str], timeout: float) -> None:
        log.info("%s: running %s", short(target), label)
        started = time.monotonic()
        try:
            run_command(
                [*NICE, *argv], label=label, timeout=timeout, cwd=stage, env=env, stream=True
            )
        except CommandFailed as exc:
            raise BuildFailed(reason, str(exc)) from exc
        log.info(
            "%s: %s passed in %s", short(target), label, fmt_duration(time.monotonic() - started)
        )

    step(REASON_BUILD, "cargo build", ["cargo", "build", "--release", "--locked"], BUILD_TIMEOUT)
    staged = paths.cargo_target_dir / "release" / BINARY_NAME
    if not staged.is_file():
        raise BuildFailed(REASON_BUILD, f"cargo build did not produce {staged}")
    # Keep exactly what `cargo build --release --locked` made: `cargo test` builds the binary
    # target again with the dev-dependencies' features unified in, and the file it may leave at
    # the same path is not the artifact a release build produces.
    built = paths.data_dir / "built" / BINARY_NAME
    copy_file_atomic(staged, built, mode=0o755)
    step(REASON_TESTS, "cargo test", ["cargo", "test", "--release", "--locked"], TEST_TIMEOUT)
    step(
        REASON_TESTS,
        "python tests",
        [python, "-m", "unittest", "discover", "-s", "tests/python"],
        PYTHON_TEST_TIMEOUT,
    )
    release = publish_release(paths, target, built, source="built and tested on this host")
    built.unlink(missing_ok=True)
    # The tests leave renders in the worktree; the next deploy would clean them anyway.
    with contextlib.suppress(CommandFailed):
        git(stage, "clean", "-ffdxq")
    return release


def prune_releases(paths: Paths, keep: set[str]) -> None:
    """Keep the KEEP_RELEASES newest releases and those in `keep` (deployed, previous)."""
    if not paths.releases.is_dir():
        return
    releases: list[tuple[float, Path]] = []
    for child in paths.releases.iterdir():
        if child.name.startswith(".") and child.name.endswith(".tmp"):
            shutil.rmtree(child, ignore_errors=True)
            continue
        if not child.is_dir():
            continue
        built: datetime.datetime | None = None
        with contextlib.suppress(OSError, ValueError):
            manifest = json.loads((child / RELEASE_MANIFEST).read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                built = parse_timestamp(_as_str(manifest.get("built_at")))
        releases.append((built.timestamp() if built else child.stat().st_mtime, child))
    releases.sort(key=lambda item: item[0], reverse=True)
    for index, (_, child) in enumerate(releases):
        if index < KEEP_RELEASES or child.name in keep:
            continue
        shutil.rmtree(child, ignore_errors=True)
        log.info("pruned the old release %s", short(child.name))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def render_units(repo: Path, python: str) -> dict[str, str]:
    """The four unit files rendered from the checkout's templates (placeholders substituted)."""
    for value, what in ((str(repo), "checkout path"), (python, "Python interpreter path")):
        if not _UNIT_SAFE_PATH_RE.match(value):
            raise DeployError(
                f"the {what} {value!r} cannot be written into a unit file unquoted "
                "(allowed: an absolute path of letters, digits and . _ / + -)"
            )
    rendered: dict[str, str] = {}
    for name in UNIT_NAMES:
        template_path = repo / UNIT_TEMPLATE_DIR / name
        try:
            template = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DeployError(f"cannot read the unit template {template_path}: {exc}") from exc
        text = template.replace(PLACEHOLDER_REPO, str(repo)).replace(PLACEHOLDER_PYTHON, python)
        leftover = _LEFTOVER_PLACEHOLDER_RE.search(text)
        if leftover:
            raise DeployError(f"{template_path} has an unknown placeholder {leftover.group()}")
        rendered[name] = text
    return rendered


def install_units(paths: Paths) -> list[str]:
    """Install the rendered units into the user unit directory; the names that changed.

    Only files whose content differs are written (atomically), so an unchanged deploy leaves
    them, and systemd, alone. The caller runs daemon-reload if anything changed.
    """
    rendered = render_units(paths.repo, python_executable())
    changed: list[str] = []
    for name, text in rendered.items():
        dest = paths.unit_dir / name
        try:
            current = dest.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = None
        if current == text:
            continue
        atomic_write_bytes(dest, text.encode("utf-8"))
        changed.append(name)
    if changed:
        log.info("installed the changed units: %s", ", ".join(changed))
    return changed


def verify_units(paths: Paths) -> None:
    """Raise DeployError unless systemd accepts the installed units (call after daemon-reload).

    Every unit must have LoadState=loaded: the manager refuses a unit it cannot use (an unknown
    section, a service without ExecStart=, a setting it rejects) with bad-setting or error.
    Where systemd-analyze is installed, `systemd-analyze --user verify` must accept the rendered
    files too, which also catches Exec*= commands that do not exist. A deploy unit broken this
    way would stop auto-deploy for good (the timer could not start the next tick), so the switch
    fails instead, and its undo puts the previous units back. Nothing short of starting a unit
    catches a missing EnvironmentFile=; the smoke test runs the commands themselves.
    """
    for name in UNIT_NAMES:
        loaded = unit_properties(name).get("LoadState", "") or "unknown"
        if loaded != "loaded":
            raise DeployError(
                f"systemd does not load {name} after daemon-reload (LoadState={loaded}); "
                f"`systemctl --user status {name}` and the user journal say why"
            )
    analyze = shutil.which(SYSTEMD_ANALYZE)
    if analyze is None:
        log.debug("%s is not installed; the units are not verified further", SYSTEMD_ANALYZE)
        return
    run_command(
        # --man=no: Documentation= names no man page, and `man` need not be installed.
        [analyze, "--user", "--man=no", "verify", *(str(paths.unit_dir / n) for n in UNIT_NAMES)],
        label="systemd-analyze verify",
        timeout=SYSTEMCTL_TIMEOUT,
        env=_c_locale_env(),
    )


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SwitchReport:
    """What a switch changed, filled in as it goes so a failure can be undone exactly."""

    old_head: str
    old_binary_sha256: str | None
    new_binary_sha256: str | None = None
    checkout_moved: bool = False
    binary_touched: bool = False
    units_touched: bool = False
    units_changed: list[str] = dataclasses.field(default_factory=list)
    sync_restart_error: str | None = None
    """Set if the switch succeeded but the sync timer could not be restarted."""


def install_binary(source: Path, dest: Path, report: SwitchReport) -> bool:
    """Install the tested binary atomically; False if the bytes are already there.

    Identical bytes are left alone, so run.py's generator identity (path, size, mtime), which
    resets its backfill ledger, only changes when the binary really changes. The replaced binary
    is kept as three_body_problem.previous.
    """
    report.new_binary_sha256 = file_sha256(source)
    if report.new_binary_sha256 is None:
        raise DeployError(f"the release binary {source} is missing")
    if report.old_binary_sha256 == report.new_binary_sha256:
        log.info("the generator binary is unchanged; keeping %s", dest)
        return False
    if dest.exists():
        copy_file_atomic(dest, dest.with_name(dest.name + PREVIOUS_SUFFIX), mode=0o755)
    report.binary_touched = True
    copy_file_atomic(source, dest, mode=0o755)
    return True


def restore_binary(dest: Path, report: SwitchReport) -> None:
    """Put back the binary a failed switch replaced (from three_body_problem.previous)."""
    if report.old_binary_sha256 is None:
        dest.unlink(missing_ok=True)
        return
    previous = dest.with_name(dest.name + PREVIOUS_SUFFIX)
    if file_sha256(previous) != report.old_binary_sha256:
        raise DeployError(f"{previous} does not hold the binary that was replaced")
    copy_file_atomic(previous, dest, mode=0o755)


def smoke_test(paths: Paths) -> None:
    """The switched checkout must start: the generator, run.py and this agent."""
    repo, python = paths.repo, python_executable()
    checks = (
        ("smoke test: generator", [str(paths.installed_binary), "--version"]),
        ("smoke test: run.py", [python, str(repo / SYNC_SCRIPT), "--help"]),
        ("smoke test: deploy agent", [python, str(repo / AGENT_PATH), "--help"]),
    )
    for label, argv in checks:
        run_command(argv, label=label, timeout=SMOKE_TIMEOUT, cwd=repo)


def apply_switch(
    paths: Paths, target: str, binary: Path, mode: SwitchMode, report: SwitchReport
) -> None:
    """Steps d-g of a switch; the caller holds run.lock and undoes `report` on failure."""
    repo = paths.repo
    collisions = untracked_collisions(repo, report.old_head, target)
    if collisions:
        raise DeployError(
            "the switch would overwrite untracked files in the checkout: "
            + ", ".join(collisions[:5])
            + " (move them away, then `retry`)"
        )
    report.checkout_moved = True
    if mode is SwitchMode.FAST_FORWARD:
        git(repo, "merge", "--ff-only", "--no-overwrite-ignore", "--quiet", target)
    else:
        git(repo, "reset", "--quiet", "--hard", target)
    head = git_out(repo, "rev-parse", "--verify", "HEAD")
    if head != target:
        raise DeployError(f"after the switch HEAD is {short(head)}, not {short(target)}")
    install_binary(binary, paths.installed_binary, report)
    report.units_touched = True
    report.units_changed = install_units(paths)
    if report.units_changed:
        systemctl("daemon-reload")
        verify_units(paths)
    smoke_test(paths)


def undo_switch(paths: Paths, report: SwitchReport) -> None:
    """Undo a failed switch: code, binary and units back to what they were."""
    if report.checkout_moved:
        git(paths.repo, "reset", "--quiet", "--hard", report.old_head)
    if report.binary_touched:
        restore_binary(paths.installed_binary, report)
    if report.units_touched:
        restored = install_units(paths)
        if restored or report.units_changed:
            systemctl("daemon-reload")


def wait_for_sync_idle(paths: Paths, *, abandon_on_pause: bool) -> None:
    """Wait, without an upper bound, until the sync service is not running.

    A sync run may render for hours and must never be interrupted. INFO when the wait starts
    and every SYNC_PROGRESS_SECONDS.
    """
    started = time.monotonic()
    next_progress: float | None = None
    while True:
        state = unit_active_state(SYNC_SERVICE)
        if state not in BUSY_STATES:
            if next_progress is not None:
                log.info(
                    "the sync run finished after %s of waiting",
                    fmt_duration(time.monotonic() - started),
                )
            return
        now = time.monotonic()
        if next_progress is None:
            log.info(
                "waiting for the running sync (%s is %s) to finish before switching; "
                "it is never interrupted",
                SYNC_SERVICE,
                state,
            )
            next_progress = now + SYNC_PROGRESS_SECONDS
        elif now >= next_progress:
            log.info("still waiting for the sync run (%s so far)", fmt_duration(now - started))
            next_progress = now + SYNC_PROGRESS_SECONDS
        interruptible_sleep(
            SYNC_POLL_SECONDS, abandon_if_paused=paths if abandon_on_pause else None
        )


def _sync_timer_wanted() -> tuple[bool, bool]:
    """Whether the sync timer is (enabled, active) before a switch stops it."""
    props = unit_properties(SYNC_TIMER)
    enabled = props.get("UnitFileState", "") in ("enabled", "enabled-runtime")
    active = props.get("ActiveState", "") in ("active", "activating")
    return enabled, active


def _stop_sync_timer() -> None:
    """Stop the sync timer so no sync run starts during the switch (absent is fine)."""
    result = systemctl("stop", SYNC_TIMER, check=False)
    if result.returncode != 0 and "not loaded" not in result.stderr:
        raise DeployError(
            f"cannot stop {SYNC_TIMER}: {_output_tail(result.stderr) or result.returncode}"
        )


def _restore_sync_timer(paths: Paths, enabled: bool, active: bool) -> None:
    """Put the sync timer back as it was (logs, never raises: the caller is failing already).

    An enabled timer that cannot be enabled again is flagged in state.json, and every tick
    retries it (restart_pending_sync_timer). A timer that was only started, not enabled (which
    only an operator does, by hand), is started again once, best effort.
    """
    try:
        if enabled:
            systemctl("enable", "--now", SYNC_TIMER)
        elif active:
            systemctl("start", SYNC_TIMER)
    except DeployError as exc:
        log.error("could not restart %s: %s", SYNC_TIMER, exc)
        if enabled:
            set_sync_timer_restart_pending(paths, True)
    else:
        if enabled:
            set_sync_timer_restart_pending(paths, False)


def _disable_sync_timer() -> str:
    """Stop and disable the sync timer, best effort; a clause saying how that went.

    For a checkout that needs a human (a failed undo, an interrupted switch): a timer that is
    merely stopped would start again at the next boot (it is still wanted by timers.target),
    and its first run would use that checkout.
    """
    try:
        result = systemctl("disable", "--now", SYNC_TIMER, check=False)
        failure = (
            None
            if result.returncode == 0
            else _output_tail(result.stderr) or f"status {result.returncode}"
        )
    except DeployError as exc:
        failure = str(exc)
    if failure is None:
        return (
            f"{SYNC_TIMER} is stopped and disabled, so no sync runs on this checkout, not even "
            "after a reboot"
        )
    log.error("cannot disable %s: %s", SYNC_TIMER, failure)
    return (
        f"{SYNC_TIMER} could not be disabled ({failure}): run "
        f"`systemctl --user disable --now {SYNC_TIMER}` so that no sync runs on this checkout"
    )


def restart_pending_sync_timer(paths: Paths) -> str | None:
    """Retry enabling the sync timer after a switch that could not; an error while it fails.

    Without this, a failed `enable --now` after a switch (a D-Bus timeout, say) would be logged
    once and forgotten: the next tick has nothing to deploy, and the sync would stay down. Only
    ticks that are not paused get here, so an operator's pause-then-disable stop still holds.
    """
    if not load_state(paths.state_file).sync_timer_restart_pending:
        return None
    try:
        systemctl("enable", "--now", SYNC_TIMER)
    except DeployError as exc:
        message = (
            f"{SYNC_TIMER} is still not running after the last switch (retrying every tick): {exc}"
        )
        log.error("%s", message)
        return message
    set_sync_timer_restart_pending(paths, False)
    log.info("enabled and started %s, which the last switch could not restart", SYNC_TIMER)
    return None


def record_switch_start(paths: Paths, target: str, mode: SwitchMode, report: SwitchReport) -> None:
    """Record in state.json that a switch is about to change the checkout (see SwitchRecord).

    Raises DeployError (nothing changed yet) if state.json cannot be written.
    """
    record = SwitchRecord(
        target=target,
        old_head=report.old_head,
        old_binary_sha256=report.old_binary_sha256,
        mode=mode.value,
        started_at=isoformat(utcnow()),
    )
    try:
        with state_transaction(paths) as state:
            state.switch_in_progress = record
    except OSError as exc:
        raise DeployError(f"cannot record the switch in {paths.state_file}: {exc}") from exc


def check_interrupted_switch(paths: Paths) -> str | None:
    """Stop for a human if an earlier switch died half way; the error message, or None.

    A SwitchRecord left in state.json means the process that switched died before it recorded
    the outcome, or that undoing the switch failed (that path keeps the record on purpose). If
    the checkout is (again) at the commit that switch started from, the record is dropped: at
    most the binary or units may differ, and the next deploy of origin/main puts them right.
    Otherwise the code may have moved without the binary or the units, and the current HEAD is
    no safe undo point (a failed retry would reset to it and restart the sync on code that was
    never deployed). So auto-deploy is paused and the sync timer disabled, as after a failed
    undo, and a human puts the checkout back. A pending restart of the sync timer is dropped
    too: no tick may enable it again on this checkout, the human does after the repair.
    """
    record = load_state(paths.state_file).switch_in_progress
    if record is None:
        return None
    head = resolve_commit(paths.repo, "HEAD")
    if head == record.old_head:
        with state_transaction(paths) as state:
            state.switch_in_progress = None
        log.warning(
            "the switch from %s to %s (started %s) did not finish cleanly: it was interrupted, "
            "or undoing it failed. The checkout is at %s, where that switch started, so "
            "carrying on",
            short(record.old_head),
            short(record.target),
            record.started_at or "?",
            short(head),
        )
        return None
    problem = (
        f"a switch from {short(record.old_head)} to {short(record.target)} (started "
        f"{record.started_at or '?'}) was interrupted, or its undo failed, and left the checkout "
        f"at {short(head)}, not at {short(record.old_head)}: its code, binary and units may be "
        "half switched"
    )
    try:
        write_pause(paths, f"automatic: {problem}")
    except OSError as exc:
        log.error("cannot pause auto-deploy: %s", exc)
    timer = _disable_sync_timer()
    set_sync_timer_restart_pending(paths, False)
    return (
        f"{problem}. Auto-deploy is paused, and {timer}. Put the checkout back on "
        f"{short(record.old_head)} (docs/deployment.md, Troubleshooting), then `resume`"
    )


def switch(
    paths: Paths, target: str, binary: Path, mode: SwitchMode, *, abandon_on_pause: bool
) -> SwitchReport:
    """Switch the checkout and the binary to `target` between sync runs (steps a-i).

    Stops the sync timer, waits for a running sync, takes run.lock, applies the switch and
    smoke-tests it; any failure undoes everything. Afterwards the timer is enabled and a sync
    run started (after a success), or the timer is put back as it was (after a failure); a timer
    that cannot be enabled again is flagged in state.json and retried by every tick. If the
    undo fails too, auto-deploy is paused and the timer is stopped and disabled: the checkout is
    in an unknown state that a human must look at.

    Raises Abandoned or Interrupted (nothing changed), DeployError (a precondition failed after
    the wait; nothing changed), SwitchFailed (undone) or RollbackFailed.
    """
    timer_enabled, timer_active = _sync_timer_wanted()
    _stop_sync_timer()
    try:
        report = _switch_while_timer_stopped(
            paths, target, binary, mode, abandon_on_pause=abandon_on_pause
        )
    except RollbackFailed:
        raise
    except BaseException:
        _restore_sync_timer(paths, timer_enabled, timer_active)
        raise
    try:
        systemctl("enable", "--now", SYNC_TIMER)
    except DeployError as exc:
        report.sync_restart_error = str(exc)
        set_sync_timer_restart_pending(paths, True)
        return report
    set_sync_timer_restart_pending(paths, False)
    try:
        systemctl("start", "--no-block", SYNC_SERVICE)
    except DeployError as exc:
        # The timer runs, and starts the sync on its own: nothing to retry.
        report.sync_restart_error = str(exc)
    return report


def _switch_while_timer_stopped(
    paths: Paths, target: str, binary: Path, mode: SwitchMode, *, abandon_on_pause: bool
) -> SwitchReport:
    """switch() between stopping and restarting the sync timer."""
    wait_for_sync_idle(paths, abandon_on_pause=abandon_on_pause)
    lock_fd = wait_for_run_lock(paths, abandon_on_pause=abandon_on_pause)
    try:
        # The last moment to give up with nothing changed: a pause (for example by `rollback`)
        # that came while the tick built or found the sync idle at once must still stop it.
        if abandon_on_pause:
            raise_if_paused(paths)
        # The wait may have taken hours: check the checkout again now that nothing can run.
        old_head = verify_checkout(paths.repo, target, mode)
        report = SwitchReport(
            old_head=old_head, old_binary_sha256=file_sha256(paths.installed_binary)
        )
        record_switch_start(paths, target, mode, report)
        try:
            apply_switch(paths, target, binary, mode, report)
        except Exception as exc:  # anything, a bug included: never leave a half-switched checkout
            log.error("the switch to %s failed: %s; rolling back", short(target), exc)
            try:
                undo_switch(paths, report)
            except Exception as undo_exc:
                reason = (
                    f"automatic: undoing the failed switch to {short(target)} failed "
                    f"({undo_exc}); the checkout needs a human"
                )
                try:
                    write_pause(paths, reason)
                except OSError as pause_exc:  # a full disk, say: still end as RollbackFailed
                    log.error("cannot pause auto-deploy: %s", pause_exc)
                # Not merely stopped: a reboot would start an enabled timer, and the sync with
                # it, on this checkout. A human enables it again (docs/deployment.md).
                timer = _disable_sync_timer()
                set_sync_timer_restart_pending(paths, False)
                raise RollbackFailed(
                    f"{exc}; undoing it failed too: {undo_exc}. The checkout needs a human: "
                    f"auto-deploy is paused, and {timer}. Inspect {paths.repo} and put it back "
                    f"on {short(old_head)} (docs/deployment.md, Troubleshooting)"
                ) from undo_exc
            log.info("rolled back to %s: code, binary and units restored", short(old_head))
            raise SwitchFailed(str(exc)) from exc
        return report
    finally:
        os.close(lock_fd)


# ---------------------------------------------------------------------------
# run: one deployment tick
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TickResult:
    """A tick's exit status and the error to record in state.json (None clears it)."""

    status: int
    error: str | None = None


def ci_recheck_due(failure: Failure, now: datetime.datetime) -> bool:
    """True once a CI failure's last check is CI_RECHECK_INTERVAL old.

    A last check in the future (the clock stepped back since) is due too: waiting for the clock
    to catch up could stall the re-checks for as long as it stepped back.
    """
    checked = parse_timestamp(failure.checked_at)
    return checked is None or checked > now or now - checked >= CI_RECHECK_INTERVAL


def fetch_origin(repo: Path) -> None:
    """Update origin/main from GitHub (raises CommandFailed)."""
    git(repo, "fetch", "--quiet", "--prune", "origin", FETCH_REFSPEC, timeout=FETCH_TIMEOUT)


def deploy_tick(paths: Paths) -> TickResult:
    """One tick (docs/deployment.md, How a deploy works: step 2 after the pause check, to 9).

    It first deals with what an earlier run may have left behind: a switch that died half way
    (check_interrupted_switch pauses auto-deploy for a human) and a sync timer that a switch
    could not enable again (retried here until it works). Problems that do not end the tick,
    such as that timer or a rejected GitHub token, go into the tick's error, so state.json's
    last_error (and `status`) keeps showing them until they are gone.
    """
    require_no_legacy_units()
    interrupted = check_interrupted_switch(paths)
    if interrupted is not None:
        log.error("%s", interrupted)
        return TickResult(1, interrupted)
    restart_error = restart_pending_sync_timer(paths)
    problems: list[str] = []
    result = _deploy_origin_main(paths, problems)
    status = result.status
    # A successful switch in this tick enables the timer itself and clears the flag.
    if restart_error is not None and load_state(paths.state_file).sync_timer_restart_pending:
        problems.insert(0, restart_error)
        status = 1
    if not problems:
        return result
    return TickResult(status, "; ".join(filter(None, (result.error, *problems))))


def _deploy_origin_main(paths: Paths, problems: list[str]) -> TickResult:
    """deploy_tick() after its housekeeping: deploy origin/main if it is new and passed CI.

    Problems that must be recorded although the tick goes on are appended to `problems`.
    """
    started = time.monotonic()
    repo = paths.repo
    try:
        fetch_origin(repo)
    except CommandFailed as exc:
        message = f"cannot fetch origin/{BRANCH}: {exc}"
        log.warning("%s (retrying next tick)", message)
        return TickResult(0, message)
    target = resolve_commit(repo, ORIGIN_MAIN)
    if target is None:
        raise DeployError(f"{ORIGIN_MAIN} does not exist in {repo} after fetching")

    state = load_state(paths.state_file)
    head = resolve_commit(repo, "HEAD")
    binary_sha256 = file_sha256(paths.installed_binary)
    if (
        state.deployed_sha == target
        and head == target
        and state.binary_sha256 is not None
        and binary_sha256 == state.binary_sha256
    ):
        log.debug("%s is deployed; nothing to do", short(target))
        return TickResult(0)

    failure = state.failed_shas.get(target)
    if failure is not None and failure.final:
        why = (
            "it was rolled back"
            if failure.reason == REASON_ROLLBACK
            else f"its {failure.reason} failed"
        )
        log.info(
            "origin/%s %s is not deployed: %s (%s). Push a fix, or run `retry` to try it again",
            BRANCH,
            short(target),
            why,
            failure.detail,
        )
        return TickResult(0)
    now = utcnow()
    if failure is not None and not ci_recheck_due(failure, now):
        log.info(
            "origin/%s %s failed CI (%s); asking GitHub again after %s",
            BRANCH,
            short(target),
            failure.detail,
            failure.checked_at,
        )
        return TickResult(0)

    verify_checkout(repo, target, SwitchMode.FAST_FORWARD)
    missing = missing_tree_paths(repo, target)
    if missing:
        detail = f"the commit lacks {', '.join(missing)}, which auto-deploy needs"
        save_failure(paths, target, REASON_SWITCH, detail)
        log.error("not deploying %s: %s", short(target), detail)
        return TickResult(1, detail)

    ci = fetch_ci_status(github_slug(repo), target)
    if ci.token_error is not None:
        problems.append(ci.token_error)
    if ci.verdict is CiVerdict.UNAVAILABLE:
        log.warning("cannot check CI for %s: %s (retrying next tick)", short(target), ci.detail)
        return TickResult(0, ci.detail)
    if ci.verdict is CiVerdict.FAILED:
        if failure is None:
            save_failure(paths, target, REASON_CI, ci.detail)
            log.error(
                "not deploying %s: %s; asking again every %d minutes in case CI is re-run",
                short(target),
                ci.detail,
                CI_RECHECK_INTERVAL.seconds // 60,
            )
        else:
            with state_transaction(paths) as current:
                current.record_failure(
                    target,
                    dataclasses.replace(failure, detail=ci.detail, checked_at=isoformat(now)),
                )
            log.info("%s still fails CI: %s", short(target), ci.detail)
        return TickResult(0)
    if failure is not None:
        with state_transaction(paths) as current:
            current.failed_shas.pop(target, None)
    if ci.verdict is CiVerdict.PENDING:
        committed = commit_time(repo, target)
        if committed is not None and now - committed >= CI_STALL_WARNING:
            minutes = int((now - committed).total_seconds() // 60)
            stalled = (
                f"still waiting for CI on {short(target)} {minutes} minutes after it was "
                f"committed: {ci.detail}. If the commit has no Actions run (its message contained "
                f"`[skip ci]` or similar), start one with `gh workflow run ci.yml --ref {BRANCH}`"
            )
            log.warning("%s", stalled)
            return TickResult(0, stalled)
        log.info("waiting for CI on %s: %s", short(target), ci.detail)
        return TickResult(0)

    subject = commit_subject(repo, target)
    log.info("deploying %s (%s): %s", short(target), subject, ci.detail)
    try:
        release = build_release(paths, target)
    except BuildFailed as exc:
        save_failure(paths, target, exc.stage, exc.detail)
        log.error(
            "not deploying %s: %s (final for this commit until `retry` or a new commit)",
            short(target),
            exc,
        )
        return TickResult(1, str(exc))

    try:
        # A pause (or a `rollback`, which pauses first) during the build: switch nothing. The
        # tested release is kept, so the tick after `resume` switches without rebuilding.
        raise_if_paused(paths)
        report = switch(paths, target, release, SwitchMode.FAST_FORWARD, abandon_on_pause=True)
    except Abandoned as exc:
        message = f"switch to {short(target)} abandoned: {exc}; nothing changed"
        if exc.paused:
            log.info("%s", message)
            return TickResult(0)
        log.warning("%s (retrying next tick)", message)
        return TickResult(0, message)
    except SwitchFailed as exc:
        save_failure(paths, target, REASON_SWITCH, str(exc), switch_ended=True)
        log.error(
            "not deploying %s: the switch failed and was rolled back: %s (final for this "
            "commit until `retry` or a new commit)",
            short(target),
            exc,
        )
        return TickResult(1, f"switch to {short(target)} failed: {exc}")
    except RollbackFailed as exc:
        log.critical("%s", exc)
        # Final like any failed switch, so `resume` after the repair does not retry it blindly.
        # The switch record stays: the next tick checks that the checkout is back where the
        # switch started. (The disk may be full: this is best effort.)
        try:
            save_failure(paths, target, REASON_SWITCH, str(exc))
        except OSError as save_exc:
            log.error("cannot record the failed switch to %s: %s", short(target), save_exc)
        return TickResult(1, str(exc))

    with state_transaction(paths) as current:
        old = current.deployed_sha
        # Only a commit the agent deployed becomes the rollback target. Before the first deploy
        # the checkout was built and switched by hand: its commit (whose CI may even have
        # failed) and its binary were never gated, so the first deploy records none.
        if old is not None and old != target:
            current.previous_sha = old
            current.previous_binary_sha256 = current.binary_sha256
        current.deployed_sha = target
        current.deployed_subject = subject
        current.deployed_at = isoformat(utcnow())
        current.binary_sha256 = report.new_binary_sha256
        current.rolled_back_from = None
        current.failed_shas.pop(target, None)
        current.switch_in_progress = None
        keep = {sha for sha in (current.deployed_sha, current.previous_sha) if sha}
    prune_releases(paths, keep)
    log.info(
        "deployed %s (%s) in %s", short(target), subject, fmt_duration(time.monotonic() - started)
    )
    if report.sync_restart_error:
        message = (
            f"deployed {short(target)}, but the sync did not restart: {report.sync_restart_error}"
        )
        log.error("%s", message)
        return TickResult(1, message)
    return TickResult(0)


def cmd_run(paths: Paths) -> int:
    """`run`: one deployment tick (what the timer starts)."""
    with held_lock(paths.deploy_lock, wait=False, what="the running deploy tick") as held:
        if not held:
            log.info("another deploy command holds %s; nothing to do", paths.deploy_lock)
            return 0
        pause = read_pause(paths)
        if pause is not None:
            log.info("auto-deploy is paused since %s: %s", pause.at or "?", pause.reason)
            return 0
        try:
            result = deploy_tick(paths)
        except Interrupted:
            log.warning("stopped by a signal before switching; nothing changed")
            return 1
        except DeployError as exc:
            log.error("%s", exc)
            result = TickResult(1, str(exc))
        except Exception as exc:  # a bug: record it and keep the timer alive for the next tick
            log.exception("unexpected error during the deploy tick")
            result = TickResult(1, f"unexpected error: {exc!r}")
        _record_tick_error(paths, result.error)
        return result.status


def _record_tick_error(paths: Paths, error: str | None) -> None:
    """Keep state.json's last_error in step with the tick (writes only when it changes)."""
    state = load_state(paths.state_file)
    if state.last_error == error:
        return
    with state_transaction(paths) as current:
        current.last_error = error
        current.last_error_at = isoformat(utcnow()) if error else None


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def rollback_binary(paths: Paths, state: State) -> Path:
    """The binary to roll back to: the previous release, or the kept .previous binary.

    A .previous binary is copied into releases/<previous_sha>/ first, because the switch
    overwrites .previous with the binary it replaces.
    """
    previous = state.previous_sha
    assert previous is not None
    release = verified_release(paths, previous)
    if release is not None:
        return release
    kept = paths.installed_binary.with_name(BINARY_NAME + PREVIOUS_SUFFIX)
    if state.previous_binary_sha256 and file_sha256(kept) == state.previous_binary_sha256:
        return publish_release(paths, previous, kept, source=f"restored from {kept}")
    raise DeployError(
        f"no binary for {short(previous)}: releases/{previous} is gone and {kept} does not "
        "hold it; roll back by hand (docs/deployment.md, Rollback)"
    )


def cmd_rollback(paths: Paths) -> int:
    """`rollback`: pause auto-deploy, then switch back to the previous deployment.

    The pause comes first: it is what the operator wants even if the rollback itself cannot
    happen, and a tick that is waiting for a sync run gives up its switch at once and frees the
    deploy lock. The rollback then uses the switch procedure of a deploy (it waits for a running
    sync, never interrupts one) with `git reset --hard` instead of a fast-forward. It rolls back
    the commit that was deployed when it was run, or nothing (see _rollback_locked).
    """
    seen = load_state(paths.state_file).deployed_sha
    write_pause(paths, f"rollback of {short(seen)} requested")
    log.info("auto-deploy paused; run `resume` once main is fixed")
    try:
        with held_lock(paths.deploy_lock, wait=True, what="the running deploy tick"):
            return _rollback_locked(paths, seen)
    except Interrupted:
        log.warning("rollback stopped by a signal before switching; nothing changed")
        return 1


def _rollback_locked(paths: Paths, seen: str | None) -> int:
    """cmd_rollback() under the deploy lock; the exit status.

    `seen` is the commit that was deployed when the operator ran `rollback`. A tick past its
    last pause check can still finish a switch while the rollback waits for the deploy lock;
    rolling back from that newer commit would land on the one the operator wanted to leave. So
    a changed deployed commit ends the rollback with an ERROR (auto-deploy stays paused).
    """
    try:
        interrupted = check_interrupted_switch(paths)
    except DeployError as exc:
        log.error("%s", exc)
        return 1
    if interrupted is not None:
        log.error("not rolling back: %s", interrupted)
        return 1
    state = load_state(paths.state_file)
    bad, previous = state.deployed_sha, state.previous_sha
    if bad != seen:
        log.error(
            "not rolling back: while this command waited for the deploy lock, the deployed "
            "commit changed from %s to %s (a deploy tick or another rollback finished first). "
            "Auto-deploy stays paused: check `status`, then `resume`, or run `rollback` again "
            "to switch back to %s",
            short(seen),
            short(bad),
            short(previous),
        )
        return 1
    if bad is None or previous is None:
        log.error(
            "nothing to roll back to: %s",
            "nothing is deployed" if bad is None else "no previous deployment is recorded",
        )
        return 1
    started = time.monotonic()
    try:
        require_no_legacy_units()
        missing = missing_tree_paths(paths.repo, previous)
        if missing:
            raise DeployError(
                f"{short(previous)} lacks {', '.join(missing)}: it predates auto-deploy and "
                "cannot be rolled back to automatically"
            )
        binary = rollback_binary(paths, state)
        verify_checkout(paths.repo, previous, SwitchMode.RESET)
        report = switch(paths, previous, binary, SwitchMode.RESET, abandon_on_pause=False)
    except SwitchFailed as exc:
        # Undone: the checkout is back where the switch started.
        with state_transaction(paths) as current:
            current.switch_in_progress = None
        log.error("rollback to %s failed: %s", short(previous), exc)
        return 1
    except (DeployError, Abandoned) as exc:
        log.error("rollback to %s failed: %s", short(previous), exc)
        return 1
    except RollbackFailed as exc:
        log.critical("%s", exc)
        return 1
    subject = commit_subject(paths.repo, previous)
    with state_transaction(paths) as current:
        current.deployed_sha = previous
        current.deployed_subject = subject
        current.deployed_at = isoformat(utcnow())
        current.binary_sha256 = report.new_binary_sha256
        current.previous_sha = None
        current.previous_binary_sha256 = None
        current.rolled_back_from = bad
        current.record_failure(
            bad, Failure(REASON_ROLLBACK, "rolled back by the operator", isoformat(utcnow()))
        )
        current.switch_in_progress = None
        current.last_error = None
        current.last_error_at = None
    log.info(
        "rolled back from %s to %s (%s) in %s; auto-deploy stays paused until `resume`",
        short(bad),
        short(previous),
        subject,
        fmt_duration(time.monotonic() - started),
    )
    if report.sync_restart_error:
        log.error("the sync did not restart: %s", report.sync_restart_error)
        return 1
    return 0


# ---------------------------------------------------------------------------
# install, pause, resume, retry
# ---------------------------------------------------------------------------


def check_linger() -> None:
    """Warn if lingering is off: the user timers would stop whenever the user logs out."""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:  # a uid without a passwd entry; loginctl takes the uid as well
        user = str(os.getuid())
    try:
        result = run_command(
            ["loginctl", "show-user", user, "--property=Linger", "--value"],
            label="loginctl",
            timeout=30,
            env=_c_locale_env(),
            check=False,
        )
    except CommandFailed as exc:
        log.debug("cannot check lingering: %s", exc)
        return
    if result.stdout.strip() == "yes":
        log.info("lingering is enabled for %s: the timers run without a login session", user)
    else:
        log.warning(
            "lingering is not enabled for %s: the timers stop when %s logs out. Run "
            "`sudo ops/server/bootstrap-root.sh %s` (or `sudo loginctl enable-linger %s`)",
            user,
            user,
            user,
            user,
        )


def cmd_install(paths: Paths) -> int:
    """`install`: install the units and start the deploy timer (idempotent, unprivileged).

    The sync timer is not enabled here: the first successful deploy enables it, so the sync
    never runs a binary that was not built and tested by the agent. Units that systemd would not
    run (verify_units) are refused before the deploy timer is enabled; the first deploy only
    checks units again if it changes them.
    """
    try:
        require_no_legacy_units()
        if _common_git_dir(paths.repo) is None:
            raise DeployError(f"{paths.repo} is not a git checkout")
        paths.state_dir.mkdir(parents=True, exist_ok=True)
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        with held_lock(paths.deploy_lock, wait=True, what="the running deploy tick"):
            install_units(paths)
            systemctl("daemon-reload")
            verify_units(paths)
            systemctl("enable", "--now", DEPLOY_TIMER)
    except DeployError as exc:
        log.error("%s", exc)
        return 1
    except Interrupted:
        log.warning("install stopped by a signal")
        return 1
    check_linger()
    if not (paths.repo / ".env").is_file():
        log.warning(
            "%s is missing: %s cannot start without it (it names the asset host; see .env.example)",
            paths.repo / ".env",
            SYNC_SERVICE,
        )
    log.info(
        "installed: %s runs every 2 minutes (its first tick starts now). Follow it with "
        "`journalctl --user -u cosmicsig-deploy -u cosmicsig-sync -f`; the first successful "
        "deploy enables %s",
        DEPLOY_TIMER,
        SYNC_TIMER,
    )
    return 0


def cmd_pause(paths: Paths, reason: str | None) -> int:
    """`pause`: no tick deploys until `resume` (a tick waiting for a sync run gives up)."""
    write_pause(paths, reason or "paused by the operator")
    log.info("auto-deploy paused: %s", reason or "paused by the operator")
    return 0


def cmd_resume(paths: Paths) -> int:
    """`resume`: undo `pause` (and the pause a rollback leaves)."""
    pause = read_pause(paths)
    paths.pause_file.unlink(missing_ok=True)
    if pause is None:
        log.info("auto-deploy was not paused")
    else:
        log.info("auto-deploy resumed (it was paused: %s)", pause.reason)
    return 0


def cmd_retry(paths: Paths, sha: str | None) -> int:
    """`retry [SHA]`: forget a recorded failure so the next tick tries the commit again."""
    if sha is not None and not _SHA_ARG_RE.match(sha):
        log.error("%r is not a commit id (4 to 40 hex digits)", sha)
        return 1
    ref = sha or ORIGIN_MAIN
    try:
        commit = resolve_commit(paths.repo, ref)
    except CommandFailed as exc:
        log.error("%s", exc)
        return 1
    if commit is None:
        state = load_state(paths.state_file)
        matches = [known for known in state.failed_shas if sha and known.startswith(sha)]
        if len(matches) != 1:
            log.error("%s does not name a commit in %s", ref, paths.repo)
            return 1
        commit = matches[0]
    with state_transaction(paths) as state:
        failure = state.failed_shas.pop(commit, None)
    if failure is None:
        log.info("%s has no recorded failure; nothing to retry", short(commit))
    else:
        log.info(
            "forgot the %s failure of %s; the next tick tries it again",
            failure.reason,
            short(commit),
        )
    pause = read_pause(paths)
    if pause is not None:
        log.info("note: auto-deploy is paused (%s); run `resume` too", pause.reason)
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def collect_status(paths: Paths) -> dict[str, object]:
    """Everything `status` shows, as JSON-ready data (never raises for a broken setup)."""
    state = load_state(paths.state_file)
    repo = paths.repo
    report: dict[str, object] = {"checkout": str(repo)}

    def subject(sha: str | None) -> str | None:
        return commit_subject(repo, sha) if sha else None

    try:
        head = resolve_commit(repo, "HEAD")
        branch_result = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
        origin_main = resolve_commit(repo, ORIGIN_MAIN)
        report["head"] = head
        report["branch"] = branch_result.stdout.strip() or None
        report["origin_main"] = {"sha": origin_main, "subject": subject(origin_main)}
    except CommandFailed as exc:
        report["git_error"] = str(exc)
        origin_main = None
    binary_sha256 = file_sha256(paths.installed_binary)
    report["deployed"] = {
        "sha": state.deployed_sha,
        "subject": state.deployed_subject,
        "at": state.deployed_at,
        "binary_sha256": state.binary_sha256,
    }
    report["binary"] = {
        "path": str(paths.installed_binary),
        "sha256": binary_sha256,
        "matches_deployed": binary_sha256 is not None and binary_sha256 == state.binary_sha256,
    }
    report["previous_sha"] = state.previous_sha
    report["rolled_back_from"] = state.rolled_back_from
    pending: dict[str, object] | None = None
    if origin_main and origin_main != state.deployed_sha:
        failure = state.failed_shas.get(origin_main)
        pending = {
            "sha": origin_main,
            "failure": failure.to_json() if failure else None,
        }
    report["pending"] = pending
    pause = read_pause(paths)
    report["paused"] = dataclasses.asdict(pause) if pause else None
    report["last_error"] = (
        {"message": state.last_error, "at": state.last_error_at} if state.last_error else None
    )
    report["failed_shas"] = {sha: f.to_json() for sha, f in state.failed_shas.items()}
    report["sync_timer_restart_pending"] = state.sync_timer_restart_pending
    report["switch_in_progress"] = (
        state.switch_in_progress.to_json() if state.switch_in_progress else None
    )
    units: dict[str, object] = {}
    for unit in UNIT_NAMES:
        try:
            units[unit] = unit_properties(unit)
        except DeployError as exc:
            units[unit] = {"error": str(exc)}
    report["units"] = units
    report["legacy_units"] = [str(path) for path in legacy_units_present()]
    return report


def _describe_time(text: object) -> str:
    """A stored timestamp with its age, e.g. '2026-09-29T02:11:00+00:00 (3h04m00s ago)'."""
    moment = parse_timestamp(text if isinstance(text, str) else None)
    if moment is None:
        return "-"
    age = (utcnow() - moment).total_seconds()
    return f"{isoformat(moment)} ({fmt_duration(max(age, 0))} ago)"


def format_status(report: Mapping[str, object]) -> str:
    """The human-readable `status` text."""
    lines: list[str] = []

    def row(name: str, value: str) -> None:
        lines.append(f"{name:<13}{value}")

    def get(key: str) -> dict[str, object]:
        value = report.get(key)
        return value if isinstance(value, dict) else {}

    deployed, binary, origin = get("deployed"), get("binary"), get("origin_main")
    row(
        "Checkout",
        f"{report.get('checkout')} (on {report.get('branch') or 'detached HEAD'} "
        f"at {short(_as_str(report.get('head')))})",
    )
    if report.get("git_error"):
        row("Git error", str(report.get("git_error")))
    if deployed.get("sha"):
        row("Deployed", f"{short(_as_str(deployed.get('sha')))} {deployed.get('subject') or ''}")
        row("", f"at {_describe_time(deployed.get('at'))}")
    else:
        row("Deployed", "nothing yet (the first successful tick deploys origin/main)")
    if binary.get("sha256") is None:
        row("Binary", f"MISSING: {binary.get('path')}")
    elif binary.get("matches_deployed"):
        row("Binary", "matches the deployed release")
    else:
        row("Binary", "DIFFERS from the deployed release (the next tick reinstalls it)")
    row("origin/main", f"{short(_as_str(origin.get('sha')))} {origin.get('subject') or ''}")
    pending = report.get("pending")
    if isinstance(pending, dict):
        failure = pending.get("failure")
        if isinstance(failure, dict):
            if failure.get("reason") == REASON_ROLLBACK:
                why = "rolled back by the operator"
            else:
                why = f"{failure.get('reason')} failed ({failure.get('detail')})"
            row("Pending", f"{short(_as_str(pending.get('sha')))} NOT deployed: {why}")
        else:
            row(
                "Pending",
                f"{short(_as_str(pending.get('sha')))} (waiting for CI, the build, "
                "or the running sync)",
            )
    else:
        row("Pending", "none: origin/main is deployed")
    row("Previous", short(_as_str(report.get("previous_sha"))))
    if report.get("rolled_back_from"):
        row("Rolled back", f"from {short(_as_str(report.get('rolled_back_from')))}")
    paused = report.get("paused")
    if isinstance(paused, dict):
        row("Auto-deploy", f"PAUSED since {paused.get('at') or '?'}: {paused.get('reason')}")
    else:
        row("Auto-deploy", "active")
    switching = report.get("switch_in_progress")
    if isinstance(switching, dict):
        row(
            "Switch",
            f"from {short(_as_str(switching.get('old_head')))} to "
            f"{short(_as_str(switching.get('target')))} since {switching.get('started_at')} "
            "(in progress, or interrupted: the next tick checks)",
        )
    if report.get("sync_timer_restart_pending"):
        row(
            "Sync timer",
            f"NOT restarted after the last switch; every tick that is not paused retries "
            f"`systemctl --user enable --now {SYNC_TIMER}`",
        )
    last_error = report.get("last_error")
    if isinstance(last_error, dict):
        row("Last error", f"{last_error.get('message')} ({last_error.get('at')})")
    else:
        row("Last error", "none")
    failed = report.get("failed_shas")
    if isinstance(failed, dict) and failed:
        for index, (sha, record) in enumerate(failed.items()):
            if isinstance(record, dict):
                row(
                    "Failed" if index == 0 else "",
                    f"{short(sha)} {record.get('reason')}: {record.get('detail')}",
                )
    units = report.get("units")
    if isinstance(units, dict):
        for index, (unit, props) in enumerate(units.items()):
            if not isinstance(props, dict):
                continue
            if "error" in props:
                text = f"unavailable: {props['error']}"
            else:
                text = (
                    f"{props.get('ActiveState', '?')} ({props.get('SubState', '?')}), "
                    f"{props.get('UnitFileState') or props.get('LoadState', '?')}"
                )
            row("Units" if index == 0 else "", f"{unit:<26}{text}")
    legacy = report.get("legacy_units")
    if isinstance(legacy, list) and legacy:
        row("LEGACY", "system units still installed: " + ", ".join(map(str, legacy)))
    return "\n".join(lines)


def cmd_status(paths: Paths, as_json: bool) -> int:
    """`status [--json]`."""
    report = collect_status(paths)
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_status(report))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="cosmicsig_deploy.py",
        description=(
            "Continuous deployment of GitHub main to the CosmicSignature generator host "
            "(see docs/deployment.md)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log DEBUG lines too")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("run", help="one deployment tick (what cosmicsig-deploy.timer runs)")
    status = commands.add_parser("status", help="what is deployed, pending, failed or paused")
    status.add_argument("--json", action="store_true", help="machine-readable output")
    commands.add_parser(
        "install", help="install the user units and start the deploy timer (idempotent)"
    )
    pause = commands.add_parser("pause", help="stop deploying until `resume`")
    pause.add_argument("--reason", help="why (shown by status and every tick)")
    commands.add_parser("resume", help="undo `pause` (and the pause a rollback leaves)")
    retry = commands.add_parser("retry", help="forget a recorded failure so it is tried again")
    retry.add_argument("sha", nargs="?", help="the commit (default: origin/main)")
    commands.add_parser(
        "rollback", help="pause auto-deploy and switch back to the previous deployment"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line entry point; the exit status."""
    args = parse_args(argv)
    setup_logging(args.verbose)
    paths = Paths.from_env()
    command: str = args.command
    if command == "status":
        return cmd_status(paths, args.json)
    if command == "pause":
        return cmd_pause(paths, args.reason)
    if command == "resume":
        return cmd_resume(paths)
    if command == "retry":
        return cmd_retry(paths, args.sha)
    with stop_signals():
        if command == "run":
            return cmd_run(paths)
        if command == "install":
            return cmd_install(paths)
        return cmd_rollback(paths)


if __name__ == "__main__":
    sys.exit(main())
