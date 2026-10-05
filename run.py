#!/usr/bin/env python3
"""
CosmicSignature NFT asset package checker and uploader.

Fetches all tokens from the CosmicGame API, determines which per-seed asset
packages are incomplete on the destination server, generates missing packages
via the Rust binary, and uploads them via SCP.

Designed to run under a systemd user timer (5 minutes after each run; see
docs/deployment.md). Runs never overlap: each holds an exclusive lock on run.lock in the working
directory for its whole duration (RUN_LOCK), so a run started by hand while another runs exits
at once, and the deploy agent takes the same lock before it switches the checkout.

Configuration is read from (in increasing priority):
    1. .env file in the working directory
    2. Environment variables
    3. CLI arguments

Copy .env.example to .env and fill in your deployment values.

Generator exit statuses (see README "Exit status"):
    0   complete package: validated against REQUIRED_PACKAGE_FILES (CORE_PACKAGE_FILES when the
        generator predates the ember edition) and uploaded.
    3   complete except the ember edition (GENERATOR_EXIT_EMBER_FAILED). A new or incomplete
        package is validated against CORE_PACKAGE_FILES and uploaded, after any stale ember file
        is deleted from its remote directory. An ember backfill seed uploads nothing: its core
        package is already live. Either way the seed counts one failed ember attempt.
    2   rejected by the generator's argument parser (unknown flag or malformed value).
    1   any other failure, including an invalid --seed or a resolution above 16,384 per side.
    Any status other than 0 and 3 (a kill by a signal included) fails the seed: nothing is
    uploaded.

Ember backfill (remote packages that lack only the ember edition's files):
    --backfill-mode ember (default) regenerates the package locally, checks that the render shows
    the same orbit in the same view as the live package (the ember edition follows the main
    edition's view), and uploads only the ember files and a merged metadata/assets.json; the
    published main art, spectral files, generation.json and nft_traits.json are never touched.
    --backfill-mode full replaces the whole remote package. A seed whose ember edition fails
    --max-backfill-attempts times with the same generator binary is given up
    (backfill_failures.json). A seed whose regenerated orbit or view differs from the live
    package's is given up at once: the same binary always regenerates the same orbit and view. A
    backfill run that fails for any other reason is not counted toward that cap, but moves the
    seed behind the seeds that have failed less often, so a seed that always fails cannot stall
    the backfill.

Stale ember editions (after a deploy that changes the ember edition's look):
    `<generator> --ember-algorithm` prints the id of the look the generator renders (ember-v<N>,
    the "algorithm" every package records in metadata/ember.json). Before planning, a run reads
    the id of every live certificate in one ssh call. A listed seed's edition whose id is older
    is stale, and its package is planned as an ember backfill seed, which the backfill renders
    again in the current look (--max-backfill per run). What happens to the stale edition until
    then is the operator's choice:
      * by default it is withdrawn at once: its certificate first, then its metadata/assets.json
        entries, then its media. The token has no ember edition until its turn in the backfill,
        so the old look is never online next to the new one;
      * with --keep-stale-ember (env COSMICSIG_KEEP_STALE_EMBER=yes) nothing is withdrawn: the
        edition stays online until the backfill replaces it in place, so no token is without an
        ember edition, and both looks are online while the backfill works through the
        collection. With --max-backfill 0 as well, every live edition is held exactly as it is.
    No edition is stale unless both ids can be read.

Uploads: the metadata files and the certificate are uploaded under temporary names and renamed
    into place, so an interrupted upload never leaves a truncated metadata file; a whole package
    first loses its remote metadata/assets.json, so an interrupted one reads as incomplete (and
    is regenerated in full) until its manifest has landed. An ember-mode backfill stages every
    file of the edition that way, media included, and swaps them in with one ssh call once all
    have landed: a failed transfer leaves the live package, any older ember edition included,
    exactly as it was.

Usage:
    python3 run.py [--dry-run]
    python3 run.py --ssh-host HOST --ssh-user USER --api-url URL --remote-dir DIR
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import decimal
import enum
import fcntl
import json
import logging
import logging.handlers
import math
import os
import posixpath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Container, Iterable, Sequence
from pathlib import Path
from typing import TypeGuard

from _utils import GENERATOR_CANDIDATES, fmt_duration

# ---------------------------------------------------------------------------
# Defaults (non-sensitive only; deployment values come from .env / env vars)
# ---------------------------------------------------------------------------

# Per-seed generator timeout. A package is a full render, and the ember edition is the most
# expensive part of it: a fine fluid grid, supersampling and three films, the film and the same film
# four and ten times slower, with up to four and ten times its frames. Its time on the production
# host is re-measured after every deploy that changes the look (the `OK  seed=... (total ...)` log
# lines); no figure is quoted here. The timeout only has to catch a render that hangs, so it leaves
# ample headroom, but it must stay well below the service's 36-hour TimeoutStartSec (RUN_CEILING): a
# render that hangs is then stopped by run.py, not by systemd (a stopped run counts no failure, so
# the same seed would come first again, every run). A backfill seed whose render overruns counts one
# failed ember attempt, so an orbit that always overruns is given up after --max-backfill-attempts
# renders; an urgent seed is rendered again without the ember edition (CORE_ONLY_TIMEOUT), so the
# token gets its main art and the edition joins the backfill.
DEFAULT_TIMEOUT = 16 * 3600  # 16 hours
# The timeout of that render without the ember edition, or --timeout if that is shorter. A core
# package takes about 50 minutes on the production host, so this too only catches a hang, and it is
# small enough that after a run's first seed overran twice (DEFAULT_TIMEOUT, then this), the next
# seed can still start in the same run: 16 + 2 hours leave an hour of the run budget's 19 for the
# uploads and checks in between (a test keeps that slack).
CORE_ONLY_TIMEOUT = 2 * 3600  # 2 hours
# The longest a whole run may take: the sync unit's TimeoutStartSec
# (ops/systemd/cosmicsig-sync.service; a test keeps the two equal), after which systemd stops it
# mid-render. A run starts a seed only while that seed's --timeout and UPLOAD_MARGIN still fit
# before the ceiling, and leaves the rest of its queue to the next run (the run budget), so the
# ceiling is a safety net that is never expected to fire.
RUN_CEILING = 36 * 3600  # 36 hours
# The room a run keeps after a render's timeout for its upload and the end of the run.
UPLOAD_MARGIN = 3600  # 1 hour
# The longest --timeout accepted: a run must be able to start a seed with it and still have room,
# within RUN_CEILING, for that seed's render without the ember edition and for its upload.
MAX_TIMEOUT = RUN_CEILING - CORE_ONLY_TIMEOUT - UPLOAD_MARGIN
API_TOKEN_FETCH_LIMIT = 999999
DEFAULT_ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
DEFAULT_NFT_CONTRACT = "0xbb84Be3500A63581d3F2d5AC3bdF8685AAedad25"

LOCAL_OUTPUT_DIR = Path("output")
LOG_FILE = "imgcheck.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
SEED_MISMATCH_REPORT = Path("seed_source_mismatch.json")
BACKFILL_FAILURES = Path("backfill_failures.json")
# The single-instance lock, held (flock, exclusive) for the whole run. It holds the pid of its
# holder, for the error message of a run that finds it taken. ops/deploy/cosmicsig_deploy.py
# takes it too while it switches the checkout and the generator binary between runs.
RUN_LOCK = Path("run.lock")

# The exit status of ssh itself failing (the connection could not be made, or was lost), as
# opposed to the remote command's own status.
SSH_FAILURE = 255
# Attempts of each idempotent ssh call made after a render (reading the live package back,
# preparing the upload, swapping the edition in, deleting staged files): a connection that fails
# once must not throw away hours of rendering.
SSH_ATTEMPTS = 3

SSH_BASE_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
]

# An scp transfer may take SCP_MIN_TIMEOUT seconds, or longer when it carries more than
# SCP_MIN_TIMEOUT * SCP_MIN_BYTES_PER_SECOND bytes. The ember edition's videos are the large
# transfers: with the ember-v3 look the slow film measured 176 to 926 MB and the archival film 134
# to 568 MB on the production host, and the medium film of ember-v4 is about half the slow film's
# size; the `UPLOAD ... (N MB, timeout Ns)` log lines give each transfer's size.
SCP_MIN_TIMEOUT = 900
SCP_MIN_BYTES_PER_SECOND = 1_000_000

# A staged upload (UploadStep.staged) writes each file as <name>.part and then renames it.
PART_SUFFIX = ".part"

# Every ssh command that changes a package on the asset host and may run twice (a retry after a
# lost connection or a timeout, while the first attempt may still be running there) holds an
# exclusive lock on this file in the package directory (util-linux flock; locked()), so a retry
# waits for the earlier attempt to finish and then runs again idempotently. The file is empty,
# stays in the package directory, and is not part of the package: list_remote_files() leaves it
# out, and no manifest lists it.
PACKAGE_LOCK = ".ember.lock"
# How long a locked command waits for the lock: less than the ssh calls' own timeouts (30 s).
PACKAGE_LOCK_WAIT = 20

# The generator capability probe: `<generator> --help` must list this flag, which the ember
# edition introduced (tests/cli.rs pins it). A binary without it predates the edition.
GENERATOR_EMBER_FLAG = "--no-ember"
GENERATOR_PROBE_TIMEOUT = 30

# The ember algorithm probe: `<generator> --ember-algorithm` prints the id of the ember look it
# renders, the "algorithm" of every metadata/ember.json it writes, and exits 0 without rendering.
# A binary that predates the flag rejects it (exit status 2). The id's number grows whenever the
# edition's rendered bits change, so a live certificate with a lower number holds a look the
# generator no longer renders: a stale edition (retire_stale_ember_editions()).
GENERATOR_EMBER_ALGORITHM_FLAG = "--ember-algorithm"
EMBER_ALGORITHM_RE = re.compile(r"ember-v(?P<number>[0-9]+)")


# Expected per-seed package emitted by the Rust generator.
SPECTRAL_BIN_COUNT = 64
SPECTRAL_FILE_RE = re.compile(r"^(?P<bin>\d{2})_\d+nm\.png$")
EXPECTED_SPECTRAL_BINS = set(range(SPECTRAL_BIN_COUNT))
ASSET_MANIFEST = "metadata/assets.json"
NFT_TRAITS = "metadata/nft_traits.json"
EMBER_CERTIFICATE = "metadata/ember.json"
CORE_PACKAGE_FILES = (
    "images/source/master.png",
    "images/web/full.webp",
    "images/web/preview.webp",
    "videos/web/main.mp4",
    "videos/web/spectral_sweep.mp4",
    "videos/hq/main.mp4",
    "videos/hq/spectral_sweep.mp4",
    "metadata/generation.json",
    ASSET_MANIFEST,
    NFT_TRAITS,
)
# The ember edition (the orbit drawn in sumi ink by the fluid it stirs, in the main edition's view)
# and its determinism certificate: the still and its two WebP derivatives, the film, the medium film
# and the slow film (the same film four and ten times slower), the archival film, the certificate.
# Packages generated before the edition existed, whose ember edition failed, or whose stale edition
# was withdrawn or lacks a file of the current look, lack only these files: they are regenerated as
# a backfill that yields to new mints (see find_missing_seeds, plan_seed_queue, --backfill-mode and
# retire_stale_ember_editions). Keep in sync with app::EMBER_OUTPUT_PATHS, order included: a Rust
# unit test reads this tuple from the source text, so it stays one plain string literal per line.
EMBER_PACKAGE_FILES = (
    "images/source/ember.png",
    "images/web/ember_full.webp",
    "images/web/ember_preview.webp",
    "videos/web/ember.mp4",
    "videos/web/ember_medium.mp4",
    "videos/web/ember_slow.mp4",
    "videos/hq/ember.mp4",
    "metadata/ember.json",
)
REQUIRED_PACKAGE_FILES = CORE_PACKAGE_FILES + EMBER_PACKAGE_FILES
# The ember edition's media: every ember file except its certificate, which is uploaded last.
EMBER_MEDIA_FILES = tuple(path for path in EMBER_PACKAGE_FILES if path != EMBER_CERTIFICATE)
# Every ember file's name starts with this, and no other package file's does (a test pins both):
# retired_ember_files() deletes nothing whose name does not.
EMBER_FILE_PREFIX = "ember"

# The roles of the ember entries in metadata/assets.json, in the order of EMBER_MEDIA_FILES
# (every role of the edition starts with EMBER_ROLE_PREFIX; the certificate has no manifest
# entry). A full package lists every one of them.
EMBER_ROLE_PREFIX = "ember_"
EMBER_MANIFEST_ROLES = (
    "ember_source_master",
    "ember_web_full",
    "ember_web_preview",
    "ember_web",
    "ember_medium_web",
    "ember_slow_web",
    "ember_hq",
)

# The metadata/nft_traits.json fields that identify the picture the ember edition must match: the
# selected orbit, and the view the main edition shows it in. An ember-mode backfill uploads the
# regenerated ember edition only if every one of IDENTITY_FIELDS equals the live package's
# (compared as exact JSON values: numbers by their exact decimal value, never as floats), so the
# edition always draws the orbit of the published main art, as that art shows it.
ORBIT_IDENTITY_FIELDS = (
    ("simulation", "masses"),
    ("generation", "borda", "selected_index"),
    ("generation", "borda", "retry_count"),
)
# The ember bodies follow the main edition's view: its projection (position space or one of the
# phase-space projections), its symmetry (which scales the primary copy the bodies follow), its
# camera drift, and the output resolution, whose size and aspect set the frame and the symmetry's
# scale (run.py never passes the generator's -r, but a package rendered at another size must not
# get an edition framed for this one). ("mode" is "none" exactly when the drift is off, so
# "enabled" adds nothing; "randomized" says how the values were chosen, not what they are.) The
# viewing rotation and the frame are derived while rendering and are not recorded in
# nft_traits.json, so they cannot be compared: they are ASSUMED to match once these fields do.
# The layer stack is compared as their recorded proxy: the rotation is the best of four by a
# score that depends on the adaptively chosen stack (computed with platform floating point), so
# another stack means the rotation was chosen for another picture.
VIEW_IDENTITY_FIELDS = (
    ("generation", "structure", "stack_label"),
    ("generation", "projection"),
    ("generation", "symmetry"),
    ("generation", "drift", "mode"),
    ("generation", "drift", "scale"),
    ("generation", "drift", "arc_fraction"),
    ("generation", "drift", "orbit_eccentricity"),
    ("generation", "resolution", "width"),
    ("generation", "resolution", "height"),
)
IDENTITY_FIELDS = ORBIT_IDENTITY_FIELDS + VIEW_IDENTITY_FIELDS

# Generator exit status for a package that is complete except for the ember edition: its
# preflight or its stage failed, the generator removed every ember file, and it wrote the rest
# of the package (metadata included) exactly as with --no-ember. A new mint's core package is
# uploaded, so the token gets its artwork and traits; an ember backfill seed uploads nothing.
# Either way the seed counts one failed ember attempt in the backfill failure ledger.
GENERATOR_EXIT_EMBER_FAILED = 3

# Backfill seeds (packages missing only the ember edition) generated per run. Each run first
# generates every seed missing a core file (new mints), so a mint waits for at most this many
# backfill packages, each a full render that takes hours (see DEFAULT_TIMEOUT), plus the timer's
# restart delay; a burst of mints larger than one run's budget (RUN_CEILING) spans several runs.
DEFAULT_MAX_BACKFILL = 1

# Failed ember attempts after which a backfill seed is given up. The count is kept per generator
# binary (see GeneratorIdentity) and resets when the binary changes, so a rebuilt generator
# retries every seed; until then a given-up seed is logged as a WARNING on every run. Backfill
# runs that fail for another reason are counted separately (BackfillLedger.other_failures): they
# only order the queue and never give a seed up. An orbit or view mismatch does not wait for the
# cap: it gives the seed up at once (Outcome.IDENTITY_MISMATCH).
MAX_BACKFILL_ATTEMPTS = 3

# Environment variable names for required config
ENV_SSH_HOST = "COSMICSIG_SSH_HOST"
ENV_SSH_USER = "COSMICSIG_SSH_USER"
ENV_API_URL = "COSMICSIG_API_URL"
ENV_REMOTE_DIR = "COSMICSIG_REMOTE_DIR"
ENV_ARBITRUM_RPC_URL = "COSMICSIG_ARBITRUM_RPC_URL"
ENV_NFT_CONTRACT = "COSMICSIG_NFT_CONTRACT"
ENV_MAX_BACKFILL = "COSMICSIG_MAX_BACKFILL"
ENV_BACKFILL_MODE = "COSMICSIG_BACKFILL_MODE"
ENV_MAX_BACKFILL_ATTEMPTS = "COSMICSIG_MAX_BACKFILL_ATTEMPTS"
ENV_KEEP_STALE_EMBER = "COSMICSIG_KEEP_STALE_EMBER"

# Minimal ABI selectors for the verified Cosmic Signature NFT contract.
SELECTOR_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()
SELECTOR_TOKEN_BY_INDEX = "0x4f6ccce7"  # tokenByIndex(uint256)
SELECTOR_GET_NFT_SEED = "0xb0c0fe4e"  # getNftSeed(uint256)


class BackfillMode(enum.Enum):
    """How a backfill seed (a live package that lacks only the current ember edition) is
    uploaded."""

    EMBER = "ember"
    """Non-destructive (the default): the package is regenerated in full locally, but only its
    ember edition (EMBER_PACKAGE_FILES and a merged metadata/assets.json) is uploaded, and only
    if the render shows the same orbit in the same view as the live package (IDENTITY_FIELDS).
    The published main art, spectral files, generation.json and nft_traits.json are never
    touched."""
    FULL = "full"
    """The whole regenerated package replaces the remote one, main art included."""


DEFAULT_BACKFILL_MODE = BackfillMode.EMBER

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

shutdown_requested = False
log = logging.getLogger("cosmicsig")

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

_ENV_LINE_RE = re.compile(r"""^\s*(?:export\s+)?(?P<key>[A-Za-z_]\w*)\s*=\s*(?P<val>.*)$""")


def load_dotenv(path: str = ".env") -> None:
    """
    Read a .env file and populate os.environ for any keys not already set.
    Supports KEY=VALUE, optional 'export' prefix, and # comments.
    Existing environment variables are never overwritten.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(line)
        if not m:
            continue
        key = m.group("key")
        val = m.group("val").strip().strip("\"'")
        if key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    """Log DEBUG and up to LOG_FILE (rotated) and INFO and up to stdout (idempotent)."""
    if log.handlers:
        return
    log.setLevel(logging.DEBUG)

    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


def acquire_run_lock(path: Path = RUN_LOCK) -> int | None:
    """Take the exclusive single-instance lock on `path` without waiting; its descriptor.

    Returns None if another process holds it. The lock is released when the descriptor is
    closed, at the latest when the process exits, however it exits (a flock dies with its
    holder, so a crash never leaves a stale lock). Python opens the file non-inheritable
    (PEP 446), so the generator, ssh and scp never hold it: once run.py is gone, the lock is
    free even if a child it started is still being killed. Raises OSError if the file cannot be
    opened or locked for another reason.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    # The pid is only for messages: a full disk must not stop the run.
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def run_lock_holder(path: Path = RUN_LOCK) -> str:
    """The pid the lock's holder wrote into it, for messages ("unknown" if unreadable)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "unknown"
    return text if text.isdigit() else "unknown"


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def install_signal_handlers() -> None:
    """On SIGTERM/SIGINT start no new seed or upload retry; a second signal exits at once.

    Under systemd the whole control group gets the signal, so the generator (or scp) in flight
    dies with it and that seed is generated again by a later run.
    """

    def handler(signum: int, _frame: types.FrameType | None) -> None:
        global shutdown_requested
        name = signal.Signals(signum).name
        if shutdown_requested:
            log.warning("Second %s received -- forcing exit", name)
            sys.exit(1)
        shutdown_requested = True
        log.info("Received %s -- will finish current seed then exit", name)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ssh_opts() -> list[str]:
    """Options shared by ssh and scp: batch mode, timeouts, and SSH_OPTS_EXTRA."""
    extra = os.environ.get("SSH_OPTS_EXTRA", "").split()
    return [*SSH_BASE_OPTS, *extra]


def ssh_cmd(host: str, user: str) -> list[str]:
    """The argv prefix that runs a command on `host` as `user`."""
    return ["ssh", *ssh_opts(), "-l", user, host]


def run_subprocess(
    cmd: list[str],
    *,
    timeout: int | None = 60,
    label: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with full logging. The caller checks the returncode.

    Output is decoded as UTF-8 (undecodable bytes are replaced). Raises TimeoutExpired or
    OSError (both logged) when the command times out or cannot be started.
    """
    log.debug("[%s] Running: %s", label, " ".join(cmd))
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        log.debug(
            "[%s] Finished in %s  rc=%d  stdout=%d bytes  stderr=%d bytes",
            label,
            fmt_duration(elapsed),
            result.returncode,
            len(result.stdout),
            len(result.stderr),
        )
        if result.stdout:
            log.debug("[%s] stdout:\n%s", label, result.stdout.rstrip()[-2000:])
        if result.stderr:
            log.debug("[%s] stderr:\n%s", label, result.stderr.rstrip()[-2000:])
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        log.error("[%s] TIMEOUT after %s (limit=%ss)", label, fmt_duration(elapsed), timeout)
        raise
    except OSError as exc:
        log.error("[%s] OS ERROR: %s", label, exc)
        raise


def run_remote(
    ssh_host: str,
    ssh_user: str,
    command: str,
    *,
    timeout: int,
    label: str,
    attempts: int = 1,
) -> subprocess.CompletedProcess[str] | None:
    """Run a shell command on the remote host.

    Returns None if ssh timed out or could not be started (already logged); any exit status is
    returned to the caller, which knows what it means (SSH_FAILURE, 255, is an ssh error). With
    `attempts` > 1, a run that ended so (a dropped or refused connection, a timeout) is tried
    again, after a backoff, up to `attempts` times in all, but no retry starts once a shutdown is
    requested. Such a run may have stopped half way through `command`, so only a command that
    can run again from the start (idempotent) may be given more than one attempt.
    """
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = run_subprocess(
                [*ssh_cmd(ssh_host, ssh_user), command], timeout=timeout, label=label
            )
        except (subprocess.TimeoutExpired, OSError):
            result = None
        if result is not None and result.returncode != SSH_FAILURE:
            return result
        if attempt == attempts or shutdown_requested:
            break
        backoff = 2**attempt
        log.warning(
            "[%s] ssh failed (attempt %d/%d); retrying in %ds", label, attempt, attempts, backoff
        )
        time.sleep(backoff)
    return result


def normalize_seed(seed: str | int) -> str:
    """Return a canonical 32-byte lowercase hex seed without 0x."""
    if isinstance(seed, int):
        value = seed
    else:
        hex_seed = seed.strip().removeprefix("0x").removeprefix("0X")
        if not hex_seed:
            raise ValueError("empty seed")
        if not re.fullmatch(r"[0-9a-fA-F]+", hex_seed):
            raise ValueError(f"non-hex seed: {seed!r}")
        value = int(hex_seed, 16)

    if value < 0 or value >= 2**256:
        raise ValueError(f"seed outside uint256 range: {seed!r}")
    return f"{value:064x}"


def safe_url_for_log(url: str | None) -> str:
    """Redact path/query details because RPC URLs often contain API keys."""
    if not url:
        return "(not set)"
    parts = urllib.parse.urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return "(configured)"
    suffix = "/..." if parts.path and parts.path != "/" else ""
    return f"{parts.scheme}://{parts.netloc}{suffix}"


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


def fetch_token_seeds(api_base_url: str, retries: int = 3) -> list[str]:
    """
    Fetch all CosmicSignature token seeds from the API.
    Returns normalized hex seed strings (without 0x prefix).
    """
    api_base_url = api_base_url.rstrip("/")
    if not api_base_url:
        raise RuntimeError("CosmicGame API URL is not configured")

    url = f"{api_base_url}/api/cosmicgame/cst/list/all/0/{API_TOKEN_FETCH_LIMIT}"
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        backoff = 2**attempt
        try:
            log.info("Fetching token list from API (attempt %d/%d)", attempt, retries)
            log.debug("GET %s", url)

            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                log.debug("API response: %d bytes, HTTP %d", len(raw), resp.status)

            data = json.loads(raw)

            api_status = data.get("status", 0)
            if str(api_status) != "1":
                err_msg = data.get("error", "unknown")
                raise ValueError(f"API returned status={api_status} error={err_msg}")

            token_list = data.get("CosmicSignatureTokenList", [])
            if not token_list:
                raise ValueError("CosmicSignatureTokenList is empty or missing")

            seen: set[str] = set()
            seeds: list[str] = []
            for token in token_list:
                seed = token.get("Seed", "")
                if not seed:
                    continue
                seed = normalize_seed(str(seed))
                if seed and seed not in seen:
                    seen.add(seed)
                    seeds.append(seed)

            if not seeds:
                raise ValueError("No valid seeds found in token list")

            log.info("Fetched %d unique token seeds from API", len(seeds))
            return seeds

        except Exception as exc:
            last_err = exc
            log.warning("API attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                log.info("Retrying in %ds ...", backoff)
                time.sleep(backoff)

    raise RuntimeError(f"Failed to fetch token seeds after {retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Blockchain fetch
# ---------------------------------------------------------------------------


def normalize_eth_address(address: str) -> str:
    """Return `address` stripped, or raise ValueError unless it is 0x plus 40 hex digits."""
    value = address.strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        raise ValueError(f"invalid Ethereum address: {address!r}")
    return value


def encode_uint256_arg(value: int) -> str:
    """ABI-encode a uint256 call argument (64 hex digits, no 0x)."""
    if value < 0 or value >= 2**256:
        raise ValueError(f"uint256 argument out of range: {value}")
    return f"{value:064x}"


def decode_uint256_result(result: str) -> int:
    """Decode a uint256 from an eth_call result (the last 32 bytes)."""
    if not isinstance(result, str) or not result.startswith("0x"):
        raise RuntimeError(f"invalid eth_call result: {result!r}")
    hex_value = result[2:]
    if len(hex_value) < 64:
        raise RuntimeError(f"short eth_call result: {result!r}")
    return int(hex_value[-64:], 16)


def rpc_request(rpc_url: str, method: str, params: list[object], timeout: int = 30) -> object:
    """Make one JSON-RPC call and return its result (RuntimeError on an RPC error)."""
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        rpc_url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cosmicsig-sync/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(f"RPC {method} failed: {data['error']}")
    if "result" not in data:
        raise RuntimeError(f"RPC {method} response missing result")
    return data["result"]


def eth_call_uint256(rpc_url: str, contract: str, calldata: str) -> int:
    """Call a view function of `contract` that returns one uint256."""
    result = rpc_request(
        rpc_url,
        "eth_call",
        [
            {
                "to": contract,
                "data": calldata,
            },
            "latest",
        ],
    )
    return decode_uint256_result(str(result))


def fetch_blockchain_token_seeds(
    rpc_url: str,
    nft_contract: str,
    retries: int = 2,
) -> list[str]:
    """Fetch token seeds directly from the Arbitrum NFT contract."""
    rpc_url = rpc_url.strip()
    if not rpc_url:
        raise RuntimeError("Arbitrum RPC URL is not configured")
    nft_contract = normalize_eth_address(nft_contract)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info(
                "Fetching token seeds from Arbitrum contract %s (attempt %d/%d)",
                nft_contract,
                attempt,
                retries,
            )
            total_supply = eth_call_uint256(rpc_url, nft_contract, SELECTOR_TOTAL_SUPPLY)
            log.info("NFT totalSupply from chain: %d", total_supply)
            if total_supply <= 0:
                raise ValueError("NFT totalSupply is zero")

            seen: set[str] = set()
            seeds: list[str] = []
            for index in range(total_supply):
                if shutdown_requested:
                    raise RuntimeError("shutdown requested while fetching blockchain seeds")

                token_id = eth_call_uint256(
                    rpc_url,
                    nft_contract,
                    f"{SELECTOR_TOKEN_BY_INDEX}{encode_uint256_arg(index)}",
                )
                seed_value = eth_call_uint256(
                    rpc_url,
                    nft_contract,
                    f"{SELECTOR_GET_NFT_SEED}{encode_uint256_arg(token_id)}",
                )
                seed = normalize_seed(seed_value)
                if seed not in seen:
                    seen.add(seed)
                    seeds.append(seed)

                if (index + 1) % 50 == 0 or index + 1 == total_supply:
                    log.info(
                        "Fetched %d/%d NFT seeds from chain (%d unique)",
                        index + 1,
                        total_supply,
                        len(seeds),
                    )

            if not seeds:
                raise ValueError("No valid seeds found on chain")

            log.info("Fetched %d unique token seeds from blockchain", len(seeds))
            return seeds
        except Exception as exc:
            last_err = exc
            log.warning("Blockchain attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                backoff = 2**attempt
                log.info("Retrying blockchain fetch in %ds ...", backoff)
                time.sleep(backoff)

    raise RuntimeError(f"Failed to fetch token seeds from blockchain: {last_err}")


def write_seed_mismatch_report(api_seeds: list[str], chain_seeds: list[str]) -> None:
    """Write the difference between the API and the chain seed lists to SEED_MISMATCH_REPORT."""
    api_set = set(api_seeds)
    chain_set = set(chain_seeds)
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_count": len(api_seeds),
        "api_unique_count": len(api_set),
        "blockchain_count": len(chain_seeds),
        "blockchain_unique_count": len(chain_set),
        "api_only": sorted(api_set - chain_set),
        "blockchain_only": sorted(chain_set - api_set),
    }
    SEED_MISMATCH_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.error("Seed source mismatch report written to %s", SEED_MISMATCH_REPORT)


def resolve_token_seeds(
    api_url: str,
    arbitrum_rpc_url: str,
    nft_contract: str,
) -> tuple[list[str], str]:
    """
    Prefer API seeds, verify against chain when possible, and fall back to chain
    if the API is unavailable.
    """
    api_seeds: list[str] | None = None
    chain_seeds: list[str] | None = None
    api_error: Exception | None = None
    chain_error: Exception | None = None

    try:
        api_seeds = fetch_token_seeds(api_url)
    except RuntimeError as exc:
        api_error = exc
        log.warning("API seed source unavailable: %s", exc)

    try:
        chain_seeds = fetch_blockchain_token_seeds(arbitrum_rpc_url, nft_contract)
    except RuntimeError as exc:
        chain_error = exc
        log.warning("Blockchain seed source unavailable: %s", exc)

    if api_seeds is not None and chain_seeds is not None:
        if set(api_seeds) != set(chain_seeds):
            write_seed_mismatch_report(api_seeds, chain_seeds)
            raise RuntimeError("API and blockchain seed lists do not match; refusing to continue")
        log.info("API and blockchain seed sources match (%d unique seeds)", len(api_seeds))
        return api_seeds, "API verified against blockchain"

    if api_seeds is not None:
        log.warning("Using API seeds without blockchain verification: %s", chain_error)
        return api_seeds, "API only"

    if chain_seeds is not None:
        log.warning("Using blockchain seed fallback because API failed: %s", api_error)
        return chain_seeds, "blockchain fallback"

    raise RuntimeError(f"Both seed sources failed: API={api_error}; blockchain={chain_error}")


# ---------------------------------------------------------------------------
# Remote file check
# ---------------------------------------------------------------------------


def remote_seed_dir(remote_dir: str, seed: str) -> str:
    """The remote package directory of `seed`: <remote_dir>/0x<seed>."""
    return f"{remote_dir.rstrip('/')}/0x{seed}"


def list_remote_files(ssh_host: str, ssh_user: str, remote_dir: str) -> set[str] | None:
    """List the files of every package beneath the remote asset directory.

    Returns paths relative to `remote_dir` (e.g. "0x<seed>/metadata/assets.json"). Returns None
    if the listing failed (an ssh timeout or error, a remote asset directory that is missing or
    cannot be entered, or a find error): planning against a partial or empty listing would take
    every published package for a new mint and replace it in full, so the caller must stop. The
    directory must exist before the first run (preflight() checks that it is writable).
    """
    quoted_dir = shlex.quote(remote_dir)
    remote_cmd = f"cd {quoted_dir} || exit 1; find . -mindepth 2 -maxdepth 4 -type f -print"
    result = run_remote(ssh_host, ssh_user, remote_cmd, timeout=60, label="ssh-find")
    if result is None:
        log.error("Could not list the remote package files (ssh did not complete)")
        return None
    if result.returncode != 0:
        log.error(
            "Listing the remote package files failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip()[:300],
        )
        return None

    files = {
        line.strip().removeprefix("./")
        for line in result.stdout.splitlines()
        if line.strip() and posixpath.basename(line.strip()) != PACKAGE_LOCK
    }
    log.info("Found %d existing package files on remote server", len(files))
    return files


def missing_remote_package_parts(seed: str, remote_files: set[str]) -> list[str]:
    """Return missing files/groups for the remote package at 0x<seed>/."""
    package_dir = f"0x{seed}"
    required_files = [f"{package_dir}/{filename}" for filename in REQUIRED_PACKAGE_FILES]
    missing = [
        path.removeprefix(f"{package_dir}/") for path in required_files if path not in remote_files
    ]

    spectral_prefix = f"{package_dir}/spectral/"
    spectral_bins: set[int] = set()
    for path in remote_files:
        if not path.startswith(spectral_prefix):
            continue
        filename = path.removeprefix(spectral_prefix)
        match = SPECTRAL_FILE_RE.match(filename)
        if match:
            bin_idx = int(match.group("bin"))
            if bin_idx in EXPECTED_SPECTRAL_BINS:
                spectral_bins.add(bin_idx)

    missing_bins = EXPECTED_SPECTRAL_BINS - spectral_bins
    if missing_bins:
        missing.append(f"spectral/*.png ({len(missing_bins)} missing)")

    return missing


def is_ember_backfill(missing_parts: list[str]) -> bool:
    """True if a package lacks only ember edition files.

    Such a package predates the edition, was uploaded after its ember edition failed (exit 3),
    lost its stale edition to a withdrawal, or holds an edition of an older look that lacks a
    file of the current one. Also used for a local package, whose missing parts then show a
    failed ember edition.
    """
    return bool(missing_parts) and all(part in EMBER_PACKAGE_FILES for part in missing_parts)


def find_missing_seeds(seeds: list[str], remote_files: set[str]) -> tuple[list[str], list[str]]:
    """Split the API seeds whose remote package is incomplete into (urgent, backfill).

    Urgent seeds lack a core file: new mints and broken uploads. Backfill seeds lack only ember
    edition files (is_ember_backfill()); retire_stale_ember_editions() takes the files of every
    stale edition out of `remote_files` first, so those packages are backfill seeds too. Both
    lists keep API order.
    """
    urgent: list[str] = []
    backfill: list[str] = []
    for seed in seeds:
        missing_parts = missing_remote_package_parts(seed, remote_files)
        if not missing_parts:
            continue
        log.debug("MISSING  0x%s  (%s)", seed, ", ".join(missing_parts))
        (backfill if is_ember_backfill(missing_parts) else urgent).append(seed)
    return urgent, backfill


def given_up_seeds(backfill: list[str], ledger: BackfillLedger, max_attempts: int) -> list[str]:
    """Backfill seeds given up with this generator binary (in API order): those whose ember
    edition failed `max_attempts` or more times, and those whose orbit or view did not match."""
    return [seed for seed in backfill if ledger.given_up(seed, max_attempts)]


def plan_seed_queue(
    urgent: list[str],
    backfill: list[str],
    max_backfill: int,
    ledger: BackfillLedger,
    max_attempts: int = MAX_BACKFILL_ATTEMPTS,
) -> list[str]:
    """Seeds to generate this run: every urgent seed, then at most `max_backfill` backfill seeds.

    Urgent seeds go in order of their overruns (BackfillLedger.urgent_overruns: fewest first,
    then API order), so a new mint whose render keeps running past --timeout goes behind the
    others, which the run budget would otherwise never reach.

    The cap bounds the latency of new mints. A run plans its queue once, at the start, so a token
    minted during a run waits for that run to finish: after the run's own new mints, at most
    `max_backfill` backfill packages, each a full render. A run also starts a seed only while its
    --timeout fits in the run budget (RUN_CEILING, less UPLOAD_MARGIN), and leaves the rest of
    its queue to the next run, which plans afresh 5 minutes later (picking up the tokens minted
    meanwhile, first): a burst of new mints, such as the several tokens a round's end mints at
    once, spans several runs. The unit's TimeoutStartSec, which systemd enforces, is only the
    safety net behind that budget, and is never expected to fire (systemd's RuntimeMaxSec has no
    effect on a Type=oneshot service).

    Backfill seeds go in order of their failed backfill runs, whatever the reason
    (BackfillLedger.failed_runs(): fewest first, then API order). A seed that failed is retried
    only once every other waiting seed has failed as often or is done, so a seed that always fails,
    for any reason, costs one render per pass over the backlog and cannot stall the backfill.
    Seeds with `max_attempts` or more failed ember attempts are left out: they are given up until
    the generator binary changes, so a seed whose ember edition always fails costs at most
    `max_attempts` renders. A seed whose regenerated orbit or view did not match the live package
    is left out after that one render. Other failures never give a seed up.
    """
    eligible = [seed for seed in backfill if not ledger.given_up(seed, max_attempts)]
    ordered = sorted(eligible, key=ledger.failed_runs)
    first = sorted(urgent, key=lambda seed: ledger.urgent_overruns.get(seed, 0))
    return [*first, *ordered[: max(max_backfill, 0)]]


# ---------------------------------------------------------------------------
# Per-seed outcomes
# ---------------------------------------------------------------------------


class Outcome(enum.Enum):
    """How far a seed got: the result of generating it, and then of processing it."""

    FAILED = "failed"
    """Nothing usable, for a reason other than the ember edition: the generator failed or was
    killed, the core package is incomplete, the live package of an ember-mode backfill cannot be
    read or used, or ssh/scp failed. Not an ember attempt: for a backfill seed the ledger counts
    it as another failure, which only moves the seed back in the queue."""
    TIMED_OUT = "timed out"
    """The generator ran past --timeout and was stopped; nothing was uploaded. For a backfill
    seed it counts one failed ember attempt (the ember edition is what makes a package slow), so
    an orbit that always overruns is given up after --max-backfill-attempts renders. An urgent
    seed is rendered again without the ember edition and uploaded as CORE_ONLY; it ends
    TIMED_OUT when that render could not start (the run budget, or a shutdown) or overran too,
    and counts one overrun (BackfillLedger.urgent_overruns), which puts it behind the new mints
    that overran less often, so a seed that keeps overrunning cannot hold up the others."""
    COMPLETE = "complete"
    """The full package, ember edition included (for an ember-mode backfill: its ember edition
    was uploaded)."""
    CORE_ONLY = "core only"
    """The package without the ember edition: generated with GENERATOR_EXIT_EMBER_FAILED, or
    (after processing) uploaded without it. Counts one failed ember attempt."""
    EMBER_FAILED = "ember failed"
    """The ember edition failed and nothing was uploaded: a backfill seed's exit 3, regenerated
    metadata that cannot show its orbit and view or give its ember entries, or an incomplete
    local ember edition. Counts one failed ember attempt."""
    IDENTITY_MISMATCH = "identity mismatch"
    """An ember-mode backfill whose regenerated package shows another orbit, or the same orbit in
    another view, than the live one (IDENTITY_FIELDS): nothing was uploaded. The generator is
    deterministic, so this binary would regenerate the same package on every retry: the seed
    counts one failed ember attempt and is given up at once (BackfillLedger.identity_mismatches)
    instead of after --max-backfill-attempts renders."""


# ---------------------------------------------------------------------------
# Backfill failure ledger
# ---------------------------------------------------------------------------


def _is_int(value: object) -> TypeGuard[int]:
    """True for an int that is not a bool (a JSON number without a fraction)."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclasses.dataclass(frozen=True)
class GeneratorIdentity:
    """The generator binary a ledger's failure counts belong to.

    Identified by its resolved path, size and modification time: a rebuild or a new binary
    changes the identity, and the counts start again from zero, so a fixed generator retries
    every seed it had given up.
    """

    path: str
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, binary: str) -> GeneratorIdentity | None:
        """The identity of the binary at `binary`, or None if it cannot be read."""
        try:
            resolved = Path(binary).resolve()
            stat = resolved.stat()
        except OSError as exc:
            log.warning("Could not read the generator's identity (%s): %s", binary, exc)
            return None
        return cls(str(resolved), stat.st_size, stat.st_mtime_ns)

    @classmethod
    def from_json(cls, data: object) -> GeneratorIdentity | None:
        """Parse the ledger's "generator" object; None if absent or malformed."""
        if not isinstance(data, dict):
            return None
        path, size, mtime_ns = data.get("path"), data.get("size"), data.get("mtime_ns")
        if isinstance(path, str) and _is_int(size) and _is_int(mtime_ns):
            return cls(path, size, mtime_ns)
        return None

    def to_json(self) -> dict[str, object]:
        """The ledger's "generator" object."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class BackfillLedger:
    """Failed backfill runs per seed with one generator binary (kept in BACKFILL_FAILURES).

    Only the ember failures count toward --max-backfill-attempts; both kinds order the queue
    (plan_seed_queue()).
    """

    ember_failures: dict[str, int] = dataclasses.field(default_factory=dict)
    """Failed ember attempts: generator exit 3, an incomplete ember edition (CORE_ONLY and
    EMBER_FAILED outcomes), an orbit or view that differs from the live package's
    (IDENTITY_MISMATCH), or a backfill render that ran past --timeout (TIMED_OUT)."""
    other_failures: dict[str, int] = dataclasses.field(default_factory=dict)
    """Backfill runs that failed for any other reason (FAILED outcomes: the generator exited 1,
    crashed or was killed, the live package cannot be used, an upload failed)."""
    identity_mismatches: set[str] = dataclasses.field(default_factory=set)
    """Seeds whose regenerated orbit or view differed from the live package's
    (IDENTITY_MISMATCH): given up at once, whatever --max-backfill-attempts says, until the
    generator binary changes."""
    urgent_overruns: dict[str, int] = dataclasses.field(default_factory=dict)
    """Urgent seeds (new mints) whose renders overran and uploaded nothing (TIMED_OUT): their
    place among the urgent seeds (plan_seed_queue()). Never a reason to give a seed up."""

    def given_up(self, seed: str, max_attempts: int) -> bool:
        """True if `seed` is given up with this generator binary: an identity mismatch, or at
        least `max_attempts` failed ember attempts."""
        return seed in self.identity_mismatches or self.ember_failures.get(seed, 0) >= max_attempts

    def failed_runs(self, seed: str) -> int:
        """Every failed backfill run of `seed`, whatever the reason: its place in the queue."""
        return self.ember_failures.get(seed, 0) + self.other_failures.get(seed, 0)

    def retain(self, seeds: Container[str], urgent: Container[str] = ()) -> None:
        """Forget the counts of every seed not in `seeds` (those still waiting for a backfill),
        and the overruns of every seed not in `urgent` (those still waiting for a package)."""
        for counts in (self.ember_failures, self.other_failures):
            for seed in [seed for seed in counts if seed not in seeds]:
                del counts[seed]
        self.identity_mismatches = {seed for seed in self.identity_mismatches if seed in seeds}
        self.urgent_overruns = {
            seed: count for seed, count in self.urgent_overruns.items() if seed in urgent
        }

    def record(
        self, seed: str, outcome: Outcome, *, backfill: bool, interrupted: bool = False
    ) -> bool:
        """Record one processed seed; True if the ledger changed and must be saved.

        COMPLETE clears the seed's counts. CORE_ONLY and EMBER_FAILED count one failed ember
        attempt, for an urgent seed too: a new mint uploaded without its ember edition becomes a
        backfill seed with one attempt. IDENTITY_MISMATCH counts one too, and gives the seed up.
        TIMED_OUT counts one failed ember attempt for a `backfill` seed, and one overrun for an
        urgent seed. FAILED counts one other failure for a `backfill` seed only (every run retries
        urgent seeds anyway). Nothing is counted while the run is `interrupted` (shutting down),
        since the failure may be the signal's doing.
        """
        if outcome is Outcome.COMPLETE:
            cleared = [counts.pop(seed) for counts in self._all_counts() if seed in counts]
            mismatched = seed in self.identity_mismatches
            self.identity_mismatches.discard(seed)
            return bool(cleared) or mismatched
        if outcome is Outcome.FAILED and not backfill:
            return False
        if interrupted:
            log.info("0x%s: failed run not counted (shutting down)", seed)
            return False
        if outcome is Outcome.TIMED_OUT and not backfill:
            counts = self.urgent_overruns
        elif outcome is Outcome.FAILED:
            counts = self.other_failures
        else:
            counts = self.ember_failures
        counts[seed] = counts.get(seed, 0) + 1
        if outcome is Outcome.IDENTITY_MISMATCH:
            self.identity_mismatches.add(seed)
        return True

    def _all_counts(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Every count map."""
        return self.ember_failures, self.other_failures, self.urgent_overruns


# The ledger's list of given-up mismatches, and its name in files written while the check
# compared the orbit only (still read, so an upgrade of run.py alone forgets no seed).
_MISMATCHES_KEY = "identity_mismatches"
_LEGACY_MISMATCHES_KEY = "orbit_mismatches"


def _ledger_counts(data: dict[object, object]) -> dict[str, int]:
    """The positive integer counts of one ledger map (anything else is dropped)."""
    return {str(seed): count for seed, count in data.items() if _is_int(count) and count > 0}


def load_backfill_ledger(
    generator: GeneratorIdentity | None, path: Path = BACKFILL_FAILURES
) -> BackfillLedger:
    """The failure counts of earlier runs with the same generator binary.

    The file is `{"generator": {"path", "size", "mtime_ns"}, "ember_failures": {seed: count},
    "other_failures": {seed: count}, "identity_mismatches": [seed, ...], "urgent_overruns":
    {seed: count}}`. "identity_mismatches" may be absent (files written before it existed);
    "orbit_mismatches", its earlier name, is then read in its place. "urgent_overruns" may be
    absent too. The ledger is empty if the file is absent, unreadable or malformed (logged), or
    was written for another generator binary (the counts reset when the binary changes).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return BackfillLedger()
    except OSError as exc:
        log.warning("Could not read %s (%s): starting with no backfill failure counts", path, exc)
        return BackfillLedger()
    try:
        data = json.loads(text)
    except ValueError:
        log.warning("%s is not valid JSON: starting with no backfill failure counts", path)
        return BackfillLedger()
    keys = ("ember_failures", "other_failures")
    if not isinstance(data, dict) or not all(isinstance(data.get(key), dict) for key in keys):
        log.warning("%s has an unknown layout: starting with no backfill failure counts", path)
        return BackfillLedger()
    if GeneratorIdentity.from_json(data.get("generator")) != generator:
        log.info(
            "The generator changed since %s was written: backfill failure counts reset, so "
            "every ember backfill seed is tried again",
            path,
        )
        return BackfillLedger()
    mismatches = data.get(_MISMATCHES_KEY, data.get(_LEGACY_MISMATCHES_KEY, []))
    overruns = data.get("urgent_overruns", {})
    return BackfillLedger(
        _ledger_counts(data["ember_failures"]),
        _ledger_counts(data["other_failures"]),
        {seed for seed in mismatches if isinstance(seed, str)}
        if isinstance(mismatches, list)
        else set(),
        _ledger_counts(overruns) if isinstance(overruns, dict) else {},
    )


def save_backfill_ledger(
    ledger: BackfillLedger,
    generator: GeneratorIdentity | None,
    path: Path = BACKFILL_FAILURES,
) -> None:
    """Persist the ledger for `generator` (atomically; errors are only logged)."""
    data = {
        "generator": generator.to_json() if generator is not None else None,
        "ember_failures": dict(sorted(ledger.ember_failures.items())),
        "other_failures": dict(sorted(ledger.other_failures.items())),
        _MISMATCHES_KEY: sorted(ledger.identity_mismatches),
        "urgent_overruns": dict(sorted(ledger.urgent_overruns.items())),
    }
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("Could not write %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------


def resolve_generator(generator_arg: str | None) -> list[str] | None:
    """Find the generator binary. Returns argv prefix list or None."""
    candidates = [generator_arg] if generator_arg else list(GENERATOR_CANDIDATES)

    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if not p.is_file():
            continue
        if os.access(p, os.X_OK):
            log.info("Using generator: %s", cand)
            return [cand]
        log.warning("Found %s but it is not executable -- skipping", cand)

    log.warning(
        "Generator not found. Tried: %s",
        ", ".join(c for c in candidates if c),
    )
    return None


def generator_supports_ember(exec_cmd: list[str]) -> bool:
    """Whether the generator knows the ember edition: its `--help` lists GENERATOR_EMBER_FLAG.

    A binary built before the edition exits 0 without any ember file, so every package it makes
    would fail validation against REQUIRED_PACKAGE_FILES. False (logged) if the probe itself
    fails: the caller then treats the binary as predating the edition.
    """
    try:
        result = run_subprocess(
            [*exec_cmd, "--help"], timeout=GENERATOR_PROBE_TIMEOUT, label="generator-help"
        )
    except (subprocess.TimeoutExpired, OSError):
        log.error(
            "Could not run %s --help to check that it supports the ember edition", exec_cmd[0]
        )
        return False
    if result.returncode != 0:
        log.error("%s --help exited with rc=%d", exec_cmd[0], result.returncode)
        return False
    return GENERATOR_EMBER_FLAG in result.stdout or GENERATOR_EMBER_FLAG in result.stderr


def generator_ember_algorithm(exec_cmd: list[str]) -> str | None:
    """The id of the ember look the generator renders (`<generator> --ember-algorithm`).

    The id is its output, stripped, if that is a single EMBER_ALGORITHM_RE id and the probe
    exited 0 (e.g. "ember-v2"). None, with one WARNING, for anything else: a binary that predates
    the flag (its argument parser exits with status 2), another failure or a timeout, or any
    other output. Without the id no live ember edition can be recognised as stale, so the caller
    withdraws or replaces none (held_editions()).
    """
    flag = GENERATOR_EMBER_ALGORITHM_FLAG
    try:
        result = run_subprocess(
            [*exec_cmd, flag], timeout=GENERATOR_PROBE_TIMEOUT, label="generator-ember-algorithm"
        )
    except (subprocess.TimeoutExpired, OSError):
        problem = "could not be run"
    else:
        algorithm = result.stdout.strip()
        if result.returncode == 0 and EMBER_ALGORITHM_RE.fullmatch(algorithm):
            log.debug("The generator renders the ember edition with %s", algorithm)
            return algorithm
        if result.returncode != 0:
            problem = f"exited with rc={result.returncode} ({result.stderr.strip()[:200]})"
        else:
            problem = f"printed {algorithm[:80]!r}, not an ember algorithm id"
    log.warning(
        "%s %s %s: stale ember editions cannot be detected with this generator, so no live "
        "ember edition is withdrawn or replaced",
        exec_cmd[0],
        flag,
        problem,
    )
    return None


def ember_algorithm_number(algorithm: str) -> int:
    """The number of an EMBER_ALGORITHM_RE id ("ember-v2" is 2); ValueError for anything else."""
    match = EMBER_ALGORITHM_RE.fullmatch(algorithm)
    if match is None:
        raise ValueError(f"not an ember algorithm id: {algorithm!r}")
    return int(match.group("number"))


def log_stale_generator(exec_cmd: list[str]) -> None:
    """The ERROR logged on every run while the generator predates the ember edition."""
    log.error(
        "Generator %s predates the ember edition (its --help does not list %s): rebuild the "
        "generator (cargo build --release --locked). Until then packages are checked against "
        "the core files only, so new mints are still uploaded, and the ember backfill is paused.",
        exec_cmd[0],
        GENERATOR_EMBER_FLAG,
    )


# ---------------------------------------------------------------------------
# Generate, upload, cleanup for a single seed
# ---------------------------------------------------------------------------


def generate(exec_cmd: list[str], seed: str, timeout: int) -> Outcome:
    """Run the generator for one seed; the outcome follows its exit status.

    COMPLETE for 0, CORE_ONLY for GENERATOR_EXIT_EMBER_FAILED, TIMED_OUT if it ran past
    `timeout` (it is killed), FAILED for anything else (a kill by a signal included).
    """
    cmd_parts = [*exec_cmd, "--seed", f"0x{seed}", "--output", f"0x{seed}"]
    log.info(
        "GENERATE  seed=0x%s%s",
        seed,
        "  (core package only)" if GENERATOR_EMBER_FLAG in exec_cmd else "",
    )

    try:
        result = run_subprocess(cmd_parts, timeout=timeout, label=f"gen-0x{seed}")
    except subprocess.TimeoutExpired:
        return Outcome.TIMED_OUT
    except OSError:
        return Outcome.FAILED

    if result.returncode == 0:
        return Outcome.COMPLETE
    if result.returncode == GENERATOR_EXIT_EMBER_FAILED:
        log.warning(
            "Generator produced 0x%s WITHOUT the ember edition (rc=%d): its preflight or stage "
            "failed. Check log for full stdout/stderr.",
            seed,
            result.returncode,
        )
        return Outcome.CORE_ONLY
    if result.returncode < 0:
        log.error(
            "Generator for 0x%s was killed by signal %d. Check log for full stdout/stderr.",
            seed,
            -result.returncode,
        )
        return Outcome.FAILED

    log.error(
        "Generator FAILED for 0x%s (rc=%d). Check log for full stdout/stderr.",
        seed,
        result.returncode,
    )
    return Outcome.FAILED


def missing_local_package_parts(
    seed_dir: Path, required_files: tuple[str, ...] = REQUIRED_PACKAGE_FILES
) -> list[str]:
    """Return missing files/groups for a generated local seed package.

    `required_files` lists the single files to check (the spectral bins are always checked):
    every package file by default, CORE_PACKAGE_FILES for a package without the ember edition.
    """
    missing: list[str] = []
    for filename in required_files:
        path = seed_dir / filename
        if not path.is_file():
            missing.append(filename)

    spectral_dir = seed_dir / "spectral"
    spectral_bins: set[int] = set()
    if spectral_dir.is_dir():
        for path in spectral_dir.iterdir():
            if not path.is_file():
                continue
            match = SPECTRAL_FILE_RE.match(path.name)
            if match:
                bin_idx = int(match.group("bin"))
                if bin_idx in EXPECTED_SPECTRAL_BINS:
                    spectral_bins.add(bin_idx)
    else:
        missing.append("spectral/")

    missing_bins = EXPECTED_SPECTRAL_BINS - spectral_bins
    if missing_bins:
        missing.append(f"spectral/*.png ({len(missing_bins)} missing)")

    return missing


def has_ember_edition(seed_dir: Path) -> bool:
    """True if a local package holds every ember edition file."""
    return all((seed_dir / filename).is_file() for filename in EMBER_PACKAGE_FILES)


def remove_ember_files(seed_dir: Path) -> bool:
    """Delete any ember edition file from a package generated without it.

    The generator removes them itself before exiting with GENERATOR_EXIT_EMBER_FAILED; this
    guards the remote state regardless, since a stray (possibly partial) ember file would be
    uploaded and could make the remote package look complete, ending the ember backfill for the
    seed. Returns False if a file cannot be deleted.
    """
    ok = True
    for filename in EMBER_PACKAGE_FILES:
        path = seed_dir / filename
        if not path.exists():
            continue
        log.warning("Removing stray ember file from a package without the ember edition: %s", path)
        try:
            path.unlink()
        except OSError as exc:
            log.error("Could not remove %s: %s", path, exc)
            ok = False
    return ok


def remove_remote_ember_files(
    ssh_host: str, ssh_user: str, remote_dir: str, seed: str, retired: Iterable[str] = ()
) -> bool:
    """Delete the ember edition's files from the remote package of `seed` (absent ones are fine).

    `retired` names further files of the package to delete in the same ssh call: those of a live
    edition of an older look that the current look does not have (retired_ember_files()).

    Run before uploading a package without the ember edition: scp adds and overwrites files but
    never deletes any, so a stale ember file of an earlier upload would stay next to a manifest
    that does not list it, and could make the package look complete. False if the removal failed.
    """
    package = remote_seed_dir(remote_dir, seed)
    filenames = (*EMBER_PACKAGE_FILES, *retired)
    paths = " ".join(shlex.quote(f"{package}/{filename}") for filename in filenames)
    result = run_remote(
        ssh_host, ssh_user, f"rm -f -- {paths}", timeout=30, label=f"ssh-rm-ember-0x{seed}"
    )
    if result is None:
        return False
    if result.returncode != 0:
        log.error(
            "Could not delete the stale ember files of %s (rc=%d): %s",
            package,
            result.returncode,
            result.stderr.strip()[:300],
        )
        return False
    return True


def locked(package: str, command: str) -> str:
    """`command`, run on the asset host under the PACKAGE_LOCK of the remote `package` directory
    (which is created first if it is absent). The lock is released when the command ends; one
    that cannot be taken within PACKAGE_LOCK_WAIT seconds fails the command (flock's status 1)."""
    lock = shlex.quote(posixpath.join(package, PACKAGE_LOCK))
    return (
        f"mkdir -p -- {shlex.quote(package)} && "
        f"flock -w {PACKAGE_LOCK_WAIT} {lock} sh -c {shlex.quote(command)}"
    )


def prepare_remote_dirs(
    ssh_host: str,
    ssh_user: str,
    package: str,
    remote_dirs: Iterable[str],
    remove: Iterable[str] = (),
) -> bool:
    """Create the remote directories (and parents) an upload into `package` writes into; delete
    `remove` first.

    One ssh call, under the package's lock (both steps can run again, so a failed connection is
    retried: SSH_ATTEMPTS). False (logged) if it failed.
    """
    command = "mkdir -p -- " + " ".join(shlex.quote(path) for path in sorted(set(remote_dirs)))
    stale = " ".join(shlex.quote(path) for path in remove)
    if stale:
        command = f"rm -f -- {stale} && {command}"
    result = run_remote(
        ssh_host,
        ssh_user,
        locked(package, command),
        timeout=30,
        label="ssh-prepare",
        attempts=SSH_ATTEMPTS,
    )
    if result is None:
        return False
    if result.returncode == 0:
        return True
    log.error(
        "Could not prepare the remote directories (rc=%d): %s",
        result.returncode,
        result.stderr.strip()[:300],
    )
    return False


@dataclasses.dataclass(frozen=True)
class UploadStep:
    """One step of an upload: local files or directories (copied recursively) into one directory."""

    sources: tuple[Path, ...]
    remote_subdir: str
    """The destination, relative to the remote package directory ("" for the directory itself)."""
    staged: bool = False
    """Upload each source (a file) as `<name>.part`, then rename them all into place, in order,
    once every one has landed. A rename is atomic, so an interrupted upload never leaves a
    truncated file under the real name: used for the metadata, which decides whether a package
    reads as complete and which an ember-mode backfill reads back."""


def full_upload_steps(local_seed_dir: Path) -> list[UploadStep]:
    """The whole package in three steps: the media, then metadata/, then its certificate.

    The metadata files and the certificate are staged, and metadata/assets.json is renamed into
    place after the other metadata files. upload_package() deletes the remote manifest and
    certificate first, so an interrupted upload leaves the package without a manifest: it reads
    as urgent and a later run regenerates and uploads it in full, whatever the backfill mode, so
    no half-replaced file survives. It reads as complete only once the certificate has landed.
    """
    metadata_dir = local_seed_dir / "metadata"
    manifest = local_seed_dir / ASSET_MANIFEST
    certificate = local_seed_dir / EMBER_CERTIFICATE
    media = tuple(sorted(p for p in local_seed_dir.iterdir() if p != metadata_dir))
    metadata = tuple(
        sorted(
            (p for p in metadata_dir.iterdir() if p != certificate),
            key=lambda p: (p == manifest, p.name),
        )
    )
    steps = [UploadStep(media, ""), UploadStep(metadata, "metadata", staged=True)]
    if certificate.is_file():
        steps.append(UploadStep((certificate,), posixpath.dirname(EMBER_CERTIFICATE), staged=True))
    return [step for step in steps if step.sources]


def tree_bytes(path: Path) -> int:
    """The size of a file, or of every file beneath a directory."""
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def scp_timeout(nbytes: int) -> int:
    """The timeout of an scp transfer of `nbytes`: SCP_MIN_TIMEOUT, or longer for large ones."""
    return max(SCP_MIN_TIMEOUT, math.ceil(nbytes / SCP_MIN_BYTES_PER_SECOND))


def scp_transfer(sources: Sequence[Path], remote_target: str, label: str, retries: int) -> bool:
    """Copy `sources` to `remote_target` (user@host:path) with scp; False (logged) on failure.

    Tried up to `retries` times, with a backoff between attempts, but no retry starts once a
    shutdown is requested. Each attempt times out after scp_timeout() of the bytes it carries.
    """
    names = ", ".join(path.name for path in sources)
    try:
        nbytes = sum(tree_bytes(path) for path in sources)
    except OSError as exc:
        log.error("UPLOAD FAILED: %s: cannot read %s: %s", label, names, exc)
        return False
    recursive = ["-r"] if any(path.is_dir() for path in sources) else []
    cmd = ["scp", *ssh_opts(), *recursive, *(str(path) for path in sources), remote_target]
    timeout = scp_timeout(nbytes)

    for attempt in range(1, retries + 1):
        log.info(
            "UPLOAD %s (attempt %d/%d)  %s -> %s  (%.1f MB, timeout %ds)",
            label,
            attempt,
            retries,
            names,
            remote_target,
            nbytes / 1e6,
            timeout,
        )
        try:
            result = run_subprocess(cmd, timeout=timeout, label=f"scp-{label}")
        except (subprocess.TimeoutExpired, OSError):
            pass
        else:
            if result.returncode == 0:
                return True
            log.warning(
                "SCP failed (attempt %d/%d) rc=%d: %s",
                attempt,
                retries,
                result.returncode,
                result.stderr.strip()[:300],
            )
        if attempt == retries or shutdown_requested:
            break
        backoff = 2**attempt
        log.info("Retrying SCP in %ds ...", backoff)
        time.sleep(backoff)

    log.error("UPLOAD FAILED: %s  %s -> %s", label, names, remote_target)
    return False


def rename_remote_files(
    ssh_host: str, ssh_user: str, renames: Sequence[tuple[str, str]], label: str
) -> bool:
    """Rename remote files in order, with one ssh call; False (logged) if it failed.

    `mv -f` within one directory is an atomic rename(2): each target is either its old file (or
    absent) or the complete new one. A failure stops at that rename, so later targets keep their
    old state.
    """
    command = " && ".join(
        f"mv -f -- {shlex.quote(source)} {shlex.quote(target)}" for source, target in renames
    )
    result = run_remote(ssh_host, ssh_user, command, timeout=30, label=f"ssh-mv-{label}")
    if result is not None and result.returncode == 0:
        return True
    log.error(
        "UPLOAD FAILED: %s  could not rename %s into place%s",
        label,
        ", ".join(posixpath.basename(target) for _source, target in renames),
        f" (rc={result.returncode}): {result.stderr.strip()[:300]}" if result is not None else "",
    )
    return False


def upload_steps(
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    local_seed_dir: Path,
    steps: Sequence[UploadStep],
    *,
    remove: Sequence[str] = (),
    retries: int = 2,
) -> bool:
    """Run the upload steps in order; False (logged) on the first failure.

    `remove` lists files, relative to the remote package, that are deleted before the first
    transfer (in the ssh call that creates the destination directories). A failed step stops
    the upload, so later steps (the metadata, the certificate) never land before earlier ones.
    Each scp transfer is retried up to `retries` times (see scp_transfer()).
    """
    remote_package = f"{remote_dir.rstrip('/')}/{local_seed_dir.name}"

    def destination(step: UploadStep) -> str:
        if step.remote_subdir:
            return posixpath.join(remote_package, step.remote_subdir)
        return remote_package

    stale = [posixpath.join(remote_package, path) for path in remove]
    if not prepare_remote_dirs(ssh_host, ssh_user, remote_package, map(destination, steps), stale):
        return False

    for index, step in enumerate(steps, 1):
        label = f"{local_seed_dir.name} [{index}/{len(steps)}]"
        target_dir = destination(step)
        remote_target = f"{ssh_user}@{ssh_host}:{target_dir}/"
        if not step.staged:
            if not scp_transfer(step.sources, remote_target, label, retries):
                return False
            continue
        directories = [path.name for path in step.sources if path.is_dir()]
        if directories:
            log.error("UPLOAD FAILED: %s  cannot stage directories: %s", label, directories)
            return False
        renames: list[tuple[str, str]] = []
        for source in step.sources:
            target = posixpath.join(target_dir, source.name)
            part = f"{target}{PART_SUFFIX}"
            if not scp_transfer((source,), f"{ssh_user}@{ssh_host}:{part}", label, retries):
                return False
            renames.append((part, target))
        if not rename_remote_files(ssh_host, ssh_user, renames, label):
            return False

    log.info("UPLOADED %s -> %s", local_seed_dir.name, remote_package)
    return True


def upload_package(
    ssh_host: str,
    ssh_user: str,
    local_seed_dir: Path,
    remote_dir: str,
    retries: int = 2,
) -> bool:
    """Upload the whole per-seed package (full_upload_steps()); False (logged) on failure.

    The remote metadata/assets.json and metadata/ember.json are deleted first: while the upload
    runs, and after an interrupted one, the package reads as incomplete (urgent), never as a mix
    of old and new files under a complete manifest.
    """
    return upload_steps(
        ssh_host,
        ssh_user,
        remote_dir,
        local_seed_dir,
        full_upload_steps(local_seed_dir),
        remove=(ASSET_MANIFEST, EMBER_CERTIFICATE),
        retries=retries,
    )


def _discard_remote_parts(
    ssh_host: str, ssh_user: str, package: str, parts: Iterable[str], label: str
) -> None:
    """Delete the staged `.part` files of a failed upload into `package`, under its lock, so that
    they do not hold room on the asset host until the package's next attempt (which deletes them
    in any case). Best effort."""
    command = "rm -f -- " + " ".join(shlex.quote(part) for part in parts)
    result = run_remote(
        ssh_host,
        ssh_user,
        locked(package, command),
        timeout=30,
        label=f"ssh-rm-parts-{label}",
        attempts=SSH_ATTEMPTS,
    )
    if result is None or result.returncode != 0:
        log.warning(
            "%s: could not delete the staged %s files of the failed upload; the next upload of "
            "this package deletes them",
            label,
            PART_SUFFIX,
        )


def ember_swap_command(renames: Sequence[tuple[str, str]], stale: Sequence[str]) -> str:
    """The shell command that swaps a staged ember edition in, which can safely run again.

    `renames` pairs each staged file with its destination, in order, the certificate last;
    `stale` lists the live files to delete before the first rename (the live certificate, which
    is the last destination, and the retired files). The deletion runs only while the staged
    certificate is still there, that is until the swap has finished (the certificate is renamed
    last), and each rename only while its staged file is there. So a run that was cut off is
    finished by running the command again, and a run after a complete one changes nothing: it
    never deletes the new certificate.

    Before any deletion or rename, while the staged certificate is there, it checks that the
    staged files form what a cut-off run leaves: a prefix of the renames done (each of those
    staged files gone and its destination present), then every later staged file still there.
    While the live certificate is there, no rename has been done yet (it is deleted first), so
    every staged file must be. Anything else (a staged medium that vanished, say) fails the
    command with status 3 before it changes anything, instead of landing the new manifest and
    certificate over an old or missing file.
    """
    certificate_part = shlex.quote(renames[-1][0])
    certificate = shlex.quote(renames[-1][1])
    # `prefix` is "yes" while the staged files checked so far may be gone because a cut-off run
    # renamed them; the first one still staged ends the prefix.
    checks = [f"prefix=yes; [ -e {certificate} ] && prefix=no"]
    for part, target in renames:
        missing = shlex.quote(f"the staged ember edition is incomplete: {part} is missing")
        checks.append(
            f"if [ -e {shlex.quote(part)} ]; then prefix=no; "
            f'elif [ "$prefix" = no ] || [ ! -e {shlex.quote(target)} ]; then '
            f"echo {missing} >&2; exit 3; fi"
        )
    deletions = " ".join(shlex.quote(path) for path in stale)
    steps = [
        f"if [ -e {certificate_part} ]; then {'; '.join(checks)}; fi",
        f"{{ [ ! -e {certificate_part} ] || rm -f -- {deletions}; }}",
    ]
    for part, target in renames:
        quoted = shlex.quote(part)
        steps.append(f"{{ [ ! -e {quoted} ] || mv -f -- {quoted} {shlex.quote(target)}; }}")
    return " && ".join(steps)


def upload_ember_edition(
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    local_seed_dir: Path,
    *,
    retired: Sequence[str] = (),
    retries: int = 2,
) -> bool:
    """Put the local package's ember edition into its live remote package, in place of any ember
    edition there; False (logged) on failure.

    The edition is staged, then swapped in:
      1. one ssh call creates the destination directories and deletes the `.part` files an
         earlier attempt may have left;
      2. every file is uploaded as `<name>.part` beside its destination, one scp transfer each:
         EMBER_MEDIA_FILES in order, then metadata/assets.json (the caller merged it with the
         live manifest), then metadata/ember.json. No live file changes meanwhile, so a transfer
         that fails (a lost connection, a full asset host) leaves the package, an ember edition
         of an older look included, byte for byte as it was; the staged files are then deleted;
      3. only when all have landed, ONE ssh call swaps the edition in (ember_swap_command()): it
         deletes the live metadata/ember.json and `retired` (the live edition's files that the
         current look does not have, retired_ember_files(); relative to the package), then
         renames each medium into place, then the manifest, then the certificate, last of all.
    Every change of step 3 is an unlink or an atomic rename(2) within one directory, so no file
    is ever truncated under its real name, and the swap refuses to run over a staged file that
    vanished (ember_swap_command()). Staging needs room on the asset host for the new media next
    to the old ones. The ssh calls of steps 1 and 3, and the deletion of the staged files after a
    failure, can all run again, so a failed connection is retried (SSH_ATTEMPTS) before the
    staged files are given up; each holds the package's lock (locked()), so a retry never
    overlaps an earlier attempt that is still running on the asset host.

    What a reader of the asset host can observe:
      * until the swap: the package as it was, with no ember edition or with the complete old
        one, plus `.part` files that no manifest lists;
      * during the swap, which transfers nothing and takes a moment, and from then on if that
        call is cut off: no certificate, and either the old manifest (without ember entries, or
        with the old edition's and their checksums) over media that are each the old edition's
        or already the new one's, or the new manifest over the new media;
      * after it: the complete new edition.
    So a package with a certificate always holds one edition throughout: that certificate's
    manifest entries and media. Without one it reads as a backfill seed, and a later run renders
    it again and repeats the upload from step 1.
    """
    remote_package = f"{remote_dir.rstrip('/')}/{local_seed_dir.name}"
    files = (*EMBER_MEDIA_FILES, ASSET_MANIFEST, EMBER_CERTIFICATE)
    targets = [posixpath.join(remote_package, path) for path in files]
    parts = [f"{target}{PART_SUFFIX}" for target in targets]
    directories = [posixpath.dirname(target) for target in targets]
    if not prepare_remote_dirs(ssh_host, ssh_user, remote_package, directories, parts):
        return False

    name = local_seed_dir.name
    for index, (path, part) in enumerate(zip(files, parts, strict=True), 1):
        label = f"{name} [{index}/{len(files)}]"
        staged = f"{ssh_user}@{ssh_host}:{part}"
        if not scp_transfer((local_seed_dir / path,), staged, label, retries):
            _discard_remote_parts(ssh_host, ssh_user, remote_package, parts, name)
            return False

    stale = [posixpath.join(remote_package, path) for path in (EMBER_CERTIFICATE, *retired)]
    command = locked(
        remote_package, ember_swap_command(list(zip(parts, targets, strict=True)), stale)
    )
    result = run_remote(
        ssh_host, ssh_user, command, timeout=30, label=f"ssh-swap-{name}", attempts=SSH_ATTEMPTS
    )
    if result is None or result.returncode != 0:
        log.error(
            "UPLOAD FAILED: %s  could not swap the staged ember edition into place%s",
            name,
            f" (rc={result.returncode}): {result.stderr.strip()[:300]}"
            if result is not None
            else "",
        )
        _discard_remote_parts(ssh_host, ssh_user, remote_package, parts, name)
        return False
    log.info("UPLOADED the ember edition of %s -> %s", name, remote_package)
    return True


def cleanup_seed_dir(seed: str) -> None:
    """Remove the entire per-seed output directory tree."""
    seed_dir = LOCAL_OUTPUT_DIR / f"0x{seed}"
    if not seed_dir.is_dir():
        return
    try:
        shutil.rmtree(seed_dir)
        log.debug("Cleaned up seed directory: %s", seed_dir)
    except OSError as exc:
        log.warning("Failed to remove %s: %s", seed_dir, exc)


# ---------------------------------------------------------------------------
# Ember-mode backfill: orbit and view check, merged manifest, retired files
# ---------------------------------------------------------------------------


class _Missing:
    """Marks a JSON field that is absent (never equal to anything, itself included)."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def parse_json_exact(text: str) -> object:
    """Parse JSON with every non-integer number as an exact decimal.Decimal."""
    return json.loads(text, parse_float=decimal.Decimal)


def _json_field(data: object, field: tuple[str, ...]) -> object:
    """The value at `field` (a key path) in parsed JSON, or _MISSING."""
    for key in field:
        if not isinstance(data, dict) or key not in data:
            return _MISSING
        data = data[key]
    return data


def _format_json_value(value: object) -> str:
    """A compact rendering of a parsed JSON value for log messages (strings in quotes)."""
    if isinstance(value, list):
        return "[" + ", ".join(_format_json_value(item) for item in value) + "]"
    if isinstance(value, dict):
        members = (
            f"{_format_json_value(str(key))}: {_format_json_value(item)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(members) + "}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def identity_differences(live_traits: object, local_traits: object) -> list[str]:
    """The IDENTITY_FIELDS (orbit, then view) in which two parsed nft_traits.json files differ.

    Parse both with parse_json_exact(), so numbers compare exactly. A field missing on either
    side counts as a difference (a live file must have them all to be used at all, see
    parse_live_package(); a regenerated file that lacks one cannot show that it matches). Each
    entry names the field and both values.
    """
    differences: list[str] = []
    for field in IDENTITY_FIELDS:
        live = _json_field(live_traits, field)
        local = _json_field(local_traits, field)
        if live is _MISSING or local is _MISSING or live != local:
            differences.append(
                f"{'.'.join(field)}: live {_format_json_value(live)}, "
                f"regenerated {_format_json_value(local)}"
            )
    return differences


def is_ember_asset(entry: dict[str, object]) -> bool:
    """True for a manifest entry of the ember edition (its role starts with "ember_")."""
    role = entry.get("role")
    return isinstance(role, str) and role.startswith(EMBER_ROLE_PREFIX)


def _manifest(manifest: object, which: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    """A parsed metadata/assets.json and its "assets" list; ValueError if it is malformed."""
    if not isinstance(manifest, dict):
        raise ValueError(f"the {which} metadata/assets.json is not a JSON object")
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"the {which} metadata/assets.json has no assets list")
    for entry in assets:
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            raise ValueError(f"the {which} metadata/assets.json has an entry without a role")
    return manifest, assets


def merge_ember_manifest(live: object, local: object) -> dict[str, object]:
    """The live metadata/assets.json with the regenerated package's ember entries.

    The live manifest's other entries and top-level fields are kept verbatim and in order; its
    ember_* entries, if any, are replaced by the local ones, appended after them. generated_at is
    the local manifest's. Raises ValueError if a manifest is malformed or the local one lacks
    one of EMBER_MANIFEST_ROLES.
    """
    live_manifest, live_assets = _manifest(live, "live")
    local_manifest, local_assets = _manifest(local, "regenerated")
    generated_at = local_manifest.get("generated_at")
    if not isinstance(generated_at, str):
        raise ValueError("the regenerated metadata/assets.json has no generated_at")
    ember_entries = [entry for entry in local_assets if is_ember_asset(entry)]
    roles = {entry["role"] for entry in ember_entries}
    missing_roles = [role for role in EMBER_MANIFEST_ROLES if role not in roles]
    if missing_roles:
        raise ValueError(
            f"the regenerated metadata/assets.json lacks the roles {', '.join(missing_roles)}"
        )
    merged = dict(live_manifest)
    merged["generated_at"] = generated_at
    merged["assets"] = [
        *(entry for entry in live_assets if not is_ember_asset(entry)),
        *ember_entries,
    ]
    return merged


def without_ember_entries(live: object) -> dict[str, object]:
    """The live metadata/assets.json without its ember entries (a withdrawn ember edition).

    Every other entry and every top-level field, generated_at included, is kept verbatim and in
    order. Raises ValueError if the manifest is malformed.
    """
    live_manifest, live_assets = _manifest(live, "live")
    stripped = dict(live_manifest)
    stripped["assets"] = [entry for entry in live_assets if not is_ember_asset(entry)]
    return stripped


def _is_ember_file_path(path: object) -> TypeGuard[str]:
    """True for a package-relative path that can only name an ember file.

    It is normalised (no "//", no "." component, no trailing "/"), stays inside the package (not
    absolute, no ".." component), and its file name starts with EMBER_FILE_PREFIX, which the name
    of no core or spectral file does.
    """
    return (
        isinstance(path, str)
        and path == posixpath.normpath(path)
        and not posixpath.isabs(path)
        and ".." not in path.split("/")
        and posixpath.basename(path).startswith(EMBER_FILE_PREFIX)
    )


def retired_ember_files(live: object, seed: str) -> list[str]:
    """The files of the live ember edition of `seed` that the current edition does not have.

    They are the paths of the live metadata/assets.json's ember entries that are not in
    EMBER_PACKAGE_FILES: an edition of an older look may hold a file that the current look no
    longer produces. No transfer ever deletes a file, so these are deleted by name (one INFO
    line lists them) by the two operations that read the live manifest anyway: the ember-mode
    backfill that replaces the edition (upload_ember_edition()) and its withdrawal
    (withdraw_ember_edition()). That is not every way an edition goes: a whole-package upload
    (an urgent seed, or --backfill-mode full), the removal before a package without the edition
    is uploaded (generator exit 3), and a withdrawal cut off after its stripped manifest landed
    delete only EMBER_PACKAGE_FILES, and would leave such a file on the asset host, listed by no
    manifest. No look so far has dropped a file (each look's files are a subset of the next one's:
    ember-v3 added the slow film, ember-v4 the medium film); a look that does must extend those
    paths.

    The manifest is data read from the asset host, so only a path that _is_ember_file_path()
    accepts is returned: nothing outside the package, and no core or spectral file, can be
    named for deletion. Any other path of an ember entry is left alone, with a WARNING. Raises
    ValueError if the manifest is malformed.
    """
    _live_manifest, live_assets = _manifest(live, "live")
    listed = [entry["path"] for entry in live_assets if is_ember_asset(entry) and "path" in entry]
    paths = [path for path in listed if path not in EMBER_PACKAGE_FILES]
    retired = sorted({path for path in paths if _is_ember_file_path(path)})
    refused = [path for path in paths if not _is_ember_file_path(path)]
    if refused:
        log.warning(
            "0x%s: the live %s lists ember entries whose paths are not ember files of the "
            "package, so those are left alone: %s",
            seed,
            ASSET_MANIFEST,
            ", ".join(repr(path) for path in refused),
        )
    if retired:
        log.info(
            "0x%s: its live ember edition has %d files that the current edition does not, "
            "which are deleted with it: %s",
            seed,
            len(retired),
            ", ".join(retired),
        )
    return retired


def manifest_json(manifest: dict[str, object]) -> str:
    """The text of a metadata/assets.json that run.py rewrote (merged or stripped)."""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def read_remote_file(ssh_host: str, ssh_user: str, remote_path: str) -> str | None:
    """The text of a remote file, or None (logged) if ssh or the read failed (a failed connection
    is retried: SSH_ATTEMPTS)."""
    command = f"cat -- {shlex.quote(remote_path)}"
    result = run_remote(
        ssh_host, ssh_user, command, timeout=60, label="ssh-cat", attempts=SSH_ATTEMPTS
    )
    if result is None:
        log.error("Could not read %s:%s (ssh did not complete)", ssh_host, remote_path)
        return None
    if result.returncode != 0:
        log.error(
            "Could not read %s:%s (rc=%d): %s",
            ssh_host,
            remote_path,
            result.returncode,
            result.stderr.strip()[:300],
        )
        return None
    return result.stdout


@dataclasses.dataclass(frozen=True)
class LivePackage:
    """The metadata of a published package that an ember-mode backfill reads back."""

    traits: object
    """metadata/nft_traits.json, parsed with parse_json_exact(): it has every IDENTITY_FIELDS
    entry."""
    manifest: dict[str, object]
    """metadata/assets.json: a JSON object whose assets list gives every entry a role."""


def parse_live_package(traits_text: str, manifest_text: str) -> LivePackage:
    """Check the live metadata an ember-mode backfill needs; ValueError names what is wrong."""
    try:
        traits = parse_json_exact(traits_text)
    except ValueError as exc:
        raise ValueError(f"{NFT_TRAITS} is not valid JSON ({exc})") from None
    # Every generator that wrote nft_traits.json wrote all of these (its schema requires them),
    # so a live file without one is damaged: its orbit or view cannot be known, and is not guessed.
    missing = [
        ".".join(field) for field in IDENTITY_FIELDS if _json_field(traits, field) is _MISSING
    ]
    if missing:
        raise ValueError(f"{NFT_TRAITS} lacks {', '.join(missing)}")
    try:
        manifest = json.loads(manifest_text)
    except ValueError as exc:
        raise ValueError(f"{ASSET_MANIFEST} is not valid JSON ({exc})") from None
    live_manifest, _assets = _manifest(manifest, "live")
    return LivePackage(traits, live_manifest)


def read_live_package(
    seed: str, ssh_host: str, ssh_user: str, remote_dir: str
) -> LivePackage | None:
    """The live metadata/nft_traits.json and metadata/assets.json of `seed`, checked.

    None (logged) if ssh or a read failed, or if a live file cannot be used (parse_live_package()).
    An unusable live file needs repair on the asset host: without the orbit and view fields and a
    valid manifest no ember edition can be checked against the package or merged into it. Neither
    case is an ember attempt, so neither gives the seed up; the ember-mode backfill checks the
    live package before rendering too, so an unusable one costs no render.
    """
    package = remote_seed_dir(remote_dir, seed)
    traits_text = read_remote_file(ssh_host, ssh_user, f"{package}/{NFT_TRAITS}")
    if traits_text is None:
        return None
    manifest_text = read_remote_file(ssh_host, ssh_user, f"{package}/{ASSET_MANIFEST}")
    if manifest_text is None:
        return None
    try:
        return parse_live_package(traits_text, manifest_text)
    except ValueError as exc:
        log.error(
            "0x%s: the live package cannot be used for an ember backfill: %s. Nothing is "
            "uploaded until it is repaired on the asset host (restore the file, or delete the "
            "package's %s to have the seed regenerated and uploaded in full).",
            seed,
            exc,
            ASSET_MANIFEST,
        )
        return None


def upload_ember_backfill(
    seed: str, seed_dir: Path, ssh_host: str, ssh_user: str, remote_dir: str
) -> Outcome:
    """Upload only the ember edition of a regenerated package whose core package is live.

    Reads the live package back (read_live_package(), again just before the upload, so the merge
    starts from the current manifest). If the regenerated package shows the same orbit in the
    same view (IDENTITY_FIELDS; the viewing rotation and the frame are not recorded in the live
    package, so they are assumed to match, not verified), stages the ember media, the live
    manifest merged with the new ember entries and metadata/ember.json, and swaps them in with
    one ssh call (upload_ember_edition()). The published main art, spectral files,
    generation.json and nft_traits.json are never touched.

    A live ember edition is replaced in place by that swap (one of an older look that
    --keep-stale-ember kept online, or what a cut-off swap or withdrawal left): its media and
    manifest entries are replaced, its certificate too, and the files it has that the current
    edition does not (retired_ember_files()) are deleted. Until the swap it stays online
    untouched, so a failed upload costs the token nothing.

    Returns COMPLETE once uploaded; IDENTITY_MISMATCH (nothing uploaded) if the regenerated
    package shows another orbit or another view; EMBER_FAILED (nothing uploaded) if its own
    metadata cannot show its orbit and view or give its ember entries; FAILED (nothing uploaded,
    not an ember attempt) if the live package cannot be read or used, or ssh, scp or the local
    disk failed.
    """
    live = read_live_package(seed, ssh_host, ssh_user, remote_dir)
    if live is None:
        return Outcome.FAILED
    try:
        local_traits_text = (seed_dir / NFT_TRAITS).read_text(encoding="utf-8")
        local_manifest_text = (seed_dir / ASSET_MANIFEST).read_text(encoding="utf-8")
    except OSError as exc:
        log.error("0x%s: could not read the regenerated metadata: %s", seed, exc)
        return Outcome.FAILED

    try:
        local_traits = parse_json_exact(local_traits_text)
    except ValueError as exc:
        log.error(
            "0x%s: the regenerated %s is not valid JSON (%s), so its orbit and view cannot be "
            "checked. Nothing is uploaded.",
            seed,
            NFT_TRAITS,
            exc,
        )
        return Outcome.EMBER_FAILED
    differences = identity_differences(live.traits, local_traits)
    if differences:
        log.error(
            "0x%s: the regenerated package shows a DIFFERENT ORBIT OR VIEW than the live one "
            "(%s). Its ember edition would not match the published art, so nothing is uploaded, "
            "and the seed is given up with this generator binary (it would regenerate the same "
            "package). Only --backfill-mode full would upload it, replacing the published "
            "package, main art included.",
            seed,
            "; ".join(differences),
        )
        return Outcome.IDENTITY_MISMATCH

    try:
        merged = merge_ember_manifest(live.manifest, json.loads(local_manifest_text))
    except ValueError as exc:
        log.error(
            "0x%s: cannot merge the regenerated ember entries into the live %s: %s. Nothing is "
            "uploaded.",
            seed,
            ASSET_MANIFEST,
            exc,
        )
        return Outcome.EMBER_FAILED
    try:
        (seed_dir / ASSET_MANIFEST).write_text(manifest_json(merged), encoding="utf-8")
    except OSError as exc:
        log.error("0x%s: could not write the merged %s: %s", seed, ASSET_MANIFEST, exc)
        return Outcome.FAILED

    log.info(
        "0x%s: same orbit and view as the live package, by every recorded field (its viewing "
        "rotation and frame are not recorded, and are assumed to match); uploading only its "
        "ember edition",
        seed,
    )
    retired = retired_ember_files(live.manifest, seed)
    if not upload_ember_edition(ssh_host, ssh_user, remote_dir, seed_dir, retired=retired):
        return Outcome.FAILED
    return Outcome.COMPLETE


# ---------------------------------------------------------------------------
# Stale ember editions: retired when the generator renders a newer look
# ---------------------------------------------------------------------------

# A sed script that prints the id of a certificate's top-level "algorithm". The generator writes
# metadata/ember.json with serde_json's pretty printer, which indents top-level keys by exactly
# two spaces (nested ones by more), so that key's line is `  "algorithm": "<id>",`. A certificate
# in any other layout reads as unreadable, and an unreadable one is never stale.
_CERTIFICATE_ALGORITHM_SED = r's/^  "algorithm": "\([^"]*\)",\{0,1\}$/\1/p'


def list_remote_ember_algorithms(
    ssh_host: str, ssh_user: str, remote_dir: str
) -> dict[str, str | None] | None:
    """The algorithm of every live ember certificate: {seed: its id, or None if unreadable}.

    One ssh call for all packages, in POSIX sh and sed only (the asset host has neither Python
    nor jq): for each 0x<seed>/metadata/ember.json it prints the package and the ids
    _CERTIFICATE_ALGORITHM_SED finds. An algorithm is read only if there is exactly one and it
    matches EMBER_ALGORITHM_RE; anything else (another layout, a truncated or unreadable file, a
    malformed id) is None. Packages without a certificate are absent. Returns None (logged) if ssh
    or the listing failed.
    """
    sed = shlex.quote(_CERTIFICATE_ALGORITHM_SED)
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} || exit 1; for f in 0x*/{EMBER_CERTIFICATE}; do "
        '[ -f "$f" ] || continue; '
        f"""printf '%s\\t%s\\n' "${{f%%/*}}" "$(sed -n {sed} "$f" | tr '\\n' ' ')"; done"""
    )
    result = run_remote(ssh_host, ssh_user, remote_cmd, timeout=60, label="ssh-ember-algorithms")
    if result is None:
        log.error("Could not read the live ember certificates (ssh did not complete)")
        return None
    if result.returncode != 0:
        log.error(
            "Reading the live ember certificates failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip()[:300],
        )
        return None

    algorithms: dict[str, str | None] = {}
    for line in result.stdout.splitlines():
        package, tab, found = line.partition("\t")
        if not tab or not package.startswith("0x"):
            continue
        ids = found.split()
        readable = len(ids) == 1 and EMBER_ALGORITHM_RE.fullmatch(ids[0]) is not None
        algorithms[package.removeprefix("0x")] = ids[0] if readable else None
    return algorithms


def stale_ember_editions(
    seeds: Sequence[str],
    live: dict[str, str | None],
    generator_algorithm: str,
    remote_files: set[str],
) -> dict[str, str]:
    """The seeds whose live ember edition is stale, in `seeds` order: {seed: its algorithm}.

    `live` is list_remote_ember_algorithms()'s result and `remote_files` list_remote_files()'s.
    An edition is stale if its certificate's algorithm is an older id than `generator_algorithm`
    (a lower number): the generator no longer renders that look. Every other edition stays:
      * one whose algorithm cannot be read (a WARNING names it): never guess;
      * a newer one (one WARNING): a rolled-back generator must not take the current look off the
        asset host, and render the old one again;
      * one whose package lacks a core file: that package is regenerated and uploaded in full
        this run, which replaces its ember edition too;
      * one whose seed is not in `seeds` (one WARNING): run.py only regenerates listed seeds, so
        a withdrawn edition would never come back.
    """
    number = ember_algorithm_number(generator_algorithm)
    stale: dict[str, str] = {}
    newer: list[str] = []
    for seed in seeds:
        if seed not in live:
            continue
        algorithm = live[seed]
        if algorithm is None:
            log.warning(
                "0x%s: the algorithm of its live %s cannot be read, so its ember edition is kept "
                "as it is (check that file on the asset host)",
                seed,
                EMBER_CERTIFICATE,
            )
        elif ember_algorithm_number(algorithm) > number:
            newer.append(seed)
        elif ember_algorithm_number(algorithm) < number:
            missing = missing_remote_package_parts(seed, remote_files)
            if missing and not is_ember_backfill(missing):
                log.debug(
                    "0x%s: its %s ember edition goes with its full regeneration", seed, algorithm
                )
            else:
                stale[seed] = algorithm
    if newer:
        log.warning(
            "%d live ember editions are newer than this generator's %s, so they are kept (was "
            "the generator rolled back?): %s",
            len(newer),
            generator_algorithm,
            ", ".join(f"0x{seed}" for seed in newer),
        )
    listed = set(seeds)
    unlisted = sorted(
        seed
        for seed, algorithm in live.items()
        if seed not in listed
        and algorithm is not None
        and ember_algorithm_number(algorithm) < number
    )
    if unlisted:
        log.warning(
            "%d stale ember editions are kept because their seeds are not in the seed list, so "
            "nothing would render them again (remove those packages from the asset host by hand "
            "if they are obsolete): %s",
            len(unlisted),
            ", ".join(f"0x{seed}" for seed in unlisted),
        )
    return stale


def withdraw_ember_edition(seed: str, ssh_host: str, ssh_user: str, remote_dir: str) -> bool:
    """Take the live ember edition of `seed` off the asset host; False (logged) if that failed.

    The live metadata/assets.json is read and checked first: one that cannot be read or used
    leaves the package untouched. Then, in this order:
      1. metadata/ember.json is deleted, so from here on the package reads as a backfill seed,
         however far the withdrawal gets (the backfill then replaces whatever is left);
      2. metadata/assets.json is replaced by the live one without its ember entries
         (without_ember_entries(); uploaded as .part and renamed into place, so it is never left
         truncated), so it no longer lists the files step 3 deletes;
      3. every ember file is deleted (remove_remote_ember_files()), and with them any file of the
         edition that the current look does not have (retired_ember_files()).
    Idempotent: withdrawing a withdrawn edition rewrites the same manifest and deletes nothing.
    """
    package = remote_seed_dir(remote_dir, seed)
    live_text = read_remote_file(ssh_host, ssh_user, f"{package}/{ASSET_MANIFEST}")
    if live_text is None:
        return False
    try:
        live = json.loads(live_text)
        stripped = without_ember_entries(live)
        retired = retired_ember_files(live, seed)
    except ValueError as exc:
        log.error(
            "0x%s: its stale ember edition cannot be withdrawn: the live %s cannot be used (%s). "
            "Nothing is changed until it is repaired on the asset host.",
            seed,
            ASSET_MANIFEST,
            exc,
        )
        return False

    seed_dir = LOCAL_OUTPUT_DIR / f"0x{seed}"
    local_manifest = seed_dir / ASSET_MANIFEST
    step = UploadStep((local_manifest,), posixpath.dirname(ASSET_MANIFEST), staged=True)
    cleanup_seed_dir(seed)  # the directory must hold this manifest only
    try:
        try:
            local_manifest.parent.mkdir(parents=True)
            local_manifest.write_text(manifest_json(stripped), encoding="utf-8")
        except OSError as exc:
            log.error(
                "0x%s: could not write its %s without ember entries: %s", seed, ASSET_MANIFEST, exc
            )
            return False
        if not upload_steps(
            ssh_host, ssh_user, remote_dir, seed_dir, [step], remove=(EMBER_CERTIFICATE,)
        ):
            return False
    finally:
        cleanup_seed_dir(seed)
    return remove_remote_ember_files(ssh_host, ssh_user, remote_dir, seed, retired)


def _withdraw_ember_editions(
    stale: dict[str, str], ssh_host: str, ssh_user: str, remote_dir: str, *, dry_run: bool
) -> tuple[list[str], list[str]]:
    """Withdraw each stale edition ({seed: its algorithm}, in that order): (withdrawn, failed).

    `withdrawn` lists the seeds whose edition is off the asset host (in a dry run: would be),
    `failed` those whose withdrawal failed (withdraw_ember_edition() logged why); the others go
    ahead. A shutdown request stops the loop: the remaining editions wait for a later run.
    """
    withdrawn: list[str] = []
    failed: list[str] = []
    for seed, algorithm in stale.items():
        if shutdown_requested:
            log.info(
                "Shutdown requested -- %d stale ember editions wait for a later run",
                len(stale) - len(withdrawn) - len(failed),
            )
            break
        if dry_run:
            log.info("DRY-RUN  would withdraw the %s ember edition of 0x%s", algorithm, seed)
        elif withdraw_ember_edition(seed, ssh_host, ssh_user, remote_dir):
            log.info(
                "WITHDRAWN  seed=0x%s  its %s ember edition is off the asset host", seed, algorithm
            )
        else:
            failed.append(seed)
            continue
        withdrawn.append(seed)
    return withdrawn, failed


def retire_stale_ember_editions(
    seeds: Sequence[str],
    remote_files: set[str],
    generator_algorithm: str,
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    *,
    keep: bool,
    dry_run: bool,
) -> tuple[set[str], bool] | None:
    """Retire every stale live ember edition (stale_ember_editions()) before the run plans.

    A stale edition shows a look the generator no longer renders, so its package is planned as a
    backfill seed, which the ember backfill renders again in the current look (--max-backfill per
    run, after new mints). Only seeds in `seeds` are touched (the seed list of this run), since
    run.py never regenerates any other package. Until its turn in the backfill, the edition is
      * withdrawn at once (withdraw_ember_edition()), by default: the artist retired the look, so
        it must not stay online next to the current one, and the token has no ember edition
        until it is rendered again;
      * left online, with `keep` (--keep-stale-ember): nothing changes on the asset host here, and
        the backfill replaces the edition in place (upload_ember_backfill(); the whole package
        with --backfill-mode full), so the token is never without an ember edition. An edition
        whose seed the backfill gives up, or whose backfill is paused (--max-backfill 0), stays.

    No edition is retired unless the certificates were read in full (one ssh call), and only
    certificates whose algorithm could be read and is older than `generator_algorithm` count, so
    no failure or unexpected file can start a mass withdrawal. The ssh call is skipped when
    `remote_files` holds no certificate. In a dry run nothing changes on the asset host.

    Returns `remote_files` without the EMBER_PACKAGE_FILES of every retired edition (withdrawn,
    kept for the backfill to replace, or that a dry run would withdraw), so this run's planning
    sees those packages as backfill seeds, and False if a withdrawal failed (logged; the other
    withdrawals go ahead). A later run retries a failed withdrawal, or backfills the package if
    its certificate is already gone. Returns None (logged) if the certificates could not be
    read: the look of every live edition is then unknown, and the caller must not plan a
    package that holds one (held_editions()).
    """
    if not any(path.endswith(f"/{EMBER_CERTIFICATE}") for path in remote_files):
        return remote_files, True
    live = list_remote_ember_algorithms(ssh_host, ssh_user, remote_dir)
    if live is None:
        log.error(
            "Stale ember editions cannot be recognised, so none is %s this run",
            "replaced" if keep else "withdrawn",
        )
        return None
    stale = stale_ember_editions(seeds, live, generator_algorithm, remote_files)
    if not stale:
        return remote_files, True

    looks = ", ".join(sorted(set(stale.values()), key=ember_algorithm_number))
    failed: list[str] = []
    if keep:
        retired = list(stale)
        for seed, algorithm in stale.items():
            log.debug(
                "0x%s: its %s ember edition stays online until it is replaced", seed, algorithm
            )
        log.info(
            "Kept %d stale ember editions online (%s -> %s): the ember backfill replaces each in "
            "place",
            len(retired),
            looks,
            generator_algorithm,
        )
    else:
        retired, failed = _withdraw_ember_editions(
            stale, ssh_host, ssh_user, remote_dir, dry_run=dry_run
        )
        log.info(
            "%s %d stale ember editions (%s -> %s): the ember backfill renders them again",
            "DRY-RUN  would withdraw" if dry_run else "Withdrew",
            len(retired),
            looks,
            generator_algorithm,
        )
    if failed:
        log.error(
            "%d stale ember editions could not be withdrawn (see above; a later run retries each, "
            "or renders it again if its certificate is already gone): %s",
            len(failed),
            ", ".join(f"0x{seed}" for seed in failed),
        )
    gone = {f"0x{seed}/{path}" for seed in retired for path in EMBER_PACKAGE_FILES}
    return remote_files - gone, not failed


def held_editions(backfill: Sequence[str], remote_files: set[str]) -> list[str]:
    """The backfill seeds whose live package holds an ember edition (a certificate), for a run
    that could not tell the look of the live editions (the generator's id or the certificates
    could not be read; logged).

    Such an edition may be of an older look: replacing it would put the new look online next to
    the old one (by default, the old one is withdrawn first), and the run could not withdraw it
    or know that it may be replaced. So the run leaves these seeds out of its backfill, and a
    later run that can read the looks plans them. An edition of an older look lacks a film of the
    current one (an ember-v3 edition the medium film, an ember-v2 edition the slow film too), so
    it would otherwise be planned for its missing files alone.
    """
    held = [seed for seed in backfill if f"0x{seed}/{EMBER_CERTIFICATE}" in remote_files]
    if held:
        log.warning(
            "%d ember backfill seeds hold a live ember edition whose look this run cannot check, "
            "so they wait for a run that can (none of them is replaced or withdrawn): %s",
            len(held),
            ", ".join(f"0x{seed}" for seed in held),
        )
    return held


# ---------------------------------------------------------------------------
# One seed, end to end
# ---------------------------------------------------------------------------


def _generate_core_only(exec_cmd: list[str], seed: str, timeout: int, deadline: float) -> Outcome:
    """After an urgent seed's render ran past `timeout`: render it again without the ember
    edition (GENERATOR_EMBER_FLAG), so the token gets its main art and the edition joins the
    backfill instead of costing a full render on every run.

    CORE_ONLY if that render worked; TIMED_OUT, without starting it (logged), if its own timeout
    (CORE_ONLY_TIMEOUT, or `timeout` if shorter) does not fit before `deadline` (the run budget)
    or a shutdown is requested; otherwise its own outcome (FAILED, or TIMED_OUT).
    """
    core_timeout = min(timeout, CORE_ONLY_TIMEOUT)
    if shutdown_requested or time.monotonic() + core_timeout > deadline:
        log.warning(
            "0x%s: its render ran past --timeout, and %s, so it is not rendered again without "
            "the ember edition this run; the next run renders it again",
            seed,
            "a shutdown is requested" if shutdown_requested else "the run budget has no room",
        )
        return Outcome.TIMED_OUT
    log.warning(
        "0x%s: its render ran past --timeout (%s): rendering it again without the ember edition, "
        "so the token gets its main art and the ember edition joins the backfill",
        seed,
        fmt_duration(timeout),
    )
    cleanup_seed_dir(seed)  # the package must hold this render's output only
    generated = generate([*exec_cmd, GENERATOR_EMBER_FLAG], seed, core_timeout)
    return Outcome.CORE_ONLY if generated is Outcome.COMPLETE else generated


def _generate_and_upload(
    seed: str,
    exec_cmd: list[str],
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    timeout: int,
    backfill: BackfillMode | None,
    ember_capable: bool,
    deadline: float,
) -> Outcome:
    """process_seed() without the dry run, the cleanup and the final log line."""
    # An ember-mode backfill needs the live package's orbit, view and manifest: check them before
    # the render, which takes hours, so a live package that cannot be used costs no render.
    if (
        backfill is BackfillMode.EMBER
        and read_live_package(seed, ssh_host, ssh_user, remote_dir) is None
    ):
        return Outcome.FAILED
    generated = generate(exec_cmd, seed, timeout)
    if generated is Outcome.TIMED_OUT and backfill is None:
        generated = _generate_core_only(exec_cmd, seed, timeout, deadline)
    if generated in (Outcome.FAILED, Outcome.TIMED_OUT):
        return generated
    if backfill is not None and generated is Outcome.CORE_ONLY:
        log.warning(
            "0x%s: its ember edition failed again (generator exit %d). Its core package is "
            "already live, so nothing is uploaded.",
            seed,
            GENERATOR_EXIT_EMBER_FAILED,
        )
        return Outcome.EMBER_FAILED

    seed_dir = LOCAL_OUTPUT_DIR / f"0x{seed}"
    if not seed_dir.is_dir():
        log.error("Package directory NOT FOUND for 0x%s. Tried: %s", seed, seed_dir)
        return Outcome.FAILED
    # A backfill exists for its ember edition, so its package is always checked for it.
    with_ember = generated is Outcome.COMPLETE and (ember_capable or backfill is not None)
    required = REQUIRED_PACKAGE_FILES if with_ember else CORE_PACKAGE_FILES
    missing = missing_local_package_parts(seed_dir, required)
    if missing:
        log.error("Package INCOMPLETE for 0x%s. Missing: %s", seed, ", ".join(missing))
        return Outcome.EMBER_FAILED if is_ember_backfill(missing) else Outcome.FAILED

    if backfill is BackfillMode.EMBER:
        return upload_ember_backfill(seed, seed_dir, ssh_host, ssh_user, remote_dir)

    # A package without the ember edition: exit 3, a render without it after a timeout, or a
    # generator that predates the edition.
    if generated is Outcome.CORE_ONLY or not has_ember_edition(seed_dir):
        if not remove_ember_files(seed_dir):
            return Outcome.FAILED
        if not remove_remote_ember_files(ssh_host, ssh_user, remote_dir, seed):
            log.error("0x%s: stale remote ember files not deleted, so nothing is uploaded", seed)
            return Outcome.FAILED
        if not upload_package(ssh_host, ssh_user, seed_dir, remote_dir):
            return Outcome.FAILED
        return Outcome.CORE_ONLY

    if not upload_package(ssh_host, ssh_user, seed_dir, remote_dir):
        return Outcome.FAILED
    return Outcome.COMPLETE


def process_seed(
    seed: str,
    exec_cmd: list[str] | None,
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    timeout: int,
    dry_run: bool,
    *,
    backfill: BackfillMode | None = None,
    ember_capable: bool = True,
    deadline: float = math.inf,
) -> Outcome:
    """Full pipeline for one seed: generate -> validate package -> upload -> cleanup.

    `backfill` is None for an urgent seed (its remote package lacks a core file: a new mint or a
    broken upload) and the backfill mode for a seed whose remote package lacks only the ember
    edition. `ember_capable` is False for a generator that predates the ember edition
    (generator_supports_ember()): its exit-0 packages are checked against the core files only.
    `deadline` (time.monotonic()) is the end of the run budget: an urgent seed's render without
    the ember edition, after a timeout, starts only if it fits before it.

    Returns:
        COMPLETE: the package was uploaded with its ember edition (an ember-mode backfill: the
            ember edition alone, after the orbit and view check).
        CORE_ONLY: an urgent seed's package was uploaded without the ember edition (generator
            exit 3, a render without the edition after a timeout, or a generator that predates
            the edition), after any stale ember file was deleted from its remote directory.
        EMBER_FAILED: the ember edition failed and nothing was uploaded (a backfill seed's exit
            3, an incomplete local ember edition).
        IDENTITY_MISMATCH: an ember-mode backfill regenerated another orbit or another view
            than the live package's; nothing was uploaded, and the seed is given up with this
            binary.
        TIMED_OUT: the render ran past `timeout` and nothing was uploaded: a backfill seed's,
            or an urgent seed's whose render without the edition could not start or run.
        FAILED: nothing was uploaded for any other reason (for an ember-mode backfill, this
            includes a live package that cannot be read or used, checked before generating).
    The local package is deleted before generating and afterwards, whatever the outcome.
    """
    if dry_run:
        if backfill is BackfillMode.EMBER:
            log.info(
                "DRY-RUN  would regenerate 0x%s and, if its orbit and view match the live "
                "package, upload only its ember edition",
                seed,
            )
        elif backfill is BackfillMode.FULL:
            log.info("DRY-RUN  would regenerate 0x%s and replace the whole remote package", seed)
        else:
            log.info("DRY-RUN  would generate and upload package 0x%s/", seed)
        return Outcome.COMPLETE

    if exec_cmd is None:
        log.error("No generator binary available for 0x%s", seed)
        return Outcome.FAILED

    t0 = time.monotonic()
    cleanup_seed_dir(seed)  # the package must hold this run's output only
    try:
        outcome = _generate_and_upload(
            seed,
            exec_cmd,
            ssh_host,
            ssh_user,
            remote_dir,
            timeout,
            backfill,
            ember_capable,
            deadline,
        )
    finally:
        cleanup_seed_dir(seed)

    elapsed = fmt_duration(time.monotonic() - t0)
    if outcome is Outcome.COMPLETE:
        what = "ember edition uploaded" if backfill is BackfillMode.EMBER else "package uploaded"
        log.info("OK  seed=0x%s  (total %s)  %s", seed, elapsed, what)
    elif outcome is Outcome.CORE_ONLY:
        log.warning("OK WITHOUT EMBER  seed=0x%s  (total %s)  core package uploaded", seed, elapsed)
    elif outcome is Outcome.EMBER_FAILED:
        log.error("EMBER FAILED  seed=0x%s  (total %s)  nothing uploaded", seed, elapsed)
    elif outcome is Outcome.IDENTITY_MISMATCH:
        log.error(
            "IDENTITY MISMATCH  seed=0x%s  (total %s)  nothing uploaded; given up with this binary",
            seed,
            elapsed,
        )
    elif outcome is Outcome.TIMED_OUT:
        log.error(
            "TIMED OUT  seed=0x%s  (total %s)  nothing uploaded%s",
            seed,
            elapsed,
            "; counts as a failed ember attempt" if backfill is not None else "",
        )
    else:
        log.error("FAILURE  seed=0x%s  (total %s)  nothing uploaded", seed, elapsed)
    return outcome


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------


def preflight(
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    api_url: str,
    arbitrum_rpc_url: str,
    nft_contract: str,
    generator_arg: str | None = None,
) -> bool:
    """
    Verify that all external dependencies are working before committing to
    lengthy generation: SSH auth, remote write permissions, seed source
    connectivity, the release generator binary (and that it supports the ember
    edition), and ffmpeg on PATH.
    """
    all_ok = True

    # 1. SSH connectivity
    log.info("[preflight] Testing SSH connection to %s@%s ...", ssh_user, ssh_host)
    try:
        result = run_subprocess(
            [*ssh_cmd(ssh_host, ssh_user), "echo ok"],
            timeout=15,
            label="preflight-ssh",
        )
        if result.returncode == 0 and "ok" in result.stdout:
            log.info("[preflight] SSH connection: OK")
        else:
            log.error("[preflight] SSH connection: FAILED (rc=%d)", result.returncode)
            all_ok = False
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.error("[preflight] SSH connection: FAILED (%s)", exc)
        all_ok = False

    # 2. Remote directory exists and is writable
    log.info("[preflight] Testing write access to %s:%s ...", ssh_host, remote_dir)
    quoted_dir = shlex.quote(remote_dir)
    probe = f"{quoted_dir}/.preflight_probe_{os.getpid()}"
    write_cmd = f"touch {probe} && rm -f {probe}"
    try:
        result = run_subprocess(
            [*ssh_cmd(ssh_host, ssh_user), write_cmd],
            timeout=15,
            label="preflight-write",
        )
        if result.returncode == 0:
            log.info("[preflight] Remote write access: OK")
        else:
            log.error(
                "[preflight] Remote write access: FAILED (rc=%d, stderr=%s)",
                result.returncode,
                result.stderr.strip()[:200],
            )
            all_ok = False
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.error("[preflight] Remote write access: FAILED (%s)", exc)
        all_ok = False

    # 3. Seed source reachability: API is preferred, blockchain is fallback.
    seed_source_ok = False

    log.info("[preflight] Testing API at %s ...", api_url or "(not configured)")
    if api_url:
        url = f"{api_url.rstrip('/')}/api/cosmicgame/cst/list/all/0/1"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                if str(data.get("status", 0)) == "1":
                    log.info("[preflight] API connectivity: OK")
                    seed_source_ok = True
                else:
                    log.warning(
                        "[preflight] API connectivity: FAILED (status=%s)",
                        data.get("status"),
                    )
        except Exception as exc:
            log.warning("[preflight] API connectivity: FAILED (%s)", exc)
    else:
        log.warning("[preflight] API connectivity: SKIPPED (not configured)")

    log.info(
        "[preflight] Testing blockchain seed source via %s ...",
        safe_url_for_log(arbitrum_rpc_url),
    )
    try:
        contract = normalize_eth_address(nft_contract)
        total_supply = eth_call_uint256(arbitrum_rpc_url, contract, SELECTOR_TOTAL_SUPPLY)
        if total_supply > 0:
            token_id = eth_call_uint256(
                arbitrum_rpc_url,
                contract,
                f"{SELECTOR_TOKEN_BY_INDEX}{encode_uint256_arg(0)}",
            )
            seed_value = eth_call_uint256(
                arbitrum_rpc_url,
                contract,
                f"{SELECTOR_GET_NFT_SEED}{encode_uint256_arg(token_id)}",
            )
            log.info(
                "[preflight] Blockchain seed source: OK "
                "(totalSupply=%d, first token=%d, first seed=0x%s)",
                total_supply,
                token_id,
                normalize_seed(seed_value),
            )
            seed_source_ok = True
        else:
            log.warning("[preflight] Blockchain seed source: FAILED (totalSupply=0)")
    except Exception as exc:
        log.warning("[preflight] Blockchain seed source: FAILED (%s)", exc)

    if not seed_source_ok:
        log.error("[preflight] No seed source is reachable")
        all_ok = False

    # 4. Generator binary, and that it supports the ember edition
    generator = resolve_generator(generator_arg)
    if generator is None:
        log.error("[preflight] Generator binary: NOT FOUND")
        all_ok = False
    elif generator_supports_ember(generator):
        # A binary without the ember algorithm probe still works (with a WARNING): it only
        # cannot recognise stale ember editions.
        algorithm = generator_ember_algorithm(generator)
        log.info(
            "[preflight] Generator binary: OK (%s, ember edition supported%s)",
            generator[0],
            f", renders {algorithm}" if algorithm else "",
        )
    else:
        log.error(
            "[preflight] Generator binary: FAILED (%s predates the ember edition: its --help "
            "does not list %s; rebuild the generator)",
            generator[0],
            GENERATOR_EMBER_FLAG,
        )
        all_ok = False

    # 5. FFmpeg (required by the Rust generator for MP4 encoding)
    if shutil.which("ffmpeg"):
        log.info("[preflight] ffmpeg: OK")
    else:
        log.error("[preflight] ffmpeg: NOT FOUND (required for MP4 encoding)")
        all_ok = False

    if all_ok:
        log.info("[preflight] All checks passed.")
    else:
        log.error("[preflight] Some checks FAILED. Fix the issues above before running.")

    return all_ok


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def non_negative_int(value: str) -> int:
    """argparse type: an integer >= 0."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"{number} is negative")
    return number


def positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    number = non_negative_int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"{number} is not positive")
    return number


def run_timeout(value: str) -> int:
    """argparse type: a per-seed timeout in seconds, from 1 to MAX_TIMEOUT."""
    number = positive_int(value)
    if number > MAX_TIMEOUT:
        raise argparse.ArgumentTypeError(
            f"{number} is more than {MAX_TIMEOUT} seconds: a run must be able to start a seed with "
            f"that timeout and still have room for its render without the ember edition "
            f"({CORE_ONLY_TIMEOUT} s) and its upload ({UPLOAD_MARGIN} s) within the sync unit's "
            f"{RUN_CEILING}-second limit"
        )
    return number


def backfill_mode(value: str) -> BackfillMode:
    """argparse type: a BackfillMode by its value ("ember" or "full")."""
    try:
        return BackfillMode(value.strip().lower())
    except ValueError:
        choices = ", ".join(mode.value for mode in BackfillMode)
        raise argparse.ArgumentTypeError(f"{value!r} is not one of {choices}") from None


def yes_no(value: str) -> bool:
    """argparse type: a switch, "yes", "true", "on" or "1", or "no", "false", "off", "0" or ""."""
    text = value.strip().lower()
    if text in ("yes", "true", "on", "1"):
        return True
    if text in ("no", "false", "off", "0", ""):
        return False
    raise argparse.ArgumentTypeError(f"{value!r} is not yes or no")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (`argv`, default sys.argv[1:]) with .env/environment defaults."""
    p = argparse.ArgumentParser(
        description="CosmicSignature NFT asset checker and uploader",
        epilog=(
            "Configuration is read from .env file, environment variables, and CLI args "
            "(in increasing priority). See .env.example for all available settings."
        ),
    )
    p.add_argument(
        "--ssh-host",
        default=os.environ.get(ENV_SSH_HOST),
        help=f"Remote SSH host (env: {ENV_SSH_HOST})",
    )
    p.add_argument(
        "--ssh-user",
        default=os.environ.get(ENV_SSH_USER),
        help=f"Remote SSH user (env: {ENV_SSH_USER})",
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get(ENV_API_URL, ""),
        help=f"CosmicGame API base URL (env: {ENV_API_URL}; optional with blockchain fallback)",
    )
    p.add_argument(
        "--remote-dir",
        default=os.environ.get(ENV_REMOTE_DIR),
        help=f"Remote asset directory (env: {ENV_REMOTE_DIR})",
    )
    p.add_argument(
        "--arbitrum-rpc-url",
        default=os.environ.get(ENV_ARBITRUM_RPC_URL, DEFAULT_ARBITRUM_RPC_URL),
        help=f"Arbitrum JSON-RPC URL (env: {ENV_ARBITRUM_RPC_URL})",
    )
    p.add_argument(
        "--nft-contract",
        default=os.environ.get(ENV_NFT_CONTRACT, DEFAULT_NFT_CONTRACT),
        help=f"Cosmic Signature NFT contract address (env: {ENV_NFT_CONTRACT})",
    )
    p.add_argument(
        "--generator", default=None, help="Path to generator binary (auto-detected if omitted)"
    )
    p.add_argument(
        "--timeout",
        type=run_timeout,
        default=DEFAULT_TIMEOUT,
        help=(
            f"Per-seed generator timeout in seconds, at most {MAX_TIMEOUT} "
            f"(default: {DEFAULT_TIMEOUT})"
        ),
    )
    p.add_argument(
        "--max-backfill",
        type=non_negative_int,
        default=os.environ.get(ENV_MAX_BACKFILL, str(DEFAULT_MAX_BACKFILL)),
        help=(
            "Packages missing only the ember edition (or holding a stale one) to regenerate "
            "per run, after every new or incomplete package; 0 pauses the backfill "
            f"(env: {ENV_MAX_BACKFILL}; default: {DEFAULT_MAX_BACKFILL})"
        ),
    )
    p.add_argument(
        "--backfill-mode",
        type=backfill_mode,
        metavar="{ember,full}",
        default=os.environ.get(ENV_BACKFILL_MODE, DEFAULT_BACKFILL_MODE.value),
        help=(
            "How a backfilled package is uploaded: 'ember' uploads only its ember edition "
            "(and a merged metadata/assets.json), after checking that the render shows the "
            "same orbit in the same view as the live package; 'full' replaces the whole remote "
            "package, main art included "
            f"(env: {ENV_BACKFILL_MODE}; default: {DEFAULT_BACKFILL_MODE.value})"
        ),
    )
    p.add_argument(
        "--max-backfill-attempts",
        type=positive_int,
        default=os.environ.get(ENV_MAX_BACKFILL_ATTEMPTS, str(MAX_BACKFILL_ATTEMPTS)),
        help=(
            "Failed ember attempts after which a backfill seed is given up until the generator "
            f"binary changes (env: {ENV_MAX_BACKFILL_ATTEMPTS}; default: {MAX_BACKFILL_ATTEMPTS})"
        ),
    )
    p.add_argument(
        "--keep-stale-ember",
        type=yes_no,
        nargs="?",
        const=True,
        metavar="{yes,no}",
        default=os.environ.get(ENV_KEEP_STALE_EMBER, "no"),
        help=(
            "Do not withdraw live ember editions that an older ember algorithm than the "
            "generator's rendered: each stays online until the backfill replaces it in place "
            "with the current look (--max-backfill per run), so no token is without an ember "
            "edition meanwhile; with --max-backfill 0 as well, every live edition is held as "
            "it is. By default each run withdraws them from the asset host at once, and a "
            "token has no ember edition until the backfill has rendered it again. The flag "
            f"alone means yes (env: {ENV_KEEP_STALE_EMBER}; default: no)"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report missing files without generating or uploading",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Verify SSH, remote write, seed sources, the release generator binary (and that it "
            "supports the ember edition), and ffmpeg, then exit"
        ),
    )
    return p.parse_args(argv)


def validate_config(args: argparse.Namespace) -> list[str]:
    """Return list of missing required config values."""
    missing = []
    for attr, env_name in [
        ("ssh_host", ENV_SSH_HOST),
        ("ssh_user", ENV_SSH_USER),
        ("remote_dir", ENV_REMOTE_DIR),
        ("arbitrum_rpc_url", ENV_ARBITRUM_RPC_URL),
        ("nft_contract", ENV_NFT_CONTRACT),
    ]:
        if not getattr(args, attr, None):
            missing.append(f"  --{attr.replace('_', '-')}  (or env {env_name})")
    return missing


def main(argv: Sequence[str] | None = None) -> int:
    """One sync run under the single-instance lock; the exit status.

    Returns 1 without doing anything if another run holds the lock (RUN_LOCK), else sync()'s
    status. --help exits during argument parsing, before the lock is taken.
    """
    load_dotenv()
    args = parse_args(argv)
    setup_logging()
    try:
        lock_fd = acquire_run_lock()
    except OSError as exc:
        log.error("Cannot take the single-instance lock %s: %s", RUN_LOCK.resolve(), exc)
        return 1
    if lock_fd is None:
        log.error(
            "Another run holds the single-instance lock %s (pid %s): exiting without doing "
            "anything",
            RUN_LOCK.resolve(),
            run_lock_holder(),
        )
        return 1
    try:
        install_signal_handlers()
        return sync(args)
    finally:
        os.close(lock_fd)


def sync(args: argparse.Namespace) -> int:
    """One sync run: retire stale ember editions, plan the incomplete packages, generate and
    upload them; the exit status.

    Returns 0 if every seed it processed succeeded (an urgent package uploaded without its ember
    edition counts as a success; seeds left to the next run by the run budget do not count) and
    1 on a configuration error, a failed seed list or remote listing, any failed seed (a failed
    ember backfill included), or live ember certificates that could not be read or a stale ember
    edition that could not be withdrawn.
    """
    missing_cfg = validate_config(args)
    if missing_cfg:
        log.error("Missing required configuration:\n%s", "\n".join(missing_cfg))
        log.error("Set them in .env, environment variables, or CLI args. See .env.example.")
        return 1

    log.info("=" * 60)
    log.info("CosmicSignature asset sync started")
    log.info("  ssh_host              = %s", args.ssh_host)
    log.info("  ssh_user              = %s", args.ssh_user)
    log.info("  api_url               = %s", args.api_url or "(not set)")
    log.info("  remote_dir            = %s", args.remote_dir)
    log.info("  arbitrum_rpc_url      = %s", safe_url_for_log(args.arbitrum_rpc_url))
    log.info("  nft_contract          = %s", args.nft_contract)
    log.info("  timeout               = %s", fmt_duration(args.timeout))
    log.info("  max_backfill          = %d", args.max_backfill)
    log.info("  backfill_mode         = %s", args.backfill_mode.value)
    log.info("  max_backfill_attempts = %d", args.max_backfill_attempts)
    log.info("  keep_stale_ember      = %s", args.keep_stale_ember)
    log.info("  dry_run               = %s", args.dry_run)
    log.info("  preflight             = %s", args.preflight)
    log.info("=" * 60)

    if args.preflight:
        ok = preflight(
            args.ssh_host,
            args.ssh_user,
            args.remote_dir,
            args.api_url,
            args.arbitrum_rpc_url,
            args.nft_contract,
            args.generator,
        )
        return 0 if ok else 1

    t_start = time.monotonic()

    exec_cmd = resolve_generator(args.generator)
    if exec_cmd is None and not args.dry_run:
        log.error("No generator available and not in dry-run mode. Exiting.")
        return 1

    ember_capable = True
    generator_identity: GeneratorIdentity | None = None
    max_backfill: int = args.max_backfill
    # The ember look the generator renders; while it is unknown (None) no live ember edition
    # can be recognised as stale, so none is withdrawn or planned for replacement.
    ember_algorithm: str | None = None
    if exec_cmd is not None:
        generator_identity = GeneratorIdentity.of(exec_cmd[0])
        ember_capable = generator_supports_ember(exec_cmd)
        if ember_capable:
            ember_algorithm = generator_ember_algorithm(exec_cmd)
        else:
            log_stale_generator(exec_cmd)
            max_backfill = 0

    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Phase 1: discover what's missing ---

    try:
        seeds, seed_source = resolve_token_seeds(
            args.api_url,
            args.arbitrum_rpc_url,
            args.nft_contract,
        )
    except RuntimeError as exc:
        log.error("Fatal: %s", exc)
        return 1

    remote_files = list_remote_files(args.ssh_host, args.ssh_user, args.remote_dir)
    if remote_files is None:
        log.error("Cannot tell which packages are incomplete without the remote listing. Exiting.")
        return 1
    # Stale editions are withdrawn or, with --keep-stale-ember, left online to be replaced in
    # place. Either way the listing no longer holds their files, so the plan below counts their
    # packages as backfill seeds. While the looks of the live editions are unknown, no package
    # that holds one is planned (held_editions()).
    retirement_ok = True
    looks_known = False
    if ember_algorithm is not None:
        retired = retire_stale_ember_editions(
            seeds,
            remote_files,
            ember_algorithm,
            args.ssh_host,
            args.ssh_user,
            args.remote_dir,
            keep=args.keep_stale_ember,
            dry_run=args.dry_run,
        )
        looks_known = retired is not None
        if retired is None:
            retirement_ok = False
        else:
            remote_files, retirement_ok = retired
    urgent, backfill = find_missing_seeds(seeds, remote_files)
    incomplete = len(urgent) + len(backfill)
    backfill_seeds = set(backfill)
    held = held_editions(backfill, remote_files) if ember_capable and not looks_known else []

    if incomplete == 0:
        elapsed = time.monotonic() - t_start
        log.info(
            "All %d tokens have complete asset packages on remote. Nothing to do. (%s)",
            len(seeds),
            fmt_duration(elapsed),
        )
        return 0 if retirement_ok else 1

    log.info(
        "Found %d seeds with incomplete asset packages (out of %d total): %d new or incomplete, "
        "%d missing only the ember edition",
        incomplete,
        len(seeds),
        len(urgent),
        len(backfill),
    )

    # Only seeds still waiting for the backfill keep their failure counts, and only new mints
    # still waiting for a package their overruns.
    ledger = load_backfill_ledger(generator_identity)
    ledger.retain(backfill_seeds, urgent=set(urgent))
    for seed in urgent:
        if seed in ledger.urgent_overruns:
            log.warning(
                "0x%s: its render ran past --timeout in %d earlier runs without a package to "
                "upload, so it goes behind the other new mints",
                seed,
                ledger.urgent_overruns[seed],
            )
    given_up = given_up_seeds(backfill, ledger, args.max_backfill_attempts)
    for seed in given_up:
        if seed in ledger.identity_mismatches:
            log.warning(
                "0x%s: ember backfill given up: this generator binary regenerates a different "
                "orbit or view than the live package (rebuild the generator, or delete the seed "
                "from %s in %s, to try again; --backfill-mode full would replace the whole "
                "published package)",
                seed,
                _MISMATCHES_KEY,
                BACKFILL_FAILURES,
            )
        else:
            log.warning(
                "0x%s: ember backfill given up after %d attempts with this generator binary "
                "(rebuild the generator, or delete the seed from ember_failures in %s, to try "
                "again)",
                seed,
                ledger.ember_failures[seed],
                BACKFILL_FAILURES,
            )
    plannable = [seed for seed in backfill if seed not in held]
    missing = plan_seed_queue(urgent, plannable, max_backfill, ledger, args.max_backfill_attempts)
    deferred = incomplete - len(given_up) - len(missing)
    if deferred:
        log.info(
            "This run generates %d of them; %d ember backfill seeds wait for later runs "
            "(--max-backfill %d%s)",
            len(missing),
            deferred,
            max_backfill,
            "; paused: the generator predates the ember edition" if not ember_capable else "",
        )
    if not missing:
        return 0 if retirement_ok else 1

    # --- Phase 2: generate and upload sequentially ---

    ok_count = 0
    fail_count = 0
    failed_seeds: list[str] = []
    core_only_seeds: list[str] = []
    ember_failed_seeds: list[str] = []
    # The run budget: a seed starts only if its render can time out and still be uploaded before
    # systemd's ceiling (RUN_CEILING); the rest waits for the next run.
    deadline = t_start + RUN_CEILING - UPLOAD_MARGIN
    budget_left = 0
    budget_stalled = False

    for i, seed in enumerate(missing, 1):
        if shutdown_requested:
            remaining = len(missing) - i + 1
            log.info("Shutdown requested -- skipping remaining %d seeds", remaining)
            break
        if time.monotonic() + args.timeout > deadline:
            budget_left = len(missing) - i + 1
            if i == 1:
                # MAX_TIMEOUT leaves room for the first seed, unless planning took hours.
                log.error(
                    "Run budget reached before the first seed, after %s of planning: nothing is "
                    "generated",
                    fmt_duration(time.monotonic() - t_start),
                )
                budget_stalled = True
            log.info(
                "Run budget reached; %d seeds wait for the next run (after %s, a seed's --timeout "
                "of %s and %s for its upload would not fit in this run's %s)",
                budget_left,
                fmt_duration(time.monotonic() - t_start),
                fmt_duration(args.timeout),
                fmt_duration(UPLOAD_MARGIN),
                fmt_duration(RUN_CEILING),
            )
            break

        is_backfill = seed in backfill_seeds
        log.info(
            "[%d/%d] seed=0x%s%s",
            i,
            len(missing),
            seed,
            f"  (ember backfill, {args.backfill_mode.value} mode)" if is_backfill else "",
        )

        outcome = process_seed(
            seed,
            exec_cmd,
            args.ssh_host,
            args.ssh_user,
            args.remote_dir,
            args.timeout,
            args.dry_run,
            backfill=args.backfill_mode if is_backfill else None,
            ember_capable=ember_capable,
            deadline=deadline,
        )
        if outcome in (
            Outcome.FAILED,
            Outcome.TIMED_OUT,
            Outcome.EMBER_FAILED,
            Outcome.IDENTITY_MISMATCH,
        ):
            fail_count += 1
            failed_seeds.append(seed)
            if outcome is not Outcome.FAILED and (is_backfill or outcome is not Outcome.TIMED_OUT):
                ember_failed_seeds.append(seed)
        else:
            ok_count += 1
            if outcome is Outcome.CORE_ONLY:
                core_only_seeds.append(seed)
        # A generator that predates the ember edition makes no ember attempts: its packages
        # lack the edition by design, so the ledger is left alone until the rebuild.
        if (
            not args.dry_run
            and ember_capable
            and ledger.record(seed, outcome, backfill=is_backfill, interrupted=shutdown_requested)
        ):
            save_backfill_ledger(ledger, generator_identity)

    # --- Summary ---

    skipped = len(missing) - ok_count - fail_count - budget_left
    elapsed = time.monotonic() - t_start

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Seed source          : %s", seed_source)
    log.info("  Total seeds          : %d", len(seeds))
    log.info("  Missing on remote    : %d", incomplete)
    log.info("  Planned this run     : %d", len(missing))
    if deferred:
        log.info("  Backfill deferred    : %d", deferred)
    if given_up:
        log.warning("  Ember given up       : %d", len(given_up))
    log.info("  Processed OK         : %d", ok_count)
    if core_only_seeds:
        log.warning(
            "  ...without ember     : %d (%s)",
            len(core_only_seeds),
            ", ".join(f"0x{s}" for s in core_only_seeds),
        )
    log.info("  Failed               : %d", fail_count)
    if ember_failed_seeds:
        log.warning(
            "  ...ember failed      : %d (%s)",
            len(ember_failed_seeds),
            ", ".join(f"0x{s}" for s in ember_failed_seeds),
        )
    if budget_left:
        log.info("  Next run (budget)    : %d", budget_left)
    if skipped > 0:
        log.info("  Skipped (shutdown)   : %d", skipped)
    if failed_seeds:
        log.info("  Failed seeds         : %s", ", ".join(f"0x{s}" for s in failed_seeds))
    log.info("  Wall time            : %s", fmt_duration(elapsed))
    log.info("=" * 60)

    return 1 if fail_count > 0 or not retirement_ok or budget_stalled else 0


if __name__ == "__main__":
    sys.exit(main())
