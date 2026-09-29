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
    the same orbit as the live package, and uploads only the ember files and a merged
    metadata/assets.json; the published main art, spectral files, generation.json and
    nft_traits.json are never touched.
    --backfill-mode full replaces the whole remote package. A seed whose ember edition fails
    --max-backfill-attempts times with the same generator binary is given up
    (backfill_failures.json). A seed whose regenerated orbit differs from the live package's is
    given up at once: the same binary always regenerates the same orbit. A backfill run that
    fails for any other reason is not counted toward that cap, but moves the seed behind the
    seeds that have failed less often, so a seed that always fails cannot stall the backfill.

Uploads: the metadata files and the certificate are uploaded under temporary names and renamed
    into place, so an interrupted upload never leaves a truncated metadata file; a whole package
    first loses its remote metadata/assets.json, so an interrupted one reads as incomplete (and
    is regenerated in full) until its manifest has landed.

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

# Per-seed generator timeout. A package takes about an hour on the production host. This must stay
# well below the service's 24-hour TimeoutStartSec: a render that hangs then fails and moves back
# in the queue, instead of using up every run until systemd stops it (a stopped run counts no
# failure, so the same seed would come first again).
DEFAULT_TIMEOUT = 4 * 3600  # 4 hours
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

SSH_BASE_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
]

# An scp transfer may take SCP_MIN_TIMEOUT seconds, or longer when it carries more than
# SCP_MIN_TIMEOUT * SCP_MIN_BYTES_PER_SECOND bytes: a package with the ember edition is about
# 0.4 GB larger than one without it (the HQ ember video alone is about 284 MB).
SCP_MIN_TIMEOUT = 900
SCP_MIN_BYTES_PER_SECOND = 1_000_000

# A staged upload (UploadStep.staged) writes each file as <name>.part and then renames it.
PART_SUFFIX = ".part"

# The generator capability probe: `<generator> --help` must list this flag, which the ember
# edition introduced (tests/cli.rs pins it). A binary without it predates the edition.
GENERATOR_EMBER_FLAG = "--no-ember"
GENERATOR_PROBE_TIMEOUT = 30


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
# The ember edition (sumi and vermilion on kozo) and its determinism certificate. Packages
# generated before it existed, or whose ember edition failed, lack only these files: they are
# regenerated as a backfill that yields to new mints (see find_missing_seeds, plan_seed_queue and
# --backfill-mode). Keep in sync with app::EMBER_OUTPUT_PATHS (a Rust unit test checks it).
EMBER_PACKAGE_FILES = (
    "images/source/ember.png",
    "images/web/ember_full.webp",
    "images/web/ember_preview.webp",
    "videos/web/ember.mp4",
    "videos/hq/ember.mp4",
    "metadata/ember.json",
)
REQUIRED_PACKAGE_FILES = CORE_PACKAGE_FILES + EMBER_PACKAGE_FILES
# The ember edition's media: every ember file except its certificate, which is uploaded last.
EMBER_MEDIA_FILES = tuple(path for path in EMBER_PACKAGE_FILES if path != EMBER_CERTIFICATE)

# The roles of the ember entries in metadata/assets.json (every role of the edition starts with
# EMBER_ROLE_PREFIX; the certificate has no manifest entry). A full package lists all five.
EMBER_ROLE_PREFIX = "ember_"
EMBER_MANIFEST_ROLES = (
    "ember_source_master",
    "ember_web_full",
    "ember_web_preview",
    "ember_web",
    "ember_hq",
)

# The metadata/nft_traits.json fields that identify the selected orbit. An ember-mode backfill
# uploads the regenerated ember edition only if all three equal the live package's (compared as
# exact JSON numbers), so the edition always draws the orbit of the published main art.
ORBIT_IDENTITY_FIELDS = (
    ("simulation", "masses"),
    ("generation", "borda", "selected_index"),
    ("generation", "borda", "retry_count"),
)

# Generator exit status for a package that is complete except for the ember edition: its
# preflight or its stage failed, the generator removed every ember file, and it wrote the rest
# of the package (metadata included) exactly as with --no-ember. A new mint's core package is
# uploaded, so the token gets its artwork and traits; an ember backfill seed uploads nothing.
# Either way the seed counts one failed ember attempt in the backfill failure ledger.
GENERATOR_EXIT_EMBER_FAILED = 3

# Backfill seeds (packages missing only the ember edition) generated per run. Each run first
# generates every seed missing a core file (new mints), so a mint waits for at most this many
# backfill packages, each a full render taking about an hour (61 min measured on the production
# host), plus the timer's restart delay.
DEFAULT_MAX_BACKFILL = 1

# Failed ember attempts after which a backfill seed is given up. The count is kept per generator
# binary (see GeneratorIdentity) and resets when the binary changes, so a rebuilt generator
# retries every seed; until then a given-up seed is logged as a WARNING on every run. Backfill
# runs that fail for another reason are counted separately (BackfillLedger.other_failures): they
# only order the queue and never give a seed up. An orbit mismatch does not wait for the cap: it
# gives the seed up at once (Outcome.ORBIT_MISMATCH).
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

# Minimal ABI selectors for the verified Cosmic Signature NFT contract.
SELECTOR_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()
SELECTOR_TOKEN_BY_INDEX = "0x4f6ccce7"  # tokenByIndex(uint256)
SELECTOR_GET_NFT_SEED = "0xb0c0fe4e"  # getNftSeed(uint256)


class BackfillMode(enum.Enum):
    """How a backfill seed (a live package that lacks only the ember edition) is uploaded."""

    EMBER = "ember"
    """Non-destructive (the default): the package is regenerated in full locally, but only its
    ember edition (the six ember files and a merged metadata/assets.json) is uploaded, and only
    if the render shows the same orbit as the live package. The published main art, spectral
    files, generation.json and nft_traits.json are never touched."""
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
    ssh_host: str, ssh_user: str, command: str, *, timeout: int, label: str
) -> subprocess.CompletedProcess[str] | None:
    """Run a shell command on the remote host.

    Returns None if ssh timed out or could not be started (already logged); any exit status is
    returned to the caller, which knows what it means (255 is an ssh error).
    """
    try:
        return run_subprocess([*ssh_cmd(ssh_host, ssh_user), command], timeout=timeout, label=label)
    except (subprocess.TimeoutExpired, OSError):
        return None


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

    files = {line.strip().removeprefix("./") for line in result.stdout.splitlines() if line.strip()}
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

    Such a package predates the edition, or was uploaded after its ember edition failed (exit
    3). Also used for a local package, whose missing parts then show a failed ember edition.
    """
    return bool(missing_parts) and all(part in EMBER_PACKAGE_FILES for part in missing_parts)


def find_missing_seeds(seeds: list[str], remote_files: set[str]) -> tuple[list[str], list[str]]:
    """Split the API seeds whose remote package is incomplete into (urgent, backfill).

    Urgent seeds lack a core file: new mints and broken uploads. Backfill seeds lack only the
    ember edition's files: their packages predate the edition, or were uploaded after their ember
    edition failed. Both lists keep API order.
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
    edition failed `max_attempts` or more times, and those whose orbit did not match."""
    return [seed for seed in backfill if ledger.given_up(seed, max_attempts)]


def plan_seed_queue(
    urgent: list[str],
    backfill: list[str],
    max_backfill: int,
    ledger: BackfillLedger,
    max_attempts: int = MAX_BACKFILL_ATTEMPTS,
) -> list[str]:
    """Seeds to generate this run: every urgent seed, then at most `max_backfill` backfill seeds.

    The cap bounds the latency of new mints. A run plans its queue once, at the start, so a token
    minted during a run waits for that run to finish: after the run's own new mints, at most
    `max_backfill` backfill packages of about an hour each. (Nothing else bounds a run: systemd's
    RuntimeMaxSec has no effect on a Type=oneshot service, and the unit's TimeoutStartSec is a
    24-hour safety net.)

    Backfill seeds go in order of their failed backfill runs, whatever the reason
    (BackfillLedger.failed_runs(): fewest first, then API order). A seed that failed is retried
    only once every other waiting seed has failed as often or is done, so a seed that always fails,
    for any reason, costs one render per pass over the backlog and cannot stall the backfill.
    Seeds with `max_attempts` or more failed ember attempts are left out: they are given up until
    the generator binary changes, so a seed whose ember edition always fails costs at most
    `max_attempts` renders. A seed whose regenerated orbit did not match the live package is left
    out after that one render. Other failures never give a seed up.
    """
    eligible = [seed for seed in backfill if not ledger.given_up(seed, max_attempts)]
    ordered = sorted(eligible, key=ledger.failed_runs)
    return [*urgent, *ordered[: max(max_backfill, 0)]]


# ---------------------------------------------------------------------------
# Per-seed outcomes
# ---------------------------------------------------------------------------


class Outcome(enum.Enum):
    """How far a seed got: the result of generating it, and then of processing it."""

    FAILED = "failed"
    """Nothing usable, for a reason other than the ember edition: the generator failed, timed
    out or was killed, the core package is incomplete, the live package of an ember-mode
    backfill cannot be read or used, or ssh/scp failed. Not an ember attempt: for a backfill seed
    the ledger counts it as another failure, which only moves the seed back in the queue."""
    COMPLETE = "complete"
    """The full package, ember edition included (for an ember-mode backfill: its ember edition
    was uploaded)."""
    CORE_ONLY = "core only"
    """The package without the ember edition: generated with GENERATOR_EXIT_EMBER_FAILED, or
    (after processing) uploaded without it. Counts one failed ember attempt."""
    EMBER_FAILED = "ember failed"
    """The ember edition failed and nothing was uploaded: a backfill seed's exit 3, regenerated
    metadata that cannot show its orbit or give its ember entries, or an incomplete local ember
    edition. Counts one failed ember attempt."""
    ORBIT_MISMATCH = "orbit mismatch"
    """An ember-mode backfill whose regenerated package shows another orbit than the live one:
    nothing was uploaded. The generator is deterministic, so this binary would regenerate the
    same orbit on every retry: the seed counts one failed ember attempt and is given up at once
    (BackfillLedger.orbit_mismatches) instead of after --max-backfill-attempts renders."""


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
    """Failed ember attempts: generator exit 3, an orbit that differs from the live package's,
    or an incomplete ember edition (CORE_ONLY and EMBER_FAILED outcomes)."""
    other_failures: dict[str, int] = dataclasses.field(default_factory=dict)
    """Backfill runs that failed for any other reason (FAILED outcomes: the generator exited 1,
    crashed, timed out or was killed, the live package cannot be used, an upload failed)."""
    orbit_mismatches: set[str] = dataclasses.field(default_factory=set)
    """Seeds whose regenerated orbit differed from the live package's (ORBIT_MISMATCH): given up
    at once, whatever --max-backfill-attempts says, until the generator binary changes."""

    def given_up(self, seed: str, max_attempts: int) -> bool:
        """True if `seed` is given up with this generator binary: an orbit mismatch, or at least
        `max_attempts` failed ember attempts."""
        return seed in self.orbit_mismatches or self.ember_failures.get(seed, 0) >= max_attempts

    def failed_runs(self, seed: str) -> int:
        """Every failed backfill run of `seed`, whatever the reason: its place in the queue."""
        return self.ember_failures.get(seed, 0) + self.other_failures.get(seed, 0)

    def retain(self, seeds: Container[str]) -> None:
        """Forget the counts of every seed not in `seeds` (those still waiting for a backfill)."""
        for counts in self._all_counts():
            for seed in [seed for seed in counts if seed not in seeds]:
                del counts[seed]
        self.orbit_mismatches = {seed for seed in self.orbit_mismatches if seed in seeds}

    def record(
        self, seed: str, outcome: Outcome, *, backfill: bool, interrupted: bool = False
    ) -> bool:
        """Record one processed seed; True if the ledger changed and must be saved.

        COMPLETE clears the seed's counts. CORE_ONLY and EMBER_FAILED count one failed ember
        attempt, for an urgent seed too: a new mint uploaded without its ember edition becomes a
        backfill seed with one attempt. ORBIT_MISMATCH counts one too, and gives the seed up.
        FAILED counts one other failure for a `backfill` seed only (every run retries urgent
        seeds anyway). Nothing is counted while the run is `interrupted` (shutting down), since
        the failure may be the signal's doing.
        """
        if outcome is Outcome.COMPLETE:
            cleared = [counts.pop(seed) for counts in self._all_counts() if seed in counts]
            mismatched = seed in self.orbit_mismatches
            self.orbit_mismatches.discard(seed)
            return bool(cleared) or mismatched
        if outcome is Outcome.FAILED and not backfill:
            return False
        if interrupted:
            log.info("0x%s: failed run not counted (shutting down)", seed)
            return False
        counts = self.other_failures if outcome is Outcome.FAILED else self.ember_failures
        counts[seed] = counts.get(seed, 0) + 1
        if outcome is Outcome.ORBIT_MISMATCH:
            self.orbit_mismatches.add(seed)
        return True

    def _all_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        """Both count maps."""
        return self.ember_failures, self.other_failures


def _ledger_counts(data: dict[object, object]) -> dict[str, int]:
    """The positive integer counts of one ledger map (anything else is dropped)."""
    return {str(seed): count for seed, count in data.items() if _is_int(count) and count > 0}


def load_backfill_ledger(
    generator: GeneratorIdentity | None, path: Path = BACKFILL_FAILURES
) -> BackfillLedger:
    """The failure counts of earlier runs with the same generator binary.

    The file is `{"generator": {"path", "size", "mtime_ns"}, "ember_failures": {seed: count},
    "other_failures": {seed: count}, "orbit_mismatches": [seed, ...]}` ("orbit_mismatches" may be
    absent: files written before it existed). The ledger is empty if the file is absent,
    unreadable or malformed (logged), or was written for another generator binary (the counts
    reset when the binary changes).
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
    mismatches = data.get("orbit_mismatches", [])
    return BackfillLedger(
        _ledger_counts(data["ember_failures"]),
        _ledger_counts(data["other_failures"]),
        {seed for seed in mismatches if isinstance(seed, str)}
        if isinstance(mismatches, list)
        else set(),
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
        "orbit_mismatches": sorted(ledger.orbit_mismatches),
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

    COMPLETE for 0, CORE_ONLY for GENERATOR_EXIT_EMBER_FAILED, FAILED for anything else
    (including a timeout, or a kill by a signal).
    """
    cmd_parts = [*exec_cmd, "--seed", f"0x{seed}", "--output", f"0x{seed}"]
    log.info("GENERATE  seed=0x%s", seed)

    try:
        result = run_subprocess(cmd_parts, timeout=timeout, label=f"gen-0x{seed}")
    except (subprocess.TimeoutExpired, OSError):
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


def remove_remote_ember_files(ssh_host: str, ssh_user: str, remote_dir: str, seed: str) -> bool:
    """Delete the ember edition's files from the remote package of `seed` (absent ones are fine).

    Run before uploading a package without the ember edition: scp adds and overwrites files but
    never deletes any, so a stale ember file of an earlier upload would stay next to a manifest
    that does not list it, and could make the package look complete. False if the removal failed.
    """
    package = remote_seed_dir(remote_dir, seed)
    paths = " ".join(shlex.quote(f"{package}/{filename}") for filename in EMBER_PACKAGE_FILES)
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


def prepare_remote_dirs(
    ssh_host: str, ssh_user: str, remote_dirs: Iterable[str], remove: Iterable[str] = ()
) -> bool:
    """Create the remote directories (and parents) an upload writes into; delete `remove` first.

    One ssh call. False (logged) if it failed.
    """
    command = "mkdir -p -- " + " ".join(shlex.quote(path) for path in sorted(set(remote_dirs)))
    stale = " ".join(shlex.quote(path) for path in remove)
    if stale:
        command = f"rm -f -- {stale} && {command}"
    result = run_remote(ssh_host, ssh_user, command, timeout=30, label="ssh-prepare")
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


def ember_upload_steps(local_seed_dir: Path) -> list[UploadStep]:
    """The ember edition of a package whose core is live: its media, the manifest, the certificate.

    The media go first, then metadata/assets.json (merged with the live manifest), and
    metadata/ember.json last of all, so the remote package looks complete only once everything
    else has landed. The manifest and the certificate are staged: an interrupted upload never
    leaves the live manifest truncated.
    """
    groups: list[tuple[str, list[Path]]] = []
    for filename in EMBER_MEDIA_FILES:
        subdir = posixpath.dirname(filename)
        if not groups or groups[-1][0] != subdir:
            groups.append((subdir, []))
        groups[-1][1].append(local_seed_dir / filename)
    steps = [UploadStep(tuple(paths), subdir) for subdir, paths in groups]
    for filename in (ASSET_MANIFEST, EMBER_CERTIFICATE):
        steps.append(
            UploadStep((local_seed_dir / filename,), posixpath.dirname(filename), staged=True)
        )
    return steps


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
    if not prepare_remote_dirs(ssh_host, ssh_user, map(destination, steps), stale):
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
# Ember-mode backfill: orbit check and merged manifest
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
    """A compact rendering of a parsed JSON value for log messages."""
    if isinstance(value, list):
        return "[" + ", ".join(_format_json_value(item) for item in value) + "]"
    return str(value)


def orbit_differences(live_traits: object, local_traits: object) -> list[str]:
    """The ORBIT_IDENTITY_FIELDS in which two parsed nft_traits.json files differ.

    Parse both with parse_json_exact(), so numbers compare exactly. A field missing on either
    side counts as a difference. Each entry names the field and both values.
    """
    differences: list[str] = []
    for field in ORBIT_IDENTITY_FIELDS:
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


def read_remote_file(ssh_host: str, ssh_user: str, remote_path: str) -> str | None:
    """The text of a remote file, or None (logged) if ssh or the read failed."""
    result = run_remote(
        ssh_host, ssh_user, f"cat -- {shlex.quote(remote_path)}", timeout=60, label="ssh-cat"
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
    """metadata/nft_traits.json, parsed with parse_json_exact(): it has every
    ORBIT_IDENTITY_FIELDS entry."""
    manifest: dict[str, object]
    """metadata/assets.json: a JSON object whose assets list gives every entry a role."""


def parse_live_package(traits_text: str, manifest_text: str) -> LivePackage:
    """Check the live metadata an ember-mode backfill needs; ValueError names what is wrong."""
    try:
        traits = parse_json_exact(traits_text)
    except ValueError as exc:
        raise ValueError(f"{NFT_TRAITS} is not valid JSON ({exc})") from None
    missing = [
        ".".join(field) for field in ORBIT_IDENTITY_FIELDS if _json_field(traits, field) is _MISSING
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
    An unusable live file needs repair on the asset host: without the orbit fields and a valid
    manifest no ember edition can be checked against the package or merged into it. Neither case
    is an ember attempt, so neither gives the seed up; the ember-mode backfill checks the live
    package before rendering too, so an unusable one costs no render.
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
    starts from the current manifest). If the regenerated package shows the same orbit
    (ORBIT_IDENTITY_FIELDS), uploads the ember media, then the live manifest merged with the new
    ember entries, then metadata/ember.json last (ember_upload_steps(); the remote certificate is
    deleted first). The published main art, spectral files, generation.json and nft_traits.json
    are never touched.

    Returns COMPLETE once uploaded; ORBIT_MISMATCH (nothing uploaded) if the regenerated package
    shows another orbit; EMBER_FAILED (nothing uploaded) if its own metadata cannot show its orbit
    or give its ember entries;
    FAILED (nothing uploaded, not an ember attempt) if the live package cannot be read or used,
    or ssh, scp or the local disk failed.
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
            "0x%s: the regenerated %s is not valid JSON (%s), so its orbit cannot be checked. "
            "Nothing is uploaded.",
            seed,
            NFT_TRAITS,
            exc,
        )
        return Outcome.EMBER_FAILED
    differences = orbit_differences(live.traits, local_traits)
    if differences:
        log.error(
            "0x%s: the regenerated package shows a DIFFERENT ORBIT than the live one (%s). Its "
            "ember edition would not match the published art, so nothing is uploaded, and the "
            "seed is given up with this generator binary (it would regenerate the same orbit). "
            "Only --backfill-mode full would upload it, replacing the published package, main "
            "art included.",
            seed,
            "; ".join(differences),
        )
        return Outcome.ORBIT_MISMATCH

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
        (seed_dir / ASSET_MANIFEST).write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        log.error("0x%s: could not write the merged %s: %s", seed, ASSET_MANIFEST, exc)
        return Outcome.FAILED

    log.info("0x%s: same orbit as the live package; uploading only its ember edition", seed)
    steps = ember_upload_steps(seed_dir)
    if not upload_steps(
        ssh_host, ssh_user, remote_dir, seed_dir, steps, remove=(EMBER_CERTIFICATE,)
    ):
        return Outcome.FAILED
    return Outcome.COMPLETE


# ---------------------------------------------------------------------------
# One seed, end to end
# ---------------------------------------------------------------------------


def _generate_and_upload(
    seed: str,
    exec_cmd: list[str],
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    timeout: int,
    backfill: BackfillMode | None,
    ember_capable: bool,
) -> Outcome:
    """process_seed() without the dry run, the cleanup and the final log line."""
    # An ember-mode backfill needs the live package's orbit and manifest: check them before the
    # hour-long render, so a live package that cannot be used costs no render.
    if (
        backfill is BackfillMode.EMBER
        and read_live_package(seed, ssh_host, ssh_user, remote_dir) is None
    ):
        return Outcome.FAILED
    generated = generate(exec_cmd, seed, timeout)
    if generated is Outcome.FAILED:
        return Outcome.FAILED
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

    # A package without the ember edition: exit 3, or a generator that predates the edition.
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
) -> Outcome:
    """Full pipeline for one seed: generate -> validate package -> upload -> cleanup.

    `backfill` is None for an urgent seed (its remote package lacks a core file: a new mint or a
    broken upload) and the backfill mode for a seed whose remote package lacks only the ember
    edition. `ember_capable` is False for a generator that predates the ember edition
    (generator_supports_ember()): its exit-0 packages are checked against the core files only.

    Returns:
        COMPLETE: the package was uploaded with its ember edition (an ember-mode backfill: the
            ember edition alone, after the orbit check).
        CORE_ONLY: an urgent seed's package was uploaded without the ember edition (generator
            exit 3, or a generator that predates the edition), after any stale ember file was
            deleted from its remote directory.
        EMBER_FAILED: the ember edition failed and nothing was uploaded (a backfill seed's exit
            3, an incomplete local ember edition).
        ORBIT_MISMATCH: an ember-mode backfill regenerated another orbit than the live
            package's; nothing was uploaded, and the seed is given up with this binary.
        FAILED: nothing was uploaded for any other reason (for an ember-mode backfill, this
            includes a live package that cannot be read or used, checked before generating).
    The local package is deleted before generating and afterwards, whatever the outcome.
    """
    if dry_run:
        if backfill is BackfillMode.EMBER:
            log.info(
                "DRY-RUN  would regenerate 0x%s and, if its orbit matches the live package, "
                "upload only its ember edition",
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
            seed, exec_cmd, ssh_host, ssh_user, remote_dir, timeout, backfill, ember_capable
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
    elif outcome is Outcome.ORBIT_MISMATCH:
        log.error(
            "ORBIT MISMATCH  seed=0x%s  (total %s)  nothing uploaded; given up with this binary",
            seed,
            elapsed,
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
        log.info("[preflight] Generator binary: OK (%s, ember edition supported)", generator[0])
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


def backfill_mode(value: str) -> BackfillMode:
    """argparse type: a BackfillMode by its value ("ember" or "full")."""
    try:
        return BackfillMode(value.strip().lower())
    except ValueError:
        choices = ", ".join(mode.value for mode in BackfillMode)
        raise argparse.ArgumentTypeError(f"{value!r} is not one of {choices}") from None


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
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-seed generator timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--max-backfill",
        type=non_negative_int,
        default=os.environ.get(ENV_MAX_BACKFILL, str(DEFAULT_MAX_BACKFILL)),
        help=(
            "Packages missing only the ember edition to regenerate per run, after every "
            "new or incomplete package; 0 pauses the backfill "
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
            "same orbit as the live package; 'full' replaces the whole remote package, main "
            f"art included (env: {ENV_BACKFILL_MODE}; default: {DEFAULT_BACKFILL_MODE.value})"
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
    """One sync run: plan the incomplete packages, generate and upload them; the exit status.

    Returns 0 if every planned seed succeeded (an urgent package uploaded without its ember
    edition counts as a success) and 1 on a configuration error, a failed seed list or remote
    listing, or any failed seed (a failed ember backfill included).
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
    if exec_cmd is not None:
        generator_identity = GeneratorIdentity.of(exec_cmd[0])
        ember_capable = generator_supports_ember(exec_cmd)
        if not ember_capable:
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
    urgent, backfill = find_missing_seeds(seeds, remote_files)
    incomplete = len(urgent) + len(backfill)
    backfill_seeds = set(backfill)

    if incomplete == 0:
        elapsed = time.monotonic() - t_start
        log.info(
            "All %d tokens have complete asset packages on remote. Nothing to do. (%s)",
            len(seeds),
            fmt_duration(elapsed),
        )
        return 0

    log.info(
        "Found %d seeds with incomplete asset packages (out of %d total): %d new or incomplete, "
        "%d missing only the ember edition",
        incomplete,
        len(seeds),
        len(urgent),
        len(backfill),
    )

    # Only seeds still waiting for the backfill keep their failure counts.
    ledger = load_backfill_ledger(generator_identity)
    ledger.retain(backfill_seeds)
    given_up = given_up_seeds(backfill, ledger, args.max_backfill_attempts)
    for seed in given_up:
        if seed in ledger.orbit_mismatches:
            log.warning(
                "0x%s: ember backfill given up: this generator binary regenerates a different "
                "orbit than the live package (rebuild the generator, or delete the seed from "
                "orbit_mismatches in %s, to try again; --backfill-mode full would replace the "
                "whole published package)",
                seed,
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
    missing = plan_seed_queue(urgent, backfill, max_backfill, ledger, args.max_backfill_attempts)
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
        return 0

    # --- Phase 2: generate and upload sequentially ---

    ok_count = 0
    fail_count = 0
    failed_seeds: list[str] = []
    core_only_seeds: list[str] = []
    ember_failed_seeds: list[str] = []

    for i, seed in enumerate(missing, 1):
        if shutdown_requested:
            remaining = len(missing) - i + 1
            log.info("Shutdown requested -- skipping remaining %d seeds", remaining)
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
        )
        if outcome in (Outcome.FAILED, Outcome.EMBER_FAILED, Outcome.ORBIT_MISMATCH):
            fail_count += 1
            failed_seeds.append(seed)
            if outcome is not Outcome.FAILED:
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

    skipped = len(missing) - ok_count - fail_count
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
    if skipped > 0:
        log.info("  Skipped (shutdown)   : %d", skipped)
    if failed_seeds:
        log.info("  Failed seeds         : %s", ", ".join(f"0x{s}" for s in failed_seeds))
    log.info("  Wall time            : %s", fmt_duration(elapsed))
    log.info("=" * 60)

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
