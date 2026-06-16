#!/usr/bin/env python3
"""
CosmicSignature NFT asset package checker and uploader.

Fetches all tokens from the CosmicGame API, determines which per-seed asset
packages are incomplete on the destination server, generates missing packages
via the Rust binary, and uploads them via SCP.

Designed to run under a systemd timer (every 5 minutes). The systemd service
unit prevents overlapping runs.

Configuration is read from (in increasing priority):
    1. .env file in the working directory
    2. Environment variables
    3. CLI arguments

Copy .env.example to .env and fill in your deployment values.

Usage:
    python3 run.py [--dry-run]
    python3 run.py --ssh-host HOST --ssh-user USER --api-url URL --remote-dir DIR
"""

import argparse
import json
import logging
import logging.handlers
import os
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
from pathlib import Path

from _utils import GENERATOR_CANDIDATES, fmt_duration

# ---------------------------------------------------------------------------
# Defaults (non-sensitive only; deployment values come from .env / env vars)
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 86400  # 24 hours
API_TOKEN_FETCH_LIMIT = 999999
DEFAULT_ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
DEFAULT_NFT_CONTRACT = "0xbb84Be3500A63581d3F2d5AC3bdF8685AAedad25"

LOCAL_OUTPUT_DIR = Path("output")
LOG_FILE = "imgcheck.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
SEED_MISMATCH_REPORT = Path("seed_source_mismatch.json")

SSH_BASE_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
]


# Expected per-seed package emitted by the Rust generator.
SPECTRAL_BIN_COUNT = 64
SPECTRAL_FILE_RE = re.compile(r"^(?P<bin>\d{2})_\d+nm\.png$")
EXPECTED_SPECTRAL_BINS = set(range(SPECTRAL_BIN_COUNT))

# Environment variable names for required config
ENV_SSH_HOST = "COSMICSIG_SSH_HOST"
ENV_SSH_USER = "COSMICSIG_SSH_USER"
ENV_API_URL = "COSMICSIG_API_URL"
ENV_REMOTE_DIR = "COSMICSIG_REMOTE_DIR"
ENV_ARBITRUM_RPC_URL = "COSMICSIG_ARBITRUM_RPC_URL"
ENV_NFT_CONTRACT = "COSMICSIG_NFT_CONTRACT"

# Minimal ABI selectors for the verified Cosmic Signature NFT contract.
SELECTOR_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()
SELECTOR_TOKEN_BY_INDEX = "0x4f6ccce7"  # tokenByIndex(uint256)
SELECTOR_GET_NFT_SEED = "0xb0c0fe4e"  # getNftSeed(uint256)

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
# Signal handling
# ---------------------------------------------------------------------------


def install_signal_handlers() -> None:
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
    extra = os.environ.get("SSH_OPTS_EXTRA", "").split()
    return [*SSH_BASE_OPTS, *extra]


def ssh_cmd(host: str, user: str) -> list[str]:
    return ["ssh", *ssh_opts(), "-l", user, host]


def run_subprocess(
    cmd: list[str],
    *,
    timeout: int | None = 60,
    label: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with full logging. Raises nothing -- caller checks returncode."""
    log.debug("[%s] Running: %s", label, " ".join(cmd))
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    value = address.strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        raise ValueError(f"invalid Ethereum address: {address!r}")
    return value


def encode_uint256_arg(value: int) -> str:
    if value < 0 or value >= 2**256:
        raise ValueError(f"uint256 argument out of range: {value}")
    return f"{value:064x}"


def decode_uint256_result(result: str) -> int:
    if not isinstance(result, str) or not result.startswith("0x"):
        raise RuntimeError(f"invalid eth_call result: {result!r}")
    hex_value = result[2:]
    if len(hex_value) < 64:
        raise RuntimeError(f"short eth_call result: {result!r}")
    return int(hex_value[-64:], 16)


def rpc_request(rpc_url: str, method: str, params: list[object], timeout: int = 30) -> object:
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
            raise RuntimeError(
                "API and blockchain seed lists do not match; refusing to continue"
            )
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


def list_remote_files(ssh_host: str, ssh_user: str, remote_dir: str) -> set[str]:
    """List known asset package files beneath the remote asset directory."""
    quoted_dir = shlex.quote(remote_dir)
    remote_cmd = (
        f"cd {quoted_dir} 2>/dev/null "
        "&& find . -mindepth 2 -maxdepth 3 -type f -print || true"
    )
    cmd = [*ssh_cmd(ssh_host, ssh_user), remote_cmd]

    try:
        result = run_subprocess(cmd, timeout=60, label="ssh-find")
    except (subprocess.TimeoutExpired, OSError):
        log.warning("Could not list remote files -- treating as empty")
        return set()

    if result.returncode != 0:
        log.warning("SSH find returned rc=%d -- treating remote as empty", result.returncode)
        return set()

    files = {
        line.strip().removeprefix("./")
        for line in result.stdout.splitlines()
        if line.strip()
    }
    log.info("Found %d existing package files on remote server", len(files))
    return files


def missing_remote_package_parts(seed: str, remote_files: set[str]) -> list[str]:
    """Return missing files/groups for the remote package at 0x<seed>/."""
    package_dir = f"0x{seed}"
    required_files = [
        f"{package_dir}/image.png",
        f"{package_dir}/video.mp4",
        f"{package_dir}/spectral_sweep.mp4",
    ]
    missing = [
        path.removeprefix(f"{package_dir}/")
        for path in required_files
        if path not in remote_files
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


def find_missing_seeds(seeds: list[str], remote_files: set[str]) -> list[str]:
    """Return API seeds missing one or more files from their remote asset package."""
    missing: list[str] = []
    for seed in seeds:
        missing_parts = missing_remote_package_parts(seed, remote_files)
        if missing_parts:
            log.debug("MISSING  0x%s  (%s)", seed, ", ".join(missing_parts))
            missing.append(seed)
    return missing


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


# ---------------------------------------------------------------------------
# Generate, upload, cleanup for a single seed
# ---------------------------------------------------------------------------


def generate(exec_cmd: list[str], seed: str, timeout: int) -> bool:
    cmd_parts = [*exec_cmd, "--seed", f"0x{seed}", "--output", f"0x{seed}"]
    log.info("GENERATE  seed=0x%s", seed)

    try:
        result = run_subprocess(cmd_parts, timeout=timeout, label=f"gen-0x{seed}")
    except (subprocess.TimeoutExpired, OSError):
        return False

    if result.returncode != 0:
        log.error(
            "Generator FAILED for 0x%s (rc=%d). Check log for full stdout/stderr.",
            seed,
            result.returncode,
        )
        return False

    return True


def missing_local_package_parts(seed_dir: Path) -> list[str]:
    """Return missing files/groups for a generated local seed package."""
    missing: list[str] = []
    for filename in ["image.png", "video.mp4", "spectral_sweep.mp4"]:
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


def find_local_package(seed: str) -> Path | None:
    """Validate and return the generated per-seed output package directory."""
    seed_dir = LOCAL_OUTPUT_DIR / f"0x{seed}"
    if not seed_dir.is_dir():
        log.error("Package directory NOT FOUND for 0x%s. Tried: %s", seed, seed_dir)
        return None

    missing = missing_local_package_parts(seed_dir)
    if missing:
        log.error("Package INCOMPLETE for 0x%s. Missing: %s", seed, ", ".join(missing))
        return None

    file_count = sum(1 for path in seed_dir.rglob("*") if path.is_file())
    log.debug("Found complete package: %s (%d files)", seed_dir, file_count)
    return seed_dir


def ensure_remote_dir(ssh_host: str, ssh_user: str, remote_dir: str) -> bool:
    """Ensure the remote asset root exists before uploading a seed package."""
    quoted_dir = shlex.quote(remote_dir)
    try:
        result = run_subprocess(
            [*ssh_cmd(ssh_host, ssh_user), f"mkdir -p -- {quoted_dir}"],
            timeout=30,
            label="ssh-mkdir",
        )
    except (subprocess.TimeoutExpired, OSError):
        return False

    if result.returncode == 0:
        return True

    log.error("Could not create remote asset directory %s (rc=%d)", remote_dir, result.returncode)
    return False


def upload_package(
    ssh_host: str,
    ssh_user: str,
    local_seed_dir: Path,
    remote_dir: str,
    retries: int = 2,
) -> bool:
    """Upload the complete per-seed output directory via recursive SCP."""
    if not ensure_remote_dir(ssh_host, ssh_user, remote_dir):
        return False

    remote_target = f"{ssh_user}@{ssh_host}:{remote_dir.rstrip('/')}/"
    cmd = ["scp", *ssh_opts(), "-r", str(local_seed_dir), remote_target]

    for attempt in range(1, retries + 1):
        backoff = 2**attempt
        log.info(
            "UPLOAD PACKAGE (attempt %d/%d)  %s -> %s",
            attempt,
            retries,
            local_seed_dir.name,
            remote_target,
        )
        try:
            result = run_subprocess(cmd, timeout=900, label=f"scp-package-{local_seed_dir.name}")
        except (subprocess.TimeoutExpired, OSError):
            if attempt < retries:
                log.info("Retrying package SCP in %ds ...", backoff)
                time.sleep(backoff)
            continue

        if result.returncode == 0:
            log.info("UPLOADED PACKAGE  %s -> %s", local_seed_dir.name, remote_target)
            return True

        log.warning(
            "Package SCP failed (attempt %d/%d) rc=%d: %s",
            attempt,
            retries,
            result.returncode,
            result.stderr.strip()[:300],
        )
        if attempt < retries:
            log.info("Retrying package SCP in %ds ...", backoff)
            time.sleep(backoff)

    log.error(
        "PACKAGE SCP FAILED after %d attempts: %s -> %s",
        retries,
        local_seed_dir,
        remote_dir,
    )
    return False


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


def process_seed(
    seed: str,
    exec_cmd: list[str] | None,
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    timeout: int,
    dry_run: bool,
) -> bool:
    """Full pipeline for one seed: generate -> validate package -> upload -> cleanup."""
    if dry_run:
        log.info("DRY-RUN  would generate and upload package 0x%s/", seed)
        return True

    t0 = time.monotonic()

    if exec_cmd is None:
        log.error("No generator binary available for 0x%s", seed)
        return False

    if not generate(exec_cmd, seed, timeout):
        return False

    package_dir = find_local_package(seed)
    if package_dir is None:
        cleanup_seed_dir(seed)
        return False

    if upload_package(ssh_host, ssh_user, package_dir, remote_dir):
        cleanup_seed_dir(seed)
        elapsed = time.monotonic() - t0
        log.info("OK  seed=0x%s  (total %s)", seed, fmt_duration(elapsed))
        return True

    log.error("UPLOAD FAILURE for 0x%s", seed)
    return False


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
) -> bool:
    """
    Verify that all external dependencies are working before committing to
    lengthy generation: SSH auth, remote write permissions, seed source
    connectivity, release generator binary, and ffmpeg on PATH.
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

    # 4. Generator binary
    generator = resolve_generator(None)
    if generator:
        log.info("[preflight] Generator binary: OK (%s)", generator[0])
    else:
        log.warning("[preflight] Generator binary: NOT FOUND")
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


def parse_args() -> argparse.Namespace:
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
        "--dry-run",
        action="store_true",
        help="Report missing files without generating or uploading",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Verify SSH, remote write, seed sources, release generator binary, "
            "and ffmpeg, then exit"
        ),
    )
    return p.parse_args()


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


def main() -> int:
    load_dotenv()
    args = parse_args()
    setup_logging()
    install_signal_handlers()

    missing_cfg = validate_config(args)
    if missing_cfg:
        log.error("Missing required configuration:\n%s", "\n".join(missing_cfg))
        log.error("Set them in .env, environment variables, or CLI args. See .env.example.")
        return 1

    log.info("=" * 60)
    log.info("CosmicSignature asset sync started")
    log.info("  ssh_host         = %s", args.ssh_host)
    log.info("  ssh_user         = %s", args.ssh_user)
    log.info("  api_url          = %s", args.api_url or "(not set)")
    log.info("  remote_dir       = %s", args.remote_dir)
    log.info("  arbitrum_rpc_url = %s", safe_url_for_log(args.arbitrum_rpc_url))
    log.info("  nft_contract     = %s", args.nft_contract)
    log.info("  timeout          = %s", fmt_duration(args.timeout))
    log.info("  dry_run          = %s", args.dry_run)
    log.info("  preflight        = %s", args.preflight)
    log.info("=" * 60)

    if args.preflight:
        ok = preflight(
            args.ssh_host,
            args.ssh_user,
            args.remote_dir,
            args.api_url,
            args.arbitrum_rpc_url,
            args.nft_contract,
        )
        return 0 if ok else 1

    t_start = time.monotonic()

    exec_cmd = resolve_generator(args.generator)
    if exec_cmd is None and not args.dry_run:
        log.error("No generator available and not in dry-run mode. Exiting.")
        return 1

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
    missing = find_missing_seeds(seeds, remote_files)

    if not missing:
        elapsed = time.monotonic() - t_start
        log.info(
            "All %d tokens have complete asset packages on remote. Nothing to do. (%s)",
            len(seeds),
            fmt_duration(elapsed),
        )
        return 0

    log.info(
        "Found %d seeds with incomplete asset packages (out of %d total)",
        len(missing),
        len(seeds),
    )

    # --- Phase 2: generate and upload sequentially ---

    ok_count = 0
    fail_count = 0
    failed_seeds: list[str] = []

    for i, seed in enumerate(missing, 1):
        if shutdown_requested:
            remaining = len(missing) - i + 1
            log.info("Shutdown requested -- skipping remaining %d seeds", remaining)
            break

        log.info("[%d/%d] seed=0x%s", i, len(missing), seed)

        success = process_seed(
            seed,
            exec_cmd,
            args.ssh_host,
            args.ssh_user,
            args.remote_dir,
            args.timeout,
            args.dry_run,
        )
        if success:
            ok_count += 1
        else:
            fail_count += 1
            failed_seeds.append(seed)

    # --- Summary ---

    skipped = len(missing) - ok_count - fail_count
    elapsed = time.monotonic() - t_start

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Seed source          : %s", seed_source)
    log.info("  Total seeds          : %d", len(seeds))
    log.info("  Missing on remote    : %d", len(missing))
    log.info("  Processed OK         : %d", ok_count)
    log.info("  Failed               : %d", fail_count)
    if skipped > 0:
        log.info("  Skipped (shutdown)   : %d", skipped)
    if failed_seeds:
        log.info("  Failed seeds         : %s", ", ".join(f"0x{s}" for s in failed_seeds))
    log.info("  Wall time            : %s", fmt_duration(elapsed))
    log.info("=" * 60)

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
