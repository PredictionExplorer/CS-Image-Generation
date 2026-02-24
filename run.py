#!/usr/bin/env python3
"""
CosmicSignature NFT image/video checker and uploader.

Fetches all tokens from the CosmicGame API, determines which images/videos
are missing on the destination server, generates missing artifacts via the
Rust binary, and uploads them via SCP.

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
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (non-sensitive only; deployment values come from .env / env vars)
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 86400  # 24 hours

LOCAL_IMG_DIR = Path("pics")
LOCAL_VID_DIR = Path("vids")
LOG_FILE = "imgcheck.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

SSH_BASE_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
]

GENERATOR_CANDIDATES = [
    "./target/release/three_body_problem",
    "./three_body_problem",
]

IMG_PREFIXES = ("0x{seed}", "enhanced_0x{seed}", "classic_0x{seed}")
VID_PREFIXES = IMG_PREFIXES

# Environment variable names for required config
ENV_SSH_HOST = "COSMICSIG_SSH_HOST"
ENV_SSH_USER = "COSMICSIG_SSH_USER"
ENV_API_URL = "COSMICSIG_API_URL"
ENV_REMOTE_DIR = "COSMICSIG_REMOTE_DIR"

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

shutdown_requested = False
log = logging.getLogger("cosmicsig")

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

_ENV_LINE_RE = re.compile(
    r"""^\s*(?:export\s+)?(?P<key>[A-Za-z_]\w*)\s*=\s*(?P<val>.*)$"""
)


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
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S",
    ))
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def install_signal_handlers() -> None:
    def handler(signum, _frame):
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


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def ssh_opts() -> list[str]:
    extra = os.environ.get("SSH_OPTS_EXTRA", "").split()
    return SSH_BASE_OPTS + extra


def ssh_cmd(host: str, user: str) -> list[str]:
    return ["ssh"] + ssh_opts() + ["-l", user, host]


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
            label, fmt_duration(elapsed), result.returncode,
            len(result.stdout), len(result.stderr),
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


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


def fetch_token_seeds(api_base_url: str, retries: int = 3) -> list[str]:
    """
    Fetch all CosmicSignature token seeds from the API.
    Returns normalized hex seed strings (without 0x prefix).
    """
    url = f"{api_base_url}/api/cosmicgame/cst/list/all/0/100000"
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        backoff = 2 ** attempt
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
                seed = seed.removeprefix("0x").removeprefix("0X")
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
# Remote file check
# ---------------------------------------------------------------------------


def list_remote_files(ssh_host: str, ssh_user: str, remote_dir: str) -> set[str]:
    """List all filenames in the remote asset directory via a single SSH call."""
    cmd = ssh_cmd(ssh_host, ssh_user) + [f"ls -1 -- {remote_dir}/ 2>/dev/null || true"]

    try:
        result = run_subprocess(cmd, timeout=30, label="ssh-ls")
    except (subprocess.TimeoutExpired, OSError):
        log.warning("Could not list remote files -- treating as empty")
        return set()

    if result.returncode != 0:
        log.warning("SSH ls returned rc=%d -- treating remote as empty", result.returncode)
        return set()

    files = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    log.info("Found %d existing files on remote server", len(files))
    return files


def find_missing_seeds(seeds: list[str], remote_files: set[str]) -> list[str]:
    """Return seeds that are missing at least one of PNG or MP4 on the remote."""
    missing: list[str] = []
    for seed in seeds:
        img_present = f"0x{seed}.png" in remote_files
        vid_present = f"0x{seed}.mp4" in remote_files
        if not img_present or not vid_present:
            parts = []
            if not img_present:
                parts.append("PNG")
            if not vid_present:
                parts.append("MP4")
            log.debug("MISSING  0x%s  (%s)", seed, " + ".join(parts))
            missing.append(seed)
    return missing


# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------


def resolve_generator(generator_arg: str | None) -> str | None:
    """Find the generator binary. Returns command string or None."""
    candidates = [generator_arg] if generator_arg else GENERATOR_CANDIDATES

    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        if os.access(cand, os.X_OK):
            log.info("Using generator: %s", cand)
            return cand
        log.info("Found %s (not executable) -- will invoke via python3", cand)
        return f"python3 {cand}"

    log.warning(
        "Generator not found. Tried: %s",
        ", ".join(c for c in candidates if c),
    )
    return None


# ---------------------------------------------------------------------------
# Generate, upload, cleanup for a single seed
# ---------------------------------------------------------------------------


def generate(exec_cmd: str, seed: str, timeout: int) -> bool:
    cmd_parts = exec_cmd.split() + ["--seed", f"0x{seed}", "--file-name", f"0x{seed}"]
    log.info("GENERATE  seed=0x%s", seed)

    try:
        result = run_subprocess(cmd_parts, timeout=timeout, label=f"gen-0x{seed}")
    except (subprocess.TimeoutExpired, OSError):
        return False

    if result.returncode != 0:
        log.error(
            "Generator FAILED for 0x%s (rc=%d). Check log for full stdout/stderr.",
            seed, result.returncode,
        )
        return False

    return True


def find_local_files(seed: str) -> tuple[Path | None, Path | None]:
    """Search for generated image and video files using known naming conventions."""
    img_path: Path | None = None
    for pattern in IMG_PREFIXES:
        cand = LOCAL_IMG_DIR / f"{pattern.format(seed=seed)}.png"
        if cand.is_file():
            img_path = cand
            log.debug("Found image: %s (%d bytes)", cand, cand.stat().st_size)
            break
    if img_path is None:
        tried = [str(LOCAL_IMG_DIR / f"{p.format(seed=seed)}.png") for p in IMG_PREFIXES]
        log.error("Image NOT FOUND for 0x%s. Tried: %s", seed, ", ".join(tried))

    vid_path: Path | None = None
    for pattern in VID_PREFIXES:
        cand = LOCAL_VID_DIR / f"{pattern.format(seed=seed)}.mp4"
        if cand.is_file():
            vid_path = cand
            log.debug("Found video: %s (%d bytes)", cand, cand.stat().st_size)
            break
    if vid_path is None:
        tried = [str(LOCAL_VID_DIR / f"{p.format(seed=seed)}.mp4") for p in VID_PREFIXES]
        log.error("Video NOT FOUND for 0x%s. Tried: %s", seed, ", ".join(tried))

    return img_path, vid_path


def upload_file(
    ssh_host: str, ssh_user: str, local_path: Path, remote_path: str, retries: int = 2,
) -> bool:
    """Upload a single file via SCP with retries."""
    cmd = ["scp"] + ssh_opts() + [str(local_path), f"{ssh_user}@{ssh_host}:{remote_path}"]

    for attempt in range(1, retries + 1):
        log.info("UPLOAD (attempt %d/%d)  %s -> %s", attempt, retries, local_path.name, remote_path)
        try:
            result = run_subprocess(cmd, timeout=300, label=f"scp-{local_path.name}")
        except (subprocess.TimeoutExpired, OSError):
            if attempt < retries:
                backoff = 5 * attempt
                log.info("Retrying SCP in %ds ...", backoff)
                time.sleep(backoff)
            continue

        if result.returncode == 0:
            log.info("UPLOADED  %s -> %s", local_path.name, remote_path)
            return True

        log.warning(
            "SCP failed (attempt %d/%d) rc=%d: %s",
            attempt, retries, result.returncode, result.stderr.strip()[:300],
        )
        if attempt < retries:
            backoff = 5 * attempt
            log.info("Retrying SCP in %ds ...", backoff)
            time.sleep(backoff)

    log.error("SCP FAILED after %d attempts: %s -> %s", retries, local_path, remote_path)
    return False


def cleanup_local_files(*paths: Path | None) -> None:
    for p in paths:
        if p is None or not p.is_file():
            continue
        try:
            p.unlink()
            log.debug("Cleaned up local file: %s", p)
        except OSError as exc:
            log.warning("Failed to remove %s: %s", p, exc)


def process_seed(
    seed: str,
    exec_cmd: str,
    ssh_host: str,
    ssh_user: str,
    remote_dir: str,
    timeout: int,
    dry_run: bool,
) -> bool:
    """Full pipeline for one seed: generate -> find -> upload -> cleanup."""
    if dry_run:
        log.info("DRY-RUN  would generate and upload 0x%s", seed)
        return True

    t0 = time.monotonic()

    if not generate(exec_cmd, seed, timeout):
        return False

    img_path, vid_path = find_local_files(seed)
    if img_path is None or vid_path is None:
        cleanup_local_files(img_path, vid_path)
        return False

    remote_img = f"{remote_dir}/0x{seed}.png"
    remote_vid = f"{remote_dir}/0x{seed}.mp4"

    img_ok = upload_file(ssh_host, ssh_user, img_path, remote_img)
    vid_ok = upload_file(ssh_host, ssh_user, vid_path, remote_vid)

    if img_ok and vid_ok:
        cleanup_local_files(img_path, vid_path)
        elapsed = time.monotonic() - t0
        log.info("OK  seed=0x%s  (total %s)", seed, fmt_duration(elapsed))
        return True

    log.error(
        "PARTIAL FAILURE for 0x%s (img_upload=%s, vid_upload=%s)",
        seed, "ok" if img_ok else "FAIL", "ok" if vid_ok else "FAIL",
    )
    return False


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------


def preflight(ssh_host: str, ssh_user: str, remote_dir: str, api_url: str) -> bool:
    """
    Verify that all external dependencies are working before committing to
    lengthy generation. Tests SSH auth, remote write permissions, and API
    connectivity. Returns True if all checks pass.
    """
    all_ok = True

    # 1. SSH connectivity
    log.info("[preflight] Testing SSH connection to %s@%s ...", ssh_user, ssh_host)
    try:
        result = run_subprocess(
            ssh_cmd(ssh_host, ssh_user) + ["echo ok"],
            timeout=15, label="preflight-ssh",
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
    probe = f"{remote_dir}/.preflight_probe_{os.getpid()}"
    write_cmd = f"touch {probe} && rm -f {probe}"
    try:
        result = run_subprocess(
            ssh_cmd(ssh_host, ssh_user) + [write_cmd],
            timeout=15, label="preflight-write",
        )
        if result.returncode == 0:
            log.info("[preflight] Remote write access: OK")
        else:
            log.error(
                "[preflight] Remote write access: FAILED (rc=%d, stderr=%s)",
                result.returncode, result.stderr.strip()[:200],
            )
            all_ok = False
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.error("[preflight] Remote write access: FAILED (%s)", exc)
        all_ok = False

    # 3. API reachability
    log.info("[preflight] Testing API at %s ...", api_url)
    url = f"{api_url}/api/cosmicgame/cst/list/all/0/1"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if str(data.get("status", 0)) == "1":
                log.info("[preflight] API connectivity: OK")
            else:
                log.error("[preflight] API connectivity: FAILED (status=%s)", data.get("status"))
                all_ok = False
    except Exception as exc:
        log.error("[preflight] API connectivity: FAILED (%s)", exc)
        all_ok = False

    # 4. Generator binary
    generator = resolve_generator(None)
    if generator:
        log.info("[preflight] Generator binary: OK (%s)", generator)
    else:
        log.warning("[preflight] Generator binary: NOT FOUND")
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
    p.add_argument("--ssh-host", default=os.environ.get(ENV_SSH_HOST),
                    help=f"Remote SSH host (env: {ENV_SSH_HOST})")
    p.add_argument("--ssh-user", default=os.environ.get(ENV_SSH_USER),
                    help=f"Remote SSH user (env: {ENV_SSH_USER})")
    p.add_argument("--api-url", default=os.environ.get(ENV_API_URL),
                    help=f"CosmicGame API base URL (env: {ENV_API_URL})")
    p.add_argument("--remote-dir", default=os.environ.get(ENV_REMOTE_DIR),
                    help=f"Remote asset directory (env: {ENV_REMOTE_DIR})")
    p.add_argument("--generator", default=None,
                    help="Path to generator binary (auto-detected if omitted)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"Per-seed generator timeout in seconds (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--dry-run", action="store_true",
                    help="Report missing files without generating or uploading")
    p.add_argument("--preflight", action="store_true",
                    help="Verify SSH access, remote write permissions, and API connectivity, then exit")
    return p.parse_args()


def validate_config(args: argparse.Namespace) -> list[str]:
    """Return list of missing required config values."""
    missing = []
    for attr, env_name in [
        ("ssh_host", ENV_SSH_HOST),
        ("ssh_user", ENV_SSH_USER),
        ("api_url", ENV_API_URL),
        ("remote_dir", ENV_REMOTE_DIR),
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
    log.info("  ssh_host   = %s", args.ssh_host)
    log.info("  ssh_user   = %s", args.ssh_user)
    log.info("  api_url    = %s", args.api_url)
    log.info("  remote_dir = %s", args.remote_dir)
    log.info("  timeout    = %s", fmt_duration(args.timeout))
    log.info("  dry_run    = %s", args.dry_run)
    log.info("  preflight  = %s", args.preflight)
    log.info("=" * 60)

    if args.preflight:
        ok = preflight(args.ssh_host, args.ssh_user, args.remote_dir, args.api_url)
        return 0 if ok else 1

    t_start = time.monotonic()

    exec_cmd = resolve_generator(args.generator)
    if exec_cmd is None and not args.dry_run:
        log.error("No generator available and not in dry-run mode. Exiting.")
        return 1

    LOCAL_IMG_DIR.mkdir(exist_ok=True)
    LOCAL_VID_DIR.mkdir(exist_ok=True)

    # --- Phase 1: discover what's missing ---

    try:
        seeds = fetch_token_seeds(args.api_url)
    except RuntimeError as exc:
        log.error("Fatal: %s", exc)
        return 1

    remote_files = list_remote_files(args.ssh_host, args.ssh_user, args.remote_dir)
    missing = find_missing_seeds(seeds, remote_files)

    if not missing:
        elapsed = time.monotonic() - t_start
        log.info(
            "All %d tokens have both PNG and MP4 on remote. Nothing to do. (%s)",
            len(seeds), fmt_duration(elapsed),
        )
        return 0

    log.info(
        "Found %d seeds with missing assets (out of %d total)",
        len(missing), len(seeds),
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
            seed, exec_cmd, args.ssh_host, args.ssh_user, args.remote_dir,
            args.timeout, args.dry_run,
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
    log.info("  Total seeds from API : %d", len(seeds))
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
