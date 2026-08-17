#!/usr/bin/env python3
"""Deploy the current branch to a remote server and run a viz batch there.

The batch runs fully detached on the server (nohup), so no persistent SSH
connection is needed: this script deploys the committed tree, uploads a
self-contained batch shell script (bootstrap, release build, one maximum
quality run per seed with ``--viz all``), launches it in the background,
and returns. Use ``--status`` to check progress later and ``--fetch`` to
download the finished artifacts.

Examples:
    python3 run_viz_batch.py                       # deploy + launch batch
    python3 run_viz_batch.py --status              # tail the remote log
    python3 run_viz_batch.py --fetch viz-results   # download artifacts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_HOST = "user@100.76.88.48"
DEFAULT_REMOTE_DIR = "viz-batch/CS-Image-Generation"
DEFAULT_SEEDS = "0xCAFE,0xBEEF,0xC0DE,0xFACE,0x1357"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
LOG_NAME = "viz_batch.log"
SCRIPT_NAME = "viz_batch.sh"

BATCH_TEMPLATE = """#!/usr/bin/env bash
set -u
echo "=== viz batch started $(date -u '+%Y-%m-%d %H:%M:%S') on $(hostname) ==="
echo "=== seeds: {seeds_display} ==="

if ! command -v cargo >/dev/null 2>&1; then
  if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
  else
    echo "=== cargo missing; installing rustup (minimal profile) ==="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \\
      || {{ echo "FATAL: rustup install failed"; exit 1; }}
    . "$HOME/.cargo/env"
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "=== ffmpeg missing; attempting passwordless apt install ==="
  if sudo -n apt-get update -qq && sudo -n apt-get install -y -qq ffmpeg; then
    echo "=== ffmpeg installed ==="
  else
    echo "FATAL: ffmpeg unavailable and passwordless install failed"
    exit 1
  fi
fi

echo "=== building release binary ($(nproc) cores) ==="
cargo build --release || {{ echo "FATAL: build failed"; exit 1; }}

BIN=./target/release/three_body_problem
FAILURES=0
for SEED in {seeds_shell}; do
  NAME="viz-${{SEED}}"
  echo "=== [$(date -u '+%H:%M:%S')] seed ${{SEED}} -> output/${{NAME}} ==="
  if ! "$BIN" --seed "${{SEED}}" --viz all --output "${{NAME}}"; then
    echo "WARN: seed ${{SEED}} failed"
    FAILURES=$((FAILURES + 1))
  fi
done

echo "=== viz batch COMPLETE $(date -u '+%Y-%m-%d %H:%M:%S') failures=${{FAILURES}} ==="
"""


def run(cmd: list[str], *, dry_run: bool, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command, echoing it first; honor --dry-run."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, check=check, text=True, capture_output=False)


def ssh_cmd(host: str, remote_command: str) -> list[str]:
    """Build an ssh invocation for one remote command."""
    return ["ssh", *SSH_OPTS, host, remote_command]


def preflight(host: str, dry_run: bool) -> None:
    """Verify passwordless SSH works and report the remote environment."""
    print("[1/4] preflight: ssh connectivity + remote environment")
    probe = (
        "echo ok && uname -sm && nproc && "
        "(command -v cargo || echo no-cargo) && (command -v ffmpeg || echo no-ffmpeg)"
    )
    run(ssh_cmd(host, probe), dry_run=dry_run)


def deploy(host: str, remote_dir: str, dry_run: bool) -> None:
    """Ship the committed tree (git archive of HEAD) to the server."""
    print("[2/4] deploy: git archive HEAD -> remote tree")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if dirty:
        print("ERROR: working tree has uncommitted changes; commit before deploying.")
        print(dirty)
        sys.exit(1)
    if dry_run:
        print(f"  $ git archive HEAD | ssh {host} 'mkdir -p {remote_dir} && tar -x -C ...'")
        return
    archive = subprocess.Popen(["git", "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
    extract = subprocess.run(
        ssh_cmd(host, f"mkdir -p {remote_dir} && tar -x -C {remote_dir}"),
        stdin=archive.stdout,
        check=True,
    )
    if archive.wait() != 0 or extract.returncode != 0:
        print("ERROR: deploy failed")
        sys.exit(1)


def upload_batch_script(host: str, remote_dir: str, seeds: list[str], dry_run: bool) -> None:
    """Generate and upload the self-contained batch script."""
    print("[3/4] upload batch script")
    script = BATCH_TEMPLATE.format(
        seeds_display=" ".join(seeds),
        seeds_shell=" ".join(seeds),
    )
    if dry_run:
        print(f"  (would write {len(script)} bytes to {remote_dir}/{SCRIPT_NAME})")
        return
    subprocess.run(
        ssh_cmd(host, f"cat > {remote_dir}/{SCRIPT_NAME}"),
        input=script,
        text=True,
        check=True,
    )


def launch(host: str, remote_dir: str, dry_run: bool) -> None:
    """Start the batch under nohup and confirm it is alive."""
    print("[4/4] launch batch in background (nohup; safe to disconnect)")
    command = (
        f"cd {remote_dir} && rm -f {LOG_NAME} && "
        f"nohup bash {SCRIPT_NAME} > {LOG_NAME} 2>&1 < /dev/null & echo launched pid=$!"
    )
    run(ssh_cmd(host, command), dry_run=dry_run)
    if dry_run:
        return
    time.sleep(5)
    print("--- first log lines ---")
    run(ssh_cmd(host, f"tail -n 20 {remote_dir}/{LOG_NAME}"), dry_run=False, check=False)
    print(
        "\nBatch is running detached. Check progress any time with:\n"
        f"  python3 run_viz_batch.py --host {host} --remote-dir {remote_dir} --status"
    )


def status(host: str, remote_dir: str) -> None:
    """Show the tail of the remote log and any running batch processes."""
    run(
        ssh_cmd(
            host,
            f"tail -n 40 {remote_dir}/{LOG_NAME} 2>/dev/null; echo '--- processes ---'; "
            "pgrep -af 'viz_batch.sh|three_body_problem' || echo '(none running)'",
        ),
        dry_run=False,
        check=False,
    )


def fetch(host: str, remote_dir: str, destination: str) -> None:
    """Download all finished viz packages into a local directory."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    run(
        ["scp", "-r", *SSH_OPTS, f"{host}:{remote_dir}/output/viz-*", str(dest)],
        dry_run=False,
        check=False,
    )
    print(f"Fetched into {dest}/")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="ssh target (user@host)")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="remote checkout path")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="comma-separated hex seeds")
    parser.add_argument("--status", action="store_true", help="show remote progress and exit")
    parser.add_argument("--fetch", metavar="DIR", help="download finished artifacts into DIR")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = parser.parse_args()

    if args.status:
        status(args.host, args.remote_dir)
        return
    if args.fetch:
        fetch(args.host, args.remote_dir, args.fetch)
        return

    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        print("ERROR: no seeds given")
        sys.exit(1)

    preflight(args.host, args.dry_run)
    deploy(args.host, args.remote_dir, args.dry_run)
    upload_batch_script(args.host, args.remote_dir, seeds, args.dry_run)
    launch(args.host, args.remote_dir, args.dry_run)


if __name__ == "__main__":
    main()
