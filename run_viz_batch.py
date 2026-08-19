#!/usr/bin/env python3
"""Deploy, operate, and fetch the continuous random visualization farm.

The remote checkout is deployed from committed ``HEAD``. Its launcher
bootstraps dependencies, compiles the Rust binary once with the optimized
release profile, verifies all 69 modes are present, then starts
``viz_farm.py`` fully detached.

Examples:
    python3 run_viz_batch.py
    python3 run_viz_batch.py --status
    python3 run_viz_batch.py --stop
    python3 run_viz_batch.py --force-stop
    python3 run_viz_batch.py --fetch ../CS-viz-random-results
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

DEFAULT_HOST = "user@100.76.88.48"
DEFAULT_REMOTE_DIR = "viz-farm/CS-Image-Generation"
DEFAULT_CONCURRENCY = 4
DEFAULT_THREADS_PER_JOB = 30
DEFAULT_MIN_FREE_GB = 500.0
DEFAULT_MAX_FAILURE_STREAK = 5
DEFAULT_TIMEOUT_HOURS = 24.0 * 14.0
EXPECTED_MODE_COUNT = 69

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
LAUNCH_SCRIPT_NAME = "viz_farm_launch.sh"


def run(
    command: list[str],
    *,
    dry_run: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, echoing it first; honor ``--dry-run``."""
    print(f"  $ {shlex.join(command)}")
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return subprocess.run(command, check=check, text=True, capture_output=False)


def ssh_cmd(host: str, remote_command: str) -> list[str]:
    """Build one noninteractive SSH invocation."""
    return ["ssh", *SSH_OPTS, host, remote_command]


def local_git_head() -> str:
    """Return the committed source revision being archived."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def preflight(host: str, dry_run: bool) -> None:
    """Verify passwordless SSH and report the remote environment."""
    print("[1/4] preflight: SSH connectivity + remote environment")
    probe = (
        "echo ok && uname -sm && nproc && "
        "(command -v cargo || echo no-cargo) && "
        "(command -v ffmpeg || echo no-ffmpeg) && "
        "(command -v python3 || echo no-python3)"
    )
    run(ssh_cmd(host, probe), dry_run=dry_run)


def deploy(host: str, remote_dir: str, dry_run: bool) -> None:
    """Ship committed HEAD to the remote tree without touching output."""
    print("[2/4] deploy: git archive HEAD -> remote tree")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        print("ERROR: working tree has uncommitted changes; commit before deploying.")
        print(dirty)
        raise SystemExit(1)

    quoted_dir = shlex.quote(remote_dir)
    if dry_run:
        print(f"  $ git archive HEAD | ssh {host} 'mkdir -p {quoted_dir} && tar -x -C ...'")
        return
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
    )
    extract = subprocess.run(
        ssh_cmd(host, f"mkdir -p {quoted_dir} && tar -x -C {quoted_dir}"),
        stdin=archive.stdout,
        check=True,
    )
    if archive.stdout is not None:
        archive.stdout.close()
    if archive.wait() != 0 or extract.returncode != 0:
        print("ERROR: deploy failed")
        raise SystemExit(1)


def build_launch_script(args: argparse.Namespace, git_head: str | None = None) -> str:
    """Return the deterministic remote bootstrap/launch script."""
    max_jobs_arg = f" --max-jobs {args.max_jobs}" if args.max_jobs is not None else ""
    deployed_head = git_head or local_git_head()
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== random viz farm bootstrap $(date -u '+%Y-%m-%d %H:%M:%S') on $(hostname) ==="

if ! command -v cargo >/dev/null 2>&1; then
  if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
  else
    echo "=== cargo missing; installing rustup (minimal profile) ==="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \\
      sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "=== ffmpeg missing; attempting passwordless apt install ==="
  sudo -n apt-get update -qq
  sudo -n apt-get install -y -qq ffmpeg
fi

echo "=== building maximum-performance release binary ($(nproc) cores) ==="
cargo build --release --locked

MODE_COUNT=$(./target/release/three_body_problem --viz-list | \\
  awk '$5 == "implemented" {{ count += 1 }} END {{ print count + 0 }}')
if [ "$MODE_COUNT" -ne {EXPECTED_MODE_COUNT} ]; then
  echo "FATAL: expected {EXPECTED_MODE_COUNT} implemented modes, got $MODE_COUNT"
  exit 1
fi
echo "=== verified $MODE_COUNT implemented visualization modes ==="

mkdir -p orchestrator/jobs output
rm -f orchestrator/STOP
if pgrep -f '^python3 viz_farm.py( |$)' >/dev/null 2>&1 || \\
   pgrep -f '[t]hree_body_problem' >/dev/null 2>&1; then
  echo "FATAL: farm supervisor or orphan Rust jobs are already running"
  exit 1
fi

setsid nohup python3 viz_farm.py \\
  --git-head {shlex.quote(deployed_head)} \\
  --concurrency {args.concurrency} \\
  --threads-per-job {args.threads_per_job} \\
  --min-free-gb {args.min_free_gb:.3f} \\
  --max-failure-streak {args.max_failure_streak} \\
  --timeout-hours {args.timeout_hours:.3f}{max_jobs_arg} \\
  > orchestrator/launcher.log 2>&1 < /dev/null &
FARM_PID=$!
echo "$FARM_PID" > orchestrator/launcher.pid
sleep 3
if ! kill -0 "$FARM_PID" 2>/dev/null; then
  echo "FATAL: viz_farm.py failed to stay alive"
  tail -n 80 orchestrator/launcher.log || true
  exit 1
fi
echo "=== farm launched pid=$FARM_PID: {args.concurrency} jobs x \\
{args.threads_per_job} Rayon threads, {args.min_free_gb:.0f} GB floor ==="
"""


def upload_launch_script(
    host: str,
    remote_dir: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    """Upload the generated release-build and farm-launch script."""
    print("[3/4] upload farm launcher")
    script = build_launch_script(args, local_git_head())
    remote_path = shlex.quote(f"{remote_dir}/{LAUNCH_SCRIPT_NAME}")
    if dry_run:
        print(f"  (would write {len(script)} bytes to {remote_path})")
        return
    subprocess.run(
        ssh_cmd(host, f"cat > {remote_path} && chmod +x {remote_path}"),
        input=script,
        text=True,
        check=True,
    )


def launch(host: str, remote_dir: str, dry_run: bool) -> None:
    """Build and start the farm, then display its initial state."""
    print("[4/4] release build + detached farm launch")
    quoted_dir = shlex.quote(remote_dir)
    run(
        ssh_cmd(host, f"cd {quoted_dir} && ./{LAUNCH_SCRIPT_NAME}"),
        dry_run=dry_run,
    )
    if dry_run:
        return
    print("\n--- initial farm status ---")
    status(host, remote_dir)
    print(
        "\nFarm is detached. Check it with:\n"
        f"  python3 run_viz_batch.py --host {host} "
        f"--remote-dir {remote_dir} --status"
    )


def status(host: str, remote_dir: str) -> None:
    """Show atomic farm state, recent logs, load/RSS, processes, and disk."""
    quoted_dir = shlex.quote(remote_dir)
    command = (
        f"cd {quoted_dir} 2>/dev/null || "
        "{ echo 'farm checkout missing'; exit 0; }; "
        "echo '--- state ---'; "
        "if [ -f orchestrator/state.json ]; then "
        "python3 -m json.tool orchestrator/state.json; "
        "else echo '(state unavailable)'; fi; "
        "echo '--- recent session log ---'; "
        "tail -n 16 orchestrator/session.log 2>/dev/null || true; "
        "echo '--- system ---'; uptime; "
        "ps -C three_body_problem -o pid=,pcpu=,rss=,etime=,args= 2>/dev/null || true; "
        "echo '--- aggregate RSS ---'; "
        "ps -C three_body_problem -o rss= 2>/dev/null | "
        "awk '{ total += $1 } END { printf \"%.1f GiB\\n\", total / 1048576 }'; "
        "echo '--- disk ---'; df -h .; du -sh output 2>/dev/null || true; "
        "echo '--- farm process ---'; "
        "pgrep -af '^python3 viz_farm.py( |$)' || echo '(farm not running)'"
    )
    run(ssh_cmd(host, command), dry_run=False, check=False)


def build_stop_command(remote_dir: str, *, force: bool) -> str:
    """Build a stop command whose process patterns cannot match its shell."""
    quoted_dir = shlex.quote(remote_dir)
    if force:
        return (
            f"cd {quoted_dir} 2>/dev/null || exit 0; "
            "touch orchestrator/STOP; "
            "pkill -TERM -f '^python3 viz_farm.py( |$)' 2>/dev/null || true; "
            "pkill -TERM -f '[t]hree_body_problem' 2>/dev/null || true; "
            "sleep 5; "
            "pkill -KILL -f '^python3 viz_farm.py( |$)' 2>/dev/null || true; "
            "pkill -KILL -f '[t]hree_body_problem' 2>/dev/null || true; "
            "echo 'forced stop complete'; "
            "pgrep -af '[v]iz_farm.py|[t]hree_body_problem' || true"
        )
    return (
        f"cd {quoted_dir} 2>/dev/null || exit 0; "
        "mkdir -p orchestrator; touch orchestrator/STOP; "
        "echo 'graceful drain requested (no new jobs will launch)'"
    )


def stop(host: str, remote_dir: str, *, force: bool) -> None:
    """Request a graceful drain, or forcibly stop every farm child."""
    command = build_stop_command(remote_dir, force=force)
    run(ssh_cmd(host, command), dry_run=False, check=False)


def fetch(host: str, remote_dir: str, destination: str) -> None:
    """Resumably merge every remote output package into a local directory."""
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    ssh_transport = "ssh " + " ".join(shlex.quote(option) for option in SSH_OPTS)
    source = f"{host}:{remote_dir}/output/"
    run(
        [
            "rsync",
            "-a",
            "--partial",
            "--stats",
            "-e",
            ssh_transport,
            source,
            f"{destination_path}/",
        ],
        dry_run=False,
        check=False,
    )
    print(f"Fetched into {destination_path}/")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH target (user@host)")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="remote checkout path")
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=DEFAULT_CONCURRENCY,
        help="simultaneous Rust jobs",
    )
    parser.add_argument(
        "--threads-per-job",
        type=positive_int,
        default=DEFAULT_THREADS_PER_JOB,
        help="RAYON_NUM_THREADS for each Rust job",
    )
    parser.add_argument(
        "--min-free-gb",
        type=positive_float,
        default=DEFAULT_MIN_FREE_GB,
        help="stop-launching disk floor",
    )
    parser.add_argument(
        "--max-failure-streak",
        type=positive_int,
        default=DEFAULT_MAX_FAILURE_STREAK,
    )
    parser.add_argument(
        "--timeout-hours",
        type=positive_float,
        default=DEFAULT_TIMEOUT_HOURS,
    )
    parser.add_argument("--max-jobs", type=positive_int, help="optional finite validation run")
    parser.add_argument("--status", action="store_true", help="show remote farm status and exit")
    parser.add_argument("--stop", action="store_true", help="request graceful drain and exit")
    parser.add_argument("--force-stop", action="store_true", help="kill farm and Rust children")
    parser.add_argument("--fetch", metavar="DIR", help="resumably download all output packages")
    parser.add_argument("--dry-run", action="store_true", help="print deploy actions only")
    args = parser.parse_args()

    action_count = sum(
        [
            args.status,
            args.stop,
            args.force_stop,
            args.fetch is not None,
        ]
    )
    if action_count > 1:
        parser.error("choose only one of --status, --stop, --force-stop, or --fetch")
    if args.status:
        status(args.host, args.remote_dir)
        return
    if args.stop:
        stop(args.host, args.remote_dir, force=False)
        return
    if args.force_stop:
        stop(args.host, args.remote_dir, force=True)
        return
    if args.fetch:
        fetch(args.host, args.remote_dir, args.fetch)
        return

    preflight(args.host, args.dry_run)
    deploy(args.host, args.remote_dir, args.dry_run)
    upload_launch_script(args.host, args.remote_dir, args, args.dry_run)
    launch(args.host, args.remote_dir, args.dry_run)


if __name__ == "__main__":
    main()
