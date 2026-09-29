#!/usr/bin/env bash
# One-time root bootstrap of the CosmicSignature generator host for continuous deployment.
#
# Usage: sudo ops/server/bootstrap-root.sh [SERVICE_USER [CHECKOUT]]
#
#   SERVICE_USER  the unprivileged user that owns the checkout and runs the sync
#                 (default: the user who ran sudo, $SUDO_USER)
#   CHECKOUT      the production checkout, only used to print the next commands
#                 (default: the checkout holding this script, else the legacy unit's
#                 WorkingDirectory=)
#
# This is the ONLY step of the deployment that needs root (docs/deployment.md). It is
# idempotent: running it again after a success changes nothing. It:
#   1. stops and disables the legacy SYSTEM timer cosmicsig-sync.timer, so it starts no new run;
#   2. waits, without a time limit, until the legacy cosmicsig-sync.service has finished a run
#      in flight: a render is never interrupted;
#   3. removes /etc/systemd/system/cosmicsig-sync.{service,timer} and reloads systemd, since
#      the user units of ops/systemd/ replace them;
#   4. enables lingering for SERVICE_USER, so that user's systemd manager, and with it the sync
#      and deploy timers, runs from boot without a login session;
#   5. prints the unprivileged commands that finish the setup.
# It never touches the checkout, its production state (.env, output/, logs, ledgers) or the
# deploy agent's state.
#
# COSMICSIG_BOOTSTRAP_UNIT_DIR and COSMICSIG_BOOTSTRAP_POLL_SECONDS exist for the tests
# (tests/python/test_deploy.py) only.

set -euo pipefail

readonly UNIT_DIR="${COSMICSIG_BOOTSTRAP_UNIT_DIR:-/etc/systemd/system}"
readonly POLL_SECONDS="${COSMICSIG_BOOTSTRAP_POLL_SECONDS:-5}"
readonly PROGRESS_SECONDS=60
readonly SERVICE=cosmicsig-sync.service
readonly TIMER=cosmicsig-sync.timer
# ActiveState values while a run is (still) in progress.
readonly BUSY_STATES='^(active|activating|deactivating|reloading|refreshing)$'

log() {
    printf 'bootstrap: %s\n' "$*"
}

die() {
    printf 'bootstrap: error: %s\n' "$*" >&2
    exit 1
}

usage() {
    sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# The ActiveState of the legacy system service ("inactive" when it does not exist).
legacy_state() {
    systemctl is-active "$SERVICE" 2>/dev/null || true
}

# One property of a system unit ("" if systemctl cannot tell).
unit_property() {
    systemctl show --property="$1" --value "$2" 2>/dev/null || true
}

# The production checkout, for the printed instructions.
find_checkout() {
    local script_dir candidate working_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    candidate="$(cd -- "$script_dir/../.." && pwd)"
    if [[ -f "$candidate/ops/deploy/cosmicsig_deploy.py" && -f "$candidate/run.py" ]]; then
        printf '%s\n' "$candidate"
        return
    fi
    working_dir="$(unit_property WorkingDirectory "$SERVICE")"
    # systemctl marks a directory that may be missing with a leading "!".
    working_dir="${working_dir#!}"
    if [[ "$working_dir" == /* ]]; then
        printf '%s\n' "$working_dir"
        return
    fi
    printf '%s\n' "<checkout>"
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        return 0
    fi
    local uid
    uid="$(id -u)"
    if [[ "$uid" != 0 ]]; then
        die "run this with sudo: sudo $0 [SERVICE_USER [CHECKOUT]]"
    fi
    command -v systemctl >/dev/null || die "systemctl not found: this host does not run systemd"
    command -v loginctl >/dev/null ||
        die "loginctl not found: this host does not run systemd-logind"

    local user="${1:-${SUDO_USER:-}}"
    [[ -n "$user" ]] || die "name the service user: sudo $0 SERVICE_USER"
    [[ "$user" != root ]] || die "the service user must not be root"
    id -u -- "$user" >/dev/null 2>&1 || die "no such user: $user"

    # Read the checkout before step 3 removes the legacy unit that may name it.
    local checkout="${2:-}"
    if [[ -z "$checkout" ]]; then
        checkout="$(find_checkout)"
    fi

    # 1. No new legacy run.
    local timer_load_state
    timer_load_state="$(unit_property LoadState "$TIMER")"
    if [[ "$timer_load_state" == loaded ]]; then
        log "stopping and disabling the legacy system timer $TIMER"
        systemctl disable --now "$TIMER"
    else
        log "the legacy system timer $TIMER is not loaded"
    fi

    # 2. Let a run in flight finish.
    local state next_progress=0
    state="$(legacy_state)"
    while [[ "$state" =~ $BUSY_STATES ]]; do
        if ((SECONDS >= next_progress)); then
            log "waiting for the running legacy sync to finish" \
                "($SERVICE is $state; it is never interrupted) ..."
            next_progress=$((SECONDS + PROGRESS_SECONDS))
        fi
        sleep "$POLL_SECONDS"
        state="$(legacy_state)"
    done
    log "no legacy sync run is in progress"

    # 3. Remove the legacy units.
    local unit removed=0
    for unit in "$SERVICE" "$TIMER"; do
        if [[ -e "$UNIT_DIR/$unit" || -L "$UNIT_DIR/$unit" ]]; then
            rm -f -- "$UNIT_DIR/$unit"
            log "removed $UNIT_DIR/$unit"
            removed=1
        fi
    done
    if ((removed)); then
        systemctl daemon-reload
        systemctl reset-failed "$SERVICE" "$TIMER" >/dev/null 2>&1 || true
    else
        log "no legacy unit files in $UNIT_DIR"
    fi

    # 4. Lingering: the user manager runs from boot and survives logouts.
    loginctl enable-linger "$user"
    log "lingering enabled for $user"

    # 5. What is left is unprivileged.
    cat <<EOF

Done. Finish the setup as $user, in a login session (for example \`ssh $user@<this host>\`;
\`sudo -u $user\` lacks the session that \`systemctl --user\` needs):

  cd $checkout
  git status --short --untracked-files=no      # must print nothing
  git switch main
  git fetch origin && git merge --ff-only origin/main
  python3 ops/deploy/cosmicsig_deploy.py install
  python3 ops/deploy/cosmicsig_deploy.py status
  journalctl --user -u cosmicsig-deploy -u cosmicsig-sync -f

The first deploy tick builds and tests origin/main, installs the binary and enables the sync
timer; until then no sync runs. See docs/deployment.md.
EOF
}

main "$@"
