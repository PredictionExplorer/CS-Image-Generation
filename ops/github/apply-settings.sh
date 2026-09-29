#!/usr/bin/env bash
# Reconcile the GitHub settings of this repository that live outside the git tree with the
# desired state declared in this file and in ruleset-main.json: the merge policy, labels,
# Dependabot, secret scanning, private vulnerability reporting, the GitHub Actions policy,
# CodeQL default setup and the `main` branch ruleset.
#
# Keeping the desired state in the repository means every change to it is reviewed in a pull
# request like code. Each setting is read first and written only when it differs, so the script
# is idempotent and safe to re-run; `--verify` detects drift introduced through the web UI.
# See ops/github/README.md.
#
# Portable to the bash 3.2 that macOS ships: no associative arrays, no mapfile, no `((i++))`
# (which returns 1 under `set -e` when i is 0), and no expansion of possibly empty arrays.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ops/github/apply-settings.sh [--dry-run | --verify] [--repo OWNER/NAME]

Reconcile the repository's GitHub settings with the desired state in this script and in
ops/github/ruleset-main.json.

  (no option)        apply: change only the settings that differ, then verify everything
  --dry-run          read the current settings and print the exact write calls an apply
                     would make; sends GET requests only and never changes anything
  --verify           read every setting back and report drift; exit 1 on any drift
  --repo OWNER/NAME  target repository (default: PredictionExplorer/CS-Image-Generation)
  -h, --help         show this help

Requires gh (https://cli.github.com), authenticated as a repository admin, and jq.
Exit status: 0 in sync (or dry run done), 1 drift or failed API call, 2 usage error or
missing prerequisite.
EOF
}

# ---------------------------------------------------------------------------------------------
# Desired state
# ---------------------------------------------------------------------------------------------

readonly DEFAULT_REPO="PredictionExplorer/CS-Image-Generation"
readonly API_VERSION="2022-11-28"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly RULESET_FILE="$SCRIPT_DIR/ruleset-main.json"

# Merge policy: pull requests land as a squash (the PR title becomes the commit subject) or a
# rebase, never as a merge commit, which keeps `main` linear for the ruleset's
# required_linear_history rule and for the deploy agent's fast-forward-only pulls. Auto-merge
# lets a PR merge itself once `CI passed` is green; merged head branches are deleted; and the
# "Update branch" button is always offered because the ruleset requires up-to-date branches.
#
# A squash commit's body starts BLANK rather than as the PR body: GitHub starts no push run for
# a commit whose message contains [skip ci], [ci skip], [no ci], [skip actions] or
# [actions skip], and a PR body can hold one (Dependabot's pull request descriptions quote
# upstream commit messages) or gain one after CI passed. The commit on `main` would then never
# get a `CI passed` check, and the deploy agent would wait for it. CI's title check rejects
# those markers in the title.
readonly MERGE_SETTINGS='{
  "allow_squash_merge": true,
  "allow_rebase_merge": true,
  "allow_merge_commit": false,
  "allow_auto_merge": true,
  "delete_branch_on_merge": true,
  "allow_update_branch": true,
  "squash_merge_commit_title": "PR_TITLE",
  "squash_merge_commit_message": "BLANK"
}'

# The GITHUB_TOKEN of every workflow starts read-only (each job asks for what it needs with a
# `permissions:` block) and can never approve a pull request, so a compromised workflow step
# cannot approve its own change into `main`.
readonly WORKFLOW_TOKEN_SETTINGS='{
  "default_workflow_permissions": "read",
  "can_approve_pull_request_reviews": false
}'

# Every action must be referenced by a full commit SHA (the workflows pin them, Dependabot
# bumps the SHAs), so a moved or hijacked tag cannot change what CI runs.
readonly ACTIONS_POLICY='{
  "enabled": true,
  "allowed_actions": "all",
  "sha_pinning_required": true
}'

# Workflows triggered by pull requests from forks wait for a maintainer's approval whenever the
# author is not a collaborator, not only on their first contribution.
readonly FORK_PR_APPROVAL='{
  "approval_policy": "all_external_contributors"
}'

# CodeQL default setup: GitHub-managed code scanning (no workflow file) of the workflows, the
# Python scripts and the Rust crate, on pushes to main, on pull requests and weekly. The
# ruleset's code_scanning rule requires its results, so it is configured before the ruleset.
readonly CODEQL_SETUP='{
  "state": "configured",
  "languages": ["actions", "python", "rust"],
  "query_suite": "default"
}'
# Rust is generally available in CodeQL and GitHub detects it here, but the documented REST
# enum for `languages` does not list `rust` yet. If GitHub rejects the body above (HTTP 422),
# it is retried without `languages`, which leaves the choice to GitHub's language detection
# (the GET of the unconfigured setup lists actions, python and rust). Verification still
# requires all three, and says how to add a missing one in the web UI.
readonly CODEQL_SETUP_AUTODETECT='{
  "state": "configured",
  "query_suite": "default"
}'

# Labels as "name|color|description". The issue forms apply bug or enhancement plus
# needs-triage, and GitHub silently drops a form's label that does not exist, so every label
# named in .github/ISSUE_TEMPLATE/ (and any `labels:` added to .github/dependabot.yml) must be
# listed here. dependencies is the label Dependabot puts on its pull requests by default, with
# Dependabot's own colour and description; ci and security are for triage by hand. bug and
# enhancement keep GitHub's default colours and descriptions.
LABELS=(
    "bug|d73a4a|Something isn't working"
    "enhancement|a2eeef|New feature or request"
    "needs-triage|fbca04|Awaiting a first look from a maintainer"
    "dependencies|0366d6|Pull requests that update a dependency file"
    "ci|1d76db|CI workflows, hooks and developer tooling"
    "security|b60205|Security fix or hardening"
)
readonly LABELS

# ---------------------------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------------------------

MODE=apply
REPO=$DEFAULT_REPO
FAILURES=0
WARNINGS=0
CHANGES=0
VERIFY_AFTER_APPLY=0
WRITTEN="|" # labels of the settings written in this run, each followed by "|"
# Notes added to a drift report: PENDING_NOTE when the verification after an apply still sees
# a setting written in the same run, DRIFT_NOTE always. A section overrides them with `local`
# (bash scopes locals dynamically, so reconcile sees the caller's values).
PENDING_NOTE="GitHub accepted the change in this run but reports another value."
DRIFT_NOTE=""
ERR_FILE=""      # stderr of the last gh call
WRITE_ERROR=""   # error message of the last failed write
# Set by an apply that could not configure CodeQL default setup: the ruleset is then written
# without its code_scanning rule, which would otherwise block every merge (see reconcile_codeql).
CODEQL_UNAVAILABLE=0

readonly JQ_DEFS='def canon: walk(if type == "array" then sort else . end);'

cleanup() {
    if [[ -n $ERR_FILE ]]; then
        rm -f -- "$ERR_FILE"
    fi
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

heading() {
    printf '\n== %s\n' "$1"
}

# status_line TAG LABEL: one aligned result line (ok, change, DRIFT, ERROR, warning).
status_line() {
    printf '%-8s %s\n' "$1" "$2"
}

# details TEXT: indent a multi-line explanation under the preceding status line.
details() {
    if [[ -n $1 ]]; then
        printf '%s\n' "$1" | sed 's/^/           /'
    fi
}

# problem SEVERITY TAG LABEL MESSAGE: record a failed check. A required setting counts as a
# failure (exit status 1); an optional one (a feature GitHub may not offer this repository, or
# a change that completes asynchronously) only as a warning.
problem() {
    local severity=$1 tag=$2 label=$3 message=$4
    if [[ $severity == required ]]; then
        FAILURES=$((FAILURES + 1))
        status_line "$tag" "$label"
    else
        WARNINGS=$((WARNINGS + 1))
        status_line warning "$label"
    fi
    details "$message"
}

# gh_api METHOD PATH [BODY]: one REST call; the response goes to stdout, gh's error message to
# $ERR_FILE. BODY, when given, is sent as the JSON request body.
gh_api() {
    local method=$1 path=$2 body=${3-}
    local -a args=(api --method "$method"
        --header "Accept: application/vnd.github+json"
        --header "X-GitHub-Api-Version: $API_VERSION")
    if [[ -n $body ]]; then
        gh "${args[@]}" --input - "$path" <<<"$body" 2>"$ERR_FILE"
    else
        gh "${args[@]}" "$path" </dev/null 2>"$ERR_FILE"
    fi
}

# api_error [RESPONSE]: the last call's error on one line, plus GitHub's validation details.
api_error() {
    local message detail=""
    message=$(tr '\n' ' ' <"$ERR_FILE" | sed 's/[[:space:]]*$//')
    if [[ -n ${1-} ]]; then
        detail=$(jq -c '.errors // empty' <<<"$1" 2>/dev/null || true)
    fi
    printf '%s%s' "${message:-unknown error}" "${detail:+ $detail}"
}

last_status_is() {
    grep -q "(HTTP $1)" "$ERR_FILE"
}

# write LABEL METHOD PATH [BODY]: perform one mutating call, or print it under --dry-run.
write() {
    local label=$1 method=$2 path=$3 body=${4-} response
    if [[ $MODE == dry-run ]]; then
        if [[ -n $body ]]; then
            printf "           would run: gh api --method %s %s --input - <<'JSON'\n" \
                "$method" "$path"
            jq . <<<"$body" | sed 's/^/           /'
            printf '           JSON\n'
        else
            printf '           would run: gh api --method %s %s\n' "$method" "$path"
        fi
        CHANGES=$((CHANGES + 1))
        return 0
    fi
    if response=$(gh_api "$method" "$path" "$body"); then
        CHANGES=$((CHANGES + 1))
        WRITTEN="$WRITTEN$label|"
        return 0
    fi
    WRITE_ERROR=$(api_error "$response")
    return 1
}

written_in_this_run() {
    [[ $WRITTEN == *"|$1|"* ]]
}

# diff_json WANT HAVE: one line per key of WANT whose value differs in HAVE (arrays compare as
# sets); nothing when HAVE contains WANT. Keys GitHub returns beyond WANT are ignored.
diff_json() {
    jq -rn --argjson want "$1" --argjson have "$2" "$JQ_DEFS"'
        $want | to_entries[]
        | select((.value | canon) != ($have[.key] | canon))
        | "\(.key): want \(.value | tojson), have \($have[.key] | tojson)"'
}

# reconcile SEVERITY LABEL WANT READER METHOD PATH BODY [HINT] [FALLBACK_BODY]
#
# Compare WANT with the JSON printed by the READER function and, depending on the mode, report
# the difference (--verify), print the write that would fix it (--dry-run) or make it (apply):
# `gh api --method METHOD PATH` with BODY as the request body (empty: no body). HINT is added
# to the report when the write fails. When GitHub rejects BODY as invalid (HTTP 422) and a
# FALLBACK_BODY is given, the write is retried once with it.
reconcile() {
    local severity=$1 label=$2 want=$3 reader=$4 method=$5 path=$6 body=$7 hint=${8-}
    local fallback=${9-} have delta
    if ! have=$("$reader"); then
        if [[ $MODE == dry-run ]]; then
            status_line unknown "$label"
            details "cannot read the current state: $(api_error)"
            write "$label" "$method" "$path" "$body"
        else
            problem "$severity" ERROR "$label" "cannot read the current state: $(api_error)"
        fi
        return 0
    fi
    if ! jq -e 'type == "object"' >/dev/null 2>&1 <<<"$have"; then
        problem "$severity" ERROR "$label" "unexpected response from GitHub: ${have:0:200}"
        return 0
    fi
    delta=$(diff_json "$want" "$have")
    if [[ -z $delta ]]; then
        status_line ok "$label"
        return 0
    fi
    case $MODE in
        verify)
            if [[ $VERIFY_AFTER_APPLY == 1 ]] && written_in_this_run "$label"; then
                delta+=$'\n'"$PENDING_NOTE"
            fi
            problem "$severity" DRIFT "$label" "$delta${DRIFT_NOTE:+$'\n'$DRIFT_NOTE}"
            ;;
        dry-run)
            status_line change "$label"
            details "$delta"
            write "$label" "$method" "$path" "$body"
            if [[ -n $fallback ]]; then
                details "if GitHub rejects that body (HTTP 422), retry once with:"
                details "$(jq -c . <<<"$fallback")"
            fi
            ;;
        apply)
            status_line change "$label"
            details "$delta"
            if write "$label" "$method" "$path" "$body"; then
                return 0
            fi
            if [[ -n $fallback ]] && last_status_is 422; then
                details "GitHub rejected the body ($WRITE_ERROR)"
                details "retrying with: $(jq -c . <<<"$fallback")"
                if write "$label" "$method" "$path" "$fallback"; then
                    return 0
                fi
            fi
            problem "$severity" ERROR "$label" "$WRITE_ERROR${hint:+$'\n'$hint}"
            ;;
        *) die "internal error: unknown mode $MODE" ;;
    esac
}

# ---------------------------------------------------------------------------------------------
# Readers: print the current state of one setting as JSON in the shape of its desired state
# ---------------------------------------------------------------------------------------------

read_repository() {
    gh_api GET "repos/$REPO"
}

read_security_features() {
    gh_api GET "repos/$REPO" | jq -c '(.security_and_analysis // {}) | map_values(.status)'
}

# GitHub answers 204 when Dependabot alerts are enabled and 404 when they are disabled.
read_vulnerability_alerts() {
    if gh_api GET "repos/$REPO/vulnerability-alerts" >/dev/null; then
        printf '{"enabled":true}\n'
    elif last_status_is 404; then
        printf '{"enabled":false}\n'
    else
        return 1
    fi
}

read_automated_security_fixes() {
    gh_api GET "repos/$REPO/automated-security-fixes"
}

read_private_vulnerability_reporting() {
    gh_api GET "repos/$REPO/private-vulnerability-reporting"
}

read_workflow_token() {
    gh_api GET "repos/$REPO/actions/permissions/workflow"
}

read_actions_policy() {
    gh_api GET "repos/$REPO/actions/permissions"
}

read_fork_pr_approval() {
    gh_api GET "repos/$REPO/actions/permissions/fork-pr-contributor-approval"
}

read_codeql_setup() {
    gh_api GET "repos/$REPO/code-scanning/default-setup"
}

# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------

reconcile_merge_policy() {
    heading "Merge policy"
    reconcile required "merge settings" "$MERGE_SETTINGS" read_repository \
        PATCH "repos/$REPO" "$MERGE_SETTINGS"
}

reconcile_labels() {
    heading "Labels"
    local entry name color description want have encoded delta
    # A label is created with its name, colour and description, and updated (PATCH by name)
    # with only the colour and description, so renaming a label here creates a new one.
    for entry in "${LABELS[@]}"; do
        IFS='|' read -r name color description <<<"$entry"
        want=$(jq -cn --arg name "$name" --arg color "$color" --arg description "$description" \
            '{name: $name, color: $color, description: $description}')
        encoded=$(jq -rn --arg name "$name" '$name | @uri')
        if have=$(gh_api GET "repos/$REPO/labels/$encoded"); then
            have=$(jq -c '{name, color, description}' <<<"$have")
        elif last_status_is 404; then
            have='{}'
        elif [[ $MODE == dry-run ]]; then
            status_line unknown "label $name"
            details "cannot read the current state: $(api_error)"
            write "label $name" POST "repos/$REPO/labels" "$want"
            continue
        else
            problem required ERROR "label $name" "cannot read the current state: $(api_error)"
            continue
        fi
        if [[ $have == '{}' ]]; then
            delta="missing"
        else
            delta=$(diff_json "$want" "$have")
        fi
        if [[ -z $delta ]]; then
            status_line ok "label $name"
        elif [[ $MODE == verify ]]; then
            problem required DRIFT "label $name" "$delta"
        else
            status_line change "label $name"
            details "$delta"
            if [[ $have == '{}' ]]; then
                write "label $name" POST "repos/$REPO/labels" "$want" ||
                    problem required ERROR "label $name" "$WRITE_ERROR"
            else
                write "label $name" PATCH "repos/$REPO/labels/$encoded" \
                    "$(jq -c '{color, description}' <<<"$want")" ||
                    problem required ERROR "label $name" "$WRITE_ERROR"
            fi
        fi
    done
}

reconcile_dependabot() {
    heading "Dependabot"
    # Alerts first: security updates need them.
    reconcile required "Dependabot alerts" '{"enabled":true}' read_vulnerability_alerts \
        PUT "repos/$REPO/vulnerability-alerts" ""
    reconcile required "Dependabot security updates" '{"enabled":true}' \
        read_automated_security_fixes PUT "repos/$REPO/automated-security-fixes" ""
}

# secret_scanning_feature SEVERITY FEATURE: enable one security_and_analysis feature. Each
# feature is its own PATCH so an optional one GitHub refuses (HTTP 422: not available for this
# repository or plan) cannot stop the others.
secret_scanning_feature() {
    local severity=$1 feature=$2
    reconcile "$severity" "$feature" "{\"$feature\":\"enabled\"}" read_security_features \
        PATCH "repos/$REPO" "{\"security_and_analysis\":{\"$feature\":{\"status\":\"enabled\"}}}" \
        "GitHub may not offer this feature to this repository; it is optional."
}

reconcile_secret_scanning() {
    heading "Secret scanning"
    # Secret scanning first: push protection builds on it.
    secret_scanning_feature required secret_scanning
    secret_scanning_feature required secret_scanning_push_protection
    secret_scanning_feature optional secret_scanning_non_provider_patterns
    secret_scanning_feature optional secret_scanning_validity_checks
}

reconcile_vulnerability_reporting() {
    heading "Private vulnerability reporting"
    reconcile required "private vulnerability reporting" '{"enabled":true}' \
        read_private_vulnerability_reporting PUT "repos/$REPO/private-vulnerability-reporting" ""
}

reconcile_actions() {
    heading "GitHub Actions"
    reconcile required "default workflow token" "$WORKFLOW_TOKEN_SETTINGS" read_workflow_token \
        PUT "repos/$REPO/actions/permissions/workflow" "$WORKFLOW_TOKEN_SETTINGS" \
        "HTTP 409 means an organization policy decides this setting; change it there."
    reconcile required "actions policy (SHA pinning)" "$ACTIONS_POLICY" read_actions_policy \
        PUT "repos/$REPO/actions/permissions" "$ACTIONS_POLICY" \
        "HTTP 409 means an organization policy decides this setting; change it there."
    reconcile required "fork pull request approval" "$FORK_PR_APPROVAL" read_fork_pr_approval \
        PUT "repos/$REPO/actions/permissions/fork-pr-contributor-approval" "$FORK_PR_APPROVAL"
}

reconcile_codeql() {
    heading "Code scanning"
    local severity=required state
    # Default setup is applied asynchronously (GitHub answers 202 and runs a first analysis),
    # so right after this run changed it, a state that is not "configured" yet is only a
    # warning. Once it is configured, any remaining difference (a language) is real drift.
    if [[ $VERIFY_AFTER_APPLY == 1 ]] && written_in_this_run "CodeQL default setup"; then
        if state=$(read_codeql_setup | jq -r '.state') && [[ $state != configured ]]; then
            severity=optional
        fi
    fi
    # shellcheck disable=SC2034 # read by reconcile through bash's dynamic scoping
    local PENDING_NOTE="Default setup is applied asynchronously: run --verify in a few minutes."
    # shellcheck disable=SC2034 # read by reconcile through bash's dynamic scoping
    local DRIFT_NOTE="If only a language is missing (rust: the REST API may not accept it yet),
add it in the web UI: Settings > Advanced Security > CodeQL analysis > Edit configuration."
    reconcile "$severity" "CodeQL default setup" "$CODEQL_SETUP" read_codeql_setup \
        PATCH "repos/$REPO/code-scanning/default-setup" "$CODEQL_SETUP" \
        "HTTP 409: a default setup run is already in progress; re-run in a few minutes." \
        "$CODEQL_SETUP_AUTODETECT"
    # The ruleset's code_scanning rule blocks every merge while its tool "is not configured for
    # the repository" (GitHub's merge protection docs). So when this apply neither found CodeQL
    # configured nor had a change to it accepted, reconcile_ruleset leaves that rule out rather
    # than lock `main` until someone disables the ruleset by hand. A change accepted in this
    # run is fine: GitHub finishes it within minutes.
    if [[ $MODE == apply ]] && ! written_in_this_run "CodeQL default setup"; then
        if ! state=$(read_codeql_setup | jq -r '.state') || [[ $state != configured ]]; then
            CODEQL_UNAVAILABLE=1
        fi
    fi
}

# The ruleset is compared field by field: name, target, enforcement, bypass actors and
# conditions exactly; the set of rule types exactly (an extra rule added in the web UI is
# drift); and each rule's parameters as a subset (GitHub adds defaults, e.g. empty
# required_reviewers). Arrays compare as sets.
ruleset_diff() {
    jq -rn --argjson want "$1" --argjson have "$2" "$JQ_DEFS"'
        def by_type: map({key: .type, value: (.parameters // {})}) | from_entries;
        ($want.rules | by_type) as $w
        | (($have.rules // []) | by_type) as $h
        | (("name", "target", "enforcement") as $k
            | select($want[$k] != $have[$k])
            | "\($k): want \($want[$k] | tojson), have \($have[$k] | tojson)"),
          ((($want.bypass_actors // []) | canon) as $wb
            | (($have.bypass_actors // []) | canon) as $hb
            | select($wb != $hb)
            | "bypass_actors: want \($wb | tojson), have \($hb | tojson)"),
          (select(($want.conditions | canon) != ($have.conditions | canon))
            | "conditions: want \($want.conditions | tojson), have \($have.conditions | tojson)"),
          ((($w | keys) - ($h | keys))[] | "rule \(.): missing"),
          ((($h | keys) - ($w | keys))[]
            | "rule \(.): present on GitHub but not in ruleset-main.json"),
          ($w | to_entries[] | .key as $t | select($h | has($t)) | .value | to_entries[]
            | select((.value | canon) != ($h[$t][.key] | canon))
            | "rule \($t).\(.key): want \(.value | tojson), have \($h[$t][.key] | tojson)")'
}

# ruleset_error LABEL: report a rejected ruleset write. The rulesets API has no dry run, so an
# HTTP 422 on the first apply is the first sign that GitHub does not offer a rule here.
ruleset_error() {
    local hint=""
    if last_status_is 422; then
        hint=$'\n'"If GitHub does not offer one of the rules to this repository (for example
code_scanning), remove it from ruleset-main.json and re-run; see ops/github/README.md."
    fi
    problem required ERROR "$1" "$WRITE_ERROR$hint"
}

reconcile_ruleset() {
    heading "Branch ruleset"
    local want name label list id have delta
    want=$(jq -c . "$RULESET_FILE")
    name=$(jq -r .name <<<"$want")
    label="ruleset \"$name\""

    if [[ $MODE == apply && $CODEQL_UNAVAILABLE == 1 ]] &&
        jq -e 'any(.rules[]; .type == "code_scanning")' >/dev/null <<<"$want"; then
        want=$(jq -c '.rules |= map(select(.type != "code_scanning"))' <<<"$want")
        problem optional warning "$label" \
            "CodeQL default setup is not configured, and the code_scanning rule would block every
merge until it is: writing the ruleset without that rule for now. Fix CodeQL (see above),
then re-run; until then the verification reports the rule as missing."
    fi

    if ! list=$(gh_api GET "repos/$REPO/rulesets?includes_parents=false&per_page=100"); then
        if [[ $MODE == dry-run ]]; then
            status_line unknown "$label"
            details "cannot list rulesets: $(api_error); showing the create call"
            write "$label" POST "repos/$REPO/rulesets" "$want"
        else
            problem required ERROR "$label" "cannot list rulesets: $(api_error)"
        fi
        return 0
    fi
    # Ruleset names are unique within a repository; inherited organization rulesets are
    # excluded by includes_parents=false and by the source_type check.
    id=$(jq -r --arg name "$name" \
        'first(.[] | select(.name == $name and .source_type == "Repository") | .id) // ""' \
        <<<"$list")

    if [[ -z $id ]]; then
        if [[ $MODE == verify ]]; then
            problem required DRIFT "$label" "no repository ruleset named \"$name\" exists"
        else
            status_line change "$label"
            details "not present yet: create it from ruleset-main.json"
            write "$label" POST "repos/$REPO/rulesets" "$want" || ruleset_error "$label"
        fi
        return 0
    fi

    if ! have=$(gh_api GET "repos/$REPO/rulesets/$id"); then
        problem required ERROR "$label" "cannot read ruleset $id: $(api_error)"
        return 0
    fi
    delta=$(ruleset_diff "$want" "$have")
    if [[ -z $delta ]]; then
        status_line ok "$label (id $id)"
    elif [[ $MODE == verify ]]; then
        problem required DRIFT "$label (id $id)" "$delta"
    else
        status_line change "$label (id $id)"
        details "$delta"
        write "$label" PUT "repos/$REPO/rulesets/$id" "$want" || ruleset_error "$label (id $id)"
    fi
}

reconcile_all() {
    reconcile_merge_policy
    reconcile_labels
    reconcile_dependabot
    reconcile_secret_scanning
    reconcile_vulnerability_reporting
    reconcile_actions
    reconcile_codeql
    # Last: its code_scanning rule needs CodeQL configured, and its status check needs the
    # merge settings above.
    reconcile_ruleset
}

# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run) MODE=dry-run ;;
            --verify) MODE=verify ;;
            --repo)
                [[ $# -ge 2 ]] || die "--repo needs a value (OWNER/NAME)"
                REPO=$2
                shift
                ;;
            --repo=*) REPO=${1#--repo=} ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                usage >&2
                die "unknown argument: $1"
                ;;
        esac
        shift
    done
    [[ $REPO =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "--repo must look like OWNER/NAME"
}

preflight() {
    local tool
    for tool in gh jq; do
        command -v "$tool" >/dev/null 2>&1 || die "$tool is required but not on PATH"
    done
    # walk (used by canon) arrived in jq 1.6; without it every comparison would abort mid-run.
    jq -en "$JQ_DEFS"' [[2, 1]] | canon == [[1, 2]]' >/dev/null 2>&1 ||
        die "jq 1.6 or newer is required (found: $(jq --version 2>&1 || true))"
    jq -e '.name and (.rules | type == "array")' "$RULESET_FILE" >/dev/null 2>&1 ||
        die "$RULESET_FILE is not a valid ruleset (needs a name and a rules array)"

    ERR_FILE=$(mktemp "${TMPDIR:-/tmp}/apply-settings.XXXXXX")
    trap cleanup EXIT
    export GH_PROMPT_DISABLED=1 GH_NO_UPDATE_NOTIFIER=1 NO_COLOR=1

    local admin
    if ! admin=$(gh_api GET "repos/$REPO" | jq -r '.permissions.admin // false'); then
        if [[ $MODE == dry-run ]]; then
            printf 'note: cannot read %s (%s);\n' "$REPO" "$(api_error)"
            printf '      listing every write without comparing against the current state.\n'
            return 0
        fi
        die "cannot read $REPO: $(api_error) (is gh authenticated? run: gh auth login)"
    fi
    if [[ $admin != true ]]; then
        if [[ $MODE == dry-run ]]; then
            printf 'note: the gh user is not an admin of %s; some reads may fail.\n' "$REPO"
        else
            die "the authenticated gh user must be an admin of $REPO"
        fi
    fi
}

main() {
    parse_args "$@"
    preflight
    printf '%s: %s\n' "$MODE" "$REPO"

    reconcile_all

    case $MODE in
        dry-run)
            printf '\nDry run: %d write(s) would be made, nothing was changed.\n' "$CHANGES"
            [[ $FAILURES -eq 0 ]] || exit 1
            ;;
        verify)
            if [[ $FAILURES -eq 0 ]]; then
                printf '\nVerified: every setting matches (%d warning(s)).\n' "$WARNINGS"
            else
                printf '\nDrift: %d setting(s) differ, %d warning(s).\n' "$FAILURES" "$WARNINGS"
                exit 1
            fi
            ;;
        apply)
            printf '\nApplied %d change(s). Verifying...\n' "$CHANGES"
            # The verification pass alone decides the exit status: it reads back what GitHub
            # actually stores, whatever the writes above reported.
            MODE=verify
            VERIFY_AFTER_APPLY=1
            FAILURES=0
            WARNINGS=0
            reconcile_all
            if [[ $FAILURES -eq 0 ]]; then
                printf '\nIn sync (%d warning(s)).\n' "$WARNINGS"
            else
                printf '\nStill drifting: %d setting(s), %d warning(s).\n' "$FAILURES" "$WARNINGS"
                exit 1
            fi
            ;;
        *) die "internal error: unknown mode $MODE" ;;
    esac
}

# Run only when executed, so a test harness can source the functions.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
