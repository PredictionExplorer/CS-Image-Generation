# GitHub repository settings as code

Some of this repository's most important controls do not live in git: who may merge into
`main`, which checks must pass first, and which security features are on. Because merging into
`main` deploys to production automatically once CI passes ([docs/deployment.md](../../docs/deployment.md)),
these settings are part of the deployment pipeline and are kept here as reviewed code:

| File | Purpose |
| --- | --- |
| [`apply-settings.sh`](apply-settings.sh) | Reads every managed setting, changes only what differs from the desired state, then reads everything back. `--dry-run` shows the plan, `--verify` reports drift. |
| [`ruleset-main.json`](ruleset-main.json) | The repository ruleset that protects the default branch, in the exact shape of the [rulesets REST API](https://docs.github.com/rest/repos/rules). |

The desired state is declared at the top of `apply-settings.sh` (one commented JSON block per
setting) and in `ruleset-main.json`. To change a setting, change it there in a pull request,
then run the script after the merge. Changes made in the web UI are drift: `--verify` reports
them and the next apply reverts them.

## Prerequisites

- [GitHub CLI](https://cli.github.com) (`gh`), logged in as a **repository admin**:
  `gh auth login` (the default OAuth token's `repo` scope is enough; check with `gh auth status`).
- `jq` 1.6 or newer (preinstalled on macOS 15+; `sudo apt install jq` on Ubuntu).
- bash 3.2 or newer (the script runs under macOS's `/bin/bash`).

## Usage

Run from anywhere; paths are resolved relative to the script.

```bash
ops/github/apply-settings.sh --dry-run   # show what would change; sends GET requests only
ops/github/apply-settings.sh             # apply the changes, then verify everything
ops/github/apply-settings.sh --verify    # report drift; exit 1 if anything differs
```

`--repo OWNER/NAME` targets another repository (for example a fork used for a trial run).

Every setting prints one line: `ok`, `change` (with the differing fields and, under
`--dry-run`, the exact `gh api` call), `DRIFT` or `ERROR` (counted as failures), `warning`
(an optional feature, or a change GitHub is still applying), or, in a dry run that cannot read
the current value, `unknown` followed by the call. Exit status: `0` in sync, `1`
drift or a failed API call, `2` usage error or missing prerequisite (including a `gh` user who
is not an admin).

An apply ends with a full verification pass, and that pass alone decides the exit status: it
reads back what GitHub actually stored, whatever the individual writes reported.

## What it manages

| Area | Desired state | Why |
| --- | --- | --- |
| Merge buttons | squash and rebase only (no merge commits); squash commit = PR title with a blank body; auto-merge allowed; head branches deleted on merge; "Update branch" always offered | Keeps `main` linear (the deploy agent only fast-forwards), makes the Conventional Commit PR title the commit subject, and lets a PR merge itself once `CI passed` is green. The body is left out because GitHub starts no push run for a commit whose message contains `[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]` or `[actions skip]`: a PR body can hold one (Dependabot's pull request descriptions quote upstream commit messages), even one added after CI passed, and the deploy agent would then wait for a `CI passed` check that never comes. CI's title check rejects those markers in the title. |
| Labels | `bug`, `enhancement`, `needs-triage`, `dependencies`, `ci`, `security` | The issue forms apply `bug` or `enhancement` plus `needs-triage`, and GitHub silently drops a form's label that does not exist. `dependencies` is the label Dependabot puts on its pull requests by default (created with Dependabot's colour and description); `ci` and `security` are for triage by hand. Keep the `LABELS` list in sync with the issue forms and with any `labels:` added to `.github/dependabot.yml`. |
| Dependabot | alerts and security updates on | Vulnerable dependencies raise alerts and get fix PRs automatically (version updates are configured in `.github/dependabot.yml`). |
| Secret scanning | secret scanning and push protection on; non-provider patterns and validity checks on where GitHub offers them | Leaked credentials are detected, and blocked at push time. The last two may not be available to this repository; GitHub's refusal (HTTP 422) is only a warning. |
| Private vulnerability reporting | on | The channel [SECURITY.md](../../SECURITY.md) points reporters to. |
| Workflow token | read-only by default; cannot approve pull requests | Least privilege for `GITHUB_TOKEN`: each job requests what it needs in its `permissions:` block, and no workflow can approve its own change. |
| Actions policy | all actions allowed, but only when pinned to a full commit SHA | A moved or compromised tag cannot change what CI runs. The workflows already pin every action and Dependabot bumps the SHAs. |
| Fork pull requests | workflows from every outside contributor wait for approval | Untrusted code never runs in CI without a maintainer looking at it first. |
| Code scanning | CodeQL default setup for `actions`, `python`, `rust`, default query suite | GitHub-managed static analysis of the workflows, the scripts and the crate, on pushes, pull requests and weekly. No workflow file to maintain. |
| Ruleset `main` | see below | Protects the branch the server deploys from. |

### The `main` ruleset

[`ruleset-main.json`](ruleset-main.json) targets the default branch (`~DEFAULT_BRANCH`), is
`active`, and has **no bypass actors**: the rules apply to admins and automation alike.

| Rule | Setting | Effect |
| --- | --- | --- |
| `deletion` | | `main` cannot be deleted. |
| `non_fast_forward` | | No force pushes: published history is never rewritten (the deploy agent refuses a rewritten `main`). |
| `required_linear_history` | | No merge commits on `main`. |
| `pull_request` | 0 approvals; stale approvals dismissed on push; all review threads resolved; merge methods `squash`, `rebase` | Every change arrives through a pull request. Approvals stay optional while there is a single maintainer; `require_code_owner_review` is off for the same reason (the owner cannot approve their own PR). Raise both when a second maintainer joins. |
| `required_status_checks` | `CI passed` from GitHub Actions (app id 15368), branches must be up to date (strict) | The aggregate job of `.github/workflows/ci.yml` must pass on the exact result of the merge. Pinning the source app stops any other integration from posting a fake `CI passed` status. The deploy agent requires the same check run before it deploys. |
| `code_scanning` | CodeQL; block on security alerts `high_or_higher` and on alerts of severity `errors` | A PR cannot merge while CodeQL is still analyzing it or if it introduces a serious finding. |

Deliberately not enabled: required signed commits (every contributor and bot would need
signing keys), a merge queue (little value for one maintainer, and `ci.yml` would need a
`merge_group` trigger), and required approvals (see above).

## First-time rollout

Order matters, because the ruleset requires checks that must already exist:

1. Merge the pull request that adds the `CI passed` job and the SHA-pinned workflows to `main`
   first (at that point `main` is still unprotected). Pull requests opened from an older
   `main` must be rebased before they can merge, since they lack that job.
2. `ops/github/apply-settings.sh --dry-run` and review the plan.
3. `ops/github/apply-settings.sh`. CodeQL default setup is applied asynchronously (a first
   analysis runs), so right after an apply it may show as a `warning`; run `--verify` again a
   few minutes later.
4. Smoke test with a trivial pull request (for example a typo fix): `CI passed` and `CodeQL`
   checks appear and pass, the merge box offers only squash and rebase, auto-merge can be
   enabled, the branch is deleted after the merge, and the deploy agent picks up the commit
   (`python3 ops/deploy/cosmicsig_deploy.py status` on the server).
5. Re-run `--verify` whenever you suspect a change in the web UI, and after editing the
   desired state.

## Caveats

- **Rust in CodeQL default setup.** CodeQL supports Rust and GitHub detects it in this
  repository, but the documented REST enum for `languages` does not list `rust` yet. The script
  first sends `actions`, `python`, `rust`; if GitHub rejects that (HTTP 422), it retries without
  `languages` so that GitHub's language detection decides. If `--verify` then still reports
  `rust` missing, add it once in the web UI (Settings, Advanced Security, CodeQL analysis,
  Edit configuration).
- **SHA pinning.** The policy covers every action a job runs, including the actions a
  composite action calls internally: a third-party composite action that references another
  action by tag fails even when our own `uses:` line is pinned, so check that before adopting
  one (the actions the workflows use today are JavaScript, Docker, or composite without nested
  actions). Reusable workflows may still be referenced by tag. GitHub documents that the
  policy does not block the dynamic workflows of code scanning, so CodeQL default setup is
  unaffected. If another GitHub-managed workflow (for example Dependabot's updates) ever fails
  with an error about actions that must be pinned to a full-length commit SHA, set
  `"sha_pinning_required": false` in `ACTIONS_POLICY`, re-run the script and report it to
  GitHub. Keep any new `uses:` pinned to a full SHA, or the workflow fails to start (zizmor,
  run by the hooks and the lint job, flags unpinned actions before that).
- **Code scanning merge protection** blocks a PR while CodeQL has not finished analyzing its
  latest commit, and blocks every PR while CodeQL is not configured at all. So when an apply
  cannot configure CodeQL default setup (an `ERROR` in the Code scanning section), it writes
  the ruleset without its `code_scanning` rule and says so in a `warning`; the verification
  then reports that rule as missing (exit 1) until a later apply, after CodeQL is fixed,
  restores it. A PR opened before default setup was enabled needs one new push. GitHub does
  not apply this rule to Dependabot PRs analyzed by default setup.
- **No dry run for rulesets.** GitHub offers no way to validate a ruleset without creating
  it, so `ruleset-main.json` was checked against the OpenAPI schema and GitHub's documentation
  (repository rulesets can require code scanning results) but first meets the real API on the
  first apply. If GitHub rejects it with HTTP 422, nothing is created; the script prints
  GitHub's reason. If a rule is not offered to this repository, remove it from
  `ruleset-main.json` in a pull request and re-run.
- **Optional features.** A `warning` for `secret_scanning_non_provider_patterns` or
  `secret_scanning_validity_checks` means GitHub does not offer the feature to this repository
  (or ignored the request); nothing else is affected.
- **Organization policies win.** An HTTP 409 on the workflow token or Actions policy means the
  organization enforces that setting; change it at the organization level instead.

## Emergencies

With no bypass actors, even an admin cannot push to `main` directly. The fast path for a hotfix
is a normal pull request with auto-merge enabled: it merges and deploys as soon as `CI passed`
is green. If CI or CodeQL itself is broken and cannot pass, an admin can set the ruleset's
enforcement to **Disabled** (Settings, Rules, Rulesets, `main`), land the fix, and then run
`ops/github/apply-settings.sh` to restore it; the ruleset history and the audit log record
both changes. Remember that the deploy agent still deploys only commits whose `CI passed` check
succeeded, so a commit landed this way is not deployed until CI passes on it.

## Beyond this repository

Two organization-level settings are worth enabling by an organization owner; the script does
not manage them because they affect every repository of `PredictionExplorer`:

- **Require two-factor authentication** for all members (Organization settings, Authentication
  security). It is currently off, yet anyone with write access can merge to production.
- **Base permissions** lowered from "Write" (the current value) to "Read" (Organization
  settings, Member privileges), with write access granted per repository only to the people
  who need it. With 0 required approvals, every member with write access can merge a green pull
  request, which deploys it.

## How the field names were checked

Every request body was validated against the JSON schemas in GitHub's official OpenAPI
description ([github/rest-api-description](https://github.com/github/rest-api-description),
`api.github.com`) and the current values were read with `gh api` GET requests. The only
mismatch is `rust` in the CodeQL `languages` enum, handled as described above.
