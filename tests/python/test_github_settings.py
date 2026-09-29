"""Tests for ops/github/apply-settings.sh and the `main` ruleset it applies.

Standard library only, and no network. `gh` is a fake on PATH: a tiny launcher that calls
fake_gh() of this module, which serves a simulated repository from a JSON state file. The
script's reads, writes, drift reports and error handling therefore run for real, and GitHub is
never contacted. The script needs bash and jq; without them the script tests skip.

Run from the repository root:

    python -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
SCRIPT = REPO_ROOT / "ops" / "github" / "apply-settings.sh"
RULESET = REPO_ROOT / "ops" / "github" / "ruleset-main.json"
REPO = "PredictionExplorer/CS-Image-Generation"
BASH = shutil.which("bash")
HAVE_TOOLS = BASH is not None and shutil.which("jq") is not None

# The check-run name of ci.yml's aggregate job, and GitHub Actions' app id: the deploy agent
# deploys only commits with this check, so the ruleset must require the same one.
CI_CHECK = "CI passed"
GITHUB_ACTIONS_APP_ID = 15368

# ---------------------------------------------------------------------------
# The simulated repository
# ---------------------------------------------------------------------------


def initial_state() -> dict[str, Any]:
    """A repository as GitHub creates it, before the script has run (like the real one was)."""
    disabled = {"status": "disabled"}
    return {
        "repo": {
            "full_name": REPO,
            "permissions": {"admin": True},
            "allow_squash_merge": True,
            "allow_rebase_merge": True,
            "allow_merge_commit": True,
            "allow_auto_merge": False,
            "delete_branch_on_merge": False,
            "allow_update_branch": False,
            "squash_merge_commit_title": "COMMIT_OR_PR_TITLE",
            "squash_merge_commit_message": "COMMIT_MESSAGES",
            "security_and_analysis": {
                "dependabot_security_updates": dict(disabled),
                "secret_scanning": dict(disabled),
                "secret_scanning_non_provider_patterns": dict(disabled),
                "secret_scanning_push_protection": dict(disabled),
                "secret_scanning_validity_checks": dict(disabled),
            },
        },
        "vulnerability_alerts": False,
        "automated_security_fixes": False,
        "private_vulnerability_reporting": False,
        "workflow": {
            "default_workflow_permissions": "write",
            "can_approve_pull_request_reviews": True,
        },
        "actions": {"enabled": True, "allowed_actions": "all", "sha_pinning_required": False},
        "fork": {"approval_policy": "first_time_contributors"},
        "codeql": {
            "state": "not-configured",
            "languages": ["actions", "python", "rust"],
            "query_suite": "default",
            "threat_model": "remote",
        },
        "codeql_previous": None,
        "codeql_lag": 0,
        "codeql_conflict": False,
        "labels": {
            "bug": {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
            "enhancement": {"name": "enhancement", "color": "a2eeef", "description": "Old text"},
        },
        "rulesets": [],
        # Returned by the listing only to prove the script ignores inherited rulesets even if
        # GitHub ever returned one despite includes_parents=false.
        "org_rulesets": [{"id": 7, "name": "main", "source_type": "Organization"}],
    }


def server_view(ruleset: dict[str, Any], ruleset_id: int) -> dict[str, Any]:
    """A ruleset as GitHub returns it: defaults added, arrays and rules in another order."""
    stored: dict[str, Any] = json.loads(json.dumps(ruleset))
    for rule in stored["rules"]:
        parameters = rule.get("parameters", {})
        if rule["type"] == "pull_request":
            parameters.setdefault("required_reviewers", [])
            parameters["allowed_merge_methods"].sort(reverse=True)
        if rule["type"] == "required_status_checks":
            parameters.setdefault("do_not_enforce_on_create", False)
    stored["rules"].reverse()
    stored.update(id=ruleset_id, source_type="Repository", source=REPO)
    return stored


class GitHubError(Exception):
    """An HTTP error response of the fake API."""

    def __init__(self, status: int, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body: dict[str, Any] = {"message": message, "status": str(status)}
        if errors:
            self.body["errors"] = errors


def _repository(state: dict[str, Any], method: str, body: dict[str, Any]) -> Any:
    """GET/PATCH /repos/{repo}. Non-provider patterns are refused (HTTP 422) and validity
    checks silently ignored, like features GitHub does not offer a repository."""
    if method == "GET":
        return state["repo"]
    features = body.pop("security_and_analysis", {})
    for feature, value in features.items():
        if feature == "secret_scanning_non_provider_patterns":
            raise GitHubError(422, "Validation Failed", ["not available for this repository"])
        if feature != "secret_scanning_validity_checks":
            state["repo"]["security_and_analysis"][feature] = value
    state["repo"].update(body)
    return state["repo"]


def _toggle(state: dict[str, Any], key: str, method: str) -> Any:
    """The enable-only endpoints: vulnerability alerts, security fixes, private reporting."""
    if method != "GET":
        if key == "automated_security_fixes" and not state["vulnerability_alerts"]:
            raise GitHubError(422, "Dependabot alerts must be enabled first")
        state[key] = True
        return None
    if key == "vulnerability_alerts":  # 204 when enabled, 404 when disabled
        if not state[key]:
            raise GitHubError(404, "Not Found")
        return None
    if key == "automated_security_fixes":
        return {"enabled": state[key], "paused": False}
    return {"enabled": state[key]}


def _codeql(state: dict[str, Any], method: str, body: dict[str, Any]) -> Any:
    """Default setup: a PATCH is applied after FAKE_GH_CODEQL_LAG reads (GitHub answers 202
    and runs a first analysis). FAKE_GH_REJECT_RUST refuses `rust` like the documented REST
    enum; without `languages`, FAKE_GH_DETECTED (default: all three) is configured."""
    if method == "GET":
        if state["codeql_lag"] > 0:
            state["codeql_lag"] -= 1
            return state["codeql_previous"]
        return state["codeql"]
    if state["codeql_conflict"]:
        raise GitHubError(409, "Conflict")
    if os.environ.get("FAKE_GH_REJECT_RUST") and "rust" in body.get("languages", []):
        raise GitHubError(422, "Invalid request", ["rust is not a valid language"])
    if "languages" not in body:
        body["languages"] = os.environ.get("FAKE_GH_DETECTED", "actions,python,rust").split(",")
    state["codeql_previous"] = dict(state["codeql"])
    state["codeql"].update(body)
    state["codeql_lag"] = int(os.environ.get("FAKE_GH_CODEQL_LAG", "0"))
    return {"run_id": 1, "run_url": "https://example.invalid/run/1"}


def _labels(state: dict[str, Any], method: str, name: str, body: dict[str, Any]) -> Any:
    labels: dict[str, Any] = state["labels"]
    if method == "GET":
        if name not in labels:
            raise GitHubError(404, "Not Found")
        return {**labels[name], "id": 1, "default": False}
    if method == "POST":
        if body["name"] in labels:
            raise GitHubError(422, "Validation Failed", ["already_exists"])
        labels[body["name"]] = body
        return body
    if set(body) - {"color", "description", "new_name"}:
        raise GitHubError(422, f"unexpected fields {sorted(body)}")
    labels[name].update(body)
    return labels[name]


def _rulesets(state: dict[str, Any], method: str, rest: str, body: dict[str, Any]) -> Any:
    rulesets: list[dict[str, Any]] = state["rulesets"]
    if rest == "":
        if method == "GET":
            listing = [
                {"id": r["id"], "name": r["name"], "source_type": "Repository"} for r in rulesets
            ]
            return listing + state["org_rulesets"]
        if os.environ.get("FAKE_GH_REJECT_RULESET"):
            raise GitHubError(422, "Validation Failed", ["Invalid rule 'code_scanning'"])
        created = server_view(body, 1000 + len(rulesets))
        rulesets.append(created)
        return created
    ruleset_id = int(rest.strip("/"))
    index = next(i for i, r in enumerate(rulesets) if r["id"] == ruleset_id)
    if method == "PUT":
        rulesets[index] = server_view(body, ruleset_id)
    return rulesets[index]


def _dispatch(state: dict[str, Any], method: str, path: str, body: dict[str, Any]) -> Any:
    """Route one call of the fake REST API."""
    if os.environ.get("FAKE_GH_UNAUTHENTICATED"):
        raise GitHubError(401, "Bad credentials")
    path, _, query = path.partition("?")
    prefix = f"repos/{REPO}"
    if not path.startswith(prefix):
        raise GitHubError(404, f"fake gh: unexpected path {path}")
    rest = path[len(prefix) :]
    toggles = {
        "/vulnerability-alerts": "vulnerability_alerts",
        "/automated-security-fixes": "automated_security_fixes",
        "/private-vulnerability-reporting": "private_vulnerability_reporting",
    }
    settings = {
        "/actions/permissions/workflow": "workflow",
        "/actions/permissions": "actions",
        "/actions/permissions/fork-pr-contributor-approval": "fork",
    }
    if rest == "":
        return _repository(state, method, body)
    if rest in toggles:
        return _toggle(state, toggles[rest], method)
    if rest in settings:
        if method != "GET":
            state[settings[rest]].update(body)
            return None
        return state[settings[rest]]
    if rest == "/code-scanning/default-setup":
        return _codeql(state, method, body)
    if rest.startswith("/labels"):
        return _labels(state, method, rest.removeprefix("/labels/"), body)
    if rest.startswith("/rulesets"):
        if rest == "/rulesets" and method == "GET" and "includes_parents=false" not in query:
            raise GitHubError(400, "the script must exclude inherited rulesets")
        return _rulesets(state, method, rest.removeprefix("/rulesets"), body)
    raise GitHubError(404, f"fake gh: unhandled {method} {path}")


def fake_gh(argv: list[str]) -> int:
    """gh: supports `gh api --method M --header H... [--input -] PATH` against FAKE_GH_STATE.

    Every call is appended to FAKE_GH_LOG as "METHOD PATH". Errors are reported like gh does:
    the response body on stdout, `gh: <message> (HTTP <status>)` on stderr, exit status 1.
    """
    if not argv or argv[0] != "api":
        print(f"fake gh: unsupported command {argv}", file=sys.stderr)
        return 2
    method, path, body = "GET", "", {}
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--method":
            method = argv[index + 1]
            index += 2
        elif arg == "--header":
            index += 2
        elif arg == "--input":
            body = json.loads(sys.stdin.read())
            index += 2
        else:
            path = arg
            index += 1
    with Path(os.environ["FAKE_GH_LOG"]).open("a", encoding="utf-8") as log:
        log.write(f"{method} {path}\n")
    state_file = Path(os.environ["FAKE_GH_STATE"])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    try:
        response = _dispatch(state, method, path, body)
    except GitHubError as error:
        print(json.dumps(error.body))
        print(f"gh: {error} (HTTP {error.status})", file=sys.stderr)
        return 1
    finally:
        state_file.write_text(json.dumps(state), encoding="utf-8")
    if response is not None:
        print(json.dumps(response))
    return 0


def install_fake_gh(bin_dir: Path) -> None:
    """Write an executable `gh` launcher for fake_gh() into `bin_dir`."""
    launcher = bin_dir / "gh"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import test_github_settings\n"
        "sys.exit(test_github_settings.fake_gh(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class RulesetContractTest(unittest.TestCase):
    """ruleset-main.json encodes the merge policy the deploy agent relies on."""

    def setUp(self) -> None:
        self.ruleset = json.loads(RULESET.read_text(encoding="utf-8"))
        self.rules = {rule["type"]: rule.get("parameters", {}) for rule in self.ruleset["rules"]}

    def test_protects_the_default_branch_for_everyone(self) -> None:
        self.assertEqual(self.ruleset["name"], "main")
        self.assertEqual(self.ruleset["target"], "branch")
        self.assertEqual(self.ruleset["enforcement"], "active")
        self.assertEqual(self.ruleset["bypass_actors"], [])
        self.assertEqual(self.ruleset["conditions"]["ref_name"]["include"], ["~DEFAULT_BRANCH"])
        for rule in ("deletion", "non_fast_forward", "required_linear_history"):
            self.assertIn(rule, self.rules)

    def test_requires_the_check_the_deploy_agent_waits_for(self) -> None:
        checks = self.rules["required_status_checks"]
        self.assertTrue(checks["strict_required_status_checks_policy"])
        self.assertEqual(
            checks["required_status_checks"],
            [{"context": CI_CHECK, "integration_id": GITHUB_ACTIONS_APP_ID}],
        )

    def test_pull_requests_merge_linearly_without_blocking_a_solo_maintainer(self) -> None:
        pull_request = self.rules["pull_request"]
        self.assertEqual(sorted(pull_request["allowed_merge_methods"]), ["rebase", "squash"])
        self.assertEqual(pull_request["required_approving_review_count"], 0)
        self.assertFalse(pull_request["require_code_owner_review"])
        self.assertTrue(pull_request["required_review_thread_resolution"])


@unittest.skipUnless(HAVE_TOOLS, "apply-settings.sh needs bash and jq")
class ApplySettingsTest(unittest.TestCase):
    """The script against the fake gh, from a fresh repository to a verified one."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        install_fake_gh(bin_dir)
        self.state_file = self.root / "state.json"
        self.log_file = self.root / "calls.log"
        self.write_state(initial_state())
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("FAKE_GH_", "GH_", "GITHUB_"))
        }
        self.env.update(
            PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            FAKE_GH_STATE=str(self.state_file),
            FAKE_GH_LOG=str(self.log_file),
            TMPDIR=str(self.root),
        )

    def state(self) -> dict[str, Any]:
        state: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
        return state

    def write_state(self, state: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def run_script(self, *args: str, **env: str) -> tuple[int, str]:
        """Run the script; return its exit status and its combined output."""
        assert BASH is not None
        result = subprocess.run(
            [BASH, str(SCRIPT), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        return result.returncode, result.stdout + result.stderr

    def writes(self) -> list[str]:
        """The non-GET calls made so far."""
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").splitlines()
        return [line for line in lines if not line.startswith("GET ")]

    def apply_converged(self) -> None:
        status, output = self.run_script()
        self.assertEqual(status, 0, output)
        self.log_file.unlink()

    def test_dry_run_plans_every_change_and_writes_nothing(self) -> None:
        before = self.state()
        status, output = self.run_script("--dry-run")
        self.assertEqual(status, 0, output)
        self.assertEqual(self.writes(), [])
        self.assertEqual(self.state(), before)
        self.assertIn(f"would run: gh api --method POST repos/{REPO}/rulesets", output)
        self.assertIn("allow_merge_commit: want false, have true", output)
        self.assertNotIn("(id 7)", output)  # the organization's ruleset is not ours
        self.assertIn("Dry run: 18 write(s) would be made", output)

    def test_dry_run_without_credentials_lists_every_write(self) -> None:
        status, output = self.run_script("--dry-run", FAKE_GH_UNAUTHENTICATED="1")
        self.assertEqual(status, 0, output)
        self.assertIn("listing every write without comparing", output)
        self.assertIn("Dry run: 19 write(s) would be made", output)

    def test_apply_converges_and_optional_features_only_warn(self) -> None:
        status, output = self.run_script()
        self.assertEqual(status, 0, output)
        self.assertIn("In sync (2 warning(s))", output)
        self.assertIn("warning  secret_scanning_non_provider_patterns", output)
        self.assertIn("warning  secret_scanning_validity_checks", output)
        state = self.state()
        repo = state["repo"]
        self.assertFalse(repo["allow_merge_commit"])
        self.assertTrue(repo["allow_auto_merge"])
        self.assertEqual(repo["squash_merge_commit_title"], "PR_TITLE")
        # A PR body holding [skip ci] must not reach the squash commit on main: GitHub would
        # start no push run for it, and the deploy agent would wait for its `CI passed`.
        self.assertEqual(repo["squash_merge_commit_message"], "BLANK")
        self.assertEqual(repo["security_and_analysis"]["secret_scanning"]["status"], "enabled")
        self.assertTrue(state["vulnerability_alerts"] and state["automated_security_fixes"])
        self.assertTrue(state["private_vulnerability_reporting"])
        self.assertEqual(state["workflow"]["default_workflow_permissions"], "read")
        self.assertTrue(state["actions"]["sha_pinning_required"])
        self.assertEqual(state["fork"]["approval_policy"], "all_external_contributors")
        self.assertEqual(state["codeql"]["state"], "configured")
        self.assertEqual(state["labels"]["enhancement"]["description"], "New feature or request")
        self.assertIn("needs-triage", state["labels"])
        self.assertEqual(len(state["rulesets"]), 1)

        status, output = self.run_script("--verify")
        self.assertEqual(status, 0, output)
        self.assertIn('ok       ruleset "main" (id 1000)', output)  # despite server defaults

    def test_second_apply_only_retries_what_github_refuses(self) -> None:
        self.apply_converged()
        status, output = self.run_script()
        self.assertEqual(status, 0, output)
        # Only the optional secret scanning features, which this repository does not get.
        self.assertEqual(self.writes(), [f"PATCH repos/{REPO}", f"PATCH repos/{REPO}"])

    def test_pending_codeql_warns_after_apply_and_drifts_until_configured(self) -> None:
        status, output = self.run_script(FAKE_GH_CODEQL_LAG="3")
        self.assertEqual(status, 0, output)
        self.assertIn("warning  CodeQL default setup", output)
        self.assertIn("run --verify in a few minutes", output)
        status, output = self.run_script("--verify")
        self.assertEqual(status, 1, output)
        self.assertIn("DRIFT    CodeQL default setup", output)
        status, output = self.run_script("--verify")
        self.assertEqual(status, 0, output)

    def test_verify_reports_drift_from_the_web_ui_and_apply_repairs_it(self) -> None:
        self.apply_converged()
        state = self.state()
        ruleset = state["rulesets"][0]
        for rule in ruleset["rules"]:
            if rule["type"] == "pull_request":
                rule["parameters"]["allowed_merge_methods"].append("merge")
            if rule["type"] == "required_status_checks":
                rule["parameters"]["strict_required_status_checks_policy"] = False
        ruleset["rules"].append({"type": "creation"})
        ruleset["bypass_actors"] = [{"actor_id": 5, "actor_type": "RepositoryRole"}]
        state["labels"]["ci"]["color"] = "000000"
        state["workflow"]["default_workflow_permissions"] = "write"
        state["repo"]["squash_merge_commit_message"] = "PR_BODY"
        self.write_state(state)

        status, output = self.run_script("--verify")
        self.assertEqual(status, 1, output)
        for line in (
            'squash_merge_commit_message: want "BLANK", have "PR_BODY"',
            "rule pull_request.allowed_merge_methods: want",
            "rule required_status_checks.strict_required_status_checks_policy: want true",
            "rule creation: present on GitHub but not in ruleset-main.json",
            'bypass_actors: want [], have [{"actor_id":5',
            'color: want "1d76db", have "000000"',
            'default_workflow_permissions: want "read", have "write"',
        ):
            self.assertIn(line, output)

        status, output = self.run_script()
        self.assertEqual(status, 0, output)
        state = self.state()
        self.assertEqual([r["id"] for r in state["rulesets"]], [1000])  # updated in place
        self.assertEqual(state["rulesets"][0]["bypass_actors"], [])

    def test_codeql_retries_without_rust_when_github_rejects_it(self) -> None:
        status, output = self.run_script(FAKE_GH_REJECT_RUST="1")
        self.assertEqual(status, 0, output)
        self.assertIn('retrying with: {"state":"configured","query_suite":"default"}', output)
        self.assertEqual(self.state()["codeql"]["languages"], ["actions", "python", "rust"])

    def test_codeql_without_rust_after_the_retry_is_drift_with_a_remedy(self) -> None:
        status, output = self.run_script(FAKE_GH_REJECT_RUST="1", FAKE_GH_DETECTED="actions,python")
        self.assertEqual(status, 1, output)
        self.assertIn(
            'languages: want ["actions","python","rust"], have ["actions","python"]', output
        )
        self.assertIn("CodeQL analysis > Edit configuration", output)

    def test_codeql_setup_in_progress_fails_with_a_hint(self) -> None:
        state = self.state()
        state["codeql_conflict"] = True
        self.write_state(state)
        status, output = self.run_script()
        self.assertEqual(status, 1, output)
        self.assertIn("a default setup run is already in progress", output)

    def test_codeql_that_cannot_be_configured_leaves_out_the_merge_blocking_rule(self) -> None:
        # Merge protection blocks every pull request while its tool is not configured, so a
        # failed CodeQL setup must not produce a ruleset that locks `main`.
        state = self.state()
        state["codeql_conflict"] = True
        self.write_state(state)
        status, output = self.run_script()
        self.assertEqual(status, 1, output)
        self.assertIn("writing the ruleset without that rule", output)
        self.assertIn("rule code_scanning: missing", output)
        rules = [rule["type"] for rule in self.state()["rulesets"][0]["rules"]]
        self.assertNotIn("code_scanning", rules)
        self.assertIn("required_status_checks", rules)

        state = self.state()
        state["codeql_conflict"] = False
        self.write_state(state)
        status, output = self.run_script()
        self.assertEqual(status, 0, output)
        self.assertNotIn("writing the ruleset without that rule", output)
        rulesets = self.state()["rulesets"]
        self.assertEqual([ruleset["id"] for ruleset in rulesets], [1000])  # updated in place
        self.assertIn("code_scanning", [rule["type"] for rule in rulesets[0]["rules"]])

    def test_a_refused_ruleset_fails_with_the_remedy(self) -> None:
        status, output = self.run_script(FAKE_GH_REJECT_RULESET="1")
        self.assertEqual(status, 1, output)
        self.assertIn("Invalid rule 'code_scanning'", output)
        self.assertIn("remove it from ruleset-main.json", output)
        self.assertIn('no repository ruleset named "main" exists', output)

    def test_usage_errors_exit_2(self) -> None:
        for args in (["--bogus"], ["--repo"], ["--repo", "not a repository"]):
            with self.subTest(args=args):
                status, output = self.run_script(*args)
                self.assertEqual(status, 2, output)
        self.assertEqual(self.writes(), [])

    def test_refuses_to_apply_without_admin_rights_or_credentials(self) -> None:
        state = self.state()
        state["repo"]["permissions"]["admin"] = False
        self.write_state(state)
        status, output = self.run_script()
        self.assertEqual(status, 2, output)
        self.assertIn("must be an admin", output)
        status, output = self.run_script("--verify", FAKE_GH_UNAUTHENTICATED="1")
        self.assertEqual(status, 2, output)
        self.assertIn("gh auth login", output)
        self.assertEqual(self.writes(), [])


if __name__ == "__main__":
    unittest.main()
