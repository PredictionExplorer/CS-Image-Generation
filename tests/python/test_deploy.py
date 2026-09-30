"""Tests for the continuous-deployment agent (ops/deploy/cosmicsig_deploy.py) and the root
bootstrap (ops/server/bootstrap-root.sh).

Standard library only, and no network. git is real: every test builds a bare "origin" (GitHub),
a developer clone that pushes commits to it, and the production checkout cloned from it. cargo,
systemctl, systemd-analyze, loginctl, id and the Python interpreter the agent uses are fakes on
PATH: the fake_* functions of this module, which install_fake() writes into standalone launcher
scripts
(their source and a few shared helpers), so that the ~15 fake calls of each deploy start in
milliseconds instead of importing this module each time. The GitHub REST API is a local
http.server on 127.0.0.1.

Every XDG directory, the git configuration and the legacy unit directory are temporary, and the
fake systemctl refuses anything but --user: the suite never touches the real systemd manager,
unit files or deploy state, including when it runs inside a deploy on the production host.

Run from the repository root:

    python -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import http.server
import inspect
import io
import json
import logging
import os
import runpy
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import cast
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
AGENT = REPO_ROOT / "ops" / "deploy" / "cosmicsig_deploy.py"
BOOTSTRAP = REPO_ROOT / "ops" / "server" / "bootstrap-root.sh"
if str(AGENT.parent) not in sys.path:
    sys.path.insert(0, str(AGENT.parent))

import cosmicsig_deploy as deploy  # noqa: E402

# ---------------------------------------------------------------------------
# Test repository contents
# ---------------------------------------------------------------------------

# The generator's file name (deploy.BINARY_NAME; the fakes cannot import the agent).
GENERATOR_NAME = "three_body_problem"

# The generator the production checkout ran before the agent's first deploy.
LEGACY_BINARY = "#!/bin/sh\necho legacy three_body_problem\n"

# The fake generator `cargo build` makes from the worktree's bin_version.txt: equal versions
# give byte-identical binaries, and version "broken" fails its --version smoke test.
FAKE_BINARY = """#!/bin/sh
# fake three_body_problem, bin_version.txt = {version}
if [ "$1" = --version ]; then
    [ "{version}" = broken ] && exit 1
    echo "three_body_problem {version}"
fi
exit 0
"""

# The fake run.py committed to the test repository: `--help` succeeds unless FAIL is True.
FAKE_RUN_PY = """import sys

FAIL = {fail}
if "--help" in sys.argv:
    print("usage: run.py")
    sys.exit(1 if FAIL else 0)
"""

GITIGNORE = ".env\noutput/\ntarget/\nrun.lock\n*.log\n"

GITCONFIG = """[user]
    name = Deploy Test
    email = deploy-test@example.invalid
[init]
    defaultBranch = main
[commit]
    gpgsign = false
[advice]
    detachedHead = false
[core]
    autocrlf = false
"""


def fake_binary_text(version: str) -> str:
    """The fake generator built for `version`."""
    return FAKE_BINARY.format(version=version)


def initial_files() -> dict[str, str]:
    """The first commit: the real agent and unit templates, a fake run.py and generator."""
    files = {
        "README.md": "# fake production repository\n",
        ".gitignore": GITIGNORE,
        "bin_version.txt": "1\n",
        "run.py": FAKE_RUN_PY.format(fail=False),
        deploy.AGENT_PATH: AGENT.read_text(encoding="utf-8"),
    }
    for name in deploy.UNIT_NAMES:
        template = REPO_ROOT / deploy.UNIT_TEMPLATE_DIR / name
        files[f"{deploy.UNIT_TEMPLATE_DIR}/{name}"] = template.read_text(encoding="utf-8")
    return files


def check_run(
    sha: str,
    *,
    conclusion: str | None = "success",
    status: str = "completed",
    app: str = "github-actions",
    name: str = "CI passed",
    run_id: int = 1,
) -> dict[str, object]:
    """One check run as GET /repos/{slug}/commits/{sha}/check-runs lists it."""
    return {
        "id": run_id,
        "name": name,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "app": {"slug": app},
        "html_url": f"https://github.invalid/checks/{run_id}",
    }


# ---------------------------------------------------------------------------
# Fakes (standalone scripts on PATH, see install_fake)
#
# A fake may only use FAKE_PRELUDE's imports, the constants and helpers in FAKE_SHARED, and
# its own code: install_fake() copies exactly that into the launcher.
# ---------------------------------------------------------------------------

FAKE_PRELUDE = """from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import cast
"""


def _sees_token() -> bool:
    """True if a GitHub token reached this process (the agent must keep it to itself)."""
    return any(os.environ.get(name) for name in ("COSMICSIG_DEPLOY_GITHUB_TOKEN", "GITHUB_TOKEN"))


def _log_call(entry: list[object]) -> None:
    """Append one call to the FAKE_LOG file (JSON lines)."""
    log_path = os.environ.get("FAKE_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def fake_cargo(argv: list[str]) -> int:
    """cargo: `build` writes the fake generator for the worktree's bin_version.txt; `test` passes.

    FAKE_CARGO_FAIL=build|test fails that subcommand; FAKE_CARGO_SLEEP=build|test makes it hang
    (timeouts, stop requests). With FAKE_CARGO_TEST_REBUILDS=1, `test` overwrites the binary, as
    a test build with the dev-dependencies' features unified in can. Logs [argv, cwd,
    CARGO_TARGET_DIR, CI, niceness, sees token].
    """
    command = argv[0] if argv else ""
    _log_call(
        [
            "cargo",
            argv,
            str(Path.cwd()),
            os.environ.get("CARGO_TARGET_DIR"),
            os.environ.get("CI"),
            os.nice(0),
            _sees_token(),
        ]
    )
    print(f"   Compiling three_body_problem v1.1.0 ({command})", flush=True)
    if os.environ.get("FAKE_CARGO_SLEEP") == command:
        time.sleep(60)
    if os.environ.get("FAKE_CARGO_FAIL") == command:
        print(f"error: fake {command} failure", file=sys.stderr)
        return 101
    rebuild = command == "test" and os.environ.get("FAKE_CARGO_TEST_REBUILDS") == "1"
    if command == "build" or rebuild:
        version = Path("bin_version.txt").read_text(encoding="utf-8").strip()
        if rebuild:
            version += "-test"
        binary = Path(os.environ["CARGO_TARGET_DIR"]) / "release" / GENERATOR_NAME
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text(fake_binary_text(version), encoding="utf-8")
        binary.chmod(0o755)
    return 0


def fake_python(argv: list[str]) -> int:
    """The interpreter the agent uses ($COSMICSIG_DEPLOY_PYTHON).

    `-m unittest ...` (the staged Python tests) logs [argv, cwd, sees token] and exits
    FAKE_PYTEST_RC; any other command runs its script in-process, as the real interpreter would.
    """
    if argv[:2] == ["-m", "unittest"]:
        _log_call(["pytests", argv, str(Path.cwd()), _sees_token()])
        status = int(os.environ.get("FAKE_PYTEST_RC", "0"))
        if status:
            print("FAILED (failures=1)", file=sys.stderr)
        return status
    _log_call(["python", argv])
    sys.argv = list(argv)
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _load_json(path: Path) -> dict[str, object]:
    """A JSON object from `path` ({} if absent)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _parse_unit(text: str) -> dict[str, list[tuple[str, str]]]:
    """A unit file's sections and their settings, in order (as much as the fakes need)."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], [])
        elif current is not None and "=" in line:
            key, _, value = line.partition("=")
            current.append((key.strip(), value.strip()))
    return sections


def _unit_load_error(path: Path) -> str | None:
    """Why systemd would refuse to load the unit file at `path`, or None if it loads.

    Models the refusals a template edit can cause: a service without ExecStart= (for example
    because its [Service] header is misspelled, so systemd ignores the whole section), a oneshot
    service with Restart=always, and a timer without any On*= trigger.
    """
    sections = _parse_unit(path.read_text(encoding="utf-8"))
    if path.name.endswith(".service"):
        service = dict(sections.get("Service", []))
        if "ExecStart" not in service:
            return "Service has no ExecStart=, ExecStop=, or SuccessAction=. Refusing."
        if service.get("Type") == "oneshot" and service.get("Restart") in ("always", "on-success"):
            return (
                "Service has Restart= set to either always or on-success, which isn't allowed "
                "for Type=oneshot services. Refusing."
            )
    elif path.name.endswith(".timer"):
        if not any(key.startswith("On") for key, _ in sections.get("Timer", [])):
            return "Timer unit lacks value setting. Refusing."
    return None


def fake_systemctl(argv: list[str]) -> int:
    """`systemctl --user` against a JSON state file (FAKE_SYSTEMCTL_STATE).

    A unit is loaded when its file is in $XDG_CONFIG_HOME/systemd/user and systemd would accept
    it (_unit_load_error; `show` reports LoadState=bad-setting otherwise). The state holds
    "active" {unit: state}, "enabled" {unit: bool} and "script" {unit: [states]}: is-active pops
    the next scripted state, if any. FAKE_SYSTEMCTL_FAIL lists verbs that fail. With
    FAKE_PAUSE_AFTER=N, the Nth is-active call writes the pause flag FAKE_PAUSE_FILE. With
    FAKE_WATCH, is-active logs that file's content (what the checkout holds during a wait).
    Anything but --user fails: the agent must never touch the system manager.
    """
    _log_call(["systemctl", *argv])
    if not argv or argv[0] != "--user":
        print("fake systemctl: these tests only allow --user", file=sys.stderr)
        return 99
    verb, args = argv[1], argv[2:]
    if verb in os.environ.get("FAKE_SYSTEMCTL_FAIL", "").split(","):
        print(f"Failed to {verb}: fake failure", file=sys.stderr)
        return 1
    state_path = Path(os.environ["FAKE_SYSTEMCTL_STATE"])
    state = _load_json(state_path)
    active = cast(dict[str, str], state.setdefault("active", {}))
    enabled = cast(dict[str, bool], state.setdefault("enabled", {}))
    script = cast(dict[str, list[str]], state.setdefault("script", {}))
    unit_dir = Path(os.environ["XDG_CONFIG_HOME"]) / "systemd" / "user"
    units = [arg for arg in args if not arg.startswith("-")]
    status = 0
    if verb == "show":
        unit = units[0]
        exists = (unit_dir / unit).exists()
        if not exists:
            load_state = "not-found"
        else:
            load_state = "bad-setting" if _unit_load_error(unit_dir / unit) else "loaded"
        current = active.get(unit, "inactive")
        print(f"LoadState={load_state}")
        print(f"ActiveState={current}")
        print(f"SubState={'running' if current == 'active' else 'dead'}")
        file_state = ("enabled" if enabled.get(unit) else "disabled") if exists else ""
        print(f"UnitFileState={file_state}")
    elif verb == "is-active":
        unit = units[0]
        queue = script.get(unit) or []
        current = queue.pop(0) if queue else active.get(unit, "inactive")
        print(current)
        status = 0 if current == "active" else 3
        calls = int(cast(int, state.get("is_active_calls", 0))) + 1
        state["is_active_calls"] = calls
        watch = os.environ.get("FAKE_WATCH")
        if watch:
            _log_call(["watch", Path(watch).read_text(encoding="utf-8")])
        if os.environ.get("FAKE_PAUSE_AFTER") == str(calls):
            Path(os.environ["FAKE_PAUSE_FILE"]).write_text(
                json.dumps({"reason": "paused mid-wait"}), encoding="utf-8"
            )
    elif verb in ("stop", "start", "enable", "disable"):
        for unit in units:
            if not (unit_dir / unit).exists():
                print(f"Failed to {verb} {unit}: Unit {unit} not loaded.", file=sys.stderr)
                return 5
            if verb == "stop":
                active[unit] = "inactive"
            elif verb == "start" and unit.endswith(".timer"):
                active[unit] = "active"
            elif verb == "enable":
                enabled[unit] = True
                if "--now" in args:
                    active[unit] = "active"
            elif verb == "disable":
                enabled[unit] = False
                if "--now" in args:
                    active[unit] = "inactive"
    elif verb != "daemon-reload":
        print(f"fake systemctl: unknown verb {verb}", file=sys.stderr)
        return 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return status


def fake_systemd_analyze(argv: list[str]) -> int:
    """`systemd-analyze --user verify FILE...`, as strict as the real one for these units.

    Logs the call, then fails (status 1) for a unit file systemd would not load
    (_unit_load_error) or whose Exec*= command is not an executable file. Anything but
    `--user verify` fails too: the tests never look at the system manager's units.
    """
    _log_call(["systemd-analyze", *argv])
    if "--user" not in argv or "verify" not in argv:
        print("fake systemd-analyze: these tests only allow --user verify", file=sys.stderr)
        return 99
    status = 0
    for name in argv[argv.index("verify") + 1 :]:
        if name.startswith("-"):
            continue
        path = Path(name)
        problems = [_unit_load_error(path)]
        for settings in _parse_unit(path.read_text(encoding="utf-8")).values():
            for key, value in settings:
                command = value.lstrip("-@+!:").split(maxsplit=1)[0] if value.strip() else ""
                if key.startswith("Exec") and command and not os.access(command, os.X_OK):
                    problems.append(f"Command {command} is not executable: No such file")
        for problem in filter(None, problems):
            print(f"{path}: {problem}", file=sys.stderr)
            status = 1
    return status


def fake_loginctl(argv: list[str]) -> int:
    """loginctl: `show-user ... --value` prints FAKE_LINGER (default yes); the rest succeeds."""
    _log_call(["loginctl", *argv])
    if argv[:1] == ["show-user"]:
        print(os.environ.get("FAKE_LINGER", "yes"))
    return 0


def fake_root_systemctl(argv: list[str]) -> int:
    """The SYSTEM systemctl the bootstrap uses, against FAKE_ROOT_STATE.

    A legacy unit is loaded if its file was in COSMICSIG_BOOTSTRAP_UNIT_DIR at the last
    daemon-reload. is-active of the legacy service pops "service_states".
    """
    _log_call(["systemctl", *argv])
    if "--user" in argv:
        return 99
    state_path = Path(os.environ["FAKE_ROOT_STATE"])
    state = _load_json(state_path)
    unit_dir = Path(os.environ["COSMICSIG_BOOTSTRAP_UNIT_DIR"])
    loaded = cast(list[str], state.setdefault("loaded", sorted(p.name for p in unit_dir.iterdir())))
    verb = argv[0]
    status = 0
    if verb == "show":
        prop = next(arg for arg in argv if arg.startswith("--property=")).split("=", 1)[1]
        unit = argv[-1]
        if prop == "LoadState":
            print("loaded" if unit in loaded else "not-found")
        elif prop == "WorkingDirectory":
            print(state.get("working_directory", "") if unit in loaded else "")
    elif verb == "is-active":
        queue = cast(list[str], state.setdefault("service_states", []))
        current = queue.pop(0) if queue else "inactive"
        print(current)
        status = 0 if current == "active" else 3
    elif verb == "daemon-reload":
        state["loaded"] = sorted(p.name for p in unit_dir.iterdir())
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return status


def fake_id(argv: list[str]) -> int:
    """id: `id -u` prints FAKE_UID (default 0, root); `id -u -- NAME` fails for nosuchuser."""
    if argv == ["-u"]:
        print(os.environ.get("FAKE_UID", "0"))
        return 0
    if argv[:2] == ["-u", "--"] and len(argv) == 3:
        if argv[2] == "nosuchuser":
            print(f"id: {argv[2]}: no such user", file=sys.stderr)
            return 1
        print("1000")
        return 0
    return 1


FAKES = {
    "cargo": fake_cargo,
    "python": fake_python,
    "systemctl": fake_systemctl,
    "systemd-analyze": fake_systemd_analyze,
    "loginctl": fake_loginctl,
    "root-systemctl": fake_root_systemctl,
    "id": fake_id,
}


# What every launcher carries besides its fake's own code.
FAKE_SHARED = (_sees_token, _log_call, _load_json, fake_binary_text, _parse_unit, _unit_load_error)


def install_fake(bin_dir: Path, name: str, fake: str | None = None) -> Path:
    """Write an executable script `name` that runs FAKES[fake or name] into `bin_dir`.

    The script holds the fake's source and FAKE_SHARED's, not an import of this module, and
    runs isolated (-I) without site-packages (-S), so it starts in a few milliseconds.
    """
    function = FAKES[fake or name]
    constants = f"GENERATOR_NAME = {GENERATOR_NAME!r}\nFAKE_BINARY = {FAKE_BINARY!r}\n"
    sources = [inspect.getsource(helper) for helper in (*FAKE_SHARED, function)]
    launcher = bin_dir / name
    launcher.write_text(
        f"#!{sys.executable} -IS\n{FAKE_PRELUDE}\n{constants}\n\n"
        + "\n\n".join(sources)
        + f"\n\nsys.exit({function.__name__}(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def git(cwd: Path, *args: str) -> str:
    """Run git in `cwd` for a test's own setup; its stripped stdout."""
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def setUpModule() -> None:
    """Keep the agent's log records out of the test output (assertLogs still sees them)."""
    deploy.log.addHandler(logging.NullHandler())
    deploy.log.propagate = False


# ---------------------------------------------------------------------------
# The fake GitHub API
# ---------------------------------------------------------------------------


class _GitHubServer(http.server.ThreadingHTTPServer):
    """The HTTP server behind FakeGitHub."""

    daemon_threads = True
    block_on_close = False
    github: FakeGitHub

    def handle_error(self, request: object, client_address: object) -> None:
        """Ignore clients that hung up (the timeout test's client does)."""


class _GitHubHandler(http.server.BaseHTTPRequestHandler):
    """Answers GET /repos/{owner}/{repo}/commits/{sha}/check-runs from FakeGitHub's tables."""

    def do_GET(self) -> None:
        github = cast(_GitHubServer, self.server).github
        github.requests.append(
            (self.path, {key.lower(): value for key, value in self.headers.items()})
        )
        if github.delay:
            time.sleep(github.delay)
        status, headers, body = github.respond(self.path, self.headers.get("Authorization"))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the request log out of the test output."""


class FakeGitHub:
    """A fake GitHub REST API on 127.0.0.1: check runs per commit, and the requests it got.

    A commit without an entry in `runs` gets one successful `CI passed` run from GitHub
    Actions (or none, if `default_conclusion` is None). `status`, `headers` and `body` replace
    every answer, and `delay` slows it down. A request with the token `rejected_token` gets
    HTTP 401 Bad credentials, as GitHub answers an expired or revoked token.
    """

    def __init__(self) -> None:
        self.runs: dict[str, list[dict[str, object]]] = {}
        self.default_conclusion: str | None = "success"
        self.status = 200
        self.headers: dict[str, str] = {}
        self.body: bytes | None = None
        self.delay = 0.0
        self.rejected_token: str | None = None
        self.requests: list[tuple[str, dict[str, str]]] = []
        self._server = _GitHubServer(("127.0.0.1", 0), _GitHubHandler)
        self._server.github = self
        # A short poll interval: shutdown() waits for serve_forever's next poll.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()

    @property
    def url(self) -> str:
        """The API base URL."""
        return f"http://127.0.0.1:{self._server.server_port}"

    def close(self) -> None:
        """Stop serving."""
        self._server.shutdown()
        self._server.server_close()

    def reset(self) -> None:
        """Answer normally again."""
        self.status, self.headers, self.body, self.delay = 200, {}, None, 0.0

    def respond(self, path: str, authorization: str | None) -> tuple[int, dict[str, str], bytes]:
        """The status, headers and body for a request path and Authorization header."""
        if self.rejected_token is not None and authorization == f"Bearer {self.rejected_token}":
            return 401, {}, b'{"message": "Bad credentials", "status": "401"}'
        if self.status != 200 or self.body is not None:
            return self.status, self.headers, self.body or b'{"message": "fake error"}'
        parts = urllib.parse.urlsplit(path).path.strip("/").split("/")
        sha = parts[4] if len(parts) == 6 and parts[5] == "check-runs" else ""
        runs = self.runs.get(sha)
        if runs is None:
            runs = (
                [check_run(sha, conclusion=self.default_conclusion)]
                if self.default_conclusion
                else []
            )
        return (
            200,
            self.headers,
            json.dumps({"total_count": len(runs), "check_runs": runs}).encode(),
        )


def closed_port_url() -> str:
    """A URL on 127.0.0.1 where nothing listens (connection refused)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class HeldLock:
    """An flock held by this process on a file, like another process's run or tick."""

    def __init__(self, path: Path, pid: str = "4242") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = os.open(path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, f"{pid}\n".encode())
        self._guard = threading.Lock()

    def release(self) -> None:
        """Release the lock (idempotent, thread-safe)."""
        with self._guard:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------


class DeployTestCase(unittest.TestCase):
    """A bare origin, a developer clone, the production checkout, fakes on PATH, a fake API."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # systemd-analyze too: the real one (on a Linux CI runner or the production host) must
        # never look at these units.
        for name in ("cargo", "systemctl", "systemd-analyze", "loginctl"):
            install_fake(self.bin, name)
        self.python = install_fake(self.bin, "fake-python", "python")
        self.calls_file = self.root / "calls.jsonl"
        self.github = FakeGitHub()
        self.addCleanup(self.github.close)
        self.legacy_dir = self.root / "etc-systemd-system"
        self.legacy_dir.mkdir()
        (self.root / "gitconfig").write_text(GITCONFIG, encoding="utf-8")
        self.origin = self.root / "origin.git"
        self.dev = self.root / "dev"
        self.repo = self.root / "prod"

        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        for key in list(os.environ):
            # GIT_* includes GIT_DIR and GIT_INDEX_FILE, which git sets for hooks: the suite also
            # runs from a pre-push hook, and those would redirect every git command below.
            if key.startswith(("GIT_", "FAKE_", "COSMICSIG_", "XDG_")) or key.lower() in (
                "github_token",
                "journal_stream",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ):
                del os.environ[key]
        os.environ.update(
            {
                "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "HOME": str(self.root / "home"),
                "XDG_STATE_HOME": str(self.root / "xdg" / "state"),
                "XDG_DATA_HOME": str(self.root / "xdg" / "data"),
                "XDG_CONFIG_HOME": str(self.root / "xdg" / "config"),
                "GIT_CONFIG_GLOBAL": str(self.root / "gitconfig"),
                "GIT_CONFIG_NOSYSTEM": "1",
                deploy.ENV_REPO: str(self.repo),
                deploy.ENV_GITHUB_API: self.github.url,
                deploy.ENV_GITHUB_REPO: "Owner/Repo",
                deploy.ENV_PYTHON: str(self.python),
                "FAKE_LOG": str(self.calls_file),
                "FAKE_SYSTEMCTL_STATE": str(self.root / "systemctl.json"),
                "no_proxy": "*",
            }
        )
        for name in ("systemctl", deploy.SYSTEMD_ANALYZE):
            self.assertEqual(shutil.which(name), str(self.bin / name))

        for name, value in {
            "LEGACY_UNIT_DIR": self.legacy_dir,
            "SYNC_POLL_SECONDS": 0.01,
            "LOCK_POLL_SECONDS": 0.01,
            "LOCK_PROGRESS_SECONDS": 0.0,
        }.items():
            patcher = mock.patch.object(deploy, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # main() would replace the handlers assertLogs relies on.
        logging_patcher = mock.patch.object(deploy, "setup_logging")
        logging_patcher.start()
        self.addCleanup(logging_patcher.stop)

        git(self.root, "init", "--quiet", "--bare", str(self.origin))
        git(self.root, "clone", "--quiet", str(self.origin), str(self.dev))
        self.initial = self.commit("feat: initial", initial_files())
        git(self.root, "clone", "--quiet", str(self.origin), str(self.repo))
        self.paths = deploy.Paths.from_env()
        self.binary = self.paths.installed_binary
        self.previous_binary = self.binary.with_name(deploy.BINARY_NAME + deploy.PREVIOUS_SUFFIX)
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text(LEGACY_BINARY, encoding="utf-8")
        self.binary.chmod(0o755)

    # --- repositories --------------------------------------------------------

    def commit(self, message: str, files: Mapping[str, str | None], *, force: bool = False) -> str:
        """Commit `files` (None deletes) in the developer clone and push it to origin main."""
        for relative, content in files.items():
            path = self.dev / relative
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        git(self.dev, "add", "--all", "--force")  # --force: the tests commit ignored paths too
        git(self.dev, "commit", "--quiet", "-m", message)
        git(self.dev, "push", "--quiet", *(["--force"] if force else []), "origin", "HEAD:main")
        return git(self.dev, "rev-parse", "HEAD")

    def head(self) -> str:
        """The production checkout's HEAD commit."""
        return git(self.repo, "rev-parse", "HEAD")

    def assert_checkout_at(self, sha: str) -> None:
        """The checkout is on main at `sha`, with a clean tracked tree."""
        self.assertEqual(self.head(), sha)
        self.assertEqual(git(self.repo, "symbolic-ref", "--short", "HEAD"), "main")
        self.assertEqual(git(self.repo, "status", "--porcelain", "--untracked-files=no"), "")

    # --- the agent -----------------------------------------------------------

    def agent(self, *argv: str) -> int:
        """cosmicsig_deploy.py `argv` (in-process)."""
        return deploy.main(list(argv))

    def tick(self) -> int:
        """One `run`."""
        return self.agent("run")

    def state(self) -> deploy.State:
        """state.json."""
        return deploy.load_state(self.paths.state_file)

    def mark_deployed(self) -> None:
        """Leave the checkout as a successful deploy of its HEAD would, without running a tick:
        the tested binary and its release, the units, the enabled sync timer, state.json."""
        head = self.head()
        version = (self.repo / "bin_version.txt").read_text(encoding="utf-8").strip()
        self.binary.write_text(fake_binary_text(version), encoding="utf-8")
        deploy.publish_release(self.paths, head, self.binary, source="test baseline")
        deploy.install_units(self.paths)
        self.set_systemctl(
            active={deploy.SYNC_TIMER: "active", deploy.DEPLOY_TIMER: "active"},
            enabled={deploy.SYNC_TIMER: True, deploy.DEPLOY_TIMER: True},
        )
        with deploy.state_transaction(self.paths) as state:
            state.deployed_sha = head
            state.deployed_subject = git(self.repo, "log", "-1", "--format=%s")
            state.deployed_at = deploy.isoformat(deploy.utcnow())
            state.binary_sha256 = deploy.file_sha256(self.binary)
        self.clear_calls()

    def systemctl_state(self) -> dict[str, object]:
        """The fake systemctl's state (FAKE_SYSTEMCTL_STATE)."""
        return _load_json(Path(os.environ["FAKE_SYSTEMCTL_STATE"]))

    def sync_timer(self) -> tuple[object, object]:
        """The fake sync timer's (ActiveState, enabled)."""
        state = self.systemctl_state()
        active = cast(dict[str, str], state.get("active", {}))
        enabled = cast(dict[str, bool], state.get("enabled", {}))
        return active.get(deploy.SYNC_TIMER, "inactive"), enabled.get(deploy.SYNC_TIMER, False)

    def status_text(self) -> str:
        """What `status` prints."""
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(self.agent("status"), 0)
        return out.getvalue()

    def age_ci_check(self, sha: str, minutes: int) -> None:
        """Pretend a CI failure was last checked `minutes` ago."""
        with deploy.state_transaction(self.paths) as state:
            failure = state.failed_shas[sha]
            moment = deploy.utcnow() - datetime.timedelta(minutes=minutes)
            failure.checked_at = deploy.isoformat(moment)

    # --- fakes ---------------------------------------------------------------

    def set_systemctl(self, **values: object) -> None:
        """Merge `values` into the fake systemctl's state."""
        path = Path(os.environ["FAKE_SYSTEMCTL_STATE"])
        state = _load_json(path)
        for key, value in values.items():
            current = state.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            else:
                state[key] = value
        path.write_text(json.dumps(state), encoding="utf-8")

    def call_log(self) -> list[list[object]]:
        """Every logged call of the fakes, in order: [kind, *details]."""
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def calls(self, kind: str) -> list[list[object]]:
        """The logged calls of one fake, in order."""
        return [entry[1:] for entry in self.call_log() if entry[0] == kind]

    def systemctl_verbs(self) -> list[list[object]]:
        """The fake systemctl's calls without the leading --user."""
        return [call[1:] for call in self.calls("systemctl")]

    def clear_calls(self) -> None:
        """Forget the logged calls."""
        self.calls_file.unlink(missing_ok=True)

    @contextlib.contextmanager
    def stop_request_after(self, seconds: float) -> Iterator[None]:
        """Send this process SIGTERM after `seconds`, which the agent's handler turns into a stop
        request. The test's own no-op handler catches it if the agent is no longer listening."""
        previous = signal.signal(signal.SIGTERM, _ignore_signal)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        timer = threading.Timer(seconds, os.kill, args=(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    @contextlib.contextmanager
    def logs(self, level: str = "INFO") -> Iterator[list[str]]:
        """The agent's log lines at `level` and up during the block."""
        lines: list[str] = []
        with self.assertLogs(deploy.log, level=level) as captured:
            yield lines
        lines.extend(captured.output)


def _ignore_signal(_signum: int, _frame: object) -> None:
    """A signal handler that does nothing."""


class _Crash(BaseException):
    """The agent dying mid-switch (SIGKILL, the OOM killer, a power loss): not an Exception, so
    the switch's undo, which handles every Exception, does not run."""


def text(lines: list[str]) -> str:
    """Log lines as one string."""
    return "\n".join(lines)


def errors(lines: list[str]) -> list[str]:
    """The ERROR and CRITICAL lines."""
    return [line for line in lines if line.startswith(("ERROR:", "CRITICAL:"))]


# ---------------------------------------------------------------------------
# install and the unit templates
# ---------------------------------------------------------------------------


class InstallTests(DeployTestCase):
    """`install` and the rendered units."""

    def test_install_renders_the_units_and_starts_only_the_deploy_timer(self) -> None:
        with self.logs() as lines:
            self.assertEqual(self.agent("install"), 0)
        self.assertIn("lingering is enabled", text(lines))
        units = self.paths.unit_dir
        self.assertEqual(
            self.systemctl_verbs(),
            [
                ["daemon-reload"],
                *(
                    ["show", "--property=LoadState,ActiveState,SubState,UnitFileState", name]
                    for name in deploy.UNIT_NAMES
                ),
                ["enable", "--now", deploy.DEPLOY_TIMER],
            ],
        )
        self.assertEqual(
            self.calls("systemd-analyze"),
            [["--user", "--man=no", "verify", *(str(units / n) for n in deploy.UNIT_NAMES)]],
        )
        python, repo = str(self.python), str(self.repo)
        deploy_service = (units / deploy.DEPLOY_SERVICE).read_text()
        self.assertIn(
            f"ExecStart={python} {repo}/ops/deploy/cosmicsig_deploy.py run\n", deploy_service
        )
        self.assertIn("EnvironmentFile=-%h/.config/cosmicsig-deploy.env\n", deploy_service)
        self.assertIn(
            "Environment=PATH=%h/.cargo/bin:/usr/local/bin:/usr/bin:/bin\n", deploy_service
        )
        self.assertIn("TimeoutStartSec=infinity\n", deploy_service)
        self.assertIn("KillMode=mixed\n", deploy_service)
        sync_service = (units / deploy.SYNC_SERVICE).read_text()
        self.assertIn(f"WorkingDirectory={repo}\n", sync_service)
        self.assertIn(f"EnvironmentFile={repo}/.env\n", sync_service)
        self.assertIn(f"ExecStart={python} {repo}/run.py\n", sync_service)
        self.assertIn("Environment=PATH=/usr/local/bin:/usr/bin:/bin\n", sync_service)
        self.assertNotIn("User=", sync_service)
        self.assertIn("OnUnitInactiveSec=5min\n", (units / deploy.SYNC_TIMER).read_text())
        deploy_timer = (units / deploy.DEPLOY_TIMER).read_text()
        self.assertIn("OnBootSec=3min\n", deploy_timer)
        self.assertIn("OnUnitInactiveSec=2min\n", deploy_timer)
        for name in deploy.UNIT_NAMES:
            self.assertNotIn("@", (units / name).read_text(), name)
        self.assertFalse(self.paths.state_file.exists())  # nothing is deployed yet

        # Idempotent: the unchanged units are not rewritten.
        mtimes = {name: (units / name).stat().st_mtime_ns for name in deploy.UNIT_NAMES}
        self.assertEqual(self.agent("install"), 0)
        self.assertEqual({n: (units / n).stat().st_mtime_ns for n in deploy.UNIT_NAMES}, mtimes)

    def test_install_warns_when_lingering_is_off(self) -> None:
        os.environ["FAKE_LINGER"] = "no"
        with self.logs("WARNING") as lines:
            self.assertEqual(self.agent("install"), 0)
        self.assertIn("lingering is not enabled", text(lines))

    def test_install_refuses_units_that_systemd_would_not_run(self) -> None:
        # The first deploy only checks the units again if it changes them, so `install` must.
        template = self.repo / deploy.UNIT_TEMPLATE_DIR / deploy.SYNC_SERVICE
        text_before = template.read_text(encoding="utf-8")
        broken = text_before.replace("\n[Service]\n", "\n[Services]\n")
        self.assertNotEqual(broken, text_before)
        template.write_text(broken, encoding="utf-8")
        with self.logs("ERROR") as lines:
            self.assertEqual(self.agent("install"), 1)
        self.assertIn(
            f"systemd does not load {deploy.SYNC_SERVICE} after daemon-reload "
            "(LoadState=bad-setting)",
            text(lines),
        )
        self.assertNotIn(["enable", "--now", deploy.DEPLOY_TIMER], self.systemctl_verbs())

        template.write_text(text_before, encoding="utf-8")
        self.assertEqual(self.agent("install"), 0)
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.DEPLOY_TIMER])

    def test_install_warns_when_the_checkout_has_no_env_file(self) -> None:
        with self.logs("WARNING") as lines:
            self.assertEqual(self.agent("install"), 0)
        self.assertEqual(
            lines,
            [
                f"WARNING:cosmicsig.deploy:{self.repo / '.env'} is missing: "
                f"{deploy.SYNC_SERVICE} cannot start without it (it names the asset host; see "
                ".env.example)"
            ],
        )
        (self.repo / ".env").write_text("COSMICSIG_SSH_HOST=assets.example\n", encoding="utf-8")
        with self.assertNoLogs(deploy.log, level="WARNING"):
            self.assertEqual(self.agent("install"), 0)

    def test_install_and_run_refuse_while_the_legacy_system_units_exist(self) -> None:
        (self.legacy_dir / deploy.SYNC_TIMER).write_text("[Timer]\n", encoding="utf-8")
        for command in ("install", "run"):
            with self.subTest(command=command), self.logs("ERROR") as lines:
                self.assertEqual(self.agent(command), 1)
            self.assertIn("bootstrap-root.sh", text(lines))
        self.assertEqual(self.call_log(), [])
        self.assertIn("legacy system units", self.state().last_error or "")

    def test_rendering_refuses_unsafe_paths_and_unknown_placeholders(self) -> None:
        with self.assertRaisesRegex(deploy.DeployError, "cannot be written into a unit file"):
            deploy.render_units(Path("/home/user/my checkout"), "/usr/bin/python3")
        with self.assertRaisesRegex(deploy.DeployError, "cannot be written into a unit file"):
            deploy.render_units(self.repo, "/usr/bin/python3 -I")
        template = self.repo / deploy.UNIT_TEMPLATE_DIR / deploy.SYNC_TIMER
        template.write_text("[Timer]\nOnBootSec=@BOOT_DELAY@\n", encoding="utf-8")
        with self.assertRaisesRegex(deploy.DeployError, "unknown placeholder @BOOT_DELAY@"):
            deploy.render_units(self.repo, "/usr/bin/python3")


# ---------------------------------------------------------------------------
# run: the deployment tick
# ---------------------------------------------------------------------------


class TickTests(DeployTestCase):
    """Deploying, skipping and refusing commits."""

    def test_the_first_deploy_goes_through_every_step_even_at_origin_main(self) -> None:
        self.assertEqual(self.head(), self.initial)  # nothing to fetch, but nothing deployed
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(f"deployed {self.initial[:12]} (feat: initial) in ", text(lines))
        self.assertIn("[cargo build]    Compiling three_body_problem", text(lines))

        cargo = self.calls("cargo")
        self.assertEqual(
            [call[0] for call in cargo],
            [["build", "--release", "--locked"], ["test", "--release", "--locked"]],
        )
        for _argv, cwd, target_dir, ci, niceness, sees_token in cargo:
            self.assertEqual(cwd, str(self.paths.stage))
            self.assertEqual(target_dir, str(self.paths.cargo_target_dir))
            self.assertEqual(ci, "true")
            assert isinstance(niceness, int)
            self.assertGreaterEqual(niceness, 10)
            self.assertFalse(sees_token)
        self.assertEqual(
            self.calls("pytests"),
            [[["-m", "unittest", "discover", "-s", "tests/python"], str(self.paths.stage), False]],
        )
        self.assertEqual(git(self.paths.stage, "rev-parse", "HEAD"), self.initial)

        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        self.assertEqual(stat.S_IMODE(self.binary.stat().st_mode), 0o755)
        self.assertEqual(self.previous_binary.read_text(), LEGACY_BINARY)
        release = self.paths.releases / self.initial / deploy.BINARY_NAME
        self.assertEqual(release.read_bytes(), self.binary.read_bytes())
        for name in deploy.UNIT_NAMES:
            self.assertTrue((self.paths.unit_dir / name).is_file(), name)
        verbs = self.systemctl_verbs()
        self.assertIn(["daemon-reload"], verbs)
        self.assertEqual(
            verbs[-2:],
            [["enable", "--now", deploy.SYNC_TIMER], ["start", "--no-block", deploy.SYNC_SERVICE]],
        )
        # smoke tests: the new run.py and agent start
        smoke = [call[0] for call in self.calls("python")]
        self.assertEqual(
            smoke,
            [[str(self.repo / "run.py"), "--help"], [str(self.repo / deploy.AGENT_PATH), "--help"]],
        )

        state = self.state()
        self.assertEqual(state.deployed_sha, self.initial)
        self.assertEqual(state.deployed_subject, "feat: initial")
        self.assertEqual(state.binary_sha256, deploy.file_sha256(self.binary))
        self.assertIsNone(state.previous_sha)  # the checkout already was at origin/main
        self.assertIsNone(state.last_error)
        self.assertEqual(state.failed_shas, {})

    def test_a_no_op_tick_is_quiet_and_cheap(self) -> None:
        self.mark_deployed()
        state_mtime = self.paths.state_file.stat().st_mtime_ns
        with self.assertNoLogs(deploy.log, level="INFO"):
            self.assertEqual(self.tick(), 0)
        self.assertEqual(self.github.requests, [])  # no API call
        self.assertEqual(self.call_log(), [])  # no systemctl, cargo or python
        self.assertEqual(self.paths.state_file.stat().st_mtime_ns, state_mtime)
        self.assertFalse(self.paths.stage.exists())

    def test_a_failed_fetch_is_a_warning_and_changes_nothing(self) -> None:
        self.mark_deployed()
        self.commit("feat: v2", {"bin_version.txt": "2\n"})
        git(self.repo, "remote", "set-url", "origin", "https://user:hunter2@127.0.0.1:1/r.git")
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("WARNING:"), lines)
        self.assertIn("cannot fetch origin/main", lines[0])
        self.assertIn("(retrying next tick)", lines[0])
        self.assertNotIn("hunter2", lines[0])
        self.assertIn("cannot fetch origin/main", self.state().last_error or "")
        self.assertEqual(self.github.requests, [])
        self.assertEqual(self.call_log(), [])
        self.assert_checkout_at(self.initial)

    def test_a_broken_staging_worktree_is_recreated(self) -> None:
        self.mark_deployed()
        self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        # Something replaced the stage's .git file (the link to the checkout's repository), so
        # the stage is no worktree any more, although git still has it registered.
        gitfile = self.paths.stage / ".git"
        gitfile.unlink()
        gitfile.mkdir()
        v3 = self.commit("feat: v3", {"bin_version.txt": "3\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(v3)
        self.assertEqual(git(self.paths.stage, "rev-parse", "HEAD"), v3)
        self.assertEqual(self.binary.read_text(), fake_binary_text("3"))

    def test_a_new_commit_is_fast_forwarded_built_and_switched(self) -> None:
        self.mark_deployed()
        old_binary = self.binary.read_bytes()
        marker = self.root / "hook-ran"
        hook = self.repo / ".git" / "hooks" / "post-merge"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
        hook.chmod(0o755)
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(f"deployed {new[:12]} (feat: v2) in ", text(lines))
        self.assertFalse(marker.exists())  # the checkout's hooks do not run unattended
        self.assert_checkout_at(new)
        self.assertEqual(self.binary.read_text(), fake_binary_text("2"))
        self.assertEqual(self.previous_binary.read_bytes(), old_binary)
        self.assertEqual(
            sorted(p.name for p in self.binary.parent.iterdir()),
            [deploy.BINARY_NAME, deploy.BINARY_NAME + deploy.PREVIOUS_SUFFIX],
        )
        state = self.state()
        self.assertEqual((state.deployed_sha, state.previous_sha), (new, self.initial))
        verbs = self.systemctl_verbs()
        self.assertEqual(verbs[1], ["stop", deploy.SYNC_TIMER])
        self.assertEqual(
            verbs[-2:],
            [["enable", "--now", deploy.SYNC_TIMER], ["start", "--no-block", deploy.SYNC_SERVICE]],
        )
        self.assertNotIn(["daemon-reload"], verbs)  # no unit changed
        # the next tick has nothing to do
        with self.assertNoLogs(deploy.log, level="INFO"):
            self.assertEqual(self.tick(), 0)

    def test_an_identical_binary_is_left_alone(self) -> None:
        self.mark_deployed()
        mtime = self.binary.stat().st_mtime_ns
        new = self.commit("docs: readme", {"README.md": "# changed\n"})
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("the generator binary is unchanged", text(lines))
        self.assertEqual(self.binary.stat().st_mtime_ns, mtime)  # run.py's identity unchanged
        self.assertFalse(self.previous_binary.exists())
        self.assert_checkout_at(new)
        self.assertEqual(self.state().deployed_sha, new)

    def test_units_are_installed_only_when_they_change(self) -> None:
        self.mark_deployed()
        units = self.paths.unit_dir
        mtimes = {name: (units / name).stat().st_mtime_ns for name in deploy.UNIT_NAMES}
        template = f"{deploy.UNIT_TEMPLATE_DIR}/{deploy.SYNC_TIMER}"
        changed = (self.dev / template).read_text().replace("=5min", "=7min")
        self.commit("feat: slower sync", {template: changed})
        self.assertEqual(self.tick(), 0)
        self.assertIn("OnUnitInactiveSec=7min", (units / deploy.SYNC_TIMER).read_text())
        for name in deploy.UNIT_NAMES:
            if name != deploy.SYNC_TIMER:
                self.assertEqual((units / name).stat().st_mtime_ns, mtimes[name], name)
        verbs = self.systemctl_verbs()
        self.assertEqual(verbs.count(["daemon-reload"]), 1)
        # After the reload, every unit must load, and systemd-analyze must accept the files.
        reload = verbs.index(["daemon-reload"])
        self.assertEqual(
            [verb[-1] for verb in verbs[reload + 1 : reload + 5]], list(deploy.UNIT_NAMES)
        )
        self.assertEqual(
            self.calls("systemd-analyze"),
            [["--user", "--man=no", "verify", *(str(units / n) for n in deploy.UNIT_NAMES)]],
        )

    def test_a_modified_binary_is_reinstalled_from_the_tested_release(self) -> None:
        self.mark_deployed()
        tested = self.binary.read_bytes()
        self.binary.write_text("#!/bin/sh\necho built by hand\n", encoding="utf-8")
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("reusing the built and tested release", text(lines))
        self.assertEqual(self.calls("cargo"), [])
        self.assertEqual(self.binary.read_bytes(), tested)

    def test_untracked_production_state_is_never_touched(self) -> None:
        production = {
            ".env": "COSMICSIG_SSH_HOST=assets.example\n",
            "output/0xabc/images/source/master.png": "art",
            "imgcheck.log": "log\n",
            "generation_log.json": "[]\n",
            "backfill_failures.json": "{}\n",
            "seed_source_mismatch.json": "{}\n",
        }
        for relative, content in production.items():
            (self.repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / relative).write_text(content, encoding="utf-8")
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(v2)
        for relative, content in production.items():
            self.assertEqual((self.repo / relative).read_text(), content, relative)

        # A commit that adds a tracked file where untracked production state lives is refused:
        # git would silently overwrite the ignored .env.
        v3 = self.commit("feat: tracked env", {".env": "leaked\n", "bin_version.txt": "3\n"})
        with self.logs() as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn("would overwrite untracked files in the checkout: .env", text(lines))
        self.assertEqual((self.repo / ".env").read_text(), production[".env"])
        self.assert_checkout_at(v2)
        self.assertEqual(self.binary.read_text(), fake_binary_text("2"))
        self.assertEqual(self.state().failed_shas[v3].reason, "switch")

    def test_a_tracked_file_that_becomes_a_directory_is_no_collision(self) -> None:
        # git replaces what the old commit tracks by itself; only untracked files are in the way.
        self.mark_deployed()
        v2 = self.commit("docs: notes", {"notes": "a file\n"})
        self.assertEqual(self.tick(), 0)
        v3 = self.commit("docs: a notes directory", {"notes": None, "notes/readme.md": "dir\n"})
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertEqual(errors(lines), [])
        self.assert_checkout_at(v3)
        self.assertEqual((self.repo / "notes" / "readme.md").read_text(), "dir\n")

        # The way back (a tracked directory becomes a file again) is refused only while an
        # untracked file lives in the directory: `git reset --hard` would delete it.
        local = self.repo / "notes" / "local.txt"
        local.write_text("mine\n", encoding="utf-8")
        with self.logs("ERROR") as lines:
            self.assertEqual(self.agent("rollback"), 1)
        self.assertIn("would overwrite untracked files in the checkout: notes", text(lines))
        self.assert_checkout_at(v3)
        self.assertEqual(local.read_text(), "mine\n")
        self.assertIsNone(self.state().switch_in_progress)
        local.unlink()
        self.assertEqual(self.agent("rollback"), 0)
        self.assert_checkout_at(v2)
        self.assertEqual((self.repo / "notes").read_text(), "a file\n")


class CiGateTests(DeployTestCase):
    """The GitHub check-run gate."""

    def test_waiting_for_ci(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        for runs, expected in (
            ([], "no 'CI passed' check run yet"),
            ([check_run(new, status="queued")], "'CI passed' is queued"),
            ([check_run(new, status="in_progress")], "'CI passed' is in_progress"),
        ):
            self.github.runs[new] = runs
            with self.subTest(expected=expected), self.logs() as lines:
                self.assertEqual(self.tick(), 0)
            self.assertIn(f"waiting for CI on {new[:12]}: {expected}", text(lines))
        self.assertEqual(self.calls("cargo"), [])
        self.assert_checkout_at(self.initial)
        self.github.runs[new] = [check_run(new)]
        self.assertEqual(self.tick(), 0)
        self.assertEqual(self.state().deployed_sha, new)

    def test_a_commit_that_waits_long_for_ci_is_a_warning(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.github.runs[new] = []  # e.g. a merge whose message said [skip ci]: no run at all
        with self.logs() as lines:  # a fresh commit: CI simply has not reported yet
            self.assertEqual(self.tick(), 0)
        self.assertEqual([line for line in lines if line.startswith("WARNING:")], [])
        self.assertIsNone(self.state().last_error)

        later = deploy.utcnow() + deploy.CI_STALL_WARNING + datetime.timedelta(minutes=1)
        with mock.patch.object(deploy, "utcnow", return_value=later), self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        warnings = [line for line in lines if line.startswith("WARNING:")]
        self.assertEqual(len(warnings), 1, text(lines))
        self.assertIn(f"still waiting for CI on {new[:12]}", warnings[0])
        self.assertIn("gh workflow run ci.yml --ref main", warnings[0])
        self.assertIn("still waiting for CI", self.state().last_error or "")
        self.assertEqual(self.calls("cargo"), [])
        self.assert_checkout_at(self.initial)

        self.github.runs[new] = [check_run(new)]  # a manual run reports: deployed, error cleared
        self.assertEqual(self.tick(), 0)
        self.assertEqual(self.state().deployed_sha, new)
        self.assertIsNone(self.state().last_error)

    def test_a_ci_failure_is_recorded_and_re_checked_every_15_minutes(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.github.runs[new] = [check_run(new, conclusion="failure")]
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("'CI passed' concluded failure", text(errors(lines)))
        self.assertEqual(self.state().failed_shas[new].reason, "ci")
        asked = len(self.github.requests)

        with self.logs() as lines:  # within 15 minutes: GitHub is not asked
            self.assertEqual(self.tick(), 0)
        self.assertEqual(len(self.github.requests), asked)
        self.assertIn("failed CI", text(lines))
        self.assertEqual(errors(lines), [])

        self.age_ci_check(new, minutes=16)  # still red: one more question, no new ERROR
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertEqual(len(self.github.requests), asked + 1)
        self.assertIn("still fails CI", text(lines))
        self.assertEqual(errors(lines), [])

        self.age_ci_check(new, minutes=16)  # re-run on GitHub: the newest run decides
        self.github.runs[new] = [
            check_run(new, conclusion="failure", run_id=1),
            check_run(new, conclusion="success", run_id=2),
        ]
        self.assertEqual(self.tick(), 0)
        state = self.state()
        self.assertEqual(state.deployed_sha, new)
        self.assertNotIn(new, state.failed_shas)
        self.assertEqual(self.calls("cargo")[0][0], ["build", "--release", "--locked"])

    def test_a_re_run_in_progress_clears_the_ci_failure(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.github.runs[new] = [check_run(new, conclusion="timed_out")]
        self.assertEqual(self.tick(), 0)
        self.age_ci_check(new, minutes=16)
        self.github.runs[new].append(check_run(new, status="in_progress", run_id=2))
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("waiting for CI", text(lines))
        self.assertNotIn(new, self.state().failed_shas)

    def test_network_errors_and_rate_limits_never_deploy(self) -> None:
        self.mark_deployed()
        self.commit("feat: v2", {"bin_version.txt": "2\n"})

        def refuse() -> None:
            os.environ[deploy.ENV_GITHUB_API] = closed_port_url()

        def rate_limit() -> None:
            self.github.status = 403
            self.github.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1900000000"}

        def secondary_rate_limit() -> None:
            self.github.status, self.github.headers = 429, {"retry-after": "60"}

        def server_error() -> None:
            self.github.status = 502

        def not_json() -> None:
            self.github.body = b"<html>unicorn</html>"

        def slow() -> None:
            self.github.delay = 1.0

        cases = (
            (refuse, "GitHub API request failed"),
            (rate_limit, "rate limit exceeded (HTTP 403; it resets at 2030-03-17T17:46:40+00:00)"),
            (secondary_rate_limit, "rate limit exceeded (HTTP 429; retry after 60s)"),
            (server_error, "GitHub API returned HTTP 502"),
            (not_json, "GitHub API response is not JSON"),
            (slow, "GitHub API request failed"),
        )
        for configure, expected in cases:
            with self.subTest(expected), mock.patch.object(deploy, "API_TIMEOUT", 0.3):
                configure()
                with self.logs("WARNING") as lines:
                    self.assertEqual(self.tick(), 0)
                self.assertIn("cannot check CI for ", text(lines))
                self.assertIn(expected, text(lines))
                self.assertEqual(self.calls("cargo"), [])
                self.assert_checkout_at(self.initial)
                self.assertIn(expected, self.state().last_error or "")
                self.github.reset()
                os.environ[deploy.ENV_GITHUB_API] = self.github.url
        self.assertIn("set COSMICSIG_DEPLOY_GITHUB_TOKEN", self.github_hint())
        self.assertEqual(self.tick(), 0)
        self.assertIsNone(self.state().last_error)  # cleared by the successful tick

    def github_hint(self) -> str:
        """The rate-limit message, which suggests a token when none is set."""
        error = mock.Mock(code=403, reason="Forbidden", headers={"x-ratelimit-remaining": "0"})
        return deploy._describe_http_error(error)

    def test_the_api_request_and_the_optional_token(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.github.runs[new] = []  # pending: every tick asks
        self.assertEqual(self.tick(), 0)
        path, headers = self.github.requests[-1]
        self.assertEqual(
            path,
            f"/repos/Owner/Repo/commits/{new}/check-runs"
            "?check_name=CI%20passed&filter=latest&per_page=100",
        )
        self.assertEqual(headers["accept"], "application/vnd.github+json")
        self.assertEqual(headers["x-github-api-version"], "2022-11-28")
        self.assertTrue(headers["user-agent"].startswith("cosmicsig-deploy/"))
        self.assertNotIn("authorization", headers)

        os.environ["GITHUB_TOKEN"] = "fallback-token"
        self.assertEqual(self.tick(), 0)
        self.assertEqual(self.github.requests[-1][1]["authorization"], "Bearer fallback-token")
        os.environ[deploy.ENV_GITHUB_TOKEN] = "primary-token"
        with self.logs("DEBUG") as lines:
            self.assertEqual(self.tick(), 0)
        self.assertEqual(self.github.requests[-1][1]["authorization"], "Bearer primary-token")
        self.assertNotIn("primary-token", text(lines))

    def test_a_rejected_token_is_an_error_and_the_anonymous_answer_decides(self) -> None:
        # GitHub answers an expired or revoked token with 401, even for a public repository.
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        os.environ["GITHUB_TOKEN"] = "expired-token"
        self.github.rejected_token = "expired-token"
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("GITHUB_TOKEN was rejected by GitHub (HTTP 401)", text(errors(lines)))
        self.assertNotIn("expired-token", text(lines))
        self.assertEqual(
            [headers.get("authorization") for _, headers in self.github.requests],
            ["Bearer expired-token", None],
        )
        self.assert_checkout_at(v2)  # the anonymous answer (CI passed) decided
        self.assertIn("GITHUB_TOKEN was rejected", self.state().last_error or "")

        # The error names the variable that holds the token, and stays in last_error while a
        # commit waits for CI (a tick that asks GitHub nothing has nothing to report).
        v3 = self.commit("feat: v3", {"bin_version.txt": "3\n"})
        self.github.runs[v3] = []
        os.environ[deploy.ENV_GITHUB_TOKEN] = "revoked-token"
        self.github.rejected_token = "revoked-token"
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(f"{deploy.ENV_GITHUB_TOKEN} was rejected", text(errors(lines)))
        self.assertIn(f"waiting for CI on {v3[:12]}", text(lines))
        self.assertIn(f"{deploy.ENV_GITHUB_TOKEN} was rejected", self.state().last_error or "")

    def test_the_repository_slug_comes_from_origin(self) -> None:
        cases = {
            "https://github.com/PredictionExplorer/CS-Image-Generation.git": (
                "PredictionExplorer/CS-Image-Generation"
            ),
            "https://github.com/PredictionExplorer/CS-Image-Generation": (
                "PredictionExplorer/CS-Image-Generation"
            ),
            "https://x-access-token:secret@github.com/owner/repo.git": "owner/repo",
            "git@github.com:owner/repo.git": "owner/repo",
            "ssh://git@github.com/owner/repo.git": "owner/repo",
            "https://gitlab.com/owner/repo.git": None,
            "/srv/git/repo.git": None,
        }
        for url, slug in cases.items():
            with self.subTest(url):
                self.assertEqual(deploy.parse_github_slug(url), slug)
        del os.environ[deploy.ENV_GITHUB_REPO]
        git(self.repo, "remote", "set-url", "origin", "https://github.com/owner/repo.git")
        self.assertEqual(deploy.github_slug(self.repo), "owner/repo")
        git(self.repo, "remote", "set-url", "origin", "https://user:hunter2@example.com/r.git")
        with self.assertRaises(deploy.DeployError) as raised:
            deploy.github_slug(self.repo)
        self.assertNotIn("hunter2", str(raised.exception))
        self.assertIn(deploy.ENV_GITHUB_REPO, str(raised.exception))


class SafetyTests(DeployTestCase):
    """Refusals: an unsafe checkout, a rewritten main, a commit that would stop auto-deploy."""

    def test_an_unsafe_checkout_is_refused_until_it_is_repaired(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})

        def dirty() -> None:
            (self.repo / "README.md").write_text("edited on the server\n", encoding="utf-8")

        cases: tuple[tuple[Callable[[], object], Callable[[], object], str], ...] = (
            (
                dirty,
                lambda: git(self.repo, "checkout", "--", "README.md"),
                "has uncommitted changes to tracked files (README.md)",
            ),
            (
                lambda: git(self.repo, "switch", "--quiet", "-c", "hotfix"),
                lambda: git(self.repo, "switch", "--quiet", "main"),
                "is on hotfix, not main",
            ),
            (
                lambda: git(self.repo, "checkout", "--quiet", "--detach"),
                lambda: git(self.repo, "switch", "--quiet", "main"),
                "is on a detached HEAD",
            ),
        )
        for break_checkout, repair, expected in cases:
            with self.subTest(expected):
                break_checkout()
                with self.logs("ERROR") as lines:
                    self.assertEqual(self.tick(), 1)
                self.assertIn(expected, text(lines))
                self.assertIn(expected, self.state().last_error or "")
                self.assertEqual(self.calls("cargo"), [])
                self.assertEqual(self.head(), self.initial)
                repair()
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(new)
        self.assertIsNone(self.state().last_error)

    def test_a_rewritten_main_is_refused(self) -> None:
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        git(self.dev, "reset", "--quiet", "--hard", self.initial)
        self.commit("feat: rewritten", {"bin_version.txt": "3\n"}, force=True)
        with self.logs("ERROR") as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn("main was rewritten", text(lines))
        self.assert_checkout_at(v2)
        self.assertEqual(self.binary.read_text(), fake_binary_text("2"))

    def test_local_commits_in_the_checkout_are_refused(self) -> None:
        self.mark_deployed()
        self.commit("feat: v2", {"bin_version.txt": "2\n"})
        (self.repo / "NOTES.md").write_text("server-side note\n", encoding="utf-8")
        git(self.repo, "add", "NOTES.md")
        git(self.repo, "commit", "--quiet", "-m", "chore: local")
        local = self.head()
        with self.logs("ERROR") as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn("does not descend from the checkout's HEAD", text(lines))
        self.assertEqual(self.head(), local)

    def test_a_commit_that_would_stop_auto_deploy_is_refused(self) -> None:
        self.mark_deployed()
        bad = self.commit("refactor: move the agent", {deploy.AGENT_PATH: None})
        with self.logs("ERROR") as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn("lacks ops/deploy/cosmicsig_deploy.py", text(lines))
        self.assertEqual(self.state().failed_shas[bad].reason, "switch")
        self.assertEqual(self.calls("cargo"), [])
        self.assert_checkout_at(self.initial)

    def test_a_second_tick_exits_quietly_while_one_runs(self) -> None:
        self.mark_deployed()
        self.commit("feat: v2", {"bin_version.txt": "2\n"})
        holder = HeldLock(self.paths.deploy_lock)
        self.addCleanup(holder.release)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertEqual(
            lines,
            [
                f"INFO:cosmicsig.deploy:another deploy command holds "
                f"{self.paths.deploy_lock}; nothing to do"
            ],
        )
        self.assertEqual(self.github.requests, [])
        self.assertEqual(self.call_log(), [])
        self.assertEqual(self.head(), self.initial)


class BuildTests(DeployTestCase):
    """Build and test failures, timeouts and releases."""

    def test_build_and_test_failures_are_final_until_retry_or_a_new_commit(self) -> None:
        self.mark_deployed()
        cases = (
            ({"FAKE_CARGO_FAIL": "build"}, "build", "error: fake build failure"),
            ({"FAKE_CARGO_FAIL": "test"}, "tests", "error: fake test failure"),
            ({"FAKE_PYTEST_RC": "1"}, "tests", "FAILED (failures=1)"),
        )
        for version, (env, reason, output) in enumerate(cases, start=2):
            with self.subTest(reason=reason, env=env):
                sha = self.commit(f"feat: v{version}", {"bin_version.txt": f"{version}\n"})
                os.environ.update(env)
                with self.logs("ERROR") as lines:
                    self.assertEqual(self.tick(), 1)
                self.assertIn(f"not deploying {sha[:12]}: {reason} failed", text(lines))
                failure = self.state().failed_shas[sha]
                self.assertEqual(failure.reason, reason)
                self.assertIn(output, failure.detail)

                self.clear_calls()  # final: the next tick neither builds nor asks GitHub
                asked = len(self.github.requests)
                with self.logs() as lines:
                    self.assertEqual(self.tick(), 0)
                self.assertIn(f"its {reason} failed", text(lines))
                self.assertEqual((self.calls("cargo"), len(self.github.requests)), ([], asked))
                self.assertNotEqual(self.head(), sha)

                for key in env:
                    del os.environ[key]
                with self.logs() as lines:
                    self.assertEqual(self.agent("retry"), 0)
                self.assertIn(f"forgot the {reason} failure of {sha[:12]}", text(lines))
                self.assertEqual(self.tick(), 0)
                self.assert_checkout_at(sha)

        # A new commit supersedes a failed one without `retry`.
        os.environ["FAKE_CARGO_FAIL"] = "build"
        failed = self.commit("feat: broken build", {"bin_version.txt": "9\n"})
        self.assertEqual(self.tick(), 1)
        del os.environ["FAKE_CARGO_FAIL"]
        fixed = self.commit("fix: build", {"bin_version.txt": "10\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(fixed)
        self.assertIn(failed, self.state().failed_shas)

    def test_the_deployed_binary_is_the_one_cargo_build_made(self) -> None:
        self.mark_deployed()
        os.environ["FAKE_CARGO_TEST_REBUILDS"] = "1"  # `cargo test` overwrites target/release
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(new)
        built = self.paths.cargo_target_dir / "release" / deploy.BINARY_NAME
        self.assertEqual(built.read_text(), fake_binary_text("2-test"))
        self.assertEqual(self.binary.read_text(), fake_binary_text("2"))
        release = self.paths.releases / new / deploy.BINARY_NAME
        self.assertEqual(release.read_text(), fake_binary_text("2"))
        self.assertFalse((self.paths.data_dir / "built" / deploy.BINARY_NAME).exists())

    def test_no_command_the_agent_runs_sees_the_github_token(self) -> None:
        os.environ[deploy.ENV_GITHUB_TOKEN] = "primary-token"
        os.environ["GITHUB_TOKEN"] = "fallback-token"
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(new)
        self.assertEqual(self.github.requests[-1][1]["authorization"], "Bearer primary-token")
        self.assertEqual([call[-1] for call in self.calls("cargo")], [False, False])
        self.assertEqual([call[-1] for call in self.calls("pytests")], [False])

    def test_retry_takes_an_abbreviated_sha_and_rejects_nonsense(self) -> None:
        self.mark_deployed()
        os.environ["FAKE_CARGO_FAIL"] = "build"
        sha = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 1)
        with self.logs("ERROR"):
            self.assertEqual(self.agent("retry", "zzzz"), 1)
        with self.logs() as lines:
            self.assertEqual(self.agent("retry", self.initial[:10]), 0)
        self.assertIn("has no recorded failure", text(lines))
        self.assertEqual(self.agent("retry", sha[:10]), 0)
        self.assertNotIn(sha, self.state().failed_shas)

    def test_a_build_that_times_out_is_a_build_failure(self) -> None:
        self.mark_deployed()
        os.environ["FAKE_CARGO_SLEEP"] = "build"
        sha = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        started = time.monotonic()
        with mock.patch.object(deploy, "BUILD_TIMEOUT", 1.0), self.logs("ERROR"):
            self.assertEqual(self.tick(), 1)
        self.assertLess(time.monotonic() - started, 30)
        failure = self.state().failed_shas[sha]
        self.assertEqual(failure.reason, "build")
        self.assertIn("timed out after 1s", failure.detail)

    def test_a_stop_request_during_the_build_kills_it_and_records_nothing(self) -> None:
        self.mark_deployed()
        os.environ["FAKE_CARGO_SLEEP"] = "build"
        sha = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        with self.stop_request_after(1.0), self.logs("WARNING") as lines:
            started = time.monotonic()
            self.assertEqual(self.tick(), 1)
        self.assertLess(time.monotonic() - started, 30)  # the sleeping cargo was killed
        self.assertIn("stopped by a signal before switching; nothing changed", text(lines))
        self.assertNotIn(sha, self.state().failed_shas)
        self.assert_checkout_at(self.initial)

    def test_old_releases_are_pruned_but_never_the_deployed_or_previous_one(self) -> None:
        source = self.root / "binary"
        source.write_text("binary\n", encoding="utf-8")
        shas = [f"{index:040x}" for index in range(8)]  # oldest first
        base = deploy.utcnow() - datetime.timedelta(days=1)
        for index, sha in enumerate(shas):
            deploy.publish_release(self.paths, sha, source, source="test")
            manifest = self.paths.releases / sha / deploy.RELEASE_MANIFEST
            data = json.loads(manifest.read_text())
            data["built_at"] = deploy.isoformat(base + datetime.timedelta(minutes=index))
            manifest.write_text(json.dumps(data))
        (self.paths.releases / f".{'f' * 40}.tmp").mkdir()  # an interrupted publish
        deploy.prune_releases(self.paths, keep={shas[0]})
        self.assertEqual(
            sorted(path.name for path in self.paths.releases.iterdir()),
            sorted([shas[0], *shas[3:]]),
        )

    def test_releases_published_within_one_second_are_pruned_by_publish_order(self) -> None:
        # Regression: built_at was written to the second, so releases published within one
        # second tied and the file system's listing order decided which one survived.
        source = self.root / "binary"
        source.write_text("binary\n", encoding="utf-8")
        second = datetime.datetime(2026, 9, 30, 6, 0, 0, tzinfo=datetime.timezone.utc)
        # Published newest last, with names that sort the other way: a name tie-break alone
        # cannot pass, and the built_at assertion catches second-precision ties whatever the
        # listing order.
        shas = [f"{index:040x}" for index in (3, 2, 1, 0)]
        for offset, sha in enumerate(shas):
            moment = second + datetime.timedelta(microseconds=10 * offset)
            with mock.patch.object(deploy, "utcnow", return_value=moment):
                deploy.publish_release(self.paths, sha, source, source="test")
        manifest = json.loads((self.paths.releases / shas[1] / deploy.RELEASE_MANIFEST).read_text())
        self.assertEqual(manifest["built_at"], "2026-09-30T06:00:00.000010+00:00")
        with mock.patch.object(deploy, "KEEP_RELEASES", 1):
            deploy.prune_releases(self.paths, keep=set())
        self.assertEqual([path.name for path in self.paths.releases.iterdir()], [shas[-1]])

    def test_releases_of_the_same_time_are_pruned_by_name_whatever_the_listing_order(self) -> None:
        # built_at written to the second by an earlier version of the tool can tie exactly.
        source = self.root / "binary"
        source.write_text("binary\n", encoding="utf-8")
        shas = [f"{index:040x}" for index in range(4)]
        survivors = []
        for listing in (sorted, lambda names: sorted(names, reverse=True)):
            shutil.rmtree(self.paths.releases, ignore_errors=True)
            for sha in shas:
                deploy.publish_release(self.paths, sha, source, source="test")
                manifest = self.paths.releases / sha / deploy.RELEASE_MANIFEST
                data = json.loads(manifest.read_text())
                data["built_at"] = "2026-09-30T06:00:00+00:00"
                manifest.write_text(json.dumps(data))
            releases = self.paths.releases
            listed = [releases / name for name in listing(p.name for p in releases.iterdir())]
            with (
                mock.patch.object(deploy, "KEEP_RELEASES", 2),
                mock.patch.object(type(releases), "iterdir", return_value=iter(listed)),
            ):
                deploy.prune_releases(self.paths, keep=set())
            survivors.append(sorted(path.name for path in releases.iterdir()))
        self.assertEqual(survivors, [shas[2:], shas[2:]])

    def test_deploys_keep_the_releases_that_rollback_needs(self) -> None:
        self.mark_deployed()
        with mock.patch.object(deploy, "KEEP_RELEASES", 1):
            v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
            self.assertEqual(self.tick(), 0)
            self.assertEqual(
                sorted(p.name for p in self.paths.releases.iterdir()), sorted([self.initial, v2])
            )
            v3 = self.commit("feat: v3", {"bin_version.txt": "3\n"})
            self.assertEqual(self.tick(), 0)
            self.assertEqual(
                sorted(p.name for p in self.paths.releases.iterdir()), sorted([v2, v3])
            )


class SwitchTests(DeployTestCase):
    """Waiting for the sync, the switch's rollback, and stop requests."""

    def test_the_switch_waits_for_the_running_sync(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.set_systemctl(
            script={deploy.SYNC_SERVICE: ["active", "activating", "deactivating", "active"]}
        )
        os.environ["FAKE_WATCH"] = str(self.binary)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(
            "waiting for the running sync (cosmicsig-sync.service is active)", text(lines)
        )
        self.assertIn("the sync run finished after", text(lines))
        # five polls (four busy, one idle); the checkout held the old binary throughout the wait
        self.assertEqual(self.calls("watch"), [[fake_binary_text("1")]] * 5)
        verbs = self.systemctl_verbs()
        self.assertLess(
            verbs.index(["stop", deploy.SYNC_TIMER]),
            verbs.index(["is-active", deploy.SYNC_SERVICE]),
        )
        self.assert_checkout_at(new)

    def test_pausing_while_waiting_for_the_sync_abandons_the_switch(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.set_systemctl(script={deploy.SYNC_SERVICE: ["active"] * 100})
        os.environ["FAKE_PAUSE_AFTER"] = "3"
        os.environ["FAKE_PAUSE_FILE"] = str(self.paths.pause_file)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(
            "abandoned: auto-deploy was paused (paused mid-wait); nothing changed", text(lines)
        )
        self.assertEqual(errors(lines), [])
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.SYNC_TIMER])
        self.assertNotIn(new, self.state().failed_shas)

        self.set_systemctl(script={deploy.SYNC_SERVICE: []})
        self.assertEqual(self.agent("resume"), 0)
        self.clear_calls()
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(new)
        self.assertEqual(self.calls("cargo"), [])  # the tested release is reused

    def test_a_pause_during_the_build_switches_nothing(self) -> None:
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        build_release = deploy.build_release

        def paused_while_building(paths: deploy.Paths, target: str) -> Path:
            release = build_release(paths, target)
            deploy.write_pause(paths, "paused during the build")
            return release

        with (
            mock.patch.object(deploy, "build_release", side_effect=paused_while_building),
            self.logs() as lines,
        ):
            self.assertEqual(self.tick(), 0)
        self.assertIn(
            f"switch to {v2[:12]} abandoned: auto-deploy was paused (paused during the build); "
            "nothing changed",
            text(lines),
        )
        self.assertEqual(errors(lines), [])
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        # The sync timer was not touched (a needless stop and restart can move its schedule).
        self.assertEqual(self.systemctl_verbs(), [])
        self.assertNotIn(v2, self.state().failed_shas)

        self.assertEqual(self.agent("resume"), 0)
        self.clear_calls()
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(v2)
        self.assertEqual(self.calls("cargo"), [])  # the tested release is reused

    def test_a_pause_that_finds_the_sync_idle_still_stops_the_switch(self) -> None:
        # The sync is idle, so the switch does not wait (and never sleeps); the pause arrives
        # after the timer was stopped. The check under run.lock catches it.
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        os.environ["FAKE_PAUSE_AFTER"] = "1"
        os.environ["FAKE_PAUSE_FILE"] = str(self.paths.pause_file)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("abandoned: auto-deploy was paused (paused mid-wait)", text(lines))
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.SYNC_TIMER])
        self.assertNotIn(v2, self.state().failed_shas)

    def test_a_rollback_during_a_build_rolls_back_the_deployed_commit(self) -> None:
        # rollback pauses first; a tick that is building the next commit meanwhile must not
        # deploy it, or the rollback would land on the bad commit it was meant to leave.
        self.mark_deployed()
        bad = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        v3 = self.commit("feat: v3", {"bin_version.txt": "3\n"})
        build_release = deploy.build_release

        def rollback_requested(paths: deploy.Paths, target: str) -> Path:
            release = build_release(paths, target)
            deploy.write_pause(paths, f"rollback of {bad[:12]} requested")
            return release

        with mock.patch.object(deploy, "build_release", side_effect=rollback_requested):
            self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(bad)
        self.assertEqual(self.agent("rollback"), 0)
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        state = self.state()
        self.assertEqual((state.deployed_sha, state.rolled_back_from), (self.initial, bad))
        self.assertNotIn(v3, state.failed_shas)

    def test_a_run_py_started_by_hand_is_waited_for(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        holder = HeldLock(self.repo / "run.lock")
        self.addCleanup(holder.release)
        sleep = deploy.interruptible_sleep

        def finish_the_run(seconds: float, **kwargs: deploy.Paths | None) -> None:
            holder.release()  # the hand-started run.py finishes while the agent waits
            sleep(seconds, **kwargs)

        with (
            mock.patch.object(deploy, "interruptible_sleep", side_effect=finish_the_run) as waits,
            self.logs() as lines,
        ):
            self.assertEqual(self.tick(), 0)
        self.assertEqual(waits.call_count, 1)
        self.assertIn("waiting for a run.py started outside systemd to finish", text(lines))
        self.assertIn("held by pid 4242", text(lines))
        self.assert_checkout_at(new)

    def test_a_run_lock_that_stays_busy_abandons_the_switch(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        holder = HeldLock(self.repo / "run.lock")
        self.addCleanup(holder.release)
        with mock.patch.object(deploy, "RUN_LOCK_TIMEOUT", 0.3), self.logs("WARNING") as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("is still held (pid 4242)", text(lines))
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.SYNC_TIMER])
        self.assertNotIn(new, self.state().failed_shas)
        self.assertIn("still held", self.state().last_error or "")

    def test_a_stop_request_while_waiting_restores_the_timer_and_changes_nothing(self) -> None:
        self.mark_deployed()
        self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.set_systemctl(script={deploy.SYNC_SERVICE: ["active"] * 100})
        previous = signal.signal(signal.SIGTERM, _ignore_signal)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        sleep = deploy.interruptible_sleep

        def systemctl_stop(seconds: float, **kwargs: deploy.Paths | None) -> None:
            os.kill(os.getpid(), signal.SIGTERM)  # `systemctl --user stop` during the wait
            sleep(seconds, **kwargs)

        with (
            mock.patch.object(deploy, "interruptible_sleep", side_effect=systemctl_stop),
            self.logs("WARNING") as lines,
        ):
            self.assertEqual(self.tick(), 1)
        self.assertIn("stopped by a signal before switching; nothing changed", text(lines))
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.SYNC_TIMER])
        self.assertIs(signal.getsignal(signal.SIGTERM), _ignore_signal)  # the agent's is gone

    def test_a_failed_smoke_test_rolls_back_code_binary_and_units(self) -> None:
        self.mark_deployed()
        tested, mtime = self.binary.read_bytes(), self.binary.stat().st_mtime_ns
        units = {name: (self.paths.unit_dir / name).read_text() for name in deploy.UNIT_NAMES}
        template = f"{deploy.UNIT_TEMPLATE_DIR}/{deploy.SYNC_TIMER}"
        changed_template = (self.dev / template).read_text().replace("=5min", "=9min")
        cases = (
            ({"bin_version.txt": "broken\n", template: changed_template}, "smoke test: generator"),
            (
                {"bin_version.txt": "2\n", "run.py": FAKE_RUN_PY.format(fail=True)},
                "smoke test: run.py",
            ),
            (
                {
                    "run.py": FAKE_RUN_PY.format(fail=False),
                    deploy.AGENT_PATH: "raise SystemExit(3)\n",
                },
                "smoke test: deploy agent",
            ),
        )
        for files, expected in cases:
            with self.subTest(expected):
                self.clear_calls()
                bad = self.commit(f"feat: breaks the {expected}", files)
                with self.logs() as lines:
                    self.assertEqual(self.tick(), 1)
                self.assertIn(expected, text(errors(lines)))
                self.assertIn(f"rolled back to {self.initial[:12]}", text(lines))
                self.assert_checkout_at(self.initial)
                self.assertEqual(self.binary.read_bytes(), tested)
                self.assertEqual(self.binary.stat().st_mtime_ns, mtime)  # same generator identity
                for name, content in units.items():
                    self.assertEqual((self.paths.unit_dir / name).read_text(), content, name)
                verbs = self.systemctl_verbs()
                self.assertEqual(verbs[-1], ["enable", "--now", deploy.SYNC_TIMER])  # restored
                self.assertNotIn(["start", "--no-block", deploy.SYNC_SERVICE], verbs)
                state = self.state()
                self.assertEqual(
                    (state.deployed_sha, state.failed_shas[bad].reason), (self.initial, "switch")
                )
        with self.logs() as lines:  # final
            self.assertEqual(self.tick(), 0)
        self.assertIn("its switch failed", text(lines))

    def test_an_unexpected_error_during_the_switch_is_rolled_back_too(self) -> None:
        self.mark_deployed()
        tested = self.binary.read_bytes()
        bad = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        with (
            mock.patch.object(deploy, "smoke_test", side_effect=TypeError("a bug")),
            self.logs() as lines,
        ):
            self.assertEqual(self.tick(), 1)
        self.assertIn(f"the switch to {bad[:12]} failed: a bug; rolling back", text(lines))
        self.assertIn(f"rolled back to {self.initial[:12]}", text(lines))
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_bytes(), tested)
        self.assertEqual(self.state().failed_shas[bad].reason, "switch")
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.SYNC_TIMER])

    def test_a_unit_change_that_fails_is_rolled_back_with_a_daemon_reload(self) -> None:
        self.mark_deployed()
        units = {name: (self.paths.unit_dir / name).read_text() for name in deploy.UNIT_NAMES}
        template = f"{deploy.UNIT_TEMPLATE_DIR}/{deploy.SYNC_TIMER}"
        changed = (self.dev / template).read_text().replace("=5min", "=9min")
        self.commit("feat: broken", {"bin_version.txt": "broken\n", template: changed})
        self.assertEqual(self.tick(), 1)
        self.assertEqual(self.systemctl_verbs().count(["daemon-reload"]), 2)  # switch and undo
        for name, content in units.items():
            self.assertEqual((self.paths.unit_dir / name).read_text(), content, name)

    def test_a_failed_undo_pauses_auto_deploy_and_disables_the_sync_timer(self) -> None:
        self.mark_deployed()
        template = f"{deploy.UNIT_TEMPLATE_DIR}/{deploy.SYNC_TIMER}"
        changed = (self.dev / template).read_text().replace("=5min", "=9min")
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n", template: changed})
        # An earlier switch also left a restart of the sync timer pending, which systemd still
        # refuses. The failed undo drops it: no tick may enable the timer on this checkout.
        with deploy.state_transaction(self.paths) as state:
            state.sync_timer_restart_pending = True
        os.environ["FAKE_SYSTEMCTL_FAIL"] = "daemon-reload,enable"
        with self.logs() as lines:
            self.assertEqual(self.tick(), 1)
        critical = [line for line in lines if line.startswith("CRITICAL:")]
        self.assertEqual(len(critical), 1)
        self.assertIn("undoing it failed too", critical[0])
        self.assertIn(f"{deploy.SYNC_TIMER} is stopped and disabled", critical[0])
        pause = deploy.read_pause(self.paths)
        assert pause is not None
        self.assertIn("undoing the failed switch", pause.reason)
        verbs = self.systemctl_verbs()
        stop = verbs.index(["stop", deploy.SYNC_TIMER])
        self.assertEqual(
            [verb for verb in verbs[stop:] if verb[0] in ("enable", "start", "disable")],
            [["disable", "--now", deploy.SYNC_TIMER]],
        )
        # Disabled, not just stopped: a reboot does not start a sync on this checkout.
        self.assertEqual(self.sync_timer(), ("inactive", False))
        self.assert_checkout_at(self.initial)  # code and binary were restored before the failure
        self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
        state = self.state()
        self.assertEqual(state.failed_shas[v2].reason, "switch")
        self.assertFalse(state.sync_timer_restart_pending)
        self.assertIsNotNone(state.switch_in_progress)  # checked by the tick after `resume`
        del os.environ["FAKE_SYSTEMCTL_FAIL"]
        with self.logs() as lines:  # paused: nothing happens until a human resumes
            self.assertEqual(self.tick(), 0)
        self.assertIn("auto-deploy is paused", text(lines))

        # The checkout is back where the switch started, so after `resume` the tick carries on
        # (and the failed commit stays failed).
        self.assertEqual(self.agent("resume"), 0)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("did not finish cleanly", text(lines))
        self.assertIn(f"{v2[:12]} is not deployed: its switch failed", text(lines))
        self.assertIsNone(self.state().switch_in_progress)
        self.assertEqual(self.sync_timer(), ("inactive", False))  # the human enables it

    def test_a_sync_timer_that_cannot_be_disabled_is_left_to_the_human(self) -> None:
        # Best effort: the message says what to run by hand instead of claiming it was done.
        self.mark_deployed()
        os.environ["FAKE_SYSTEMCTL_FAIL"] = "disable"
        with self.logs("ERROR") as lines:
            clause = deploy._disable_sync_timer()
        self.assertIn(f"cannot disable {deploy.SYNC_TIMER}: Failed to disable", text(lines))
        self.assertEqual(
            clause,
            f"{deploy.SYNC_TIMER} could not be disabled (Failed to disable: fake failure): run "
            f"`systemctl --user disable --now {deploy.SYNC_TIMER}` so that no sync runs on this "
            "checkout",
        )
        self.assertEqual(self.sync_timer(), ("active", True))

    def test_units_that_systemd_would_not_run_are_rolled_back(self) -> None:
        # A deploy unit that does not load would stop auto-deploy for good, so the switch fails.
        self.mark_deployed()
        units = {name: (self.paths.unit_dir / name).read_text() for name in deploy.UNIT_NAMES}
        deploy_template = f"{deploy.UNIT_TEMPLATE_DIR}/{deploy.DEPLOY_SERVICE}"
        sync_template = f"{deploy.UNIT_TEMPLATE_DIR}/{deploy.SYNC_SERVICE}"
        original = {t: (self.dev / t).read_text() for t in (deploy_template, sync_template)}
        cases = (
            (  # an unknown section: systemd ignores it, and the service has no ExecStart=
                deploy_template,
                ("\n[Service]\n", "\n[Services]\n"),
                f"systemd does not load {deploy.DEPLOY_SERVICE} after daemon-reload "
                "(LoadState=bad-setting)",
            ),
            (  # a setting systemd refuses for a oneshot service
                sync_template,
                ("\nTimeoutStartSec=86400\n", "\nTimeoutStartSec=86400\nRestart=always\n"),
                f"systemd does not load {deploy.SYNC_SERVICE} after daemon-reload",
            ),
            (  # loads, but systemd-analyze verify finds the missing command
                sync_template,
                ("\nExecStart=", "\nExecStartPre=/nonexistent/preflight\nExecStart="),
                "Command /nonexistent/preflight is not executable",
            ),
        )
        bad = ""
        for version, (template, (old, new), expected) in enumerate(cases, start=2):
            with self.subTest(expected):
                self.assertEqual(original[template].count(old), 1)
                files = {**original, template: original[template].replace(old, new)}
                bad = self.commit(f"feat: v{version}", {**files, "bin_version.txt": f"{version}\n"})
                self.clear_calls()
                with self.logs() as lines:
                    self.assertEqual(self.tick(), 1)
                self.assertIn(expected, text(errors(lines)))
                self.assertIn(f"rolled back to {self.initial[:12]}", text(lines))
                state = self.state()
                failure = state.failed_shas[bad]
                self.assertEqual(failure.reason, "switch")
                self.assertIn(expected, failure.detail)
                # Undone: the same update that recorded the failure dropped the switch record.
                self.assertIsNone(state.switch_in_progress)
                self.assert_checkout_at(self.initial)
                self.assertEqual(self.binary.read_text(), fake_binary_text("1"))
                for name, content in units.items():
                    self.assertEqual((self.paths.unit_dir / name).read_text(), content, name)
                verbs = self.systemctl_verbs()
                self.assertEqual(verbs.count(["daemon-reload"]), 2)  # the switch and the undo
                self.assertEqual(verbs[-1], ["enable", "--now", deploy.SYNC_TIMER])

        # Without systemd-analyze only LoadState is checked, and the last commit loads.
        with mock.patch.object(deploy, "SYSTEMD_ANALYZE", "systemd-analyze-not-installed"):
            self.assertEqual(self.agent("retry"), 0)
            self.clear_calls()
            self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(bad)
        self.assertEqual(self.calls("systemd-analyze"), [])

    def test_a_sync_timer_that_did_not_restart_is_retried_every_tick(self) -> None:
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        os.environ["FAKE_SYSTEMCTL_FAIL"] = "enable"
        with self.logs() as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn(f"deployed {v2[:12]}, but the sync did not restart", text(errors(lines)))
        self.assert_checkout_at(v2)
        state = self.state()
        self.assertEqual(state.deployed_sha, v2)
        self.assertTrue(state.sync_timer_restart_pending)
        self.assertEqual(self.sync_timer(), ("inactive", True))

        # Nothing to deploy, but the tick retries the timer, fails again and keeps the error.
        self.clear_calls()
        with self.logs() as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn("is still not running after the last switch", text(errors(lines)))
        self.assertEqual(self.systemctl_verbs(), [["enable", "--now", deploy.SYNC_TIMER]])
        self.assertIn("is still not running", self.state().last_error or "")
        self.assertIn("Sync timer   NOT restarted after the last switch", self.status_text())

        # A paused tick leaves systemd alone (an operator may have stopped the sync on purpose).
        self.assertEqual(self.agent("pause"), 0)
        self.clear_calls()
        self.assertEqual(self.tick(), 0)
        self.assertEqual(self.call_log(), [])
        self.assertEqual(self.agent("resume"), 0)

        # Once systemd accepts it, the timer runs again and the error is cleared.
        del os.environ["FAKE_SYSTEMCTL_FAIL"]
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(f"enabled and started {deploy.SYNC_TIMER}", text(lines))
        self.assertEqual(self.sync_timer(), ("active", True))
        state = self.state()
        self.assertFalse(state.sync_timer_restart_pending)
        self.assertIsNone(state.last_error)
        self.assertNotIn("Sync timer", self.status_text())
        self.clear_calls()
        with self.assertNoLogs(deploy.log, level="INFO"):  # a quiet no-op again
            self.assertEqual(self.tick(), 0)
        self.assertEqual(self.call_log(), [])

    def test_a_sync_timer_that_could_not_be_restored_is_retried_too(self) -> None:
        self.mark_deployed()
        bad = self.commit("feat: breaks the generator", {"bin_version.txt": "broken\n"})
        os.environ["FAKE_SYSTEMCTL_FAIL"] = "enable"
        with self.logs() as lines:
            self.assertEqual(self.tick(), 1)
        self.assertIn(f"could not restart {deploy.SYNC_TIMER}", text(errors(lines)))
        state = self.state()
        self.assertEqual(state.failed_shas[bad].reason, "switch")
        self.assertTrue(state.sync_timer_restart_pending)

        del os.environ["FAKE_SYSTEMCTL_FAIL"]
        self.clear_calls()
        with self.logs() as lines:  # the failure is final, but the timer is restarted
            self.assertEqual(self.tick(), 0)
        self.assertIn("its switch failed", text(lines))
        self.assertEqual(self.systemctl_verbs(), [["enable", "--now", deploy.SYNC_TIMER]])
        self.assertEqual(self.sync_timer(), ("active", True))
        self.assertFalse(self.state().sync_timer_restart_pending)
        self.assertIsNone(self.state().last_error)

    def test_a_switch_that_died_half_way_pauses_for_a_human(self) -> None:
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        v3 = self.commit("feat: v3", {"bin_version.txt": "3\n"})
        # The agent dies after the fast-forward, before it installs the binary.
        with (
            mock.patch.object(deploy, "install_binary", side_effect=_Crash),
            self.assertRaises(_Crash),
        ):
            self.tick()
        self.assertEqual(self.head(), v3)
        self.assertEqual(self.state().deployed_sha, v2)
        self.assertEqual(self.binary.read_text(), fake_binary_text("2"))
        self.assertIn(f"Switch       from {v2[:12]} to {v3[:12]} since ", self.status_text())
        # Say an earlier switch also left a sync timer restart pending.
        with deploy.state_transaction(self.paths) as state:
            state.sync_timer_restart_pending = True

        # HEAD is no undo point: the next tick switches nothing and hands over to a human.
        self.clear_calls()
        with self.logs() as lines:
            self.assertEqual(self.tick(), 1)
        message = text(errors(lines))
        self.assertIn(f"a switch from {v2[:12]} to {v3[:12]}", message)
        self.assertIn(f"left the checkout at {v3[:12]}, not at {v2[:12]}", message)
        self.assertIn(f"{deploy.SYNC_TIMER} is stopped and disabled", message)
        self.assertEqual((self.head(), self.calls("cargo")), (v3, []))
        self.assertEqual(self.sync_timer(), ("inactive", False))
        # No tick may enable the timer on this checkout again: the human does, after the repair.
        self.assertFalse(self.state().sync_timer_restart_pending)
        pause = deploy.read_pause(self.paths)
        assert pause is not None
        self.assertIn("was interrupted", pause.reason)
        self.assertIn("was interrupted", self.state().last_error or "")
        with self.logs("ERROR") as lines:  # rollback does not use HEAD as the undo point either
            self.assertEqual(self.agent("rollback"), 1)
        self.assertIn(f"not rolling back: a switch from {v2[:12]} to {v3[:12]}", text(lines))
        self.assertEqual(self.head(), v3)

        # The documented repair: the checkout back where the switch started, then `resume`.
        git(self.repo, "reset", "--quiet", "--hard", v2)
        self.assertEqual(self.agent("resume"), 0)
        self.clear_calls()
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("did not finish cleanly", text(lines))
        self.assert_checkout_at(v3)
        self.assertEqual(self.binary.read_text(), fake_binary_text("3"))
        state = self.state()
        self.assertEqual((state.deployed_sha, state.previous_sha), (v3, v2))
        self.assertIsNone(state.switch_in_progress)
        self.assertEqual(self.calls("cargo"), [])  # the tested release is reused
        self.assertEqual(self.sync_timer(), ("active", True))

    def test_a_switch_that_died_before_the_checkout_moved_is_forgotten(self) -> None:
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        with (
            mock.patch.object(deploy, "untracked_collisions", side_effect=_Crash),
            self.assertRaises(_Crash),
        ):
            self.tick()
        self.assertEqual(self.head(), self.initial)
        self.assertIsNotNone(self.state().switch_in_progress)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("did not finish cleanly", text(lines))
        self.assertEqual(errors(lines), [])
        self.assert_checkout_at(v2)
        self.assertIsNone(self.state().switch_in_progress)

    def test_a_failed_first_deploy_never_enables_the_sync_timer(self) -> None:
        self.commit("feat: broken", {"bin_version.txt": "broken\n"})
        with self.logs("ERROR"):
            self.assertEqual(self.tick(), 1)
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_text(), LEGACY_BINARY)
        verbs = self.systemctl_verbs()
        self.assertFalse([verb for verb in verbs if verb[0] in ("enable", "start")])
        self.assertIsNone(self.state().deployed_sha)


class OperatorTests(DeployTestCase):
    """pause, resume, rollback and status."""

    def test_pause_and_resume(self) -> None:
        self.mark_deployed()
        new = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.agent("pause", "--reason", "investigating a render"), 0)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn("auto-deploy is paused since ", text(lines))
        self.assertIn("investigating a render", text(lines))
        self.assertEqual(self.github.requests, [])
        self.assertEqual(self.call_log(), [])
        self.assertEqual(self.head(), self.initial)
        with self.logs() as lines:
            self.assertEqual(self.agent("resume"), 0)
        self.assertIn("auto-deploy resumed (it was paused: investigating a render)", text(lines))
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(new)

    def test_rollback_switches_back_pauses_and_blocks_the_bad_commit(self) -> None:
        self.mark_deployed()
        tested, mtime = self.binary.read_bytes(), self.binary.stat().st_mtime_ns
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.clear_calls()
        with self.logs() as lines:
            self.assertEqual(self.agent("rollback"), 0)
        self.assertIn(
            f"rolled back from {v2[:12]} to {self.initial[:12]} (feat: initial)", text(lines)
        )
        self.assert_checkout_at(self.initial)
        self.assertEqual(self.binary.read_bytes(), tested)
        self.assertEqual(self.binary.stat().st_mtime_ns, mtime)  # the generator identity is back
        self.assertEqual(
            self.systemctl_verbs()[-2:],
            [["enable", "--now", deploy.SYNC_TIMER], ["start", "--no-block", deploy.SYNC_SERVICE]],
        )
        state = self.state()
        self.assertEqual(
            (state.deployed_sha, state.previous_sha, state.rolled_back_from),
            (self.initial, None, v2),
        )
        self.assertEqual(state.failed_shas[v2].reason, "rollback")
        self.assertIsNone(state.switch_in_progress)
        self.assertIsNotNone(deploy.read_pause(self.paths))
        report = self.status_text()
        self.assertIn(f"Pending      {v2[:12]} NOT deployed: rolled back by the operator\n", report)
        self.assertNotIn("rollback failed", report)

        # Resuming does not redeploy the rolled-back commit ...
        self.assertEqual(self.agent("resume"), 0)
        with self.logs() as lines:
            self.assertEqual(self.tick(), 0)
        self.assertIn(f"{v2[:12]} is not deployed: it was rolled back", text(lines))
        self.assertEqual(self.head(), self.initial)
        # ... but a fix on main is deployed.
        v3 = self.commit("fix: v3", {"bin_version.txt": "3\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(v3)
        self.assertEqual(self.state().previous_sha, self.initial)

    def test_rollback_needs_a_previous_deployment(self) -> None:
        self.mark_deployed()
        with self.logs("ERROR") as lines:
            self.assertEqual(self.agent("rollback"), 1)
        self.assertIn("no previous deployment is recorded", text(lines))
        self.assertIsNotNone(deploy.read_pause(self.paths))  # deploys stay stopped
        self.assertEqual(self.call_log(), [])

    def test_rollback_refuses_a_commit_that_predates_auto_deploy(self) -> None:
        # The first deploy's previous commit is the legacy checkout, without the agent.
        git(self.dev, "rm", "--quiet", "-r", "ops")
        git(self.dev, "commit", "--quiet", "-m", "chore: the legacy layout")
        git(self.dev, "push", "--quiet", "--force", "origin", "HEAD:main")
        git(self.repo, "fetch", "--quiet", "origin")
        git(self.repo, "reset", "--quiet", "--hard", "origin/main")
        current = self.commit("feat: auto-deploy", {**initial_files(), "bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.assertIsNone(self.state().previous_sha)
        with self.logs("ERROR") as lines:
            self.assertEqual(self.agent("rollback"), 1)
        self.assertIn("no previous deployment is recorded", text(lines))
        self.assert_checkout_at(current)

    def test_the_first_deploy_records_no_rollback_target(self) -> None:
        # Before the first deploy the checkout was switched by hand (here to a commit that
        # already has the agent, as the documented fast-forward does), and origin/main moved on.
        # That commit (its CI may even have failed) and the hand-built binary were never gated
        # by the agent, so rollback must not switch back to them.
        current = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.assert_checkout_at(current)
        self.assertIsNone(self.state().previous_sha)
        with self.logs("ERROR") as lines:
            self.assertEqual(self.agent("rollback"), 1)
        self.assertIn("no previous deployment is recorded", text(lines))
        self.assert_checkout_at(current)
        self.assertEqual(self.binary.read_text(), fake_binary_text("2"))

    def test_rollback_leaves_only_the_commit_it_saw_deployed(self) -> None:
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        v3 = self.commit("feat: v3", {"bin_version.txt": "3\n"})
        # A tick that passed its last pause check before `rollback` paused holds the deploy lock,
        # and deploys v3 while the rollback waits for the lock.
        holder = HeldLock(self.paths.deploy_lock)
        self.addCleanup(holder.release)
        sleep = deploy.interruptible_sleep
        ticked: list[int] = []

        def the_tick_finishes(seconds: float, **kwargs: deploy.Paths | None) -> None:
            if not ticked:
                pause = self.paths.pause_file.read_bytes()
                self.paths.pause_file.unlink()
                holder.release()
                ticked.append(deploy.cmd_run(self.paths))
                self.paths.pause_file.write_bytes(pause)
            sleep(seconds, **kwargs)

        with (
            mock.patch.object(deploy, "interruptible_sleep", side_effect=the_tick_finishes),
            self.logs() as lines,
        ):
            self.assertEqual(self.agent("rollback"), 1)
        self.assertEqual(ticked, [0])
        self.assertIn(
            f"the deployed commit changed from {v2[:12]} to {v3[:12]}", text(errors(lines))
        )
        self.assert_checkout_at(v3)
        state = self.state()
        self.assertEqual((state.deployed_sha, state.previous_sha), (v3, v2))
        self.assertNotIn(v3, state.failed_shas)
        self.assertIsNotNone(deploy.read_pause(self.paths))  # deploys stay stopped

    def test_a_hangup_while_rollback_waits_restores_the_timer_and_changes_nothing(self) -> None:
        # An SSH session that drops sends the waiting rollback SIGHUP. By default that kills the
        # process on the spot, leaving the sync timer stopped.
        self.mark_deployed()
        v2 = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 0)
        self.set_systemctl(script={deploy.SYNC_SERVICE: ["active"] * 100})
        previous = signal.signal(signal.SIGHUP, _ignore_signal)
        self.addCleanup(signal.signal, signal.SIGHUP, previous)
        sleep = deploy.interruptible_sleep

        def hang_up(seconds: float, **kwargs: deploy.Paths | None) -> None:
            os.kill(os.getpid(), signal.SIGHUP)
            sleep(seconds, **kwargs)

        self.clear_calls()
        with (
            mock.patch.object(deploy, "interruptible_sleep", side_effect=hang_up),
            self.logs("WARNING") as lines,
        ):
            self.assertEqual(self.agent("rollback"), 1)
        self.assertIn("rollback stopped by a signal before switching; nothing changed", text(lines))
        self.assert_checkout_at(v2)
        self.assertEqual(self.state().deployed_sha, v2)
        self.assertEqual(self.systemctl_verbs()[-1], ["enable", "--now", deploy.SYNC_TIMER])
        self.assertEqual(self.sync_timer(), ("active", True))
        self.assertIsNotNone(deploy.read_pause(self.paths))
        self.assertIs(signal.getsignal(signal.SIGHUP), _ignore_signal)  # the agent's is gone

    def test_status(self) -> None:
        self.mark_deployed()
        os.environ["FAKE_CARGO_FAIL"] = "build"
        bad = self.commit("feat: v2", {"bin_version.txt": "2\n"})
        self.assertEqual(self.tick(), 1)
        self.assertEqual(self.agent("pause", "--reason", "looking into it"), 0)

        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(self.agent("status"), 0)
        report = out.getvalue()
        self.assertIn(f"Deployed     {self.initial[:12]} feat: initial", report)
        self.assertIn("Binary       matches the deployed release", report)
        self.assertIn(f"origin/main  {bad[:12]} feat: v2", report)
        self.assertIn(f"Pending      {bad[:12]} NOT deployed: build failed", report)
        self.assertIn("Auto-deploy  PAUSED since ", report)
        self.assertIn(f"Failed       {bad[:12]} build: ", report)
        self.assertIn("cosmicsig-sync.timer      active (running), enabled", report)

        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(self.agent("status", "--json"), 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["deployed"]["sha"], self.initial)
        self.assertEqual(data["origin_main"], {"sha": bad, "subject": "feat: v2"})
        self.assertEqual(data["pending"]["failure"]["reason"], "build")
        self.assertEqual(data["paused"]["reason"], "looking into it")
        self.assertTrue(data["binary"]["matches_deployed"])
        self.assertEqual(data["units"][deploy.SYNC_TIMER]["UnitFileState"], "enabled")
        self.assertEqual(data["legacy_units"], [])

        self.binary.unlink()
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(self.agent("status"), 0)
        self.assertIn("Binary       MISSING", out.getvalue())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A state.json as the agent wrote it before it numbered failures (Failure.seq): written by that
# version's save_failure, recording f, c, e and d in this order (c and e in the same second).
PREVIOUS_STATE_JSON = """{
  "binary_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "deployed_at": "2026-09-01T12:00:00+00:00",
  "deployed_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "deployed_subject": "feat: v1",
  "failed_shas": {
    "cccccccccccccccccccccccccccccccccccccccc": {
      "at": "2026-09-01T12:02:00+00:00",
      "checked_at": "2026-09-01T12:02:00+00:00",
      "detail": "'CI passed' concluded failure",
      "reason": "ci"
    },
    "dddddddddddddddddddddddddddddddddddddddd": {
      "at": "2026-09-01T12:03:00+00:00",
      "detail": "smoke test failed",
      "reason": "switch"
    },
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee": {
      "at": "2026-09-01T12:02:00+00:00",
      "detail": "cargo test failed",
      "reason": "tests"
    },
    "ffffffffffffffffffffffffffffffffffffffff": {
      "at": "2026-09-01T12:01:00+00:00",
      "detail": "cargo build failed",
      "reason": "build"
    }
  },
  "last_error": null,
  "last_error_at": null,
  "previous_binary_sha256": null,
  "previous_sha": "9999999999999999999999999999999999999999",
  "rolled_back_from": null,
  "schema_version": 1,
  "switch_in_progress": null,
  "sync_timer_restart_pending": false
}
"""


class HelperTests(unittest.TestCase):
    """Small pure helpers."""

    def temp_paths(self) -> deploy.Paths:
        """Paths under a fresh temporary directory (for the state.json helpers)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        return deploy.Paths(
            repo=root / "repo",
            state_dir=root / "state",
            data_dir=root / "data",
            unit_dir=root / "units",
        )

    def test_journal_lines_carry_their_priority(self) -> None:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "bad\nworse", None, None)
        self.assertEqual(deploy._Formatter(journal=True).format(record), "<3>ERROR bad\n<3>worse")
        self.assertEqual(deploy._Formatter(journal=False).format(record), "ERROR bad\nworse")
        with tempfile.TemporaryFile("w+") as stream:
            info = os.fstat(stream.fileno())
            with mock.patch.dict(os.environ, {"JOURNAL_STREAM": f"{info.st_dev}:{info.st_ino}"}):
                self.assertTrue(deploy.stream_is_journal(stream))
            with mock.patch.dict(os.environ, {"JOURNAL_STREAM": "1:2"}):
                self.assertFalse(deploy.stream_is_journal(stream))

    def test_redact_and_durations(self) -> None:
        self.assertEqual(
            deploy.redact("fatal: https://user:hunter2@github.com/o/r.git failed"),
            "fatal: https://***@github.com/o/r.git failed",
        )
        self.assertEqual(deploy.fmt_duration(42), "42s")
        self.assertEqual(deploy.fmt_duration(187), "3m07s")
        self.assertEqual(deploy.fmt_duration(3723), "1h02m03s")

    def test_state_round_trip_and_damaged_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = deploy.State(
                deployed_sha="a" * 40,
                binary_sha256="b" * 64,
                sync_timer_restart_pending=True,
                switch_in_progress=deploy.SwitchRecord(
                    "d" * 40, "a" * 40, "b" * 64, "fast-forward", "2026-01-01T00:00:00+00:00"
                ),
            )
            state.record_failure(
                "c" * 40,
                deploy.Failure(
                    "ci", "red", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"
                ),
            )
            deploy.atomic_write_json(path, state.to_json())
            self.assertEqual(deploy.load_state(path), state)
            for damaged in ("{not json", "[]"):
                path.write_text(damaged, encoding="utf-8")
                with self.assertLogs(deploy.log, level="WARNING"):
                    self.assertEqual(deploy.load_state(path), deploy.State())

    def test_only_the_newest_github_actions_ci_passed_run_of_the_commit_counts(self) -> None:
        sha, other = "a" * 40, "b" * 40
        verdict = deploy.CiVerdict
        cases: list[tuple[str, object, deploy.CiVerdict]] = [
            ("another app", {"check_runs": [check_run(sha, app="imposter")]}, verdict.PENDING),
            ("another check", {"check_runs": [check_run(sha, name="Tests")]}, verdict.PENDING),
            ("another commit", {"check_runs": [check_run(other)]}, verdict.PENDING),
            (
                "an imposter's success does not hide the failure",
                {
                    "check_runs": [
                        check_run(sha, app="imposter", run_id=9),
                        check_run(sha, conclusion="failure", run_id=2),
                    ]
                },
                verdict.FAILED,
            ),
            (
                "the newest run wins",
                {
                    "check_runs": [
                        check_run(sha, run_id=3),
                        check_run(sha, conclusion="failure", run_id=2),
                    ]
                },
                verdict.PASSED,
            ),
            ("cancelled", {"check_runs": [check_run(sha, conclusion="cancelled")]}, verdict.FAILED),
            ("skipped", {"check_runs": [check_run(sha, conclusion="skipped")]}, verdict.FAILED),
            ("malformed", {"message": "Not Found"}, verdict.UNAVAILABLE),
            ("not an object", ["check_runs"], verdict.UNAVAILABLE),
        ]
        for label, body, expected in cases:
            with self.subTest(label):
                self.assertIs(deploy.ci_status_from_check_runs(body, sha).verdict, expected)

    def test_the_fakes_agree_with_the_agent(self) -> None:
        self.assertEqual(GENERATOR_NAME, deploy.BINARY_NAME)

    def test_a_stop_signal_ends_a_wait_promptly(self) -> None:
        # The handler only stores a flag (no lock it could deadlock on), and a wait notices it
        # within a poll interval. A late signal meets this test's no-op handler. SIGHUP is a
        # stop request too: an operator's SSH session that drops during `rollback` sends it.
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=signal.Signals(signum).name):
                previous = signal.signal(signum, _ignore_signal)
                self.addCleanup(signal.signal, signum, previous)
                timer = threading.Timer(0.2, os.kill, args=(os.getpid(), signum))
                with deploy.stop_signals():
                    started = time.monotonic()
                    timer.start()
                    try:
                        with self.assertRaises(deploy.Interrupted):
                            deploy.interruptible_sleep(10.0)
                    finally:
                        timer.cancel()
                    self.assertLess(time.monotonic() - started, 5.0)
                    self.assertTrue(deploy._stop.is_set())
                self.assertIs(signal.getsignal(signum), _ignore_signal)  # restored
                self.assertFalse(deploy._stop.is_set())  # cleared for the next command
                self.assertFalse(deploy._stop.wait(0.05))

    def test_a_signal_the_caller_ignores_stays_ignored(self) -> None:
        # `nohup cosmicsig_deploy.py rollback` starts the agent with SIGHUP ignored: the hangup of
        # a dropped SSH session must not stop the rollback the operator protected from it.
        for signum, handler in ((signal.SIGHUP, signal.SIG_IGN), (signal.SIGTERM, signal.SIG_DFL)):
            previous = signal.signal(signum, handler)
            self.addCleanup(signal.signal, signum, previous)
        with deploy.stop_signals():
            self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
            self.assertTrue(callable(signal.getsignal(signal.SIGTERM)))  # the others still stop
            os.kill(os.getpid(), signal.SIGHUP)
            self.assertFalse(deploy._stop.wait(0.2))
        self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_git_command_labels_name_the_subcommand(self) -> None:
        subcommand = deploy._git_subcommand
        self.assertEqual(
            subcommand(["-c", "advice.detachedHead=false", "checkout", "x"]), "checkout"
        )
        self.assertEqual(subcommand(["--no-pager", "log", "-1"]), "log")
        self.assertEqual(subcommand(["fetch", "--quiet"]), "fetch")
        self.assertEqual(subcommand([]), "")

    def test_failed_shas_are_capped(self) -> None:
        state = deploy.State()
        for index in range(deploy.MAX_FAILED_SHAS + 5):
            state.record_failure(f"{index:040x}", deploy.Failure("build", "", ""))
        self.assertEqual(len(state.failed_shas), deploy.MAX_FAILED_SHAS)
        self.assertNotIn(f"{0:040x}", state.failed_shas)
        self.assertIn(f"{deploy.MAX_FAILED_SHAS + 4:040x}", state.failed_shas)

    def test_the_cap_forgets_the_oldest_recorded_failure_across_saves(self) -> None:
        # state.json's keys are sorted, so a reload lists the failures alphabetically. Recorded
        # in reverse alphabetical order within one second (so `at` cannot tell them apart), the
        # oldest records are the alphabetically last: the cap must forget those, not the first.
        paths = self.temp_paths()
        extra = 3
        shas = [f"{n:040x}" for n in reversed(range(deploy.MAX_FAILED_SHAS + extra))]
        now = datetime.datetime(2026, 9, 1, 12, 0, tzinfo=datetime.timezone.utc)
        with mock.patch.object(deploy, "utcnow", return_value=now):
            for index, sha in enumerate(shas):
                recorded = deploy.save_failure(paths, sha, deploy.REASON_BUILD, f"failure {index}")
                self.assertEqual(recorded, deploy.load_state(paths.state_file).failed_shas[sha])
        on_disk = json.loads(paths.state_file.read_text(encoding="utf-8"))["failed_shas"]
        self.assertEqual(list(on_disk), sorted(shas[extra:]))  # the file's order is lost...
        state = deploy.load_state(paths.state_file)
        self.assertEqual(list(state.failed_shas), shas[extra:])  # ...but not the recording order
        self.assertEqual(state.failed_shas[shas[-1]].detail, f"failure {len(shas) - 1}")

        # Recording a commit again (as a CI re-check does) makes it the newest record.
        again, next_oldest = shas[extra], shas[extra + 1]
        deploy.save_failure(paths, again, deploy.REASON_CI, "still red")
        deploy.save_failure(paths, "f" * 40, deploy.REASON_BUILD, "newest")
        state = deploy.load_state(paths.state_file)
        self.assertNotIn(next_oldest, state.failed_shas)
        self.assertEqual(list(state.failed_shas)[-2:], [again, "f" * 40])

    def test_a_state_json_of_the_previous_version_still_loads(self) -> None:
        paths = self.temp_paths()
        paths.state_dir.mkdir(parents=True)
        paths.state_file.write_text(PREVIOUS_STATE_JSON, encoding="utf-8")
        state = deploy.load_state(paths.state_file)
        self.assertEqual((state.deployed_sha, state.previous_sha), ("a" * 40, "9" * 40))
        self.assertEqual(state.binary_sha256, "b" * 64)
        self.assertEqual(
            state.failed_shas["c" * 40],
            deploy.Failure(
                deploy.REASON_CI,
                "'CI passed' concluded failure",
                "2026-09-01T12:02:00+00:00",
                "2026-09-01T12:02:00+00:00",
            ),
        )
        # Unnumbered records are ordered by `at`, then (within one second) by commit id.
        self.assertEqual(list(state.failed_shas), ["f" * 40, "c" * 40, "e" * 40, "d" * 40])
        self.assertEqual({failure.seq for failure in state.failed_shas.values()}, {0})

        # New records are newer than all of them, and the cap forgets the oldest `at` first. (The
        # previous version forgot the alphabetically first commit: here, a new record.)
        for index in range(deploy.MAX_FAILED_SHAS - 2):
            deploy.save_failure(paths, f"{index:040x}", deploy.REASON_BUILD, "new")
        state = deploy.load_state(paths.state_file)
        self.assertEqual(len(state.failed_shas), deploy.MAX_FAILED_SHAS)
        self.assertEqual(list(state.failed_shas)[:3], ["e" * 40, "d" * 40, f"{0:040x}"])

        # The file keeps the layout the previous version reads ({sha: {reason, detail, at}},
        # where it ignores seq), so the agent of an older commit keeps the records.
        record = json.loads(paths.state_file.read_text(encoding="utf-8"))["failed_shas"]["d" * 40]
        self.assertEqual(
            record,
            {
                "reason": deploy.REASON_SWITCH,
                "detail": "smoke test failed",
                "at": "2026-09-01T12:03:00+00:00",
                "seq": 0,
            },
        )

    def test_a_malformed_failure_number_reads_as_unnumbered(self) -> None:
        for seq in (True, -1, 0, "3", 2.0, None):
            with self.subTest(seq=seq):
                failure = deploy.Failure.from_json({"reason": "build", "seq": seq})
                self.assertEqual(failure, deploy.Failure("build", "", ""))

    def test_a_ci_failure_checked_in_the_future_is_due_again(self) -> None:
        # After the clock steps back, waiting for it to reach checked_at would stall the
        # re-checks for as long as it stepped back.
        now = datetime.datetime(2026, 9, 1, 12, 0, tzinfo=datetime.timezone.utc)
        cases: list[tuple[int | None, bool]] = [
            (None, True),
            (-15, True),
            (-14, False),
            (0, False),
            (1, True),
            (24 * 60, True),
        ]
        for minutes, due in cases:
            checked = (
                None
                if minutes is None
                else deploy.isoformat(now + datetime.timedelta(minutes=minutes))
            )
            with self.subTest(minutes=minutes):
                failure = deploy.Failure(deploy.REASON_CI, "", "", checked)
                self.assertIs(deploy.ci_recheck_due(failure, now), due)


# ---------------------------------------------------------------------------
# ops/server/bootstrap-root.sh
# ---------------------------------------------------------------------------


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class BootstrapTests(unittest.TestCase):
    """The root bootstrap, with fake id, systemctl (system scope) and loginctl."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        install_fake(self.bin, "systemctl", "root-systemctl")
        install_fake(self.bin, "loginctl")
        install_fake(self.bin, "id")
        self.unit_dir = self.root / "etc-systemd-system"
        self.unit_dir.mkdir()
        for name in (deploy.SYNC_SERVICE, deploy.SYNC_TIMER):
            (self.unit_dir / name).write_text("[Unit]\n", encoding="utf-8")
        self.calls_file = self.root / "calls.jsonl"
        self.state_file = self.root / "root-systemctl.json"
        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        for key in list(os.environ):
            if key.startswith(("FAKE_", "COSMICSIG_")) or key == "SUDO_USER":
                del os.environ[key]
        os.environ.update(
            {
                "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "FAKE_LOG": str(self.calls_file),
                "FAKE_ROOT_STATE": str(self.state_file),
                "COSMICSIG_BOOTSTRAP_UNIT_DIR": str(self.unit_dir),
                "COSMICSIG_BOOTSTRAP_POLL_SECONDS": "0.05",
                "SUDO_USER": "builder",
            }
        )
        for name in ("systemctl", "loginctl", "id"):
            self.assertEqual(shutil.which(name), str(self.bin / name))

    def bootstrap(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the script."""
        return subprocess.run(
            ["bash", str(BOOTSTRAP), *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def calls(self) -> list[list[object]]:
        """The fakes' calls, in order."""
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def test_it_refuses_to_run_without_root(self) -> None:
        os.environ["FAKE_UID"] = "1000"
        result = self.bootstrap()
        self.assertEqual(result.returncode, 1)
        self.assertIn("run this with sudo", result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertTrue((self.unit_dir / deploy.SYNC_TIMER).exists())

    def test_it_refuses_an_unknown_or_root_service_user(self) -> None:
        for args, expected in (
            (["nosuchuser"], "no such user: nosuchuser"),
            (["root"], "must not be root"),
        ):
            with self.subTest(args=args):
                result = self.bootstrap(*args)
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
        del os.environ["SUDO_USER"]
        result = self.bootstrap()
        self.assertEqual(result.returncode, 1)
        self.assertIn("name the service user", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_it_retires_the_legacy_units_after_the_running_sync_and_enables_lingering(self) -> None:
        self.state_file.write_text(
            json.dumps(
                {
                    "service_states": ["active", "activating", "deactivating", "refreshing"],
                    "working_directory": "/home/builder/checkout",
                }
            ),
            encoding="utf-8",
        )
        result = self.bootstrap()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [call[1:] for call in self.calls()]
        self.assertEqual(
            calls,
            [
                ["show", "--property=LoadState", "--value", deploy.SYNC_TIMER],
                ["disable", "--now", deploy.SYNC_TIMER],
                *[["is-active", deploy.SYNC_SERVICE]] * 5,  # four busy states, then idle
                ["daemon-reload"],
                ["reset-failed", deploy.SYNC_SERVICE, deploy.SYNC_TIMER],
                ["enable-linger", "builder"],
            ],
        )
        self.assertIn("waiting for the running legacy sync to finish", result.stdout)
        self.assertEqual(list(self.unit_dir.iterdir()), [])
        # The script lives in a checkout, so it names that checkout in the next steps.
        self.assertIn(f"  cd {REPO_ROOT}\n", result.stdout)
        self.assertIn("python3 ops/deploy/cosmicsig_deploy.py install", result.stdout)
        self.assertIn("git fetch origin && git merge --ff-only origin/main", result.stdout)

        # Idempotent: nothing left to stop, wait for or remove.
        self.calls_file.unlink()
        result = self.bootstrap("builder")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call[1:] for call in self.calls()],
            [
                ["show", "--property=LoadState", "--value", deploy.SYNC_TIMER],
                ["is-active", deploy.SYNC_SERVICE],
                ["enable-linger", "builder"],
            ],
        )
        self.assertIn("no legacy unit files", result.stdout)

    def test_help(self) -> None:
        result = subprocess.run(
            ["bash", str(BOOTSTRAP), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith("Usage: sudo ops/server/bootstrap-root.sh"))


if __name__ == "__main__":
    unittest.main()
