"""Tests for the tooling configuration and the developer tooling around it.

Settings kept in two places must stay equal. .pre-commit-config.yaml runs ruff and mypy from
their pre-commit mirrors, while pyproject.toml pins the same tools in its `dev` extra for
editors, `just py-check` and `pip install -e ".[dev]"`. Two ruff versions disagree about
formatting and two mypy versions about types, so the hooks would fight the editor. Dependabot's
pip and pre-commit ecosystems share one group (.github/dependabot.yml) and normally move both
sides in one pull request, but a mirror can tag a release later than PyPI publishes it, and a
hand edit can move one side alone: these tests fail CI until both sides agree. The same goes for
the shellcheck release actionlint uses and the one the shellcheck hook runs, for the pre-commit
version the config requires, and for the commit types the commit-msg hook and CI's pull request
title check accept (a squash merge turns the title into the commit subject on main). Another
test holds every hook to a frozen commit SHA (a tag can be moved upstream; a SHA cannot).

The rest run the tooling itself: CI's pull request title check (its bash script, with a fake
`gh`), the macOS FFmpeg cache steps of CI (their scripts, with the steps' env and a fake
Homebrew), `just setup` (the recipe's script, with a fake Python and pre-commit, in a real git
repository and in a linked worktree of it), and run_parallel.py, the runner of the pre-push
hook and `just py-test`; and they hold the zizmor hooks to their split (offline at commit time,
online only in CI's manual-stage run).

Standard library only. The standard library has no YAML parser, so the YAML files are read with
line-oriented regexes, strict about the layout `pre-commit autoupdate --freeze` writes (and the
one the files have today), which fail loudly on anything else rather than guess. pyproject.toml
is read with tomllib on Python 3.11+ and, on 3.10 (no tomllib), with a regex for the `dev`
array; on 3.11+ a test checks the regex reader against tomllib, so the 3.10 path is exercised on
every interpreter CI runs.

Run from the repository root:

    python -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path

import run_parallel

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
JUSTFILE = REPO_ROOT / "justfile"

RUFF_HOOKS = "https://github.com/astral-sh/ruff-pre-commit"
MYPY_HOOKS = "https://github.com/pre-commit/mirrors-mypy"
SHELLCHECK_HOOKS = "https://github.com/shellcheck-py/shellcheck-py"
ZIZMOR_HOOKS = "https://github.com/zizmorcore/zizmor-pre-commit"

BASH = shutil.which("bash")
GIT = shutil.which("git")

# ---------------------------------------------------------------------------
# .pre-commit-config.yaml
# ---------------------------------------------------------------------------

# `  - repo: <url>`: the start of one entry of the top-level `repos:` list.
REPO_LINE = re.compile(r"^\s*-\s+repo:\s*(?P<repo>\S+)\s*$")
# `    rev: <rev>` with an optional `# frozen: <tag>` comment (autoupdate --freeze writes it).
REV_LINE = re.compile(r"^\s*rev:\s*(?P<rev>[^\s#]+)\s*(?:#\s*frozen:\s*(?P<frozen>\S+)\s*)?$")
# A pip requirement pinned exactly inside an `additional_dependencies` list.
SHELLCHECK_PY_PIN = re.compile(r"\bshellcheck-py==(?P<version>[0-9][0-9A-Za-z.]*)")
MINIMUM_PRE_COMMIT = re.compile(
    r"^minimum_pre_commit_version:\s*[\"']?(?P<version>[0-9][0-9.]*)[\"']?\s*$", re.MULTILINE
)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
# `repo: local` and `repo: meta` have no revision: pre-commit runs them from this repository.
REVLESS_REPOS = frozenset({"local", "meta"})


@dataclass(frozen=True)
class HookRepo:
    """One remote entry of `repos:`: its URL, the `rev:` value and the `# frozen:` tag, if any."""

    url: str
    rev: str
    frozen: str | None

    @property
    def release(self) -> str:
        """The release the rev stands for: the frozen tag, else the rev itself (a tag)."""
        return self.frozen if self.frozen is not None else self.rev


def parse_hook_repos(text: str) -> dict[str, HookRepo]:
    """Map every remote `repo:` URL in a pre-commit config to its revision.

    Raises ValueError when the layout is not the one pre-commit writes: a remote repo without a
    `rev:` line before the next repo, a `rev:` outside a repo, or a repo listed twice.
    """
    repos: dict[str, HookRepo] = {}
    pending: str | None = None  # a remote repo whose rev: has not been seen yet
    for number, line in enumerate(text.splitlines(), start=1):
        if repo_match := REPO_LINE.match(line):
            if pending is not None:
                raise ValueError(f"repo {pending} has no rev: line (line {number})")
            url = repo_match["repo"]
            if url in repos:
                raise ValueError(f"repo {url} is listed twice (line {number})")
            pending = None if url in REVLESS_REPOS else url
        elif rev_match := REV_LINE.match(line):
            if pending is None:
                raise ValueError(f"rev: without a remote repo (line {number})")
            repos[pending] = HookRepo(pending, rev_match["rev"], rev_match["frozen"])
            pending = None
        elif line.lstrip().startswith("rev:"):
            raise ValueError(f"unparsable rev: line {number}: {line.strip()!r}")
    if pending is not None:
        raise ValueError(f"repo {pending} has no rev: line")
    return repos


def pre_commit_config_text() -> str:
    """The repository's .pre-commit-config.yaml."""
    return PRE_COMMIT_CONFIG.read_text(encoding="utf-8")


# `      - id: conventional-pre-commit`, `        args:` and one `          - item` of a block list.
CONVENTIONAL_HOOK_LINE = re.compile(r"^\s*-\s+id:\s*conventional-pre-commit\s*$")
ARGS_LINE = re.compile(r"^(?P<indent>\s*)args:\s*$")
LIST_ITEM_LINE = re.compile(r"^(?P<indent>\s*)-\s+(?P<item>[^\s#]+)\s*$")
# The start of the next hook or repo entry, where the conventional-pre-commit hook has ended.
NEXT_ENTRY_LINE = re.compile(r"^\s*-\s+(?:id|repo):")


def conventional_commit_types(text: str) -> list[str]:
    """The commit types the conventional-pre-commit hook accepts: its `args:` minus `--options`.

    The hook takes the types as positional arguments, one per item of the block list pre-commit
    configs use. Raises ValueError when there is no such hook, when it has no block-style
    `args:` list (a flow list `[a, b]` is not read), or when the list names no type.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if CONVENTIONAL_HOOK_LINE.match(line)), None)
    if start is None:
        raise ValueError("no conventional-pre-commit hook")
    args_indent: int | None = None
    types: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if args_indent is None:
            if NEXT_ENTRY_LINE.match(line):
                break
            if args_match := ARGS_LINE.match(line):
                args_indent = len(args_match["indent"])
            continue
        item = LIST_ITEM_LINE.match(line)
        if item is None or len(item["indent"]) <= args_indent:
            break
        if not item["item"].startswith("-"):
            types.append(item["item"])
    if args_indent is None:
        raise ValueError("the conventional-pre-commit hook has no block-style args: list")
    if not types:
        raise ValueError("the conventional-pre-commit hook's args: name no commit type")
    return types


# `      - id: <hook>` opens a hook entry; `        key: value` lines follow (the value kept raw,
# so a flow list `[a, b]` stays one string).
HOOK_ID_LINE = re.compile(r"^\s*-\s+id:\s*(?P<id>\S+)\s*$")
HOOK_KEY_LINE = re.compile(r"^\s+(?P<key>[A-Za-z_][A-Za-z0-9_-]*):\s*(?P<value>.*?)\s*$")


def hook_entries(text: str, repo_url: str) -> list[dict[str, str]]:
    """The hooks of the `repo:` entry `repo_url`, each as {key: raw value} of its `key: value`
    lines (a block list under a key is not read). Raises ValueError without that repo."""
    lines = text.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if (match := REPO_LINE.match(line)) and match["repo"] == repo_url
        ),
        None,
    )
    if start is None:
        raise ValueError(f"no repo {repo_url} in the pre-commit config")
    hooks: list[dict[str, str]] = []
    for line in lines[start + 1 :]:
        if REPO_LINE.match(line):
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if id_match := HOOK_ID_LINE.match(line):
            hooks.append({"id": id_match["id"]})
        elif hooks and (key_match := HOOK_KEY_LINE.match(line)):
            hooks[-1][key_match["key"]] = re.sub(r"\s+#.*$", "", key_match["value"])
    return hooks


def flow_list(value: str) -> list[str]:
    """`[a, "b"]` -> ["a", "b"]: a YAML flow list of plain or quoted scalars."""
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"not a flow list: {value!r}")
    items = (item.strip().strip("\"'") for item in value[1:-1].split(","))
    return [item for item in items if item]


# ---------------------------------------------------------------------------
# .github/workflows/ci.yml
# ---------------------------------------------------------------------------

# The pull request title check's `types='feat|fix|...'` shell assignment: the alternation its
# Conventional Commit regex is built from.
PR_TITLE_TYPES = re.compile(
    r"""^\s*types=(?P<quote>["'])(?P<types>[a-z]+(?:\|[a-z]+)*)(?P=quote)\s*$""", re.MULTILINE
)


def pull_request_title_types(text: str) -> list[str]:
    """The commit types CI's pull request title check accepts. Raises ValueError without one."""
    matches = list(PR_TITLE_TYPES.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected one types='a|b|...' line in ci.yml, found {len(matches)}")
    return matches[0]["types"].split("|")


# `- name: <step>` opens a step; `key:` (a block follows) and `key: value` are its keys.
STEP_NAME_LINE = re.compile(r"^(?P<indent>\s*)-\s+name:\s*(?P<name>.+?)\s*$")
BLOCK_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_-]+):\s*\|?\s*$")
MAPPING_ITEM_LINE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*?)\s*$")


def indentation(line: str) -> int:
    """The number of leading spaces of `line`."""
    return len(line) - len(line.lstrip(" "))


def workflow_step_block(text: str, step_name: str, key: str) -> list[str]:
    """The lines nested under `key:` (or `key: |`) in the workflow step named `step_name`.

    Raises ValueError when there is no such step, or when the step has no such block.
    """
    lines = text.splitlines()
    step = next(
        (
            (i, len(match["indent"]))
            for i, line in enumerate(lines)
            if (match := STEP_NAME_LINE.match(line)) and match["name"] == step_name
        ),
        None,
    )
    if step is None:
        raise ValueError(f"no step named {step_name!r}")
    start, step_indent = step
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and indentation(lines[i]) <= step_indent:
            break  # the next step
        if (match := BLOCK_KEY_LINE.match(lines[i])) and match["key"] == key:
            block: list[str] = []
            for line in lines[i + 1 :]:
                if line.strip() and indentation(line) <= len(match["indent"]):
                    break
                block.append(line)
            if any(line.strip() for line in block):
                return block
            break
    raise ValueError(f"step {step_name!r} has no {key}: block")


def workflow_step_script(text: str, step_name: str) -> str:
    """The `run: |` script of the workflow step named `step_name`, dedented."""
    script = textwrap.dedent("\n".join(workflow_step_block(text, step_name, "run")))
    return script.strip("\n") + "\n"


def workflow_step_mapping(text: str, step_name: str, key: str) -> dict[str, str]:
    """The `env:` or `with:` mapping of the workflow step named `step_name`, quotes removed.

    Only one-line `name: value` items are read (a comment line is skipped).
    """
    mapping: dict[str, str] = {}
    for line in workflow_step_block(text, step_name, key):
        if line.strip() and not line.lstrip().startswith("#"):
            item = MAPPING_ITEM_LINE.match(line)
            if item is None:
                raise ValueError(f"step {step_name!r}: unreadable {key}: line {line.strip()!r}")
            mapping[item["key"]] = item["value"].strip("\"'")
    return mapping


# ---------------------------------------------------------------------------
# justfile
# ---------------------------------------------------------------------------


def recipe_body(text: str, name: str) -> str:
    """The body of the justfile recipe `name`, dedented: the indented and blank lines after its
    header, up to the next line at column 0. Raises ValueError without that recipe."""
    header = re.compile(rf"^{re.escape(name)}(?:\s[^:]*)?:(?!=)(?:\s.*)?$")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        raise ValueError(f"the justfile has no recipe {name!r}")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and not line[0].isspace():
            break
        body.append(line)
    script = textwrap.dedent("\n".join(body)).strip("\n")
    if not script:
        raise ValueError(f"the recipe {name!r} has no body")
    return script + "\n"


def setup_script(python: str) -> str:
    """The shebang script `just setup <python>` runs, interpolated the way just does it."""
    script = recipe_body(JUSTFILE.read_text(encoding="utf-8"), "setup")
    script = script.replace("{{ python }}", python)
    if "{{" in script:
        raise ValueError("the setup recipe interpolates more than {{ python }}; extend the test")
    return script


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

# An exact pin, `name==version`, as the dev extra writes them (no extras, markers or ranges).
EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;,\[\]]+)$")
# The `dev = [ ... ]` array in the [project.optional-dependencies] table, which runs until the
# next line that opens a table. The array body is whitespace, commas, comments and strings, so a
# `]` inside a string or a comment does not end it early.
DEV_ARRAY = re.compile(
    r"^\[project\.optional-dependencies\]\s*$"
    r"(?:(?!^\[)[\s\S])*?"
    r"^dev\s*=\s*\[(?P<items>(?:\s|,|#[^\n]*|\"[^\"\\\n]*\"|'[^'\n]*')*)\]",
    re.MULTILINE,
)
# A basic (no escapes: requirement strings need none) or literal TOML string, or a comment.
TOML_STRING_OR_COMMENT = re.compile(r"\"(?P<basic>[^\"\\\n]*)\"|'(?P<literal>[^'\n]*)'|#[^\n]*")


def normalize_name(name: str) -> str:
    """PEP 503 normalization, so `Pre_Commit` and `pre-commit` are the same distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def exact_pins(requirements: list[str]) -> dict[str, str]:
    """{normalized name: version} for the requirements pinned exactly; others are ignored."""
    pins: dict[str, str] = {}
    for requirement in requirements:
        if match := EXACT_PIN.match(requirement.strip()):
            pins[normalize_name(match["name"])] = match["version"]
    return pins


def dev_pins_regex(text: str) -> dict[str, str]:
    """The exact pins of pyproject's `dev` extra, read without a TOML parser (Python 3.10).

    Handles the forms a hand-edited or Dependabot-edited array takes: one or many strings per
    line, double or single quotes, trailing commas, comments. Raises ValueError when there is
    no `dev` array under [project.optional-dependencies].
    """
    match = DEV_ARRAY.search(text)
    if match is None:
        raise ValueError("pyproject.toml has no dev array in [project.optional-dependencies]")
    strings = [
        token["basic"] if token["basic"] is not None else token["literal"]
        for token in TOML_STRING_OR_COMMENT.finditer(match["items"])
        if token["basic"] is not None or token["literal"] is not None
    ]
    return exact_pins(strings)


def dev_pins(text: str) -> dict[str, str]:
    """The exact pins of pyproject's `dev` extra: tomllib on 3.11+, the regex reader on 3.10."""
    if sys.version_info >= (3, 11):
        import tomllib

        extras = tomllib.loads(text).get("project", {}).get("optional-dependencies", {})
        if "dev" not in extras:
            raise ValueError("pyproject.toml has no dev array in [project.optional-dependencies]")
        return exact_pins(list(extras["dev"]))
    return dev_pins_regex(text)


def pyproject_text() -> str:
    """The repository's pyproject.toml."""
    return PYPROJECT.read_text(encoding="utf-8")


def version_tuple(version: str) -> tuple[int, ...]:
    """`4.6.2` -> (4, 6, 2): enough for the plain release numbers compared here."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version):
        raise ValueError(f"not a plain release number: {version!r}")
    return tuple(int(part) for part in version.split("."))


def mirror_version(release: str) -> str:
    """A pre-commit mirror's tag -> the PyPI version it installs: strip the `v` prefix and a
    `-N` mirror rebuild suffix (shellcheck-py's v0.11.0.1-1 repackages 0.11.0.1)."""
    return re.sub(r"-[0-9]+$", "", release.removeprefix("v"))


# ---------------------------------------------------------------------------
# The parsers themselves (synthetic inputs)
# ---------------------------------------------------------------------------


# A pyproject.toml as hand edits and Dependabot leave it: one or many strings per line, both
# quote styles, comments (one holding a `]`), a range and a requirement with extras (neither is
# an exact pin), a non-normalized name, and a `dev` key in another table that must not count.
SYNTHETIC_PYPROJECT = (
    "[project]\n"
    'name = "x"\n'
    "\n"
    "[project.optional-dependencies]\n"
    'docs = ["sphinx==1.0"]\n'
    "dev = [\n"
    '    "mypy==1.20.1",  # type checker\n'
    "    'Pre_Commit==4.6.2', \"ruff==0.15.10\",\n"
    '    "types-foo>=1.0",  # a range, not a pin [ignored]\n'
    '    "foo[bar]==2.0",\n'
    "]\n"
    "\n"
    "[tool.ruff]\n"
    'dev = ["not==this"]\n'
)
SYNTHETIC_DEV_PINS = {"mypy": "1.20.1", "pre-commit": "4.6.2", "ruff": "0.15.10"}


class HookRepoParsingTests(unittest.TestCase):
    """parse_hook_repos() reads what pre-commit writes and refuses what it cannot read."""

    def test_reads_frozen_and_tag_revisions(self) -> None:
        text = (
            "repos:\n"
            "  - repo: meta\n"
            "    hooks:\n"
            "      - id: check-hooks-apply\n"
            "  - repo: https://example.invalid/frozen\n"
            f"    rev: {'a' * 40}  # frozen: v1.2.3\n"
            "  - repo: local\n"
            "    hooks: []\n"
            "  - repo: https://example.invalid/tag\n"
            "    rev: v4.5.6\n"
        )
        repos = parse_hook_repos(text)
        self.assertEqual(
            repos,
            {
                "https://example.invalid/frozen": HookRepo(
                    "https://example.invalid/frozen", "a" * 40, "v1.2.3"
                ),
                "https://example.invalid/tag": HookRepo(
                    "https://example.invalid/tag", "v4.5.6", None
                ),
            },
        )
        self.assertEqual(repos["https://example.invalid/frozen"].release, "v1.2.3")
        self.assertEqual(repos["https://example.invalid/tag"].release, "v4.5.6")

    def test_refuses_layouts_it_cannot_read(self) -> None:
        cases = {
            "missing rev": "  - repo: https://example.invalid/a\n  - repo: local\n",
            "missing final rev": "  - repo: https://example.invalid/a\n",
            "orphan rev": "  - repo: local\n    rev: v1\n",
            "duplicate repo": (
                "  - repo: https://example.invalid/a\n    rev: v1\n"
                "  - repo: https://example.invalid/a\n    rev: v2\n"
            ),
            "odd rev line": "  - repo: https://example.invalid/a\n    rev: v1 # pinned\n",
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_hook_repos(text)


class DevPinParsingTests(unittest.TestCase):
    """The pyproject readers agree, on the real file and on the forms an edit can produce."""

    def test_regex_reader_handles_edited_arrays(self) -> None:
        self.assertEqual(dev_pins_regex(SYNTHETIC_PYPROJECT), SYNTHETIC_DEV_PINS)

    def test_readers_refuse_a_missing_dev_extra(self) -> None:
        text = '[project]\nname = "x"\n\n[tool.ruff]\ndev = ["ruff==1"]\n'
        with self.assertRaises(ValueError):
            dev_pins_regex(text)
        with self.assertRaises(ValueError):
            dev_pins(text)

    @unittest.skipIf(sys.version_info < (3, 11), "tomllib needs Python 3.11+")
    def test_regex_reader_agrees_with_tomllib(self) -> None:
        for name, text in (
            ("synthetic", SYNTHETIC_PYPROJECT),
            ("pyproject.toml", pyproject_text()),
        ):
            with self.subTest(name):
                self.assertEqual(dev_pins_regex(text), dev_pins(text))

    def test_mirror_version_strips_prefix_and_rebuild_suffix(self) -> None:
        self.assertEqual(mirror_version("v0.11.0.1-1"), "0.11.0.1")
        self.assertEqual(mirror_version("v1.20.1"), "1.20.1")
        self.assertEqual(mirror_version("0.38.2"), "0.38.2")


class CommitTypeParsingTests(unittest.TestCase):
    """The commit type readers take the lists as the two files write them, and nothing else."""

    HOOK = (
        "  - repo: https://example.invalid/conventional\n"
        f"    rev: {'b' * 40}  # frozen: v4.4.0\n"
        "    hooks:\n"
        "      - id: conventional-pre-commit\n"
        "        args:\n"
        "          - --verbose\n"
        "          # a comment inside the list\n"
        "          - feat\n"
        "          - fix\n"
        "\n"
        "          - deps\n"
    )

    def test_reads_the_hook_types_and_drops_options(self) -> None:
        for name, tail in (
            ("end of file", ""),
            ("next hook", "      - id: other\n        args:\n          - no\n"),
            ("next repo", "  - repo: local\n    hooks:\n      - id: x\n        args:\n"),
        ):
            with self.subTest(name):
                self.assertEqual(
                    conventional_commit_types(self.HOOK + tail), ["feat", "fix", "deps"]
                )

    def test_refuses_hooks_it_cannot_read(self) -> None:
        cases = {
            "no hook": "  - repo: local\n    hooks:\n      - id: other\n        args:\n",
            "no args": "      - id: conventional-pre-commit\n      - id: other\n        args:\n",
            "flow list": "      - id: conventional-pre-commit\n        args: [feat, fix]\n",
            "options only": (
                "      - id: conventional-pre-commit\n        args:\n          - --strict\n"
            ),
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                conventional_commit_types(text)

    def test_reads_the_title_check_types(self) -> None:
        for quote in "'\"":
            with self.subTest(quote=quote):
                text = f"        run: |\n          types={quote}feat|fix|ci{quote}\n"
                self.assertEqual(pull_request_title_types(text), ["feat", "fix", "ci"])
        for name, text in (
            ("none", "run: echo\n"),
            ("two", "types='feat'\ntypes='fix'\n"),
            ("unquoted", "types=feat|fix\n"),
        ):
            with self.subTest(name), self.assertRaises(ValueError):
                pull_request_title_types(text)


# ---------------------------------------------------------------------------
# The repository's configuration
# ---------------------------------------------------------------------------


class PinSyncTests(unittest.TestCase):
    """Versions and commit types set both in .pre-commit-config.yaml and elsewhere are equal."""

    def setUp(self) -> None:
        self.repos = parse_hook_repos(pre_commit_config_text())
        self.pins = dev_pins(pyproject_text())

    def assert_hook_matches_pin(self, hook_repo: str, package: str) -> None:
        """The hook mirror's release equals pyproject's exact `dev` pin of `package`."""
        self.assertIn(hook_repo, self.repos, f"{hook_repo} is not in .pre-commit-config.yaml")
        self.assertIn(package, self.pins, f'pyproject.toml\'s dev extra does not pin "{package}=="')
        self.assertEqual(
            mirror_version(self.repos[hook_repo].release),
            self.pins[package],
            f"{package}: .pre-commit-config.yaml ({hook_repo}) and pyproject.toml's dev extra "
            "pin different versions; move both to the same release (the rev as a frozen SHA "
            "with its `# frozen: vX.Y.Z` comment)",
        )

    def test_ruff_hook_matches_pyproject_pin(self) -> None:
        self.assert_hook_matches_pin(RUFF_HOOKS, "ruff")

    def test_mypy_hook_matches_pyproject_pin(self) -> None:
        self.assert_hook_matches_pin(MYPY_HOOKS, "mypy")

    def test_pyproject_pre_commit_meets_the_config_minimum(self) -> None:
        # `just setup` installs pre-commit from the dev extra; an older one refuses the config.
        match = MINIMUM_PRE_COMMIT.search(pre_commit_config_text())
        if match is None:
            self.fail("minimum_pre_commit_version is not set in .pre-commit-config.yaml")
        self.assertIn(
            "pre-commit", self.pins, 'pyproject.toml\'s dev extra must pin "pre-commit=="'
        )
        self.assertGreaterEqual(
            version_tuple(self.pins["pre-commit"]), version_tuple(match["version"])
        )

    def test_actionlint_uses_the_shellcheck_hook_release(self) -> None:
        # actionlint shellchecks `run:` blocks with the shellcheck-py in its own environment;
        # the shellcheck hook runs its own copy. Both must be the same shellcheck.
        pins = SHELLCHECK_PY_PIN.findall(pre_commit_config_text())
        self.assertEqual(len(pins), 1, "expected one shellcheck-py== pin (actionlint's)")
        self.assertIn(SHELLCHECK_HOOKS, self.repos)
        self.assertEqual(pins[0], mirror_version(self.repos[SHELLCHECK_HOOKS].release))

    def test_pull_request_titles_accept_the_commit_msg_types(self) -> None:
        # A squash merge makes the pull request title the commit subject on main, so the title
        # check in CI's lint job must accept exactly the types the commit-msg hook accepts.
        hook_types = conventional_commit_types(pre_commit_config_text())
        try:
            title_types = pull_request_title_types(CI_WORKFLOW.read_text(encoding="utf-8"))
        except ValueError as error:
            self.fail(f"the pull request title check in .github/workflows/ci.yml: {error}")
        self.assertEqual(len(hook_types), len(set(hook_types)), "a commit type is listed twice")
        self.assertEqual(
            sorted(title_types),
            sorted(hook_types),
            "the conventional-pre-commit args in .pre-commit-config.yaml and the types='...' of "
            "the pull request title check in .github/workflows/ci.yml differ; list the same types",
        )


class SupplyChainTests(unittest.TestCase):
    """Every remote hook runs code frozen at a reviewed commit."""

    def test_every_remote_rev_is_a_frozen_commit_sha(self) -> None:
        repos = parse_hook_repos(pre_commit_config_text())
        self.assertTrue(repos, "no remote hook repositories found")
        for url, repo in repos.items():
            with self.subTest(url):
                self.assertRegex(repo.rev, FULL_SHA, "rev must be a full 40-hex commit SHA")
                self.assertIsNotNone(
                    repo.frozen, "a frozen rev needs its `# frozen: <tag>` comment"
                )


class ZizmorHookTests(unittest.TestCase):
    """zizmor never goes online at commit time, and CI still runs its online audits."""

    def setUp(self) -> None:
        self.hooks = hook_entries(pre_commit_config_text(), ZIZMOR_HOOKS)

    def test_the_commit_time_hook_runs_offline(self) -> None:
        # With GH_TOKEN or GITHUB_TOKEN exported, zizmor would otherwise go online and fail
        # the commit whenever GitHub is unreachable.
        commit_time = [hook for hook in self.hooks if "alias" not in hook]
        self.assertEqual(len(commit_time), 1, self.hooks)
        hook = commit_time[0]
        self.assertEqual(hook["id"], "zizmor")
        if "stages" in hook:
            self.assertIn("pre-commit", flow_list(hook["stages"]))
        self.assertIn("--offline", flow_list(hook.get("args", "[]")))
        # `args` replaces the hook's own, which disable the progress bar.
        self.assertIn("--no-progress", flow_list(hook.get("args", "[]")))

    def test_ci_runs_the_online_audits_from_the_manual_stage(self) -> None:
        online = [hook for hook in self.hooks if hook.get("alias") == "zizmor-online"]
        self.assertEqual(len(online), 1, self.hooks)
        self.assertEqual(online[0]["id"], "zizmor")
        self.assertEqual(flow_list(online[0].get("stages", "[]")), ["manual"])
        self.assertNotIn("--offline", flow_list(online[0].get("args", "[]")))
        ci = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pre-commit run --hook-stage manual zizmor-online --all-files", ci)
        # Only clippy (its own job) is skipped, so the lint step runs the offline zizmor too.
        self.assertEqual(re.findall(r"^\s*SKIP:\s*(\S+)\s*$", ci, re.MULTILINE), ["cargo-clippy"])

    def test_hook_entries_reads_keys_and_flow_lists(self) -> None:
        text = (
            "  - repo: https://example.invalid/a\n"
            f"    rev: {'c' * 40}  # frozen: v1\n"
            "    hooks:\n"
            "      # a comment\n"
            "      - id: one\n"
            "        args: [--x, '--y']  # trailing comment\n"
            "      - id: one\n"
            "        alias: two\n"
            "        stages: [manual]\n"
            "  - repo: local\n"
            "    hooks:\n"
            "      - id: other\n"
        )
        self.assertEqual(
            hook_entries(text, "https://example.invalid/a"),
            [
                {"id": "one", "args": "[--x, '--y']"},
                {"id": "one", "alias": "two", "stages": "[manual]"},
            ],
        )
        self.assertEqual(flow_list("[--x, '--y', \"z\"]"), ["--x", "--y", "z"])
        with self.assertRaises(ValueError):
            hook_entries(text, "https://example.invalid/missing")
        with self.assertRaises(ValueError):
            flow_list("--x")


@unittest.skipUnless(BASH, "the title check is a bash script")
class PullRequestTitleCheckTests(unittest.TestCase):
    """CI's pull request title check, run as GitHub runs it, against a fake `gh`."""

    STEP = "Check the pull request title"

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        # `gh api repos/.../pulls/N --jq .title` prints the title.
        gh = bin_dir / "gh"
        gh.write_text("#!/bin/sh\nprintf '%s\\n' \"$FAKE_TITLE\"\n", encoding="utf-8")
        gh.chmod(0o755)
        self.script = root / "check-title.sh"
        self.script.write_text(
            workflow_step_script(CI_WORKFLOW.read_text(encoding="utf-8"), self.STEP),
            encoding="utf-8",
        )
        self.env = {
            **{key: value for key, value in os.environ.items() if not key.startswith("FAKE_")},
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "GH_TOKEN": "fake",
            "REPOSITORY": "owner/repository",
            "PR_NUMBER": "1",
        }

    def check(self, title: str) -> tuple[int, str]:
        """Run the check on `title`: its exit status and output."""
        assert BASH is not None
        result = subprocess.run(
            # GitHub's `shell: bash`.
            [BASH, "--noprofile", "--norc", "-eo", "pipefail", str(self.script)],
            env={**self.env, "FAKE_TITLE": title},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return result.returncode, result.stdout + result.stderr

    def test_accepts_conventional_commit_and_revert_titles(self) -> None:
        for title in (
            "fix(ember): clamp the ink density",
            "feat!: drop the legacy output layout",
            "build(deps): bump libc from 0.2.1 to 0.2.2",
            'Revert "feat: add the ember edition"',
            "docs: explain when CI skips a commit",  # no bracketed instruction
        ):
            with self.subTest(title):
                status, output = self.check(title)
                self.assertEqual(status, 0, output)

    def test_rejects_skip_instructions_in_any_letter_case(self) -> None:
        # GitHub starts no push run for a commit whose message holds one of these, so the squash
        # commit on main would never get the `CI passed` check the deploy agent waits for.
        for title in (
            "ci: document [skip ci]",
            "fix: typo [CI SKIP]",
            "feat: [No Ci] faster renders",
            "build(deps): bump foo [skip actions]",
            "chore: tidy [Actions Skip]",
        ):
            with self.subTest(title):
                status, output = self.check(title)
                self.assertEqual(status, 1, output)
                self.assertIn("would stop CI from running", output)

    def test_rejects_titles_that_are_not_conventional_commits(self) -> None:
        for title in ("Update README", "fix:missing space", "Fix: uppercase type"):
            with self.subTest(title):
                status, output = self.check(title)
                self.assertEqual(status, 1, output)
                self.assertIn("must be a Conventional Commit subject", output)

    def test_workflow_step_script_reads_only_its_step(self) -> None:
        text = (
            "    steps:\n"
            "      - name: First\n"
            "        run: |\n"
            "          echo one\n"
            "\n"
            "          echo two\n"
            "      - name: Second\n"
            "        run: echo three\n"
        )
        self.assertEqual(workflow_step_script(text, "First"), "echo one\n\necho two\n")
        for name in ("Second", "Missing"):
            with self.subTest(name), self.assertRaises(ValueError):
                workflow_step_script(text, name)

    def test_workflow_step_mapping_reads_env_and_with(self) -> None:
        text = (
            "      - name: Step\n"
            "        uses: some/action@sha\n"
            "        with:\n"
            "          # a comment\n"
            "          path: ${{ steps.x.outputs.path }}\n"
            "        env:\n"
            '          ONE: "1"\n'
            "          TWO: two\n"
            "        run: |\n"
            "          echo\n"
            "      - name: Next\n"
            "        env:\n"
            "          THREE: '3'\n"
        )
        self.assertEqual(workflow_step_mapping(text, "Step", "env"), {"ONE": "1", "TWO": "two"})
        self.assertEqual(
            workflow_step_mapping(text, "Step", "with"), {"path": "${{ steps.x.outputs.path }}"}
        )
        with self.assertRaises(ValueError):
            workflow_step_mapping(text, "Next", "with")


# Homebrew as far as the macOS FFmpeg steps use it, around the files of FAKE_ROOT: `installed`
# ("name version" lines), `tap/Formula/ffmpeg.rb`, `Cellar/` and `ffmpeg-state` (whether the
# linked ffmpeg works). Every call is logged with HOMEBREW_NO_AUTO_UPDATE's value in brackets.
FAKE_BREW = """\
#!/bin/sh
echo "brew[${HOMEBREW_NO_AUTO_UPDATE:-}] $*" >> "$FAKE_LOG"
case "$1" in
--cellar) echo "$FAKE_ROOT/Cellar" ;;
--repository) echo "$FAKE_ROOT/tap" ;;
deps) printf 'libpng\\nx265\\n' ;;
list)
    shift 2  # --formula --versions
    if [ $# -eq 0 ]; then
        cat "$FAKE_ROOT/installed"
        exit 0
    fi
    status=0
    for name in "$@"; do
        grep "^$name " "$FAKE_ROOT/installed" || status=1
    done
    exit $status
    ;;
link) [ -d "$FAKE_ROOT/Cellar/ffmpeg" ] || exit 1 ;;
install)
    case "$*" in
    *--only-dependencies*) ;;
    *)
        mkdir -p "$FAKE_ROOT/Cellar/ffmpeg/8.0"
        echo works > "$FAKE_ROOT/ffmpeg-state"
        ;;
    esac
    ;;
esac
exit 0
"""
# The linked ffmpeg: lists the three encoders, or fails to load a library (a restored keg whose
# dependency moved on).
FAKE_FFMPEG = """\
#!/bin/sh
if [ "$(cat "$FAKE_ROOT/ffmpeg-state" 2>/dev/null)" != works ]; then
    echo "dyld: Library not loaded" >&2
    exit 1
fi
printf ' V....D libx264   H.264\\n V....D libx265   H.265\\n V....D libwebp   WebP\\n'
"""
FFMPEG_PREP_STEP = "Prepare the FFmpeg build (macOS)"
FFMPEG_RESTORE_STEP = "Restore the cached FFmpeg build (macOS)"
FFMPEG_BUILD_STEP = "Install FFmpeg with libwebp (macOS)"
FFMPEG_ARGS = "homebrew-ffmpeg/ffmpeg/ffmpeg --with-webp"


@unittest.skipUnless(BASH and shutil.which("shasum"), "the FFmpeg steps need bash and shasum")
class MacFfmpegCacheTests(unittest.TestCase):
    """The macOS FFmpeg cache steps of ci.yml, run with the step's env against a fake brew."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name, text in (("brew", FAKE_BREW), ("ffmpeg", FAKE_FFMPEG)):
            (bin_dir / name).write_text(text, encoding="utf-8")
            (bin_dir / name).chmod(0o755)
        self.formula = self.root / "tap" / "Formula" / "ffmpeg.rb"
        self.formula.parent.mkdir(parents=True)
        self.formula.write_text("class Ffmpeg < Formula\nend\n", encoding="utf-8")
        self.set_installed(libpng="1.6.58", x265="4.1", webp="1.5.0", unrelated="1.0")
        (self.root / "Cellar").mkdir()
        self.log = self.root / "calls.log"
        self.workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        self.env = {
            **{key: value for key, value in os.environ.items() if not key.startswith("FAKE_")},
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "FAKE_ROOT": str(self.root),
            "FAKE_LOG": str(self.log),
            "ImageOS": "macos26",
            "RUNNER_ARCH": "ARM64",
        }
        self.env.pop("HOMEBREW_NO_AUTO_UPDATE", None)

    def set_installed(self, **versions: str) -> None:
        lines = "".join(f"{name} {version}\n" for name, version in versions.items())
        (self.root / "installed").write_text(lines, encoding="utf-8")

    def run_step(self, step: str, **env: str) -> tuple[dict[str, str], str]:
        """Run a step's script with its literal `env:` values; its outputs and its output."""
        assert BASH is not None
        script = self.root / "step.sh"
        script.write_text(workflow_step_script(self.workflow, step), encoding="utf-8")
        outputs = self.root / "github_output"
        outputs.write_text("", encoding="utf-8")
        try:
            step_env = workflow_step_mapping(self.workflow, step, "env")
        except ValueError:
            step_env = {}
        literal_env = {key: value for key, value in step_env.items() if "${{" not in value}
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-eo", "pipefail", str(script)],
            env={**self.env, **literal_env, "GITHUB_OUTPUT": str(outputs), **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        pairs = (line.split("=", 1) for line in outputs.read_text(encoding="utf-8").splitlines())
        return {key: value for key, value in pairs}, output

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def restore_keg(self, *, works: bool) -> None:
        """What actions/cache/restore leaves: a keg in the Cellar, not linked yet."""
        (self.root / "Cellar" / "ffmpeg" / "7.1").mkdir(parents=True)
        (self.root / "ffmpeg-state").write_text("works" if works else "broken", encoding="utf-8")

    def build(self, *, cache_hit: str, matched_key: str) -> tuple[dict[str, str], str]:
        return self.run_step(FFMPEG_BUILD_STEP, CACHE_HIT=cache_hit, MATCHED_KEY=matched_key)

    def test_the_key_covers_the_formula_and_its_dependencies_only(self) -> None:
        outputs, _ = self.run_step(FFMPEG_PREP_STEP)
        key, prefix = outputs["key"], outputs["prefix"]
        self.assertEqual(prefix, "ffmpeg-webp-macos26-ARM64-")
        self.assertRegex(key, rf"^{re.escape(prefix)}[0-9a-f]{{16}}$")
        # A formula outside the dependency closure (a runner image update) keeps the key.
        self.set_installed(libpng="1.6.58", x265="4.1", webp="1.5.0", unrelated="2.0")
        self.assertEqual(self.run_step(FFMPEG_PREP_STEP)[0]["key"], key)
        # A dependency update, or a new formula, changes it.
        self.set_installed(libpng="1.6.59", x265="4.1", webp="1.5.0", unrelated="2.0")
        dependency_key = self.run_step(FFMPEG_PREP_STEP)[0]["key"]
        self.assertNotEqual(dependency_key, key)
        self.formula.write_text("class Ffmpeg < Formula # 8.1\nend\n", encoding="utf-8")
        self.assertNotEqual(self.run_step(FFMPEG_PREP_STEP)[0]["key"], dependency_key)

    def test_homebrew_updates_once_then_holds_still(self) -> None:
        # The first install may update Homebrew, so the core formulae match the fresh tap;
        # every later call keeps the dependencies the key describes.
        self.run_step(FFMPEG_PREP_STEP)
        calls = self.calls()
        first_install = next(call for call in calls if re.match(r"brew\[1?\] install ", call))
        self.assertEqual(first_install, "brew[] install --only-dependencies " + FFMPEG_ARGS)
        later = calls[calls.index(first_install) + 1 :]
        self.assertTrue(later)
        self.assertEqual([call for call in later if not call.startswith("brew[1] ")], [])
        self.run_step(FFMPEG_BUILD_STEP, CACHE_HIT="false", MATCHED_KEY="")
        self.assertIn("brew[1] install " + FFMPEG_ARGS, self.calls())

    def test_the_restore_falls_back_to_any_keg_of_the_same_image(self) -> None:
        restore = workflow_step_mapping(self.workflow, FFMPEG_RESTORE_STEP, "with")
        self.assertEqual(restore["restore-keys"], "${{ steps.ffmpeg-prep.outputs.prefix }}")
        self.assertEqual(restore["key"], "${{ steps.ffmpeg-prep.outputs.key }}")
        # The save step saves whatever the build step asks to.
        self.assertIn("steps.ffmpeg-build.outputs.save == 'true'", self.workflow)

    def test_an_exact_hit_is_used_and_not_saved_again(self) -> None:
        self.restore_keg(works=True)
        outputs, output = self.build(cache_hit="true", matched_key="ffmpeg-webp-x-new")
        self.assertEqual(outputs, {"save": "false"})
        self.assertIn("Using the cached FFmpeg build.", output)
        self.assertNotIn("brew[1] install " + FFMPEG_ARGS, self.calls())

    def test_an_older_keg_that_still_works_is_reused_and_saved_under_the_new_key(self) -> None:
        self.restore_keg(works=True)
        outputs, output = self.build(cache_hit="false", matched_key="ffmpeg-webp-x-old")
        self.assertEqual(outputs, {"save": "true"})
        self.assertIn("cached under ffmpeg-webp-x-old", output)
        self.assertNotIn("brew[1] install " + FFMPEG_ARGS, self.calls())
        self.assertTrue((self.root / "Cellar" / "ffmpeg" / "7.1").is_dir())

    def test_a_keg_that_does_not_work_is_rebuilt(self) -> None:
        self.restore_keg(works=False)
        outputs, output = self.build(cache_hit="false", matched_key="ffmpeg-webp-x-old")
        self.assertEqual(outputs, {"save": "true"})
        self.assertIn("::warning title=FFmpeg cache::", output)
        self.assertIn("brew[1] install " + FFMPEG_ARGS, self.calls())
        self.assertFalse((self.root / "Cellar" / "ffmpeg" / "7.1").exists())

    def test_without_a_cached_keg_it_builds(self) -> None:
        outputs, _ = self.build(cache_hit="false", matched_key="")
        self.assertEqual(outputs, {"save": "true"})
        self.assertIn("brew[1] install " + FFMPEG_ARGS, self.calls())


# The interpreter a venv gets: every call is logged; `-m pip` fails until pip is installed (a
# `pip-installed` file next to it), like a venv whose ensurepip step failed.
FAKE_VENV_PYTHON = """\
#!/bin/sh
echo "venv-python $*" >> "$FAKE_LOG"
if [ "$1" = -m ] && [ "$2" = pip ] && [ ! -e "$(dirname "$0")/pip-installed" ]; then
    echo "No module named pip" >&2
    exit 1
fi
exit 0
"""
# The interpreter `just setup` is given. `-m venv DIR` creates a fake venv; with
# FAKE_ENSUREPIP_MISSING set it fails half-way, as python3 does on Debian and Ubuntu without
# python3-venv: the directory and bin/python are left behind, without pip.
FAKE_BASE_PYTHON = """\
#!/bin/sh
echo "base-python $*" >> "$FAKE_LOG"
if [ "$1" = -m ] && [ "$2" = venv ]; then
    mkdir -p "$3/bin"
    cp "$FAKE_TEMPLATES/python" "$FAKE_TEMPLATES/pre-commit" "$3/bin/"
    if [ -n "${FAKE_ENSUREPIP_MISSING:-}" ]; then
        echo "ensurepip is not available" >&2
        exit 1
    fi
    touch "$3/bin/pip-installed"
fi
exit 0
"""
FAKE_PRE_COMMIT = """\
#!/bin/sh
echo "pre-commit $*" >> "$FAKE_LOG"
"""


@unittest.skipUnless(BASH and GIT, "the setup recipe needs bash and git")
class SetupRecipeTests(unittest.TestCase):
    """`just setup`'s script in a real git repository, with fake interpreters and pre-commit."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        templates = self.root / "templates"
        templates.mkdir()
        for name, text in (("python", FAKE_VENV_PYTHON), ("pre-commit", FAKE_PRE_COMMIT)):
            (templates / name).write_text(text, encoding="utf-8")
            (templates / name).chmod(0o755)
        self.base_python = self.root / "python3-base"
        self.base_python.write_text(FAKE_BASE_PYTHON, encoding="utf-8")
        self.base_python.chmod(0o755)
        self.log = self.root / "calls.log"
        gitconfig = self.root / "gitconfig"
        gitconfig.write_text(
            "[user]\n\tname = Test\n\temail = test@example.invalid\n", encoding="utf-8"
        )
        # GIT_* includes GIT_DIR, which git sets for hooks: the suite also runs from pre-push.
        self.env = {
            key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "FAKE_"))
        }
        self.env.update(
            HOME=str(self.root),
            GIT_CONFIG_GLOBAL=str(gitconfig),
            GIT_CONFIG_NOSYSTEM="1",
            FAKE_LOG=str(self.log),
            FAKE_TEMPLATES=str(templates),
        )
        self.checkout = self.root / "checkout"
        self.git(self.root, "init", "--quiet", str(self.checkout))
        self.git(self.checkout, "commit", "--quiet", "--allow-empty", "--message", "init")

    def git(self, cwd: Path, *args: str) -> None:
        assert GIT is not None
        subprocess.run([GIT, "-C", str(cwd), *args], env=self.env, check=True, capture_output=True)

    def run_setup(self, cwd: Path, **env: str) -> tuple[int, str, str]:
        """Run the recipe in `cwd` as just would; its exit status, stdout and stderr."""
        assert BASH is not None and GIT is not None
        script = self.root / "setup.sh"
        script.write_text(setup_script(str(self.base_python)), encoding="utf-8")
        # The justfile puts .venv/bin first. The rest is minimal, so no real ffmpeg, python or
        # pre-commit on the developer's PATH takes part.
        path = [cwd / ".venv" / "bin", Path(GIT).parent, Path("/usr/bin"), Path("/bin")]
        result = subprocess.run(
            [BASH, str(script)],
            cwd=cwd,
            env={**self.env, "PATH": os.pathsep.join(map(str, path)), **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr

    def calls(self) -> list[str]:
        """The logged calls of the fake interpreters and pre-commit so far."""
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()

    def test_main_checkout_installs_hooks_that_tolerate_a_missing_config(self) -> None:
        # Older branches have no .pre-commit-config.yaml; their commits must not fail.
        status, out, err = self.run_setup(self.checkout)
        self.assertEqual(status, 0, err)
        self.assertIn("pre-commit install --install-hooks --allow-missing-config", self.calls())
        self.assertIn("Ready: hooks installed", out)

    def test_linked_worktree_leaves_the_shared_hooks_alone(self) -> None:
        # The hooks directory is shared by every worktree; hooks that run this worktree's .venv
        # would break every commit once the worktree is removed.
        worktree = self.root / "worktree"
        self.git(self.checkout, "worktree", "add", "--quiet", "--detach", str(worktree))
        status, out, err = self.run_setup(worktree)
        self.assertEqual(status, 0, err)
        calls = self.calls()
        self.assertIn("pre-commit install-hooks", calls)  # environments only
        self.assertEqual([call for call in calls if call.startswith("pre-commit install ")], [])
        self.assertIn("linked git worktree", err)
        self.assertIn(f"main checkout ({self.checkout})", err)
        self.assertIn("Ready, without git hooks", out)

    def test_a_venv_without_pip_is_recreated(self) -> None:
        # What a failed `python3 -m venv .venv` leaves behind on Ubuntu without python3-venv.
        venv_bin = self.checkout / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        shutil.copy2(self.root / "templates" / "python", venv_bin / "python")
        status, _, err = self.run_setup(self.checkout)
        self.assertEqual(status, 0, err)
        self.assertIn("removing .venv", err)
        self.assertIn("base-python -m venv .venv", self.calls())
        self.assertTrue((venv_bin / "pip-installed").exists())

    def test_a_failed_venv_is_removed_and_the_rerun_succeeds(self) -> None:
        status, _, err = self.run_setup(self.checkout, FAKE_ENSUREPIP_MISSING="1")
        self.assertEqual(status, 1, err)
        self.assertIn("sudo apt-get install python3-venv", err)
        self.assertFalse((self.checkout / ".venv").exists())
        status, _, err = self.run_setup(self.checkout)  # after installing python3-venv
        self.assertEqual(status, 0, err)

    def test_recipe_body_reads_one_recipe(self) -> None:
        text = (
            "# comment\n"
            "[group('x')]\n"
            'first arg="a":\n'
            "    #!/usr/bin/env bash\n"
            "    echo {{ arg }}\n"
            "\n"
            "        indented more\n"
            "second: first\n"
            "    echo second\n"
            "export PATH := 'x'\n"
        )
        self.assertEqual(
            recipe_body(text, "first"), "#!/usr/bin/env bash\necho {{ arg }}\n\n    indented more\n"
        )
        self.assertEqual(recipe_body(text, "second"), "echo second\n")
        with self.assertRaises(ValueError):
            recipe_body(text, "PATH")


# A test module that passes only while `other` runs at the same time: each one announces itself
# with a file, then waits for the other's.
RENDEZVOUS_MODULE = """\
import time
import unittest
from pathlib import Path


class Rendezvous(unittest.TestCase):
    def test_meets_the_other_module(self):
        here = Path(__file__).resolve().parent
        (here / "{me}.arrived").touch()
        deadline = time.monotonic() + 30
        while not (here / "{other}.arrived").exists():
            if time.monotonic() > deadline:
                self.fail("{other} never ran while {me} ran")
            time.sleep(0.01)
        print("{me} met {other}")
"""
FAILING_MODULE = """\
import unittest


class Failing(unittest.TestCase):
    def test_fails(self):
        self.fail("a deliberate failure")
"""
PASSING_MODULE = """\
import unittest


class Passing(unittest.TestCase):
    def test_passes(self):
        pass
"""


class ParallelRunnerTests(unittest.TestCase):
    """run_parallel.py runs the modules concurrently and fails when any of them fails."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write_module(self, name: str, source: str) -> None:
        (self.dir / f"{name}.py").write_text(source, encoding="utf-8")

    def run_runner(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        status = run_parallel.main(["--start-dir", str(self.dir), *args], out=out)
        return status, out.getvalue()

    def test_modules_run_concurrently_and_one_failure_fails_the_run(self) -> None:
        self.write_module("test_a", RENDEZVOUS_MODULE.format(me="a", other="b"))
        self.write_module("test_b", RENDEZVOUS_MODULE.format(me="b", other="a"))
        self.write_module("test_c", FAILING_MODULE)
        self.write_module("helper", "raise SystemExit('not a test module')\n")
        status, output = self.run_runner()
        self.assertEqual(status, run_parallel.EXIT_FAILED, output)
        self.assertIn("Running 3 test module(s), 3 at a time", output)
        self.assertIn("===== test_c: FAILED (exit status 1)", output)
        self.assertIn("a deliberate failure", output)
        self.assertIn("FAILED: test_c (3 test module(s)", output)
        # Each module's output forms one block under its own header.
        for me, other in (("a", "b"), ("b", "a")):
            with self.subTest(module=me):
                block = output.split(f"===== test_{me}: ok")[1].split("=====")[0]
                self.assertIn(f"{me} met {other}", block)
                self.assertIn("OK", block)

    def test_passing_modules_exit_zero_and_verbose_lists_each_test(self) -> None:
        self.write_module("test_ok", PASSING_MODULE)
        status, output = self.run_runner("--verbose")
        self.assertEqual(status, run_parallel.EXIT_OK, output)
        self.assertRegex(output, r"test_passes .*\.\.\. ok")
        self.assertIn("OK: 1 test module(s)", output)

    def test_no_test_modules_is_a_usage_error(self) -> None:
        self.write_module("helper", PASSING_MODULE)
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            status, _ = self.run_runner()
        self.assertEqual(status, run_parallel.EXIT_USAGE)
        self.assertIn("error: no test_*.py in", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
