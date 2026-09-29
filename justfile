# Development task runner for the three_body_problem generator and its Python scripts.
# Install just 1.27 or newer (https://just.systems; the recipe groups need it):
# `brew install just` or `cargo install --locked just`. `just` lists the recipes; `just setup`
# prepares a fresh clone (CONTRIBUTING.md).

set shell := ["bash", "-euo", "pipefail", "-c"]

# The project virtualenv made by `just setup` comes first on PATH, so every recipe runs the
# pinned ruff, mypy and pre-commit from pyproject.toml's dev extra without activating it.
export PATH := justfile_directory() / ".venv" / "bin" + ":" + env_var("PATH")

# List the recipes
[private]
default:
    @just --list --unsorted

# ---------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------

# Create .venv with the pinned dev tools and, in the main checkout, install the git hooks
[group('setup')]
setup python="python3":
    #!/usr/bin/env bash
    set -euo pipefail
    is_310_plus() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }
    # An existing .venv is reused, so it must be new enough too (PATH puts it first, so a bare
    # `python3` here may already be the .venv's interpreter).
    if [ -x .venv/bin/python ] && ! is_310_plus .venv/bin/python; then
        echo "error: .venv runs a Python older than 3.10; remove it (rm -rf .venv) and rerun" >&2
        exit 1
    fi
    if ! is_310_plus "{{ python }}"; then
        # macOS's /usr/bin/python3 (Xcode command line tools) is 3.9.
        echo "error: the scripts and their tooling need Python 3.10+, and '{{ python }}' is older" \
            "or missing; name a newer interpreter, e.g. 'just setup python3.12'" >&2
        exit 1
    fi
    # A .venv without a working pip is recreated rather than reused: `python3 -m venv` on
    # Debian or Ubuntu without python3-venv leaves the directory and bin/python behind and
    # fails at ensurepip, so every rerun would fail again at the pip install below.
    if [ -d .venv ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
        echo "note: removing .venv, which has no working pip (an interrupted or failed setup?)" >&2
        rm -rf .venv
    fi
    if [ ! -d .venv ] && ! "{{ python }}" -m venv .venv; then
        rm -rf .venv
        echo "error: '{{ python }} -m venv' failed; on Debian or Ubuntu, install the venv module" \
            "(sudo apt-get install python3-venv) and rerun 'just setup'" >&2
        exit 1
    fi
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -e ".[dev]"
    # Git hooks belong to the repository, not to a worktree: pre-commit installs them in the
    # hooks directory that every worktree of this clone shares, and they run the pre-commit of
    # the .venv that installed them. Installed from a linked worktree, they would break every
    # commit once that worktree (and its .venv) is removed, so only the main checkout installs
    # them. --allow-missing-config lets the hooks pass, with a note, on a branch that predates
    # .pre-commit-config.yaml instead of failing every commit there.
    git_dir="$(git rev-parse --path-format=absolute --git-dir)"
    common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
    if [ "$git_dir" != "$common_dir" ]; then
        # The hook environments live in ~/.cache/pre-commit, shared by every checkout, so
        # installing them here still makes the first `just lint` fast.
        .venv/bin/pre-commit install-hooks
        main_checkout="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
        echo "note: this is a linked git worktree. Git hooks are shared by every worktree of the" \
            "clone and would run this worktree's .venv, so they were not installed; run" \
            "'just setup' in the main checkout (${main_checkout:-see 'git worktree list'})" \
            "to install them for every worktree." >&2
        hooks_installed=false
    else
        # The pre-commit framework replaced the hand-written .githooks/ directory, and
        # pre-commit refuses to install while core.hooksPath is set, so drop this clone's old
        # setting.
        case "$(git config --local --get core.hooksPath || true)" in
            .githooks | .githooks/ | "$PWD/.githooks" | "$PWD/.githooks/")
                git config --local --unset core.hooksPath
                ;;
        esac
        .venv/bin/pre-commit install --install-hooks --allow-missing-config
        hooks_installed=true
    fi
    # The FFmpeg-dependent tests skip locally without ffmpeg, and fail without these encoders.
    # The listing is captured first: `grep -q` at the end of a pipe can exit before ffmpeg has
    # written everything, and pipefail would then report the SIGPIPE as "not found".
    if ! command -v ffmpeg >/dev/null; then
        echo "note: ffmpeg is not installed; the video and package tests will skip (CONTRIBUTING.md)"
    else
        encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
        for encoder in libwebp libx264 libx265; do
            if ! awk -v want="$encoder" '$2 == want { found = 1 } END { exit !found }' \
                <<<"$encoders"; then
                echo "note: this ffmpeg has no $encoder encoder; the package tests will fail" \
                    "(CONTRIBUTING.md)"
            fi
        done
    fi
    if [ "$hooks_installed" = true ]; then
        echo "Ready: hooks installed for commit, commit-msg and push. Run 'just ci' before a PR."
    else
        echo "Ready, without git hooks (see the note above). Run 'just ci' before a PR."
    fi

# Bump every hook to its latest release (frozen SHAs), then check the pins pyproject.toml shares
[group('setup')]
hooks-update:
    pre-commit autoupdate --freeze
    @echo "Hooks updated. The ruff and mypy revs must equal the ruff==/mypy== pins in pyproject.toml's"
    @echo "dev extra, and actionlint's shellcheck-py== pin the shellcheck-py release: update them to"
    @echo "match, run 'just setup' to reinstall, then 'just lint'. Checking the pins now:"
    python3 -m unittest discover -s tests/python -p test_tooling.py

# ---------------------------------------------------------------------------------------------
# Checks (CI runs the same commands; see .github/workflows/ci.yml)
# ---------------------------------------------------------------------------------------------

# Run every pre-commit hook on every file, or only the named hook (e.g. `just lint mypy`)
[group('check')]
lint *hook:
    pre-commit run --all-files {{ hook }}

# What CI gates a merge on, on this machine (CI adds coverage and the OS/CPU matrix)
[group('check')]
ci: lint test py-test doc-check bench-check
    #!/usr/bin/env bash
    set -euo pipefail
    # cargo-deny is an optional local install; CI always runs it.
    if cargo deny --version >/dev/null 2>&1; then
        cargo deny --locked check
    else
        echo "note: skipped the dependency policy check (cargo install --locked cargo-deny)"
    fi

# Run all checks and tests (same as `ci`)
[group('check')]
all: ci

# Rust formatting and clippy
[group('check')]
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --locked -- -D warnings

# Clippy on every target, warnings as errors
[group('check')]
clippy:
    cargo clippy --all-targets --locked -- -D warnings

# Python: ruff format check, ruff lint and strict mypy (the versions pinned in pyproject.toml)
[group('check')]
py-check:
    ruff format --check .
    ruff check .
    mypy

# Dependency policy in deny.toml: advisories, licenses, bans, sources (needs cargo-deny)
[group('check')]
deny:
    cargo deny --locked check

# RustSec advisories only (the advisories part of `just deny`)
[group('check')]
audit:
    cargo deny --locked check advisories

# Build the API docs as CI does (broken intra-doc links are errors)
[group('check')]
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --locked

# Compile the benchmarks without running them
[group('check')]
bench-check:
    cargo bench --no-run --locked

# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------

# Run the Rust test suite in release mode (the FFmpeg tests skip without ffmpeg on PATH)
[group('test')]
test:
    cargo test --release --locked

# Run the Python unit tests (stdlib unittest, no network), one process per module in parallel
[group('test')]
py-test:
    python3 tests/python/run_parallel.py -v

# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------

# Format all code: rustfmt; ruff's safe autofixes (import order, ...) then its formatter
[group('format')]
fmt:
    cargo fmt --all
    ruff check --fix --exit-zero --quiet .
    ruff format .

# Format the Python scripts only
[group('format')]
py-fmt:
    ruff format .

# ---------------------------------------------------------------------------------------------
# Build, docs, benchmarks and look development
# ---------------------------------------------------------------------------------------------

# Build the release binary (target/release/three_body_problem)
[group('build')]
build:
    cargo build --release --locked

# Build the API docs and open them in a browser
[group('build')]
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --locked --open

# Run the benchmarks
[group('build')]
bench:
    cargo bench --locked

# Render a random-seed contact sheet for fast visual curation
[group('look')]
contact-sheet count="24":
    python3 contact_sheet.py --count {{ count }}

# Re-render the fixed golden seed set into golden_gallery.png for look regression
[group('look')]
golden-gallery:
    python3 contact_sheet.py --golden

# Regenerate the reference baseline image and metadata in ci/reference/
[group('look')]
reference:
    cd ci/reference && ./generate_reference.sh
