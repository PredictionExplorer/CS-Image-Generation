# Contributing

This repository holds the `three_body_problem` generator (Rust) and the stdlib-only Python
scripts around it, most importantly `run.py`, the production sync loop. **Every merge into `main`
ships to the production server automatically** once CI passes, so the local hooks, CI and review
all exist to keep `main` releasable at every commit.

- [Setup](#setup)
- [Workflow: branch, pull request, merge, deploy](#workflow-branch-pull-request-merge-deploy)
- [The hooks](#the-hooks)
- [Commit messages](#commit-messages)
- [Tests](#tests)
- [Determinism rules](#determinism-rules)
- [Dependencies and tool versions](#dependencies-and-tool-versions)
- [Security and license](#security-and-license)

## Setup

Prerequisites:

| Tool | Why | Install |
|------|-----|---------|
| Rust via [rustup](https://rustup.rs) | builds the generator; `rust-toolchain.toml` pins the toolchain, rustup fetches it on first use | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh`, plus a C toolchain for linking (Ubuntu / Debian: `sudo apt-get install build-essential`; macOS: `xcode-select --install`) |
| Python 3.10+ with `venv` | the scripts and their tooling (production runs 3.12) | your OS or [python.org](https://www.python.org); Ubuntu / Debian ship `venv` separately: `sudo apt-get install python3-venv` |
| [just](https://just.systems) 1.27+ | the task runner (`just` lists every recipe) | `brew install just` or `cargo install --locked just` (Ubuntu 24.04's apt package is too old) |
| FFmpeg with libwebp, libx264, libx265 | video and WebP encoding in the package tests | see below |
| git | version control; the hooks and the Python tests run it | your OS |

FFmpeg must encode WebP. Homebrew's own `ffmpeg` formula no longer links libwebp, so on macOS
install it from the `homebrew-ffmpeg` tap, which builds from source with the option:

```bash
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-webp   # macOS
sudo apt-get install ffmpeg                               # Ubuntu / Debian: already has all three
ffmpeg -hide_banner -encoders | grep -E 'libwebp|libx264|libx265'   # all three must be listed
```

Then, from the repository root:

```bash
just setup              # uses python3 from PATH
just setup python3.12   # or name the interpreter (macOS's /usr/bin/python3 is 3.9: too old)
```

`just setup` creates `.venv/` with the pinned development tools from `pyproject.toml`'s `dev`
extra (ruff, mypy, pre-commit), and installs the git hooks for three stages: `pre-commit`,
`commit-msg` and `pre-push`. The `justfile` puts `.venv/bin` first on `PATH`, so `just` recipes
use those tools without activating the virtualenv; activate it (`source .venv/bin/activate`) to
run them by hand. `just setup` is idempotent: rerun it after pulling a change to
`pyproject.toml` or `.pre-commit-config.yaml`. A `.venv` left without a working pip (for example
by a first run on Ubuntu before `python3-venv` was installed) is removed and created again.

Git hooks belong to the repository, not to a worktree: every `git worktree` of a clone runs the
same hooks, and they call the pre-commit of the `.venv` that installed them. So install them
from the main checkout. In a linked worktree, `just setup` creates the worktree's `.venv` but
leaves the hooks alone and says so. The hooks pass, with a note, on a branch that predates
`.pre-commit-config.yaml` (`--allow-missing-config`).

Upgrading from the old `.githooks/pre-commit` script: `just setup` removes the
`core.hooksPath = .githooks` setting it needed (pre-commit refuses to install while it is set).

## Workflow: branch, pull request, merge, deploy

1. Branch from an up-to-date `main`. Nobody pushes to `main` directly: a repository ruleset
   rejects it (no bypass, admins included), along with force pushes and deleting `main`.
2. Commit. The hooks check each commit (see [The hooks](#the-hooks)); `git push` runs the test
   suites first.
3. Open a pull request. Fill in the template, in particular the testing done and the deployment
   impact.
4. CI runs on every push to the pull request. The single required check is **`CI passed`**, a
   job that succeeds only when every other CI job did. The branch must also be up to date with
   `main` before it can merge (GitHub offers an "Update branch" button).
5. Merge with **squash** (the pull request title becomes the commit subject, so it must be a
   [Conventional Commit](#commit-messages); the commit body starts empty, since a skip
   instruction such as `[skip ci]` in it would leave the commit on `main` without CI, and so
   undeployed) or **rebase** (every commit lands as is, so each one must be clean). `main` has
   linear history; merge commits are disabled. Auto-merge is enabled:
   `gh pr merge --auto --squash` merges as soon as the checks pass. The head branch is deleted
   on merge.
6. **Merging deploys.** CI runs again on the new commit on `main`. When its `CI passed` check
   succeeds, the production server's deploy agent (it polls every two minutes) fast-forwards the
   production checkout, builds and tests the generator natively, switches to the new build
   between two sync runs, and starts a sync run, which regenerates whatever the change requires.
   A commit whose CI fails is never deployed. Pausing, rolling back and reading the logs are
   covered in [docs/deployment.md](docs/deployment.md).

Treat a merge as a release: if a change needs a manual step on the server, or regenerates
published tokens, say so in the pull request and coordinate before merging.

## The hooks

The [pre-commit](https://pre-commit.com) framework runs them, configured in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml). The same file drives CI's lint job
(`pre-commit run --all-files --show-diff-on-failure`), so passing locally means passing there.

| Stage | Hook | What it checks |
|-------|------|----------------|
| commit | `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending`, `fix-byte-order-marker` | whitespace, final newline, LF line endings, no BOM (these fix the file) |
| commit | `check-yaml`, `check-toml`, `check-json` | the data files parse |
| commit | `check-merge-conflict`, `check-case-conflict`, `check-illegal-windows-names`, `check-symlinks`, `destroyed-symlinks`, `forbid-submodules` | mistakes git lets through |
| commit | `check-executables-have-shebangs`, `check-shebang-scripts-are-executable` | a script's shebang and executable bit agree |
| commit | `check-added-large-files` | no file over 1 MiB (rendered output belongs in `output/`) |
| commit | `detect-private-key`, `debug-statements` | no private keys; no leftover `breakpoint()` or `pdb` |
| commit | `ruff-check` (with `--fix`), `ruff-format` | Python lint and formatting (`pyproject.toml`) |
| commit | `mypy` | strict typing of the files listed in `pyproject.toml` |
| commit | `cargo-fmt`, `cargo-clippy` | `cargo fmt --check`; clippy on all targets with warnings denied |
| commit | `shellcheck` | shell scripts |
| commit | `actionlint`, `zizmor` | workflow correctness (with shellcheck on every `run:` block) and workflow security (zizmor's offline audits: `--offline`, even with a GitHub token exported) |
| commit | `check-github-workflows`, `check-github-workflows-require-timeout`, `check-dependabot`, `check-github-issue-forms`, `check-github-issue-config` | GitHub configuration against its JSON schemas; every workflow job sets `timeout-minutes` |
| commit | `check-metaschema`, `check-jsonschema` | `docs/nft_traits.schema.json` is a valid schema and `docs/fixtures/nft_traits.example.json` satisfies it |
| commit | `typos` | spelling (intentional spellings are listed in [`_typos.toml`](_typos.toml)) |
| commit | `check-hooks-apply`, `check-useless-excludes` | the hook configuration itself has not gone stale |
| commit-msg | `conventional-pre-commit` | the message is a [Conventional Commit](#commit-messages) |
| push | `python-unittest`, `cargo-test` | the Python tests, each `tests/python/test_*.py` in its own process, in parallel (`tests/python/run_parallel.py`, on the `.venv` interpreter); `cargo test --release --locked` |
| manual | `zizmor-online` | zizmor with its online audits too (a pinned SHA belongs to its action and matches its version comment; no pinned release has a known vulnerability). CI's lint job runs it; locally: `GH_TOKEN=$(gh auth token) pre-commit run --hook-stage manual zizmor-online --all-files` |

Commit-stage hooks only look at the staged files of their type, so most commits take seconds;
the cargo hooks run only when Rust code or its configuration changed. Hooks that fix files
(the whitespace fixers, `ruff-check --fix`, `ruff-format`) still fail the commit: review the
change, `git add` it and commit again. At push time the whitespace, large-file, executable-bit
and spelling checks also re-check the files of the pushed commits, which catches a commit made
with `--no-verify`. The first push builds the optimized test binaries (with the release
profile's fat LTO, so a minute or more); later pushes reuse the build cache, and the Rust suite
then takes seconds. The Python suite takes about a minute: the hook runs its modules side by
side, so a push waits only as long as the slowest one (`test_deploy.py`, whose fake commands and
temporary git repositories start many short-lived processes). Each module's output is printed
in one piece when it finishes, followed by a summary.

Useful commands:

```bash
just lint                     # every hook on every file (what CI's lint job does)
just lint mypy                # one hook, every file
pre-commit run --files a.py   # every hook, some files
just ci                       # hooks + Rust and Python tests + docs + benchmark build + cargo-deny
SKIP=cargo-clippy git commit  # skip one hook for one commit
```

`git commit --no-verify` and `git push --no-verify` skip the hooks entirely. Keep that for
emergencies: CI runs every check again and blocks the merge anyway.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
`type(optional scope)!: subject`, the subject in the imperative and lowercase, as in this
history:

```text
feat: add the ember edition to every package and backfill published tokens
fix: make the AVX2 exp match libm and skip NaN bins in the scalar path
ci: build and test the AVX2 spectral kernel
docs: correct the augur-explorer trait URL contract
```

| Type | Use for |
|------|---------|
| `feat` | a new capability of the generator, the scripts or the deployment |
| `fix` | a bug fix |
| `perf` | a speed-up with no change in behaviour |
| `refactor` | restructuring with no change in behaviour |
| `test` | tests only |
| `docs` | documentation only |
| `style` | formatting only |
| `build` | the build, Cargo or pyproject configuration; `build(deps)` for dependency bumps |
| `ci` | workflows and CI configuration; `ci(deps)` for action bumps |
| `deps` | a dependency update (older history uses it; prefer `build(deps)` or `ci(deps)`) |
| `chore` | maintenance that fits nothing above (tooling, hooks, repository settings) |
| `revert` | reverting an earlier commit: `revert: <subject of the reverted commit>` |

Mark a breaking change with `!` after the type or scope (`feat(nft)!: rename ...`) and explain it
in a `BREAKING CHANGE:` footer. This applies to anything a consumer depends on: the package
layout, `nft_traits.json` (versioned by its own `schema_version`), the CLI, `run.py`'s
environment variables, and rendered bits (see [Determinism rules](#determinism-rules)).
`fixup!` and `squash!` commits (`git commit --fixup`) pass the hook, and so do merge commits.
`git revert`'s default `Revert "..."` message does not: reword it to `revert: ...`. CI checks
pull request titles against the same types, since a squash merge turns the title into the commit
subject; it also accepts the `Revert "..."` title of GitHub's Revert button, and rejects a title
containing `[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]` or `[actions skip]` (in any
letter case), which would stop CI, and so the deploy, on the merged commit. To add a type, add it
to both lists (the hook's `args` and the `types=` of the title check in
`.github/workflows/ci.yml`); `tests/python/test_tooling.py` fails until they match.

## Tests

```bash
just test      # Rust: cargo test --release --locked (unit, integration and golden tests)
just py-test   # Python: every tests/python/test_*.py in parallel (tests/python/run_parallel.py -v)
just ci        # everything CI gates a merge on that runs locally
```

- **Python tests** (`tests/python/`) use `unittest` from the standard library and no network.
  External programs (the generator, `ssh`, `scp`, `cargo`, `systemctl`) are replaced by fakes
  on `PATH` that record their calls, and the GitHub API by a local HTTP server; git runs for
  real on temporary repositories. Follow `tests/python/test_run.py` when adding a fake.
  Every Python file is listed in `[tool.mypy] files` in `pyproject.toml` and passes `mypy
  --strict`. Runtime code stays standard-library only: the production server installs nothing
  from PyPI.
- **FFmpeg-dependent tests** (a full package, the ember videos) skip with a message when
  `ffmpeg` is not on `PATH`. Under CI (`CI` set in the environment) they fail instead, so a
  runner without FFmpeg cannot pass silently. Before touching encoding or the package layout,
  install FFmpeg locally so these tests run.
- **CI** runs the Rust suite on Linux x86_64, Linux aarch64 and macOS aarch64, and once more
  built for AVX2/FMA (`x86-64-v3`); the Python suite on 3.10, 3.12 and 3.13. See
  [`ci/README.md`](ci/README.md).

## Determinism rules

Every piece of art is a deterministic function of its on-chain seed: the same seed must render
the same picture, and the production server regenerates published tokens from their seeds. The
ember edition goes further: its raw frames and still are **bit-identical across CPU
architectures** and thread counts, and a certificate records their digests (encoded MP4 and WebP
bytes are outside that guarantee). The contract is in
[docs/ember-design.md](docs/ember-design.md) §0.

- **`libm` is pinned exactly** (`libm = "=0.2.16"` in `Cargo.toml`) and Dependabot ignores it:
  a bump can change results in the last bit. Bump it by hand, as an algorithm change.
- **Golden digests** pin small complete computations (the ember unit goldens in
  `src/ember/*.rs`) and a small end-to-end render (`tests/ember_determinism.rs`). A digest that
  changes on one architecture only is a determinism bug to fix, never to re-bless. A digest that
  changes everywhere is an algorithm change: bump `ember::certificate::ALGORITHM_VERSION`,
  re-bless every affected golden with the procedure in
  [docs/ember-design.md §0.2](docs/ember-design.md#02-golden-canaries), and say so in the pull
  request. Never re-bless to make an unexplained failure go away.
- **Portable arithmetic** in the ember modules and the orbit integrator
  ([docs/ember-design.md §0.1](docs/ember-design.md#01-bit-identical-on-every-cpu)):
  transcendentals only through `crate::ember::math` (never the `f64`/`f32` std methods, which
  call the platform's C library, and never `mul_add`), no architecture-specific code paths,
  no float reductions across threads, no `HashMap` iteration that affects output. Unit tests
  scan the ember sources for the forbidden calls; parallel modules have 1-thread versus
  3-thread parity tests.
- The spectral master's reference image (`ci/reference/`, `just reference`) and the golden
  gallery (`just golden-gallery`, seeds in `ci/golden_seeds.txt`) are for look regressions:
  regenerate and compare them side by side after a tuning change.

## Dependencies and tool versions

- **Dependabot** opens weekly pull requests for Cargo crates, the Rust toolchain, GitHub
  Actions and the Python tooling ([`.github/dependabot.yml`](.github/dependabot.yml)). They go
  through CI and review like any other change, and deploy when merged.
- **Hook revisions** are frozen to commit SHAs with the release in a `# frozen:` comment.
  `just hooks-update` bumps them all (`pre-commit autoupdate --freeze`); Dependabot's
  `pre-commit` ecosystem does the same weekly.
- **Pins that move together**: the ruff and mypy hook revisions must equal the `ruff==` and
  `mypy==` pins in `pyproject.toml`, and actionlint's `shellcheck-py==` pin must equal the
  shellcheck hook release. `tests/python/test_tooling.py` fails CI until they agree. Dependabot
  updates the pip pins and the hooks in one pull request for that reason; if a side is still
  behind, push the matching change onto the same branch.
- **Rust toolchain**: `rust-toolchain.toml` pins the toolchain everyone builds with (CI, the
  hooks and the production server); `rust-version` in `Cargo.toml` records the minimum
  supported version.
- **Cargo.lock** is committed and every build and test uses `--locked`. `cargo-deny`
  (`deny.toml`, `just deny`) checks advisories, licenses, banned and duplicate crates, and
  crate sources.

## Security and license

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), never in a public
issue. Never commit secrets: deployment settings belong in the untracked `.env` (template:
[`.env.example`](.env.example)), and the `detect-private-key` hook and GitHub's secret scanning
with push protection back that up.

The project is dedicated to the public domain under [CC0 1.0 Universal](LICENSE). By
contributing, you dedicate your contribution under the same terms.
