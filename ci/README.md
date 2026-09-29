# CI and supply chain

This directory holds the manual reference-image tooling. This page also documents the GitHub
Actions pipeline and the supply-chain configuration around it:

| File | Role |
|------|------|
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | The CI pipeline: lint, tests, docs, dependency policy, and the `CI passed` gate |
| [`.github/workflows/scorecard.yml`](../.github/workflows/scorecard.yml) | OpenSSF Scorecard analysis |
| [`.github/dependabot.yml`](../.github/dependabot.yml) | Dependency and action version updates |
| [`deny.toml`](../deny.toml) | cargo-deny policy: advisories, licenses, bans, sources |
| [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) | The file-level checks, which run locally and in CI's lint job |
| `ci/reference/`, `ci/verify_reference.py` | Manual reference-image regression check (below) |

## The `CI passed` gate

Every job in `ci.yml` feeds one final job, **`CI passed`**. It runs even when a job it depends on
failed or the run was cancelled, and it fails unless every one of those jobs succeeded. It is the
one check that matters outside the workflow:

- the `main` ruleset requires it before a pull request can merge (branches must also be up to
  date with `main`);
- the production deploy agent (`ops/deploy/cosmicsig_deploy.py`, see
  [`docs/deployment.md`](../docs/deployment.md)) deploys a commit on `main` only when that
  commit's latest `CI passed` check run, from GitHub Actions, concluded with success.

So a green `CI passed` on a pull request means it can merge, and merging it deploys it. To add a
job, add it to `ci-passed.needs` as well; a job that is missing there gates neither merges nor
deploys. Never give a job a job-level `if:` that can skip it: `CI passed` treats a skipped job as
a failure.

## Jobs

| Job | What it checks |
|-----|----------------|
| **Lint (pre-commit)** | Every hook in `.pre-commit-config.yaml` on every file: whitespace, line endings, YAML/TOML/JSON syntax, large files, private keys, ruff (lint and format), strict mypy, rustfmt, shellcheck, actionlint, zizmor (offline, as at commit time), JSON Schema validation (workflows, Dependabot, issue forms), typos. Then zizmor again with its online audits (the manual-stage `zizmor-online` hook, with the run's read-only token), and on pull requests the title check (below); both run even when a hook failed, so one run reports every problem. clippy is skipped here (`SKIP=cargo-clippy`) because it has its own jobs. |
| **Python tests (3.10, 3.12, 3.13)** | `python -m unittest discover -s tests/python -v`: `run.py`'s sync policy and the deploy agent, with fake generator, `ssh`, `scp`, `cargo` and `systemctl` commands on `PATH` and a fake GitHub API. Standard library only, no network. 3.10 is the floor in `pyproject.toml`; the production server runs 3.12. |
| **Clippy** | `cargo clippy --all-targets --locked -- -D warnings` (the crate enables `clippy::pedantic`, see `Cargo.toml`). |
| **Tests (Linux x86-64, coverage)** | The Rust test suite under `cargo llvm-cov` (doctests excepted: covering them needs a nightly toolchain; the other three entries run them); the LCOV report is uploaded as the `coverage-lcov` artifact. Baseline x86-64, so the scalar spectral fallback. |
| **Tests (Linux x86-64, AVX2 + FMA)** | Production parity: `cargo test --release --locked` (the deploy agent's command) with `-C target-cpu=x86-64-v3 -C target-feature=+avx2,+fma`, so the AVX2 kernel in `src/spectrum_simd.rs` that production x86 builds use is compiled, linted with clippy and tested. |
| **Tests (Linux aarch64)** | `cargo test --release --locked` on `ubuntu-24.04-arm`: the ember golden digests on Linux ARM (NEON). |
| **Tests (macOS aarch64)** | `cargo test --release --locked` on `macos-26` (Apple Silicon, NEON). |
| **Benchmarks (build only)** | `cargo bench --no-run --locked`. |
| **Documentation** | `cargo doc --no-deps --document-private-items --locked` with `RUSTDOCFLAGS=-D warnings`. |
| **Supply chain (cargo-deny)** | `cargo deny --locked check` against `deny.toml`. |
| **CI passed** | The gate described above. |

Every Rust job passes `--locked`, so a `Cargo.lock` that is out of date with `Cargo.toml` fails
CI instead of being rewritten on the runner. The test suite includes the ember edition's golden
digests (`tests/ember_determinism.rs` and the unit goldens in `src/ember/`), which must come out
bit-identical on all four test entries (two CPU architectures, two operating systems, with and
without AVX2/FMA code generation); see [`docs/ember-design.md`](../docs/ember-design.md).

### Build flags

The workflow sets `RUSTFLAGS="-D warnings"`. Setting `RUSTFLAGS` replaces the per-target flags in
`.cargo/config.toml` (`-C target-cpu=native`), so CI builds target each architecture's baseline:
the builds run on any runner CPU and rust-cache can restore them anywhere. The AVX2 test entry
sets its own `RUSTFLAGS` (x86-64-v3 plus AVX2 and FMA), which stands in for `target-cpu=native` on
the production server's CPU while staying valid on whatever runner restores the cache. A step in
that entry checks that the flags really enable `avx2` and `fma`.

### FFmpeg

The tests that render a package or encode the web images and videos need `ffmpeg` and `ffprobe`
with the **libwebp**, **libx264** and **libx265** encoders. Locally they skip when `ffmpeg` is not
on `PATH`. GitHub Actions sets `CI=true`, under which they fail instead, so a runner without FFmpeg
cannot pass silently. Every test entry installs FFmpeg and then checks that it has those three
encoders, failing early with a clear message if one is missing:

- **Linux**: Ubuntu's `ffmpeg` package (`apt-get install --no-install-recommends ffmpeg`) has all
  three on both architectures.
- **macOS**: Homebrew's own `ffmpeg` formula no longer links libwebp, so CI builds FFmpeg from the
  [homebrew-ffmpeg tap](https://github.com/homebrew-ffmpeg/homebrew-ffmpeg) with `--with-webp`
  (its defaults already include x264 and x265). The source build takes about ten minutes, so the
  finished keg is cached, but only once it has passed the encoder check. The cache key covers the
  tap's formula and the versions of the formulae in its runtime dependency closure (not every
  formula on the runner image). Homebrew updates itself at most once, when the tap is added and
  the dependencies are installed, and not again while the key is computed and FFmpeg is built.
  Without an exact match, the most recent keg of the same macOS image and architecture is
  restored (`restore-keys`); if it runs and lists the three encoders it is used, and on `main`
  saved under the new key. So a dependency update that leaves the keg working costs no rebuild,
  and cannot turn CI red while the tap fails to build against the new version. A new formula
  release is treated the same way: macOS CI keeps the FFmpeg it has until that keg stops
  working. A restored keg that does not run or lacks an encoder (a library it links was
  replaced by an incompatible one) is rebuilt from the current formula, with a warning.
  Developers on macOS install FFmpeg the same way:
  `brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-webp`.

### Pull request titles

A squash merge uses the pull request title as the commit subject on `main`, so on pull requests
the lint job checks that the title is a
[Conventional Commit](https://www.conventionalcommits.org/) subject: `type(scope)!: subject`
with one of the types the commit-msg hook accepts (`feat`, `fix`, `docs`, `ci`, `build`, `chore`,
`refactor`, `perf`, `test`, `style`, `revert`, `deps`); the scope and `!` are optional.
`Revert "..."` titles, which GitHub's revert button creates, pass too. A title containing
`[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]` or `[actions skip]`, in any letter case,
fails: GitHub starts no push run for a commit whose message holds one, so the squash commit on
`main` would never get its `CI passed` check and the deploy agent would never deploy it. (For
the same reason, squash commits get a blank body rather than the pull request's description;
see [`ops/github/README.md`](../ops/github/README.md).) The job reads the current title through
the API, so after fixing a title, re-run the failed jobs: editing a title alone does not start a
new run.

## Triggers and concurrency

The pipeline runs on every pull request, on every push to `main`, weekly (Monday 04:23 UTC) and on
demand (`workflow_dispatch`). The weekly run catches breakage that no commit causes: a new RustSec
advisory against `Cargo.lock`, a yanked crate, a runner image or Homebrew change.

A new push to a pull request cancels the run it supersedes. Runs for `main` are never cancelled or
queued behind one another: each commit on `main` gets its own `CI passed` result.

## Caching

- **Rust builds**: [rust-cache](https://github.com/Swatinem/rust-cache), one cache per job and
  toolchain/flags combination. Caches are saved only from `main`; pull requests restore `main`'s.
  A branch cannot restore another branch's caches, so per-branch saves would only evict `main`'s.
- **pre-commit hook environments**: keyed on `.pre-commit-config.yaml`, so a hook update rebuilds
  them.
- **pip**: `actions/setup-python`'s cache, keyed on `pyproject.toml`.
- **FFmpeg on macOS**: see above; saved from `main` only.

A stale or poisoned cache can be deleted under *Actions > Caches* or with
`gh cache delete <key>`.

## Supply chain

### Actions

- Every action is pinned to a full 40-character commit SHA, with its release in a trailing
  comment (`uses: actions/checkout@<sha> # v7.0.1`). A tag can be moved to other code; a SHA
  cannot. Dependabot updates the SHA and the comment together.
- The workflow token is read-only (`permissions: contents: read` at the top), and a job asks for
  more only when it needs it (the lint job reads pull requests; the Scorecard job uploads SARIF and
  signs its results). The aggregate job has no permissions at all.
- Every checkout sets `persist-credentials: false`, so no step can push with the token.
- No `${{ }}` expression is expanded inside a `run:` script: values reach scripts through `env:`,
  which rules out script injection from branch names or titles.
- Tools installed by [install-action](https://github.com/taiki-e/install-action) (cargo-deny,
  cargo-llvm-cov) are the versions that the pinned install-action release lists, verified by
  checksum, with the unverified `cargo-binstall` fallback disabled.
- [zizmor](https://docs.zizmor.sh) and [actionlint](https://github.com/rhysd/actionlint) (with
  shellcheck on every `run:` block) check the workflows in the lint job and in the local
  pre-commit hooks. The commit-time zizmor hook runs offline (`--offline`, so a commit never
  depends on reaching GitHub); the lint job also runs its online audits through the manual-stage
  `zizmor-online` hook.

### Rust dependencies: cargo-deny

[`deny.toml`](../deny.toml) (cargo-deny's version 2 schema) holds the policy. Each setting carries
a comment with its reason. In short:

- **advisories**: every RustSec advisory is an error: vulnerable, unsound and unmaintained crates
  anywhere in the graph, and yanked versions in `Cargo.lock`. Accepting one takes an `ignore`
  entry with its reason. There is one today: `paste` (RUSTSEC-2024-0436, unmaintained), a
  compile-time proc macro that `nalgebra` pulls in through `simba`. cargo-deny warns about an
  entry that no longer matches any crate; remove it then.
- **licenses**: a list of accepted permissive licenses. This crate itself is `CC0-1.0` and is
  checked too, so its `license` field in `Cargo.toml` must stay present and valid.
- **bans**: no wildcard version requirements; duplicate versions are reported; crates that run at
  build time may not ship prebuilt executables.
- **sources**: crates.io only.

The graph is limited to the targets we build for (x86-64 and aarch64 Linux, both macOS
architectures). Run it locally with `just deny` (after `cargo install --locked cargo-deny`).

### Dependabot

[`.github/dependabot.yml`](../.github/dependabot.yml) proposes version updates weekly (Monday
06:00 UTC) for Cargo dependencies, the Rust toolchain (`rust-toolchain.toml`), GitHub Actions, and
the Python developer tools. Each release must be 7 days old (cooldown) before it is proposed.
Minor and patch updates arrive as one grouped pull request per ecosystem; majors come one pull
request each. The Python tools are pinned in two places, `pyproject.toml`'s `dev` extra and the
hook revisions in `.pre-commit-config.yaml`, which must agree (`tests/python/test_tooling.py`),
so pip and pre-commit updates arrive together in a single `python-tooling` pull request. `libm`
is excluded: it is pinned exactly for the ember edition's determinism and is only upgraded by
hand, together with re-blessed goldens. Commit subjects follow Conventional Commits:
`build(deps)` / `build(deps-dev)` for dependencies and toolchains, `ci(deps)` for actions.

Security updates are separate: GitHub opens them as soon as an advisory affects a dependency,
without cooldown.

Merging a Dependabot pull request deploys it to production once CI passes, like any other merge.

### OpenSSF Scorecard

[`scorecard.yml`](../.github/workflows/scorecard.yml) runs
[OpenSSF Scorecard](https://scorecard.dev) on every push to `main`, weekly, and when classic
branch protection changes. It uploads the results to *Security > Code scanning* and publishes
them to the public Scorecard API (the source of a Scorecard badge). Publishing restricts that
workflow's shape; its header comment lists the rules. CodeQL is not a workflow here: it runs
through GitHub's CodeQL default setup, which `ops/github/apply-settings.sh` enables.

## Running the same checks locally

With [just](https://just.systems) and a one-time `just setup` (see
[`CONTRIBUTING.md`](../CONTRIBUTING.md)):

```bash
just lint     # every pre-commit hook on every file (as the lint job, plus clippy; zizmor offline)
just ci       # lint + the Rust and Python test suites + docs + benchmarks + cargo-deny
just deny     # cargo-deny alone
```

`just setup` also installs the git hooks, so the file checks run on every commit and the test
suites on every push. Without `just`:

```bash
pre-commit run --all-files
GH_TOKEN="$(gh auth token)" pre-commit run --hook-stage manual zizmor-online --all-files
cargo clippy --all-targets --locked -- -D warnings
cargo test --release --locked
python3 -m unittest discover -s tests/python -v
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --locked
cargo deny --locked check
```

The FFmpeg-dependent tests skip locally without `ffmpeg` on `PATH`. To run them the way CI does,
install an FFmpeg with libwebp (see above) and set `CI=true`, which makes a missing FFmpeg fail
instead of skip.

## Troubleshooting

- **`CI passed` failed**: open the run; the job's log lists the result of every job it depends
  on. Fix the failing job; `CI passed` follows.
- **Lint failed with a diff**: a fixer hook (whitespace, end of file, ruff) would change a file.
  Run `just lint` (or `pre-commit run --all-files`) locally, commit what it changed, push.
- **Pull request title check failed**: edit the title, then *Re-run failed jobs* (editing a
  title does not start a run by itself).
- **No CI run at all for a commit on `main`**: GitHub skips the push run of a commit whose
  message holds a skip instruction such as `[skip ci]` (a squash message edited by hand in the
  merge box, say), and the deploy agent then waits for its `CI passed`. Start a run for the tip
  of `main` with `gh workflow run ci.yml --ref main`.
- **cargo-deny failed on an advisory**: update the affected crate (`cargo update -p <crate>`). If
  no fixed version exists and the advisory does not apply, add an `ignore` entry with its id and
  the reason to `deny.toml`.
- **macOS FFmpeg problems**: the step's log shows whether it used the cache or built FFmpeg. To
  force a rebuild, delete the `ffmpeg-webp-*` caches under *Actions > Caches*.
- **A test passes locally but fails in CI**: CI runs with `CI=true` (FFmpeg tests fail instead of
  skip) and without `target-cpu=native`; reproduce with
  `CI=true RUSTFLAGS="-D warnings" cargo test --release --locked`.

## Reference images

The reference image is a manual deterministic-output check for the main (non-ember) master; CI
does not run it. The ember edition carries its own cross-architecture digests in
`metadata/ember.json` and in the golden tests (see
[`docs/ember-edition.md`](../docs/ember-edition.md)).

Regenerate the baseline from the repository root:

```bash
just reference   # or: cd ci/reference && ./generate_reference.sh
```

This renders a fixed seed with `--image-only --no-ember` and writes:

- `ci/reference/baseline_512x288.png`: the reference image
- `ci/reference/baseline_512x288.json`: its parameters and SHA-256

Verify a render against it:

```bash
python3 ci/verify_reference.py output/test/images/source/master.png
```

Pass a second path to compare against another reference image; its `.json` (next to it) supplies
the expected hash.

## Output naming

The generator writes each package under one explicit output name:

```bash
./target/release/three_body_problem --seed 0x0123 --output experiment-1
# Creates: output/experiment-1/images/source/{master,ember}.png,
#          output/experiment-1/images/web/*.webp,
#          output/experiment-1/videos/{web,hq}/*.mp4,
#          output/experiment-1/metadata/*.json
```
