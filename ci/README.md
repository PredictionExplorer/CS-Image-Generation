# CI Infrastructure

This directory contains CI support files and manual reference-image tooling for the Three Body Problem simulator.

## Structure

- `reference/` - Contains reference images and metadata for regression testing
- `verify_reference.py` - Python script to verify generated images against references
- `README.md` - This file

## Reference Images

Reference images are used for manual deterministic-output checks. They are not run automatically by the GitHub Actions workflow today. To generate or update reference images:

```bash
cd ci/reference
./generate_reference.sh
```

This will create:
- `baseline_512x288.png` - The reference image
- `baseline_512x288.json` - Metadata including parameters and SHA256 hash

## CI Workflow

The GitHub Actions workflow (`.github/workflows/ci.yml`) performs:

1. **Python** — `ruff format --check`, `ruff check`, and `mypy` on the repository scripts (`pyproject.toml`), then the `run.py` unit tests: `python -m unittest discover -s tests/python -v`. They live in `tests/python/`, use the standard library only, and cover the sync policy (seed planning, backfill modes, the retry cap and its failure ledger, generator exit statuses, uploads) with fake generator, `ssh` and `scp` commands on `PATH`, so they need no network, remote host or real generator
2. **Formatting** — `cargo fmt --all -- --check`
3. **Linting** — `cargo clippy --all-targets -- -D warnings`
4. **Tests** — `cargo nextest run --release` on Ubuntu and macOS
   - The job installs FFmpeg first (`sudo apt-get install -y --no-install-recommends ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS), so the tests that render a package or encode the ember videos run, such as the generator's exit-status-3 end-to-end test and the app-level ember renders. Locally these tests skip with a message when `ffmpeg` is not on `PATH`. GitHub Actions sets the `CI` environment variable, and under it they panic instead, so a runner without FFmpeg fails rather than passing silently
   - The workflow-level `RUSTFLAGS="-D warnings"` overrides `.cargo/config.toml`, so these builds target baseline x86-64 (the scalar spectral fallback on Linux, NEON on Apple Silicon)
   - A separate **x86-64 AVX2** job runs clippy and the tests with `-C target-cpu=x86-64-v3 -C target-feature=+avx2,+fma` so the AVX2 kernel used by production x86 builds is compiled and tested; this also checks that the ember golden digests (`tests/ember_determinism.rs`) come out the same under AVX2/FMA code generation. It installs FFmpeg too
5. **Benchmarks** — compile-check benchmark targets with `cargo bench --no-run`
6. **Documentation** — `cargo doc` with `-D warnings` to catch broken links
7. **Security Audit** — `cargo audit` (installed with `taiki-e/install-action@cargo-audit`) against the RustSec advisory database, on the committed `Cargo.lock`, which is the lockfile release builds use (`cargo build --release --locked`). It needs no permission beyond `contents: read`
   - This replaces `rustsec/audit-check`, which failed on every run on `main` whatever the advisories said. That action creates a check run, which needs `checks: write`, a permission the workflow does not grant; later runs also ended with cargo exit status 101 inside the action
   - `Cargo.lock` pins `crossbeam-epoch` 0.9.20 (reached through `rayon`), which fixes RUSTSEC-2026-0204 in 0.9.18
8. **Coverage** — `cargo-llvm-cov` with LCOV output uploaded as artifact; it installs FFmpeg like the test job, so the FFmpeg-dependent ember code is covered

Additional automation:
- **Dependabot** (`.github/dependabot.yml`) — weekly Cargo and GitHub Actions dependency updates
- **cargo-deny** (`deny.toml`) — license allowlist, advisory checks, and source restrictions

## Local Development

Install [just](https://github.com/casey/just) and run:

```bash
just check    # fmt + clippy
just py-check # ruff + mypy (after `pip install -e ".[dev]"` in a venv)
just test     # full test suite
just all      # check + test
```

The `run.py` unit tests need no extra packages:

```bash
python -m unittest discover -s tests/python -v
```

The tests that render a package or encode video need `ffmpeg` (with `ffprobe`) on `PATH`.
Without it they skip locally, and fail when the `CI` environment variable is set.

A pre-commit hook is available at `.githooks/pre-commit`. Enable it with:

```bash
git config core.hooksPath .githooks
```

## Output Naming

The generator uses a single explicit output name:

```bash
./target/release/three_body_problem --seed 0x0123 --output experiment-1
# Creates: output/experiment-1/images/source/{master,ember}.png,
#          output/experiment-1/images/web/*.webp,
#          output/experiment-1/videos/{web,hq}/*.mp4,
#          output/experiment-1/metadata/*.json
```

The reference baseline (`reference/generate_reference.sh`) renders with `--image-only --no-ember`:
it covers the main master only. The ember edition carries its own cross-architecture digests in
`metadata/ember.json` (see [`docs/ember-edition.md`](../docs/ember-edition.md)).
