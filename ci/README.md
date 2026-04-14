# CI Infrastructure

This directory contains the continuous integration setup for the Three Body Problem simulator.

## Structure

- `reference/` - Contains reference images and metadata for regression testing
- `verify_reference.py` - Python script to verify generated images against references
- `README.md` - This file

## Reference Images

Reference images are used to ensure the simulator produces deterministic output across different runs and platforms. To generate or update reference images:

```bash
cd ci/reference
./generate_reference.sh
```

This will create:
- `baseline_512x288.png` - The reference image
- `baseline_512x288.json` - Metadata including parameters and SHA256 hash

## CI Workflow

The GitHub Actions workflow (`.github/workflows/ci.yml`) performs:

1. **Formatting** — `cargo fmt --all -- --check`
2. **Linting** — `cargo clippy --all-targets -- -D warnings`
3. **Tests** — `cargo nextest run --release` on Ubuntu and macOS
4. **Benchmarks** — compile-check with `cargo bench --no-run`
5. **Documentation** — `cargo doc` with `-D warnings` to catch broken links
6. **Security Audit** — `rustsec/audit-check` against the RustSec advisory database
7. **Coverage** — `cargo-llvm-cov` with LCOV output uploaded as artifact

Additional automation:
- **Dependabot** (`.github/dependabot.yml`) — weekly Cargo and GitHub Actions dependency updates
- **cargo-deny** (`deny.toml`) — license allowlist, advisory checks, and source restrictions

## Local Development

Install [just](https://github.com/casey/just) and run:

```bash
just check   # fmt + clippy
just test    # full test suite
just all     # check + test
```

A pre-commit hook is available at `.githooks/pre-commit`. Enable it with:

```bash
git config core.hooksPath .githooks
```

## Output Naming

The generator uses a single explicit output name:

```bash
./target/release/three_body_problem --seed 0x123 --output experiment-1
# Creates: output/experiment-1/image.png and output/experiment-1/video.mp4
```
