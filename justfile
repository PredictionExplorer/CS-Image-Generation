# Three Body Problem — development task runner
# Install: brew install just  # or: cargo install just

# Run all quality checks
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

# Run the full Rust quality gate used before commits
gate:
    cargo fmt --all -- --check
    cargo check --all-targets --all-features
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --release
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
    git diff --check

# Python: ruff + mypy (requires dev tools on PATH, e.g. `pip install -e ".[dev]"` in a venv)
py-check:
    @if [ -x .venv/bin/ruff ]; then RUFF=.venv/bin/ruff; else RUFF=ruff; fi; \
      if [ -x .venv/bin/mypy ]; then MYPY=.venv/bin/mypy; else MYPY=mypy; fi; \
      "$RUFF" format --check .; \
      "$RUFF" check .; \
      "$MYPY"

py-fmt:
    @if [ -x .venv/bin/ruff ]; then RUFF=.venv/bin/ruff; else RUFF=ruff; fi; \
      "$RUFF" format .

# Run the full test suite in release mode
test:
    cargo test --release

# Run release coverage and fail below the repository threshold
coverage:
    cargo llvm-cov --release --fail-under-lines 95

# Run benchmarks
bench:
    cargo bench

# Build release binary
build:
    cargo build --release

# Generate reference baseline images for CI
reference:
    cd ci/reference && ./generate_reference.sh

# Run all checks and tests
all: check test

# Generate documentation
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --open

# Run cargo-deny license and advisory checks (requires: cargo install cargo-deny)
deny:
    cargo deny check

# Run security audit (requires: cargo install cargo-audit)
audit:
    cargo audit

# Format all code
fmt:
    cargo fmt --all
