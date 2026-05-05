# Three Body Problem — development task runner
# Install: brew install just  # or: cargo install just

# Run fast Rust quality checks
check:
    cargo fmt --all -- --check
    cargo check --all-targets --all-features
    cargo clippy --all-targets --all-features -- -D warnings

# Run the full Rust quality gate used before commits
gate:
    just check
    just test
    just doc-check
    just bench-check
    just deny
    git diff --check

# Run every local quality gate, including slower coverage and audit checks
full-gate:
    just gate
    just py-check
    just audit
    just coverage

# Python: ruff + mypy + pytest (requires dev tools on PATH, e.g. `pip install -e ".[dev]"` in a venv)
py-check:
    @if [ -x .venv/bin/ruff ]; then RUFF=.venv/bin/ruff; else RUFF=ruff; fi; \
      if [ -x .venv/bin/mypy ]; then MYPY=.venv/bin/mypy; else MYPY=mypy; fi; \
      if [ -x .venv/bin/pytest ]; then PYTEST=.venv/bin/pytest; else PYTEST=pytest; fi; \
      "$RUFF" format --check .; \
      "$RUFF" check .; \
      "$MYPY"; \
      "$PYTEST"

py-fmt:
    @if [ -x .venv/bin/ruff ]; then RUFF=.venv/bin/ruff; else RUFF=ruff; fi; \
      "$RUFF" format .

# Run the full test suite in release mode
test:
    @if cargo nextest --version >/dev/null 2>&1; then \
      cargo nextest run --release; \
    else \
      cargo test --release; \
    fi

# Run release coverage and fail below the repository threshold
coverage:
    cargo llvm-cov --release --fail-under-lines 95

# Compile benchmarks without running timing-sensitive measurements
bench-check:
    cargo bench --no-run

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

# Check documentation without opening a browser
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items

# Run cargo-deny license and advisory checks (requires: cargo install cargo-deny)
deny:
    cargo deny check

# Run security audit (requires: cargo install cargo-audit)
audit:
    cargo audit

# Format all code
fmt:
    cargo fmt --all
