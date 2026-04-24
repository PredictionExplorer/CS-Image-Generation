# Three Body Problem — development task runner
# Install: cargo install just

# Run all quality checks
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

# Python: ruff + mypy (requires dev tools on PATH, e.g. `pip install -e ".[dev]"` in a venv)
py-check:
    ruff format --check .
    ruff check .
    mypy

py-fmt:
    ruff format .

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
