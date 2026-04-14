# Three Body Problem — development task runner
# Install: cargo install just

# Run all quality checks
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings

# Run the full test suite in release mode
test:
    cargo test --release

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
