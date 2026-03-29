.PHONY: check clippy clippy-native clippy-cross test fmt

CROSS_TARGETS := \
	x86_64-unknown-linux-gnu \
	aarch64-unknown-linux-gnu \
	x86_64-apple-darwin \
	aarch64-apple-darwin

# Run all checks: native clippy + cross-compilation clippy + tests
check: clippy test
	@echo "All checks passed."

# Clippy on native + every cross target (catches #[cfg] blind spots)
clippy: clippy-native clippy-cross

clippy-native:
	cargo clippy --all-targets

clippy-cross:
	@for target in $(CROSS_TARGETS); do \
		echo "--- clippy --lib --target $$target ---"; \
		RUSTFLAGS="" cargo clippy --lib --target $$target --target-dir target/cross-check || exit 1; \
	done

test:
	cargo test

fmt:
	cargo fmt -- --check
