//! Cross-platform compilation tests.
//!
//! Each test runs `cargo clippy --lib` against a foreign target triple, catching
//! lint and type errors hidden behind `#[cfg(target_arch)]` or `#[cfg(target_os)]`
//! gates that are invisible to the compiler on the host platform.
//!
//! A separate `--target-dir` avoids deadlocking with the parent `cargo test`
//! process, and `RUSTFLAGS=""` overrides the per-target `target-cpu=native` in
//! `.cargo/config.toml` that would produce spurious warnings on foreign targets.

use std::process::Command;

const CROSS_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
];

fn target_is_installed(triple: &str) -> bool {
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("failed to run rustup");
    let list = String::from_utf8_lossy(&output.stdout);
    list.lines().any(|line| line.trim() == triple)
}

fn assert_clippy_clean(triple: &str) {
    if !target_is_installed(triple) {
        eprintln!("SKIP: target {triple} not installed (run `rustup target add {triple}`)");
        return;
    }

    let output = Command::new("cargo")
        .args([
            "clippy",
            "--lib",
            "--target",
            triple,
            "--target-dir",
            "target/cross-check",
        ])
        .env("RUSTFLAGS", "")
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn cargo clippy for {triple}: {e}"));

    assert!(
        output.status.success(),
        "cargo clippy --lib --target {triple} failed (exit {}):\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn cross_compile_x86_64_linux() {
    assert_clippy_clean("x86_64-unknown-linux-gnu");
}

#[test]
fn cross_compile_aarch64_linux() {
    assert_clippy_clean("aarch64-unknown-linux-gnu");
}

#[test]
fn cross_compile_x86_64_macos() {
    assert_clippy_clean("x86_64-apple-darwin");
}

#[test]
fn cross_compile_aarch64_macos() {
    assert_clippy_clean("aarch64-apple-darwin");
}

#[test]
fn all_cross_targets_installed() {
    let mut missing = Vec::new();
    for triple in CROSS_TARGETS {
        if !target_is_installed(triple) {
            missing.push(*triple);
        }
    }
    assert!(
        missing.is_empty(),
        "cross-compilation targets not installed (rust-toolchain.toml should auto-install these): {}",
        missing.join(", "),
    );
}
