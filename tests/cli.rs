//! CLI binary integration tests.
//!
//! Validates that the binary accepts valid arguments and rejects invalid ones.

use std::process::Command;

fn binary_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().expect("failed to get test binary path");
    path.pop();
    path.pop();
    path.push("three_body_problem");
    path
}

fn run_binary(args: &[&str]) -> std::process::Output {
    Command::new(binary_path()).args(args).output().expect("failed to execute binary")
}

#[test]
fn help_flag_exits_successfully() {
    let output = run_binary(&["--help"]);
    assert!(output.status.success(), "--help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage"), "help output should contain Usage");
}

#[test]
fn version_flag_exits_successfully() {
    let output = run_binary(&["--version"]);
    assert!(output.status.success(), "--version should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1.0.0"), "version should contain 1.0.0");
}

#[test]
fn invalid_resolution_is_rejected() {
    let output = run_binary(&["--resolution", "notaresolution", "--seed", "0xdeadbeef"]);
    assert!(!output.status.success(), "invalid resolution should cause non-zero exit");
}

#[test]
fn zero_resolution_is_rejected() {
    let output = run_binary(&["--resolution", "0x0", "--seed", "0xdeadbeef"]);
    assert!(!output.status.success(), "zero resolution should cause non-zero exit");
}

#[test]
fn unknown_flags_are_rejected() {
    let output = run_binary(&["--nonexistent-flag"]);
    assert!(!output.status.success(), "unknown flags should cause non-zero exit");
}
