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
fn help_lists_core_generation_flags() {
    let output = run_binary(&["--help"]);
    assert!(output.status.success(), "--help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in ["--seed", "--output", "--resolution", "--drift", "--fast-encode"] {
        assert!(stdout.contains(flag), "help output should document {flag}: {stdout}");
    }
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
fn empty_seed_is_rejected_before_generation() {
    let output = run_binary(&["--seed", "0x"]);
    assert!(!output.status.success(), "empty seed should cause non-zero exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("seed"), "stderr should mention seed validation: {stderr}");
}

#[test]
fn invalid_log_level_is_rejected() {
    let output = run_binary(&["--log-level", "three_body_problem=not-a-level"]);
    assert!(!output.status.success(), "invalid log level should cause non-zero exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("log_level"), "stderr should mention log_level: {stderr}");
}

#[test]
fn unknown_flags_are_rejected() {
    let output = run_binary(&["--nonexistent-flag"]);
    assert!(!output.status.success(), "unknown flags should cause non-zero exit");
}
