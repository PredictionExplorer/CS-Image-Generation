//! Integration tests for error handling across module boundaries.
//!
//! Validates that invalid inputs produce the correct error types and messages.

use three_body_problem::app;
use three_body_problem::error::{AppError, ConfigError};

#[test]
fn parse_seed_rejects_invalid_hex() {
    let result = app::parse_seed("not-a-hex-string");
    assert!(result.is_err(), "invalid hex should fail");
    let err = result.unwrap_err();
    assert!(
        matches!(err, AppError::Config(ConfigError::InvalidSeed { .. })),
        "expected InvalidSeed, got: {err:?}"
    );
}

#[test]
fn parse_seed_accepts_with_prefix() {
    let result = app::parse_seed("0xdeadbeef");
    assert!(result.is_ok(), "0x-prefixed hex should succeed");
}

#[test]
fn parse_seed_accepts_without_prefix() {
    let result = app::parse_seed("deadbeef");
    assert!(result.is_ok(), "bare hex should succeed");
}

#[test]
fn parse_seed_rejects_odd_length_hex() {
    let result = app::parse_seed("0xabc");
    assert!(result.is_err(), "odd-length hex should fail");
}

#[test]
fn parse_seed_rejects_empty_seed_with_specific_error() {
    for seed in ["", "0x", "0X"] {
        let result = app::parse_seed(seed);
        assert!(
            matches!(result, Err(AppError::Config(ConfigError::EmptySeed { .. }))),
            "expected EmptySeed for {seed:?}, got: {result:?}"
        );
    }
}

#[test]
fn error_types_implement_display() {
    let err =
        AppError::Config(ConfigError::InvalidResolution { reason: "test reason".to_string() });
    let msg = format!("{err}");
    assert!(msg.contains("test reason"), "Display should contain the reason");
}

#[test]
fn error_types_implement_debug() {
    let err =
        AppError::Config(ConfigError::InvalidResolution { reason: "test reason".to_string() });
    let debug = format!("{err:?}");
    assert!(!debug.is_empty(), "Debug should produce non-empty output");
}

#[test]
fn io_error_converts_to_app_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let app_err: AppError = io_err.into();
    assert!(matches!(app_err, AppError::Io(_)), "io error should convert to AppError::Io");
}
