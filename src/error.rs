//! Comprehensive error handling for the Three Body Problem visualization system
//!
//! This module provides a unified error type hierarchy for all operations in the system,
//! enabling proper error propagation and recovery instead of panics.

use thiserror::Error;

/// Result type alias using our custom error type
pub type Result<T> = std::result::Result<T, AppError>;

/// Top-level application error encompassing all possible failure modes
#[derive(Debug, Error)]
pub enum AppError {
    /// Simulation-related errors
    #[error("Simulation error: {0}")]
    Simulation(#[from] SimulationError),

    /// Rendering-related errors (app-level wrapper)
    #[error("Rendering error: {0}")]
    Render(#[from] AppRenderError),

    /// Configuration and input validation errors
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// File I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Render-module internal errors
    #[error("{0}")]
    RenderInternal(#[from] crate::render::error::RenderError),
}

/// Errors that can occur during physics simulation
#[derive(Debug, Error)]
pub enum SimulationError {
    /// No valid orbits found after filtering and escape checks
    #[error(
        "No valid orbits found after filtering {discarded}/{total_attempted} candidates. Reason: {reason}"
    )]
    NoValidOrbits {
        /// Total number of orbit candidates that were attempted.
        total_attempted: usize,
        /// Number of candidates that were discarded.
        discarded: usize,
        /// Human-readable explanation for the failure.
        reason: String,
    },
}

/// Errors that can occur during rendering operations (app-level wrapper)
#[derive(Debug, Error)]
pub enum AppRenderError {
    /// Wraps the existing `render::error::RenderError`
    #[error("{0}")]
    Inner(#[from] crate::render::error::RenderError),

    /// Invalid rendering dimensions
    #[error("Invalid dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
        /// Why the dimensions are invalid.
        reason: String,
    },
}

/// Configuration and validation errors
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Invalid seed format
    #[error("Invalid hex seed '{seed}': {error}")]
    InvalidSeed {
        /// The seed string that failed to parse.
        seed: String,
        /// The underlying hex-decode error.
        error: hex::FromHexError,
    },

    /// File system error
    #[error("Failed to {operation} '{path}': {error}")]
    FileSystem {
        /// The file-system operation that failed (e.g. "read", "write").
        operation: String,
        /// Path that was being accessed.
        path: String,
        /// The underlying I/O error.
        error: std::io::Error,
    },

    /// Invalid resolution format
    #[error("Invalid resolution: {reason}")]
    InvalidResolution {
        /// Why the resolution is invalid.
        reason: String,
    },

    /// Partial drift configuration (must be all-or-nothing)
    #[error(
        "Drift parameters must be either all specified or all omitted. \
         Provided: scale={scale:?}, arc_fraction={arc_fraction:?}, eccentricity={eccentricity:?}"
    )]
    InvalidDriftConfig {
        /// Drift scale, if provided.
        scale: Option<f64>,
        /// Arc fraction, if provided.
        arc_fraction: Option<f64>,
        /// Eccentricity, if provided.
        eccentricity: Option<f64>,
    },
}

/// Helper functions for common validation patterns
pub mod validation {
    use super::{AppRenderError, Result};

    /// Validate that dimensions are non-zero and reasonable.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero or exceeds the maximum
    /// supported size.
    pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(AppRenderError::InvalidDimensions {
                width,
                height,
                reason: "Dimensions must be greater than zero".to_string(),
            }
            .into());
        }

        const MAX_DIMENSION: u32 = 16384; // 16K
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(AppRenderError::InvalidDimensions {
                width,
                height,
                reason: format!("Dimensions must not exceed {MAX_DIMENSION}"),
            }
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_dimensions() {
        assert!(validation::validate_dimensions(1920, 1080).is_ok());
        assert!(validation::validate_dimensions(0, 1080).is_err());
        assert!(validation::validate_dimensions(1920, 0).is_err());
        assert!(validation::validate_dimensions(20000, 1080).is_err());
    }

    #[test]
    fn test_error_display() {
        let err = SimulationError::NoValidOrbits {
            total_attempted: 100,
            discarded: 95,
            reason: "All orbits escaped".to_string(),
        };

        let display = format!("{err}");
        assert!(display.contains("95/100"));
        assert!(display.contains("escaped"));
    }

    #[test]
    fn test_config_error_invalid_resolution_display() {
        let err = ConfigError::InvalidResolution { reason: "negative width".to_string() };
        let display = format!("{err}");
        assert!(display.contains("Invalid resolution"));
        assert!(display.contains("negative width"));
    }

    #[test]
    fn test_config_error_invalid_drift_display() {
        let err = ConfigError::InvalidDriftConfig {
            scale: Some(1.0),
            arc_fraction: None,
            eccentricity: None,
        };
        let display = format!("{err}");
        assert!(display.contains("must be either all specified or all omitted"));
        assert!(display.contains("1.0"));
    }

    #[test]
    fn test_app_error_from_config_error() {
        let cfg_err = ConfigError::InvalidResolution { reason: "test".to_string() };
        let app_err: AppError = cfg_err.into();
        let display = format!("{app_err}");
        assert!(display.contains("Configuration error"));
    }

    #[test]
    fn test_app_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let app_err: AppError = io_err.into();
        let display = format!("{app_err}");
        assert!(display.contains("I/O error"));
    }
}
