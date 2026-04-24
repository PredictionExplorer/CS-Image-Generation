//! Three Body Problem Visualization Library
//!
//! This library provides simulation and rendering capabilities for the
//! three-body gravitational problem, producing museum-quality visualizations
//! with physically-based spectral rendering and cinematic post-processing.
//!
//! # Modules
//!
//! - [`app`] -- high-level generation pipeline (seed parsing, orchestration).
//! - [`drift`] -- body drift transforms (Brownian, linear, elliptical).
//! - [`drift_config`] -- drift parameter resolution and validation.
//! - [`error`] -- error types and the crate-wide [`Result`] alias.
//! - [`generation_log`] -- persistent JSON generation log.
//! - [`post_effects`] -- composable image post-processing effects.
//! - [`pipeline`] -- top-level generation request and orchestration boundary.
//! - [`render`] -- rendering pipeline (histogram, tonemapping, effects, video).
//! - [`sim`] -- N-body gravitational simulation with RNG.
//! - [`spectral_constants`] -- wavelength/bin conversion constants.
//! - [`spectrum`] -- spectral power distribution to RGB conversion.
//! - [`spectrum_simd`] -- SIMD-accelerated spectral conversion (AVX2/NEON).
//!
//! # Error types
//!
//! - [`AppError`] is the top-level error for the application.
//! - [`AppRenderError`] wraps render-module errors at the app level.
//! - [`render::error::RenderError`] is the render-module's internal error type.

pub(crate) mod analysis;
/// High-level generation pipeline: seed parsing, simulation orchestration, and output.
pub mod app;
/// Body drift transforms applied between simulation frames.
pub mod drift;
/// Drift parameter resolution and validation.
pub mod drift_config;
/// Error types and the crate-wide [`Result`] alias.
pub mod error;
/// Persistent JSON generation log for tracking produced seeds.
pub mod generation_log;
pub(crate) mod oklab;
/// Top-level generation request and orchestration boundary.
pub mod pipeline;
/// Composable image post-processing effect pipeline.
pub mod post_effects;
/// Rendering pipeline: histogram passes, tonemapping, effects, and video output.
pub mod render;
/// N-body gravitational simulation with deterministic RNG.
pub mod sim;
/// Wavelength-to-bin and bin-to-wavelength conversion constants.
pub mod spectral_constants;
/// Spectral power distribution (SPD) to RGB conversion via pre-computed LUTs.
pub mod spectrum;
/// SIMD-accelerated spectral conversion with AVX2, NEON, and scalar fallback.
pub mod spectrum_simd;
pub(crate) mod utils;

/// Re-exported common types for convenience.
pub use error::{AppError, AppRenderError, ConfigError, Result, SimulationError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_exported_result_alias_is_usable() {
        fn fallible(succeed: bool) -> Result<i32> {
            if succeed {
                Ok(42)
            } else {
                Err(ConfigError::InvalidResolution { reason: "test".into() }.into())
            }
        }
        assert!(fallible(true).is_ok());
        assert!(fallible(false).is_err());
    }

    #[test]
    fn re_exported_error_types_are_accessible() {
        let _app: AppError = ConfigError::InvalidResolution { reason: "test".into() }.into();
        let _render: AppRenderError =
            AppRenderError::InvalidDimensions { width: 0, height: 0, reason: "test".into() };
        let _sim: SimulationError = SimulationError::NoValidOrbits {
            total_attempted: 0,
            discarded: 0,
            reason: "test".into(),
        };
    }

    #[test]
    fn public_modules_are_accessible() {
        let _ = spectrum::NUM_BINS;
        let _ = spectral_constants::LAMBDA_START;
        let _ = std::mem::size_of::<sim::Body>();
        let _ = std::mem::size_of::<render::BloomMode>();
    }
}
