//! Three Body Problem Visualization Library
//!
//! This library provides simulation and rendering capabilities for the three-body problem.
//!
//! ## Error types
//!
//! - [`AppError`] is the top-level error for the application.
//! - [`AppRenderError`] wraps render-module errors at the app level.
//! - [`render::error::RenderError`] is the render-module's internal error type.

pub(crate) mod analysis;
pub mod app;
pub mod drift;
pub mod drift_config;
pub mod error;
pub mod generation_log;
pub(crate) mod oklab;
pub mod post_effects;
pub mod render;
pub mod sim;
pub mod spectral_constants;
pub mod spectrum;
pub mod spectrum_simd;
pub(crate) mod utils;

// Re-export common types for convenience
pub use error::{AppError, AppRenderError, ConfigError, Result, SimulationError};
