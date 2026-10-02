//! Errors of the ember edition.

use thiserror::Error;

/// Result alias for ember operations.
pub type EmberResult<T> = std::result::Result<T, EmberError>;

/// Everything that can go wrong while rendering the ember edition.
#[derive(Debug, Error)]
pub enum EmberError {
    /// A configuration value is out of its valid range.
    #[error("invalid ember configuration `{parameter}`: {reason}")]
    InvalidConfig {
        /// Dotted path of the offending parameter (e.g. `fluid.cfl`).
        parameter: String,
        /// Why the value is rejected.
        reason: String,
    },

    /// The orbit cannot be drawn: it is not three equally long recordings of at least two steps,
    /// or its track on the canvas is non-finite or stationary.
    #[error("degenerate orbit: {reason}")]
    DegenerateOrbit {
        /// What made the orbit unusable.
        reason: String,
    },

    /// The main edition's view cannot be followed (a non-finite or inconsistent recorded view, or
    /// a drift whose path is not recorded).
    #[error("invalid ember view: {reason}")]
    InvalidView {
        /// What is wrong with the view.
        reason: String,
    },

    /// The orbit is too short in fluid time for the configured pre-roll and valve.
    #[error(
        "orbit lasts {duration:.4} fluid time units, but pre-roll and valve need more than {required:.4}"
    )]
    OrbitTooShort {
        /// Orbit duration in fluid time units.
        duration: f64,
        /// Minimum duration required by the configuration.
        required: f64,
    },

    /// The frame schedule is empty, unsorted, or points past the recorded orbit.
    #[error("invalid frame schedule: {reason}")]
    InvalidSchedule {
        /// What is wrong with the schedule.
        reason: String,
    },

    /// A grid or transform size is unsupported.
    #[error("unsupported grid size {nx}x{ny}: {reason}")]
    UnsupportedGrid {
        /// Columns.
        nx: usize,
        /// Rows.
        ny: usize,
        /// Why the size is unsupported.
        reason: String,
    },

    /// A NaN or infinity appeared in a simulated or shaded quantity.
    #[error("non-finite value in {stage} at fluid time {time:.6}")]
    NonFinite {
        /// Pipeline stage that detected the value.
        stage: &'static str,
        /// Fluid time at which it was detected.
        time: f64,
    },

    /// The frame consumer (video encoder, hasher, ...) failed.
    #[error("frame sink failed: {0}")]
    Sink(String),
}
