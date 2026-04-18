//! Error types for render module

use thiserror::Error;

/// Errors that can occur during rendering
#[derive(Debug, Error)]
pub enum RenderError {
    /// A post-processing effect chain step failed.
    #[error("Effect '{effect_name}' failed: {reason}")]
    EffectChain {
        /// Name of the effect that failed.
        effect_name: String,
        /// Description of the failure.
        reason: String,
    },

    /// Video encoder encountered an I/O failure.
    #[error("Video encoding failed")]
    VideoEncoding(#[from] std::io::Error),

    /// Render configuration is invalid or inconsistent.
    #[error("Invalid configuration for '{parameter}': {reason}")]
    InvalidConfig {
        /// Name of the invalid parameter.
        parameter: String,
        /// Why the value is invalid.
        reason: String,
    },

    /// Output dimensions are zero or otherwise unsupported.
    #[error("Invalid dimensions: width={width}, height={height}")]
    InvalidDimensions {
        /// Requested output width.
        width: u32,
        /// Requested output height.
        height: u32,
    },

    /// Image file encoding (e.g. PNG write) failed.
    #[error("Image encoding failed: {reason}")]
    ImageEncoding {
        /// Description of the encoding failure.
        reason: String,
    },

    /// Post-render quality gate rejected the final image. The seed is
    /// considered unshippable in its current form; the orchestrator
    /// (run.py) can retry with a rescue-salted seed or log and move on.
    #[error("Quality gate rejected render: {reason}")]
    QualityGateRejected {
        /// Human-readable description of which check failed and by how
        /// much (e.g. "near-white fraction 0.28 > 0.15").
        reason: String,
    },
}

/// Convenience type alias
pub type Result<T> = std::result::Result<T, RenderError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_chain_display() {
        let err =
            RenderError::EffectChain { effect_name: "blur".into(), reason: "blur failed".into() };
        assert_eq!(err.to_string(), "Effect 'blur' failed: blur failed");
    }

    #[test]
    fn test_video_encoding_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err: RenderError = io_err.into();
        assert!(matches!(err, RenderError::VideoEncoding(_)));
        assert_eq!(err.to_string(), "Video encoding failed");
    }

    #[test]
    fn test_invalid_config_display() {
        let err = RenderError::InvalidConfig {
            parameter: "frame_rate".into(),
            reason: "bad param".into(),
        };
        assert_eq!(err.to_string(), "Invalid configuration for 'frame_rate': bad param");
    }

    #[test]
    fn test_invalid_dimensions_display() {
        let err = RenderError::InvalidDimensions { width: 0, height: 100 };
        assert_eq!(err.to_string(), "Invalid dimensions: width=0, height=100");
    }

    #[test]
    fn test_image_encoding_display() {
        let err = RenderError::ImageEncoding { reason: "PNG write failed".into() };
        assert_eq!(err.to_string(), "Image encoding failed: PNG write failed");
    }

    #[test]
    fn test_result_type_alias() {
        let ok: Result<i32> = Ok(42);
        assert!(ok.is_ok());

        let err: Result<i32> =
            Err(RenderError::EffectChain { effect_name: "test".into(), reason: "test".into() });
        assert!(err.is_err());
    }
}
