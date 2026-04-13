//! Error types for render module

use thiserror::Error;

/// Errors that can occur during rendering
#[derive(Debug, Error)]
pub enum RenderError {
    #[error("Effect chain failed: {0}")]
    EffectChain(String),

    #[error("Video encoding failed")]
    VideoEncoding(#[from] std::io::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Invalid dimensions: width={width}, height={height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Image encoding failed: {0}")]
    ImageEncoding(String),
}

/// Convenience type alias
pub type Result<T> = std::result::Result<T, RenderError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_chain_display() {
        let err = RenderError::EffectChain("blur failed".into());
        assert_eq!(err.to_string(), "Effect chain failed: blur failed");
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
        let err = RenderError::InvalidConfig("bad param".into());
        assert_eq!(err.to_string(), "Invalid configuration: bad param");
    }

    #[test]
    fn test_invalid_dimensions_display() {
        let err = RenderError::InvalidDimensions { width: 0, height: 100 };
        assert_eq!(err.to_string(), "Invalid dimensions: width=0, height=100");
    }

    #[test]
    fn test_image_encoding_display() {
        let err = RenderError::ImageEncoding("PNG write failed".into());
        assert_eq!(err.to_string(), "Image encoding failed: PNG write failed");
    }

    #[test]
    fn test_result_type_alias() {
        let ok: Result<i32> = Ok(42);
        assert!(ok.is_ok());

        let err: Result<i32> = Err(RenderError::EffectChain("test".into()));
        assert!(err.is_err());
    }
}
