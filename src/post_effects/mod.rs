//! Post-processing effects pipeline for the Three Body Problem renderer.
//!
//! This module provides a trait-based system for applying visual effects
//! in a composable, modular fashion.

use thiserror::Error;

pub use crate::render::context::PixelBuffer;

/// Error type for post-processing pipeline failures.
#[derive(Debug, Error)]
pub enum PostEffectError {
    /// An effect encountered an I/O error.
    #[error("PostEffect I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A named effect reported an error.
    #[error("PostEffect '{effect_name}' error: {message}")]
    EffectFailed {
        /// Name of the effect that failed.
        effect_name: String,
        /// Human-readable error description.
        message: String,
    },

    /// A post-processing buffer has dimensions or length inconsistent with the
    /// requested image shape.
    #[error("PostEffect buffer invalid at {stage}: {message}")]
    InvalidBuffer {
        /// Stage or effect that observed the invalid buffer.
        stage: String,
        /// Human-readable error description.
        message: String,
    },
}

fn checked_pixel_count(stage: &str, width: usize, height: usize) -> Result<usize, PostEffectError> {
    if width == 0 || height == 0 {
        return Err(PostEffectError::InvalidBuffer {
            stage: stage.into(),
            message: format!("dimensions must be non-zero, got {width}x{height}"),
        });
    }

    width.checked_mul(height).ok_or_else(|| PostEffectError::InvalidBuffer {
        stage: stage.into(),
        message: format!("dimensions overflow usize: {width}x{height}"),
    })
}

fn validate_buffer_shape(
    stage: &str,
    buffer_len: usize,
    width: usize,
    height: usize,
) -> Result<(), PostEffectError> {
    let pixel_count = checked_pixel_count(stage, width, height)?;
    if buffer_len != pixel_count {
        return Err(PostEffectError::InvalidBuffer {
            stage: stage.into(),
            message: format!(
                "buffer length ({buffer_len}) does not match dimensions {width}x{height} ({pixel_count} pixels)"
            ),
        });
    }

    Ok(())
}

/// Trait for implementing post-processing effects.
///
/// Each effect transforms an input buffer and returns a new buffer.
/// Effects should be stateless and safe to call multiple times.
pub trait PostEffect: Send + Sync {
    /// Process the input buffer and return the result.
    ///
    /// # Arguments
    /// * `input` - Input pixel buffer
    /// * `width` - Buffer width in pixels
    /// * `height` - Buffer height in pixels
    ///
    /// # Returns
    /// Processed pixel buffer or error
    ///
    /// # Errors
    ///
    /// Returns an error when the effect cannot process the supplied buffer.
    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError>;

    /// Returns whether this effect is currently enabled.
    /// Default implementation returns true.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Returns a diagnostic name for this effect.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// A chain of post-processing effects applied in sequence.
pub struct PostEffectChain {
    effects: Vec<Box<dyn PostEffect>>,
}

impl PostEffectChain {
    /// Creates a new, empty effect chain.
    #[must_use]
    pub fn new() -> Self {
        Self { effects: Vec::new() }
    }

    /// Adds an effect to the end of the chain.
    pub fn add(&mut self, effect: Box<dyn PostEffect>) {
        self.effects.push(effect);
    }

    /// Processes a buffer through all enabled effects in order.
    ///
    /// # Arguments
    /// * `buffer` - Input pixel buffer
    /// * `width` - Buffer width in pixels
    /// * `height` - Buffer height in pixels
    ///
    /// # Returns
    /// Final processed buffer or first error encountered
    ///
    /// # Errors
    ///
    /// Returns the first error produced by an enabled effect in the chain.
    pub fn process(
        &self,
        mut buffer: PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        validate_buffer_shape("post_effect_chain input", buffer.len(), width, height)?;
        for effect in &self.effects {
            if effect.is_enabled() {
                buffer = effect.process(&buffer, width, height)?;
                validate_buffer_shape(effect.name(), buffer.len(), width, height)?;
            }
        }
        Ok(buffer)
    }

    /// Returns the number of effects in the chain.
    #[cfg(test)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Returns true if the chain has no effects.
    #[cfg(test)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for PostEffectChain {
    fn default() -> Self {
        Self::new()
    }
}

pub mod aether;
pub mod atmospheric_depth;
pub mod champleve;
pub mod chromatic_bloom;
pub mod color_grade;
pub mod crystal_facets;
pub mod dog_bloom;
pub mod edge_luminance;
pub mod fine_texture;
pub mod gaussian_bloom;
pub mod glow_enhancement;
pub mod gradient_map;
pub mod ink_cut_edges;
pub mod micro_contrast;
pub mod opalescence;
pub mod perceptual_blur;
pub mod prismatic_sparkle;
pub mod temporal_smoothing;
pub(super) mod utils;

pub use aether::{AetherConfig, apply_aether_weave, try_apply_aether_weave};
pub use atmospheric_depth::{AtmosphericDepth, AtmosphericDepthConfig};
pub use champleve::{
    ChampleveConfig, apply_champleve_iridescence, try_apply_champleve_iridescence,
};
pub use chromatic_bloom::{ChromaticBloom, ChromaticBloomConfig};
pub use color_grade::{CinematicColorGrade, ColorGradeParams};
pub use crystal_facets::{CrystalFacetConfig, CrystalFacetContrast};
pub use dog_bloom::DogBloom;
pub use edge_luminance::{EdgeLuminance, EdgeLuminanceConfig};
pub use fine_texture::{FineTexture, FineTextureConfig};
pub use gaussian_bloom::GaussianBloom;
pub use glow_enhancement::{GlowEnhancement, GlowEnhancementConfig};
pub use gradient_map::{GradientMap, GradientMapConfig, LuxuryPalette};
pub use ink_cut_edges::{InkCutConfig, InkCutEdges};
pub use micro_contrast::{MicroContrast, MicroContrastConfig};
pub use opalescence::{Opalescence, OpalescenceConfig};
pub use perceptual_blur::{PerceptualBlur, PerceptualBlurConfig};
pub use prismatic_sparkle::{PrismaticSparkle, PrismaticSparkleConfig};
pub use temporal_smoothing::{TemporalSmoothing, TemporalSmoothingConfig};

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test effect that adds a constant value
    struct AddEffect {
        value: f64,
        enabled: bool,
    }

    impl PostEffect for AddEffect {
        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn process(
            &self,
            input: &PixelBuffer,
            _width: usize,
            _height: usize,
        ) -> Result<PixelBuffer, PostEffectError> {
            let mut output = input.clone();
            for pixel in &mut output {
                pixel.0 += self.value;
                pixel.1 += self.value;
                pixel.2 += self.value;
            }
            Ok(output)
        }
    }

    struct BadShapeEffect;

    impl PostEffect for BadShapeEffect {
        fn process(
            &self,
            _input: &PixelBuffer,
            _width: usize,
            _height: usize,
        ) -> Result<PixelBuffer, PostEffectError> {
            Ok(vec![(0.0, 0.0, 0.0, 1.0)])
        }
    }

    #[test]
    fn test_empty_chain() {
        let chain = PostEffectChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result =
            chain.process(input.clone(), 1, 1).expect("empty chain process should succeed");
        assert_eq!(result, input);
    }

    #[test]
    fn test_single_effect() {
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(AddEffect { value: 0.1, enabled: true }));

        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result = chain.process(input, 1, 1).expect("single effect chain should succeed");
        assert_eq!(result[0].0, 0.6);
        assert_eq!(result[0].1, 0.6);
        assert_eq!(result[0].2, 0.6);
        assert_eq!(result[0].3, 1.0); // Alpha unchanged
    }

    #[test]
    fn test_multiple_effects() {
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(AddEffect { value: 0.1, enabled: true }));
        chain.add(Box::new(AddEffect { value: 0.2, enabled: true }));

        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result = chain.process(input, 1, 1).expect("multi-effect chain should succeed");
        assert_eq!(result[0].0, 0.8); // 0.5 + 0.1 + 0.2
        assert_eq!(result[0].1, 0.8);
        assert_eq!(result[0].2, 0.8);
        assert_eq!(result[0].3, 1.0);
    }

    #[test]
    fn test_disabled_effect() {
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(AddEffect { value: 0.1, enabled: false }));
        chain.add(Box::new(AddEffect { value: 0.2, enabled: true }));

        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result = chain.process(input, 1, 1).expect("chain with disabled effect should succeed");
        assert_eq!(result[0].0, 0.7); // 0.5 + 0.2 (first effect disabled)
        assert_eq!(result[0].1, 0.7);
        assert_eq!(result[0].2, 0.7);
        assert_eq!(result[0].3, 1.0);
    }

    #[test]
    fn test_chain_rejects_zero_dimensions_before_effects() {
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(AddEffect { value: 0.1, enabled: true }));

        let err = chain.process(Vec::new(), 0, 1).expect_err("zero width should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
        assert!(err.to_string().contains("dimensions must be non-zero"));
    }

    #[test]
    fn test_chain_rejects_input_shape_mismatch() {
        let chain = PostEffectChain::new();
        let input = vec![(0.5, 0.5, 0.5, 1.0); 3];

        let err = chain.process(input, 2, 2).expect_err("mismatched input should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
        assert!(err.to_string().contains("post_effect_chain input"));
        assert!(err.to_string().contains("buffer length"));
    }

    #[test]
    fn test_chain_rejects_effect_output_shape_mismatch() {
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(BadShapeEffect));
        let input = vec![(0.5, 0.5, 0.5, 1.0); 4];

        let err = chain.process(input, 2, 2).expect_err("mismatched output should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
        assert!(err.to_string().contains("BadShapeEffect"));
        assert!(err.to_string().contains("buffer length"));
    }

    #[test]
    fn test_error_type() {
        let error = PostEffectError::EffectFailed {
            effect_name: "Test Effect".to_string(),
            message: "Test error".to_string(),
        };

        assert_eq!(format!("{error}"), "PostEffect 'Test Effect' error: Test error");
    }

    #[test]
    fn test_invalid_buffer_error_type() {
        let error = PostEffectError::InvalidBuffer {
            stage: "chain".to_string(),
            message: "bad shape".to_string(),
        };

        assert_eq!(format!("{error}"), "PostEffect buffer invalid at chain: bad shape");
    }

    #[test]
    fn test_post_effect_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let effect_err: PostEffectError = io_err.into();
        let display = format!("{effect_err}");
        assert!(display.contains("I/O error"));
        assert!(display.contains("pipe broke"));
    }
}
