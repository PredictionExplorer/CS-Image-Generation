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
    pub fn process(
        &self,
        mut buffer: PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        for effect in &self.effects {
            if effect.is_enabled() {
                buffer = effect.process(&buffer, width, height)?;
            }
        }
        Ok(buffer)
    }

    /// Returns the number of effects in the chain.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Returns true if the chain has no effects.
    #[cfg(test)]
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
pub mod bloom_pyramid;
pub mod champleve;
pub mod chromatic_bloom;
pub mod color_grade;
pub mod diffraction_spikes;
pub mod dog_bloom;
pub mod edge_luminance;
pub mod fine_texture;
pub mod gaussian_bloom;
pub mod glaze;
pub mod glow_enhancement;
pub mod god_rays;
pub mod gradient_map;
pub mod lens_flare;
pub mod micro_contrast;
pub mod nebula_clouds;
pub mod opalescence;
pub mod palette_harmony;
pub mod perceptual_blur;
pub mod star_field;
pub mod temporal_smoothing;
pub(super) mod utils;

pub use aether::{AetherConfig, apply_aether_weave};
pub use atmospheric_depth::{AtmosphericDepth, AtmosphericDepthConfig};
pub use bloom_pyramid::BloomPyramid;
pub use champleve::{ChampleveConfig, apply_champleve_iridescence};
pub use chromatic_bloom::{ChromaticBloom, ChromaticBloomConfig};
pub use color_grade::{CinematicColorGrade, ColorGradeParams};
pub use diffraction_spikes::DiffractionSpikes;
pub use dog_bloom::DogBloom;
pub use edge_luminance::{EdgeLuminance, EdgeLuminanceConfig};
pub use fine_texture::{FineTexture, FineTextureConfig};
pub use gaussian_bloom::GaussianBloom;
pub use glaze::Glaze;
pub use glow_enhancement::{GlowEnhancement, GlowEnhancementConfig};
pub use god_rays::GodRays;
pub use gradient_map::{GradientMap, GradientMapConfig, LuxuryPalette};
pub use lens_flare::LensFlareDiffractive;
pub use micro_contrast::{MicroContrast, MicroContrastConfig};
pub use nebula_clouds::{NebulaCloudConfig, NebulaClouds};
pub use opalescence::{Opalescence, OpalescenceConfig};
pub use palette_harmony::{PaletteHarmony, PaletteScheme};
pub use perceptual_blur::{PerceptualBlur, PerceptualBlurConfig};
pub use star_field::StarField;
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
    fn test_error_type() {
        let error = PostEffectError::EffectFailed {
            effect_name: "Test Effect".to_string(),
            message: "Test error".to_string(),
        };

        assert_eq!(format!("{error}"), "PostEffect 'Test Effect' error: Test error");
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
