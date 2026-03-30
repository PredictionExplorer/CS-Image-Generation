//! Post-processing effects pipeline for the Three Body Problem renderer.
//!
//! This module provides a trait-based system for applying visual effects
//! in a composable, modular fashion.

use std::error::Error;
#[cfg(test)]
use std::fmt;

/// Type alias for pixel buffers used throughout the pipeline.
/// Format: (R, G, B, A) with premultiplied alpha.
/// Color channels may be linear or display-space depending on the render stage.
pub type PixelBuffer = Vec<(f64, f64, f64, f64)>;

/// Camera basis vectors needed to map screen pixels to world-space directions.
#[derive(Clone, Debug)]
pub struct CameraOrientation {
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub fwd: [f64; 3],
    pub half_fov_tan: f64,
}

/// Per-frame context passed through the effect chain so procedural effects
/// can generate patterns in world-stable coordinates instead of screen space.
#[derive(Clone, Debug, Default)]
pub struct EffectContext {
    /// Camera orientation for the current frame.
    pub current_camera: Option<CameraOrientation>,
    /// Camera orientation for the reference frame (typically frame 0).
    /// When both cameras are present, effects can project screen pixels
    /// through the current camera into the reference camera's coordinate
    /// system, yielding coordinates that are stable under camera rotation.
    pub reference_camera: Option<CameraOrientation>,
}

/// Error type for post-processing pipeline failures.
#[derive(Debug)]
#[cfg(test)]
pub struct PostEffectError {
    effect_name: String,
    message: String,
}

#[cfg(test)]
impl fmt::Display for PostEffectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PostEffect '{}' error: {}", self.effect_name, self.message)
    }
}

#[cfg(test)]
impl Error for PostEffectError {}

/// Trait for implementing post-processing effects.
///
/// Each effect transforms an input buffer and returns a new buffer.
/// Effects should be stateless and safe to call multiple times.
pub trait PostEffect: Send + Sync {
    /// Process the input buffer and return the result.
    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, Box<dyn Error>>;

    /// Process the buffer in-place to avoid an extra allocation+copy.
    ///
    /// Default implementation falls back to the allocating `process` path.
    /// Override this for effects that can mutate the buffer directly.
    fn process_in_place(
        &self,
        buffer: &mut PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn Error>> {
        let result = self.process(buffer, width, height)?;
        *buffer = result;
        Ok(())
    }

    /// Process in-place with camera context for world-stable procedural patterns.
    ///
    /// Default delegates to `process_in_place`, ignoring the context.
    /// Override in effects that generate procedural patterns from pixel coordinates
    /// (e.g. Voronoi cells, noise textures) so they can use world-stable coords.
    fn process_in_place_with_context(
        &self,
        buffer: &mut PixelBuffer,
        width: usize,
        height: usize,
        _ctx: &EffectContext,
    ) -> Result<(), Box<dyn Error>> {
        self.process_in_place(buffer, width, height)
    }

    /// Returns whether this effect is currently enabled.
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
    pub fn new() -> Self {
        Self { effects: Vec::new() }
    }

    /// Adds an effect to the end of the chain.
    pub fn add(&mut self, effect: Box<dyn PostEffect>) {
        self.effects.push(effect);
    }

    /// Processes a buffer through all enabled effects in order.
    ///
    /// Uses context-aware in-place processing so procedural effects can
    /// generate world-stable patterns when camera data is available.
    pub fn process(
        &self,
        mut buffer: PixelBuffer,
        width: usize,
        height: usize,
        ctx: &EffectContext,
    ) -> Result<PixelBuffer, Box<dyn Error>> {
        for effect in &self.effects {
            if effect.is_enabled() {
                effect.process_in_place_with_context(&mut buffer, width, height, ctx)?;
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
pub mod champleve;
pub mod chromatic_bloom;
pub mod color_grade;
pub mod dog_bloom;
pub mod edge_luminance;
pub mod fine_texture;
pub mod gaussian_bloom;
pub mod glow_enhancement;
pub mod gradient_map;
pub mod micro_contrast;
pub mod opalescence;
pub mod perceptual_blur;
pub mod temporal_smoothing;
pub mod utils;

pub use aether::{AetherConfig, apply_aether_weave};
pub use atmospheric_depth::{AtmosphericDepth, AtmosphericDepthConfig};
pub use champleve::{ChampleveConfig, apply_champleve_iridescence};
pub use chromatic_bloom::{ChromaticBloom, ChromaticBloomConfig};
pub use color_grade::{CinematicColorGrade, ColorGradeParams};
pub use dog_bloom::DogBloom;
pub use edge_luminance::{EdgeLuminance, EdgeLuminanceConfig};
pub use fine_texture::{FineTexture, FineTextureConfig};
pub use gaussian_bloom::GaussianBloom;
pub use glow_enhancement::{GlowEnhancement, GlowEnhancementConfig};
pub use gradient_map::{GradientMap, GradientMapConfig, LuxuryPalette};
pub use micro_contrast::{MicroContrast, MicroContrastConfig};
pub use opalescence::{Opalescence, OpalescenceConfig};
pub use perceptual_blur::{PerceptualBlur, PerceptualBlurConfig};
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
        ) -> Result<PixelBuffer, Box<dyn Error>> {
            let mut output = input.clone();
            for pixel in &mut output {
                pixel.0 += self.value;
                pixel.1 += self.value;
                pixel.2 += self.value;
                // Alpha unchanged
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
        let ctx = EffectContext::default();
        let result = chain.process(input.clone(), 1, 1, &ctx).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_single_effect() {
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(AddEffect { value: 0.1, enabled: true }));

        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        let ctx = EffectContext::default();
        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result = chain.process(input, 1, 1, &ctx).unwrap();
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

        let ctx = EffectContext::default();
        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result = chain.process(input, 1, 1, &ctx).unwrap();
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

        let ctx = EffectContext::default();
        let input = vec![(0.5, 0.5, 0.5, 1.0)];
        let result = chain.process(input, 1, 1, &ctx).unwrap();
        assert_eq!(result[0].0, 0.7); // 0.5 + 0.2 (first effect disabled)
        assert_eq!(result[0].1, 0.7);
        assert_eq!(result[0].2, 0.7);
        assert_eq!(result[0].3, 1.0);
    }

    #[test]
    fn test_error_type() {
        let error = PostEffectError {
            effect_name: "Test Effect".to_string(),
            message: "Test error".to_string(),
        };

        assert_eq!(format!("{}", error), "PostEffect 'Test Effect' error: Test error");
    }

    // ── Context-aware pipeline tests ─────────────────────────────

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    struct ContextAwareEffect {
        saw_context: Arc<AtomicBool>,
    }

    impl PostEffect for ContextAwareEffect {
        fn process(
            &self,
            input: &PixelBuffer,
            _w: usize,
            _h: usize,
        ) -> Result<PixelBuffer, Box<dyn Error>> {
            Ok(input.clone())
        }

        fn process_in_place_with_context(
            &self,
            _buffer: &mut PixelBuffer,
            _w: usize,
            _h: usize,
            ctx: &EffectContext,
        ) -> Result<(), Box<dyn Error>> {
            if ctx.current_camera.is_some() {
                self.saw_context.store(true, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    #[test]
    fn test_chain_passes_context_to_effects() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(ContextAwareEffect { saw_context: Arc::clone(&flag) }));

        let ctx = EffectContext {
            current_camera: Some(CameraOrientation {
                right: [1.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fwd: [0.0, 0.0, 1.0],
                half_fov_tan: 0.5,
            }),
            reference_camera: None,
        };

        let buf = vec![(0.5, 0.5, 0.5, 1.0)];
        let _ = chain.process(buf, 1, 1, &ctx).unwrap();
        assert!(flag.load(Ordering::SeqCst), "context with camera should reach the effect");
    }

    #[test]
    fn test_chain_default_context_does_not_set_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut chain = PostEffectChain::new();
        chain.add(Box::new(ContextAwareEffect { saw_context: Arc::clone(&flag) }));

        let ctx = EffectContext::default();
        let buf = vec![(0.5, 0.5, 0.5, 1.0)];
        let _ = chain.process(buf, 1, 1, &ctx).unwrap();
        assert!(!flag.load(Ordering::SeqCst), "default context should not set the flag");
    }

    #[test]
    fn test_default_process_in_place_with_context_delegates() {
        let effect = AddEffect { value: 0.25, enabled: true };
        let mut buf = vec![(0.0, 0.0, 0.0, 1.0)];
        let ctx = EffectContext::default();
        effect.process_in_place_with_context(&mut buf, 1, 1, &ctx).unwrap();
        assert!((buf[0].0 - 0.25).abs() < 1e-12, "default should delegate to process_in_place");
    }

    #[test]
    fn test_effect_context_default_has_no_cameras() {
        let ctx = EffectContext::default();
        assert!(ctx.current_camera.is_none());
        assert!(ctx.reference_camera.is_none());
    }

    #[test]
    fn test_camera_orientation_clone() {
        let cam = CameraOrientation {
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fwd: [0.0, 0.0, 1.0],
            half_fov_tan: 0.5,
        };
        let cloned = cam.clone();
        assert_eq!(cloned.right, cam.right);
        assert_eq!(cloned.fwd, cam.fwd);
        assert_eq!(cloned.half_fov_tan, cam.half_fov_tan);
    }
}
