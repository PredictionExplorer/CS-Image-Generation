//! Gaussian blur-based bloom effect implementation.

use super::{PixelBuffer, PostEffect};
use crate::render::{constants, parallel_blur_2d_rgba};
use rayon::prelude::*;
use std::error::Error;

/// Gaussian bloom post-processing effect.
///
/// Applies a Gaussian blur to create a soft glow effect, then composites
/// it with the original image using screen blending.
pub struct GaussianBloom {
    /// Blur radius in pixels.
    pub radius: usize,

    /// Strength of the bloom effect (multiplier for blurred component).
    pub strength: f64,

    /// Brightness multiplier for the core (unblurred) image.
    pub core_brightness: f64,

    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl GaussianBloom {
    /// Creates a new Gaussian bloom effect with the given parameters.
    pub fn new(radius: usize, strength: f64, core_brightness: f64) -> Self {
        Self { radius, strength, core_brightness, enabled: true }
    }

    #[inline]
    fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    #[inline]
    fn highlight_extract_factor(luminance: f64) -> f64 {
        let knee = constants::DEFAULT_HIGHLIGHT_EXTRACT_KNEE;
        let threshold = constants::DEFAULT_HIGHLIGHT_EXTRACT_THRESHOLD;
        Self::smoothstep(threshold - knee * 0.5, threshold + knee * 0.5, luminance)
    }

    fn extract_highlights(&self, input: &PixelBuffer) -> PixelBuffer {
        input
            .par_iter()
            .map(|&(r, g, b, a)| {
                if a <= 0.0 {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                let sr = r / a;
                let sg = g / a;
                let sb = b / a;
                let luminance = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
                let factor = Self::highlight_extract_factor(luminance);
                (r * factor, g * factor, b * factor, a * factor)
            })
            .collect()
    }

    #[inline]
    fn core_gain(&self) -> f64 {
        (1.0 - (-0.05 * self.core_brightness.max(0.0)).exp()).min(0.55)
    }
}

impl PostEffect for GaussianBloom {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, Box<dyn Error>> {
        let highlights = self.extract_highlights(input);
        let core_gain = self.core_gain();

        if self.radius == 0 {
            let mut output = Vec::with_capacity(input.len());
            let iter = input.par_iter().zip(highlights.par_iter()).map(
                |(&(r, g, b, a), &(hr, hg, hb, _ha))| {
                    (r + hr * core_gain, g + hg * core_gain, b + hb * core_gain, a)
                },
            );
            output.par_extend(iter);
            return Ok(output);
        }

        // Create blurred highlight residual
        let mut blurred = highlights.clone();
        parallel_blur_2d_rgba(&mut blurred, width, height, self.radius);

        // Composite extracted highlight residuals over the untouched base image.
        let mut output = Vec::with_capacity(input.len());
        let iter = input.par_iter().zip(highlights.par_iter()).zip(blurred.par_iter()).map(
            |((&(base_r, base_g, base_b, base_a), &(hr, hg, hb, _ha)), &(br, bg, bb, _ba))| {
                (
                    base_r + hr * core_gain + br * self.strength,
                    base_g + hg * core_gain + bg * self.strength,
                    base_b + hb * core_gain + bb * self.strength,
                    base_a,
                )
            },
        );
        output.par_extend(iter);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_bloom_preserves_dark_pixels() {
        let bloom = GaussianBloom::new(0, 0.5, 12.0);
        let input = vec![(0.08, 0.08, 0.08, 1.0)];
        let output = bloom.process(&input, 1, 1).unwrap();

        assert!((output[0].0 - input[0].0).abs() < 1e-3);
        assert!((output[0].1 - input[0].1).abs() < 1e-3);
        assert!((output[0].2 - input[0].2).abs() < 1e-3);
    }

    #[test]
    fn test_gaussian_bloom_lifts_highlights() {
        let bloom = GaussianBloom::new(0, 0.5, 12.0);
        let input = vec![(0.85, 0.85, 0.85, 1.0)];
        let output = bloom.process(&input, 1, 1).unwrap();

        assert!(output[0].0 > input[0].0);
        assert!(output[0].1 > input[0].1);
        assert!(output[0].2 > input[0].2);
    }
}
