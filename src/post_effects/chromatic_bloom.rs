//! Chromatic bloom effect for prismatic color separation.
//!
//! This effect creates a magical, lens-aberration-like glow by separating RGB channels
//! spatially and blurring them independently, then compositing back with additive blending.

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// Configuration for chromatic bloom effect
#[derive(Clone, Debug)]
pub struct ChromaticBloomConfig {
    /// Blur radius in pixels
    pub radius: usize,
    /// Overall effect strength (0.0-1.0)
    pub strength: f64,
    /// RGB channel separation distance in pixels
    pub separation: f64,
    /// Luminance threshold for bloom activation (0.0-1.0)
    pub threshold: f64,
}

impl Default for ChromaticBloomConfig {
    fn default() -> Self {
        // Default for ~1080p resolution
        Self::from_resolution(1920, 1080)
    }
}

impl ChromaticBloomConfig {
    /// Create configuration scaled for the given resolution.
    /// This ensures the effect looks consistent across different resolutions.
    #[must_use]
    pub fn from_resolution(width: usize, height: usize) -> Self {
        let min_dim = width.min(height) as f64;
        Self {
            // Scale radius: 12px @ 1080p, 24px @ 4K
            radius: (0.0111 * min_dim).round() as usize,
            // Scale separation: 2.5px @ 1080p, 5px @ 4K
            separation: 0.0023 * min_dim,
            // These are ratios, no scaling needed
            strength: 0.65,
            threshold: 0.15,
        }
    }
}

/// Chromatic bloom post-effect
pub struct ChromaticBloom {
    config: ChromaticBloomConfig,
    enabled: bool,
}

impl ChromaticBloom {
    /// Creates a chromatic bloom stage from the given configuration.
    #[must_use]
    pub fn new(config: ChromaticBloomConfig) -> Self {
        Self { config, enabled: true }
    }

    /// Extract bright pixels above threshold
    fn extract_bright_pixels(&self, input: &PixelBuffer) -> PixelBuffer {
        input
            .par_iter()
            .map(|&(r, g, b, a)| {
                if a <= 0.0 {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                // Calculate luminance (Rec. 709)
                let lum = crate::render::constants::rec709_luminance(r, g, b);

                // Threshold-based extraction with smooth falloff
                let brightness =
                    (lum - self.config.threshold).max(0.0) / (1.0 - self.config.threshold);
                let factor = brightness.min(1.0).powf(1.5); // Smooth curve

                (r * factor, g * factor, b * factor, a * factor)
            })
            .collect()
    }

    /// Sample pixel with bilinear interpolation
    #[inline]
    fn sample_bilinear(
        buffer: &PixelBuffer,
        width: usize,
        height: usize,
        x: f64,
        y: f64,
    ) -> (f64, f64, f64, f64) {
        let x = x.clamp(0.0, (width - 1) as f64);
        let y = y.clamp(0.0, (height - 1) as f64);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        let p00 = buffer[y0 * width + x0];
        let p01 = buffer[y0 * width + x1];
        let p10 = buffer[y1 * width + x0];
        let p11 = buffer[y1 * width + x1];

        // Bilinear interpolation
        let top = (
            p00.0 * (1.0 - fx) + p01.0 * fx,
            p00.1 * (1.0 - fx) + p01.1 * fx,
            p00.2 * (1.0 - fx) + p01.2 * fx,
            p00.3 * (1.0 - fx) + p01.3 * fx,
        );

        let bottom = (
            p10.0 * (1.0 - fx) + p11.0 * fx,
            p10.1 * (1.0 - fx) + p11.1 * fx,
            p10.2 * (1.0 - fx) + p11.2 * fx,
            p10.3 * (1.0 - fx) + p11.3 * fx,
        );

        (
            top.0 * (1.0 - fy) + bottom.0 * fy,
            top.1 * (1.0 - fy) + bottom.1 * fy,
            top.2 * (1.0 - fy) + bottom.2 * fy,
            top.3 * (1.0 - fy) + bottom.3 * fy,
        )
    }

    /// Create spatially offset channel buffers
    fn create_separated_channels(
        &self,
        bright: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> (PixelBuffer, PixelBuffer, PixelBuffer) {
        let sep = self.config.separation;
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let size = width * height;

        // Red channel: offset outward from center
        let red_buffer: PixelBuffer = (0..size)
            .into_par_iter()
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                let offset_x = x as f64 + (dx / dist) * sep;
                let offset_y = y as f64 + (dy / dist) * sep;

                let (r, _, _, a) = Self::sample_bilinear(bright, width, height, offset_x, offset_y);
                (r, 0.0, 0.0, a)
            })
            .collect();

        // Green channel: centered (no offset)
        let green_buffer: PixelBuffer =
            bright.par_iter().map(|&(_, g, _, a)| (0.0, g, 0.0, a)).collect();

        // Blue channel: offset inward toward center
        let blue_buffer: PixelBuffer = (0..size)
            .into_par_iter()
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                let offset_x = x as f64 - (dx / dist) * sep;
                let offset_y = y as f64 - (dy / dist) * sep;

                let (_, _, b, a) = Self::sample_bilinear(bright, width, height, offset_x, offset_y);
                (0.0, 0.0, b, a)
            })
            .collect();

        (red_buffer, green_buffer, blue_buffer)
    }

    /// Apply box blur (separable) for efficient Gaussian approximation
    fn box_blur_channel(&self, buffer: &mut PixelBuffer, width: usize, height: usize) {
        if self.config.radius == 0 {
            return;
        }

        let radius = self.config.radius;

        // Horizontal pass
        let mut temp = buffer.clone();
        for y in 0..height {
            for x in 0..width {
                let mut sum = (0.0, 0.0, 0.0, 0.0);
                let mut count = 0;

                for dx in -(radius as i32)..=(radius as i32) {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                    let idx = y * width + nx;
                    sum.0 += buffer[idx].0;
                    sum.1 += buffer[idx].1;
                    sum.2 += buffer[idx].2;
                    sum.3 += buffer[idx].3;
                    count += 1;
                }

                let inv_count = 1.0 / count as f64;
                temp[y * width + x] =
                    (sum.0 * inv_count, sum.1 * inv_count, sum.2 * inv_count, sum.3 * inv_count);
            }
        }

        // Vertical pass
        for y in 0..height {
            for x in 0..width {
                let mut sum = (0.0, 0.0, 0.0, 0.0);
                let mut count = 0;

                for dy in -(radius as i32)..=(radius as i32) {
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    let idx = ny * width + x;
                    sum.0 += temp[idx].0;
                    sum.1 += temp[idx].1;
                    sum.2 += temp[idx].2;
                    sum.3 += temp[idx].3;
                    count += 1;
                }

                let inv_count = 1.0 / count as f64;
                buffer[y * width + x] =
                    (sum.0 * inv_count, sum.1 * inv_count, sum.2 * inv_count, sum.3 * inv_count);
            }
        }
    }
}

impl PostEffect for ChromaticBloom {
    fn is_enabled(&self) -> bool {
        self.enabled && self.config.strength > 0.0 && self.config.radius > 0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if !self.is_enabled() {
            return Ok(input.clone());
        }

        // Extract bright pixels
        let bright = self.extract_bright_pixels(input);

        // Separate into RGB channels with spatial offsets
        let (mut red, mut green, mut blue) = self.create_separated_channels(&bright, width, height);

        // Blur each channel independently
        self.box_blur_channel(&mut red, width, height);
        self.box_blur_channel(&mut green, width, height);
        self.box_blur_channel(&mut blue, width, height);

        // Composite back with additive blending
        let output: PixelBuffer = input
            .par_iter()
            .zip(red.par_iter())
            .zip(green.par_iter())
            .zip(blue.par_iter())
            .map(|(((orig, r), g), b)| {
                let bloom_r = r.0;
                let bloom_g = g.1;
                let bloom_b = b.2;

                (
                    (orig.0 + bloom_r * self.config.strength).min(10.0), // Clamp to reasonable HDR range
                    (orig.1 + bloom_g * self.config.strength).min(10.0),
                    (orig.2 + bloom_b * self.config.strength).min(10.0),
                    orig.3,
                )
            })
            .collect();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_matches_1080p() {
        let cfg = ChromaticBloomConfig::default();
        let explicit = ChromaticBloomConfig::from_resolution(1920, 1080);
        assert_eq!(cfg.radius, explicit.radius);
        assert_eq!(cfg.separation, explicit.separation);
        assert_eq!(cfg.strength, explicit.strength);
        assert_eq!(cfg.threshold, explicit.threshold);
    }

    #[test]
    fn test_from_resolution_scales_radius() {
        let small = ChromaticBloomConfig::from_resolution(640, 480);
        let large = ChromaticBloomConfig::from_resolution(3840, 2160);
        assert!(large.radius > small.radius);
        assert!(large.separation > small.separation);
    }

    #[test]
    fn test_is_enabled_default() {
        let bloom = ChromaticBloom::new(ChromaticBloomConfig::default());
        assert!(bloom.is_enabled());
    }

    #[test]
    fn test_is_enabled_zero_strength() {
        let cfg = ChromaticBloomConfig { strength: 0.0, ..ChromaticBloomConfig::default() };
        let bloom = ChromaticBloom::new(cfg);
        assert!(!bloom.is_enabled());
    }

    #[test]
    fn test_is_enabled_zero_radius() {
        let cfg = ChromaticBloomConfig { radius: 0, ..ChromaticBloomConfig::default() };
        let bloom = ChromaticBloom::new(cfg);
        assert!(!bloom.is_enabled());
    }

    #[test]
    fn test_disabled_passthrough() {
        let cfg = ChromaticBloomConfig { strength: 0.0, ..ChromaticBloomConfig::default() };
        let bloom = ChromaticBloom::new(cfg);
        let input = vec![(0.5, 0.3, 0.2, 1.0); 100];
        let output = bloom.process(&input, 10, 10).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_process_preserves_alpha() {
        let bloom = ChromaticBloom::new(ChromaticBloomConfig::from_resolution(100, 100));
        let input: PixelBuffer = (0..100)
            .map(|i| {
                let v = i as f64 / 100.0;
                (v, v, v, 0.8)
            })
            .collect();
        let output = bloom.process(&input, 10, 10).unwrap();
        for (inp, out) in input.iter().zip(output.iter()) {
            assert_eq!(inp.3, out.3, "Alpha must be preserved");
        }
    }

    #[test]
    fn test_bright_extraction_threshold() {
        let cfg = ChromaticBloomConfig { threshold: 0.5, ..ChromaticBloomConfig::default() };
        let bloom = ChromaticBloom::new(cfg);
        let dark = vec![(0.1, 0.1, 0.1, 1.0)];
        let bright = vec![(0.9, 0.9, 0.9, 1.0)];

        let dark_out = bloom.extract_bright_pixels(&dark);
        let bright_out = bloom.extract_bright_pixels(&bright);

        assert!(dark_out[0].0 < bright_out[0].0, "Bright pixels should have higher extraction");
    }

    #[test]
    fn test_transparent_pixel_extraction() {
        let bloom = ChromaticBloom::new(ChromaticBloomConfig::default());
        let input = vec![(0.0, 0.0, 0.0, 0.0)];
        let output = bloom.extract_bright_pixels(&input);
        assert_eq!(output[0], (0.0, 0.0, 0.0, 0.0));
    }
}
