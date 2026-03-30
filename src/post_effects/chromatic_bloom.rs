//! Chromatic bloom effect for prismatic color separation.
//!
//! This effect creates a magical, lens-aberration-like glow by separating RGB channels
//! spatially and blurring them independently, then compositing back with additive blending.

use super::{PixelBuffer, PostEffect};
use rayon::prelude::*;
use std::error::Error;

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
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

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

    /// Apply box blur (separable) for efficient Gaussian approximation.
    /// Both passes are parallelised over rows via rayon.
    fn box_blur_channel(&self, buffer: &mut PixelBuffer, width: usize, height: usize) {
        if self.config.radius == 0 {
            return;
        }

        let radius = self.config.radius;

        let mut temp = vec![(0.0, 0.0, 0.0, 0.0); buffer.len()];
        temp.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for (x, pixel_out) in row.iter_mut().enumerate() {
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
                *pixel_out =
                    (sum.0 * inv_count, sum.1 * inv_count, sum.2 * inv_count, sum.3 * inv_count);
            }
        });

        buffer.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for (x, pixel_out) in row.iter_mut().enumerate() {
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
                *pixel_out =
                    (sum.0 * inv_count, sum.1 * inv_count, sum.2 * inv_count, sum.3 * inv_count);
            }
        });
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
    ) -> Result<PixelBuffer, Box<dyn Error>> {
        if !self.is_enabled() {
            return Ok(input.clone());
        }

        let bright = self.extract_bright_pixels(input);
        let (mut red, mut green, mut blue) = self.create_separated_channels(&bright, width, height);

        rayon::join(
            || self.box_blur_channel(&mut red, width, height),
            || {
                rayon::join(
                    || self.box_blur_channel(&mut green, width, height),
                    || self.box_blur_channel(&mut blue, width, height),
                );
            },
        );

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
    use crate::post_effects::PostEffect;

    fn uniform_buffer(width: usize, height: usize, val: f64) -> PixelBuffer {
        vec![(val, val, val, 1.0); width * height]
    }

    #[test]
    fn test_chromatic_bloom_preserves_dimensions() {
        let config = ChromaticBloomConfig { radius: 3, strength: 0.5, separation: 2.0, threshold: 0.1 };
        let bloom = ChromaticBloom::new(config);
        let input = uniform_buffer(32, 24, 0.8);
        let output = bloom.process(&input, 32, 24).unwrap();
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_chromatic_bloom_disabled_when_zero_strength() {
        let config = ChromaticBloomConfig { radius: 5, strength: 0.0, separation: 2.0, threshold: 0.1 };
        let bloom = ChromaticBloom::new(config);
        assert!(!bloom.is_enabled());
    }

    #[test]
    fn test_chromatic_bloom_disabled_when_zero_radius() {
        let config = ChromaticBloomConfig { radius: 0, strength: 0.5, separation: 2.0, threshold: 0.1 };
        let bloom = ChromaticBloom::new(config);
        assert!(!bloom.is_enabled());
    }

    #[test]
    fn test_chromatic_bloom_below_threshold_unchanged() {
        let config = ChromaticBloomConfig { radius: 3, strength: 0.5, separation: 2.0, threshold: 0.9 };
        let bloom = ChromaticBloom::new(config);
        let input = uniform_buffer(16, 16, 0.1);
        let output = bloom.process(&input, 16, 16).unwrap();
        for (&(ir, ig, ib, _), &(or, og, ob, _)) in input.iter().zip(output.iter()) {
            assert!((ir - or).abs() < 1e-10);
            assert!((ig - og).abs() < 1e-10);
            assert!((ib - ob).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chromatic_bloom_adds_energy_above_threshold() {
        let config = ChromaticBloomConfig { radius: 2, strength: 1.0, separation: 3.0, threshold: 0.0 };
        let bloom = ChromaticBloom::new(config);
        let mut input = uniform_buffer(32, 32, 0.0);
        input[16 * 32 + 16] = (2.0, 2.0, 2.0, 1.0);
        let output = bloom.process(&input, 32, 32).unwrap();
        let input_energy: f64 = input.iter().map(|(r, g, b, _)| r + g + b).sum();
        let output_energy: f64 = output.iter().map(|(r, g, b, _)| r + g + b).sum();
        assert!(output_energy >= input_energy);
    }

    #[test]
    fn test_from_resolution_scales_params() {
        let small = ChromaticBloomConfig::from_resolution(640, 360);
        let large = ChromaticBloomConfig::from_resolution(3840, 2160);
        assert!(large.radius >= small.radius);
    }

    #[test]
    fn test_parallel_blur_no_nan_or_inf() {
        let config = ChromaticBloomConfig { radius: 5, strength: 1.0, separation: 4.0, threshold: 0.0 };
        let bloom = ChromaticBloom::new(config);
        let mut input = uniform_buffer(64, 48, 0.0);
        for i in (0..input.len()).step_by(17) {
            input[i] = (3.0, 1.5, 2.0, 1.0);
        }
        let output = bloom.process(&input, 64, 48).unwrap();
        for (i, &(r, g, b, a)) in output.iter().enumerate() {
            assert!(r.is_finite(), "pixel {i} R is not finite: {r}");
            assert!(g.is_finite(), "pixel {i} G is not finite: {g}");
            assert!(b.is_finite(), "pixel {i} B is not finite: {b}");
            assert!(a.is_finite(), "pixel {i} A is not finite: {a}");
        }
    }

    #[test]
    fn test_parallel_blur_deterministic() {
        let config = ChromaticBloomConfig { radius: 4, strength: 0.8, separation: 3.0, threshold: 0.05 };
        let bloom = ChromaticBloom::new(config);
        let mut input = uniform_buffer(48, 36, 0.2);
        input[18 * 48 + 24] = (5.0, 4.0, 3.0, 1.0);

        let output1 = bloom.process(&input, 48, 36).unwrap();
        let output2 = bloom.process(&input, 48, 36).unwrap();
        for (i, (a, b)) in output1.iter().zip(output2.iter()).enumerate() {
            assert_eq!(a.0.to_bits(), b.0.to_bits(), "pixel {i} R not deterministic");
            assert_eq!(a.1.to_bits(), b.1.to_bits(), "pixel {i} G not deterministic");
            assert_eq!(a.2.to_bits(), b.2.to_bits(), "pixel {i} B not deterministic");
        }
    }

    #[test]
    fn test_blur_smooths_sharp_edge() {
        let config = ChromaticBloomConfig { radius: 3, strength: 1.0, separation: 0.0, threshold: 0.0 };
        let bloom = ChromaticBloom::new(config);
        let w = 32;
        let h = 32;
        let mut input: PixelBuffer = vec![(0.0, 0.0, 0.0, 1.0); w * h];
        for y in 0..h {
            for x in 0..w {
                if x >= w / 2 {
                    input[y * w + x] = (2.0, 2.0, 2.0, 1.0);
                }
            }
        }
        let output = bloom.process(&input, w, h).unwrap();
        let mid_y = h / 2;
        let left_of_edge = output[mid_y * w + (w / 2 - 2)].0;
        let right_of_edge = output[mid_y * w + (w / 2 + 2)].0;
        assert!(
            left_of_edge > 0.0,
            "bloom should spread light leftward from edge: {left_of_edge}"
        );
        assert!(
            right_of_edge > 1.5,
            "right side should retain most energy: {right_of_edge}"
        );
    }
}
