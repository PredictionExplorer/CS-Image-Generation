//! Sparse prismatic glints for crisp, jewel-like highlights.

use super::{PixelBuffer, PostEffect, PostEffectError, validate_buffer_shape};
use crate::render::constants;
use rayon::prelude::*;

/// Configuration for the prismatic sparkle finish.
#[derive(Clone, Debug)]
pub struct PrismaticSparkleConfig {
    /// Overall glint intensity.
    pub strength: f64,
    /// Display luminance required before a pixel can seed a sparkle.
    pub threshold: f64,
    /// Probability gate for eligible local maxima.
    pub density: f64,
    /// Sparkle radius in pixels.
    pub radius: usize,
}

impl Default for PrismaticSparkleConfig {
    fn default() -> Self {
        Self { strength: 0.16, threshold: 0.74, density: 0.018, radius: 2 }
    }
}

/// Adds tiny deterministic RGB-split glints to bright local maxima.
pub struct PrismaticSparkle {
    config: PrismaticSparkleConfig,
    enabled: bool,
}

impl PrismaticSparkle {
    /// Creates a new prismatic sparkle effect from the given configuration.
    #[must_use]
    pub fn new(config: PrismaticSparkleConfig) -> Self {
        let enabled = config.strength > 0.0 && config.density > 0.0 && config.radius > 0;
        Self { config, enabled }
    }

    #[inline]
    fn luminance(pixel: (f64, f64, f64, f64)) -> f64 {
        let (r, g, b, a) = pixel;
        if a <= 0.0 {
            return 0.0;
        }
        constants::rec709_luminance(r, g, b)
    }

    #[inline]
    fn hash01(x: usize, y: usize, salt: u64) -> f64 {
        let mut n = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ salt;
        n ^= n >> 30;
        n = n.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        n ^= n >> 27;
        n = n.wrapping_mul(0x94D0_49BB_1331_11EB);
        n ^= n >> 31;
        (n >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn is_local_max(luma: &[f64], width: usize, height: usize, x: usize, y: usize) -> bool {
        let idx = y * width + x;
        let center = luma[idx];
        let x_min = x.saturating_sub(1);
        let x_max = (x + 1).min(width - 1);
        let y_min = y.saturating_sub(1);
        let y_max = (y + 1).min(height - 1);

        for ny in y_min..=y_max {
            for nx in x_min..=x_max {
                let nidx = ny * width + nx;
                if nidx != idx && luma[nidx] > center {
                    return false;
                }
            }
        }
        true
    }

    fn sparkle_centers(&self, luma: &[f64], width: usize, height: usize) -> Vec<bool> {
        (0..luma.len())
            .into_par_iter()
            .map(|idx| {
                let lum = luma[idx];
                if lum < self.config.threshold {
                    return false;
                }
                let x = idx % width;
                let y = idx / width;
                Self::is_local_max(luma, width, height, x, y)
                    && Self::hash01(x, y, 0xA11C_EC7A_5C1E) < self.config.density
            })
            .collect()
    }

    #[inline]
    fn prismatic_color(x: usize, y: usize) -> [f64; 3] {
        let t = Self::hash01(x, y, 0xC010_2A1A_5EED);
        if t < 0.33 {
            [1.0, 0.78, 0.42]
        } else if t < 0.66 {
            [0.38, 0.78, 1.0]
        } else {
            [0.95, 0.52, 1.0]
        }
    }

    fn sparkle_boost_for_pixel(
        &self,
        centers: &[bool],
        luma: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
    ) -> [f64; 3] {
        let radius = self.config.radius as isize;
        let mut boost = [0.0, 0.0, 0.0];

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let distance_sq = dx * dx + dy * dy;
                if distance_sq > radius * radius {
                    continue;
                }
                let cx = x as isize + dx;
                let cy = y as isize + dy;
                if cx < 0 || cy < 0 || cx >= width as isize || cy >= height as isize {
                    continue;
                }
                let cidx = cy as usize * width + cx as usize;
                if !centers[cidx] {
                    continue;
                }

                let distance = (distance_sq as f64).sqrt();
                let falloff = 1.0 - distance / (radius as f64 + 1.0);
                let gate = ((luma[cidx] - self.config.threshold) / (1.0 - self.config.threshold))
                    .clamp(0.0, 1.0);
                let amount = self.config.strength * gate * falloff * falloff;
                let prism = Self::prismatic_color(cx as usize, cy as usize);
                boost[0] += prism[0] * amount;
                boost[1] += prism[1] * amount;
                boost[2] += prism[2] * amount;
            }
        }

        boost
    }
}

impl PostEffect for PrismaticSparkle {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        validate_buffer_shape(self.name(), input.len(), width, height)?;
        if !self.is_enabled() {
            return Ok(input.clone());
        }

        let luma: Vec<f64> = input.par_iter().map(|&pixel| Self::luminance(pixel)).collect();
        let centers = self.sparkle_centers(&luma, width, height);

        let output = input
            .par_iter()
            .enumerate()
            .map(|(idx, &(r, g, b, a))| {
                if a <= 0.0 {
                    return (r, g, b, a);
                }
                let x = idx % width;
                let y = idx / width;
                let boost = self.sparkle_boost_for_pixel(&centers, &luma, width, height, x, y);
                if boost == [0.0, 0.0, 0.0] {
                    return (r, g, b, a);
                }

                ((r + boost[0]).min(1.35), (g + boost[1]).min(1.35), (b + boost[2]).min(1.35), a)
            })
            .collect();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparkle_disabled_with_zero_strength() {
        let effect =
            PrismaticSparkle::new(PrismaticSparkleConfig { strength: 0.0, ..Default::default() });
        assert!(!effect.is_enabled());
    }

    #[test]
    fn test_sparkle_preserves_buffer_shape() {
        let effect = PrismaticSparkle::new(PrismaticSparkleConfig {
            strength: 0.2,
            threshold: 0.5,
            density: 1.0,
            radius: 1,
        });
        let mut input = vec![(0.02, 0.02, 0.02, 1.0); 25];
        input[12] = (1.0, 1.0, 1.0, 1.0);

        let output = effect.process(&input, 5, 5).expect("sparkle effect should process");

        assert_eq!(output.len(), input.len());
        assert!(
            output.iter().zip(input.iter()).any(|(out, inp)| out.0 > inp.0),
            "sparkle should brighten at least one highlight-adjacent pixel",
        );
    }

    #[test]
    fn test_sparkle_does_not_collapse_low_alpha_display_pixels() {
        let effect = PrismaticSparkle::new(PrismaticSparkleConfig {
            strength: 0.2,
            threshold: 0.5,
            density: 1.0,
            radius: 3,
        });
        let mut input = vec![(0.02, 0.02, 0.02, 1e-7); 81];
        input[4 * 9 + 4] = (0.8, 0.8, 0.8, 1e-7);

        let output = effect.process(&input, 9, 9).expect("sparkle effect should process");

        assert!(
            output[4 * 9 + 4].0 >= input[4 * 9 + 4].0,
            "display RGB should not be premultiplied again"
        );
        assert_eq!(output[4 * 9 + 4].3, input[4 * 9 + 4].3, "alpha should be preserved");
    }

    #[test]
    fn test_sparkle_uses_isotropic_glints_not_cross_arms() {
        let effect = PrismaticSparkle::new(PrismaticSparkleConfig {
            strength: 0.2,
            threshold: 0.5,
            density: 1.0,
            radius: 3,
        });
        let mut input = vec![(0.02, 0.02, 0.02, 1.0); 81];
        input[4 * 9 + 4] = (0.8, 0.8, 0.8, 1.0);

        let output = effect.process(&input, 9, 9).expect("sparkle effect should process");

        assert!(
            output[(4 + 2) * 9 + (4 + 1)].0 > input[(4 + 2) * 9 + (4 + 1)].0,
            "off-axis pixels inside the sparkle radius should brighten; otherwise the sparkle reads as cross arms",
        );
    }
}
