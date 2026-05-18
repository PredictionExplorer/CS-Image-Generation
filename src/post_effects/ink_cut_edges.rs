//! Crisp ink-cut edge definition for premium print-like silhouettes.

use super::{PixelBuffer, PostEffect, PostEffectError, validate_buffer_shape};
use crate::render::constants;
use rayon::prelude::*;

/// Configuration for crisp ink-cut edge definition.
#[derive(Clone, Debug)]
pub struct InkCutConfig {
    /// Overall edge treatment strength.
    pub strength: f64,
    /// Sobel luminance threshold required to affect a pixel.
    pub threshold: f64,
    /// Dark-side line weight.
    pub darken: f64,
    /// Bright-side glint weight.
    pub glint: f64,
}

impl Default for InkCutConfig {
    fn default() -> Self {
        Self { strength: 0.18, threshold: 0.18, darken: 0.12, glint: 0.10 }
    }
}

/// Adds bright-side glints and dark-side cuts around strong luminance edges.
pub struct InkCutEdges {
    config: InkCutConfig,
    enabled: bool,
}

impl InkCutEdges {
    /// Creates a new ink-cut edge effect.
    #[must_use]
    pub fn new(config: InkCutConfig) -> Self {
        let enabled = config.strength > 0.0 && (config.darken > 0.0 || config.glint > 0.0);
        Self { config, enabled }
    }

    #[inline]
    fn pixel_luma(pixel: (f64, f64, f64, f64)) -> f64 {
        let (r, g, b, a) = pixel;
        if a <= 0.0 {
            return 0.0;
        }
        constants::rec709_luminance(r, g, b)
    }

    #[inline]
    fn luma_at(luma: &[f64], width: usize, height: usize, x: isize, y: isize) -> f64 {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            return 0.0;
        }
        luma[y as usize * width + x as usize]
    }

    fn sobel(luma: &[f64], width: usize, height: usize, x: usize, y: usize) -> f64 {
        let x = x as isize;
        let y = y as isize;
        let gx = -Self::luma_at(luma, width, height, x - 1, y - 1)
            - 2.0 * Self::luma_at(luma, width, height, x - 1, y)
            - Self::luma_at(luma, width, height, x - 1, y + 1)
            + Self::luma_at(luma, width, height, x + 1, y - 1)
            + 2.0 * Self::luma_at(luma, width, height, x + 1, y)
            + Self::luma_at(luma, width, height, x + 1, y + 1);
        let gy = -Self::luma_at(luma, width, height, x - 1, y - 1)
            - 2.0 * Self::luma_at(luma, width, height, x, y - 1)
            - Self::luma_at(luma, width, height, x + 1, y - 1)
            + Self::luma_at(luma, width, height, x - 1, y + 1)
            + 2.0 * Self::luma_at(luma, width, height, x, y + 1)
            + Self::luma_at(luma, width, height, x + 1, y + 1);
        (gx * gx + gy * gy).sqrt()
    }

    fn neighbor_average(luma: &[f64], width: usize, height: usize, x: usize, y: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0.0;
        let x = x as isize;
        let y = y as isize;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                sum += Self::luma_at(luma, width, height, x + dx, y + dy);
                count += 1.0;
            }
        }
        sum / count
    }
}

impl PostEffect for InkCutEdges {
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

        let luma: Vec<f64> = input.par_iter().map(|&pixel| Self::pixel_luma(pixel)).collect();
        let output = input
            .par_iter()
            .enumerate()
            .map(|(idx, &(r, g, b, a))| {
                if a <= 0.0 {
                    return (r, g, b, a);
                }
                let x = idx % width;
                let y = idx / width;
                if x == 0 || y == 0 || x >= width - 1 || y >= height - 1 {
                    return (r, g, b, a);
                }

                let edge = Self::sobel(&luma, width, height, x, y);
                if edge < self.config.threshold {
                    return (r, g, b, a);
                }

                let center = luma[idx];
                let neighbor_avg = Self::neighbor_average(&luma, width, height, x, y);
                let edge_gate = ((edge - self.config.threshold) / (1.0 - self.config.threshold))
                    .clamp(0.0, 1.0);
                let bright_side = center >= neighbor_avg;
                let factor = if bright_side {
                    1.0 + self.config.strength * self.config.glint * edge_gate
                } else {
                    1.0 - self.config.strength * self.config.darken * edge_gate
                };

                (
                    (r * factor).clamp(0.0, 1.35),
                    (g * factor).clamp(0.0, 1.35),
                    (b * factor).clamp(0.0, 1.35),
                    a,
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
    fn test_ink_cut_disabled_with_zero_strength() {
        let effect = InkCutEdges::new(InkCutConfig { strength: 0.0, ..Default::default() });
        assert!(!effect.is_enabled());
    }

    #[test]
    fn test_ink_cut_defines_both_sides_of_edge() {
        let effect = InkCutEdges::new(InkCutConfig {
            strength: 1.0,
            threshold: 0.05,
            darken: 0.30,
            glint: 0.30,
        });
        let mut input = vec![(0.2, 0.2, 0.2, 1.0); 25];
        for y in 0..5 {
            for x in 2..5 {
                input[y * 5 + x] = (0.8, 0.8, 0.8, 1.0);
            }
        }

        let output = effect.process(&input, 5, 5).expect("ink cut should process");

        assert!(output[2 * 5 + 1].0 < input[2 * 5 + 1].0, "dark edge side should darken");
        assert!(output[2 * 5 + 2].0 > input[2 * 5 + 2].0, "bright edge side should glint");
    }

    #[test]
    fn test_ink_cut_does_not_collapse_low_alpha_display_pixels() {
        let effect = InkCutEdges::new(InkCutConfig {
            strength: 1.0,
            threshold: 0.05,
            darken: 0.30,
            glint: 0.30,
        });
        let mut input = vec![(0.2, 0.2, 0.2, 1e-7); 25];
        for y in 0..5 {
            for x in 2..5 {
                input[y * 5 + x] = (0.8, 0.8, 0.8, 1e-7);
            }
        }

        let output = effect.process(&input, 5, 5).expect("ink cut should process");

        assert!(output[2 * 5 + 2].0 > 0.5, "bright display edge should remain visible");
        assert_eq!(output[2 * 5 + 2].3, input[2 * 5 + 2].3, "alpha should be preserved");
    }

    #[test]
    fn test_ink_cut_is_deterministic() {
        let effect = InkCutEdges::new(InkCutConfig::default());
        let input = vec![(0.4, 0.3, 0.2, 1.0); 36];

        let first = effect.process(&input, 6, 6).expect("ink cut should process");
        let second = effect.process(&input, 6, 6).expect("ink cut should be deterministic");

        assert_eq!(first, second);
    }
}
