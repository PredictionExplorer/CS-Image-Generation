//! Angular facet contrast for a crisp cut-gem finish.

use super::{PixelBuffer, PostEffect, PostEffectError, validate_buffer_shape};
use crate::render::constants;
use rayon::prelude::*;
use std::f64::consts::TAU;

/// Configuration for angular crystal facet contrast.
#[derive(Clone, Debug)]
pub struct CrystalFacetConfig {
    /// Overall contrast modulation strength.
    pub strength: f64,
    /// Facet cell size in pixels.
    pub cell_size: usize,
    /// Luminance where the facet finish starts appearing.
    pub threshold: f64,
    /// Additional lift for bright facet planes.
    pub highlight_gain: f64,
}

impl Default for CrystalFacetConfig {
    fn default() -> Self {
        Self { strength: 0.08, cell_size: 18, threshold: 0.18, highlight_gain: 0.05 }
    }
}

/// Adds deterministic angular contrast planes without spatial blur.
pub struct CrystalFacetContrast {
    config: CrystalFacetConfig,
    enabled: bool,
}

impl CrystalFacetContrast {
    /// Creates a new crystal facet contrast effect.
    #[must_use]
    pub fn new(config: CrystalFacetConfig) -> Self {
        let enabled = config.strength > 0.0 && config.cell_size >= 2;
        Self { config, enabled }
    }

    #[inline]
    fn luminance(r: f64, g: f64, b: f64) -> f64 {
        constants::rec709_luminance(r, g, b)
    }

    fn facet_value(&self, x: usize, y: usize) -> f64 {
        let scale = self.config.cell_size as f64;
        let xf = x as f64 / scale;
        let yf = y as f64 / scale;

        // Use continuous, angled bands instead of per-cell random values. The old
        // grid-cell hash created visible square seams at `cell_size` boundaries.
        let wave_a = (TAU * (xf * 0.73 + yf * 0.41)).sin();
        let wave_b = (TAU * (xf * -0.29 + yf * 0.91) / 1.618).sin();
        let wave_c = (TAU * (xf * 0.17 - yf * 0.67) / 2.414).sin();

        (wave_a * 0.50 + wave_b * 0.32 + wave_c * 0.18).clamp(-1.0, 1.0)
    }
}

impl PostEffect for CrystalFacetContrast {
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

        let output = input
            .par_iter()
            .enumerate()
            .map(|(idx, &(r, g, b, a))| {
                if a <= 0.0 {
                    return (r, g, b, a);
                }
                let lum = Self::luminance(r, g, b);
                let gate =
                    ((lum - self.config.threshold) / (1.0 - self.config.threshold)).clamp(0.0, 1.0);
                if gate <= 0.0 {
                    return (r, g, b, a);
                }

                let x = idx % width;
                let y = idx / width;
                let facet = self.facet_value(x, y);
                let lift = facet.max(0.0) * self.config.highlight_gain * gate;
                let modulation = 1.0 + facet * self.config.strength * gate + lift;

                (
                    (r * modulation).clamp(0.0, 1.35),
                    (g * modulation).clamp(0.0, 1.35),
                    (b * modulation).clamp(0.0, 1.35),
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
    fn test_crystal_facets_disabled_with_zero_strength() {
        let effect =
            CrystalFacetContrast::new(CrystalFacetConfig { strength: 0.0, ..Default::default() });
        assert!(!effect.is_enabled());
    }

    #[test]
    fn test_crystal_facets_add_deterministic_variation() {
        let effect = CrystalFacetContrast::new(CrystalFacetConfig {
            strength: 0.25,
            cell_size: 4,
            threshold: 0.0,
            highlight_gain: 0.08,
        });
        let input = vec![(0.55, 0.45, 0.35, 1.0); 64];

        let first = effect.process(&input, 8, 8).expect("facet effect should process");
        let second = effect.process(&input, 8, 8).expect("facet effect should be deterministic");

        assert_eq!(first, second);
        assert!(first.windows(2).any(|pair| (pair[0].0 - pair[1].0).abs() > 1e-6));
    }

    #[test]
    fn test_crystal_facets_do_not_collapse_low_alpha_display_pixels() {
        let effect = CrystalFacetContrast::new(CrystalFacetConfig {
            strength: 0.20,
            cell_size: 4,
            threshold: 0.0,
            highlight_gain: 0.04,
        });
        let input = vec![(0.6, 0.5, 0.4, 1e-7); 64];

        let output = effect.process(&input, 8, 8).expect("facet effect should process");

        assert!(
            output.iter().any(|pixel| pixel.0 > 0.1),
            "display RGB should stay visible even when alpha is tiny",
        );
        assert!(output.iter().all(|pixel| pixel.3 == 1e-7), "alpha should be preserved");
    }

    #[test]
    fn test_crystal_facets_do_not_imprint_square_cell_boundaries() {
        let effect = CrystalFacetContrast::new(CrystalFacetConfig {
            strength: 0.30,
            cell_size: 14,
            threshold: 0.0,
            highlight_gain: 0.08,
        });
        let width = 140;
        let height = 140;
        let input = vec![(0.55, 0.48, 0.36, 1.0); width * height];

        let output = effect.process(&input, width, height).expect("facet effect should process");

        let mut boundary = Vec::new();
        let mut interior = Vec::new();
        for y in 0..height {
            for x in 1..width {
                let diff = (output[y * width + x].0 - output[y * width + x - 1].0).abs();
                if x % 14 == 0 {
                    boundary.push(diff);
                } else {
                    interior.push(diff);
                }
            }
        }
        let boundary_mean = boundary.iter().sum::<f64>() / boundary.len() as f64;
        let interior_mean = interior.iter().sum::<f64>() / interior.len() as f64;

        assert!(
            boundary_mean <= interior_mean * 1.25,
            "facet finish should not create square-cell seams (boundary={boundary_mean:.6}, interior={interior_mean:.6})",
        );
    }

    #[test]
    fn test_crystal_facets_reject_invalid_buffer_shape() {
        let effect = CrystalFacetContrast::new(CrystalFacetConfig::default());
        let input = vec![(0.2, 0.2, 0.2, 1.0)];
        let err = effect.process(&input, 2, 2).expect_err("mismatched shape should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
    }
}
