//! Angular facet contrast for a crisp cut-gem finish.

use super::{PixelBuffer, PostEffect, PostEffectError, validate_buffer_shape};
use crate::render::constants;
use rayon::prelude::*;

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

    #[inline]
    fn hash_signed(cell_x: usize, cell_y: usize, salt: u64) -> f64 {
        let mut n = (cell_x as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (cell_y as u64).wrapping_mul(0xA5A3_58B9_3217_58B5)
            ^ salt;
        n ^= n >> 32;
        n = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        n ^= n >> 29;
        let unit = (n >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        unit * 2.0 - 1.0
    }

    fn facet_value(&self, x: usize, y: usize) -> f64 {
        let cell = self.config.cell_size;
        let cell_x = x / cell;
        let cell_y = y / cell;
        let local_x = (x % cell) as f64 / cell as f64;
        let local_y = (y % cell) as f64 / cell as f64;
        let diagonal = if (cell_x + cell_y).is_multiple_of(2) {
            local_x - local_y
        } else {
            local_x + local_y - 1.0
        };
        let grain = Self::hash_signed(cell_x, cell_y, 0xC875_FAC7);
        (diagonal * 0.65 + grain * 0.35).clamp(-1.0, 1.0)
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
    fn test_crystal_facets_reject_invalid_buffer_shape() {
        let effect = CrystalFacetContrast::new(CrystalFacetConfig::default());
        let input = vec![(0.2, 0.2, 0.2, 1.0)];
        let err = effect.process(&input, 2, 2).expect_err("mismatched shape should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
    }
}
