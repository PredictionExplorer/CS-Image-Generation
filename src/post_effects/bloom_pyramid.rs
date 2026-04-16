//! Multi-tier bloom pyramid.
//!
//! Three Gaussian blur scales (tight / medium / wide) are stacked with
//! decreasing weights and composited additively on top of the untouched base
//! image. This produces a much richer highlight wash than a single-radius
//! Gaussian bloom: the tight layer keeps cores punchy while the wide layer
//! gives images their "film glow" halation.
//!
//! Used primarily by the Cinematic mood.

use super::{PixelBuffer, PostEffect, PostEffectError, utils};
use crate::render::parallel_blur_2d_rgba;
use rayon::prelude::*;

/// Multi-tier bloom pyramid post-processing effect.
#[derive(Debug)]
pub struct BloomPyramid {
    /// Tight radius in pixels (sharpest tier).
    pub radius_tight: usize,
    /// Medium radius in pixels.
    pub radius_medium: usize,
    /// Wide radius in pixels (softest halation tier).
    pub radius_wide: usize,
    /// Overall strength multiplier applied to the summed blurred highlights.
    pub strength: f64,
    /// Core brightness (same meaning as `GaussianBloom`).
    pub core_brightness: f64,
    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl BloomPyramid {
    /// Creates a new bloom pyramid with three tiers derived from a single
    /// base radius. The tight tier matches `base_radius`, medium = 2.2x, wide = 5x.
    #[must_use]
    pub fn new(base_radius: usize, strength: f64, core_brightness: f64) -> Self {
        let tight = base_radius.max(1);
        let medium = ((tight as f64) * 2.2).round() as usize;
        let wide = ((tight as f64) * 5.0).round() as usize;
        Self {
            radius_tight: tight,
            radius_medium: medium.max(tight + 1),
            radius_wide: wide.max(medium + 1),
            strength,
            core_brightness,
            enabled: true,
        }
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
                let luminance = crate::render::constants::rec709_luminance(sr, sg, sb);
                let factor = utils::highlight_extract_factor(luminance);
                (r * factor, g * factor, b * factor, a * factor)
            })
            .collect()
    }

    #[inline]
    fn core_gain(&self) -> f64 {
        (1.0 - (-0.05 * self.core_brightness.max(0.0)).exp()).min(0.55)
    }
}

impl PostEffect for BloomPyramid {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        let highlights = self.extract_highlights(input);
        let core_gain = self.core_gain();

        let mut tight = highlights.clone();
        parallel_blur_2d_rgba(&mut tight, width, height, self.radius_tight);
        let mut medium = highlights.clone();
        parallel_blur_2d_rgba(&mut medium, width, height, self.radius_medium);
        let mut wide = highlights.clone();
        parallel_blur_2d_rgba(&mut wide, width, height, self.radius_wide);

        // Weight the tiers so the tight layer dominates but wide halation is visible.
        let w_tight = 0.55 * self.strength;
        let w_med = 0.30 * self.strength;
        let w_wide = 0.18 * self.strength;

        let mut output = Vec::with_capacity(input.len());
        let iter = input
            .par_iter()
            .zip(highlights.par_iter())
            .zip(tight.par_iter())
            .zip(medium.par_iter())
            .zip(wide.par_iter())
            .map(
                |((((&(br, bg, bb, ba), &(hr, hg, hb, _)), &(tr, tg, tb, _)), &(mr, mg, mb, _)), &(wr, wg, wb, _))| {
                    (
                        br + hr * core_gain + tr * w_tight + mr * w_med + wr * w_wide,
                        bg + hg * core_gain + tg * w_tight + mg * w_med + wg * w_wide,
                        bb + hb * core_gain + tb * w_tight + mb * w_med + wb * w_wide,
                        ba,
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
    fn test_bloom_pyramid_preserves_dark_pixels() {
        let pyramid = BloomPyramid::new(2, 1.0, 12.0);
        let input = vec![(0.05, 0.05, 0.05, 1.0); 4];
        let out = pyramid.process(&input, 2, 2).expect("process ok");
        // Dark pixels should change only by the tiny highlight leak.
        for (p_in, p_out) in input.iter().zip(out.iter()) {
            assert!(p_out.0 <= p_in.0 + 0.02);
        }
    }

    #[test]
    fn test_bloom_pyramid_lifts_bright_pixel() {
        let pyramid = BloomPyramid::new(1, 1.0, 12.0);
        let mut input = vec![(0.0, 0.0, 0.0, 1.0); 9];
        input[4] = (2.0, 2.0, 2.0, 1.0);
        let out = pyramid.process(&input, 3, 3).expect("process ok");
        // Neighbors should be brighter than input (they were 0.0).
        assert!(out[0].0 > 0.0 || out[1].0 > 0.0 || out[3].0 > 0.0);
    }

    #[test]
    fn test_radii_are_increasing() {
        let p = BloomPyramid::new(3, 1.0, 12.0);
        assert!(p.radius_tight < p.radius_medium);
        assert!(p.radius_medium < p.radius_wide);
    }
}
