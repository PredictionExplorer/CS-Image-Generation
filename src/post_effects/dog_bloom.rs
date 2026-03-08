//! Difference of Gaussians (DoG) bloom effect implementation.

use super::{PixelBuffer, PostEffect, utils};
use crate::render::{DogBloomConfig, apply_dog_bloom};
use rayon::prelude::*;
use std::error::Error;

/// Difference of Gaussians bloom post-processing effect.
///
/// Creates sharper, more defined bloom by subtracting two Gaussian blurs
/// of different radii, emphasizing edges and reducing overall haziness.
pub struct DogBloom {
    /// Configuration for the DoG algorithm.
    pub config: DogBloomConfig,

    /// Brightness multiplier for the core (unblurred) image.
    pub core_brightness: f64,

    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl DogBloom {
    /// Creates a new DoG bloom effect with the given configuration.
    pub fn new(config: DogBloomConfig, core_brightness: f64) -> Self {
        Self { config, core_brightness, enabled: true }
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

impl PostEffect for DogBloom {
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
        let dog_bloom = apply_dog_bloom(&highlights, width, height, &self.config);
        let core_gain = self.core_gain();

        // Composite the residual over the untouched base image.
        let mut output = Vec::with_capacity(input.len());
        output.par_extend(
            input.par_iter().zip(highlights.par_iter()).zip(dog_bloom.par_iter()).map(
                |(
                    (&(base_r, base_g, base_b, base_a), &(hr, hg, hb, _ha)),
                    &(dog_r, dog_g, dog_b, _dog_a),
                )| {
                    (
                        base_r + hr * core_gain + dog_r,
                        base_g + hg * core_gain + dog_g,
                        base_b + hb * core_gain + dog_b,
                        base_a,
                    )
                },
            ),
        );

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dog_bloom_preserves_dark_pixels() {
        let bloom = DogBloom::new(DogBloomConfig::default(), 12.0);
        let input = vec![(0.08, 0.08, 0.08, 1.0)];
        let output = bloom.process(&input, 1, 1).unwrap();

        assert!((output[0].0 - input[0].0).abs() < 1e-3);
        assert!((output[0].1 - input[0].1).abs() < 1e-3);
        assert!((output[0].2 - input[0].2).abs() < 1e-3);
    }
}
