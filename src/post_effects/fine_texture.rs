//! Fine art texture overlay for tactile richness and material quality.
//!
//! Adds subtle procedural grain that simulates fine art materials without
//! directional weave or screen-door artifacts.
//!
//! These textures add a sense of physicality and craftsmanship to digital renders.

use super::{PixelBuffer, PostEffect, PostEffectError, validate_buffer_shape};
use rayon::prelude::*;

/// Configuration for fine texture overlay
#[derive(Clone, Debug)]
pub struct FineTextureConfig {
    /// Overall strength of the texture (0.0-1.0)
    pub strength: f64,
    /// Scale of texture features (larger = coarser texture)
    pub scale: f64,
    /// Contrast of texture variations
    pub contrast: f64,
    /// Directional anisotropy (0.0 = isotropic, 1.0 = strongly directional)
    pub anisotropy: f64,
    /// Angle for directional features (in degrees)
    pub angle: f64,
}

impl Default for FineTextureConfig {
    fn default() -> Self {
        let base_scale = (1920.0_f64 * 1080.0).sqrt();
        Self {
            strength: 0.12,
            scale: base_scale * 0.0018,
            contrast: 0.35,
            anisotropy: 0.25,
            angle: 45.0,
        }
    }
}

/// Fine texture post-effect
pub struct FineTexture {
    config: FineTextureConfig,
    enabled: bool,
}

impl FineTexture {
    /// Creates a new fine texture overlay effect from the given configuration.
    #[must_use]
    pub fn new(config: FineTextureConfig) -> Self {
        let enabled = config.strength > 0.0;
        Self { config, enabled }
    }

    /// Fast 2D hash function
    #[inline]
    fn hash2d(x: f64, y: f64) -> f64 {
        let h = (x * 127.1 + y * 311.7).sin() * 43758.5453;
        let unit = h - h.floor();
        unit * 2.0 - 1.0
    }

    /// Value noise (smooth interpolated noise)
    fn value_noise(x: f64, y: f64) -> f64 {
        let ix = x.floor();
        let iy = y.floor();
        let fx = x - ix;
        let fy = y - iy;

        // Smoothstep interpolation
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sy = fy * fy * (3.0 - 2.0 * fy);

        // Corner values
        let v00 = Self::hash2d(ix, iy);
        let v10 = Self::hash2d(ix + 1.0, iy);
        let v01 = Self::hash2d(ix, iy + 1.0);
        let v11 = Self::hash2d(ix + 1.0, iy + 1.0);

        // Bilinear interpolation
        let v0 = v00 * (1.0 - sx) + v10 * sx;
        let v1 = v01 * (1.0 - sx) + v11 * sx;
        v0 * (1.0 - sy) + v1 * sy
    }

    /// Isotropic paper-grain pattern with no directional weave.
    fn canvas_pattern(&self, x: f64, y: f64) -> f64 {
        let scale = 1.0 / self.config.scale.max(1e-6);

        let fine = Self::value_noise(x * scale * 17.0 + 19.7, y * scale * 17.0 - 3.1) * 0.55;
        let mid = Self::value_noise(x * scale * 7.0 + 101.3, y * scale * 7.0 + 47.9) * 0.30;
        let coarse = Self::value_noise(x * scale * 2.7 - 61.0, y * scale * 2.7 + 13.0) * 0.15;

        fine + mid + coarse
    }

    /// Get texture value for a given position
    fn get_texture_value(&self, x: f64, y: f64) -> f64 {
        let raw_value = self.canvas_pattern(x, y);

        raw_value * self.config.contrast
    }
}

impl PostEffect for FineTexture {
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

        let output: PixelBuffer = input
            .par_iter()
            .enumerate()
            .map(|(idx, &(r, g, b, a))| {
                if a <= 0.0 {
                    return (r, g, b, a);
                }

                let x = (idx % width) as f64;
                let y = (idx / width) as f64;

                // Get texture modulation
                let texture = self.get_texture_value(x, y);
                let modulation = 1.0 + texture * self.config.strength;

                // Apply texture modulation (multiplicative for natural look)
                let final_r = (r * modulation).max(0.0);
                let final_g = (g * modulation).max(0.0);
                let final_b = (b * modulation).max(0.0);

                (final_r, final_g, final_b, a)
            })
            .collect();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_disabled() {
        let config = FineTextureConfig { strength: 0.0, ..FineTextureConfig::default() };
        let texture = FineTexture::new(config);
        assert!(!texture.is_enabled());
    }

    #[test]
    fn test_texture_enabled() {
        let config = FineTextureConfig::default();
        let texture = FineTexture::new(config);
        assert!(texture.is_enabled());
    }

    #[test]
    fn test_hash_determinism() {
        let h1 = FineTexture::hash2d(100.0, 200.0);
        let h2 = FineTexture::hash2d(100.0, 200.0);
        assert_eq!(h1, h2, "Hash should be deterministic");
    }

    #[test]
    fn test_buffer_processing() {
        let config = FineTextureConfig::default();
        let texture = FineTexture::new(config);

        // Create uniform test buffer
        let buffer: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0); 10000];
        let result =
            texture.process(&buffer, 100, 100).expect("fine texture process should succeed");

        // Verify texture has been applied (values should vary)
        assert_eq!(result.len(), buffer.len());
        let has_variation = result.windows(2).any(|w| (w[0].0 - w[1].0).abs() > 0.001);
        assert!(has_variation, "Texture should add variation");
    }

    #[test]
    fn test_canvas_produces_varying_values() {
        let config = FineTextureConfig {
            strength: 0.1,
            scale: 10.0,
            contrast: 0.5,
            anisotropy: 0.0,
            angle: 0.0,
        };
        let texture = FineTexture::new(config);

        assert!(texture.is_enabled());

        let v1 = texture.get_texture_value(0.0, 0.0);
        let v2 = texture.get_texture_value(10.0, 10.0);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_texture_is_not_directional_weave() {
        let texture = FineTexture::new(FineTextureConfig {
            strength: 0.2,
            scale: 12.0,
            contrast: 0.6,
            anisotropy: 1.0,
            angle: 0.0,
        });
        let input = vec![(0.5, 0.5, 0.5, 1.0); 128 * 128];

        let output = texture.process(&input, 128, 128).expect("texture should process");

        let mut horizontal = Vec::new();
        let mut vertical = Vec::new();
        for y in 1..127 {
            for x in 1..127 {
                let idx = y * 128 + x;
                horizontal.push((output[idx].0 - output[idx - 1].0).abs());
                vertical.push((output[idx].0 - output[idx - 128].0).abs());
            }
        }
        let h_mean = horizontal.iter().sum::<f64>() / horizontal.len() as f64;
        let v_mean = vertical.iter().sum::<f64>() / vertical.len() as f64;
        let ratio = h_mean / v_mean.max(1e-12);

        assert!(
            (0.70..=1.30).contains(&ratio),
            "texture should be isotropic grain, not horizontal/vertical weave (ratio={ratio:.3})",
        );
    }

    #[test]
    fn test_fine_texture_rejects_invalid_buffer_shape() {
        let texture = FineTexture::new(FineTextureConfig::default());
        let buffer: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0); 99];

        let err = texture.process(&buffer, 10, 10).expect_err("mismatched buffer should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
        assert!(err.to_string().contains("FineTexture"));
        assert!(err.to_string().contains("buffer length"));
    }
}
