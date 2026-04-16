//! Four-point diffraction spikes on very bright pixels.
//!
//! For each pixel whose luminance exceeds `threshold`, stamp a small spike
//! cross (optionally rotated) across it. Unlike the full lens-flare pass,
//! this runs only on the brightest ~1% of pixels (body cores, brightest
//! highlights, and very bright stars from [`StarField`]) and produces the
//! classic "telescope" look.

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// Four-point diffraction spike post-effect.
#[derive(Clone, Debug)]
pub struct DiffractionSpikes {
    /// Blend strength (0 disables).
    pub strength: f64,
    /// Luminance threshold on the un-premultiplied pixel.
    pub threshold: f64,
    /// Half-length of each spike arm in pixels.
    pub arm_length: usize,
    /// Rotation angle in radians (0 = axis-aligned `+`, π/4 = `x`).
    pub rotation: f64,
    /// Whether to also draw the 45°-rotated secondary spikes (8-point star).
    pub eight_point: bool,
    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl Default for DiffractionSpikes {
    fn default() -> Self {
        Self {
            strength: 0.25,
            threshold: 0.90,
            arm_length: 22,
            rotation: 0.0,
            eight_point: true,
            enabled: true,
        }
    }
}

impl DiffractionSpikes {
    /// Create a cinema-telescope variant tuned for the given image dimensions.
    #[must_use]
    pub fn for_image(width: usize, height: usize, strength: f64) -> Self {
        let diag = ((width * width + height * height) as f64).sqrt();
        let arm = (diag / 110.0).round() as usize;
        Self {
            strength: strength.clamp(0.0, 2.0),
            threshold: 0.88,
            arm_length: arm.clamp(8, 80),
            rotation: 0.0,
            eight_point: true,
            enabled: true,
        }
    }

    #[inline]
    fn premul_luma(p: (f64, f64, f64, f64)) -> f64 {
        let a = p.3.max(1e-9);
        0.2126 * (p.0 / a) + 0.7152 * (p.1 / a) + 0.0722 * (p.2 / a)
    }
}

impl PostEffect for DiffractionSpikes {
    fn is_enabled(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if self.arm_length == 0 {
            return Ok(input.clone());
        }
        let mut output = input.clone();
        let thr = self.threshold;
        let (c, s) = (self.rotation.cos(), self.rotation.sin());
        let arms: Vec<(f64, f64)> = if self.eight_point {
            vec![
                (c, s),
                (-c, -s),
                (-s, c),
                (s, -c),
                ((c - s) * std::f64::consts::FRAC_1_SQRT_2, (s + c) * std::f64::consts::FRAC_1_SQRT_2),
                (-(c - s) * std::f64::consts::FRAC_1_SQRT_2, -(s + c) * std::f64::consts::FRAC_1_SQRT_2),
                ((c + s) * std::f64::consts::FRAC_1_SQRT_2, (s - c) * std::f64::consts::FRAC_1_SQRT_2),
                (-(c + s) * std::f64::consts::FRAC_1_SQRT_2, -(s - c) * std::f64::consts::FRAC_1_SQRT_2),
            ]
        } else {
            vec![(c, s), (-c, -s), (-s, c), (s, -c)]
        };

        // Find bright source pixels first; stamping is then straightforward.
        let sources: Vec<(usize, usize, (f64, f64, f64, f64))> = (0..height)
            .into_par_iter()
            .flat_map_iter(|y| {
                (0..width).filter_map(move |x| {
                    let p = input[y * width + x];
                    let lum = Self::premul_luma(p);
                    if lum >= thr {
                        Some((x, y, p))
                    } else {
                        None
                    }
                })
            })
            .collect();

        let arm_len = self.arm_length as i64;
        let strength = self.strength;

        for (sx, sy, p) in sources {
            let lum = Self::premul_luma(p);
            let gate = ((lum - thr) / (1.0 - thr + 1e-6)).clamp(0.0, 1.0);
            for &(ax, ay) in &arms {
                for t in 1..=arm_len {
                    let px = sx as i64 + (ax * t as f64).round() as i64;
                    let py = sy as i64 + (ay * t as f64).round() as i64;
                    if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                        break;
                    }
                    let falloff = 1.0 - (t as f64 / arm_len as f64);
                    let w = strength * gate * falloff * falloff;
                    if w < 0.001 {
                        continue;
                    }
                    let dst = &mut output[py as usize * width + px as usize];
                    dst.0 = (dst.0 + p.0 * w).min(2.0);
                    dst.1 = (dst.1 + p.1 * w).min(2.0);
                    dst.2 = (dst.2 + p.2 * w).min(2.0);
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffraction_spikes_on_hot_centre() {
        let w = 21;
        let h = 21;
        let mut input = vec![(0.0, 0.0, 0.0, 1.0); w * h];
        input[10 * w + 10] = (1.5, 1.5, 1.5, 1.0);
        let ds = DiffractionSpikes {
            strength: 0.5,
            threshold: 0.5,
            arm_length: 5,
            rotation: 0.0,
            eight_point: false,
            enabled: true,
        };
        let out = ds.process(&input, w, h).expect("ok");
        // Pixel on the horizontal arm should have picked up energy.
        let arm_pixel = out[10 * w + 13];
        assert!(arm_pixel.0 > 0.0, "horizontal spike arm should be lit");
        let arm_pixel2 = out[10 * w + 7];
        assert!(arm_pixel2.0 > 0.0, "left horizontal spike arm should be lit");
        // Pixel far off-axis should remain dark.
        let off_axis = out[3 * w + 3];
        assert_eq!(off_axis, (0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_zero_strength_passthrough() {
        let w = 9;
        let h = 9;
        let mut input = vec![(0.0, 0.0, 0.0, 1.0); w * h];
        input[4 * w + 4] = (1.5, 1.5, 1.5, 1.0);
        let ds = DiffractionSpikes { strength: 0.0, ..Default::default() };
        let out = ds.process(&input, w, h).expect("ok");
        assert!(!ds.is_enabled());
        assert_eq!(out, input);
    }
}
