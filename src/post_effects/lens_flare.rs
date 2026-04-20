//! Anamorphic lens-flare post-effect.
//!
//! Detects the brightest luminous centroid in the image (the focal attractor),
//! then paints a series of lightly tinted "ghosts" along the line through the
//! image centre, plus a horizontal anamorphic streak. The result evokes the
//! classic cinematic lens aesthetic without requiring per-frame motion data.
//!
//! The centroid detection is deterministic (serial scan), and all colour
//! derivations are closed-form given the configuration, so the effect is
//! reproducible across runs.

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// Configuration for the lens-flare post-effect.
#[derive(Clone, Debug)]
pub struct LensFlareConfig {
    /// Overall opacity of the flare (0 disables).
    pub strength: f64,
    /// Minimum luminance for a pixel to contribute to the centroid.
    pub luminance_threshold: f64,
    /// Number of ghost highlights along the flare axis.
    pub ghost_count: usize,
    /// Maximum ghost spread (fraction of image half-diagonal).
    pub ghost_spread: f64,
    /// Anamorphic streak strength.
    pub streak_strength: f64,
    /// Length of anamorphic streak as fraction of image width.
    pub streak_length: f64,
    /// Streak tint [r, g, b].
    pub streak_tint: [f64; 3],
    /// Ghost tint [r, g, b].
    pub ghost_tint: [f64; 3],
}

impl Default for LensFlareConfig {
    fn default() -> Self {
        Self {
            strength: 0.0,
            luminance_threshold: 0.35,
            ghost_count: 5,
            ghost_spread: 0.55,
            streak_strength: 0.25,
            streak_length: 0.30,
            streak_tint: [1.0, 0.96, 0.85],
            ghost_tint: [0.9, 0.95, 1.05],
        }
    }
}

/// Post-effect implementing the anamorphic lens flare.
pub struct LensFlare {
    /// Active configuration.
    pub config: LensFlareConfig,
}

impl LensFlare {
    /// Create a new lens flare effect.
    #[must_use]
    pub fn new(config: LensFlareConfig) -> Self {
        Self { config }
    }

    fn find_centroid(&self, input: &PixelBuffer, width: usize, height: usize) -> Option<(f64, f64, f64)> {
        let threshold = self.config.luminance_threshold.max(0.0);
        let mut sum_lum = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for (idx, p) in input.iter().enumerate() {
            let a = p.3.max(1e-6);
            let r = p.0 / a;
            let g = p.1 / a;
            let b = p.2 / a;
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            if lum < threshold {
                continue;
            }
            let weight = lum - threshold;
            let x = (idx % width) as f64;
            let y = (idx / width) as f64;
            sum_x += x * weight;
            sum_y += y * weight;
            sum_lum += weight;
        }
        let _ = height;
        if sum_lum <= 1e-6 {
            None
        } else {
            Some((sum_x / sum_lum, sum_y / sum_lum, sum_lum))
        }
    }
}

impl PostEffect for LensFlare {
    fn is_enabled(&self) -> bool {
        self.config.strength > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if !self.is_enabled() || width < 4 || height < 4 {
            return Ok(input.clone());
        }
        let Some((cx, cy, _)) = self.find_centroid(input, width, height) else {
            return Ok(input.clone());
        };

        let mut output = input.clone();
        let strength = self.config.strength.clamp(0.0, 1.5);
        let image_cx = width as f64 * 0.5;
        let image_cy = height as f64 * 0.5;
        let half_diag = ((width as f64).hypot(height as f64)) * 0.5;

        // Ghost highlights mirrored about image centre.
        let ghost_count = self.config.ghost_count.max(0);
        for i in 0..ghost_count {
            let t = (i as f64 + 1.0) / (ghost_count as f64 + 1.0);
            let offset = (t * 2.0 - 1.0) * self.config.ghost_spread;
            let gx = image_cx + (cx - image_cx) * -offset;
            let gy = image_cy + (cy - image_cy) * -offset;
            let radius = half_diag * 0.04 * (0.6 + (1.0 - (t - 0.5).abs()));
            let intensity = strength * (1.0 - (t - 0.5).abs()) * 0.6;
            let r_i = radius.ceil() as isize;
            let x0 = (gx.floor() as isize - r_i).max(0) as usize;
            let y0 = (gy.floor() as isize - r_i).max(0) as usize;
            let x1 = ((gx.floor() as isize + r_i) as usize).min(width.saturating_sub(1));
            let y1 = ((gy.floor() as isize + r_i) as usize).min(height.saturating_sub(1));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let dx = x as f64 - gx;
                    let dy = y as f64 - gy;
                    let d2 = dx * dx + dy * dy;
                    let falloff = (-d2 / (radius * radius * 0.5)).exp();
                    if falloff < 1e-4 {
                        continue;
                    }
                    let k = falloff * intensity;
                    let idx = y * width + x;
                    output[idx].0 += self.config.ghost_tint[0] * k;
                    output[idx].1 += self.config.ghost_tint[1] * k;
                    output[idx].2 += self.config.ghost_tint[2] * k;
                }
            }
        }

        // Anamorphic horizontal streak through the focal attractor.
        if self.config.streak_strength > 0.0 {
            let length = (self.config.streak_length * width as f64).max(4.0);
            let height_radius = (height as f64 * 0.002).max(0.75);
            let y_c = cy;
            let x0 = 0usize;
            let x1 = width - 1;
            let row_half = height_radius.ceil() as isize;
            let y_min = ((y_c.floor() as isize) - row_half).max(0) as usize;
            let y_max =
                (((y_c.floor() as isize) + row_half) as usize).min(height.saturating_sub(1));
            let streak_intensity = strength * self.config.streak_strength;
            for y in y_min..=y_max {
                let dy = (y as f64 - y_c) / height_radius.max(0.5);
                let row_fall = (-(dy * dy) * 1.5).exp();
                for x in x0..=x1 {
                    let dx = (x as f64 - cx).abs();
                    let col_fall = (-(dx * dx) / (length * length * 0.25)).exp();
                    let k = row_fall * col_fall * streak_intensity;
                    if k < 1e-4 {
                        continue;
                    }
                    let idx = y * width + x;
                    output[idx].0 += self.config.streak_tint[0] * k;
                    output[idx].1 += self.config.streak_tint[1] * k;
                    output[idx].2 += self.config.streak_tint[2] * k;
                }
            }
        }

        output.par_iter_mut().for_each(|p| {
            if p.3 < 0.0 {
                p.3 = 0.0;
            }
        });

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_input() {
        let lens = LensFlare::new(LensFlareConfig::default());
        assert!(!lens.is_enabled());
        let input = vec![(0.0, 0.0, 0.0, 0.0); 16 * 16];
        let out = lens.process(&input, 16, 16).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn no_bright_pixel_passes_through() {
        let cfg = LensFlareConfig { strength: 0.5, ..Default::default() };
        let lens = LensFlare::new(cfg);
        let input = vec![(0.05, 0.05, 0.05, 0.2); 32 * 32];
        let out = lens.process(&input, 32, 32).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn bright_pixel_produces_flare() {
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); 32 * 32];
        let center = 10 * 32 + 10;
        input[center] = (1.5, 1.5, 1.5, 1.0);
        let cfg =
            LensFlareConfig { strength: 0.8, ghost_count: 3, ..Default::default() };
        let lens = LensFlare::new(cfg);
        let out = lens.process(&input, 32, 32).unwrap();
        let sum_in: f64 = input.iter().map(|p| p.0 + p.1 + p.2).sum();
        let sum_out: f64 = out.iter().map(|p| p.0 + p.1 + p.2).sum();
        assert!(sum_out > sum_in, "flare should add luminous energy");
    }

    #[test]
    fn deterministic() {
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); 32 * 32];
        input[20 * 32 + 15] = (1.2, 1.2, 1.2, 1.0);
        let cfg = LensFlareConfig { strength: 0.7, ..Default::default() };
        let lens = LensFlare::new(cfg);
        let a = lens.process(&input, 32, 32).unwrap();
        let b = lens.process(&input, 32, 32).unwrap();
        assert_eq!(a, b);
    }
}
