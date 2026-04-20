//! Procedural starfield post-effect.
//!
//! Sprinkles deterministic, hash-positioned star points onto the background.
//! Each star has a spectral color, a brightness drawn from a power-law
//! distribution (mimicking real stellar magnitude counts), and a soft
//! spherical profile so individual pixels feel like luminous point sources
//! rather than solid squares.
//!
//! The effect is fully deterministic: given the same `seed` and dimensions,
//! the same star pattern is produced every time. No RNG state is mutated at
//! render time, so it is trivially thread-safe.

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// Configuration for the procedural starfield effect.
#[derive(Clone, Debug)]
pub struct StarfieldConfig {
    /// Overall opacity/strength of the starfield (0 disables).
    pub strength: f64,
    /// Average number of stars per million pixels.
    pub density: f64,
    /// Minimum brightness of a star (0..1).
    pub min_brightness: f64,
    /// Maximum brightness of a star (0..1).
    pub max_brightness: f64,
    /// Radius in pixels of the brightest stars (sub-pixel soft edges).
    pub max_radius: f64,
    /// Warmth bias: `0.0` = neutral white, `>0` tints toward warm tones,
    /// `<0` toward cool. Range `[-1, 1]`.
    pub warmth_bias: f64,
    /// Seed used to position and colour stars.
    pub seed: u64,
    /// Fade stars away from existing luminous regions (0 = no fade).
    pub avoid_luminous_regions: f64,
}

impl Default for StarfieldConfig {
    fn default() -> Self {
        Self {
            strength: 0.0,
            density: 160.0,
            min_brightness: 0.04,
            max_brightness: 0.95,
            max_radius: 1.6,
            warmth_bias: 0.0,
            seed: 0,
            avoid_luminous_regions: 0.35,
        }
    }
}

/// Post-effect implementing the starfield.
pub struct Starfield {
    /// Active configuration.
    pub config: StarfieldConfig,
}

impl Starfield {
    /// Create a new starfield post-effect.
    #[must_use]
    pub fn new(config: StarfieldConfig) -> Self {
        Self { config }
    }
}

/// 64-bit integer hash (`SplitMix64` finaliser).
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn hash_u01(seed: u64, a: u64, b: u64) -> f64 {
    let mix = splitmix64(seed ^ a.wrapping_mul(0x9E37_79B9) ^ b.wrapping_mul(0x8528_7F34));
    (mix >> 11) as f64 / (1u64 << 53) as f64
}

fn star_color(idx: u64, seed: u64, warmth_bias: f64) -> (f64, f64, f64) {
    let r_bias = hash_u01(seed ^ 0xA1B2, idx, 1);
    // Shift along a warm/cool axis. Positive warmth pushes toward 580K gold,
    // negative pushes toward icy blue-white.
    let t = (r_bias - 0.5) * 0.8 + warmth_bias * 0.4;
    let t = t.clamp(-0.5, 0.5);
    let r = 0.95 + t * 0.25;
    let g = 0.95 - t.abs() * 0.08;
    let b = 0.95 - t * 0.35;
    (r.clamp(0.0, 1.2), g.clamp(0.0, 1.2), b.clamp(0.0, 1.2))
}

impl PostEffect for Starfield {
    fn is_enabled(&self) -> bool {
        self.config.strength > 0.0 && self.config.density > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if !self.is_enabled() || width == 0 || height == 0 {
            return Ok(input.clone());
        }

        let pixels = (width as f64) * (height as f64);
        let star_count = (self.config.density * pixels / 1_000_000.0).round() as u64;
        if star_count == 0 {
            return Ok(input.clone());
        }

        let mut output = input.clone();
        let min_b = self.config.min_brightness.clamp(0.0, 1.0);
        let max_b = self.config.max_brightness.clamp(min_b, 2.0);
        let max_radius = self.config.max_radius.max(0.3);
        let strength = self.config.strength.clamp(0.0, 1.5);
        let avoid = self.config.avoid_luminous_regions.clamp(0.0, 1.0);

        // Compute per-star splats sequentially; each star affects a small
        // neighbourhood, so we batch the final overlay as a parallel additive
        // pass using the splats vector.
        let splats: Vec<(f64, f64, f64, f64, f64, f64)> = (0..star_count)
            .map(|i| {
                let fx = hash_u01(self.config.seed, i, 0);
                let fy = hash_u01(self.config.seed, i, 1);
                let raw_b = hash_u01(self.config.seed, i, 2);
                // Power-law distribution favouring dim stars (realism).
                let gamma = 3.0;
                let norm_b = raw_b.powf(gamma);
                let brightness = min_b + norm_b * (max_b - min_b);
                let r_frac = hash_u01(self.config.seed, i, 3);
                let radius = (0.4 + r_frac * max_radius).max(0.35);
                let (cr, cg, cb) = star_color(i, self.config.seed, self.config.warmth_bias);
                (fx * width as f64, fy * height as f64, brightness, cr * brightness, cg * brightness, cb * brightness + radius * 0.0)
            })
            .collect();

        // Accumulate splats into output.
        for &(cx, cy, brightness, cr, cg, cb) in &splats {
            let radius = 0.4 + brightness * max_radius;
            let r_int = radius.ceil() as isize;
            let x0 = (cx as isize - r_int).max(0);
            let y0 = (cy as isize - r_int).max(0);
            let x1 = ((cx as isize + r_int) as usize).min(width - 1);
            let y1 = ((cy as isize + r_int) as usize).min(height - 1);
            for y in y0 as usize..=y1 {
                for x in x0 as usize..=x1 {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let d2 = dx * dx + dy * dy;
                    let falloff = (-d2 / (radius * radius * 0.5)).exp();
                    if falloff < 1e-4 {
                        continue;
                    }
                    let idx = y * width + x;
                    let existing = output[idx];
                    let existing_lum = 0.2126 * existing.0 + 0.7152 * existing.1 + 0.0722 * existing.2;
                    let mask = (1.0 - existing_lum.min(1.0) * avoid).max(0.0);
                    let k = falloff * strength * mask;
                    output[idx].0 += cr * k;
                    output[idx].1 += cg * k;
                    output[idx].2 += cb * k;
                    if existing.3 <= 0.0 {
                        output[idx].3 = k.min(1.0);
                    }
                }
            }
        }

        // Re-premultiply sane alphas.
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

    fn black(w: usize, h: usize) -> PixelBuffer {
        vec![(0.0, 0.0, 0.0, 0.0); w * h]
    }

    #[test]
    fn disabled_when_strength_is_zero() {
        let cfg = StarfieldConfig::default();
        let sf = Starfield::new(cfg);
        assert!(!sf.is_enabled());
        let input = black(32, 32);
        let out = sf.process(&input, 32, 32).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn strength_nonzero_adds_light() {
        let cfg = StarfieldConfig { strength: 0.5, density: 5_000.0, seed: 42, ..Default::default() };
        let sf = Starfield::new(cfg);
        let input = black(64, 64);
        let out = sf.process(&input, 64, 64).unwrap();
        let total: f64 = out.iter().map(|p| p.0 + p.1 + p.2).sum();
        assert!(total > 0.0, "starfield should add luminous energy");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let cfg =
            StarfieldConfig { strength: 0.5, density: 5_000.0, seed: 123, ..Default::default() };
        let sf = Starfield::new(cfg);
        let input = black(48, 48);
        let a = sf.process(&input, 48, 48).unwrap();
        let b = sf.process(&input, 48, 48).unwrap();
        assert_eq!(a, b, "starfield must be deterministic");
    }

    #[test]
    fn different_seeds_diverge() {
        let input = black(48, 48);
        let a = Starfield::new(StarfieldConfig {
            strength: 0.5,
            density: 5_000.0,
            seed: 1,
            ..Default::default()
        })
        .process(&input, 48, 48)
        .unwrap();
        let b = Starfield::new(StarfieldConfig {
            strength: 0.5,
            density: 5_000.0,
            seed: 2,
            ..Default::default()
        })
        .process(&input, 48, 48)
        .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn monotonic_in_strength() {
        let input = black(48, 48);
        let weak = Starfield::new(StarfieldConfig {
            strength: 0.2,
            density: 5_000.0,
            seed: 9,
            ..Default::default()
        })
        .process(&input, 48, 48)
        .unwrap();
        let strong = Starfield::new(StarfieldConfig {
            strength: 0.8,
            density: 5_000.0,
            seed: 9,
            ..Default::default()
        })
        .process(&input, 48, 48)
        .unwrap();
        let sum_weak: f64 = weak.iter().map(|p| p.0 + p.1 + p.2).sum();
        let sum_strong: f64 = strong.iter().map(|p| p.0 + p.1 + p.2).sum();
        assert!(sum_strong > sum_weak);
    }

    #[test]
    fn alpha_zero_passthrough_for_zero_strength() {
        let cfg =
            StarfieldConfig { strength: 0.0, density: 5_000.0, seed: 1, ..Default::default() };
        let sf = Starfield::new(cfg);
        let input = black(16, 16);
        let out = sf.process(&input, 16, 16).unwrap();
        assert_eq!(out, input);
    }
}
