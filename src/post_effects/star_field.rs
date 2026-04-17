//! Procedural star field background.
//!
//! Generates a deterministic star layout seeded from the simulation noise seed
//! using a simple rejection-sampled pseudo-poisson distribution, then paints
//! each star as a tiny Gaussian splat (airy-disc-ish) directly into the
//! trajectory buffer (before bloom so the bloom lights up the brighter stars).
//!
//! Used primarily by the Cosmic mood.

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// A single procedural star.
#[derive(Clone, Copy, Debug)]
struct Star {
    x: f32,
    y: f32,
    /// Brightness scale in linear energy. Few stars are very bright; most dim.
    brightness: f32,
    /// Airy/Gaussian PSF radius in pixels (larger for brighter stars).
    radius: f32,
    /// Color temperature bias: negative = cooler/blue, positive = warmer/red.
    temp_bias: f32,
}

/// Procedural star field post-effect.
#[derive(Clone, Debug)]
pub struct StarField {
    /// Deterministic stars, generated once at construction.
    stars: Vec<Star>,
    /// Overall blend strength; stars are additive.
    pub strength: f64,
    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl StarField {
    /// Build a star field for the given image dimensions using a deterministic
    /// `seed`. Density scales with image area (about 1 star per 1600 px²).
    #[must_use]
    pub fn new(width: usize, height: usize, seed: i32, strength: f64) -> Self {
        let area = width * height;
        let target = (area as f64 / 1600.0).round() as usize;
        let target = target.clamp(50, 3000);
        let mut stars = Vec::with_capacity(target);
        let mut state = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xCAFE);
        for _ in 0..target {
            let rx = next_u64(&mut state);
            let ry = next_u64(&mut state);
            let rb = next_u64(&mut state);
            let rt = next_u64(&mut state);
            let x = ((rx as u32) % width as u32) as f32 + 0.5;
            let y = ((ry as u32) % height as u32) as f32 + 0.5;
            // Star brightness follows an exponential distribution so most are
            // faint and a few are very bright.
            let u01 = (rb >> 11) as f64 / (1u64 << 53) as f64;
            let bright = (-u01.ln()).min(8.0) as f32;
            let radius = (0.6 + bright * 0.25).clamp(0.6, 3.5);
            let temp = ((rt >> 11) as i64 as f32) / (1i64 << 53) as f32;
            let temp_bias = (temp * 2.0 - 1.0).clamp(-1.0, 1.0);
            stars.push(Star { x, y, brightness: bright, radius, temp_bias });
        }
        Self { stars, strength, enabled: true }
    }

    /// Number of stars (testing hook).
    #[must_use]
    pub fn star_count(&self) -> usize {
        self.stars.len()
    }
}

impl PostEffect for StarField {
    fn is_enabled(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if self.stars.is_empty() {
            return Ok(input.clone());
        }
        let mut output = input.clone();
        let strength = self.strength as f32;

        // Draw each star by stamping a small Gaussian splat. We accumulate into
        // a per-pixel (dr, dg, db) delta that we then apply in one parallel pass.
        let wu = width;
        let hu = height;
        let mut deltas = vec![(0.0f32, 0.0f32, 0.0f32); wu * hu];
        for star in &self.stars {
            let r = star.radius;
            let pad = (r * 2.8).ceil() as i32;
            let ix = star.x as i32;
            let iy = star.y as i32;
            let (tint_r, tint_g, tint_b) = temp_to_tint(star.temp_bias);
            let amp = star.brightness * strength;
            let min_x = (ix - pad).max(0);
            let max_x = (ix + pad).min(wu as i32 - 1);
            let min_y = (iy - pad).max(0);
            let max_y = (iy + pad).min(hu as i32 - 1);
            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let dx = px as f32 - star.x;
                    let dy = py as f32 - star.y;
                    let d2 = dx * dx + dy * dy;
                    let e = (-d2 / (r * r)).exp();
                    if e < 0.02 {
                        continue;
                    }
                    let g = amp * e;
                    let d = &mut deltas[py as usize * wu + px as usize];
                    d.0 += g * tint_r;
                    d.1 += g * tint_g;
                    d.2 += g * tint_b;
                }
            }
        }

        output.par_iter_mut().zip(deltas.par_iter()).for_each(|(p, &(dr, dg, db))| {
            // Stars are added in straight-alpha space; because we don't know
            // the alpha of the existing pixel here (it's premultiplied), we
            // just lift RGB and leave alpha unchanged.
            p.0 += f64::from(dr);
            p.1 += f64::from(dg);
            p.2 += f64::from(db);
        });

        Ok(output)
    }
}

/// Deterministic xorshift64* PRNG.
#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

/// Map a bias `[-1, 1]` to a (r, g, b) tint. -1 → cool blue-white, 0 → white,
/// +1 → warm orange-yellow. Values are multiplied by `brightness` so very
/// bright stars get close to white regardless.
#[inline]
fn temp_to_tint(bias: f32) -> (f32, f32, f32) {
    if bias < 0.0 {
        let t = (-bias).min(1.0);
        (1.0 - 0.35 * t, 1.0 - 0.10 * t, 1.0)
    } else {
        let t = bias.min(1.0);
        (1.0, 1.0 - 0.15 * t, 1.0 - 0.40 * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_field_has_stars_and_is_deterministic() {
        let a = StarField::new(320, 180, 12345, 0.5);
        let b = StarField::new(320, 180, 12345, 0.5);
        assert_eq!(a.star_count(), b.star_count());
        assert!(a.star_count() > 10);
        assert_eq!(a.stars[0].x, b.stars[0].x);
        assert_eq!(a.stars[0].y, b.stars[0].y);
    }

    #[test]
    fn test_star_field_different_seeds_differ() {
        let a = StarField::new(320, 180, 1, 0.5);
        let b = StarField::new(320, 180, 2, 0.5);
        // With at least ~36 stars, positions should differ somewhere.
        let any_diff = (0..a.star_count().min(b.star_count())).any(|i| {
            (a.stars[i].x - b.stars[i].x).abs() > 0.5 || (a.stars[i].y - b.stars[i].y).abs() > 0.5
        });
        assert!(any_diff, "different seeds should produce different star positions");
    }

    #[test]
    fn test_zero_strength_passthrough() {
        let field = StarField::new(64, 36, 42, 0.0);
        let input = vec![(0.0, 0.0, 0.0, 1.0); 64 * 36];
        let out = field.process(&input, 64, 36).expect("ok");
        assert_eq!(out, input);
    }

    #[test]
    fn test_lifts_some_pixels() {
        let field = StarField::new(128, 72, 7, 1.0);
        let input = vec![(0.0, 0.0, 0.0, 1.0); 128 * 72];
        let out = field.process(&input, 128, 72).expect("ok");
        let any_lifted = out.iter().any(|p| p.0 > 0.0 || p.1 > 0.0 || p.2 > 0.0);
        assert!(any_lifted, "at least one pixel should pick up a star");
    }
}
