//! `OKLab` palette harmonizer.
//!
//! Pulls each pixel's hue toward the nearest "anchor" hue from a seeded
//! triadic or analogous palette. Chroma magnitude and lightness are preserved
//! so the image reads with the same luminance structure, just with a more
//! coherent colour identity — a key ingredient for the Painterly mood.

use super::{PixelBuffer, PostEffect, PostEffectError};
use crate::oklab::{linear_srgb_to_oklab, oklab_to_linear_srgb};
use rayon::prelude::*;

/// Palette scheme.
#[derive(Clone, Copy, Debug)]
pub enum PaletteScheme {
    /// Three hues evenly spaced 120° apart.
    Triadic,
    /// Three hues within 60° of the seed (gentle harmony).
    Analogous,
    /// Two hues 180° apart.
    Complementary,
}

/// Palette harmony post-effect.
#[derive(Clone, Debug)]
pub struct PaletteHarmony {
    /// Anchor hues (radians, `atan2(b, a)` convention).
    anchors: Vec<f64>,
    /// Blend strength `[0, 1]`.
    pub strength: f64,
    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl PaletteHarmony {
    /// Construct a palette harmony pass from a deterministic seed.
    #[must_use]
    pub fn from_seed(seed: i32, strength: f64, scheme: PaletteScheme) -> Self {
        let mut state = (seed as u64).wrapping_mul(0xBF58476D1CE4E5B9).wrapping_add(0xA11CE);
        let raw = next_u64(&mut state);
        let base = ((raw >> 11) as f64 / (1u64 << 53) as f64) * std::f64::consts::TAU;
        let anchors = match scheme {
            PaletteScheme::Triadic => vec![
                base,
                base + std::f64::consts::TAU / 3.0,
                base + 2.0 * std::f64::consts::TAU / 3.0,
            ],
            PaletteScheme::Analogous => {
                vec![base - 0.35, base, base + 0.35]
            }
            PaletteScheme::Complementary => vec![base, base + std::f64::consts::PI],
        };
        Self { anchors, strength: strength.clamp(0.0, 1.0), enabled: true }
    }

    /// Number of anchor hues (for tests).
    #[must_use]
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Closest anchor to a given hue (handles the ±π wrap).
    #[inline]
    fn closest_anchor(&self, hue: f64) -> f64 {
        let mut best = self.anchors[0];
        let mut best_dist = angular_distance(hue, best);
        for &a in self.anchors.iter().skip(1) {
            let d = angular_distance(hue, a);
            if d < best_dist {
                best_dist = d;
                best = a;
            }
        }
        best
    }
}

#[inline]
fn angular_distance(a: f64, b: f64) -> f64 {
    let mut d = (a - b).rem_euclid(std::f64::consts::TAU);
    if d > std::f64::consts::PI {
        d = std::f64::consts::TAU - d;
    }
    d
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

impl PostEffect for PaletteHarmony {
    fn is_enabled(&self) -> bool {
        self.enabled && self.strength > 0.0 && !self.anchors.is_empty()
    }

    fn process(
        &self,
        input: &PixelBuffer,
        _width: usize,
        _height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        let s = self.strength;
        if s <= 0.0 {
            return Ok(input.clone());
        }
        let out: PixelBuffer = input
            .par_iter()
            .map(|&(r, g, b, a)| {
                if a <= 1e-9 {
                    return (r, g, b, a);
                }
                // Un-premultiply, convert to OKLab, snap hue, convert back.
                let inv_a = 1.0 / a;
                let sr = (r * inv_a).max(0.0);
                let sg = (g * inv_a).max(0.0);
                let sb = (b * inv_a).max(0.0);
                let (l, oa, ob) = linear_srgb_to_oklab(sr, sg, sb);
                let chroma = (oa * oa + ob * ob).sqrt();
                if chroma < 0.015 {
                    return (r, g, b, a);
                }
                let hue = ob.atan2(oa);
                let target = self.closest_anchor(hue);
                // Shortest-arc interpolation.
                let mut delta = target - hue;
                if delta > std::f64::consts::PI {
                    delta -= std::f64::consts::TAU;
                } else if delta < -std::f64::consts::PI {
                    delta += std::f64::consts::TAU;
                }
                let new_hue = hue + delta * s;
                let new_a = chroma * new_hue.cos();
                let new_b = chroma * new_hue.sin();
                let (nr, ng, nb) = oklab_to_linear_srgb(l, new_a, new_b);
                (nr.max(0.0) * a, ng.max(0.0) * a, nb.max(0.0) * a, a)
            })
            .collect();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triadic_has_three_anchors() {
        let p = PaletteHarmony::from_seed(12345, 0.5, PaletteScheme::Triadic);
        assert_eq!(p.anchor_count(), 3);
    }

    #[test]
    fn test_zero_strength_passthrough() {
        let p = PaletteHarmony::from_seed(1, 0.0, PaletteScheme::Triadic);
        let input = vec![(0.5, 0.1, 0.1, 1.0), (0.1, 0.5, 0.1, 1.0)];
        let out = p.process(&input, 2, 1).expect("ok");
        assert_eq!(out, input);
    }

    #[test]
    fn test_palette_changes_vivid_hues_not_greys() {
        let p = PaletteHarmony::from_seed(7, 1.0, PaletteScheme::Triadic);
        let vivid = vec![(0.6, 0.05, 0.05, 1.0)];
        let out_vivid = p.process(&vivid, 1, 1).expect("ok");
        assert!(
            (out_vivid[0].0 - vivid[0].0).abs() > 0.005
                || (out_vivid[0].1 - vivid[0].1).abs() > 0.005
                || (out_vivid[0].2 - vivid[0].2).abs() > 0.005,
            "vivid pixel should shift toward an anchor"
        );

        let grey = vec![(0.35, 0.35, 0.35, 1.0)];
        let out_grey = p.process(&grey, 1, 1).expect("ok");
        assert!(
            (out_grey[0].0 - grey[0].0).abs() < 0.01
                && (out_grey[0].1 - grey[0].1).abs() < 0.01
                && (out_grey[0].2 - grey[0].2).abs() < 0.01,
            "grey should be preserved"
        );
    }
}
