//! Warm highlight glaze.
//!
//! A soft warm-tinted lift applied to highlights using soft-light compositing.
//! Mimics the luminous wet-ink / oil-glaze quality of illuminated manuscripts
//! and Renaissance glazing techniques — the core aesthetic move of the
//! Painterly mood.

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// Warm highlight glaze post-effect.
#[derive(Clone, Debug)]
pub struct Glaze {
    /// Blend strength (0 disables).
    pub strength: f64,
    /// Highlight threshold in luminance (below = no effect).
    pub threshold: f64,
    /// Warm tint `(r, g, b)` multiplier applied to the glazed layer.
    pub tint: (f64, f64, f64),
    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl Default for Glaze {
    fn default() -> Self {
        Self { strength: 0.35, threshold: 0.55, tint: (1.15, 1.02, 0.82), enabled: true }
    }
}

impl PostEffect for Glaze {
    fn is_enabled(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        _width: usize,
        _height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        let thr = self.threshold;
        let s = self.strength;
        let (tr, tg, tb) = self.tint;

        let out: PixelBuffer = input
            .par_iter()
            .map(|&(r, g, b, a)| {
                if a <= 1e-9 {
                    return (r, g, b, a);
                }
                let inv_a = 1.0 / a;
                let sr = (r * inv_a).clamp(0.0, 2.0);
                let sg = (g * inv_a).clamp(0.0, 2.0);
                let sb = (b * inv_a).clamp(0.0, 2.0);
                let lum = crate::render::constants::rec709_luminance(sr, sg, sb);
                let gate = ((lum - thr) / (1.0 - thr + 1e-6)).clamp(0.0, 1.0);
                if gate < 1e-3 {
                    return (r, g, b, a);
                }
                // Soft-light blend: classic Pegtop formula (fast, good enough).
                // result = base + gate * s * (tint * base * (1 - base) + tint - base)
                let gr = soft_light(sr, (sr * tr).min(1.0));
                let gg = soft_light(sg, (sg * tg).min(1.0));
                let gb_ = soft_light(sb, (sb * tb).min(1.0));
                let w = gate * s;
                let fr = sr + (gr - sr) * w;
                let fg = sg + (gg - sg) * w;
                let fb = sb + (gb_ - sb) * w;
                (fr * a, fg * a, fb * a, a)
            })
            .collect();
        Ok(out)
    }
}

#[inline]
fn soft_light(base: f64, blend: f64) -> f64 {
    // Pegtop's variant: smooth and monotonic.
    (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_pixels_untouched() {
        let glaze = Glaze::default();
        let input = vec![(0.1, 0.1, 0.1, 1.0); 4];
        let out = glaze.process(&input, 4, 1).expect("ok");
        for (before, after) in input.iter().zip(out.iter()) {
            assert!((before.0 - after.0).abs() < 1e-6);
            assert!((before.1 - after.1).abs() < 1e-6);
            assert!((before.2 - after.2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bright_pixels_get_warmer() {
        let glaze = Glaze::default();
        let input = vec![(0.85, 0.85, 0.85, 1.0)];
        let out = glaze.process(&input, 1, 1).expect("ok");
        // Red channel should end up >= blue channel after warm tint.
        assert!(out[0].0 >= out[0].2, "glaze should push highlights warm");
    }

    #[test]
    fn test_zero_strength_passthrough() {
        let glaze = Glaze { strength: 0.0, ..Default::default() };
        let input = vec![(0.85, 0.85, 0.85, 1.0)];
        let out = glaze.process(&input, 1, 1).expect("ok");
        assert_eq!(out, input);
    }
}
