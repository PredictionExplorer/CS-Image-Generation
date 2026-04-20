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
                (
                    (c - s) * std::f64::consts::FRAC_1_SQRT_2,
                    (s + c) * std::f64::consts::FRAC_1_SQRT_2,
                ),
                (
                    -(c - s) * std::f64::consts::FRAC_1_SQRT_2,
                    -(s + c) * std::f64::consts::FRAC_1_SQRT_2,
                ),
                (
                    (c + s) * std::f64::consts::FRAC_1_SQRT_2,
                    (s - c) * std::f64::consts::FRAC_1_SQRT_2,
                ),
                (
                    -(c + s) * std::f64::consts::FRAC_1_SQRT_2,
                    -(s - c) * std::f64::consts::FRAC_1_SQRT_2,
                ),
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
                    if lum >= thr { Some((x, y, p)) } else { None }
                })
            })
            .collect();

        let arm_len = self.arm_length as i64;
        // Adaptive strength attenuation for dense source populations.
        //
        // The legacy kernel paints spike arms around every pixel whose
        // luminance exceeds `threshold`. On a sparse starfield (~0.5% of
        // pixels above threshold) this produces a handful of crisp spikes.
        // On a dense orbit where 20-40% of pixels are above threshold, the
        // arms of each source overlap the arms of its neighbours — each
        // destination pixel receives dozens of additive stamps, saturating
        // every channel inside the inked region and destroying internal
        // structure. This is the primary contributor to the "white blob"
        // failure mode (seeds `0x3d8c8c21b240`, `0x4d5af082584d`,
        // `0x8df990f92766`, etc.).
        //
        // We attenuate strength by `sqrt(DENSITY_REF / source_density)`
        // when source density exceeds the reference. Square root is a
        // good empirical fit because each destination pixel's stamp count
        // scales roughly linearly with density, and linear strength
        // attenuation leaves cumulative brightness super-linear; sqrt is
        // the fixed-point. A hard floor keeps dense scenes visually
        // legible instead of collapsing to zero.
        let density = Self::compute_source_density(sources.len(), width, height);
        let density_factor = Self::density_attenuation(density);
        let strength = self.strength * density_factor;

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
                    // Hue-preserving additive write. The legacy per-channel
                    // `.min(2.0)` clamp shifted R:G:B ratios whenever the
                    // brightest channel saturated first, so a tinted core
                    // receiving a tinted arm drifted toward neutral. Rescale
                    // all three channels uniformly when the max exceeds
                    // `CEILING` so hue is preserved exactly.
                    let r = dst.0 + p.0 * w;
                    let g = dst.1 + p.1 * w;
                    let b = dst.2 + p.2 * w;
                    const CEILING: f64 = 2.0;
                    let max_ch = r.max(g).max(b);
                    let scale = if max_ch > CEILING { CEILING / max_ch } else { 1.0 };
                    dst.0 = r * scale;
                    dst.1 = g * scale;
                    dst.2 = b * scale;
                }
            }
        }

        Ok(output)
    }
}

impl DiffractionSpikes {
    /// Reference source density: 0.5% of pixels above threshold is the
    /// "typical" starfield case the effect was tuned for. Below this
    /// density the effect runs at full strength.
    pub(crate) const DENSITY_REF: f64 = 0.005;
    /// Minimum density-attenuation factor. Even on a fully-bright input
    /// the spike still contributes a visible (if very subtle) accent so
    /// the effect never silently disappears.
    pub(crate) const DENSITY_MIN_FACTOR: f64 = 0.1;

    /// Fraction of pixels that passed the luminance threshold. Used to
    /// scale back spike strength on dense bright regions so overlapping
    /// arms stop compounding into channel saturation.
    #[inline]
    fn compute_source_density(source_count: usize, width: usize, height: usize) -> f64 {
        let total = width.saturating_mul(height).max(1);
        #[allow(clippy::cast_precision_loss)]
        let density = source_count as f64 / total as f64;
        density
    }

    /// Density-aware strength multiplier in `[DENSITY_MIN_FACTOR, 1.0]`.
    /// At or below `DENSITY_REF` this is exactly 1.0 (no change); above
    /// it decays as `sqrt(DENSITY_REF / density)` with a hard floor.
    #[inline]
    #[must_use]
    pub(crate) fn density_attenuation(source_density: f64) -> f64 {
        if source_density <= Self::DENSITY_REF {
            return 1.0;
        }
        (Self::DENSITY_REF / source_density).sqrt().max(Self::DENSITY_MIN_FACTOR)
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

    /// At or below the reference density, density attenuation is a no-op —
    /// standard starfield renders are untouched by the new adaptive logic.
    #[test]
    fn density_attenuation_below_reference_is_noop() {
        assert_eq!(DiffractionSpikes::density_attenuation(0.0), 1.0);
        assert_eq!(
            DiffractionSpikes::density_attenuation(DiffractionSpikes::DENSITY_REF * 0.5),
            1.0
        );
        assert_eq!(DiffractionSpikes::density_attenuation(DiffractionSpikes::DENSITY_REF), 1.0);
    }

    /// Above the reference, attenuation scales as `sqrt(ref / density)`
    /// and never drops below the hard floor.
    #[test]
    fn density_attenuation_decays_sqrt_above_reference_with_floor() {
        // 4x density -> sqrt(1/4) = 0.5.
        let f4 = DiffractionSpikes::density_attenuation(DiffractionSpikes::DENSITY_REF * 4.0);
        assert!((f4 - 0.5).abs() < 1e-12);
        // 25x density -> sqrt(1/25) = 0.2.
        let f25 = DiffractionSpikes::density_attenuation(DiffractionSpikes::DENSITY_REF * 25.0);
        assert!((f25 - 0.2).abs() < 1e-12);
        // Past the floor: anything >= 100x ref hits MIN_FACTOR = 0.1.
        let f1000 = DiffractionSpikes::density_attenuation(DiffractionSpikes::DENSITY_REF * 1000.0);
        assert_eq!(f1000, DiffractionSpikes::DENSITY_MIN_FACTOR);
    }

    /// Regression for the "white blob" failure mode (seeds `0x3d8c8c21b240`,
    /// `0x4d5af082584d`, `0x8df990f92766`): on a dense inked region with
    /// a short arm length (so accumulation is bounded), the adaptive
    /// density attenuation must produce substantially lower per-pixel
    /// additions compared to the pre-fix kernel. We simulate the
    /// pre-fix behaviour by forcing the density attenuation to 1.0 via
    /// the [`DiffractionSpikes::density_attenuation`] helper and comparing
    /// to the actual (adaptive) output.
    #[test]
    fn dense_inked_input_is_attenuated_vs_unit_strength_reference() {
        let w = 64;
        let h = 64;
        let mut dense = vec![(0.0, 0.0, 0.0, 0.0); w * h];
        // Fill a 32x32 block (25% of the frame) with a mid-bright tint.
        for y in 16..48 {
            for x in 16..48 {
                dense[y * w + x] = (0.40, 0.80, 0.60, 1.0);
            }
        }
        // Short arms so accumulation doesn't hit the ceiling.
        let ds_adaptive = DiffractionSpikes {
            strength: 0.3,
            threshold: 0.5,
            arm_length: 4,
            rotation: 0.0,
            eight_point: true,
            enabled: true,
        };
        // Reference with the legacy (no-attenuation) strength: we
        // multiply `strength` by the density_factor the adaptive path
        // would have applied, so it is mathematically equivalent to
        // "kernel *without* density_attenuation running on dense input".
        let density = 1024.0 / 4096.0;
        let factor = DiffractionSpikes::density_attenuation(density);
        assert!(
            factor < 0.2,
            "density attenuation factor too high for 25% inked: {factor}"
        );
        let ds_legacy = DiffractionSpikes {
            strength: ds_adaptive.strength / factor,
            threshold: 0.5,
            arm_length: 4,
            rotation: 0.0,
            eight_point: true,
            enabled: true,
        };

        let out_adaptive = ds_adaptive.process(&dense, w, h).expect("ok");
        let out_legacy = ds_legacy.process(&dense, w, h).expect("ok");

        // Pick the centre pixel of the inked region and compare the
        // G-channel accumulation. The adaptive path must deposit
        // strictly less energy.
        let centre = 32 * w + 32;
        let g_adaptive = out_adaptive[centre].1;
        let g_legacy = out_legacy[centre].1;
        assert!(
            g_adaptive + 1e-9 < g_legacy,
            "adaptive path did not reduce deposit at centre: adaptive={g_adaptive}, legacy={g_legacy}"
        );
        // And the reduction must be substantial: at 25% density, the
        // factor is ~0.14, so adaptive should be dramatically smaller.
        // Account for ceiling effects in the legacy path by comparing
        // ratios rather than raw values.
        let raw_legacy_add = g_legacy - 0.80; // The base value.
        let raw_adaptive_add = g_adaptive - 0.80;
        assert!(
            raw_adaptive_add < raw_legacy_add * 0.5,
            "adaptive deposit not substantially lower: {raw_adaptive_add} vs {raw_legacy_add}"
        );
    }

    /// On a dense inked region, the hue-preserving rescale must preserve
    /// R:G:B ratios within the core. Legacy per-channel `.min(2.0)` would
    /// allow one channel to saturate while the others grow past the clamp
    /// point at different rates, drifting hue toward neutral.
    #[test]
    fn dense_inked_input_preserves_hue_ratio() {
        let w = 48;
        let h = 48;
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); w * h];
        // Strong red tint — R dominates luma — filling 25% of the frame.
        for y in 12..36 {
            for x in 12..36 {
                input[y * w + x] = (1.0, 0.25, 0.10, 1.0);
            }
        }
        let ds = DiffractionSpikes {
            strength: 0.6,
            threshold: 0.3,
            arm_length: 20,
            rotation: 0.0,
            eight_point: true,
            enabled: true,
        };
        let out = ds.process(&input, w, h).expect("ok");

        // Sample the centre of the inked region. Legacy kernel would push
        // this to R=G=B=2.0 (clamped) — a neutral white. With the fixes,
        // the red dominance must survive.
        let centre = out[24 * w + 24];
        assert!(
            centre.0 > centre.1 * 2.5,
            "hue drift in inked core: r={} g={} (expected r/g > 2.5)",
            centre.0,
            centre.1
        );
    }

    /// Isolated bright spike (sparse starfield case) is untouched by the
    /// density attenuation — we must not weaken the effect for its intended
    /// use case.
    #[test]
    fn isolated_bright_spike_is_untouched_by_density_attenuation() {
        let w = 64;
        let h = 64;
        let mut input = vec![(0.0, 0.0, 0.0, 1.0); w * h];
        input[32 * w + 32] = (1.0, 1.0, 1.0, 1.0);
        // Compare legacy-equivalent output (compute strength manually at
        // full factor) with effect output.
        let ds = DiffractionSpikes {
            strength: 0.5,
            threshold: 0.5,
            arm_length: 8,
            rotation: 0.0,
            eight_point: false,
            enabled: true,
        };
        let out = ds.process(&input, w, h).expect("ok");
        // Arm at distance 2 from the centre:
        //   falloff = 1 - 2/8 = 0.75 → falloff² = 0.5625.
        //   gate = (lum - thr) / (1 - thr + 1e-6) with a tiny 1e-6 to
        //         avoid division by zero at thr = 1; here lum = 1,
        //         thr = 0.5, so gate = 0.5 / 0.500001 ≈ 0.999998.
        //   density_factor = 1.0 (1 source in 4096 pixels << DENSITY_REF).
        //   w = strength * gate * falloff² = 0.5 * 0.999998 * 0.5625
        //     ≈ 0.28124943750...
        // The sparse case must be attenuated by *exactly* no density
        // factor — any regression of the adaptive logic over-firing on
        // isolated sources would break this test.
        let arm = out[32 * w + 34];
        let expected = 0.5 * (0.5 / (1.0 - 0.5 + 1e-6)) * 0.75_f64.powi(2);
        assert!(
            (arm.0 - expected).abs() < 1e-9,
            "sparse case attenuated unexpectedly (arm.r={}, expected {expected})",
            arm.0
        );
    }

    /// Long horizontal run of bright pixels (~4% of frame) should receive
    /// a moderate attenuation but not be crushed to the floor.
    #[test]
    fn moderate_density_gets_moderate_attenuation() {
        let w = 200;
        let h = 50;
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); w * h];
        // ~4% of pixels above threshold (400 / 10000).
        for x in 0..400 {
            let px = x % w;
            let py = (x / w) % h;
            input[py * w + px] = (0.9, 0.9, 0.9, 1.0);
        }
        // Density = 0.04 = 8 * DENSITY_REF, so attenuation = sqrt(1/8) ~ 0.354.
        let factor = DiffractionSpikes::density_attenuation(0.04);
        assert!(
            (factor - (1.0f64 / 8.0_f64).sqrt()).abs() < 1e-9,
            "density attenuation for 0.04 = {factor}"
        );
        assert!(factor > DiffractionSpikes::DENSITY_MIN_FACTOR);
        // Simply smoke-running the effect must succeed.
        let ds = DiffractionSpikes::default();
        let out = ds.process(&input, w, h).expect("ok");
        assert_eq!(out.len(), w * h);
    }
}
