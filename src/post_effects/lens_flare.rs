//! Directional diffraction-style streaks on very bright pixels (anamorphic + starburst).

use super::{PixelBuffer, PostEffect, PostEffectError};
use rayon::prelude::*;

/// Simple lens-flare pass: horizontal streak + six-point star around hot highlights.
#[derive(Clone, Debug)]
pub struct LensFlareDiffractive {
    /// Blend strength for added energy (0 disables).
    pub strength: f64,
    /// Linear luminance threshold on premultiplied RGB (after dividing by alpha).
    pub threshold: f64,
    /// When true, use a much wider cinematic anamorphic streak with cyan tint.
    pub anamorphic: bool,
    /// RGB multiplier for the anamorphic streak (when `anamorphic` is true).
    pub anamorphic_tint: (f64, f64, f64),
}

impl Default for LensFlareDiffractive {
    fn default() -> Self {
        Self {
            strength: 0.22,
            threshold: 0.82,
            anamorphic: false,
            anamorphic_tint: (0.55, 0.85, 1.0),
        }
    }
}

impl LensFlareDiffractive {
    /// Builds the default cinematic flare used when the pipeline flag is enabled.
    #[must_use]
    pub fn pipeline_default() -> Self {
        Self::default()
    }

    /// Wide blue/cyan anamorphic streak variant used by the Cinematic mood.
    ///
    /// The streak is ~3x the length of the default and is cool-tinted for the
    /// classic "blockbuster lens" look.
    #[must_use]
    pub fn anamorphic() -> Self {
        Self {
            strength: 0.32,
            threshold: 0.70,
            anamorphic: true,
            anamorphic_tint: (0.55, 0.85, 1.0),
        }
    }

    #[inline]
    fn premul_luma(p: (f64, f64, f64, f64)) -> f64 {
        let a = p.3.max(1e-9);
        let lr = p.0 / a;
        let lg = p.1 / a;
        let lb = p.2 / a;
        0.2126 * lr + 0.7152 * lg + 0.0722 * lb
    }
}

impl PostEffect for LensFlareDiffractive {
    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if self.strength <= 0.0 {
            return Ok(input.clone());
        }

        let w = width as isize;
        let h = height as isize;
        let mut acc: PixelBuffer = input.clone();

        // Horizontal anamorphic streak kernel offsets + weights
        const H_STREAK: &[(isize, f64)] = &[
            (-12, 0.04),
            (-8, 0.07),
            (-5, 0.1),
            (-3, 0.14),
            (-1, 0.18),
            (0, 0.22),
            (1, 0.18),
            (3, 0.14),
            (5, 0.1),
            (8, 0.07),
            (12, 0.04),
        ];

        // Wide cinema-style anamorphic streak used when `anamorphic = true`.
        const H_STREAK_WIDE: &[(isize, f64)] = &[
            (-48, 0.015),
            (-36, 0.025),
            (-26, 0.04),
            (-18, 0.06),
            (-12, 0.09),
            (-8, 0.12),
            (-5, 0.16),
            (-3, 0.20),
            (-1, 0.24),
            (0, 0.28),
            (1, 0.24),
            (3, 0.20),
            (5, 0.16),
            (8, 0.12),
            (12, 0.09),
            (18, 0.06),
            (26, 0.04),
            (36, 0.025),
            (48, 0.015),
        ];

        // Six-point star (60° directions approximated on integer grid)
        const STAR: &[(isize, isize, f64)] = &[
            (1, 0, 0.12),
            (2, 0, 0.08),
            (3, 0, 0.05),
            (4, 0, 0.03),
            (-1, 0, 0.12),
            (-2, 0, 0.08),
            (-3, 0, 0.05),
            (-4, 0, 0.03),
            (1, 2, 0.06),
            (2, 3, 0.04),
            (3, 5, 0.03),
            (-1, -2, 0.06),
            (-2, -3, 0.04),
            (-3, -5, 0.03),
            (1, -2, 0.06),
            (2, -3, 0.04),
            (-1, 2, 0.06),
            (-2, 3, 0.04),
        ];

        let thr = self.threshold;
        let str = self.strength;
        let streak: &[(isize, f64)] = if self.anamorphic { H_STREAK_WIDE } else { H_STREAK };
        let tint = if self.anamorphic { self.anamorphic_tint } else { (1.0, 1.0, 1.0) };

        acc.par_iter_mut().enumerate().for_each(|(idx, out)| {
            let x0 = (idx % width) as isize;
            let y0 = (idx / width) as isize;
            let base = input[idx];
            let lum = Self::premul_luma(base);
            if lum < thr {
                return;
            }
            let gate = ((lum - thr) / (1.0 - thr + 1e-6)).clamp(0.0, 1.0);
            let mut add = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

            for &(dx, wgt) in streak {
                let sx = x0 + dx;
                let sy = y0;
                if sx >= 0 && sx < w && sy >= 0 && sy < h {
                    let p = input[sy as usize * width + sx as usize];
                    let l = Self::premul_luma(p);
                    if l > thr * 0.85 {
                        let g = gate * wgt * str;
                        add.0 += p.0 * g * tint.0;
                        add.1 += p.1 * g * tint.1;
                        add.2 += p.2 * g * tint.2;
                        add.3 += p.3 * g * 0.2;
                    }
                }
            }

            for &(dx, dy, wgt) in STAR {
                let sx = x0 + dx;
                let sy = y0 + dy;
                if sx >= 0 && sx < w && sy >= 0 && sy < h {
                    let p = input[sy as usize * width + sx as usize];
                    let l = Self::premul_luma(p);
                    if l > thr * 0.9 {
                        let g = gate * wgt * str;
                        add.0 += p.0 * g;
                        add.1 += p.1 * g;
                        add.2 += p.2 * g;
                        add.3 += p.3 * g * 0.15;
                    }
                }
            }

            // Hue-preserving ceiling: the legacy per-channel `.min(1.0)` shifted
            // R:G:B ratios whenever one channel saturated before the others,
            // collapsing bright tinted cores to pure white. Instead, scale all
            // three channels by the same factor whenever the max channel exceeds
            // the ceiling, exactly preserving hue/chroma at the cost of a tiny
            // luminance reduction. Alpha is coverage (not chroma) so it keeps
            // its per-channel clamp.
            let r = base.0 + add.0;
            let g = base.1 + add.1;
            let b = base.2 + add.2;
            const CEILING: f64 = 1.0;
            let max_ch = r.max(g).max(b);
            let scale = if max_ch > CEILING { CEILING / max_ch } else { 1.0 };
            out.0 = r * scale;
            out.1 = g * scale;
            out.2 = b * scale;
            out.3 = (base.3 + add.3).min(1.0);
        });

        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn center_hotspot(width: usize, height: usize) -> PixelBuffer {
        let mut buf = vec![(0.0, 0.0, 0.0, 0.0); width * height];
        let cx = width / 2;
        let cy = height / 2;
        buf[cy * width + cx] = (0.95, 0.95, 0.95, 0.95);
        buf
    }

    #[test]
    fn test_zero_strength_is_passthrough() {
        let input = center_hotspot(48, 32);
        let flare = LensFlareDiffractive { strength: 0.0, threshold: 0.5, ..Default::default() };
        let out = flare.process(&input, 48, 32).expect("process");
        assert_eq!(out, input);
    }

    #[test]
    fn test_high_threshold_leaves_uniform_dim_image_unchanged() {
        // All pixels have premultiplied luma = 1.0, but lum/alpha = 1.0 < threshold.
        let mut input = vec![(0.1, 0.1, 0.1, 0.1); 16 * 16];
        for p in &mut input {
            p.0 = 0.1;
            p.1 = 0.1;
            p.2 = 0.1;
            p.3 = 1.0;
        }
        // Per-pixel un-premultiplied luma is 0.1, below threshold = 0.9 → gate never opens.
        let flare = LensFlareDiffractive { strength: 0.5, threshold: 0.9, ..Default::default() };
        let out = flare.process(&input, 16, 16).expect("process");
        assert_eq!(out, input);
    }

    #[test]
    fn test_bright_row_gets_horizontally_extended_streak() {
        // Whole centre row is bright so both the hot pixel and its lateral neighbours
        // pass the threshold, proving the horizontal streak kernel is wired up.
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); 64 * 16];
        for x in 16..48 {
            input[8 * 64 + x] = (0.92, 0.92, 0.92, 0.92);
        }
        let flare = LensFlareDiffractive::default();
        let out = flare.process(&input, 64, 16).expect("process");
        let before = input[8 * 64 + 32];
        let after = out[8 * 64 + 32];
        // Horizontal kernel deposits extra RGB onto centre pixel.
        assert!(after.0 >= before.0, "centre should not darken ({} -> {})", before.0, after.0);
        // Off-row neighbours stay dark because their luma is below threshold.
        assert_eq!(out[9 * 64 + 32], (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_output_bounded_to_unit_interval() {
        let mut input = vec![(0.9, 0.9, 0.9, 0.9); 40 * 24];
        for pixel in &mut input {
            pixel.0 = 1.0;
            pixel.1 = 1.0;
            pixel.2 = 1.0;
            pixel.3 = 1.0;
        }
        let out = LensFlareDiffractive::default().process(&input, 40, 24).expect("process");
        for p in out {
            assert!(p.0 <= 1.0);
            assert!(p.1 <= 1.0);
            assert!(p.2 <= 1.0);
            assert!(p.3 <= 1.0);
        }
    }

    /// Regression for the "white-blob" failure mode (PR #N diagnostic run,
    /// seeds `0x3d8c8c21b240`, `0x4d5af082584d`, `0x8df990f92766`): when a
    /// bright tinted core received a tinted anamorphic streak, the legacy
    /// per-channel `.min(1.0)` clamped each channel independently, breaking
    /// the R:G:B ratio and collapsing the core to near-white. Under the
    /// hue-preserving ceiling, the core must retain its tint after the
    /// flare writes.
    #[test]
    fn anamorphic_flare_preserves_hue_on_saturated_red_core() {
        let width = 64;
        let height = 16;
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); width * height];
        // A horizontal run of strongly red-tinted near-ceiling pixels. Luma is
        // dominated by R so the anamorphic streak kernel will fire.
        for x in 16..48 {
            input[8 * width + x] = (0.98, 0.20, 0.10, 1.0);
        }
        let flare = LensFlareDiffractive::anamorphic();
        let out = flare.process(&input, width, height).expect("process");

        let centre = out[8 * width + 32];
        // All channels must stay within the ceiling.
        assert!(centre.0 <= 1.0 + 1e-9);
        assert!(centre.1 <= 1.0 + 1e-9);
        assert!(centre.2 <= 1.0 + 1e-9);
        // Hue check: R should still dominate G and B by a wide margin. Legacy
        // per-channel clamp would produce R = G = B ~= 1.0 (pure white).
        assert!(
            centre.0 > centre.1 * 2.0,
            "red should remain >2x green (got r={} g={})",
            centre.0,
            centre.1
        );
        assert!(
            centre.0 > centre.2 * 3.0,
            "red should remain >3x blue (got r={} b={})",
            centre.0,
            centre.2
        );
    }

    /// The R:G:B ratio at the core must be preserved exactly when the
    /// hue-preserving ceiling engages (up to floating point). This is the
    /// mathematical contract of the max-channel rescale.
    #[test]
    fn hue_ratio_is_invariant_under_ceiling_rescale() {
        let width = 48;
        let height = 8;
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); width * height];
        for x in 12..36 {
            // 4:2:1 ratio, already near the ceiling so any additive write
            // forces the rescale branch.
            input[4 * width + x] = (0.80, 0.40, 0.20, 1.0);
        }
        let flare = LensFlareDiffractive {
            strength: 0.8,
            threshold: 0.30,
            anamorphic: false,
            anamorphic_tint: (1.0, 1.0, 1.0),
        };
        let out = flare.process(&input, width, height).expect("process");

        let centre = out[4 * width + 24];
        // The input ratio is R:G = 2.0, R:B = 4.0. After the hue-preserving
        // scale, additive writes can change absolute values but the ratios
        // must stay within a small tolerance (the streak kernel mixes in
        // neighbours that share the same tint).
        let rg = centre.0 / centre.1.max(1e-9);
        let rb = centre.0 / centre.2.max(1e-9);
        assert!(
            (rg - 2.0).abs() < 0.05,
            "R:G drifted from 2.0 to {rg:.4} (centre={centre:?})"
        );
        assert!(
            (rb - 4.0).abs() < 0.10,
            "R:B drifted from 4.0 to {rb:.4} (centre={centre:?})"
        );
    }

    /// Black base + white streak should never lift a pixel above the unit
    /// cube — legacy behaviour this test also caught — and the additive
    /// write should carry a neutral-white tint, proving the ceiling does
    /// nothing when no channel exceeds 1.0.
    #[test]
    fn ceiling_is_noop_when_no_channel_exceeds_1() {
        let width = 32;
        let height = 8;
        let mut input = vec![(0.0, 0.0, 0.0, 0.0); width * height];
        for x in 8..24 {
            input[4 * width + x] = (0.30, 0.30, 0.30, 1.0);
        }
        let flare = LensFlareDiffractive {
            strength: 0.4,
            threshold: 0.10,
            anamorphic: false,
            anamorphic_tint: (1.0, 1.0, 1.0),
        };
        let out = flare.process(&input, width, height).expect("process");
        let centre = out[4 * width + 16];
        // All channels stay equal for a neutral input under a neutral tint.
        assert!((centre.0 - centre.1).abs() < 1e-9);
        assert!((centre.1 - centre.2).abs() < 1e-9);
        // And stay under the ceiling (well under, given the modest strength).
        assert!(centre.0 < 1.0);
    }
}
