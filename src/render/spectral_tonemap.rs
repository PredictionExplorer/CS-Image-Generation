//! Hue-preserving `OKLab` tonemap and gamut mapper.
//!
//! The legacy `tonemap_core` compressed each linear-RGB channel through an
//! `AgX`-like curve independently and then clamped to `[0, 1]`. For an
//! extremely energetic trajectory core all three channels saturate to 1.0
//! and the pixel reads as **pure white**, even though the pre-tonemap
//! spectrum carried plenty of hue.
//!
//! This module provides:
//! - a **luminance-space** `AgX`-inspired tonemap that asymptotes to
//!   `paper_white` without clamping, and
//! - a **gamut map** that preserves `OKLab` hue and lightness while reducing
//!   chroma until the result fits inside the `sRGB` cube.
//!
//! The combined effect: bright hot cores retain their chroma (red flares
//! stay red, violet streamers stay violet) instead of collapsing to
//! featureless white.

use crate::oklab::{linear_srgb_to_oklab, oklab_to_linear_srgb};

/// Smooth sigmoid that asymptotes to 1.0 without clamping.
///
/// Applied in `OKLab` `L` after the `min_ev..=max_ev` log-allocation so the
/// full HDR range is compressed smoothly. The polynomial is the same `AgX`
/// base spline used per-channel in the legacy path, but its role here is to
/// shape **luminance only**.
#[inline]
#[must_use]
fn agx_spline(x: f64) -> f64 {
    // High quality fit for the AgX base curve (monotone on [0,1]).
    let x = x.clamp(0.0, 1.0);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;
    let x6 = x5 * x;
    12.0625 * x6 - 36.3262 * x5 + 39.5298 * x4 - 17.6534 * x3 + 3.0135 * x2 + 0.3707 * x
}

/// Log-space allocation of a scene-linear value onto `[0, 1]`.
#[inline]
#[must_use]
fn log_allocate(v: f64, min_ev: f64, range: f64) -> f64 {
    let val = v.max(1e-10).log2();
    ((val - min_ev) / range).clamp(0.0, 1.0)
}

/// Apply an `AgX`-style luminance compression in linear RGB.
///
/// The curve maps scene-linear luminance `L ∈ [0, +∞)` to display luminance
/// `L' ∈ [0, 1)`. Chroma (`OKLab` `a`, `b`) is scaled by `L'/L` so that a
/// **brighter** input does not become **more saturated** than its scene-lit
/// counterpart. Crucially no per-channel clamp is applied here — the output
/// may still lie slightly outside the `sRGB` cube, to be handled by
/// [`gamut_map_preserve_hue`].
#[must_use]
pub fn oklab_tonemap(rgb: [f64; 3]) -> [f64; 3] {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);

    let (l, a_ch, b_ch) = linear_srgb_to_oklab(r, g, b);

    // Allocate luminance across ~12.5 EV (same coverage as AgX base).
    const MIN_EV: f64 = -10.0;
    const MAX_EV: f64 = 2.5;
    const RANGE: f64 = MAX_EV - MIN_EV;

    let allocated = log_allocate(l.max(0.0), MIN_EV, RANGE);
    let l_out = agx_spline(allocated);

    // Preserve perceived saturation as luminance is compressed:
    // scale chroma by (L'/L)^0.65 so extreme highlights don't overshoot and
    // extreme shadows don't desaturate beyond plausibility.
    let chroma_scale = if l > 1e-6 { (l_out / l).max(0.0).powf(0.65) } else { 1.0 };

    let (r2, g2, b2) = oklab_to_linear_srgb(l_out, a_ch * chroma_scale, b_ch * chroma_scale);
    [r2, g2, b2]
}

/// Reduce `OKLab` chroma until the RGB point fits inside `[0, 1]^3`, keeping
/// `OKLab` `L` and hue angle unchanged.
///
/// This is a binary search on a single scalar `t ∈ [0, 1]` where `t = 1`
/// keeps the full chroma and `t = 0` collapses to neutral at the same
/// luminance. The returned RGB is always inside the `sRGB` cube.
#[must_use]
pub fn gamut_map_preserve_hue(rgb: [f64; 3]) -> [f64; 3] {
    if in_gamut(rgb) {
        return [rgb[0].clamp(0.0, 1.0), rgb[1].clamp(0.0, 1.0), rgb[2].clamp(0.0, 1.0)];
    }

    let (l, a_ch, b_ch) = linear_srgb_to_oklab(rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0));
    // Clamp L to a safe physical range so the search always converges.
    let l_clamped = l.clamp(0.0, 1.0);

    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        let candidate = oklab_to_linear_srgb(l_clamped, a_ch * mid, b_ch * mid);
        if in_gamut([candidate.0, candidate.1, candidate.2]) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let (r, g, b) = oklab_to_linear_srgb(l_clamped, a_ch * lo, b_ch * lo);
    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

#[inline]
fn in_gamut(rgb: [f64; 3]) -> bool {
    rgb[0] >= 0.0
        && rgb[0] <= 1.0
        && rgb[1] >= 0.0
        && rgb[1] <= 1.0
        && rgb[2] >= 0.0
        && rgb[2] <= 1.0
}

/// One-shot: tonemap scene-linear RGB, then gamut-map into `[0, 1]^3`.
///
/// This is the canonical entry point used by the tonemap core. Extreme
/// inputs no longer clip to white; they instead become the most saturated
/// in-gamut representation of their hue at the compressed luminance.
#[must_use]
#[inline]
pub fn oklab_tonemap_and_gamut_map(rgb: [f64; 3]) -> [f64; 3] {
    gamut_map_preserve_hue(oklab_tonemap(rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tonemap_preserves_black() {
        let out = oklab_tonemap_and_gamut_map([0.0, 0.0, 0.0]);
        for c in &out {
            assert!(c.abs() < 1e-3);
        }
    }

    #[test]
    fn tonemap_never_exceeds_gamut() {
        for (r, g, b) in [
            (0.0, 0.0, 0.0),
            (0.2, 0.3, 0.4),
            (1.0, 0.0, 0.0),
            (10.0, 0.2, 0.0),
            (100.0, 50.0, 20.0),
            (1e6, 1e6, 1e6),
            (50.0, 0.01, 100.0),
        ] {
            let out = oklab_tonemap_and_gamut_map([r, g, b]);
            assert!(out[0].is_finite() && out[1].is_finite() && out[2].is_finite());
            assert!((0.0..=1.0).contains(&out[0]), "r={r},{g},{b} -> {out:?}");
            assert!((0.0..=1.0).contains(&out[1]), "r={r},{g},{b} -> {out:?}");
            assert!((0.0..=1.0).contains(&out[2]), "r={r},{g},{b} -> {out:?}");
        }
    }

    #[test]
    fn bright_red_stays_red_not_white() {
        // A huge red input must not collapse to all-white.
        let out = oklab_tonemap_and_gamut_map([50.0, 0.1, 0.1]);
        assert!(
            out[0] > out[1] + 0.05 && out[0] > out[2] + 0.05,
            "red should dominate, got {out:?}"
        );
    }

    #[test]
    fn bright_blue_stays_blue_not_white() {
        let out = oklab_tonemap_and_gamut_map([0.1, 0.1, 50.0]);
        assert!(
            out[2] > out[0] + 0.05 && out[2] > out[1] + 0.05,
            "blue should dominate, got {out:?}"
        );
    }

    #[test]
    fn mid_grays_are_close_to_neutral() {
        let out = oklab_tonemap_and_gamut_map([0.5, 0.5, 0.5]);
        let max = out[0].max(out[1]).max(out[2]);
        let min = out[0].min(out[1]).min(out[2]);
        assert!(max - min < 0.01, "neutral should stay neutral, got {out:?}");
    }

    #[test]
    fn oklab_tonemap_is_monotone_for_gray() {
        let a = oklab_tonemap([0.2, 0.2, 0.2])[0];
        let b = oklab_tonemap([0.6, 0.6, 0.6])[0];
        let c = oklab_tonemap([2.0, 2.0, 2.0])[0];
        assert!(a <= b && b <= c);
    }
}
