//! Hue-palette distribution modes for per-body color sequences.
//!
//! The previous pipeline always placed the three bodies 120° apart in `OKLab`
//! hue, producing a near-identical triadic look across every generation.
//! [`HuePaletteMode`] introduces eight well-known harmonic relationships
//! that are chosen per seed for dramatic palette variety.
//!
//! [`hues_for_mode`] returns three hues in degrees, guaranteed to be in
//! `[0.0, 360.0)`.

use crate::sim::Sha3RandomByteStream;
use serde::{Deserialize, Serialize};

/// One of a fixed set of harmonic hue-distribution strategies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HuePaletteMode {
    /// Classic 120° triad (legacy default).
    #[default]
    Triadic,
    /// Two complementary hues + one mid-tone.
    Complementary,
    /// Complementary split: primary + two opposite neighbors.
    SplitComplementary,
    /// Three analogous (closely spaced) hues.
    Analogous,
    /// Three very close hues in a single family.
    Monochromatic,
    /// Only two hues repeated - high-impact duotone.
    Duotone,
    /// Four hues spaced 90°, three picked (square tetrad).
    TetradicSquare,
    /// Warm / cool cosmic pair with an intermediate accent.
    CosmicWarmCool,
}

/// Static list of every hue palette mode variant.
pub const ALL_HUE_PALETTE_MODES: &[HuePaletteMode] = &[
    HuePaletteMode::Triadic,
    HuePaletteMode::Complementary,
    HuePaletteMode::SplitComplementary,
    HuePaletteMode::Analogous,
    HuePaletteMode::Monochromatic,
    HuePaletteMode::Duotone,
    HuePaletteMode::TetradicSquare,
    HuePaletteMode::CosmicWarmCool,
];

impl HuePaletteMode {
    /// Every variant in declaration order.
    #[must_use]
    pub fn all() -> &'static [HuePaletteMode] {
        ALL_HUE_PALETTE_MODES
    }

    /// Select a mode by index using modulo arithmetic.
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        ALL_HUE_PALETTE_MODES[index % ALL_HUE_PALETTE_MODES.len()]
    }

    /// Machine-readable name for logs.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            HuePaletteMode::Triadic => "triadic",
            HuePaletteMode::Complementary => "complementary",
            HuePaletteMode::SplitComplementary => "split_complementary",
            HuePaletteMode::Analogous => "analogous",
            HuePaletteMode::Monochromatic => "monochromatic",
            HuePaletteMode::Duotone => "duotone",
            HuePaletteMode::TetradicSquare => "tetradic_square",
            HuePaletteMode::CosmicWarmCool => "cosmic_warm_cool",
        }
    }
}

fn draw_unit(rng: &mut Sha3RandomByteStream) -> f64 {
    // Use 32-bit fixed-point from the existing byte stream for consistency
    // with the rest of the codebase.
    let b0 = u32::from(rng.next_byte());
    let b1 = u32::from(rng.next_byte());
    let b2 = u32::from(rng.next_byte());
    let b3 = u32::from(rng.next_byte());
    let bits = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    f64::from(bits) / f64::from(u32::MAX)
}

fn wrap_hue(h: f64) -> f64 {
    let mut v = h % 360.0;
    if v < 0.0 {
        v += 360.0;
    }
    // Numerical-safety: guarantee strict [0, 360).
    if v >= 360.0 {
        v -= 360.0;
    }
    v
}

/// Compute three body hues (degrees) given a mode + deterministic RNG.
///
/// All returned values are guaranteed to be finite and strictly in
/// `[0.0, 360.0)`. The ordering of returned hues is meaningful: index 0 is
/// the lead / subject body.
#[must_use]
pub fn hues_for_mode(mode: HuePaletteMode, rng: &mut Sha3RandomByteStream) -> [f64; 3] {
    let base = draw_unit(rng) * 360.0;
    match mode {
        HuePaletteMode::Triadic => [wrap_hue(base), wrap_hue(base + 120.0), wrap_hue(base + 240.0)],
        HuePaletteMode::Complementary => {
            // Two opposite poles + a neutral midpoint.
            let jitter = (draw_unit(rng) - 0.5) * 12.0;
            [wrap_hue(base), wrap_hue(base + 180.0 + jitter), wrap_hue(base + 90.0 + jitter * 0.5)]
        }
        HuePaletteMode::SplitComplementary => {
            // Primary + two hues flanking the complement.
            let split = 24.0 + draw_unit(rng) * 24.0;
            [wrap_hue(base), wrap_hue(base + 180.0 - split), wrap_hue(base + 180.0 + split)]
        }
        HuePaletteMode::Analogous => {
            // Three close hues, 18-36° apart.
            let step = 18.0 + draw_unit(rng) * 18.0;
            [wrap_hue(base), wrap_hue(base + step), wrap_hue(base + 2.0 * step)]
        }
        HuePaletteMode::Monochromatic => {
            // Very tight cluster - difference shows up via chroma/lightness.
            let jitter = draw_unit(rng) * 8.0;
            [wrap_hue(base - jitter), wrap_hue(base), wrap_hue(base + jitter)]
        }
        HuePaletteMode::Duotone => {
            // Two hues repeated — body 0 and body 2 share a hue.
            let gap = 140.0 + draw_unit(rng) * 40.0;
            let secondary = wrap_hue(base + gap);
            [wrap_hue(base), secondary, wrap_hue(base)]
        }
        HuePaletteMode::TetradicSquare => {
            // Drop one of the four square points.
            let drop = (draw_unit(rng) * 4.0) as usize % 4;
            let mut points = [
                wrap_hue(base),
                wrap_hue(base + 90.0),
                wrap_hue(base + 180.0),
                wrap_hue(base + 270.0),
            ];
            points.rotate_left(drop);
            [points[0], points[1], points[2]]
        }
        HuePaletteMode::CosmicWarmCool => {
            // Two cosmic poles + a neutral bridge: warm (10-70°), cool (170-250°), bridge.
            let warm = 10.0 + draw_unit(rng) * 60.0;
            let cool = 170.0 + draw_unit(rng) * 80.0;
            let bridge = 280.0 + draw_unit(rng) * 60.0;
            [wrap_hue(warm), wrap_hue(cool), wrap_hue(bridge)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed_byte: u8) -> Sha3RandomByteStream {
        let seed = vec![seed_byte, seed_byte.wrapping_add(1), seed_byte.wrapping_add(2)];
        Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0)
    }

    #[test]
    fn hues_always_in_range_for_every_mode() {
        for &mode in HuePaletteMode::all() {
            for seed in 0..16u8 {
                let mut rng = make_rng(seed);
                let hues = hues_for_mode(mode, &mut rng);
                for (i, &h) in hues.iter().enumerate() {
                    assert!(h.is_finite(), "mode {} seed {seed} hue {i} = {h}", mode.name());
                    assert!(
                        (0.0..360.0).contains(&h),
                        "mode {} seed {seed} hue {i} = {h}",
                        mode.name()
                    );
                }
            }
        }
    }

    #[test]
    fn triadic_has_120_degree_separation() {
        let mut rng = make_rng(7);
        let hues = hues_for_mode(HuePaletteMode::Triadic, &mut rng);
        let d01 = wrap_distance(hues[0], hues[1]);
        let d12 = wrap_distance(hues[1], hues[2]);
        assert!((d01 - 120.0).abs() < 0.5);
        assert!((d12 - 120.0).abs() < 0.5);
    }

    #[test]
    fn complementary_puts_bodies_near_opposite_ends() {
        for seed in 0..8u8 {
            let mut rng = make_rng(seed);
            let hues = hues_for_mode(HuePaletteMode::Complementary, &mut rng);
            let d = wrap_distance(hues[0], hues[1]);
            assert!((d - 180.0).abs() < 10.0, "seed {seed} got d01 = {d} (hues = {hues:?})",);
        }
    }

    #[test]
    fn monochromatic_stays_tight() {
        for seed in 0..8u8 {
            let mut rng = make_rng(seed);
            let hues = hues_for_mode(HuePaletteMode::Monochromatic, &mut rng);
            let spread = wrap_distance(hues[0], hues[2]);
            assert!(spread <= 20.0, "seed {seed} spread {spread}");
        }
    }

    #[test]
    fn duotone_has_only_two_unique_hues() {
        for seed in 0..8u8 {
            let mut rng = make_rng(seed);
            let hues = hues_for_mode(HuePaletteMode::Duotone, &mut rng);
            let d02 = wrap_distance(hues[0], hues[2]);
            assert!(
                d02 < 0.001,
                "duotone must repeat hue 0 on body 2, got {} vs {}",
                hues[0],
                hues[2]
            );
            let d01 = wrap_distance(hues[0], hues[1]);
            assert!(d01 > 130.0 && d01 < 185.0);
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        for &mode in HuePaletteMode::all() {
            let mut r1 = make_rng(42);
            let mut r2 = make_rng(42);
            let h1 = hues_for_mode(mode, &mut r1);
            let h2 = hues_for_mode(mode, &mut r2);
            assert_eq!(h1, h2, "mode {} not deterministic", mode.name());
        }
    }

    #[test]
    fn all_modes_have_distinct_names() {
        use std::collections::HashSet;
        let names: HashSet<&'static str> = HuePaletteMode::all().iter().map(|m| m.name()).collect();
        assert_eq!(names.len(), HuePaletteMode::all().len());
    }

    #[test]
    fn at_least_eight_modes() {
        assert!(HuePaletteMode::all().len() >= 8);
    }

    fn wrap_distance(a: f64, b: f64) -> f64 {
        let raw = (a - b).abs() % 360.0;
        raw.min(360.0 - raw)
    }
}
