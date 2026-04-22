//! Statistical variety tests for the randomization pipeline.
//!
//! These tests assert that the art-style meta-system produces *statistically
//! diverse* outputs across a large seed population. Concretely we measure:
//!
//! * Shannon entropy over `ArtStyle`, `GradePreset`, `HuePaletteMode`, and
//!   `DriftCharacter` distributions.
//! * Absence of single-variant dominance: no single enum value may claim more
//!   than ~30-40% of 256 seeds.
//!
//! The thresholds are chosen conservatively so that normal RNG variation does
//! not cause flakiness, but a regression that accidentally collapses the
//! distribution (e.g. resolving everything to `ArtStyle::DeepCosmos`) would
//! immediately fail.

use std::collections::HashMap;

use three_body_problem::render::art_style::{ArtStyle, DriftCharacter};
use three_body_problem::render::grade_presets::GradePreset;
use three_body_problem::render::hue_palette::HuePaletteMode;
use three_body_problem::render::randomizable_config::RandomizableEffectConfig;
use three_body_problem::sim::Sha3RandomByteStream;

const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;

const SAMPLE_COUNT: usize = 256;
const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;

fn shannon_entropy_bits<K: Eq + std::hash::Hash>(counts: &HashMap<K, usize>) -> f64 {
    let total: f64 = counts.values().copied().map(|c| c as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    counts
        .values()
        .copied()
        .map(|c| {
            let p = (c as f64) / total;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum()
}

fn max_share<K: Eq + std::hash::Hash>(counts: &HashMap<K, usize>) -> f64 {
    let total: f64 = counts.values().copied().map(|c| c as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    counts.values().copied().max().unwrap_or(0) as f64 / total
}

fn resolve_sample(
    seed: u64,
) -> three_body_problem::render::randomizable_config::ResolvedEffectConfig {
    let bytes = seed.to_le_bytes();
    let mut rng = Sha3RandomByteStream::new(&bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY);
    let cfg = RandomizableEffectConfig::default();
    let (resolved, _) = cfg.resolve(&mut rng, WIDTH, HEIGHT);
    resolved
}

#[test]
fn art_style_distribution_has_high_entropy() {
    let mut counts: HashMap<ArtStyle, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_5A5A_5A5A;
        let resolved = resolve_sample(seed);
        *counts.entry(resolved.art_style).or_default() += 1;
    }

    let entropy = shannon_entropy_bits(&counts);
    let max_entropy = (17f64).log2();
    // The pick() weights are intentionally flat (every style in 7..=8), so the
    // empirical entropy over 256 seeds is typically within ~0.05 bits of the
    // theoretical maximum log2(17) ~ 4.087. Requiring >= 3.55 bits gives
    // generous headroom for reshuffles or new seeds while still catching any
    // future regression that collapses the tail.
    assert!(
        entropy >= 3.55,
        "ArtStyle entropy {entropy:.3} bits too low (max {max_entropy:.3}); counts: {counts:?}"
    );
    let dominance = max_share(&counts);
    assert!(dominance <= 0.40, "ArtStyle dominance {dominance:.3} too high; counts: {counts:?}");
    assert!(
        counts.len() >= 12,
        "Only {} of 17 ArtStyle variants appeared in {SAMPLE_COUNT} seeds: {counts:?}",
        counts.len()
    );
}

#[test]
fn grade_preset_distribution_covers_most_variants() {
    let mut counts: HashMap<GradePreset, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0xC6BC_2796_3A0C_43B3) ^ 0xFEED_F00D_BAAD_F00D;
        let resolved = resolve_sample(seed);
        *counts.entry(resolved.grade_preset).or_default() += 1;
    }
    let entropy = shannon_entropy_bits(&counts);
    assert!(entropy >= 2.8, "GradePreset entropy {entropy:.3} bits too low; counts: {counts:?}");
    assert!(
        counts.len() >= 10,
        "Only {} of 16 GradePreset variants in {SAMPLE_COUNT} seeds: {counts:?}",
        counts.len()
    );
}

#[test]
fn hue_palette_mode_distribution_shows_spread() {
    let mut counts: HashMap<HuePaletteMode, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x0F0F_F0F0_CAFE_BABE;
        let resolved = resolve_sample(seed);
        *counts.entry(resolved.hue_palette_mode).or_default() += 1;
    }
    let entropy = shannon_entropy_bits(&counts);
    assert!(entropy >= 1.8, "HuePaletteMode entropy {entropy:.3} bits too low; counts: {counts:?}");
    assert!(
        counts.len() >= 4,
        "Only {} of 8 HuePaletteMode variants in {SAMPLE_COUNT} seeds: {counts:?}",
        counts.len()
    );
}

#[test]
fn drift_character_is_not_always_identical() {
    let mut counts: HashMap<DriftCharacter, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        let resolved = resolve_sample(seed);
        *counts.entry(resolved.drift_character).or_default() += 1;
    }
    assert!(counts.len() >= 3, "drift_character distribution collapsed: {counts:?}");
    let dominance = max_share(&counts);
    assert!(dominance <= 0.80, "drift_character dominance {dominance:.3} too high: {counts:?}");
}

#[test]
fn framing_zoom_has_both_tight_and_wide_samples() {
    let mut tight = 0usize;
    let mut wide = 0usize;
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0xD137_4DAC_C2D4_6EDB);
        let resolved = resolve_sample(seed);
        if (resolved.framing_zoom - 1.0).abs() < 1e-9 {
            tight += 1;
        } else if resolved.framing_zoom > 1.0 {
            wide += 1;
        }
    }
    assert!(tight > 20, "too few tight-framed samples: {tight}/{SAMPLE_COUNT}");
    assert!(wide > 20, "too few wide-framed samples: {wide}/{SAMPLE_COUNT}");
}
