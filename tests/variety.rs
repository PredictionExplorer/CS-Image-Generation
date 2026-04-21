//! Statistical variety checks for the compact curation layer.

use std::collections::HashMap;

use three_body_problem::render::curation::{CompositionMode, FinishCluster, GradeFamily};
use three_body_problem::render::grade_presets::GradePreset;
use three_body_problem::render::randomizable_config::RandomizableEffectConfig;
use three_body_problem::sim::Sha3RandomByteStream;

const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;
const SAMPLE_COUNT: usize = 256;

fn resolve_sample(seed: u64) -> three_body_problem::render::randomizable_config::ResolvedEffectConfig {
    let bytes = seed.to_le_bytes();
    let mut rng = Sha3RandomByteStream::new(&bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY);
    let (resolved, _) = RandomizableEffectConfig::default().resolve(&mut rng, 640, 360);
    resolved
}

fn shannon_entropy_bits<K: Eq + std::hash::Hash>(counts: &HashMap<K, usize>) -> f64 {
    let total: f64 = counts.values().copied().map(|count| count as f64).sum();
    counts
        .values()
        .copied()
        .map(|count| {
            let p = count as f64 / total.max(1.0);
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum()
}

fn max_share<K: Eq + std::hash::Hash>(counts: &HashMap<K, usize>) -> f64 {
    let total: f64 = counts.values().copied().map(|count| count as f64).sum();
    counts.values().copied().max().unwrap_or(0) as f64 / total.max(1.0)
}

#[test]
fn composition_modes_have_real_spread() {
    let mut counts: HashMap<CompositionMode, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_5A5A_5A5A;
        let resolved = resolve_sample(seed);
        *counts.entry(resolved.composition_mode).or_default() += 1;
    }

    assert!(counts.len() >= 4, "composition spread collapsed: {counts:?}");
    assert!(
        shannon_entropy_bits(&counts) >= 1.9,
        "composition entropy too low: {counts:?}",
    );
    assert!(max_share(&counts) <= 0.45, "one composition mode dominates: {counts:?}");
}

#[test]
fn finish_clusters_bias_toward_sharper_outputs() {
    let mut counts: HashMap<FinishCluster, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0xC6BC_2796_3A0C_43B3) ^ 0xDEAD_BEEF_F00D_BAAD;
        let resolved = resolve_sample(seed);
        *counts.entry(resolved.finish_cluster).or_default() += 1;
    }

    let crisp = *counts.get(&FinishCluster::Crisp).unwrap_or(&0);
    let balanced = *counts.get(&FinishCluster::Balanced).unwrap_or(&0);
    let velvet = *counts.get(&FinishCluster::Velvet).unwrap_or(&0);

    assert!(crisp > velvet, "crisp should appear more often than velvet: {counts:?}");
    assert!(
        crisp + balanced > velvet * 2,
        "sharp/balanced outcomes should substantially outweigh velvet ones: {counts:?}",
    );
    assert!(
        velvet as f64 / SAMPLE_COUNT as f64 <= 0.30,
        "velvet share should stay intentionally rare: {counts:?}",
    );
}

#[test]
fn grade_families_and_presets_cover_multiple_moods() {
    let mut family_counts: HashMap<GradeFamily, usize> = HashMap::new();
    let mut preset_counts: HashMap<GradePreset, usize> = HashMap::new();
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0x94D0_49BB_1331_11EB) ^ 0x1234_5678_CAFE_BABE;
        let resolved = resolve_sample(seed);
        *family_counts.entry(resolved.grade_family).or_default() += 1;
        *preset_counts.entry(resolved.grade_preset).or_default() += 1;
    }

    assert_eq!(family_counts.len(), 4, "all grade families should appear: {family_counts:?}");
    assert!(
        shannon_entropy_bits(&family_counts) >= 1.6,
        "grade family entropy too low: {family_counts:?}",
    );
    assert!(
        preset_counts.len() >= 8,
        "too few grade presets surfaced through family selection: {preset_counts:?}",
    );
}

#[test]
fn framing_has_both_centered_and_shifted_layouts() {
    let mut centered = 0usize;
    let mut shifted = 0usize;
    let mut wide = 0usize;
    for i in 0..SAMPLE_COUNT {
        let seed = (i as u64).wrapping_mul(0xD137_4DAC_C2D4_6EDB);
        let resolved = resolve_sample(seed);
        if resolved.framing_shift_x.abs() < 1e-9 && resolved.framing_shift_y.abs() < 1e-9 {
            centered += 1;
        } else {
            shifted += 1;
        }
        if resolved.framing_zoom > 1.08 {
            wide += 1;
        }
    }

    assert!(centered > 12, "too few centered layouts: {centered}");
    assert!(shifted > 40, "too few shifted layouts: {shifted}");
    assert!(wide > 24, "too few wide layouts: {wide}");
}
