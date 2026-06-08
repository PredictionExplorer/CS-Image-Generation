//! Procedural color generation for the spectral renderer.

use crate::oklab::{max_display_p3_chroma_for_lh, oklch_to_oklab};
use crate::render::constants::{
    BASE_HUE_DRIFT, HUE_DRIFT_SCALE, HUE_FULL_CIRCLE, HUE_WAVE_AMPLITUDE,
};
use crate::sim::Sha3RandomByteStream;
use std::sync::{LazyLock, Mutex};
use tracing::info;

/// Type alias for `OKLab` color (L, a, b components)
pub type OklabColor = (f64, f64, f64);

/// Small random hue variation for visual interest
const HUE_DRIFT_JITTER: f64 = 0.1;
const COLOR_RNG_DOMAIN: &[u8] = b"cosmic-color/v2";
const GLOW_LIGHTNESS_FLOOR: f64 = 0.62;
const DOMINANT_CHROMA_FRACTION_FLOOR: f64 = 0.62;

/// Minimum inter-body hue gap (degrees) at full spread. Scaling by `spread`
/// still allows near-monochrome palettes at low spread, while this floor keeps
/// two bodies from landing on a near-identical hue when the palette is wide.
const MIN_BODY_HUE_GAP_DEG: f64 = 10.0;
/// Maximum inter-body hue gap (degrees) at full spread, large enough to reach
/// complementary and widely separated relationships.
const MAX_BODY_HUE_GAP_DEG: f64 = 175.0;

static LAST_PALETTE_METADATA: LazyLock<Mutex<(String, String)>> =
    LazyLock::new(|| Mutex::new(("unresolved".to_string(), "unresolved".to_string())));

#[derive(Clone, Copy)]
struct BodyColorPlan {
    base_hue: f64,
    target_lightness: f64,
    lightness_range: f64,
    lightness_wave: f64,
    chroma_fraction: f64,
    chroma_noise: f64,
    chroma_wave: f64,
    hue_journey: f64,
    phase: f64,
    is_dominant: bool,
}

#[derive(Clone)]
struct PaletteSpec {
    harmony: String,
    mood_label: String,
    bodies: [BodyColorPlan; 3],
    palette_phase: f64,
    hue_accent_strength: f64,
    lightness_contrast: f64,
}

/// Return the harmony and mood chosen by the most recent body palette generation.
#[must_use]
pub fn current_palette_metadata() -> (String, String) {
    LAST_PALETTE_METADATA.lock().map_or_else(
        |_| ("unresolved".to_string(), "unresolved".to_string()),
        |metadata| metadata.clone(),
    )
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn shuffle3(rng: &mut Sha3RandomByteStream, values: &mut [usize; 3]) {
    for i in (1..3).rev() {
        let j = (rng.next_f64() * (i + 1) as f64).floor() as usize;
        values.swap(i, j);
    }
}

/// Three body hues from an anchor and a continuous `spread` scalar.
///
/// The two inter-body hue gaps are drawn independently and scaled by `spread`,
/// so palettes range continuously and without bias from near-monochrome (small
/// spread) through analogous, split, complementary, and widely separated
/// relationships. No fixed angular structure (such as a 120-degree triad) and
/// no warm/cool axis is privileged; the downstream lightness/chroma hierarchy
/// and gamut-relative chroma keep every relationship tasteful.
fn body_hues(rng: &mut Sha3RandomByteStream, anchor: f64, spread: f64) -> [f64; 3] {
    let gap1 = lerp(MIN_BODY_HUE_GAP_DEG, MAX_BODY_HUE_GAP_DEG, rng.next_f64()) * spread;
    let gap2 = lerp(MIN_BODY_HUE_GAP_DEG, MAX_BODY_HUE_GAP_DEG, rng.next_f64()) * spread;
    [anchor, anchor + gap1, anchor + gap1 + gap2].map(|hue| hue.rem_euclid(HUE_FULL_CIRCLE))
}

fn assign_body_plans(
    rng: &mut Sha3RandomByteStream,
    hues: [f64; 3],
    chroma_boost: bool,
    key: f64,
    spread: f64,
) -> [BodyColorPlan; 3] {
    let mut lightness_order = [0, 1, 2];
    let mut chroma_order = [0, 1, 2];
    shuffle3(rng, &mut lightness_order);
    shuffle3(rng, &mut chroma_order);

    let center = (lerp(0.50, 0.73, key) + (rng.next_f64() - 0.5) * 0.05).clamp(0.48, 0.75);
    let lightness_values = [
        (center + lerp(0.105, 0.175, rng.next_f64())).clamp(0.63, 0.90),
        (center + (rng.next_f64() - 0.5) * 0.035).clamp(0.47, 0.78),
        (center - lerp(0.115, 0.19, rng.next_f64())).clamp(0.34, 0.62),
    ];

    let chroma_values = if chroma_boost {
        [
            lerp(0.82, 0.97, rng.next_f64()),
            lerp(0.58, 0.76, rng.next_f64()),
            lerp(0.34, 0.54, rng.next_f64()),
        ]
    } else {
        [
            lerp(0.66, 0.84, rng.next_f64()),
            lerp(0.46, 0.64, rng.next_f64()),
            lerp(0.26, 0.44, rng.next_f64()),
        ]
    };

    let journey_max = lerp(30.0, 96.0, spread);

    let mut plans = [BodyColorPlan {
        base_hue: 0.0,
        target_lightness: 0.65,
        lightness_range: 0.08,
        lightness_wave: 0.08,
        chroma_fraction: 0.6,
        chroma_noise: 0.08,
        chroma_wave: 0.04,
        hue_journey: 0.0,
        phase: 0.0,
        is_dominant: false,
    }; 3];

    for body in 0..3 {
        let lightness_rank = lightness_order.iter().position(|idx| *idx == body).unwrap_or(1);
        let chroma_rank = chroma_order.iter().position(|idx| *idx == body).unwrap_or(1);
        let is_dominant = chroma_rank == 0;
        let mut target_lightness = lightness_values[lightness_rank];
        if is_dominant {
            target_lightness = target_lightness.max(GLOW_LIGHTNESS_FLOOR);
        }

        plans[body] = BodyColorPlan {
            base_hue: hues[body],
            target_lightness,
            lightness_range: lerp(0.035, 0.105, rng.next_f64()),
            lightness_wave: lerp(0.045, 0.13, rng.next_f64()),
            chroma_fraction: chroma_values[chroma_rank],
            chroma_noise: lerp(0.04, 0.14, rng.next_f64()),
            chroma_wave: lerp(0.025, 0.095, rng.next_f64()),
            hue_journey: (rng.next_f64() - 0.5) * 2.0 * journey_max,
            phase: rng.next_f64(),
            is_dominant,
        };
    }

    plans
}

fn resolve_palette_spec(
    rng: &mut Sha3RandomByteStream,
    chroma_boost: bool,
    palette_phase: f64,
) -> PaletteSpec {
    let palette_phase = palette_phase.clamp(0.0, 1.0);

    // A few continuous knobs, each a plain uniform draw, define the whole palette
    // - no modes, no scoring, no rejection sampling. The hue anchor is uniform
    // and the inter-body spacing is randomized (see `body_hues`), so no colour or
    // relationship is privileged. Beauty is guaranteed by construction downstream
    // via gamut-relative chroma and the lightness/chroma hierarchy.
    let anchor = rng.next_f64() * HUE_FULL_CIRCLE;
    let spread = rng.next_f64();
    let key = rng.next_f64();

    let hues = body_hues(rng, anchor, spread);
    let bodies = assign_body_plans(rng, hues, chroma_boost, key, spread);

    PaletteSpec {
        harmony: format!("randomized_spread_{spread:.2}"),
        mood_label: "hue_neutral".to_string(),
        bodies,
        palette_phase,
        hue_accent_strength: lerp(8.0, 34.0, rng.next_f64()),
        lightness_contrast: lerp(0.88, 1.20, rng.next_f64()),
    }
}

/// Generate color gradient optimized for `OKLab` space.
///
/// Generates colors in `OKLCh` (cylindrical `OKLab`) for perceptually
/// uniform distribution. `chroma_boost` selects richer saturation
/// constants; `hue_wave_freq` controls per-seed color rhythm.
pub fn generate_color_gradient_oklab(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue_offset: f64,
    chroma_boost: bool,
    hue_wave_freq: f64,
) -> Vec<OklabColor> {
    let palette = resolve_palette_spec(rng, chroma_boost, 0.5);
    generate_color_gradient_with_palette(
        rng,
        length,
        body_index,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    )
}

fn generate_color_gradient_with_palette(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue_offset: f64,
    hue_wave_freq: f64,
    palette: &PaletteSpec,
) -> Vec<OklabColor> {
    let mut colors = Vec::with_capacity(length);

    let body = palette.bodies[body_index % palette.bodies.len()];
    let base_hue = body.base_hue;
    let phase_jitter = rng.next_f64() * 0.1 + palette.palette_phase + body.phase * 0.07;

    let ln_cache: Vec<f64> =
        (0..length).map(|i| if i > 0 { (i as f64).ln() } else { 0.0 }).collect();
    let wave_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let phase_offset = body_index as f64 * 0.33 + phase_jitter + body.phase;
            ((phase_offset + t * hue_wave_freq) * std::f64::consts::TAU).sin()
        })
        .collect();
    let accent_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let accent_phase = palette.palette_phase + body_index as f64 * 0.618 + body.phase;
            ((accent_phase + t * (hue_wave_freq * 0.37 + 0.71)) * std::f64::consts::TAU).sin()
        })
        .collect();

    let random_bits: Vec<u8> = (0..length).map(|_| rng.next_byte()).collect();
    let random_chromas: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();
    let random_lightnesses: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();

    for step in 0..length {
        let t = step as f64 / length.max(1) as f64;
        let journey = (smoothstep(t) - 0.5) * body.hue_journey;
        let mut current_hue = base_hue
            + journey
            + base_hue_offset * (1.0 + ln_cache[step]) * HUE_DRIFT_SCALE
            + wave_cache[step] * HUE_WAVE_AMPLITUDE
            + accent_cache[step] * palette.hue_accent_strength;

        if random_bits[step] & 1 == 0 {
            current_hue += HUE_DRIFT_JITTER;
        } else {
            current_hue -= HUE_DRIFT_JITTER;
        }
        current_hue = current_hue.rem_euclid(HUE_FULL_CIRCLE);

        let wave_factor = wave_cache[step];
        let accent_factor = accent_cache[step];
        let mut lightness = body.target_lightness
            + (random_lightnesses[step] - 0.5) * body.lightness_range
            + wave_factor * body.lightness_wave * palette.lightness_contrast
            + accent_factor * 0.018;
        if body.is_dominant {
            lightness = lightness.max(GLOW_LIGHTNESS_FLOOR);
        }
        lightness = lightness.clamp(0.34, 0.94);

        let mut chroma_fraction = body.chroma_fraction
            + (random_chromas[step] - 0.5) * body.chroma_noise
            + wave_factor * body.chroma_wave
            + accent_factor.abs() * 0.035;
        if body.is_dominant {
            chroma_fraction = chroma_fraction.max(DOMINANT_CHROMA_FRACTION_FLOOR);
        }
        chroma_fraction = chroma_fraction.clamp(0.10, 0.985);
        let max_chroma = max_display_p3_chroma_for_lh(lightness, current_hue);
        let chroma_floor = if body.is_dominant { 0.055 } else { 0.026 };
        let mut chroma = max_chroma * chroma_fraction;
        if max_chroma > chroma_floor {
            chroma = chroma.max(chroma_floor);
        }
        chroma = chroma.min(max_chroma * 0.995);

        colors.push(oklch_to_oklab(lightness, chroma, current_hue));
    }

    colors
}

/// Generate 3 color sequences + per-body alphas.
///
/// `chroma_boost`: use richer saturation constants.
/// `alpha_variation`: give each body a slightly different alpha for depth.
pub fn generate_body_color_sequences(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    alpha_denom: usize,
    chroma_boost: bool,
    alpha_variation: bool,
    palette_phase: f64,
) -> (Vec<Vec<OklabColor>>, Vec<f64>) {
    let mut rng = rng.fork(COLOR_RNG_DOMAIN);
    let base_hue_offset = BASE_HUE_DRIFT;

    let palette_phase = palette_phase.clamp(0.0, 1.0);
    let hue_wave_freq = 1.8 + rng.next_f64() * 2.2 + palette_phase * 0.45; // [1.8, 4.45]
    let palette = resolve_palette_spec(&mut rng, chroma_boost, palette_phase);
    if let Ok(mut metadata) = LAST_PALETTE_METADATA.lock() {
        *metadata = (palette.harmony.clone(), palette.mood_label.clone());
    }
    info!(
        "   => Palette harmony={} mood={} phase={:.3}",
        palette.harmony, palette.mood_label, palette.palette_phase
    );

    let b1 = generate_color_gradient_with_palette(
        &mut rng,
        length,
        0,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    );
    let b2 = generate_color_gradient_with_palette(
        &mut rng,
        length,
        1,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    );
    let b3 = generate_color_gradient_with_palette(
        &mut rng,
        length,
        2,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    );

    let body_alphas = if alpha_variation {
        let mut denoms = [13_000_000.0_f64, 15_000_000.0, 17_000_000.0];
        for i in (1..3).rev() {
            let j = (rng.next_f64() * (i + 1) as f64).floor() as usize;
            denoms.swap(i, j);
        }
        let alphas = vec![1.0 / denoms[0], 1.0 / denoms[1], 1.0 / denoms[2]];
        info!(
            "   => Per-body alpha variation: {:.3e}, {:.3e}, {:.3e}",
            alphas[0], alphas[1], alphas[2]
        );
        alphas
    } else {
        let alpha_value = 1.0 / alpha_denom as f64;
        info!("   => Uniform body alpha: 1/{alpha_denom} = {alpha_value:.3e}");
        vec![alpha_value; 3]
    };

    (vec![b1, b2, b3], body_alphas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sha3RandomByteStream;

    fn avg_chroma(cols: &[(f64, f64, f64)]) -> f64 {
        cols.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>() / cols.len() as f64
    }

    #[test]
    fn test_color_gradient_generation() {
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 1.0, 1.0, 1.0, 1.0);
        let length = 100;
        let colors = generate_color_gradient_oklab(&mut rng, length, 0, BASE_HUE_DRIFT, false, 2.6);

        assert_eq!(colors.len(), length);
        for (l, a, b) in &colors {
            assert!(*l >= 0.0 && *l <= 1.0);
            assert!(*a >= -0.5 && *a <= 0.5);
            assert!(*b >= -0.5 && *b <= 0.5);
        }
    }

    #[test]
    fn test_color_gradient_chroma_boost() {
        let mut rng1 = Sha3RandomByteStream::new(&[1, 2, 3, 4], 1.0, 1.0, 1.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&[1, 2, 3, 4], 1.0, 1.0, 1.0, 1.0);

        let normal = generate_color_gradient_oklab(&mut rng1, 100, 0, BASE_HUE_DRIFT, false, 2.6);
        let boosted = generate_color_gradient_oklab(&mut rng2, 100, 0, BASE_HUE_DRIFT, true, 2.6);

        assert!(
            avg_chroma(&boosted) > avg_chroma(&normal),
            "Boosted chroma should produce higher average saturation"
        );
    }

    #[test]
    fn test_body_color_sequences_uniform_alpha() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (colors, alphas) =
            generate_body_color_sequences(&mut rng, 50, 15_000_000, false, false, 0.5);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for &a in &alphas {
            assert_eq!(a, 1.0 / 15_000_000.0);
        }
    }

    #[test]
    fn test_body_color_sequences_alpha_variation() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (_, alphas) = generate_body_color_sequences(&mut rng, 50, 15_000_000, false, true, 0.5);

        assert_eq!(alphas.len(), 3);
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "alpha_variation should produce different per-body alphas");
    }

    #[test]
    fn test_color_generation_determinism() {
        let seed = [0x10, 0x00, 0x33];
        let steps = 200;

        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors1, alphas1) =
            generate_body_color_sequences(&mut rng1, steps, 15_000_000, true, true, 0.37);

        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors2, alphas2) =
            generate_body_color_sequences(&mut rng2, steps, 15_000_000, true, true, 0.37);

        for body in 0..3 {
            assert_eq!(
                alphas1[body].to_bits(),
                alphas2[body].to_bits(),
                "alpha for body {body} diverged"
            );
            for step in 0..steps {
                let (l1, a1, b1) = colors1[body][step];
                let (l2, a2, b2) = colors2[body][step];
                assert_eq!(l1.to_bits(), l2.to_bits(), "body {body} step {step} L diverged");
                assert_eq!(a1.to_bits(), a2.to_bits(), "body {body} step {step} a diverged");
                assert_eq!(b1.to_bits(), b2.to_bits(), "body {body} step {step} b diverged");
            }
        }
    }

    #[test]
    fn test_color_subseed_is_independent_of_parent_rng_position() {
        let seed = [0x44, 0x22, 0x11];
        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        for _ in 0..512 {
            rng2.next_byte();
        }

        let (colors1, alphas1) =
            generate_body_color_sequences(&mut rng1, 80, 15_000_000, true, true, 0.41);
        let (colors2, alphas2) =
            generate_body_color_sequences(&mut rng2, 80, 15_000_000, true, true, 0.41);

        assert_eq!(alphas1, alphas2);
        assert_eq!(colors1, colors2);
    }

    #[test]
    fn test_procedural_palettes_cover_all_hue_regions() {
        let mut hue_bins = std::collections::HashSet::new();

        for seed in 0u8..64 {
            let mut rng = Sha3RandomByteStream::new(&[seed, 0xA5, 0x5A], 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(
                &mut rng,
                32,
                15_000_000,
                true,
                false,
                f64::from(seed) / 63.0,
            );

            for body_colors in &colors {
                for &(_, a, b) in body_colors.iter().step_by(8) {
                    let hue = b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE);
                    hue_bins.insert((hue / 30.0).floor() as u8);
                }
            }
        }

        assert_eq!(hue_bins.len(), 12, "procedural palettes should occupy every hue bin");
    }

    #[test]
    fn test_procedural_palettes_stay_display_p3_safe() {
        for seed in 0u8..24 {
            let mut rng = Sha3RandomByteStream::new(&[0x33, seed, 0x77], 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(
                &mut rng,
                64,
                15_000_000,
                true,
                false,
                f64::from(seed) / 23.0,
            );

            for body_colors in &colors {
                for &(l, a, b) in body_colors {
                    let chroma = (a * a + b * b).sqrt();
                    let hue = b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE);
                    let max_chroma = max_display_p3_chroma_for_lh(l, hue);
                    assert!((0.34..=0.94).contains(&l), "lightness out of bounds: {l}");
                    assert!(
                        chroma <= max_chroma * 1.000_001,
                        "chroma {chroma} exceeds Display P3 cusp {max_chroma}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_palettes_keep_lightness_hierarchy_and_vividness() {
        let span = |values: &[f64]| {
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min)
        };

        for seed in 0u8..32 {
            let mut rng = Sha3RandomByteStream::new(&[0x81, seed, 0x19], 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(
                &mut rng,
                96,
                15_000_000,
                true,
                false,
                f64::from(seed) / 31.0,
            );

            // Average over each body's sequence so the structural hierarchy is
            // measured rather than a single wave-modulated step.
            let mean_l: Vec<f64> = colors
                .iter()
                .map(|body| body.iter().map(|(l, _, _)| *l).sum::<f64>() / body.len() as f64)
                .collect();
            let mean_c: Vec<f64> = colors
                .iter()
                .map(|body| {
                    body.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>()
                        / body.len() as f64
                })
                .collect();
            let max_c = mean_c.iter().copied().fold(0.0, f64::max);

            assert!(
                span(&mean_l) > 0.05,
                "lightness hierarchy collapsed for seed {seed}: {mean_l:?}"
            );
            assert!(max_c > 0.05, "palette should stay vivid for seed {seed}: {mean_c:?}");
        }
    }

    #[test]
    fn test_body_hue_relationships_are_varied_not_triadic() {
        fn circular_gap(a: f64, b: f64) -> f64 {
            let d = (a - b).rem_euclid(HUE_FULL_CIRCLE);
            d.min(HUE_FULL_CIRCLE - d)
        }

        let total = 256usize;
        let mut near_triadic = 0usize;
        // Track the smallest and largest "widest pairwise gap" across seeds, so we
        // can confirm both tight (analogous/monochrome) and wide (complementary)
        // relationships occur rather than a single fixed structure.
        let mut tightest_max_gap = HUE_FULL_CIRCLE;
        let mut widest_max_gap = 0.0f64;

        for s in 0..total as u32 {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0x5a, 0xa5];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let palette = resolve_palette_spec(&mut rng, true, f64::from(s % 97) / 97.0);
            let hues: Vec<f64> = palette.bodies.iter().map(|body| body.base_hue).collect();
            let gaps = [
                circular_gap(hues[0], hues[1]),
                circular_gap(hues[1], hues[2]),
                circular_gap(hues[0], hues[2]),
            ];
            let max_gap = gaps.iter().copied().fold(0.0, f64::max);
            tightest_max_gap = tightest_max_gap.min(max_gap);
            widest_max_gap = widest_max_gap.max(max_gap);
            if gaps.iter().all(|g| (g - 120.0).abs() < 15.0) {
                near_triadic += 1;
            }
        }

        // The previous generator pinned nearly every palette to a 120-degree triad.
        assert!(near_triadic * 4 < total, "too many near-triadic palettes: {near_triadic}/{total}");
        assert!(
            tightest_max_gap < 45.0,
            "expected at least one tight palette, smallest widest-gap={tightest_max_gap:.1}"
        );
        assert!(
            widest_max_gap > 150.0,
            "expected at least one wide palette, largest widest-gap={widest_max_gap:.1}"
        );
    }
}
