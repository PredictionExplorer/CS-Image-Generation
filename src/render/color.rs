//! Color space conversions and utilities

use crate::render::constants::{
    BASE_HUE_DRIFT, HUE_DRIFT_SCALE, HUE_FULL_CIRCLE, HUE_WAVE_AMPLITUDE, OKLAB_CHROMA_BASE,
    OKLAB_CHROMA_BASE_BOOSTED, OKLAB_CHROMA_RANGE, OKLAB_CHROMA_RANGE_BOOSTED,
    OKLAB_CHROMA_WAVE_AMPLITUDE, OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED, OKLAB_LIGHTNESS_BASE,
    OKLAB_LIGHTNESS_RANGE, OKLAB_LIGHTNESS_WAVE_AMPLITUDE,
};
use crate::sim::Sha3RandomByteStream;
use std::sync::{LazyLock, Mutex};
use tracing::info;

/// Type alias for `OKLab` color (L, a, b components)
pub type OklabColor = (f64, f64, f64);

/// Small random hue variation for visual interest
const HUE_DRIFT_JITTER: f64 = 0.1;

static LAST_PALETTE_METADATA: LazyLock<Mutex<(&'static str, &'static str)>> =
    LazyLock::new(|| Mutex::new(("unresolved", "unresolved")));

#[derive(Clone, Copy)]
struct MoodEnvelope {
    name: &'static str,
    center_hue: f64,
    hue_span: f64,
    chroma_base: f64,
    chroma_range: f64,
    chroma_wave: f64,
    lightness_base: f64,
    lightness_range: f64,
    lightness_wave: f64,
}

#[derive(Clone, Copy)]
struct PaletteSpec {
    harmony: &'static str,
    mood: MoodEnvelope,
    base_hue: f64,
    offsets: [f64; 3],
}

const MOODS: [MoodEnvelope; 6] = [
    MoodEnvelope {
        name: "aurora",
        center_hue: 168.0,
        hue_span: 132.0,
        chroma_base: 0.22,
        chroma_range: 0.11,
        chroma_wave: 0.055,
        lightness_base: 0.69,
        lightness_range: 0.20,
        lightness_wave: 0.15,
    },
    MoodEnvelope {
        name: "ember",
        center_hue: 28.0,
        hue_span: 86.0,
        chroma_base: 0.24,
        chroma_range: 0.10,
        chroma_wave: 0.050,
        lightness_base: 0.64,
        lightness_range: 0.24,
        lightness_wave: 0.16,
    },
    MoodEnvelope {
        name: "nebula",
        center_hue: 286.0,
        hue_span: 112.0,
        chroma_base: 0.23,
        chroma_range: 0.10,
        chroma_wave: 0.060,
        lightness_base: 0.66,
        lightness_range: 0.21,
        lightness_wave: 0.17,
    },
    MoodEnvelope {
        name: "bioluminescence",
        center_hue: 196.0,
        hue_span: 76.0,
        chroma_base: 0.21,
        chroma_range: 0.12,
        chroma_wave: 0.065,
        lightness_base: 0.70,
        lightness_range: 0.18,
        lightness_wave: 0.14,
    },
    MoodEnvelope {
        name: "deep_ocean",
        center_hue: 222.0,
        hue_span: 92.0,
        chroma_base: 0.18,
        chroma_range: 0.11,
        chroma_wave: 0.050,
        lightness_base: 0.58,
        lightness_range: 0.25,
        lightness_wave: 0.17,
    },
    MoodEnvelope {
        name: "solar_opal",
        center_hue: 58.0,
        hue_span: 72.0,
        chroma_base: 0.23,
        chroma_range: 0.10,
        chroma_wave: 0.060,
        lightness_base: 0.71,
        lightness_range: 0.17,
        lightness_wave: 0.13,
    },
];

/// Return the harmony and mood chosen by the most recent body palette generation.
#[must_use]
pub fn current_palette_metadata() -> (&'static str, &'static str) {
    LAST_PALETTE_METADATA.lock().map_or(("unresolved", "unresolved"), |metadata| *metadata)
}

fn harmony_offsets(rng: &mut Sha3RandomByteStream) -> (&'static str, [f64; 3]) {
    match (rng.next_f64() * 6.0).floor() as usize {
        0 => ("analogous", [0.0, 28.0, -32.0]),
        1 => ("complementary", [0.0, 180.0, 150.0]),
        2 => ("split_complementary", [0.0, 150.0, 210.0]),
        3 => ("triadic", [0.0, 118.0, 242.0]),
        4 => ("tetradic", [0.0, 88.0, 180.0]),
        _ => ("golden_angle", [0.0, 137.5, 275.0]),
    }
}

fn resolve_palette_spec(rng: &mut Sha3RandomByteStream, chroma_boost: bool) -> PaletteSpec {
    let mood_index = ((rng.next_f64() * MOODS.len() as f64).floor() as usize).min(MOODS.len() - 1);
    let mut mood = MOODS[mood_index];
    if !chroma_boost {
        mood.chroma_base = (mood.chroma_base - 0.04).max(OKLAB_CHROMA_BASE);
        mood.chroma_range = (mood.chroma_range + 0.02).min(OKLAB_CHROMA_RANGE + 0.04);
        mood.chroma_wave = mood.chroma_wave.min(OKLAB_CHROMA_WAVE_AMPLITUDE);
    }

    let (harmony, offsets) = harmony_offsets(rng);
    let base_hue = mood.center_hue + (rng.next_f64() - 0.5) * mood.hue_span + rng.next_f64() * 11.0;

    PaletteSpec { harmony, mood, base_hue: base_hue.rem_euclid(HUE_FULL_CIRCLE), offsets }
}

fn classic_palette_spec(rng: &mut Sha3RandomByteStream, chroma_boost: bool) -> PaletteSpec {
    let mood = MoodEnvelope {
        name: "classic",
        center_hue: 0.0,
        hue_span: HUE_FULL_CIRCLE,
        chroma_base: if chroma_boost { OKLAB_CHROMA_BASE_BOOSTED } else { OKLAB_CHROMA_BASE },
        chroma_range: if chroma_boost { OKLAB_CHROMA_RANGE_BOOSTED } else { OKLAB_CHROMA_RANGE },
        chroma_wave: if chroma_boost {
            OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED
        } else {
            OKLAB_CHROMA_WAVE_AMPLITUDE
        },
        lightness_base: OKLAB_LIGHTNESS_BASE,
        lightness_range: OKLAB_LIGHTNESS_RANGE,
        lightness_wave: OKLAB_LIGHTNESS_WAVE_AMPLITUDE,
    };

    PaletteSpec {
        harmony: "classic_triad",
        mood,
        base_hue: rng.next_f64() * HUE_FULL_CIRCLE,
        offsets: [0.0, 120.0, 240.0],
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
    let palette = classic_palette_spec(rng, chroma_boost);
    generate_color_gradient_with_palette(
        rng,
        length,
        body_index,
        base_hue_offset,
        hue_wave_freq,
        palette,
    )
}

fn generate_color_gradient_with_palette(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue_offset: f64,
    hue_wave_freq: f64,
    palette: PaletteSpec,
) -> Vec<OklabColor> {
    let mut colors = Vec::with_capacity(length);

    let body_offset = palette.offsets[body_index % palette.offsets.len()];
    let base_hue = palette.base_hue + body_offset;
    let phase_jitter = rng.next_f64() * 0.1;

    let ln_cache: Vec<f64> =
        (0..length).map(|i| if i > 0 { (i as f64).ln() } else { 0.0 }).collect();
    let wave_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let phase_offset = body_index as f64 * 0.33 + phase_jitter;
            ((phase_offset + t * hue_wave_freq) * std::f64::consts::TAU).sin()
        })
        .collect();

    let random_bits: Vec<u8> = (0..length).map(|_| rng.next_byte()).collect();
    let random_chromas: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();
    let random_lightnesses: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();

    for step in 0..length {
        let mut current_hue = base_hue
            + base_hue_offset * (1.0 + ln_cache[step]) * HUE_DRIFT_SCALE
            + wave_cache[step] * HUE_WAVE_AMPLITUDE;

        if random_bits[step] & 1 == 0 {
            current_hue += HUE_DRIFT_JITTER;
        } else {
            current_hue -= HUE_DRIFT_JITTER;
        }
        current_hue = current_hue.rem_euclid(HUE_FULL_CIRCLE);

        let wave_factor = wave_cache[step];
        let chroma = (palette.mood.chroma_base
            + random_chromas[step] * palette.mood.chroma_range
            + wave_factor * palette.mood.chroma_wave
            + body_index as f64 * 0.01)
            .max(0.0);

        let lightness = (palette.mood.lightness_base
            + random_lightnesses[step] * palette.mood.lightness_range
            + wave_factor * palette.mood.lightness_wave
            + body_index as f64 * 0.015)
            .clamp(0.0, 1.0);

        let hue_rad = current_hue.to_radians();
        let a = chroma * hue_rad.cos();
        let b = chroma * hue_rad.sin();

        colors.push((lightness, a, b));
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
) -> (Vec<Vec<OklabColor>>, Vec<f64>) {
    let base_hue_offset = BASE_HUE_DRIFT;

    // #14: randomize hue wave frequency per seed for unique color rhythm
    let hue_wave_freq = 1.8 + rng.next_f64() * 2.2; // [1.8, 4.0]
    let palette = resolve_palette_spec(rng, chroma_boost);
    if let Ok(mut metadata) = LAST_PALETTE_METADATA.lock() {
        *metadata = (palette.harmony, palette.mood.name);
    }
    info!(
        "   => Palette harmony={} mood={} base_hue={:.1}",
        palette.harmony, palette.mood.name, palette.base_hue
    );

    let b1 = generate_color_gradient_with_palette(
        rng,
        length,
        0,
        base_hue_offset,
        hue_wave_freq,
        palette,
    );
    let b2 = generate_color_gradient_with_palette(
        rng,
        length,
        1,
        base_hue_offset,
        hue_wave_freq,
        palette,
    );
    let b3 = generate_color_gradient_with_palette(
        rng,
        length,
        2,
        base_hue_offset,
        hue_wave_freq,
        palette,
    );

    let body_alphas = if alpha_variation {
        // Shuffle [13M, 15M, 17M] using the RNG for per-body depth hierarchy
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

        let avg_chroma = |cols: &[(f64, f64, f64)]| {
            cols.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>() / cols.len() as f64
        };
        assert!(
            avg_chroma(&boosted) > avg_chroma(&normal),
            "Boosted chroma should produce higher average saturation"
        );
    }

    #[test]
    fn test_body_color_sequences_uniform_alpha() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (colors, alphas) =
            generate_body_color_sequences(&mut rng, 50, 15_000_000, false, false);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for &a in &alphas {
            assert_eq!(a, 1.0 / 15_000_000.0);
        }
    }

    #[test]
    fn test_body_color_sequences_alpha_variation() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (_, alphas) = generate_body_color_sequences(&mut rng, 50, 15_000_000, false, true);

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
            generate_body_color_sequences(&mut rng1, steps, 15_000_000, true, true);

        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors2, alphas2) =
            generate_body_color_sequences(&mut rng2, steps, 15_000_000, true, true);

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
}
