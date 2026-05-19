//! Color space conversions and utilities

use crate::render::constants::{
    BASE_HUE_DRIFT, HUE_DRIFT_SCALE, HUE_FULL_CIRCLE, HUE_WAVE_AMPLITUDE, OKLAB_CHROMA_BASE,
    OKLAB_CHROMA_BASE_BOOSTED, OKLAB_CHROMA_RANGE, OKLAB_CHROMA_RANGE_BOOSTED,
    OKLAB_CHROMA_WAVE_AMPLITUDE, OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED, OKLAB_LIGHTNESS_BASE,
    OKLAB_LIGHTNESS_RANGE, OKLAB_LIGHTNESS_WAVE_AMPLITUDE,
};
use crate::sim::Sha3RandomByteStream;
use tracing::info;

/// Type alias for `OKLab` color (L, a, b components)
pub type OklabColor = (f64, f64, f64);

/// Small random hue variation for visual interest
const HUE_DRIFT_JITTER: f64 = 0.1;
const BODY_COUNT: usize = 3;
const MAX_THEME_ATTEMPTS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaletteThemeKind {
    NoirGold,
    SolarFlare,
    AlienJewel,
    RoyalPlasma,
    ArcticAurora,
    RoseChrome,
    ToxicLuxury,
    PrismaticBlack,
}

#[derive(Clone, Copy, Debug)]
struct PaletteTheme {
    kind: PaletteThemeKind,
    role_hues: [f64; BODY_COUNT],
    drift_scale: f64,
    wave_amplitude: f64,
    chroma_offsets: [f64; BODY_COUNT],
    lightness_offsets: [f64; BODY_COUNT],
}

fn wrap_hue(hue: f64) -> f64 {
    hue.rem_euclid(HUE_FULL_CIRCLE)
}

fn hue_distance(a: f64, b: f64) -> f64 {
    let delta = (a - b).abs().rem_euclid(HUE_FULL_CIRCLE);
    delta.min(HUE_FULL_CIRCLE - delta)
}

fn is_blue_green_cluster(hue: f64) -> bool {
    (95.0..=250.0).contains(&wrap_hue(hue))
}

fn is_warm_sector(hue: f64) -> bool {
    let hue = wrap_hue(hue);
    hue <= 78.0 || hue >= 300.0
}

fn jitter(rng: &mut Sha3RandomByteStream, degrees: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * degrees
}

fn pick_theme_kind(rng: &mut Sha3RandomByteStream) -> PaletteThemeKind {
    // Warm and warm/cool contrast themes are intentionally heavier than cool themes.
    const WEIGHTED_THEMES: &[PaletteThemeKind] = &[
        PaletteThemeKind::NoirGold,
        PaletteThemeKind::NoirGold,
        PaletteThemeKind::SolarFlare,
        PaletteThemeKind::SolarFlare,
        PaletteThemeKind::AlienJewel,
        PaletteThemeKind::RoyalPlasma,
        PaletteThemeKind::RoyalPlasma,
        PaletteThemeKind::ArcticAurora,
        PaletteThemeKind::RoseChrome,
        PaletteThemeKind::RoseChrome,
        PaletteThemeKind::ToxicLuxury,
        PaletteThemeKind::PrismaticBlack,
    ];
    let idx = (rng.next_f64() * WEIGHTED_THEMES.len() as f64).floor() as usize;
    WEIGHTED_THEMES[idx.min(WEIGHTED_THEMES.len() - 1)]
}

impl PaletteTheme {
    fn choose(rng: &mut Sha3RandomByteStream) -> Self {
        let fallback_root = rng.next_f64() * HUE_FULL_CIRCLE;
        for _ in 0..MAX_THEME_ATTEMPTS {
            let kind = pick_theme_kind(rng);
            let root = rng.next_f64() * HUE_FULL_CIRCLE;
            let theme = Self::from_kind(kind, root, rng);
            if theme.quality_score() >= 0.78 {
                return theme;
            }
        }

        Self::from_kind(PaletteThemeKind::NoirGold, fallback_root, rng)
    }

    fn from_kind(kind: PaletteThemeKind, root: f64, rng: &mut Sha3RandomByteStream) -> Self {
        let root = wrap_hue(root);
        match kind {
            PaletteThemeKind::NoirGold => Self {
                kind,
                role_hues: [
                    wrap_hue(280.0 + jitter(rng, 16.0)),
                    wrap_hue(45.0 + jitter(rng, 12.0)),
                    wrap_hue(190.0 + jitter(rng, 10.0)),
                ],
                drift_scale: 0.88,
                wave_amplitude: 26.0,
                chroma_offsets: [0.00, 0.04, 0.02],
                lightness_offsets: [-0.05, 0.05, 0.02],
            },
            PaletteThemeKind::SolarFlare => Self {
                kind,
                role_hues: [
                    wrap_hue(8.0 + jitter(rng, 18.0)),
                    wrap_hue(36.0 + jitter(rng, 14.0)),
                    wrap_hue(214.0 + jitter(rng, 12.0)),
                ],
                drift_scale: 1.02,
                wave_amplitude: 32.0,
                chroma_offsets: [0.05, 0.02, 0.03],
                lightness_offsets: [0.02, 0.06, -0.02],
            },
            PaletteThemeKind::AlienJewel => Self {
                kind,
                role_hues: [
                    wrap_hue(154.0 + jitter(rng, 12.0)),
                    wrap_hue(310.0 + jitter(rng, 16.0)),
                    wrap_hue(22.0 + jitter(rng, 12.0)),
                ],
                drift_scale: 0.96,
                wave_amplitude: 30.0,
                chroma_offsets: [0.02, 0.05, 0.03],
                lightness_offsets: [-0.01, 0.05, 0.02],
            },
            PaletteThemeKind::RoyalPlasma => Self {
                kind,
                role_hues: [
                    wrap_hue(268.0 + jitter(rng, 16.0)),
                    wrap_hue(320.0 + jitter(rng, 16.0)),
                    wrap_hue(48.0 + jitter(rng, 10.0)),
                ],
                drift_scale: 0.92,
                wave_amplitude: 34.0,
                chroma_offsets: [0.03, 0.05, 0.02],
                lightness_offsets: [-0.02, 0.04, 0.06],
            },
            PaletteThemeKind::ArcticAurora => Self {
                kind,
                role_hues: [
                    wrap_hue(205.0 + jitter(rng, 12.0)),
                    wrap_hue(284.0 + jitter(rng, 16.0)),
                    wrap_hue(326.0 + jitter(rng, 14.0)),
                ],
                drift_scale: 0.82,
                wave_amplitude: 24.0,
                chroma_offsets: [0.00, 0.04, 0.05],
                lightness_offsets: [0.02, 0.03, 0.05],
            },
            PaletteThemeKind::RoseChrome => Self {
                kind,
                role_hues: [
                    wrap_hue(340.0 + jitter(rng, 14.0)),
                    wrap_hue(268.0 + jitter(rng, 12.0)),
                    wrap_hue(204.0 + jitter(rng, 10.0)),
                ],
                drift_scale: 0.86,
                wave_amplitude: 25.0,
                chroma_offsets: [0.05, 0.02, 0.01],
                lightness_offsets: [0.05, 0.02, 0.04],
            },
            PaletteThemeKind::ToxicLuxury => Self {
                kind,
                role_hues: [
                    wrap_hue(74.0 + jitter(rng, 10.0)),
                    wrap_hue(286.0 + jitter(rng, 14.0)),
                    wrap_hue(24.0 + jitter(rng, 12.0)),
                ],
                drift_scale: 0.90,
                wave_amplitude: 22.0,
                chroma_offsets: [0.05, 0.02, 0.03],
                lightness_offsets: [0.06, -0.03, 0.02],
            },
            PaletteThemeKind::PrismaticBlack => {
                let second = wrap_hue(root + 138.0 + jitter(rng, 12.0));
                let mut third = wrap_hue(root + 274.0 + jitter(rng, 12.0));
                if [root, second, third].iter().all(|&h| is_blue_green_cluster(h)) {
                    third = wrap_hue(32.0 + jitter(rng, 10.0));
                }
                Self {
                    kind,
                    role_hues: [root, second, third],
                    drift_scale: 1.12,
                    wave_amplitude: 42.0,
                    chroma_offsets: [0.00, 0.02, 0.06],
                    lightness_offsets: [-0.06, 0.01, 0.07],
                }
            }
        }
    }

    fn quality_score(&self) -> f64 {
        let cool_count = self.role_hues.iter().filter(|&&h| is_blue_green_cluster(h)).count();
        let warm_count = self.role_hues.iter().filter(|&&h| is_warm_sector(h)).count();
        let min_distance = self
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                self.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(HUE_FULL_CIRCLE, f64::min);
        let max_distance = self
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                self.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(0.0_f64, f64::max);

        let has_analogous_pair_with_accent =
            min_distance < 35.0 && max_distance >= 130.0 && warm_count >= 2;
        let separation_score = if has_analogous_pair_with_accent {
            0.78
        } else {
            (min_distance / 105.0).clamp(0.0, 1.0)
        };
        let warm_score = if warm_count > 0 { 1.0 } else { 0.0 };
        let cool_balance_score = match cool_count {
            0 | 1 => 1.0,
            2 => 0.70,
            _ => 0.0,
        };
        let accent_score =
            if self.chroma_offsets.iter().any(|&offset| offset >= 0.05) { 1.0 } else { 0.75 };

        separation_score * 0.36
            + warm_score * 0.28
            + cool_balance_score * 0.26
            + accent_score * 0.10
    }

    fn hue_for_body(&self, body_index: usize) -> f64 {
        self.role_hues[body_index % BODY_COUNT]
    }
}

fn generate_color_gradient_from_hue(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue: f64,
    base_hue_offset: f64,
    chroma_boost: bool,
    hue_wave_freq: f64,
    drift_scale: f64,
    wave_amplitude: f64,
    chroma_offset: f64,
    lightness_offset: f64,
) -> Vec<OklabColor> {
    let chroma_base = if chroma_boost { OKLAB_CHROMA_BASE_BOOSTED } else { OKLAB_CHROMA_BASE };
    let chroma_range = if chroma_boost { OKLAB_CHROMA_RANGE_BOOSTED } else { OKLAB_CHROMA_RANGE };
    let chroma_wave = if chroma_boost {
        OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED
    } else {
        OKLAB_CHROMA_WAVE_AMPLITUDE
    };

    let mut colors = Vec::with_capacity(length);

    let ln_cache: Vec<f64> =
        (0..length).map(|i| if i > 0 { (i as f64).ln() } else { 0.0 }).collect();
    let wave_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let phase_offset = body_index as f64 * 0.33 + rng.next_f64() * 0.1;
            ((phase_offset + t * hue_wave_freq) * std::f64::consts::TAU).sin()
        })
        .collect();

    let random_bits: Vec<u8> = (0..length).map(|_| rng.next_byte()).collect();
    let random_chromas: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();
    let random_lightnesses: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();

    for step in 0..length {
        let mut current_hue = base_hue
            + base_hue_offset * (1.0 + ln_cache[step]) * drift_scale
            + wave_cache[step] * wave_amplitude;

        if random_bits[step] & 1 == 0 {
            current_hue += HUE_DRIFT_JITTER;
        } else {
            current_hue -= HUE_DRIFT_JITTER;
        }
        current_hue = wrap_hue(current_hue);

        let wave_factor = wave_cache[step];
        let chroma = (chroma_base
            + random_chromas[step] * chroma_range
            + wave_factor * chroma_wave
            + body_index as f64 * 0.01
            + chroma_offset)
            .max(0.0);

        let lightness = (OKLAB_LIGHTNESS_BASE
            + random_lightnesses[step] * OKLAB_LIGHTNESS_RANGE
            + wave_factor * OKLAB_LIGHTNESS_WAVE_AMPLITUDE
            + body_index as f64 * 0.015
            + lightness_offset)
            .clamp(0.0, 1.0);

        let hue_rad = current_hue.to_radians();
        let a = chroma * hue_rad.cos();
        let b = chroma * hue_rad.sin();

        colors.push((lightness, a, b));
    }

    colors
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
    let base_hue = rng.next_f64() * HUE_FULL_CIRCLE
        + body_index as f64 * 120.0
        + [0.0, 120.0, 240.0][body_index % BODY_COUNT];
    generate_color_gradient_from_hue(
        rng,
        length,
        body_index,
        base_hue,
        base_hue_offset,
        chroma_boost,
        hue_wave_freq,
        HUE_DRIFT_SCALE,
        HUE_WAVE_AMPLITUDE,
        0.0,
        0.0,
    )
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
    let palette = PaletteTheme::choose(rng);
    info!(
        "   => Color harmony: {:?} hues [{:.1}, {:.1}, {:.1}]",
        palette.kind, palette.role_hues[0], palette.role_hues[1], palette.role_hues[2]
    );

    let b1 = generate_color_gradient_from_hue(
        rng,
        length,
        0,
        palette.hue_for_body(0),
        base_hue_offset,
        chroma_boost,
        hue_wave_freq,
        palette.drift_scale,
        palette.wave_amplitude,
        palette.chroma_offsets[0],
        palette.lightness_offsets[0],
    );
    let b2 = generate_color_gradient_from_hue(
        rng,
        length,
        1,
        palette.hue_for_body(1),
        base_hue_offset,
        chroma_boost,
        hue_wave_freq,
        palette.drift_scale,
        palette.wave_amplitude,
        palette.chroma_offsets[1],
        palette.lightness_offsets[1],
    );
    let b3 = generate_color_gradient_from_hue(
        rng,
        length,
        2,
        palette.hue_for_body(2),
        base_hue_offset,
        chroma_boost,
        hue_wave_freq,
        palette.drift_scale,
        palette.wave_amplitude,
        palette.chroma_offsets[2],
        palette.lightness_offsets[2],
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

    fn hue_from_oklab((_, a, b): OklabColor) -> f64 {
        b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE)
    }

    fn circular_mean_hue(colors: &[OklabColor]) -> f64 {
        let (sum_cos, sum_sin) = colors.iter().fold((0.0, 0.0), |(sum_cos, sum_sin), &color| {
            let hue = hue_from_oklab(color).to_radians();
            (sum_cos + hue.cos(), sum_sin + hue.sin())
        });
        sum_sin.atan2(sum_cos).to_degrees().rem_euclid(HUE_FULL_CIRCLE)
    }

    fn assert_theme_quality(theme: PaletteTheme) {
        let cool_count = theme.role_hues.iter().filter(|&&hue| is_blue_green_cluster(hue)).count();
        let warm_count = theme.role_hues.iter().filter(|&&hue| is_warm_sector(hue)).count();
        let min_distance = theme
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                theme.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(HUE_FULL_CIRCLE, f64::min);
        let max_distance = theme
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                theme.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(0.0_f64, f64::max);
        let has_analogous_pair_with_accent =
            min_distance < 35.0 && max_distance >= 130.0 && warm_count >= 2;

        assert!(cool_count < 3, "{:?} produced an all-cool cluster", theme);
        assert!(warm_count > 0, "{:?} needs at least one warm anchor", theme);
        assert!(
            min_distance >= 35.0 || has_analogous_pair_with_accent,
            "{:?} roles are too tightly clustered without a far accent",
            theme
        );
        assert!(
            (0.70..=1.20).contains(&theme.drift_scale),
            "{:?} has unreasonable drift scale",
            theme
        );
        assert!(
            (20.0..=45.0).contains(&theme.wave_amplitude),
            "{:?} has unreasonable wave amplitude",
            theme
        );
        assert!(theme.quality_score() >= 0.78, "{:?} should pass the quality threshold", theme);
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

    #[test]
    fn test_palette_theme_quality_rejects_all_cool_cluster() {
        let theme = PaletteTheme {
            kind: PaletteThemeKind::ArcticAurora,
            role_hues: [130.0, 180.0, 230.0],
            drift_scale: 0.8,
            wave_amplitude: 24.0,
            chroma_offsets: [0.0, 0.0, 0.0],
            lightness_offsets: [0.0, 0.0, 0.0],
        };

        assert!(
            theme.quality_score() < 0.78,
            "all green/cyan/blue palettes should fail the quality gate"
        );
    }

    #[test]
    fn test_palette_theme_choice_keeps_warm_or_balanced_roles() {
        for seed_val in 0u8..80 {
            let seed = [seed_val, 0x42, 0x91, 0x17];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let theme = PaletteTheme::choose(&mut rng);

            assert_theme_quality(theme);
        }
    }

    #[test]
    fn test_all_curated_palette_theme_templates_pass_quality_gate() {
        let kinds = [
            PaletteThemeKind::NoirGold,
            PaletteThemeKind::SolarFlare,
            PaletteThemeKind::AlienJewel,
            PaletteThemeKind::RoyalPlasma,
            PaletteThemeKind::ArcticAurora,
            PaletteThemeKind::RoseChrome,
            PaletteThemeKind::ToxicLuxury,
            PaletteThemeKind::PrismaticBlack,
        ];

        for (idx, kind) in kinds.iter().copied().enumerate() {
            let seed = [idx as u8, 0xBA, 0x5E, 0x11];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let theme = PaletteTheme::from_kind(kind, 142.0, &mut rng);

            assert_theme_quality(theme);
        }
    }

    #[test]
    fn test_generated_body_hues_follow_coherent_theme_roles() {
        for seed_val in 0u8..32 {
            let seed = [seed_val, 0x10, 0x20, 0x30];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(&mut rng, 160, 15_000_000, true, false);

            let mean_hues = [
                circular_mean_hue(&colors[0]),
                circular_mean_hue(&colors[1]),
                circular_mean_hue(&colors[2]),
            ];
            let cool_count = mean_hues.iter().filter(|&&hue| is_blue_green_cluster(hue)).count();
            let warm_count = mean_hues.iter().filter(|&&hue| is_warm_sector(hue)).count();
            let min_distance = mean_hues
                .iter()
                .enumerate()
                .flat_map(|(idx, &hue)| {
                    mean_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
                })
                .fold(HUE_FULL_CIRCLE, f64::min);
            let max_distance = mean_hues
                .iter()
                .enumerate()
                .flat_map(|(idx, &hue)| {
                    mean_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
                })
                .fold(0.0_f64, f64::max);
            let has_analogous_pair_with_accent =
                min_distance < 28.0 && max_distance >= 130.0 && warm_count >= 2;

            assert!(cool_count < 3, "seed {seed_val} produced all-cool body hues: {mean_hues:?}");
            assert!(warm_count > 0, "seed {seed_val} lost its warm body anchor: {mean_hues:?}");
            assert!(
                min_distance >= 28.0 || has_analogous_pair_with_accent,
                "seed {seed_val} body hues collapsed without a far accent: {mean_hues:?}"
            );
        }
    }

    #[test]
    fn test_palette_theme_fallback_is_quality_checked() {
        let seed = [0xFE, 0xED, 0xFA, 0xCE];
        let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let theme = PaletteTheme::from_kind(PaletteThemeKind::NoirGold, 180.0, &mut rng);

        assert_theme_quality(theme);
    }
}
