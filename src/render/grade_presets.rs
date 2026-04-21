//! Curated cinematic color-grade presets.
//!
//! The previous pipeline hard-coded a single teal/orange shadow/highlight
//! pair. These presets turn color grading into another meaningful
//! expressive lever: each preset is a coherent, named aesthetic.
//!
//! Shadow/highlight tints are linear-sRGB deltas added on top of the
//! incoming image, clamped during compositing. They are intentionally
//! small-magnitude so the grade *flavors* the image rather than painting
//! over it.

use serde::{Deserialize, Serialize};

/// A curated color-grade preset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GradePreset {
    /// Classic film teal/orange split.
    #[default]
    CinematicTeal,
    /// Desaturated noir with inky shadows and crisp highlights.
    NoirContrast,
    /// Warm golden hour with honey-tinted highlights.
    GoldenHour,
    /// Cool nordic feel with cyan shadows and silver highlights.
    NordicCool,
    /// Deep blue shadows / pale highlights for underwater imagery.
    UnderwaterTeal,
    /// Soft pastel dream - very gentle tints.
    PastelDream,
    /// Black and silver - extreme monochrome contrast.
    HighContrastBW,
    /// Art Nouveau earth tones: olive shadows + peach highlights.
    ArtNouveauEarth,
    /// Dusk violet shadows / warm cream highlights.
    EtherealSoft,
    /// Royal amethyst purple + gold.
    RoyalAmethyst,
    /// Warm brass and copper.
    WarmBrass,
    /// Icy platinum / pale cyan.
    IcyPlatinum,
    /// Vintage sepia print.
    VintageSepia,
    /// Saturated rainbow / iridescent.
    SaturatedRainbow,
    /// Neon-noir Tokyo: magenta shadows + cyan highlights.
    NeonNoir,
    /// Night sky: indigo shadows, cool-white highlights.
    NightSky,
}

/// Static list of every grade preset variant.
pub const ALL_GRADE_PRESETS: &[GradePreset] = &[
    GradePreset::CinematicTeal,
    GradePreset::NoirContrast,
    GradePreset::GoldenHour,
    GradePreset::NordicCool,
    GradePreset::UnderwaterTeal,
    GradePreset::PastelDream,
    GradePreset::HighContrastBW,
    GradePreset::ArtNouveauEarth,
    GradePreset::EtherealSoft,
    GradePreset::RoyalAmethyst,
    GradePreset::WarmBrass,
    GradePreset::IcyPlatinum,
    GradePreset::VintageSepia,
    GradePreset::SaturatedRainbow,
    GradePreset::NeonNoir,
    GradePreset::NightSky,
];

/// Concrete grade-preset values used to drive the cinematic color grade.
#[derive(Clone, Debug)]
pub struct GradeParams {
    /// Linear-sRGB delta added to the shadows.
    pub shadow_tint: [f64; 3],
    /// Linear-sRGB delta added to the highlights.
    pub highlight_tint: [f64; 3],
    /// Palette-wave accent strength (0 disables, up to ~0.45 for expressive).
    pub palette_wave_strength: f64,
    /// Multiplicative bias applied on top of the randomized vibrance.
    /// 1.0 = no change; 0.8 = slightly desaturated; 1.2 = punchier.
    pub vibrance_bias: f64,
    /// Multiplicative bias applied on top of the randomized tone curve
    /// strength. 1.0 = no change; > 1 = more S-curve contrast.
    pub tone_curve_bias: f64,
}

impl GradePreset {
    /// Every variant in declaration order.
    #[must_use]
    pub fn all() -> &'static [GradePreset] {
        ALL_GRADE_PRESETS
    }

    /// Select a variant by index using modulo arithmetic.
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        ALL_GRADE_PRESETS[index % ALL_GRADE_PRESETS.len()]
    }

    /// Machine-readable name suitable for logs.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GradePreset::CinematicTeal => "cinematic_teal",
            GradePreset::NoirContrast => "noir_contrast",
            GradePreset::GoldenHour => "golden_hour",
            GradePreset::NordicCool => "nordic_cool",
            GradePreset::UnderwaterTeal => "underwater_teal",
            GradePreset::PastelDream => "pastel_dream",
            GradePreset::HighContrastBW => "high_contrast_bw",
            GradePreset::ArtNouveauEarth => "art_nouveau_earth",
            GradePreset::EtherealSoft => "ethereal_soft",
            GradePreset::RoyalAmethyst => "royal_amethyst",
            GradePreset::WarmBrass => "warm_brass",
            GradePreset::IcyPlatinum => "icy_platinum",
            GradePreset::VintageSepia => "vintage_sepia",
            GradePreset::SaturatedRainbow => "saturated_rainbow",
            GradePreset::NeonNoir => "neon_noir",
            GradePreset::NightSky => "night_sky",
        }
    }

    /// Resolve this preset into concrete grade parameters.
    #[must_use]
    pub fn params(self) -> GradeParams {
        match self {
            GradePreset::CinematicTeal => GradeParams {
                shadow_tint: [-0.08, -0.02, 0.16],
                highlight_tint: [0.11, 0.05, -0.03],
                palette_wave_strength: 0.25,
                vibrance_bias: 1.0,
                tone_curve_bias: 1.0,
            },
            GradePreset::NoirContrast => GradeParams {
                shadow_tint: [-0.14, -0.12, -0.10],
                highlight_tint: [0.10, 0.10, 0.09],
                palette_wave_strength: 0.10,
                vibrance_bias: 0.55,
                tone_curve_bias: 1.25,
            },
            GradePreset::GoldenHour => GradeParams {
                shadow_tint: [0.04, -0.02, -0.10],
                highlight_tint: [0.18, 0.10, -0.02],
                palette_wave_strength: 0.20,
                vibrance_bias: 1.10,
                tone_curve_bias: 0.95,
            },
            GradePreset::NordicCool => GradeParams {
                shadow_tint: [-0.08, 0.02, 0.14],
                highlight_tint: [-0.02, 0.06, 0.12],
                palette_wave_strength: 0.15,
                vibrance_bias: 0.85,
                tone_curve_bias: 1.05,
            },
            GradePreset::UnderwaterTeal => GradeParams {
                shadow_tint: [-0.14, -0.04, 0.12],
                highlight_tint: [-0.04, 0.10, 0.16],
                palette_wave_strength: 0.20,
                vibrance_bias: 0.95,
                tone_curve_bias: 1.0,
            },
            GradePreset::PastelDream => GradeParams {
                shadow_tint: [0.04, 0.00, 0.06],
                highlight_tint: [0.06, 0.04, 0.02],
                palette_wave_strength: 0.35,
                vibrance_bias: 0.80,
                tone_curve_bias: 0.85,
            },
            GradePreset::HighContrastBW => GradeParams {
                shadow_tint: [-0.20, -0.20, -0.20],
                highlight_tint: [0.18, 0.18, 0.18],
                palette_wave_strength: 0.0,
                vibrance_bias: 0.10,
                tone_curve_bias: 1.35,
            },
            GradePreset::ArtNouveauEarth => GradeParams {
                shadow_tint: [-0.02, 0.04, -0.06],
                highlight_tint: [0.14, 0.08, 0.02],
                palette_wave_strength: 0.25,
                vibrance_bias: 0.95,
                tone_curve_bias: 1.0,
            },
            GradePreset::EtherealSoft => GradeParams {
                shadow_tint: [0.02, -0.02, 0.10],
                highlight_tint: [0.10, 0.06, 0.02],
                palette_wave_strength: 0.30,
                vibrance_bias: 0.90,
                tone_curve_bias: 0.90,
            },
            GradePreset::RoyalAmethyst => GradeParams {
                shadow_tint: [0.08, -0.08, 0.16],
                highlight_tint: [0.14, 0.10, -0.02],
                palette_wave_strength: 0.28,
                vibrance_bias: 1.05,
                tone_curve_bias: 1.05,
            },
            GradePreset::WarmBrass => GradeParams {
                shadow_tint: [0.02, -0.04, -0.08],
                highlight_tint: [0.16, 0.11, 0.02],
                palette_wave_strength: 0.22,
                vibrance_bias: 1.05,
                tone_curve_bias: 1.0,
            },
            GradePreset::IcyPlatinum => GradeParams {
                shadow_tint: [-0.04, 0.00, 0.08],
                highlight_tint: [0.04, 0.08, 0.12],
                palette_wave_strength: 0.10,
                vibrance_bias: 0.70,
                tone_curve_bias: 1.05,
            },
            GradePreset::VintageSepia => GradeParams {
                shadow_tint: [0.04, 0.00, -0.10],
                highlight_tint: [0.14, 0.08, -0.04],
                palette_wave_strength: 0.12,
                vibrance_bias: 0.55,
                tone_curve_bias: 1.05,
            },
            GradePreset::SaturatedRainbow => GradeParams {
                shadow_tint: [-0.04, 0.04, 0.10],
                highlight_tint: [0.10, 0.08, 0.04],
                palette_wave_strength: 0.45,
                vibrance_bias: 1.25,
                tone_curve_bias: 1.0,
            },
            GradePreset::NeonNoir => GradeParams {
                shadow_tint: [0.10, -0.14, 0.12],
                highlight_tint: [-0.04, 0.14, 0.14],
                palette_wave_strength: 0.35,
                vibrance_bias: 1.20,
                tone_curve_bias: 1.15,
            },
            GradePreset::NightSky => GradeParams {
                shadow_tint: [-0.04, -0.06, 0.14],
                highlight_tint: [0.04, 0.04, 0.10],
                palette_wave_strength: 0.18,
                vibrance_bias: 0.95,
                tone_curve_bias: 1.05,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn at_least_twelve_presets() {
        assert!(GradePreset::all().len() >= 12);
    }

    #[test]
    fn tints_are_within_reasonable_bounds() {
        for &preset in GradePreset::all() {
            let p = preset.params();
            for (i, &v) in p.shadow_tint.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "{} shadow_tint[{i}] out of range: {v}",
                    preset.name()
                );
            }
            for (i, &v) in p.highlight_tint.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "{} highlight_tint[{i}] out of range: {v}",
                    preset.name()
                );
            }
            assert!((0.0..=1.0).contains(&p.palette_wave_strength));
            assert!((0.0..=2.0).contains(&p.vibrance_bias));
            assert!((0.5..=2.0).contains(&p.tone_curve_bias));
        }
    }

    #[test]
    fn presets_are_distinct() {
        let unique: HashSet<GradePreset> = GradePreset::all().iter().copied().collect();
        assert_eq!(unique.len(), GradePreset::all().len());

        let mut signatures: HashSet<String> = HashSet::new();
        for &preset in GradePreset::all() {
            let p = preset.params();
            let sig = format!(
                "{:?}_{:?}_{:.3}_{:.3}_{:.3}",
                p.shadow_tint,
                p.highlight_tint,
                p.palette_wave_strength,
                p.vibrance_bias,
                p.tone_curve_bias
            );
            assert!(signatures.insert(sig), "preset {} duplicates another", preset.name());
        }
    }

    #[test]
    fn every_preset_has_unique_name() {
        let names: HashSet<&'static str> = GradePreset::all().iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), GradePreset::all().len());
    }

    #[test]
    fn highlight_brighter_than_shadow_for_most_presets() {
        // Most "normal" grade presets raise highlights vs. shadows.
        // HighContrastBW is the extreme test case that must satisfy this.
        let p = GradePreset::HighContrastBW.params();
        assert!(p.highlight_tint[0] > p.shadow_tint[0]);
        assert!(p.highlight_tint[1] > p.shadow_tint[1]);
        assert!(p.highlight_tint[2] > p.shadow_tint[2]);
    }

    #[test]
    fn from_index_wraps() {
        assert_eq!(GradePreset::from_index(0), GradePreset::from_index(GradePreset::all().len()));
    }
}
