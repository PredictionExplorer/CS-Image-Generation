//! Curated cinematic color-grade presets.
//!
//! These presets provide a compact library of coherent looks that can be
//! combined with composition and finish coordination without resorting to a
//! heavyweight style meta-system.

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
    /// Deep blue shadows / pale highlights for nocturnal imagery.
    NightSky,
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
    /// Soft pastel dream with gentle tints.
    PastelDream,
}

/// Static list of every grade preset variant.
pub const ALL_GRADE_PRESETS: &[GradePreset] = &[
    GradePreset::CinematicTeal,
    GradePreset::NoirContrast,
    GradePreset::GoldenHour,
    GradePreset::NordicCool,
    GradePreset::NightSky,
    GradePreset::ArtNouveauEarth,
    GradePreset::EtherealSoft,
    GradePreset::RoyalAmethyst,
    GradePreset::WarmBrass,
    GradePreset::IcyPlatinum,
    GradePreset::VintageSepia,
    GradePreset::PastelDream,
];

/// Concrete grade-preset values used to drive the cinematic color grade.
#[derive(Clone, Debug)]
pub struct GradeParams {
    /// Linear-sRGB delta added to the shadows.
    pub shadow_tint: [f64; 3],
    /// Linear-sRGB delta added to the highlights.
    pub highlight_tint: [f64; 3],
    /// Palette-wave accent strength.
    pub palette_wave_strength: f64,
    /// Multiplicative bias applied on top of the randomized vibrance.
    pub vibrance_bias: f64,
    /// Multiplicative bias applied on top of the randomized tone curve strength.
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

    /// Machine-readable name suitable for logs and tests.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GradePreset::CinematicTeal => "cinematic_teal",
            GradePreset::NoirContrast => "noir_contrast",
            GradePreset::GoldenHour => "golden_hour",
            GradePreset::NordicCool => "nordic_cool",
            GradePreset::NightSky => "night_sky",
            GradePreset::ArtNouveauEarth => "art_nouveau_earth",
            GradePreset::EtherealSoft => "ethereal_soft",
            GradePreset::RoyalAmethyst => "royal_amethyst",
            GradePreset::WarmBrass => "warm_brass",
            GradePreset::IcyPlatinum => "icy_platinum",
            GradePreset::VintageSepia => "vintage_sepia",
            GradePreset::PastelDream => "pastel_dream",
        }
    }

    /// Resolve this preset into concrete grade parameters.
    #[must_use]
    pub fn params(self) -> GradeParams {
        match self {
            GradePreset::CinematicTeal => GradeParams {
                shadow_tint: [-0.08, -0.02, 0.16],
                highlight_tint: [0.11, 0.05, -0.03],
                palette_wave_strength: 0.18,
                vibrance_bias: 1.0,
                tone_curve_bias: 1.0,
            },
            GradePreset::NoirContrast => GradeParams {
                shadow_tint: [-0.14, -0.12, -0.10],
                highlight_tint: [0.10, 0.10, 0.09],
                palette_wave_strength: 0.06,
                vibrance_bias: 0.55,
                tone_curve_bias: 1.22,
            },
            GradePreset::GoldenHour => GradeParams {
                shadow_tint: [0.04, -0.02, -0.10],
                highlight_tint: [0.18, 0.10, -0.02],
                palette_wave_strength: 0.18,
                vibrance_bias: 1.08,
                tone_curve_bias: 0.96,
            },
            GradePreset::NordicCool => GradeParams {
                shadow_tint: [-0.08, 0.02, 0.14],
                highlight_tint: [-0.02, 0.06, 0.12],
                palette_wave_strength: 0.10,
                vibrance_bias: 0.85,
                tone_curve_bias: 1.05,
            },
            GradePreset::NightSky => GradeParams {
                shadow_tint: [-0.04, -0.06, 0.14],
                highlight_tint: [0.04, 0.04, 0.10],
                palette_wave_strength: 0.10,
                vibrance_bias: 0.92,
                tone_curve_bias: 1.06,
            },
            GradePreset::ArtNouveauEarth => GradeParams {
                shadow_tint: [-0.02, 0.04, -0.06],
                highlight_tint: [0.14, 0.08, 0.02],
                palette_wave_strength: 0.20,
                vibrance_bias: 0.95,
                tone_curve_bias: 1.0,
            },
            GradePreset::EtherealSoft => GradeParams {
                shadow_tint: [0.02, -0.02, 0.10],
                highlight_tint: [0.10, 0.06, 0.02],
                palette_wave_strength: 0.22,
                vibrance_bias: 0.88,
                tone_curve_bias: 0.92,
            },
            GradePreset::RoyalAmethyst => GradeParams {
                shadow_tint: [0.08, -0.08, 0.16],
                highlight_tint: [0.14, 0.10, -0.02],
                palette_wave_strength: 0.18,
                vibrance_bias: 1.03,
                tone_curve_bias: 1.04,
            },
            GradePreset::WarmBrass => GradeParams {
                shadow_tint: [0.02, -0.04, -0.08],
                highlight_tint: [0.16, 0.11, 0.02],
                palette_wave_strength: 0.14,
                vibrance_bias: 1.03,
                tone_curve_bias: 1.0,
            },
            GradePreset::IcyPlatinum => GradeParams {
                shadow_tint: [-0.04, 0.00, 0.08],
                highlight_tint: [0.04, 0.08, 0.12],
                palette_wave_strength: 0.06,
                vibrance_bias: 0.72,
                tone_curve_bias: 1.04,
            },
            GradePreset::VintageSepia => GradeParams {
                shadow_tint: [0.04, 0.00, -0.10],
                highlight_tint: [0.14, 0.08, -0.04],
                palette_wave_strength: 0.08,
                vibrance_bias: 0.58,
                tone_curve_bias: 1.04,
            },
            GradePreset::PastelDream => GradeParams {
                shadow_tint: [0.04, 0.00, 0.06],
                highlight_tint: [0.06, 0.04, 0.02],
                palette_wave_strength: 0.20,
                vibrance_bias: 0.78,
                tone_curve_bias: 0.86,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn presets_are_distinct() {
        let unique: HashSet<GradePreset> = GradePreset::all().iter().copied().collect();
        assert_eq!(unique.len(), GradePreset::all().len());
    }

    #[test]
    fn preset_parameters_are_bounded() {
        for &preset in GradePreset::all() {
            let params = preset.params();
            for &value in params.shadow_tint.iter().chain(params.highlight_tint.iter()) {
                assert!((-1.0..=1.0).contains(&value));
            }
            assert!((0.0..=1.0).contains(&params.palette_wave_strength));
            assert!((0.0..=2.0).contains(&params.vibrance_bias));
            assert!((0.5..=2.0).contains(&params.tone_curve_bias));
        }
    }

    #[test]
    fn every_preset_has_unique_name() {
        let names: HashSet<&'static str> =
            GradePreset::all().iter().map(|preset| preset.name()).collect();
        assert_eq!(names.len(), GradePreset::all().len());
    }

    #[test]
    fn from_index_wraps() {
        assert_eq!(GradePreset::from_index(0), GradePreset::from_index(GradePreset::all().len()));
    }
}
