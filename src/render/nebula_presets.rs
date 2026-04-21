//! Curated nebula background palettes.
//!
//! Each [`NebulaPalette`] variant is a cohesive 4-color cosmic background
//! paired with curated noise parameters. The previous pipeline hard-coded a
//! single dark palette and kept `NEBULA_STRENGTH` pinned to zero, which made
//! every image a trajectory-on-black composition. These presets unlock real
//! atmospheric variety while remaining tasteful and museum-worthy.
//!
//! All RGB triplets are straight-alpha linear-sRGB values. They are designed
//! to sit BEHIND the trajectory and lift the background out of pure black
//! without ever competing with the subject.

use serde::{Deserialize, Serialize};

/// Number of color stops in every nebula palette.
pub const NEBULA_PALETTE_STOPS: usize = 4;

/// A single curated nebula palette + noise parameter preset.
///
/// These are intentionally named after the mood they evoke so artistic
/// intent stays legible in the generation log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NebulaPalette {
    /// Deep purple / teal / magenta / indigo - the previous default.
    #[default]
    DeepSpace,
    /// Emerald / electric violet / ice blue / magenta - polar light show.
    Aurora,
    /// Ember / flame / solar gold / deep red - close to a star.
    SolarFire,
    /// Jade / seafoam / moss / teal - underwater reef at depth.
    EmeraldReef,
    /// Rose / blush / mauve / plum - soft nebular nursery.
    RoseNebula,
    /// Brass / amber / bronze / dark gold - molten metal river.
    MoltenGold,
    /// Platinum / pale sky / frost blue / icy lilac - Arctic silence.
    ArcticCrystal,
    /// Royal purple / violet / lavender / silver - crown jewel.
    Amethyst,
    /// Abyssal indigo / midnight / phosphor teal / cyan - deep ocean.
    AbyssalOcean,
    /// Neutral grey scale - baroque chiaroscuro ground.
    Monochrome,
    /// Warm ivory / sepia / tan / coffee - vintage print.
    Sepia,
    /// Hot pink / electric cyan / neon violet / black - Tokyo street.
    TokyoNeon,
}

/// Static list of every nebula palette variant. Used for enumeration and tests.
pub const ALL_NEBULA_PALETTES: &[NebulaPalette] = &[
    NebulaPalette::DeepSpace,
    NebulaPalette::Aurora,
    NebulaPalette::SolarFire,
    NebulaPalette::EmeraldReef,
    NebulaPalette::RoseNebula,
    NebulaPalette::MoltenGold,
    NebulaPalette::ArcticCrystal,
    NebulaPalette::Amethyst,
    NebulaPalette::AbyssalOcean,
    NebulaPalette::Monochrome,
    NebulaPalette::Sepia,
    NebulaPalette::TokyoNeon,
];

/// Resolved nebula noise + color parameters ready for `NebulaCloudConfig`.
#[derive(Clone, Debug)]
pub struct NebulaPreset {
    /// Four linear-sRGB color stops cycled by the opensimplex noise.
    pub colors: [[f64; 3]; NEBULA_PALETTE_STOPS],
    /// Persistence (amplitude reduction per octave).
    pub persistence: f64,
    /// Lacunarity (frequency multiplier per octave).
    pub lacunarity: f64,
    /// Time-scale factor for video motion.
    pub time_scale: f64,
    /// Edge-fade (radial vignette) factor.
    pub edge_fade: f64,
}

impl NebulaPalette {
    /// Every variant in declaration order.
    #[must_use]
    pub fn all() -> &'static [NebulaPalette] {
        ALL_NEBULA_PALETTES
    }

    /// Select a variant by index using modulo arithmetic.
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        ALL_NEBULA_PALETTES[index % ALL_NEBULA_PALETTES.len()]
    }

    /// Machine-readable name suitable for logs.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            NebulaPalette::DeepSpace => "deep_space",
            NebulaPalette::Aurora => "aurora",
            NebulaPalette::SolarFire => "solar_fire",
            NebulaPalette::EmeraldReef => "emerald_reef",
            NebulaPalette::RoseNebula => "rose_nebula",
            NebulaPalette::MoltenGold => "molten_gold",
            NebulaPalette::ArcticCrystal => "arctic_crystal",
            NebulaPalette::Amethyst => "amethyst",
            NebulaPalette::AbyssalOcean => "abyssal_ocean",
            NebulaPalette::Monochrome => "monochrome",
            NebulaPalette::Sepia => "sepia",
            NebulaPalette::TokyoNeon => "tokyo_neon",
        }
    }

    /// Resolve this palette into a concrete preset.
    ///
    /// All colors are kept deliberately dim (<= ~0.30 per channel) so the
    /// trajectory stays the dominant subject and highlight headroom is
    /// preserved for the tonemapper.
    #[must_use]
    pub fn preset(self) -> NebulaPreset {
        match self {
            NebulaPalette::DeepSpace => NebulaPreset {
                colors: [
                    [0.08, 0.10, 0.22],
                    [0.15, 0.08, 0.26],
                    [0.22, 0.10, 0.18],
                    [0.10, 0.14, 0.28],
                ],
                persistence: 0.5,
                lacunarity: 2.0,
                time_scale: 1.0,
                edge_fade: 0.3,
            },
            NebulaPalette::Aurora => NebulaPreset {
                colors: [
                    [0.04, 0.22, 0.18],
                    [0.05, 0.28, 0.12],
                    [0.18, 0.08, 0.26],
                    [0.06, 0.18, 0.24],
                ],
                persistence: 0.55,
                lacunarity: 2.1,
                time_scale: 0.8,
                edge_fade: 0.32,
            },
            NebulaPalette::SolarFire => NebulaPreset {
                colors: [
                    [0.28, 0.08, 0.03],
                    [0.28, 0.16, 0.04],
                    [0.30, 0.22, 0.06],
                    [0.18, 0.04, 0.02],
                ],
                persistence: 0.52,
                lacunarity: 2.05,
                time_scale: 1.1,
                edge_fade: 0.35,
            },
            NebulaPalette::EmeraldReef => NebulaPreset {
                colors: [
                    [0.04, 0.20, 0.14],
                    [0.08, 0.22, 0.18],
                    [0.02, 0.14, 0.18],
                    [0.05, 0.22, 0.20],
                ],
                persistence: 0.5,
                lacunarity: 2.0,
                time_scale: 0.9,
                edge_fade: 0.30,
            },
            NebulaPalette::RoseNebula => NebulaPreset {
                colors: [
                    [0.24, 0.10, 0.18],
                    [0.26, 0.14, 0.22],
                    [0.18, 0.06, 0.16],
                    [0.22, 0.12, 0.14],
                ],
                persistence: 0.48,
                lacunarity: 2.05,
                time_scale: 0.85,
                edge_fade: 0.33,
            },
            NebulaPalette::MoltenGold => NebulaPreset {
                colors: [
                    [0.22, 0.14, 0.04],
                    [0.26, 0.18, 0.06],
                    [0.30, 0.22, 0.10],
                    [0.14, 0.08, 0.02],
                ],
                persistence: 0.52,
                lacunarity: 2.1,
                time_scale: 1.0,
                edge_fade: 0.32,
            },
            NebulaPalette::ArcticCrystal => NebulaPreset {
                colors: [
                    [0.18, 0.22, 0.26],
                    [0.14, 0.20, 0.24],
                    [0.10, 0.16, 0.22],
                    [0.16, 0.18, 0.22],
                ],
                persistence: 0.45,
                lacunarity: 2.0,
                time_scale: 0.7,
                edge_fade: 0.38,
            },
            NebulaPalette::Amethyst => NebulaPreset {
                colors: [
                    [0.14, 0.06, 0.26],
                    [0.18, 0.10, 0.28],
                    [0.22, 0.16, 0.30],
                    [0.10, 0.04, 0.20],
                ],
                persistence: 0.5,
                lacunarity: 2.05,
                time_scale: 0.95,
                edge_fade: 0.32,
            },
            NebulaPalette::AbyssalOcean => NebulaPreset {
                colors: [
                    [0.02, 0.08, 0.20],
                    [0.04, 0.14, 0.24],
                    [0.02, 0.18, 0.22],
                    [0.04, 0.06, 0.16],
                ],
                persistence: 0.48,
                lacunarity: 2.0,
                time_scale: 0.8,
                edge_fade: 0.34,
            },
            NebulaPalette::Monochrome => NebulaPreset {
                colors: [
                    [0.10, 0.10, 0.10],
                    [0.14, 0.14, 0.14],
                    [0.18, 0.18, 0.18],
                    [0.06, 0.06, 0.06],
                ],
                persistence: 0.5,
                lacunarity: 2.0,
                time_scale: 0.9,
                edge_fade: 0.36,
            },
            NebulaPalette::Sepia => NebulaPreset {
                colors: [
                    [0.22, 0.16, 0.10],
                    [0.26, 0.20, 0.14],
                    [0.18, 0.12, 0.08],
                    [0.14, 0.10, 0.06],
                ],
                persistence: 0.5,
                lacunarity: 2.0,
                time_scale: 0.9,
                edge_fade: 0.34,
            },
            NebulaPalette::TokyoNeon => NebulaPreset {
                colors: [
                    [0.24, 0.04, 0.18],
                    [0.04, 0.22, 0.26],
                    [0.18, 0.04, 0.26],
                    [0.02, 0.02, 0.08],
                ],
                persistence: 0.55,
                lacunarity: 2.15,
                time_scale: 1.05,
                edge_fade: 0.28,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn there_are_at_least_twelve_palettes() {
        assert!(
            NebulaPalette::all().len() >= 12,
            "expected >= 12 nebula palettes, got {}",
            NebulaPalette::all().len()
        );
    }

    #[test]
    fn every_palette_has_unique_name() {
        let names: HashSet<&'static str> = NebulaPalette::all().iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), NebulaPalette::all().len(), "duplicate palette names");
    }

    #[test]
    fn all_colors_are_in_unit_range() {
        for &palette in NebulaPalette::all() {
            let preset = palette.preset();
            for (i, stop) in preset.colors.iter().enumerate() {
                for (c, &channel) in stop.iter().enumerate() {
                    assert!(
                        (0.0..=1.0).contains(&channel),
                        "palette {} stop {i} channel {c} out of range: {channel}",
                        palette.name()
                    );
                }
            }
        }
    }

    #[test]
    fn every_palette_stays_dim_enough_for_subject() {
        // Nebula must never exceed 0.35 per channel, else it overwhelms the
        // trajectory on the way through the tonemapper.
        for &palette in NebulaPalette::all() {
            let preset = palette.preset();
            for stop in &preset.colors {
                for &channel in stop {
                    assert!(
                        channel <= 0.35,
                        "palette {} has channel {channel} > 0.35 (too bright)",
                        palette.name()
                    );
                }
            }
        }
    }

    #[test]
    fn noise_params_are_in_sane_ranges() {
        for &palette in NebulaPalette::all() {
            let preset = palette.preset();
            assert!((0.30..=0.70).contains(&preset.persistence), "{}", palette.name());
            assert!((1.50..=2.50).contains(&preset.lacunarity), "{}", palette.name());
            assert!((0.5..=1.2).contains(&preset.time_scale), "{}", palette.name());
            assert!((0.2..=0.45).contains(&preset.edge_fade), "{}", palette.name());
        }
    }

    #[test]
    fn every_palette_has_some_chromatic_variety() {
        // Palettes (even Monochrome) must show non-trivial variance between stops.
        for &palette in NebulaPalette::all() {
            let preset = palette.preset();
            let (mut min_sum, mut max_sum) = (f64::INFINITY, f64::NEG_INFINITY);
            for stop in &preset.colors {
                let sum = stop.iter().sum::<f64>();
                min_sum = min_sum.min(sum);
                max_sum = max_sum.max(sum);
            }
            assert!(
                max_sum - min_sum > 0.02,
                "palette {} has no per-stop variance ({min_sum} vs {max_sum})",
                palette.name()
            );
        }
    }

    #[test]
    fn every_nonmonochrome_palette_is_actually_chromatic() {
        for &palette in NebulaPalette::all() {
            if matches!(palette, NebulaPalette::Monochrome) {
                continue;
            }
            let preset = palette.preset();
            let mut seen_chromatic_stop = false;
            for stop in &preset.colors {
                let min = stop.iter().copied().fold(f64::INFINITY, f64::min);
                let max = stop.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if max - min > 0.04 {
                    seen_chromatic_stop = true;
                    break;
                }
            }
            assert!(seen_chromatic_stop, "palette {} has no chromatic stops", palette.name());
        }
    }

    #[test]
    fn from_index_wraps_and_is_deterministic() {
        let first = NebulaPalette::from_index(0);
        let wrapped = NebulaPalette::from_index(NebulaPalette::all().len());
        assert_eq!(first, wrapped);
    }

    #[test]
    fn all_palettes_distinct() {
        let unique: HashSet<NebulaPalette> = NebulaPalette::all().iter().copied().collect();
        assert_eq!(unique.len(), NebulaPalette::all().len());
    }
}
