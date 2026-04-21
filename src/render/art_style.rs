//! Per-seed art-style meta randomizer.
//!
//! Instead of sampling every effect parameter independently (which tends
//! to regress toward a mean "teal/orange cinematic" look) we pick a single
//! cohesive [`ArtStyle`] per seed. Each variant is a fully-considered
//! aesthetic that bundles a nebula palette, a grade preset, a hue palette
//! mode, a drift character, a bloom mode, and optional emphasis knobs.
//!
//! The style is chosen early (with weighted probability) and then baked
//! into the effect config before per-effect randomization runs, which
//! ensures every image is a coherent composition.

use super::BloomMode;
use super::grade_presets::GradePreset;
use super::hue_palette::HuePaletteMode;
use super::nebula_presets::NebulaPalette;
use crate::sim::Sha3RandomByteStream;
use serde::{Deserialize, Serialize};

/// Drift character associated with an art style.
///
/// Distinct from the low-level [`crate::drift::DriftMode`] because it
/// encodes intent rather than a specific math-form choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriftCharacter {
    /// Pure gravitational orbit - no drift.
    #[default]
    None,
    /// Gentle linear translation.
    Linear,
    /// Jittery Brownian exploration.
    Brownian,
    /// Steady elliptical sweep.
    Elliptical,
    /// Circular sweep.
    Circular,
    /// Spiral inward/outward.
    Spiral,
}

impl DriftCharacter {
    /// Machine-readable name for logs.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            DriftCharacter::None => "none",
            DriftCharacter::Linear => "linear",
            DriftCharacter::Brownian => "brownian",
            DriftCharacter::Elliptical => "elliptical",
            DriftCharacter::Circular => "circular",
            DriftCharacter::Spiral => "spiral",
        }
    }
}

/// Optional emphasis flags that a style can turn on or off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StyleEmphasis {
    /// Force-enable champlevé accent.
    pub emphasize_champleve: bool,
    /// Force-enable opalescence heavily.
    pub emphasize_opalescence: bool,
    /// Force-enable fine texture strongly.
    pub emphasize_fine_texture: bool,
    /// Force-enable perceptual blur (otherwise probabilistic).
    pub force_perceptual_blur: bool,
    /// Force-enable starfield effect (when wired).
    pub emphasize_starfield: bool,
    /// Force-enable lens flare (when wired).
    pub emphasize_lens_flare: bool,
    /// Pump vignette heavily.
    pub heavy_vignette: bool,
    /// Boost dispersion / chromatic spread.
    pub boost_dispersion: bool,
}

/// A curated art style - one of the cohesive aesthetics the pipeline can produce.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtStyle {
    /// Classic trajectory-on-black cosmos.
    #[default]
    DeepCosmos,
    /// Polar aurora color fields.
    AuroraBorealis,
    /// Blazing solar furnace.
    SolarFurnace,
    /// Deep abyssal ocean blues.
    OceanicAbyss,
    /// Soft rose-nebula nursery.
    RoseNebula,
    /// Mucha-inspired art nouveau.
    ArtNouveau,
    /// High-contrast baroque chiaroscuro.
    BaroqueChiaroscuro,
    /// Neo-noir Tokyo neon.
    TokyoNeon,
    /// Silent arctic crystal.
    ArcticSilence,
    /// Molten metal gold.
    MoltenGold,
    /// Jewel-tone royal velvet.
    RoyalVelvet,
    /// Vintage desert mirage / sepia print.
    DesertMirage,
    /// Emerald city: fantasy teal.
    EmeraldCity,
    /// Soft porcelain pastel.
    PorcelainPastel,
    /// Iridescent prism: highly saturated.
    IridescentPrism,
    /// Pure monochrome.
    Monochrome,
    /// Two-tone duotone.
    Duotone,
    /// Soft, diffused cosmic ethereal.
    CosmicEthereal,
}

/// Static list of every art style variant in declaration order.
pub const ALL_ART_STYLES: &[ArtStyle] = &[
    ArtStyle::DeepCosmos,
    ArtStyle::AuroraBorealis,
    ArtStyle::SolarFurnace,
    ArtStyle::OceanicAbyss,
    ArtStyle::RoseNebula,
    ArtStyle::ArtNouveau,
    ArtStyle::BaroqueChiaroscuro,
    ArtStyle::TokyoNeon,
    ArtStyle::ArcticSilence,
    ArtStyle::MoltenGold,
    ArtStyle::RoyalVelvet,
    ArtStyle::DesertMirage,
    ArtStyle::EmeraldCity,
    ArtStyle::PorcelainPastel,
    ArtStyle::IridescentPrism,
    ArtStyle::Monochrome,
    ArtStyle::Duotone,
    ArtStyle::CosmicEthereal,
];

/// Resolved aesthetic bundle chosen per-seed.
#[derive(Clone, Debug)]
pub struct StyleBundle {
    /// Chosen style.
    pub style: ArtStyle,
    /// Matching nebula palette.
    pub nebula: NebulaPalette,
    /// Matching color-grade preset.
    pub grade: GradePreset,
    /// Matching hue-palette mode.
    pub hue_mode: HuePaletteMode,
    /// Drift character.
    pub drift: DriftCharacter,
    /// Bloom mode.
    pub bloom: BloomMode,
    /// Optional emphasis knobs.
    pub emphasis: StyleEmphasis,
    /// Nebula strength multiplier (0..=1.2) applied on top of the randomized base.
    pub nebula_strength_bias: f64,
    /// Multiplier applied to HDR scale after resolve.
    pub hdr_scale_bias: f64,
}

impl ArtStyle {
    /// Every variant in declaration order.
    #[must_use]
    pub fn all() -> &'static [ArtStyle] {
        ALL_ART_STYLES
    }

    /// Machine-readable name for logs and schema.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            ArtStyle::DeepCosmos => "deep_cosmos",
            ArtStyle::AuroraBorealis => "aurora_borealis",
            ArtStyle::SolarFurnace => "solar_furnace",
            ArtStyle::OceanicAbyss => "oceanic_abyss",
            ArtStyle::RoseNebula => "rose_nebula",
            ArtStyle::ArtNouveau => "art_nouveau",
            ArtStyle::BaroqueChiaroscuro => "baroque_chiaroscuro",
            ArtStyle::TokyoNeon => "tokyo_neon",
            ArtStyle::ArcticSilence => "arctic_silence",
            ArtStyle::MoltenGold => "molten_gold",
            ArtStyle::RoyalVelvet => "royal_velvet",
            ArtStyle::DesertMirage => "desert_mirage",
            ArtStyle::EmeraldCity => "emerald_city",
            ArtStyle::PorcelainPastel => "porcelain_pastel",
            ArtStyle::IridescentPrism => "iridescent_prism",
            ArtStyle::Monochrome => "monochrome",
            ArtStyle::Duotone => "duotone",
            ArtStyle::CosmicEthereal => "cosmic_ethereal",
        }
    }

    /// Pick a style from the RNG with a curated near-uniform distribution.
    ///
    /// Weights are intentionally flat: no single style exceeds 8/130 of the
    /// probability mass. The Shannon entropy over 1024+ seeds stays
    /// comfortably above 3.8 bits, and no legacy style dominates the
    /// aesthetic sample. Every style is already curated for museum-tier
    /// output, so flatter weights translate directly into higher variety
    /// without any quality trade-off.
    #[must_use]
    pub fn pick(rng: &mut Sha3RandomByteStream) -> ArtStyle {
        const WEIGHTS: [(ArtStyle, u32); 18] = [
            (ArtStyle::DeepCosmos, 8),
            (ArtStyle::AuroraBorealis, 8),
            (ArtStyle::SolarFurnace, 8),
            (ArtStyle::OceanicAbyss, 8),
            (ArtStyle::RoseNebula, 8),
            (ArtStyle::ArtNouveau, 7),
            (ArtStyle::BaroqueChiaroscuro, 7),
            (ArtStyle::TokyoNeon, 8),
            (ArtStyle::ArcticSilence, 7),
            (ArtStyle::MoltenGold, 8),
            (ArtStyle::RoyalVelvet, 7),
            (ArtStyle::DesertMirage, 7),
            (ArtStyle::EmeraldCity, 8),
            (ArtStyle::PorcelainPastel, 7),
            (ArtStyle::IridescentPrism, 7),
            (ArtStyle::Monochrome, 7),
            (ArtStyle::Duotone, 7),
            (ArtStyle::CosmicEthereal, 8),
        ];
        let total: u32 = WEIGHTS.iter().map(|(_, w)| *w).sum();
        let b0 = u32::from(rng.next_byte());
        let b1 = u32::from(rng.next_byte());
        let b2 = u32::from(rng.next_byte());
        let b3 = u32::from(rng.next_byte());
        let bits = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
        let t = (f64::from(bits) / f64::from(u32::MAX)) * f64::from(total);
        let mut cumulative = 0u32;
        for (style, weight) in WEIGHTS.iter().copied() {
            cumulative = cumulative.saturating_add(weight);
            if t < f64::from(cumulative) {
                return style;
            }
        }
        ArtStyle::DeepCosmos
    }

    /// Resolve the fully-curated style bundle.
    #[must_use]
    pub fn bundle(self) -> StyleBundle {
        match self {
            ArtStyle::DeepCosmos => StyleBundle {
                style: self,
                nebula: NebulaPalette::DeepSpace,
                grade: GradePreset::NightSky,
                hue_mode: HuePaletteMode::Triadic,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis { emphasize_starfield: true, ..Default::default() },
                nebula_strength_bias: 0.9,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::AuroraBorealis => StyleBundle {
                style: self,
                nebula: NebulaPalette::Aurora,
                grade: GradePreset::NordicCool,
                hue_mode: HuePaletteMode::Complementary,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 1.15,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::SolarFurnace => StyleBundle {
                style: self,
                nebula: NebulaPalette::SolarFire,
                grade: GradePreset::GoldenHour,
                hue_mode: HuePaletteMode::Analogous,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis { emphasize_lens_flare: true, ..Default::default() },
                nebula_strength_bias: 1.1,
                hdr_scale_bias: 1.05,
            },
            ArtStyle::OceanicAbyss => StyleBundle {
                style: self,
                nebula: NebulaPalette::AbyssalOcean,
                grade: GradePreset::UnderwaterTeal,
                hue_mode: HuePaletteMode::Monochromatic,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 1.0,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::RoseNebula => StyleBundle {
                style: self,
                nebula: NebulaPalette::RoseNebula,
                grade: GradePreset::PastelDream,
                hue_mode: HuePaletteMode::SplitComplementary,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 1.1,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::ArtNouveau => StyleBundle {
                style: self,
                nebula: NebulaPalette::EmeraldReef,
                grade: GradePreset::ArtNouveauEarth,
                hue_mode: HuePaletteMode::TetradicSquare,
                drift: DriftCharacter::Circular,
                bloom: BloomMode::None,
                emphasis: StyleEmphasis { emphasize_champleve: true, ..Default::default() },
                nebula_strength_bias: 0.95,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::BaroqueChiaroscuro => StyleBundle {
                style: self,
                nebula: NebulaPalette::Monochrome,
                grade: GradePreset::NoirContrast,
                hue_mode: HuePaletteMode::Duotone,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis { heavy_vignette: true, ..Default::default() },
                nebula_strength_bias: 0.7,
                hdr_scale_bias: 1.1,
            },
            ArtStyle::TokyoNeon => StyleBundle {
                style: self,
                nebula: NebulaPalette::TokyoNeon,
                grade: GradePreset::NeonNoir,
                hue_mode: HuePaletteMode::Complementary,
                drift: DriftCharacter::Linear,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis {
                    emphasize_lens_flare: true,
                    boost_dispersion: true,
                    ..Default::default()
                },
                nebula_strength_bias: 1.15,
                hdr_scale_bias: 1.1,
            },
            ArtStyle::ArcticSilence => StyleBundle {
                style: self,
                nebula: NebulaPalette::ArcticCrystal,
                grade: GradePreset::IcyPlatinum,
                hue_mode: HuePaletteMode::Monochromatic,
                drift: DriftCharacter::None,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 0.85,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::MoltenGold => StyleBundle {
                style: self,
                nebula: NebulaPalette::MoltenGold,
                grade: GradePreset::WarmBrass,
                hue_mode: HuePaletteMode::Analogous,
                drift: DriftCharacter::Spiral,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis {
                    emphasize_lens_flare: true,
                    emphasize_fine_texture: true,
                    ..Default::default()
                },
                nebula_strength_bias: 1.05,
                hdr_scale_bias: 1.05,
            },
            ArtStyle::RoyalVelvet => StyleBundle {
                style: self,
                nebula: NebulaPalette::Amethyst,
                grade: GradePreset::RoyalAmethyst,
                hue_mode: HuePaletteMode::Complementary,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis { emphasize_champleve: true, ..Default::default() },
                nebula_strength_bias: 1.0,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::DesertMirage => StyleBundle {
                style: self,
                nebula: NebulaPalette::Sepia,
                grade: GradePreset::VintageSepia,
                hue_mode: HuePaletteMode::Analogous,
                drift: DriftCharacter::Linear,
                bloom: BloomMode::None,
                emphasis: StyleEmphasis { emphasize_fine_texture: true, ..Default::default() },
                nebula_strength_bias: 0.95,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::EmeraldCity => StyleBundle {
                style: self,
                nebula: NebulaPalette::EmeraldReef,
                grade: GradePreset::CinematicTeal,
                hue_mode: HuePaletteMode::Triadic,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 1.05,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::PorcelainPastel => StyleBundle {
                style: self,
                nebula: NebulaPalette::RoseNebula,
                grade: GradePreset::PastelDream,
                hue_mode: HuePaletteMode::Analogous,
                drift: DriftCharacter::Circular,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 1.0,
                hdr_scale_bias: 0.95,
            },
            ArtStyle::IridescentPrism => StyleBundle {
                style: self,
                nebula: NebulaPalette::DeepSpace,
                grade: GradePreset::SaturatedRainbow,
                hue_mode: HuePaletteMode::TetradicSquare,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis {
                    emphasize_opalescence: true,
                    boost_dispersion: true,
                    ..Default::default()
                },
                nebula_strength_bias: 0.85,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::Monochrome => StyleBundle {
                style: self,
                nebula: NebulaPalette::Monochrome,
                grade: GradePreset::HighContrastBW,
                hue_mode: HuePaletteMode::Monochromatic,
                drift: DriftCharacter::Elliptical,
                bloom: BloomMode::Dog,
                emphasis: StyleEmphasis {
                    emphasize_fine_texture: true,
                    heavy_vignette: true,
                    ..Default::default()
                },
                nebula_strength_bias: 0.85,
                hdr_scale_bias: 1.05,
            },
            ArtStyle::Duotone => StyleBundle {
                style: self,
                nebula: NebulaPalette::DeepSpace,
                grade: GradePreset::NeonNoir,
                hue_mode: HuePaletteMode::Duotone,
                drift: DriftCharacter::Linear,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis::default(),
                nebula_strength_bias: 0.95,
                hdr_scale_bias: 1.0,
            },
            ArtStyle::CosmicEthereal => StyleBundle {
                style: self,
                nebula: NebulaPalette::Aurora,
                grade: GradePreset::EtherealSoft,
                hue_mode: HuePaletteMode::CosmicWarmCool,
                drift: DriftCharacter::Brownian,
                bloom: BloomMode::Gaussian,
                emphasis: StyleEmphasis { force_perceptual_blur: true, ..Default::default() },
                nebula_strength_bias: 1.10,
                hdr_scale_bias: 0.95,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn make_rng(seed_byte: u8) -> Sha3RandomByteStream {
        let seed = vec![seed_byte, seed_byte.wrapping_add(17), seed_byte.wrapping_add(29)];
        Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0)
    }

    #[test]
    fn exactly_eighteen_styles() {
        assert_eq!(ArtStyle::all().len(), 18);
    }

    #[test]
    fn every_style_has_unique_name() {
        let names: HashSet<&'static str> = ArtStyle::all().iter().map(|s| s.name()).collect();
        assert_eq!(names.len(), ArtStyle::all().len());
    }

    #[test]
    fn every_bundle_is_self_consistent() {
        for &style in ArtStyle::all() {
            let bundle = style.bundle();
            assert_eq!(bundle.style, style);
            assert!(
                (0.5..=1.5).contains(&bundle.nebula_strength_bias),
                "{} nebula_bias out of range",
                style.name()
            );
            assert!(
                (0.8..=1.25).contains(&bundle.hdr_scale_bias),
                "{} hdr_bias out of range",
                style.name()
            );
        }
    }

    #[test]
    fn pick_returns_valid_variant() {
        for seed in 0..32u8 {
            let mut rng = make_rng(seed);
            let style = ArtStyle::pick(&mut rng);
            assert!(ArtStyle::all().contains(&style));
        }
    }

    #[test]
    fn pick_is_deterministic_per_seed() {
        for seed in 0..8u8 {
            let mut r1 = make_rng(seed);
            let mut r2 = make_rng(seed);
            assert_eq!(ArtStyle::pick(&mut r1), ArtStyle::pick(&mut r2));
        }
    }

    #[test]
    fn pick_distribution_has_high_entropy() {
        // Over many seeds, Shannon entropy of picked styles must exceed 3.8 bits
        // (18 variants => log2(18) ~ 4.17 maximum).
        let mut counts: HashMap<ArtStyle, u32> = HashMap::new();
        let total = 1024u32;
        for seed in 0..total {
            let bytes = seed.to_le_bytes();
            let mut rng = Sha3RandomByteStream::new(&bytes, 100.0, 300.0, 300.0, 1.0);
            let style = ArtStyle::pick(&mut rng);
            *counts.entry(style).or_insert(0) += 1;
        }
        assert!(counts.len() >= 12, "only {} distinct styles over {total} seeds", counts.len());
        let n = f64::from(total);
        let mut entropy = 0.0f64;
        for &c in counts.values() {
            if c > 0 {
                let p = f64::from(c) / n;
                entropy -= p * p.log2();
            }
        }
        assert!(entropy >= 3.8, "entropy too low: {entropy}");
    }

    #[test]
    fn drift_character_names_unique() {
        let characters = [
            DriftCharacter::None,
            DriftCharacter::Linear,
            DriftCharacter::Brownian,
            DriftCharacter::Elliptical,
            DriftCharacter::Circular,
            DriftCharacter::Spiral,
        ];
        let names: HashSet<&'static str> = characters.iter().map(|d| d.name()).collect();
        assert_eq!(names.len(), characters.len());
    }

    #[test]
    fn coverage_is_broad_across_nebula_palettes() {
        let nebulas: HashSet<NebulaPalette> =
            ArtStyle::all().iter().map(|s| s.bundle().nebula).collect();
        assert!(nebulas.len() >= 10, "only {} distinct nebulas", nebulas.len());
    }

    #[test]
    fn coverage_is_broad_across_grade_presets() {
        let grades: HashSet<GradePreset> =
            ArtStyle::all().iter().map(|s| s.bundle().grade).collect();
        assert!(grades.len() >= 12, "only {} distinct grades", grades.len());
    }

    #[test]
    fn coverage_covers_all_hue_modes() {
        let modes: HashSet<HuePaletteMode> =
            ArtStyle::all().iter().map(|s| s.bundle().hue_mode).collect();
        assert_eq!(modes.len(), HuePaletteMode::all().len(), "missing modes: {modes:?}");
    }
}
