//! Compact aesthetic coordination for composition, finish, and grading.
//!
//! This module intentionally replaces the old heavyweight style graph with a
//! small number of high-signal artistic axes. Each seed chooses one
//! [`CompositionMode`], one [`FinishCluster`], and one [`GradeFamily`], then
//! derives a coherent set of framing, finish, and grading biases from them.

use super::effect_randomizer::EffectRandomizer;
use super::grade_presets::GradePreset;
use serde::{Deserialize, Serialize};

/// Top-level composition families for framing and motion character.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositionMode {
    /// Stable centered framing with minimal drift.
    #[default]
    CenteredMonument,
    /// Wider stage with breathing room around the subject.
    WideStage,
    /// Off-center composition with deliberate negative space.
    OffAxisTension,
    /// Diagonal energy with stronger motion bias.
    DiagonalSweep,
    /// Tighter, quieter compositions.
    IntimateFocus,
}

impl CompositionMode {
    /// Machine-readable identifier for tests and logging.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CompositionMode::CenteredMonument => "centered_monument",
            CompositionMode::WideStage => "wide_stage",
            CompositionMode::OffAxisTension => "off_axis_tension",
            CompositionMode::DiagonalSweep => "diagonal_sweep",
            CompositionMode::IntimateFocus => "intimate_focus",
        }
    }
}

/// Correlated finish families controlling softness vs detail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinishCluster {
    /// Cleanest and most detail-forward.
    #[default]
    Crisp,
    /// Balanced and versatile.
    Balanced,
    /// Rare softer, velvety finish.
    Velvet,
}

impl FinishCluster {
    /// Machine-readable identifier for tests and logging.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            FinishCluster::Crisp => "crisp",
            FinishCluster::Balanced => "balanced",
            FinishCluster::Velvet => "velvet",
        }
    }
}

/// High-level grade families used to keep palettes and grading coherent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GradeFamily {
    /// Grounded, cinematic grading.
    #[default]
    Filmic,
    /// Jewel-tone, luxurious grading.
    Jewel,
    /// Graphic, print-like grading.
    Graphic,
    /// Dreamier atmospheric grading.
    Atmospheric,
}

impl GradeFamily {
    /// Machine-readable identifier for tests and logging.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GradeFamily::Filmic => "filmic",
            GradeFamily::Jewel => "jewel",
            GradeFamily::Graphic => "graphic",
            GradeFamily::Atmospheric => "atmospheric",
        }
    }

    #[must_use]
    fn grade_candidates(self) -> &'static [GradePreset] {
        match self {
            GradeFamily::Filmic => &[
                GradePreset::NoirContrast,
                GradePreset::GoldenHour,
                GradePreset::NordicCool,
                GradePreset::NightSky,
            ],
            GradeFamily::Jewel => &[
                GradePreset::RoyalAmethyst,
                GradePreset::WarmBrass,
                GradePreset::IcyPlatinum,
                GradePreset::EtherealSoft,
            ],
            GradeFamily::Graphic => &[
                GradePreset::ArtNouveauEarth,
                GradePreset::VintageSepia,
                GradePreset::NoirContrast,
                GradePreset::IcyPlatinum,
            ],
            GradeFamily::Atmospheric => &[
                GradePreset::CinematicTeal,
                GradePreset::NightSky,
                GradePreset::GoldenHour,
                GradePreset::PastelDream,
            ],
        }
    }

    #[must_use]
    fn gradient_palette_candidates(self) -> &'static [usize] {
        match self {
            GradeFamily::Filmic => &[2, 3, 4, 5],
            GradeFamily::Jewel => &[8, 12, 13, 14],
            GradeFamily::Graphic => &[5, 6, 7],
            GradeFamily::Atmospheric => &[1, 8, 10, 11],
        }
    }
}

/// Probabilities for enabling coordinated effects.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EnableProbabilities {
    pub(crate) bloom: f64,
    pub(crate) glow: f64,
    pub(crate) chromatic_bloom: f64,
    pub(crate) perceptual_blur: f64,
    pub(crate) micro_contrast: f64,
    pub(crate) gradient_map: f64,
    pub(crate) color_grade: f64,
    pub(crate) champleve: f64,
    pub(crate) aether: f64,
    pub(crate) opalescence: f64,
    pub(crate) edge_luminance: f64,
    pub(crate) atmospheric_depth: f64,
    pub(crate) fine_texture: f64,
}

/// Curated surface treatment for fine texture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfacePreset {
    pub(crate) fine_texture_anisotropy: f64,
    pub(crate) fine_texture_angle: f64,
}

/// Curated material treatment for layered finish effects.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaterialPreset {
    pub(crate) opalescence_chromatic_shift: f64,
    pub(crate) opalescence_pearl_sheen: f64,
    pub(crate) champleve_cell_density: f64,
    pub(crate) champleve_rim_sharpness: f64,
    pub(crate) aether_filament_density: f64,
    pub(crate) aether_iridescence_frequency: f64,
}

/// Fully-resolved artistic choices for a single seed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuratedChoices {
    pub(crate) composition_mode: CompositionMode,
    pub(crate) finish_cluster: FinishCluster,
    pub(crate) grade_family: GradeFamily,
    pub(crate) grade_preset: GradePreset,
    pub(crate) gradient_palette_index: usize,
    pub(crate) framing_zoom: f64,
    pub(crate) framing_shift_x: f64,
    pub(crate) framing_shift_y: f64,
    pub(crate) vignette_offset_x: f64,
    pub(crate) vignette_offset_y: f64,
    pub(crate) drift_scale_bias: f64,
    pub(crate) drift_arc_bias: f64,
    pub(crate) drift_eccentricity_bias: f64,
    pub(crate) probabilities: EnableProbabilities,
    pub(crate) softness_scale: f64,
    pub(crate) detail_scale: f64,
    pub(crate) grade_strength_scale: f64,
    pub(crate) vibrance_scale: f64,
    pub(crate) tone_curve_scale: f64,
    pub(crate) gradient_strength_scale: f64,
    pub(crate) gradient_hue_preservation_floor: f64,
    pub(crate) surface_preset: SurfacePreset,
    pub(crate) material_preset: MaterialPreset,
}

impl CuratedChoices {
    /// Choose a complete set of curated directions for a seed.
    #[must_use]
    pub(crate) fn choose(randomizer: &mut EffectRandomizer<'_>) -> Self {
        let composition_mode = weighted_pick(
            randomizer,
            &[
                (CompositionMode::CenteredMonument, 26),
                (CompositionMode::WideStage, 18),
                (CompositionMode::OffAxisTension, 22),
                (CompositionMode::DiagonalSweep, 20),
                (CompositionMode::IntimateFocus, 14),
            ],
        );
        let finish_cluster = weighted_pick(
            randomizer,
            &[
                (FinishCluster::Crisp, 50),
                (FinishCluster::Balanced, 38),
                (FinishCluster::Velvet, 12),
            ],
        );
        let grade_family = grade_family_for(randomizer, composition_mode, finish_cluster);

        let grade_candidates = grade_family.grade_candidates();
        let grade_preset = grade_candidates[random_index(randomizer, grade_candidates.len())];
        let palette_candidates = grade_family.gradient_palette_candidates();
        let gradient_palette_index = palette_candidates[random_index(randomizer, palette_candidates.len())];

        let (framing_zoom, framing_shift_x, framing_shift_y, vignette_offset_x, vignette_offset_y) =
            composition_profile(randomizer, composition_mode);
        let (drift_scale_bias, drift_arc_bias, drift_eccentricity_bias) =
            drift_profile(composition_mode);
        let (
            probabilities,
            softness_scale,
            detail_scale,
            grade_strength_scale,
            vibrance_scale,
            tone_curve_scale,
            gradient_strength_scale,
            gradient_hue_preservation_floor,
        ) = finish_profile(finish_cluster);
        let surface_preset = surface_preset(randomizer, finish_cluster);
        let material_preset = material_preset(randomizer, finish_cluster);

        Self {
            composition_mode,
            finish_cluster,
            grade_family,
            grade_preset,
            gradient_palette_index,
            framing_zoom,
            framing_shift_x,
            framing_shift_y,
            vignette_offset_x,
            vignette_offset_y,
            drift_scale_bias,
            drift_arc_bias,
            drift_eccentricity_bias,
            probabilities,
            softness_scale,
            detail_scale,
            grade_strength_scale,
            vibrance_scale,
            tone_curve_scale,
            gradient_strength_scale,
            gradient_hue_preservation_floor,
            surface_preset,
            material_preset,
        }
    }
}

fn random_index(randomizer: &mut EffectRandomizer<'_>, len: usize) -> usize {
    debug_assert!(len > 0);
    ((randomizer.random_unit() * len as f64).floor() as usize).min(len.saturating_sub(1))
}

fn sample_range(randomizer: &mut EffectRandomizer<'_>, min: f64, max: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        min
    } else {
        min + randomizer.random_unit() * (max - min)
    }
}

fn random_sign(randomizer: &mut EffectRandomizer<'_>) -> f64 {
    if randomizer.random_unit() < 0.5 { -1.0 } else { 1.0 }
}

fn weighted_pick<T: Copy>(randomizer: &mut EffectRandomizer<'_>, weighted: &[(T, u32)]) -> T {
    debug_assert!(!weighted.is_empty());
    let total_weight: u32 = weighted.iter().map(|(_, weight)| *weight).sum();
    let mut target = (randomizer.random_unit() * total_weight as f64).floor() as u32;
    for &(value, weight) in weighted {
        if target < weight {
            return value;
        }
        target -= weight;
    }
    weighted[weighted.len() - 1].0
}

fn composition_profile(
    randomizer: &mut EffectRandomizer<'_>,
    mode: CompositionMode,
) -> (f64, f64, f64, f64, f64) {
    match mode {
        CompositionMode::CenteredMonument => (
            sample_range(randomizer, 1.00, 1.05),
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        CompositionMode::WideStage => {
            let shift_x = random_sign(randomizer) * sample_range(randomizer, 0.04, 0.10);
            let shift_y = random_sign(randomizer) * sample_range(randomizer, 0.00, 0.05);
            (
                sample_range(randomizer, 1.12, 1.20),
                shift_x,
                shift_y,
                shift_x * 1.15,
                shift_y * 1.10,
            )
        }
        CompositionMode::OffAxisTension => {
            let shift_x = random_sign(randomizer) * sample_range(randomizer, 0.10, 0.16);
            let shift_y = random_sign(randomizer) * sample_range(randomizer, 0.04, 0.09);
            (
                sample_range(randomizer, 1.04, 1.12),
                shift_x,
                shift_y,
                shift_x * 1.20,
                shift_y * 1.15,
            )
        }
        CompositionMode::DiagonalSweep => {
            let shift_x = random_sign(randomizer) * sample_range(randomizer, 0.08, 0.14);
            let shift_y = -shift_x.signum() * sample_range(randomizer, 0.05, 0.10);
            (
                sample_range(randomizer, 1.02, 1.14),
                shift_x,
                shift_y,
                shift_x * 1.10,
                shift_y * 1.25,
            )
        }
        CompositionMode::IntimateFocus => {
            let shift_x = random_sign(randomizer) * sample_range(randomizer, 0.00, 0.06);
            let shift_y = random_sign(randomizer) * sample_range(randomizer, 0.00, 0.04);
            (
                sample_range(randomizer, 1.00, 1.03),
                shift_x,
                shift_y,
                shift_x * 0.90,
                shift_y * 0.90,
            )
        }
    }
}

fn drift_profile(mode: CompositionMode) -> (f64, f64, f64) {
    match mode {
        CompositionMode::CenteredMonument => (0.70, 0.55, 0.98),
        CompositionMode::WideStage => (0.82, 0.78, 1.00),
        CompositionMode::OffAxisTension => (0.76, 0.72, 1.00),
        CompositionMode::DiagonalSweep => (1.12, 1.18, 1.05),
        CompositionMode::IntimateFocus => (0.74, 0.62, 0.96),
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_profile(
    cluster: FinishCluster,
) -> (EnableProbabilities, f64, f64, f64, f64, f64, f64, f64) {
    match cluster {
        FinishCluster::Crisp => (
            EnableProbabilities {
                bloom: 0.18,
                glow: 0.30,
                chromatic_bloom: 0.08,
                perceptual_blur: 0.01,
                micro_contrast: 0.95,
                gradient_map: 0.14,
                color_grade: 0.72,
                champleve: 0.18,
                aether: 0.22,
                opalescence: 0.16,
                edge_luminance: 0.75,
                atmospheric_depth: 0.08,
                fine_texture: 0.58,
            },
            0.82,
            1.18,
            0.98,
            0.96,
            1.05,
            0.90,
            0.56,
        ),
        FinishCluster::Balanced => (
            EnableProbabilities {
                bloom: 0.22,
                glow: 0.42,
                chromatic_bloom: 0.10,
                perceptual_blur: 0.03,
                micro_contrast: 0.92,
                gradient_map: 0.16,
                color_grade: 0.78,
                champleve: 0.22,
                aether: 0.28,
                opalescence: 0.20,
                edge_luminance: 0.76,
                atmospheric_depth: 0.10,
                fine_texture: 0.52,
            },
            0.90,
            1.10,
            1.00,
            1.00,
            1.02,
            1.00,
            0.54,
        ),
        FinishCluster::Velvet => (
            EnableProbabilities {
                bloom: 0.22,
                glow: 0.40,
                chromatic_bloom: 0.10,
                perceptual_blur: 0.05,
                micro_contrast: 0.80,
                gradient_map: 0.08,
                color_grade: 0.84,
                champleve: 0.26,
                aether: 0.36,
                opalescence: 0.24,
                edge_luminance: 0.58,
                atmospheric_depth: 0.10,
                fine_texture: 0.22,
            },
            1.02,
            0.98,
            0.98,
            0.90,
            0.94,
            0.82,
            0.58,
        ),
    }
}

fn grade_family_for(
    randomizer: &mut EffectRandomizer<'_>,
    composition_mode: CompositionMode,
    finish_cluster: FinishCluster,
) -> GradeFamily {
    let weighted: &[(GradeFamily, u32)] = match finish_cluster {
        FinishCluster::Crisp => &[
            (GradeFamily::Filmic, 38),
            (GradeFamily::Graphic, 26),
            (GradeFamily::Jewel, 24),
            (GradeFamily::Atmospheric, 12),
        ],
        FinishCluster::Balanced => &[
            (GradeFamily::Filmic, 36),
            (GradeFamily::Jewel, 24),
            (GradeFamily::Graphic, 14),
            (GradeFamily::Atmospheric, 26),
        ],
        FinishCluster::Velvet => &[
            (GradeFamily::Filmic, 34),
            (GradeFamily::Jewel, 12),
            (GradeFamily::Graphic, 4),
            (GradeFamily::Atmospheric, 50),
        ],
    };
    let mut family = weighted_pick(randomizer, weighted);

    if matches!(composition_mode, CompositionMode::OffAxisTension | CompositionMode::IntimateFocus)
        && family == GradeFamily::Graphic
        && finish_cluster != FinishCluster::Crisp
    {
        family = GradeFamily::Filmic;
    }
    if composition_mode == CompositionMode::IntimateFocus && family == GradeFamily::Jewel {
        family = GradeFamily::Atmospheric;
    }

    family
}

fn surface_preset(randomizer: &mut EffectRandomizer<'_>, cluster: FinishCluster) -> SurfacePreset {
    let candidates: &[SurfacePreset] = match cluster {
        FinishCluster::Crisp => &[
            SurfacePreset { fine_texture_anisotropy: 0.52, fine_texture_angle: 12.0 },
            SurfacePreset { fine_texture_anisotropy: 0.68, fine_texture_angle: 90.0 },
        ],
        FinishCluster::Balanced => &[
            SurfacePreset { fine_texture_anisotropy: 0.24, fine_texture_angle: 36.0 },
            SurfacePreset { fine_texture_anisotropy: 0.42, fine_texture_angle: 58.0 },
        ],
        FinishCluster::Velvet => &[
            SurfacePreset { fine_texture_anisotropy: 0.10, fine_texture_angle: 78.0 },
            SurfacePreset { fine_texture_anisotropy: 0.18, fine_texture_angle: 112.0 },
        ],
    };
    candidates[random_index(randomizer, candidates.len())]
}

fn material_preset(randomizer: &mut EffectRandomizer<'_>, cluster: FinishCluster) -> MaterialPreset {
    let candidates: &[MaterialPreset] = match cluster {
        FinishCluster::Crisp => &[
            MaterialPreset {
                opalescence_chromatic_shift: 0.26,
                opalescence_pearl_sheen: 0.36,
                champleve_cell_density: 1.00,
                champleve_rim_sharpness: 0.62,
                aether_filament_density: 0.86,
                aether_iridescence_frequency: 0.92,
            },
            MaterialPreset {
                opalescence_chromatic_shift: 0.34,
                opalescence_pearl_sheen: 0.42,
                champleve_cell_density: 1.18,
                champleve_rim_sharpness: 0.68,
                aether_filament_density: 0.92,
                aether_iridescence_frequency: 1.00,
            },
        ],
        FinishCluster::Balanced => &[
            MaterialPreset {
                opalescence_chromatic_shift: 0.32,
                opalescence_pearl_sheen: 0.44,
                champleve_cell_density: 0.90,
                champleve_rim_sharpness: 0.48,
                aether_filament_density: 1.05,
                aether_iridescence_frequency: 1.05,
            },
            MaterialPreset {
                opalescence_chromatic_shift: 0.40,
                opalescence_pearl_sheen: 0.56,
                champleve_cell_density: 1.10,
                champleve_rim_sharpness: 0.58,
                aether_filament_density: 1.12,
                aether_iridescence_frequency: 1.12,
            },
        ],
        FinishCluster::Velvet => &[
            MaterialPreset {
                opalescence_chromatic_shift: 0.22,
                opalescence_pearl_sheen: 0.28,
                champleve_cell_density: 0.80,
                champleve_rim_sharpness: 0.40,
                aether_filament_density: 1.20,
                aether_iridescence_frequency: 0.86,
            },
            MaterialPreset {
                opalescence_chromatic_shift: 0.30,
                opalescence_pearl_sheen: 0.34,
                champleve_cell_density: 0.92,
                champleve_rim_sharpness: 0.46,
                aether_filament_density: 1.28,
                aether_iridescence_frequency: 0.92,
            },
        ],
    };
    candidates[random_index(randomizer, candidates.len())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sha3RandomByteStream;

    fn make_randomizer(seed_byte: u8) -> EffectRandomizer<'static> {
        let seed = Box::new([seed_byte; 32]);
        let leaked = Box::leak(seed);
        let rng = Box::new(Sha3RandomByteStream::new(leaked, 100.0, 300.0, 300.0, 1.0));
        let leaked_rng = Box::leak(rng);
        EffectRandomizer::new(leaked_rng)
    }

    #[test]
    fn curated_choices_stay_in_expected_bounds() {
        for seed in 0u8..=64 {
            let mut randomizer = make_randomizer(seed);
            let choices = CuratedChoices::choose(&mut randomizer);
            assert!((1.0..=1.20).contains(&choices.framing_zoom));
            assert!(choices.framing_shift_x.abs() <= 0.18);
            assert!(choices.framing_shift_y.abs() <= 0.12);
            assert!(choices.vignette_offset_x.abs() <= 0.22);
            assert!(choices.vignette_offset_y.abs() <= 0.16);
            assert!(choices.gradient_palette_index <= 14);
        }
    }

    #[test]
    fn centered_and_intimate_modes_keep_drift_calmer_than_diagonal() {
        let centered = drift_profile(CompositionMode::CenteredMonument);
        let intimate = drift_profile(CompositionMode::IntimateFocus);
        let diagonal = drift_profile(CompositionMode::DiagonalSweep);

        assert!(centered.0 < diagonal.0);
        assert!(centered.1 < diagonal.1);
        assert!(intimate.0 < diagonal.0);
        assert!(intimate.1 < diagonal.1);
    }

    #[test]
    fn softer_off_center_modes_avoid_graphic_family() {
        let mut randomizer = make_randomizer(17);
        for _ in 0..128 {
            let family = grade_family_for(
                &mut randomizer,
                CompositionMode::OffAxisTension,
                FinishCluster::Velvet,
            );
            assert_ne!(family, GradeFamily::Graphic);
        }
    }
}
