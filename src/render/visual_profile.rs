//! Visual profiles for the renderer.
//!
//! The production generator uses a single crisp `CosmicSignature` profile: no
//! randomized post-effect stack, a clean black field, saturated spectral colour,
//! and restrained HDR values that preserve thin luminous structure.

use super::effect_randomizer::{RandomizationLog, RandomizationRecord};
use super::randomizable_config::ResolvedEffectConfig;
use crate::sim::Sha3RandomByteStream;

/// Canonical name recorded in generation metadata for the default visual style.
pub const COSMIC_SIGNATURE_PROFILE_NAME: &str = "cosmic_signature";

/// Probability that a seed receives the subtle halation (`DoG` bloom) finish.
pub const HALATION_PROBABILITY: f64 = 0.30;

/// How the three-body geometry is converted into luminous strokes.
///
/// Selected per seed; every mode reuses the same crisp spectral splatter, so
/// the family identity (thin luminous structure on black) is preserved while
/// the large-scale composition changes dramatically between modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureMode {
    /// Classic look: all three inter-body edges drawn each step.
    TriangleWeb,
    /// Each body paints its own trajectory as a continuous luminous ribbon.
    OrbitRibbons,
    /// Faint triangle web layered beneath full-strength orbit ribbons.
    WebRibbonHybrid,
    /// Two of the three edges drawn; `dropped_edge` (0..=2) is omitted.
    Duet {
        /// Index of the omitted edge (0 = body0-body1, 1 = body1-body2, 2 = body2-body0).
        dropped_edge: u8,
    },
    /// Lines from each body to the instantaneous triangle centroid.
    Spokes,
}

impl StructureMode {
    /// Stable identifier used for logging and generation metadata.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::TriangleWeb => "triangle_web",
            Self::OrbitRibbons => "orbit_ribbons",
            Self::WebRibbonHybrid => "web_ribbon_hybrid",
            Self::Duet { .. } => "duet",
            Self::Spokes => "spokes",
        }
    }

    /// Numeric index recorded in the randomization log.
    #[must_use]
    pub fn log_index(self) -> usize {
        match self {
            Self::TriangleWeb => 0,
            Self::OrbitRibbons => 1,
            Self::WebRibbonHybrid => 2,
            Self::Duet { .. } => 3,
            Self::Spokes => 4,
        }
    }
}

/// Seed-resolved scene traits consumed directly by the spectral accumulator.
#[derive(Clone, Copy, Debug)]
pub struct SceneTraits {
    /// Geometry-to-stroke conversion mode.
    pub structure: StructureMode,
    /// Global line weight multiplier (hairline seeds vs bold seeds).
    pub line_weight: f64,
    /// Trail-age exposure ramp in [-0.4, 0.4]; positive brightens late steps.
    pub age_ramp: f64,
}

impl Default for SceneTraits {
    fn default() -> Self {
        Self { structure: StructureMode::TriangleWeb, line_weight: 1.0, age_ramp: 0.0 }
    }
}

/// Small set of seed-varying aesthetic values that define the `CosmicSignature` look.
#[derive(Clone, Copy, Debug)]
pub struct CosmicSignatureParameters {
    /// Multiplier applied during spectral accumulation.
    pub hdr_scale: f64,
    /// Lower percentile used as the tonemapping black point.
    pub clip_black: f64,
    /// Upper percentile used as the tonemapping white point.
    pub clip_white: f64,
    /// Palette phase value reserved for deterministic colour evolution.
    pub palette_phase: f64,
    /// Relative line energy reserved for future edge/ribbon tuning.
    pub edge_energy: f64,
    /// Display exposure key: < 1 renders darker/ember seeds, > 1 brighter/airier seeds.
    pub exposure_key: f64,
    /// Geometry structure mode for this seed.
    pub structure: StructureMode,
    /// Global line weight multiplier.
    pub line_weight: f64,
    /// Trail-age exposure ramp (negative = early steps brighter).
    pub age_ramp: f64,
    /// Subtle halation (tight `DoG` bloom) strength; 0 disables the pass entirely.
    pub halation_strength: f64,
    /// Halation inner sigma, relative to the output short edge.
    pub halation_radius_scale: f64,
    /// Halation outer/inner sigma ratio (controls halo softness).
    pub halation_softness: f64,
}

impl CosmicSignatureParameters {
    fn resolve(rng: &mut Sha3RandomByteStream) -> Self {
        let hdr_scale = sample_range(rng, 0.155, 0.215);
        let clip_black = sample_range(rng, 0.0045, 0.0085);
        let clip_white = sample_range(rng, 0.9935, 0.9985);
        let palette_phase = sample_range(rng, 0.0, 1.0);
        let edge_energy = sample_range(rng, 0.92, 1.18);

        let exposure_key = sample_range(rng, 0.85, 1.12);
        let line_weight = sample_range(rng, 0.85, 1.45);
        let age_ramp = sample_range(rng, -0.35, 0.35);
        let structure = resolve_structure_mode(rng);

        let halation_roll = rng.next_f64();
        let (halation_strength, halation_radius_scale, halation_softness) =
            if halation_roll < HALATION_PROBABILITY {
                (
                    sample_range(rng, 0.05, 0.14),
                    sample_range(rng, 0.0026, 0.0046),
                    sample_range(rng, 2.8, 4.0),
                )
            } else {
                (0.0, 0.0035, 3.2)
            };

        Self {
            hdr_scale,
            clip_black,
            clip_white,
            palette_phase,
            edge_energy,
            exposure_key,
            structure,
            line_weight,
            age_ramp,
            halation_strength,
            halation_radius_scale,
            halation_softness,
        }
    }

    /// Bundle the accumulation-facing traits for the renderer.
    #[must_use]
    pub fn scene_traits(&self) -> SceneTraits {
        SceneTraits {
            structure: self.structure,
            line_weight: self.line_weight,
            age_ramp: self.age_ramp,
        }
    }
}

/// Sample the seeded structure mode with curated weights.
///
/// Weights keep the classic triangle web as the most common outcome while
/// making each alternative mode a meaningful (non-rare) population.
fn resolve_structure_mode(rng: &mut Sha3RandomByteStream) -> StructureMode {
    let roll = rng.next_f64();
    if roll < 0.40 {
        StructureMode::TriangleWeb
    } else if roll < 0.60 {
        StructureMode::OrbitRibbons
    } else if roll < 0.76 {
        StructureMode::WebRibbonHybrid
    } else if roll < 0.90 {
        let dropped_edge = ((rng.next_f64() * 3.0).floor() as u8).min(2);
        StructureMode::Duet { dropped_edge }
    } else {
        StructureMode::Spokes
    }
}

/// Fully resolved `CosmicSignature` profile plus metadata for logging.
#[derive(Clone, Debug)]
pub struct ResolvedVisualProfile {
    /// Human-readable profile identifier.
    pub name: &'static str,
    /// Core seed-varying visual parameters.
    pub parameters: CosmicSignatureParameters,
    /// Render/effect configuration consumed by the existing pipeline.
    pub effect_config: ResolvedEffectConfig,
    /// Metadata describing deterministic profile randomization.
    pub randomization_log: RandomizationLog,
}

impl ResolvedVisualProfile {
    /// Resolve the default `CosmicSignature` profile for the output dimensions.
    pub fn cosmic_signature(rng: &mut Sha3RandomByteStream, width: u32, height: u32) -> Self {
        let parameters = CosmicSignatureParameters::resolve(rng);
        let effect_config = ResolvedEffectConfig {
            width,
            height,
            enable_bloom: parameters.halation_strength > 0.0,
            enable_glow: false,
            enable_chromatic_bloom: false,
            enable_perceptual_blur: false,
            enable_micro_contrast: false,
            enable_gradient_map: false,
            enable_color_grade: false,
            enable_champleve: false,
            enable_aether: false,
            enable_opalescence: false,
            enable_edge_luminance: false,
            enable_atmospheric_depth: false,
            enable_fine_texture: false,
            blur_strength: 0.0,
            blur_radius_scale: 0.0,
            blur_core_brightness: 1.0,
            dog_strength: parameters.halation_strength,
            dog_sigma_scale: parameters.halation_radius_scale,
            dog_ratio: parameters.halation_softness,
            glow_strength: 0.0,
            glow_threshold: 1.0,
            glow_radius_scale: 0.0,
            glow_sharpness: 1.0,
            glow_saturation_boost: 0.0,
            chromatic_bloom_strength: 0.0,
            chromatic_bloom_radius_scale: 0.0,
            chromatic_bloom_separation_scale: 0.0,
            chromatic_bloom_threshold: 1.0,
            perceptual_blur_strength: 0.0,
            color_grade_strength: 0.0,
            vignette_strength: 0.0,
            vignette_softness: 1.0,
            vibrance: 1.0,
            clarity_strength: 0.0,
            tone_curve_strength: 0.0,
            gradient_map_strength: 0.0,
            gradient_map_hue_preservation: 1.0,
            gradient_map_palette: 0,
            opalescence_strength: 0.0,
            opalescence_scale: 0.0,
            opalescence_layers: 1,
            champleve_flow_alignment: 0.0,
            champleve_interference_amplitude: 0.0,
            champleve_rim_intensity: 0.0,
            champleve_rim_warmth: 0.0,
            champleve_interior_lift: 0.0,
            aether_flow_alignment: 0.0,
            aether_scattering_strength: 0.0,
            aether_iridescence_amplitude: 0.0,
            aether_caustic_strength: 0.0,
            micro_contrast_strength: 0.0,
            micro_contrast_radius: 1,
            edge_luminance_strength: 0.0,
            edge_luminance_threshold: 1.0,
            edge_luminance_brightness_boost: 0.0,
            atmospheric_depth_strength: 0.0,
            atmospheric_desaturation: 0.0,
            atmospheric_darkening: 0.0,
            atmospheric_fog_color_r: 0.0,
            atmospheric_fog_color_g: 0.0,
            atmospheric_fog_color_b: 0.0,
            fine_texture_strength: 0.0,
            fine_texture_scale: 0.0,
            fine_texture_contrast: 0.0,
            hdr_scale: parameters.hdr_scale,
            clip_black: parameters.clip_black,
            clip_white: parameters.clip_white,
            nebula_strength: 0.0,
            nebula_octaves: 1,
            nebula_base_frequency: 0.0,
        };

        Self {
            name: COSMIC_SIGNATURE_PROFILE_NAME,
            parameters,
            effect_config,
            randomization_log: build_profile_log(parameters),
        }
    }
}

fn sample_range(rng: &mut Sha3RandomByteStream, min: f64, max: f64) -> f64 {
    min + rng.next_f64() * (max - min)
}

fn build_profile_log(parameters: CosmicSignatureParameters) -> RandomizationLog {
    let mut record = RandomizationRecord::new(COSMIC_SIGNATURE_PROFILE_NAME, true, true);
    record.add_float("hdr_scale", parameters.hdr_scale, true, (0.155, 0.215));
    record.add_float("clip_black", parameters.clip_black, true, (0.0045, 0.0085));
    record.add_float("clip_white", parameters.clip_white, true, (0.9935, 0.9985));
    record.add_float("palette_phase", parameters.palette_phase, true, (0.0, 1.0));
    record.add_float("edge_energy", parameters.edge_energy, true, (0.92, 1.18));
    record.add_float("exposure_key", parameters.exposure_key, true, (0.85, 1.12));
    record.add_float("line_weight", parameters.line_weight, true, (0.85, 1.45));
    record.add_float("age_ramp", parameters.age_ramp, true, (-0.35, 0.35));
    record.add_int("structure_mode", parameters.structure.log_index(), true, (0, 4));
    record.add_float("halation_strength", parameters.halation_strength, true, (0.0, 0.14));

    let mut log = RandomizationLog::new();
    log.add_record(record);
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: &[u8]) -> Sha3RandomByteStream {
        Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0)
    }

    #[test]
    fn cosmic_signature_disables_legacy_effects() {
        let mut rng = make_rng(&[0x10, 0x00, 0x33]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 1920, 1080);
        let c = profile.effect_config;

        // Halation (DoG bloom) is the only seed-gated finish; everything else stays off.
        assert_eq!(c.enable_bloom, profile.parameters.halation_strength > 0.0);
        assert!(!c.enable_glow);
        assert!(!c.enable_chromatic_bloom);
        assert!(!c.enable_perceptual_blur);
        assert!(!c.enable_gradient_map);
        assert!(!c.enable_color_grade);
        assert!(!c.enable_champleve);
        assert!(!c.enable_aether);
        assert!(!c.enable_opalescence);
        assert!(!c.enable_edge_luminance);
        assert!(!c.enable_atmospheric_depth);
        assert!(!c.enable_fine_texture);
        assert_eq!(c.nebula_strength, 0.0);
    }

    #[test]
    fn cosmic_signature_halation_is_gated_and_curated() {
        let mut enabled = 0usize;
        let total = 128usize;
        for seed in 0..total {
            let mut rng = make_rng(&[seed as u8, 0x77, 0x21]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            let p = profile.parameters;

            assert_eq!(profile.effect_config.enable_bloom, p.halation_strength > 0.0);
            if p.halation_strength > 0.0 {
                enabled += 1;
                assert!((0.05..=0.14).contains(&p.halation_strength));
                assert!((0.0026..=0.0046).contains(&p.halation_radius_scale));
                assert!((2.8..=4.0).contains(&p.halation_softness));
                assert_eq!(profile.effect_config.dog_strength, p.halation_strength);
            }
        }

        // ~30% gate: bounds are loose so the test is robust to seed-set drift.
        assert!(enabled > total / 10, "halation almost never enabled: {enabled}/{total}");
        assert!(enabled < total * 6 / 10, "halation enabled too often: {enabled}/{total}");
    }

    #[test]
    fn cosmic_signature_structure_modes_cover_all_variants() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0u16..512 {
            let bytes = seed.to_le_bytes();
            let mut rng = make_rng(&[bytes[0], bytes[1], 0x5A]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            seen.insert(profile.parameters.structure.log_index());
            if let StructureMode::Duet { dropped_edge } = profile.parameters.structure {
                assert!(dropped_edge <= 2, "dropped edge out of range: {dropped_edge}");
            }
        }
        assert_eq!(seen.len(), 5, "all structure modes should occur across seeds: {seen:?}");
    }

    #[test]
    fn cosmic_signature_is_deterministic_for_seed() {
        let mut rng_a = make_rng(&[0xCA, 0xFE]);
        let mut rng_b = make_rng(&[0xCA, 0xFE]);

        let a = ResolvedVisualProfile::cosmic_signature(&mut rng_a, 800, 450);
        let b = ResolvedVisualProfile::cosmic_signature(&mut rng_b, 800, 450);

        assert_eq!(a.parameters.hdr_scale.to_bits(), b.parameters.hdr_scale.to_bits());
        assert_eq!(a.parameters.clip_black.to_bits(), b.parameters.clip_black.to_bits());
        assert_eq!(a.parameters.clip_white.to_bits(), b.parameters.clip_white.to_bits());
        assert_eq!(a.parameters.palette_phase.to_bits(), b.parameters.palette_phase.to_bits());
        assert_eq!(a.parameters.edge_energy.to_bits(), b.parameters.edge_energy.to_bits());
    }

    #[test]
    fn cosmic_signature_parameters_stay_in_crisp_ranges() {
        for seed in 0u8..64 {
            let mut rng = make_rng(&[seed, 2, 3, 4]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            let p = profile.parameters;

            assert!((0.155..=0.215).contains(&p.hdr_scale));
            assert!((0.0045..=0.0085).contains(&p.clip_black));
            assert!((0.9935..=0.9985).contains(&p.clip_white));
            assert!((0.0..=1.0).contains(&p.palette_phase));
            assert!((0.92..=1.18).contains(&p.edge_energy));
            assert!((0.85..=1.12).contains(&p.exposure_key));
            assert!((0.85..=1.45).contains(&p.line_weight));
            assert!((-0.35..=0.35).contains(&p.age_ramp));
            assert!((0.0..=0.14).contains(&p.halation_strength));
        }
    }

    #[test]
    fn cosmic_signature_records_profile_randomization() {
        let mut rng = make_rng(&[1, 2, 3, 4]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);

        assert_eq!(profile.name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects.len(), 1);
        assert_eq!(profile.randomization_log.effects[0].effect_name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects[0].parameters.len(), 10);
    }

    #[test]
    fn scene_traits_default_matches_classic_look() {
        let traits = SceneTraits::default();
        assert_eq!(traits.structure, StructureMode::TriangleWeb);
        assert_eq!(traits.line_weight, 1.0);
        assert_eq!(traits.age_ramp, 0.0);
    }

    #[test]
    fn cosmic_signature_finish_pipeline_is_halation_only() {
        let mut saw_empty = false;
        let mut saw_halation = false;

        for seed in 0u8..64 {
            let mut rng = make_rng(&[seed, 8, 7, 6]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 1280, 720);
            let render_config = crate::render::RenderConfig {
                hdr_scale: profile.effect_config.hdr_scale,
                bloom_mode: crate::render::BloomMode::Dog,
            };
            let effect_config = crate::render::build_effect_config_from_resolved(
                &profile.effect_config,
                &render_config,
                crate::render::FinishOutputMode::Still,
            );
            let pipeline = crate::render::effects::FinishEffectPipeline::new(effect_config);

            // The image (post-tonemap) chain stays empty in every case; the
            // trajectory chain may contain exactly the gated halation bloom.
            assert_eq!(pipeline.image_len(), 0);
            if profile.parameters.halation_strength > 0.0 {
                assert_eq!(pipeline.trajectory_len(), 1, "halation seed should add one pass");
                saw_halation = true;
            } else {
                assert_eq!(pipeline.trajectory_len(), 0, "non-halation seed must stay empty");
                saw_empty = true;
            }
        }

        assert!(saw_empty, "expected at least one seed without halation");
        assert!(saw_halation, "expected at least one seed with halation");
    }
}
