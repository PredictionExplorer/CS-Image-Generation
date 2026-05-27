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
}

impl CosmicSignatureParameters {
    fn resolve(rng: &mut Sha3RandomByteStream) -> Self {
        Self {
            hdr_scale: sample_range(rng, 0.155, 0.215),
            clip_black: sample_range(rng, 0.0045, 0.0085),
            clip_white: sample_range(rng, 0.9935, 0.9985),
            palette_phase: sample_range(rng, 0.0, 1.0),
            edge_energy: sample_range(rng, 0.92, 1.18),
        }
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
            enable_bloom: false,
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
            dog_strength: 0.0,
            dog_sigma_scale: 0.0,
            dog_ratio: 1.0,
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

        assert!(!c.enable_bloom);
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
        }
    }

    #[test]
    fn cosmic_signature_records_profile_randomization() {
        let mut rng = make_rng(&[1, 2, 3, 4]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);

        assert_eq!(profile.name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects.len(), 1);
        assert_eq!(profile.randomization_log.effects[0].effect_name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects[0].parameters.len(), 5);
    }

    #[test]
    fn cosmic_signature_builds_empty_finish_pipeline() {
        let mut rng = make_rng(&[9, 8, 7, 6]);
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

        assert_eq!(pipeline.trajectory_len(), 0);
        assert_eq!(pipeline.image_len(), 0);
    }
}
