//! Visual profiles for the renderer.
//!
//! The production generator uses a single crisp `CosmicSignature` profile: no
//! randomized post-effect stack, a clean black field, saturated spectral colour,
//! and restrained HDR values that preserve thin luminous structure.
//!
//! Within that family identity each seed resolves a structure mode (how the
//! three-body geometry becomes strokes), mode-aware stroke tuning (ribbon-style
//! modes draw bolder and always receive halation so they read as luminous
//! bands rather than sparse hairlines), and a small set of rare, seed-gated
//! traits (prism dispersion, nebula whisper, mirrored composition) that give a
//! minority of outputs a collectible twist without breaking the family look.

use super::effect_randomizer::{RandomizationLog, RandomizationRecord};
use super::randomizable_config::ResolvedEffectConfig;
use crate::sim::Sha3RandomByteStream;

/// Canonical name recorded in generation metadata for the default visual style.
pub const COSMIC_SIGNATURE_PROFILE_NAME: &str = "cosmic_signature";

/// Probability that a seed receives the rare prism (chromatic bloom) trait.
pub const PRISM_PROBABILITY: f64 = 0.05;

/// Probability that a seed receives the rare nebula-whisper background trait.
pub const NEBULA_WHISPER_PROBABILITY: f64 = 0.05;

/// Probability that a seed receives the rare mirrored composition trait.
pub const MIRROR_PROBABILITY: f64 = 0.03;

/// Probability that an `OrbitRibbons` seed layers a time-lagged echo band.
pub const RIBBON_ECHO_PROBABILITY: f64 = 0.60;

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
    /// String-art chords from each body to its own time-lagged future position,
    /// layered over a faint ribbon underlay (ruled-surface sheets).
    TimeChords,
    /// Orbit ribbons trailed by decaying time-lagged echo bands (comet tails).
    CometRibbons,
    /// Reduced-alpha triangle web interleaved with centroid spokes (lace).
    WebSpokesLace,
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
            Self::TimeChords => "time_chords",
            Self::CometRibbons => "comet_ribbons",
            Self::WebSpokesLace => "web_spokes_lace",
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
            Self::TimeChords => 5,
            Self::CometRibbons => 6,
            Self::WebSpokesLace => 7,
        }
    }

    /// True for modes whose primary geometry is per-body trails (sparse ink).
    #[must_use]
    pub fn is_ribbon_like(self) -> bool {
        matches!(self, Self::OrbitRibbons | Self::CometRibbons)
    }
}

/// Seed-resolved scene traits consumed directly by the spectral accumulator.
#[derive(Clone, Copy, Debug)]
pub struct SceneTraits {
    /// Geometry-to-stroke conversion mode.
    pub structure: StructureMode,
    /// Global line weight multiplier (hairline seeds vs bold seeds).
    pub line_weight: f64,
    /// Trail-age exposure ramp in [-0.6, 0.6]; positive brightens late steps.
    pub age_ramp: f64,
    /// Per-edge energy asymmetry for web-style modes (1.0 = symmetric).
    pub edge_energy: f64,
    /// Time lag, as a fraction of total steps, for chord/echo strokes.
    pub chord_lag_fraction: f64,
    /// Alpha scale of the time-lagged ribbon echo layer (0 disables it).
    pub ribbon_echo_alpha: f64,
    /// Rare trait: mirror every stroke across the vertical frame axis.
    pub mirror: bool,
}

impl Default for SceneTraits {
    fn default() -> Self {
        Self {
            structure: StructureMode::TriangleWeb,
            line_weight: 1.0,
            age_ramp: 0.0,
            edge_energy: 1.0,
            chord_lag_fraction: 0.008,
            ribbon_echo_alpha: 0.0,
            mirror: false,
        }
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
    /// Per-edge energy asymmetry consumed by web-style structure modes.
    pub edge_energy: f64,
    /// Display exposure key: < 1 renders darker/ember seeds, > 1 brighter/airier seeds.
    pub exposure_key: f64,
    /// Geometry structure mode for this seed.
    pub structure: StructureMode,
    /// Global line weight multiplier (range depends on the structure mode).
    pub line_weight: f64,
    /// Trail-age exposure ramp (negative = early steps brighter).
    pub age_ramp: f64,
    /// Time lag, as a fraction of total steps, for chord/echo strokes.
    pub chord_lag_fraction: f64,
    /// Alpha scale of the ribbon echo layer (0 disables the pass).
    pub ribbon_echo_alpha: f64,
    /// Subtle halation (tight `DoG` bloom) strength; 0 disables the pass entirely.
    pub halation_strength: f64,
    /// Halation inner sigma, relative to the output short edge.
    pub halation_radius_scale: f64,
    /// Halation outer/inner sigma ratio (controls halo softness).
    pub halation_softness: f64,
    /// Rare prism trait: chromatic bloom strength (0 disables).
    pub prism_strength: f64,
    /// Prism blur radius relative to the output short edge.
    pub prism_radius_scale: f64,
    /// Prism RGB channel separation relative to the output short edge.
    pub prism_separation_scale: f64,
    /// Prism luminance activation threshold.
    pub prism_threshold: f64,
    /// Rare nebula-whisper trait: faint procedural background strength (0 disables).
    pub nebula_whisper_strength: f64,
    /// Nebula noise octaves used when the whisper trait is active.
    pub nebula_octaves: usize,
    /// Nebula noise base frequency used when the whisper trait is active.
    pub nebula_base_frequency: f64,
    /// Rare trait: mirrored (kaleidoscopic) composition.
    pub mirror: bool,
}

/// Mode-aware line weight range: ribbon-style modes draw bolder strokes so the
/// sparse per-body trails read as luminous bands instead of hairlines.
fn line_weight_range(structure: StructureMode) -> (f64, f64) {
    match structure {
        StructureMode::TriangleWeb | StructureMode::Duet { .. } => (0.75, 1.60),
        StructureMode::OrbitRibbons => (1.30, 2.20),
        StructureMode::CometRibbons => (1.20, 2.00),
        StructureMode::WebRibbonHybrid => (0.95, 1.70),
        StructureMode::Spokes => (0.85, 1.50),
        StructureMode::TimeChords => (0.90, 1.70),
        StructureMode::WebSpokesLace => (0.80, 1.45),
    }
}

/// Mode-aware halation gate: sparse modes always glow, dense webs rarely do.
fn halation_profile(structure: StructureMode) -> (f64, (f64, f64)) {
    match structure {
        StructureMode::OrbitRibbons | StructureMode::CometRibbons => (1.0, (0.10, 0.18)),
        StructureMode::TimeChords => (0.60, (0.06, 0.15)),
        StructureMode::Spokes | StructureMode::WebSpokesLace => (0.50, (0.05, 0.14)),
        StructureMode::WebRibbonHybrid => (0.45, (0.05, 0.14)),
        StructureMode::TriangleWeb | StructureMode::Duet { .. } => (0.25, (0.05, 0.14)),
    }
}

impl CosmicSignatureParameters {
    fn resolve(rng: &mut Sha3RandomByteStream) -> Self {
        let hdr_scale = sample_range(rng, 0.155, 0.215);
        let clip_black = sample_range(rng, 0.0045, 0.0085);
        let clip_white = sample_range(rng, 0.9935, 0.9985);
        let palette_phase = sample_range(rng, 0.0, 1.0);
        let edge_energy = sample_range(rng, 0.92, 1.18);
        let exposure_key = sample_range(rng, 0.85, 1.12);

        let structure = resolve_structure_mode(rng);
        let (lw_min, lw_max) = line_weight_range(structure);
        let line_weight = sample_range(rng, lw_min, lw_max);
        let age_ramp = sample_range(rng, -0.6, 0.6);
        let chord_lag_fraction = sample_range(rng, 0.004, 0.020);

        let ribbon_echo_alpha = match structure {
            StructureMode::OrbitRibbons => {
                if rng.next_f64() < RIBBON_ECHO_PROBABILITY {
                    sample_range(rng, 0.18, 0.35)
                } else {
                    0.0
                }
            }
            StructureMode::CometRibbons => sample_range(rng, 0.30, 0.50),
            _ => 0.0,
        };

        let (halation_probability, (hal_min, hal_max)) = halation_profile(structure);
        let halation_roll = rng.next_f64();
        let (halation_strength, halation_radius_scale, halation_softness) =
            if halation_roll < halation_probability {
                let radius_range =
                    if structure.is_ribbon_like() { (0.0030, 0.0050) } else { (0.0026, 0.0046) };
                (
                    sample_range(rng, hal_min, hal_max),
                    sample_range(rng, radius_range.0, radius_range.1),
                    sample_range(rng, 2.8, 4.0),
                )
            } else {
                (0.0, 0.0035, 3.2)
            };

        // Rare seed-gated traits. Each is deterministic per seed and logged so
        // rarity is auditable from generation metadata.
        let prism_roll = rng.next_f64();
        let (prism_strength, prism_radius_scale, prism_separation_scale, prism_threshold) =
            if prism_roll < PRISM_PROBABILITY {
                (
                    sample_range(rng, 0.16, 0.30),
                    sample_range(rng, 0.0035, 0.0050),
                    sample_range(rng, 0.0008, 0.0014),
                    sample_range(rng, 0.20, 0.26),
                )
            } else {
                (0.0, 0.0042, 0.0011, 0.23)
            };

        let nebula_roll = rng.next_f64();
        let (nebula_whisper_strength, nebula_octaves, nebula_base_frequency) =
            if nebula_roll < NEBULA_WHISPER_PROBABILITY {
                let strength = sample_range(rng, 0.04, 0.09);
                let octaves = if rng.next_f64() < 0.5 { 3 } else { 4 };
                (strength, octaves, sample_range(rng, 0.0010, 0.0018))
            } else {
                (0.0, 4, 0.0015)
            };

        let mirror = rng.next_f64() < MIRROR_PROBABILITY;

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
            chord_lag_fraction,
            ribbon_echo_alpha,
            halation_strength,
            halation_radius_scale,
            halation_softness,
            prism_strength,
            prism_radius_scale,
            prism_separation_scale,
            prism_threshold,
            nebula_whisper_strength,
            nebula_octaves,
            nebula_base_frequency,
            mirror,
        }
    }

    /// Bundle the accumulation-facing traits for the renderer.
    #[must_use]
    pub fn scene_traits(&self) -> SceneTraits {
        SceneTraits {
            structure: self.structure,
            line_weight: self.line_weight,
            age_ramp: self.age_ramp,
            edge_energy: self.edge_energy,
            chord_lag_fraction: self.chord_lag_fraction,
            ribbon_echo_alpha: self.ribbon_echo_alpha,
            mirror: self.mirror,
        }
    }
}

/// Sample the seeded structure mode with curated weights.
///
/// The classic triangle web stays the most common single outcome while every
/// alternative mode (including the chord/comet/lace additions) remains a
/// meaningful, non-rare population.
fn resolve_structure_mode(rng: &mut Sha3RandomByteStream) -> StructureMode {
    let roll = rng.next_f64();
    if roll < 0.26 {
        StructureMode::TriangleWeb
    } else if roll < 0.42 {
        StructureMode::OrbitRibbons
    } else if roll < 0.55 {
        StructureMode::WebRibbonHybrid
    } else if roll < 0.66 {
        let dropped_edge = ((rng.next_f64() * 3.0).floor() as u8).min(2);
        StructureMode::Duet { dropped_edge }
    } else if roll < 0.74 {
        StructureMode::Spokes
    } else if roll < 0.86 {
        StructureMode::TimeChords
    } else if roll < 0.94 {
        StructureMode::CometRibbons
    } else {
        StructureMode::WebSpokesLace
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
            enable_chromatic_bloom: parameters.prism_strength > 0.0,
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
            chromatic_bloom_strength: parameters.prism_strength,
            chromatic_bloom_radius_scale: parameters.prism_radius_scale,
            chromatic_bloom_separation_scale: parameters.prism_separation_scale,
            chromatic_bloom_threshold: parameters.prism_threshold,
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
            nebula_strength: parameters.nebula_whisper_strength,
            nebula_octaves: parameters.nebula_octaves,
            nebula_base_frequency: parameters.nebula_base_frequency,
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
    record.add_int("structure_mode", parameters.structure.log_index(), true, (0, 7));
    record.add_float("line_weight", parameters.line_weight, true, (0.75, 2.20));
    record.add_float("age_ramp", parameters.age_ramp, true, (-0.6, 0.6));
    record.add_float("chord_lag_fraction", parameters.chord_lag_fraction, true, (0.004, 0.020));
    record.add_float("ribbon_echo_alpha", parameters.ribbon_echo_alpha, true, (0.0, 0.50));
    record.add_float("halation_strength", parameters.halation_strength, true, (0.0, 0.18));
    record.add_float("prism_strength", parameters.prism_strength, true, (0.0, 0.30));
    record.add_float(
        "nebula_whisper_strength",
        parameters.nebula_whisper_strength,
        true,
        (0.0, 0.09),
    );
    record.add_int("mirror", usize::from(parameters.mirror), true, (0, 1));

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
    fn cosmic_signature_keeps_softening_legacy_effects_off() {
        let mut rng = make_rng(&[0x10, 0x00, 0x33]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 1920, 1080);
        let c = profile.effect_config;

        // Halation, prism and nebula whisper are the only seed-gated finishes;
        // every other legacy effect must stay off for all seeds.
        assert_eq!(c.enable_bloom, profile.parameters.halation_strength > 0.0);
        assert_eq!(c.enable_chromatic_bloom, profile.parameters.prism_strength > 0.0);
        assert_eq!(c.nebula_strength, profile.parameters.nebula_whisper_strength);
        assert!(!c.enable_glow);
        assert!(!c.enable_perceptual_blur);
        assert!(!c.enable_gradient_map);
        assert!(!c.enable_color_grade);
        assert!(!c.enable_champleve);
        assert!(!c.enable_aether);
        assert!(!c.enable_opalescence);
        assert!(!c.enable_edge_luminance);
        assert!(!c.enable_atmospheric_depth);
        assert!(!c.enable_fine_texture);
    }

    #[test]
    fn cosmic_signature_halation_is_gated_and_curated() {
        let mut enabled = 0usize;
        let mut ribbon_like = 0usize;
        let mut ribbon_like_with_halation = 0usize;
        let total = 256usize;
        for seed in 0..total {
            let mut rng = make_rng(&[(seed & 0xff) as u8, (seed >> 8) as u8, 0x77, 0x21]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            let p = profile.parameters;

            assert_eq!(profile.effect_config.enable_bloom, p.halation_strength > 0.0);
            if p.structure.is_ribbon_like() {
                ribbon_like += 1;
                if p.halation_strength > 0.0 {
                    ribbon_like_with_halation += 1;
                }
            }
            if p.halation_strength > 0.0 {
                enabled += 1;
                assert!((0.05..=0.18).contains(&p.halation_strength));
                assert!((0.0026..=0.0050).contains(&p.halation_radius_scale));
                assert!((2.8..=4.0).contains(&p.halation_softness));
                assert_eq!(profile.effect_config.dog_strength, p.halation_strength);
            }
        }

        // Per-mode gates put the population average near ~50%; bounds are loose
        // so the test is robust to seed-set drift.
        assert!(enabled > total / 5, "halation almost never enabled: {enabled}/{total}");
        assert!(enabled < total * 4 / 5, "halation enabled too often: {enabled}/{total}");
        // Ribbon-style modes are guaranteed halation so sparse trails glow.
        assert!(ribbon_like > 0, "expected ribbon-like seeds in the sample");
        assert_eq!(
            ribbon_like, ribbon_like_with_halation,
            "every ribbon-like seed must receive halation"
        );
    }

    #[test]
    fn cosmic_signature_structure_modes_cover_all_variants() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0u16..1024 {
            let bytes = seed.to_le_bytes();
            let mut rng = make_rng(&[bytes[0], bytes[1], 0x5A]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            seen.insert(profile.parameters.structure.log_index());
            if let StructureMode::Duet { dropped_edge } = profile.parameters.structure {
                assert!(dropped_edge <= 2, "dropped edge out of range: {dropped_edge}");
            }
        }
        assert_eq!(seen.len(), 8, "all structure modes should occur across seeds: {seen:?}");
    }

    #[test]
    fn ribbon_modes_resolve_bolder_line_weights() {
        let mut ribbon_min = f64::INFINITY;
        let mut web_max = f64::NEG_INFINITY;
        for seed in 0u16..512 {
            let bytes = seed.to_le_bytes();
            let mut rng = make_rng(&[bytes[0], bytes[1], 0x9C]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            let p = profile.parameters;
            match p.structure {
                StructureMode::OrbitRibbons => ribbon_min = ribbon_min.min(p.line_weight),
                StructureMode::TriangleWeb => web_max = web_max.max(p.line_weight),
                _ => {}
            }
        }
        assert!(
            ribbon_min >= 1.30,
            "orbit ribbons must draw bold strokes, found line weight {ribbon_min}"
        );
        assert!(web_max <= 1.60, "triangle web line weight exceeded mode cap: {web_max}");
    }

    #[test]
    fn ribbon_echo_and_chord_lag_stay_in_curated_ranges() {
        let mut echo_seen = false;
        for seed in 0u16..768 {
            let bytes = seed.to_le_bytes();
            let mut rng = make_rng(&[bytes[0], bytes[1], 0x44]);
            let p = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360).parameters;

            assert!((0.004..=0.020).contains(&p.chord_lag_fraction));
            match p.structure {
                StructureMode::OrbitRibbons => {
                    if p.ribbon_echo_alpha > 0.0 {
                        echo_seen = true;
                        assert!((0.18..=0.35).contains(&p.ribbon_echo_alpha));
                    }
                }
                StructureMode::CometRibbons => {
                    assert!(
                        (0.30..=0.50).contains(&p.ribbon_echo_alpha),
                        "comet ribbons always carry echo bands"
                    );
                }
                _ => assert_eq!(p.ribbon_echo_alpha, 0.0),
            }
        }
        assert!(echo_seen, "expected at least one orbit-ribbons seed with the echo layer");
    }

    #[test]
    fn rare_traits_are_gated_and_curated() {
        let total = 2048usize;
        let mut prism = 0usize;
        let mut nebula = 0usize;
        let mut mirrored = 0usize;
        for seed in 0..total {
            let mut rng = make_rng(&[(seed & 0xff) as u8, (seed >> 8) as u8, 0xE7]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            let p = profile.parameters;

            if p.prism_strength > 0.0 {
                prism += 1;
                assert!((0.16..=0.30).contains(&p.prism_strength));
                assert!((0.0035..=0.0050).contains(&p.prism_radius_scale));
                assert!((0.0008..=0.0014).contains(&p.prism_separation_scale));
                assert!((0.20..=0.26).contains(&p.prism_threshold));
                assert!(profile.effect_config.enable_chromatic_bloom);
            } else {
                assert!(!profile.effect_config.enable_chromatic_bloom);
            }

            if p.nebula_whisper_strength > 0.0 {
                nebula += 1;
                assert!((0.04..=0.09).contains(&p.nebula_whisper_strength));
                assert!((3..=4).contains(&p.nebula_octaves));
                assert!((0.0010..=0.0018).contains(&p.nebula_base_frequency));
            }
            assert_eq!(profile.effect_config.nebula_strength, p.nebula_whisper_strength);

            if p.mirror {
                mirrored += 1;
            }
        }

        // Each rare trait should appear, but stay a small minority (loose bounds).
        for (name, count, max_fraction) in
            [("prism", prism, 0.15), ("nebula", nebula, 0.15), ("mirror", mirrored, 0.10)]
        {
            assert!(count > 0, "{name} trait never appeared in {total} seeds");
            // usize→f64: counts are bounded by `total`.
            let fraction = count as f64 / total as f64;
            assert!(fraction < max_fraction, "{name} trait too common: {fraction:.3}");
        }
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
        assert_eq!(a.parameters.structure, b.parameters.structure);
        assert_eq!(a.parameters.line_weight.to_bits(), b.parameters.line_weight.to_bits());
        assert_eq!(a.parameters.prism_strength.to_bits(), b.parameters.prism_strength.to_bits());
        assert_eq!(a.parameters.mirror, b.parameters.mirror);
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
            assert!((0.75..=2.20).contains(&p.line_weight));
            assert!((-0.6..=0.6).contains(&p.age_ramp));
            assert!((0.004..=0.020).contains(&p.chord_lag_fraction));
            assert!((0.0..=0.50).contains(&p.ribbon_echo_alpha));
            assert!((0.0..=0.18).contains(&p.halation_strength));
            assert!((0.0..=0.30).contains(&p.prism_strength));
            assert!((0.0..=0.09).contains(&p.nebula_whisper_strength));

            let (lw_min, lw_max) = line_weight_range(p.structure);
            assert!(
                (lw_min..=lw_max).contains(&p.line_weight),
                "line weight {} outside mode range [{lw_min}, {lw_max}] for {}",
                p.line_weight,
                p.structure.label()
            );
        }
    }

    #[test]
    fn cosmic_signature_records_profile_randomization() {
        let mut rng = make_rng(&[1, 2, 3, 4]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);

        assert_eq!(profile.name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects.len(), 1);
        assert_eq!(profile.randomization_log.effects[0].effect_name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects[0].parameters.len(), 15);
    }

    #[test]
    fn scene_traits_default_matches_classic_look() {
        let traits = SceneTraits::default();
        assert_eq!(traits.structure, StructureMode::TriangleWeb);
        assert_eq!(traits.line_weight, 1.0);
        assert_eq!(traits.age_ramp, 0.0);
        assert_eq!(traits.edge_energy, 1.0);
        assert_eq!(traits.ribbon_echo_alpha, 0.0);
        assert!(!traits.mirror);
    }

    #[test]
    fn cosmic_signature_finish_pipeline_matches_gated_traits() {
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
            // trajectory chain contains exactly the gated halation bloom plus
            // the rare prism pass when those traits are active.
            assert_eq!(pipeline.image_len(), 0);
            let expected = usize::from(profile.parameters.halation_strength > 0.0)
                + usize::from(profile.parameters.prism_strength > 0.0);
            assert_eq!(
                pipeline.trajectory_len(),
                expected,
                "trajectory chain length must match gated traits"
            );
            if profile.parameters.halation_strength > 0.0 {
                saw_halation = true;
            } else if expected == 0 {
                saw_empty = true;
            }
        }

        assert!(saw_empty, "expected at least one seed without finish passes");
        assert!(saw_halation, "expected at least one seed with halation");
    }
}
