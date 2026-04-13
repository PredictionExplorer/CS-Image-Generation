//! Spectral effect modes for creative post-processing of per-bin images.
//!
//! Each mode defines a unique strategy for applying the post-effects pipeline
//! to the 64 spectral bin images and cycle video frames, producing visually
//! distinct variants of the spectral gallery and cycle videos.

use super::constants;
use super::effects::{DogBloomConfig, EffectConfig, FinishEffectPipeline, FrameParams};
use crate::post_effects::{
    AetherConfig, AtmosphericDepthConfig, ChampleveConfig, ChromaticBloomConfig, ColorGradeParams,
    EdgeLuminanceConfig, FineTextureConfig, GlowEnhancementConfig, GradientMapConfig,
    LuxuryPalette, MicroContrastConfig, OpalescenceConfig, PerceptualBlurConfig,
};
use crate::sim::Sha3RandomByteStream;
use crate::spectrum::NUM_BINS;
use rayon::prelude::*;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// SpectralEffectMode
// ---------------------------------------------------------------------------

/// Identifies one of the ten spectral output variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpectralEffectMode {
    Default,
    Dispersion,
    Resonance,
    Weathering,
    Fingerprint,
    Interference,
    Cascade,
    Masking,
    Gravity,
    Chaos,
}

impl SpectralEffectMode {
    /// All ten modes in a fixed order (Default first).
    pub const ALL: &'static [SpectralEffectMode] = &[
        Self::Default,
        Self::Dispersion,
        Self::Resonance,
        Self::Weathering,
        Self::Fingerprint,
        Self::Interference,
        Self::Cascade,
        Self::Masking,
        Self::Gravity,
        Self::Chaos,
    ];

    /// Subdirectory name for this mode's outputs.
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Dispersion => "dispersion",
            Self::Resonance => "resonance",
            Self::Weathering => "weathering",
            Self::Fingerprint => "fingerprint",
            Self::Interference => "interference",
            Self::Cascade => "cascade",
            Self::Masking => "masking",
            Self::Gravity => "gravity",
            Self::Chaos => "chaos",
        }
    }

    /// Human-readable label.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Default => "Default (raw)",
            Self::Dispersion => "Chromatic Dispersion",
            Self::Resonance => "Resonance Bins",
            Self::Weathering => "Spectral Weathering",
            Self::Fingerprint => "Stochastic Fingerprint",
            Self::Interference => "Interference Pattern",
            Self::Cascade => "Cascade Reveal",
            Self::Masking => "Cross-Spectral Masking",
            Self::Gravity => "Spectral Gravity Well",
            Self::Chaos => "Maximum Chaos",
        }
    }

    /// Whether this mode requires an RNG for planning.
    pub fn needs_rng(self) -> bool {
        matches!(self, Self::Resonance | Self::Fingerprint | Self::Gravity | Self::Chaos)
    }
}

// ---------------------------------------------------------------------------
// EnabledEffects
// ---------------------------------------------------------------------------

/// Per-effect enable flags for a single spectral bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnabledEffects {
    pub bloom: bool,
    pub dog_bloom: bool,
    pub glow: bool,
    pub chromatic_bloom: bool,
    pub perceptual_blur: bool,
    pub micro_contrast: bool,
    pub gradient_map: bool,
    pub color_grade: bool,
    pub opalescence: bool,
    pub champleve: bool,
    pub aether: bool,
    pub edge_luminance: bool,
    pub atmospheric_depth: bool,
    pub fine_texture: bool,
}

impl EnabledEffects {
    pub fn all() -> Self {
        Self {
            bloom: true,
            dog_bloom: true,
            glow: true,
            chromatic_bloom: true,
            perceptual_blur: true,
            micro_contrast: true,
            gradient_map: true,
            color_grade: true,
            opalescence: true,
            champleve: true,
            aether: true,
            edge_luminance: true,
            atmospheric_depth: true,
            fine_texture: true,
        }
    }

    pub fn none() -> Self {
        Self {
            bloom: false,
            dog_bloom: false,
            glow: false,
            chromatic_bloom: false,
            perceptual_blur: false,
            micro_contrast: false,
            gradient_map: false,
            color_grade: false,
            opalescence: false,
            champleve: false,
            aether: false,
            edge_luminance: false,
            atmospheric_depth: false,
            fine_texture: false,
        }
    }

    /// Gaseous zone: atmosphere, blur, glow.
    pub fn gaseous() -> Self {
        Self {
            atmospheric_depth: true,
            perceptual_blur: true,
            glow: true,
            bloom: true,
            ..Self::none()
        }
    }

    /// Metallic zone: structure, texture, contrast, edges.
    pub fn metallic() -> Self {
        Self {
            champleve: true,
            fine_texture: true,
            micro_contrast: true,
            edge_luminance: true,
            ..Self::none()
        }
    }

    /// Luminous zone: aether, opalescence, chromatic bloom, dog bloom.
    pub fn luminous() -> Self {
        Self {
            aether: true,
            opalescence: true,
            chromatic_bloom: true,
            dog_bloom: true,
            ..Self::none()
        }
    }

    /// Count how many effects are enabled.
    pub fn count(self) -> usize {
        [
            self.bloom,
            self.dog_bloom,
            self.glow,
            self.chromatic_bloom,
            self.perceptual_blur,
            self.micro_contrast,
            self.gradient_map,
            self.color_grade,
            self.opalescence,
            self.champleve,
            self.aether,
            self.edge_luminance,
            self.atmospheric_depth,
            self.fine_texture,
        ]
        .iter()
        .filter(|&&v| v)
        .count()
    }

    /// Total number of available effects.
    pub const TOTAL_EFFECTS: usize = 14;

    /// Enable the effect at the given index (0..14).
    pub fn set_by_index(&mut self, index: usize, value: bool) {
        match index {
            0 => self.bloom = value,
            1 => self.dog_bloom = value,
            2 => self.glow = value,
            3 => self.chromatic_bloom = value,
            4 => self.perceptual_blur = value,
            5 => self.micro_contrast = value,
            6 => self.gradient_map = value,
            7 => self.color_grade = value,
            8 => self.opalescence = value,
            9 => self.champleve = value,
            10 => self.aether = value,
            11 => self.edge_luminance = value,
            12 => self.atmospheric_depth = value,
            13 => self.fine_texture = value,
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// BinEffectPlan
// ---------------------------------------------------------------------------

/// Describes how effects should be applied to a single spectral bin.
#[derive(Clone, Debug)]
pub struct BinEffectPlan {
    /// Overall effect strength (0.0 = raw, 1.0 = full).
    pub strength: f64,
    /// Multiplier on the default bloom/DoG radius.
    pub bloom_radius_scale: f64,
    /// Which effects are enabled for this bin.
    pub enabled_effects: EnabledEffects,
    /// When set, used directly instead of modifying the base config.
    /// Used by Chaos mode which fully randomizes every parameter.
    pub custom_config: Option<Box<EffectConfig>>,
}

impl BinEffectPlan {
    pub fn raw() -> Self {
        Self {
            strength: 0.0,
            bloom_radius_scale: 1.0,
            enabled_effects: EnabledEffects::none(),
            custom_config: None,
        }
    }

    pub fn full() -> Self {
        Self {
            strength: 1.0,
            bloom_radius_scale: 1.0,
            enabled_effects: EnabledEffects::all(),
            custom_config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Planning functions
// ---------------------------------------------------------------------------

/// Plan for Default mode: no effects, raw bin images.
pub(crate) fn plan_default() -> Vec<BinEffectPlan> {
    vec![BinEffectPlan::raw(); NUM_BINS]
}

/// Chromatic dispersion: bloom radius scales inversely with wavelength.
/// Violet (bin 0) gets 3x bloom, red (bin 63) gets 1x.
pub(crate) fn plan_dispersion() -> Vec<BinEffectPlan> {
    (0..NUM_BINS)
        .map(|bin| {
            let t = bin as f64 / (NUM_BINS - 1) as f64;
            BinEffectPlan {
                strength: 1.0,
                bloom_radius_scale: constants::SPECTRAL_DISPERSION_BLOOM_MAX
                    - (constants::SPECTRAL_DISPERSION_BLOOM_MAX
                        - constants::SPECTRAL_DISPERSION_BLOOM_MIN)
                        * t,
                enabled_effects: EnabledEffects::all(),
                custom_config: None,
            }
        })
        .collect()
}

/// Resonance bins: RNG picks 5-8 bins for full effects; others get minimal treatment.
pub(crate) fn plan_resonance(rng: &mut Sha3RandomByteStream) -> Vec<BinEffectPlan> {
    let count_range =
        constants::SPECTRAL_RESONANCE_MAX_BINS - constants::SPECTRAL_RESONANCE_MIN_BINS + 1;
    let num_resonance =
        constants::SPECTRAL_RESONANCE_MIN_BINS + (rng.next_byte() as usize % count_range);

    let mut resonance_bins = Vec::with_capacity(num_resonance);
    while resonance_bins.len() < num_resonance {
        let candidate = rng.next_byte() as usize % NUM_BINS;
        if !resonance_bins.contains(&candidate) {
            resonance_bins.push(candidate);
        }
    }

    (0..NUM_BINS)
        .map(|bin| {
            if resonance_bins.contains(&bin) {
                BinEffectPlan::full()
            } else {
                BinEffectPlan {
                    strength: 0.15,
                    bloom_radius_scale: 1.0,
                    enabled_effects: EnabledEffects {
                        micro_contrast: true,
                        edge_luminance: true,
                        ..EnabledEffects::none()
                    },
                    custom_config: None,
                }
            }
        })
        .collect()
}

/// Spectral weathering: three material zones across the spectrum.
pub(crate) fn plan_weathering() -> Vec<BinEffectPlan> {
    let [z1, z2] = constants::SPECTRAL_WEATHERING_ZONE_BOUNDARIES;
    (0..NUM_BINS)
        .map(|bin| {
            let enabled = if bin < z1 {
                EnabledEffects::gaseous()
            } else if bin < z2 {
                EnabledEffects::metallic()
            } else {
                EnabledEffects::luminous()
            };
            BinEffectPlan {
                strength: 1.0,
                bloom_radius_scale: 1.0,
                enabled_effects: enabled,
                custom_config: None,
            }
        })
        .collect()
}

/// Stochastic fingerprint: each bin gets a unique random subset of 2-4 effects.
pub(crate) fn plan_fingerprint(rng: &mut Sha3RandomByteStream) -> Vec<BinEffectPlan> {
    let min_fx = constants::SPECTRAL_FINGERPRINT_MIN_EFFECTS;
    let max_fx = constants::SPECTRAL_FINGERPRINT_MAX_EFFECTS;
    let fx_range = max_fx - min_fx + 1;

    (0..NUM_BINS)
        .map(|_| {
            let num_effects = min_fx + (rng.next_byte() as usize % fx_range);
            let mut enabled = EnabledEffects::none();
            let mut chosen = 0;
            while chosen < num_effects {
                let idx = rng.next_byte() as usize % EnabledEffects::TOTAL_EFFECTS;
                enabled.set_by_index(idx, true);
                chosen = enabled.count();
            }
            BinEffectPlan {
                strength: 0.85,
                bloom_radius_scale: 1.0,
                enabled_effects: enabled,
                custom_config: None,
            }
        })
        .collect()
}

/// Sinusoidal interference: effect strength oscillates across the spectrum.
pub(crate) fn plan_interference() -> Vec<BinEffectPlan> {
    let freq = constants::SPECTRAL_INTERFERENCE_FREQUENCY;
    (0..NUM_BINS)
        .map(|bin| {
            let phase = 2.0 * PI * bin as f64 * freq / NUM_BINS as f64;
            let strength = 0.5 + 0.5 * phase.sin();
            BinEffectPlan {
                strength,
                bloom_radius_scale: 1.0,
                enabled_effects: EnabledEffects::all(),
                custom_config: None,
            }
        })
        .collect()
}

/// Cascade: progressive effect stacking across the bin spectrum.
/// Gallery: strength = bin / 63. Cycle videos use frame-based scaling.
pub(crate) fn plan_cascade() -> Vec<BinEffectPlan> {
    (0..NUM_BINS)
        .map(|bin| {
            let strength = bin as f64 / (NUM_BINS - 1) as f64;
            BinEffectPlan {
                strength,
                bloom_radius_scale: 1.0,
                enabled_effects: EnabledEffects::all(),
                custom_config: None,
            }
        })
        .collect()
}

/// Cross-spectral masking: all bins processed at full strength, then blended
/// with raw data using complementary bin intensity as the mask.
pub(crate) fn plan_masking() -> Vec<BinEffectPlan> {
    vec![BinEffectPlan::full(); NUM_BINS]
}

/// Gravity well: Gaussian falloff from a seed-random attractor bin.
pub(crate) fn plan_gravity(rng: &mut Sha3RandomByteStream) -> Vec<BinEffectPlan> {
    let attractor = rng.next_byte() as usize % NUM_BINS;
    let sigma = constants::SPECTRAL_GRAVITY_SIGMA;
    let two_sigma_sq = 2.0 * sigma * sigma;

    (0..NUM_BINS)
        .map(|bin| {
            let dist = (bin as f64 - attractor as f64).abs();
            let strength = (-dist * dist / two_sigma_sq).exp();
            BinEffectPlan {
                strength,
                bloom_radius_scale: 1.0,
                enabled_effects: EnabledEffects::all(),
                custom_config: None,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Chaos mode: maximum per-bin randomization
// ---------------------------------------------------------------------------

/// RNG helper: random f64 in `[lo, hi]`.
fn rand_range(rng: &mut Sha3RandomByteStream, lo: f64, hi: f64) -> f64 {
    rng.next_f64() * (hi - lo) + lo
}

/// RNG helper: random usize in `[lo, hi]` inclusive.
fn rand_usize(rng: &mut Sha3RandomByteStream, lo: usize, hi: usize) -> usize {
    lo + (rng.next_byte() as usize % (hi - lo + 1))
}

fn randomize_dog_bloom(rng: &mut Sha3RandomByteStream, _base: &DogBloomConfig) -> DogBloomConfig {
    DogBloomConfig {
        inner_sigma: rand_range(rng, 2.0, 12.0),
        outer_ratio: rand_range(rng, 1.5, 3.5),
        strength: rand_range(rng, 0.1, 0.9),
        threshold: rand_range(rng, 0.005, 0.04),
    }
}

fn randomize_glow(
    rng: &mut Sha3RandomByteStream,
    _base: &GlowEnhancementConfig,
) -> GlowEnhancementConfig {
    GlowEnhancementConfig {
        strength: rand_range(rng, 0.2, 0.8),
        threshold: rand_range(rng, 0.3, 0.9),
        radius: rand_usize(rng, 3, 18),
        sharpness: rand_range(rng, 1.5, 4.0),
        saturation_boost: rand_range(rng, 0.0, 0.6),
    }
}

fn randomize_chromatic_bloom(
    rng: &mut Sha3RandomByteStream,
    base: &ChromaticBloomConfig,
) -> ChromaticBloomConfig {
    let scale = rand_range(rng, 0.5, 2.0);
    ChromaticBloomConfig {
        strength: rand_range(rng, 0.3, 1.0),
        separation: base.separation * scale,
        threshold: rand_range(rng, 0.05, 0.35),
        radius: (base.radius as f64 * rand_range(rng, 0.5, 2.0)).round() as usize,
    }
}

fn randomize_perceptual_blur(
    rng: &mut Sha3RandomByteStream,
    base: &PerceptualBlurConfig,
) -> PerceptualBlurConfig {
    PerceptualBlurConfig {
        strength: rand_range(rng, 0.2, 0.9),
        radius: (base.radius as f64 * rand_range(rng, 0.5, 2.0)).round().max(1.0) as usize,
        ..*base
    }
}

fn randomize_micro_contrast(
    rng: &mut Sha3RandomByteStream,
    _base: &MicroContrastConfig,
) -> MicroContrastConfig {
    MicroContrastConfig {
        strength: rand_range(rng, 0.1, 0.7),
        radius: rand_usize(rng, 1, 5),
        edge_threshold: rand_range(rng, 0.05, 0.25),
        luminance_weight: rand_range(rng, 0.3, 0.9),
    }
}

fn randomize_gradient_map(
    rng: &mut Sha3RandomByteStream,
    _base: &GradientMapConfig,
) -> GradientMapConfig {
    GradientMapConfig {
        palette: LuxuryPalette::from_index(rand_usize(rng, 0, 14)),
        strength: rand_range(rng, 0.2, 0.9),
        hue_preservation: rand_range(rng, 0.0, 0.6),
    }
}

fn randomize_color_grade(
    rng: &mut Sha3RandomByteStream,
    base: &ColorGradeParams,
) -> ColorGradeParams {
    ColorGradeParams {
        strength: rand_range(rng, 0.1, 0.6),
        vignette_strength: rand_range(rng, 0.0, 0.6),
        vignette_softness: rand_range(rng, 1.5, 3.5),
        vibrance: rand_range(rng, 0.7, 1.4),
        clarity_strength: rand_range(rng, 0.1, 0.5),
        tone_curve: rand_range(rng, 0.2, 0.8),
        shadow_tint: [
            rand_range(rng, -0.12, 0.04),
            rand_range(rng, -0.06, 0.04),
            rand_range(rng, -0.04, 0.2),
        ],
        highlight_tint: [
            rand_range(rng, -0.04, 0.15),
            rand_range(rng, -0.04, 0.08),
            rand_range(rng, -0.08, 0.04),
        ],
        ..*base
    }
}

fn randomize_opalescence(
    rng: &mut Sha3RandomByteStream,
    base: &OpalescenceConfig,
) -> OpalescenceConfig {
    OpalescenceConfig {
        strength: rand_range(rng, 0.05, 0.4),
        layers: rand_usize(rng, 2, 4),
        chromatic_shift: rand_range(rng, 0.1, 0.7),
        pearl_sheen: rand_range(rng, 0.05, 0.5),
        angle_sensitivity: rand_range(rng, 0.5, 1.5),
        ..*base
    }
}

fn randomize_champleve(rng: &mut Sha3RandomByteStream, base: &ChampleveConfig) -> ChampleveConfig {
    ChampleveConfig {
        cell_density: rand_range(rng, 20.0, 100.0),
        flow_alignment: rand_range(rng, 0.3, 1.0),
        interference_amplitude: rand_range(rng, 0.2, 1.0),
        interference_frequency: rand_range(rng, 10.0, 50.0),
        rim_intensity: rand_range(rng, 0.5, 3.5),
        rim_warmth: rand_range(rng, 0.3, 1.0),
        interior_lift: rand_range(rng, 0.3, 1.0),
        ..*base
    }
}

fn randomize_aether(rng: &mut Sha3RandomByteStream, base: &AetherConfig) -> AetherConfig {
    AetherConfig {
        filament_density: rand_range(rng, 40.0, 150.0),
        flow_alignment: rand_range(rng, 0.4, 1.0),
        scattering_strength: rand_range(rng, 0.3, 2.0),
        scattering_falloff: rand_range(rng, 1.5, 4.0),
        iridescence_amplitude: rand_range(rng, 0.2, 1.0),
        iridescence_frequency: rand_range(rng, 6.0, 20.0),
        caustic_strength: rand_range(rng, 0.1, 0.7),
        caustic_softness: rand_range(rng, 1.5, 5.0),
        ..*base
    }
}

fn randomize_edge_luminance(
    rng: &mut Sha3RandomByteStream,
    base: &EdgeLuminanceConfig,
) -> EdgeLuminanceConfig {
    EdgeLuminanceConfig {
        strength: rand_range(rng, 0.1, 0.5),
        threshold: rand_range(rng, 0.05, 0.35),
        brightness_boost: rand_range(rng, 0.1, 0.6),
        min_luminance: rand_range(rng, 0.1, 0.4),
        ..*base
    }
}

fn randomize_atmospheric_depth(
    rng: &mut Sha3RandomByteStream,
    base: &AtmosphericDepthConfig,
) -> AtmosphericDepthConfig {
    AtmosphericDepthConfig {
        strength: rand_range(rng, 0.1, 0.5),
        fog_color: (
            rand_range(rng, 0.0, 0.3),
            rand_range(rng, 0.0, 0.3),
            rand_range(rng, 0.0, 0.3),
        ),
        desaturation: rand_range(rng, 0.1, 0.7),
        darkening: rand_range(rng, 0.05, 0.4),
        ..*base
    }
}

fn randomize_fine_texture(
    rng: &mut Sha3RandomByteStream,
    base: &FineTextureConfig,
) -> FineTextureConfig {
    FineTextureConfig {
        strength: rand_range(rng, 0.05, 0.3),
        contrast: rand_range(rng, 0.1, 0.6),
        anisotropy: rand_range(rng, 0.0, 0.5),
        angle: rand_range(rng, 0.0, 180.0),
        ..*base
    }
}

/// Maximum chaos: every bin gets an independent coin-flip for each effect
/// plus fully randomized numeric parameters.
pub(crate) fn plan_chaos(
    rng: &mut Sha3RandomByteStream,
    base_config: &EffectConfig,
) -> Vec<BinEffectPlan> {
    (0..NUM_BINS)
        .map(|_| {
            let strength = rand_range(
                rng,
                constants::SPECTRAL_CHAOS_STRENGTH_MIN,
                constants::SPECTRAL_CHAOS_STRENGTH_MAX,
            );
            let bloom_radius_scale = rand_range(
                rng,
                constants::SPECTRAL_CHAOS_BLOOM_SCALE_MIN,
                constants::SPECTRAL_CHAOS_BLOOM_SCALE_MAX,
            );

            let mut enabled = EnabledEffects::none();
            for idx in 0..EnabledEffects::TOTAL_EFFECTS {
                enabled.set_by_index(idx, rng.next_byte() >= 128);
            }

            let mut cfg = base_config.clone();

            if enabled.bloom {
                cfg.blur_radius_px =
                    (cfg.blur_radius_px as f64 * bloom_radius_scale).round() as usize;
            } else {
                cfg.blur_radius_px = 0;
            }

            if enabled.dog_bloom {
                cfg.dog_config = randomize_dog_bloom(rng, &cfg.dog_config);
                cfg.dog_config.inner_sigma *= bloom_radius_scale;
            } else {
                cfg.bloom_mode = "none".to_string();
            }

            cfg.glow_enhancement_enabled = enabled.glow;
            if enabled.glow {
                cfg.glow_enhancement_config = randomize_glow(rng, &cfg.glow_enhancement_config);
            }

            cfg.chromatic_bloom_enabled = enabled.chromatic_bloom;
            if enabled.chromatic_bloom {
                cfg.chromatic_bloom_config =
                    randomize_chromatic_bloom(rng, &cfg.chromatic_bloom_config);
            }

            cfg.perceptual_blur_enabled = enabled.perceptual_blur;
            if enabled.perceptual_blur
                && let Some(ref blur_cfg) = cfg.perceptual_blur_config
            {
                cfg.perceptual_blur_config = Some(randomize_perceptual_blur(rng, blur_cfg));
            }

            cfg.micro_contrast_enabled = enabled.micro_contrast;
            if enabled.micro_contrast {
                cfg.micro_contrast_config =
                    randomize_micro_contrast(rng, &cfg.micro_contrast_config);
            }

            cfg.gradient_map_enabled = enabled.gradient_map;
            if enabled.gradient_map {
                cfg.gradient_map_config = randomize_gradient_map(rng, &cfg.gradient_map_config);
            }

            cfg.color_grade_enabled = enabled.color_grade;
            if enabled.color_grade {
                cfg.color_grade_params = randomize_color_grade(rng, &cfg.color_grade_params);
            }

            cfg.opalescence_enabled = enabled.opalescence;
            if enabled.opalescence {
                cfg.opalescence_config = randomize_opalescence(rng, &cfg.opalescence_config);
            }

            cfg.champleve_enabled = enabled.champleve;
            if enabled.champleve {
                cfg.champleve_config = randomize_champleve(rng, &cfg.champleve_config);
            }

            cfg.aether_enabled = enabled.aether;
            if enabled.aether {
                cfg.aether_config = randomize_aether(rng, &cfg.aether_config);
            }

            cfg.edge_luminance_enabled = enabled.edge_luminance;
            if enabled.edge_luminance {
                cfg.edge_luminance_config =
                    randomize_edge_luminance(rng, &cfg.edge_luminance_config);
            }

            cfg.atmospheric_depth_enabled = enabled.atmospheric_depth;
            if enabled.atmospheric_depth {
                cfg.atmospheric_depth_config =
                    randomize_atmospheric_depth(rng, &cfg.atmospheric_depth_config);
            }

            cfg.fine_texture_enabled = enabled.fine_texture;
            if enabled.fine_texture {
                cfg.fine_texture_config = randomize_fine_texture(rng, &cfg.fine_texture_config);
            }

            BinEffectPlan {
                strength,
                bloom_radius_scale,
                enabled_effects: enabled,
                custom_config: Some(Box::new(cfg)),
            }
        })
        .collect()
}

/// Generate the full bin plan for any mode.
///
/// `base_config` is required only for `Chaos` mode which fully randomizes
/// per-effect parameters; other modes ignore it.
pub(crate) fn plan_for_mode(
    mode: SpectralEffectMode,
    rng: &mut Sha3RandomByteStream,
    base_config: Option<&EffectConfig>,
) -> Vec<BinEffectPlan> {
    match mode {
        SpectralEffectMode::Default => plan_default(),
        SpectralEffectMode::Dispersion => plan_dispersion(),
        SpectralEffectMode::Resonance => plan_resonance(rng),
        SpectralEffectMode::Weathering => plan_weathering(),
        SpectralEffectMode::Fingerprint => plan_fingerprint(rng),
        SpectralEffectMode::Interference => plan_interference(),
        SpectralEffectMode::Cascade => plan_cascade(),
        SpectralEffectMode::Masking => plan_masking(),
        SpectralEffectMode::Gravity => plan_gravity(rng),
        SpectralEffectMode::Chaos => {
            let cfg = base_config.expect("Chaos mode requires a base EffectConfig");
            plan_chaos(rng, cfg)
        }
    }
}

// ---------------------------------------------------------------------------
// Effect config construction
// ---------------------------------------------------------------------------

/// Build a per-bin `EffectConfig` from a base config and a `BinEffectPlan`.
///
/// When the plan carries a `custom_config` (Chaos mode), it is used directly.
/// Otherwise, toggles each effect's enable flag according to the plan, and
/// scales the bloom / DoG radius by `plan.bloom_radius_scale`.
pub fn build_bin_effect_config(base: &EffectConfig, plan: &BinEffectPlan) -> EffectConfig {
    if let Some(custom) = &plan.custom_config {
        return *custom.clone();
    }

    let mut cfg = base.clone();
    let fx = &plan.enabled_effects;

    if !fx.bloom {
        cfg.blur_radius_px = 0;
    } else {
        cfg.blur_radius_px = (cfg.blur_radius_px as f64 * plan.bloom_radius_scale).round() as usize;
    }

    if !fx.dog_bloom {
        cfg.bloom_mode = "none".to_string();
    } else {
        cfg.dog_config.inner_sigma *= plan.bloom_radius_scale;
    }

    cfg.glow_enhancement_enabled = cfg.glow_enhancement_enabled && fx.glow;
    cfg.chromatic_bloom_enabled = cfg.chromatic_bloom_enabled && fx.chromatic_bloom;
    cfg.perceptual_blur_enabled = cfg.perceptual_blur_enabled && fx.perceptual_blur;
    cfg.micro_contrast_enabled = cfg.micro_contrast_enabled && fx.micro_contrast;
    cfg.gradient_map_enabled = cfg.gradient_map_enabled && fx.gradient_map;
    cfg.color_grade_enabled = cfg.color_grade_enabled && fx.color_grade;
    cfg.opalescence_enabled = cfg.opalescence_enabled && fx.opalescence;
    cfg.champleve_enabled = cfg.champleve_enabled && fx.champleve;
    cfg.aether_enabled = cfg.aether_enabled && fx.aether;
    cfg.edge_luminance_enabled = cfg.edge_luminance_enabled && fx.edge_luminance;
    cfg.atmospheric_depth_enabled = cfg.atmospheric_depth_enabled && fx.atmospheric_depth;
    cfg.fine_texture_enabled = cfg.fine_texture_enabled && fx.fine_texture;

    cfg
}

// ---------------------------------------------------------------------------
// Effect application
// ---------------------------------------------------------------------------

/// Convert a linear-space `[f32; 3]` bin image to a `PixelBuffer` for the
/// effect pipeline. Alpha is set to the pixel's luminance.
pub fn bin_to_pixel_buffer(pixels: &[[f32; 3]]) -> Vec<(f64, f64, f64, f64)> {
    pixels
        .iter()
        .map(|p| {
            let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let a = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            (r, g, b, a)
        })
        .collect()
}

/// Convert a `PixelBuffer` back to linear `[f32; 3]`.
pub fn pixel_buffer_to_bin(pixels: &[(f64, f64, f64, f64)]) -> Vec<[f32; 3]> {
    pixels
        .iter()
        .map(|&(r, g, b, _)| {
            [r.clamp(0.0, 1.0) as f32, g.clamp(0.0, 1.0) as f32, b.clamp(0.0, 1.0) as f32]
        })
        .collect()
}

/// Apply the post-effects pipeline to a single linear-space bin image.
///
/// When `plan.strength` is 0.0 the input is returned unmodified.
/// When `plan.strength` is between 0.0 and 1.0 the result is a linear blend
/// of the raw input and the fully processed output.
pub fn apply_effects_to_bin_linear(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    plan: &BinEffectPlan,
    base_config: &EffectConfig,
) -> Vec<[f32; 3]> {
    if plan.strength <= 0.0 || plan.enabled_effects.count() == 0 {
        return pixels.to_vec();
    }

    let cfg = build_bin_effect_config(base_config, plan);
    let pipeline = FinishEffectPipeline::new(cfg);
    let buffer = bin_to_pixel_buffer(pixels);
    let frame_params = FrameParams { frame_number: 0, density: None };

    let processed = match pipeline.process_trajectory(buffer, width, height, &frame_params) {
        Ok(buf) => pixel_buffer_to_bin(&buf),
        Err(_) => return pixels.to_vec(),
    };

    if (plan.strength - 1.0).abs() < 1e-9 {
        return processed;
    }

    let s = plan.strength as f32;
    let inv = 1.0 - s;
    pixels
        .iter()
        .zip(processed.iter())
        .map(|(raw, fx)| {
            [raw[0] * inv + fx[0] * s, raw[1] * inv + fx[1] * s, raw[2] * inv + fx[2] * s]
        })
        .collect()
}

/// Apply display gamma and save a linear `[f32; 3]` image as a 16-bit PNG.
pub fn apply_gamma_and_save(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    path: &str,
) -> super::error::Result<()> {
    let inv_gamma = 1.0 / constants::DISPLAY_GAMMA;
    let pixel_count = (width * height) as usize;
    let mut raw = Vec::with_capacity(pixel_count * 3);

    for p in pixels {
        raw.push(((p[0].clamp(0.0, 1.0) as f64).powf(inv_gamma) * 65535.0).round() as u16);
        raw.push(((p[1].clamp(0.0, 1.0) as f64).powf(inv_gamma) * 65535.0).round() as u16);
        raw.push(((p[2].clamp(0.0, 1.0) as f64).powf(inv_gamma) * 65535.0).round() as u16);
    }

    let img: image::ImageBuffer<image::Rgb<u16>, Vec<u16>> =
        image::ImageBuffer::from_raw(width, height, raw).ok_or_else(|| {
            super::error::RenderError::InvalidConfig(
                "Failed to create effect bin image buffer".to_string(),
            )
        })?;

    image::DynamicImage::ImageRgb16(img)
        .save(path)
        .map_err(|e| super::error::RenderError::ImageEncoding(e.to_string()))?;
    Ok(())
}

/// Apply effects to a cycle video frame (raw 16-bit RGB buffer).
///
/// `frame_pixels_f32` is the float-space frame (linear, pre-gamma).
/// Returns a gamma-corrected 16-bit buffer ready for encoding.
pub fn apply_effects_to_cycle_frame(
    frame_pixels_f32: &[[f32; 3]],
    width: usize,
    height: usize,
    plan: &BinEffectPlan,
    base_config: &EffectConfig,
) -> Vec<u16> {
    let processed = apply_effects_to_bin_linear(frame_pixels_f32, width, height, plan, base_config);
    let inv_gamma = 1.0 / constants::DISPLAY_GAMMA;

    let mut out = Vec::with_capacity(processed.len() * 3);
    for p in &processed {
        out.push(((p[0].clamp(0.0, 1.0) as f64).powf(inv_gamma) * 65535.0).round() as u16);
        out.push(((p[1].clamp(0.0, 1.0) as f64).powf(inv_gamma) * 65535.0).round() as u16);
        out.push(((p[2].clamp(0.0, 1.0) as f64).powf(inv_gamma) * 65535.0).round() as u16);
    }
    out
}

/// For Masking mode, blend a processed bin image with its raw version
/// using the complementary bin's intensity as the per-pixel mask.
pub fn apply_masking_blend(
    raw: &[[f32; 3]],
    processed: &[[f32; 3]],
    complement_raw: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    raw.par_iter()
        .zip(processed.par_iter())
        .zip(complement_raw.par_iter())
        .map(|((r, p), c)| {
            let mask = (0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
                .clamp(0.0, 1.0) as f32;
            let inv_mask = 1.0 - mask;
            [
                r[0] * mask + p[0] * inv_mask,
                r[1] * mask + p[1] * inv_mask,
                r[2] * mask + p[2] * inv_mask,
            ]
        })
        .collect()
}

/// Determine the `BinEffectPlan` to use for a cycle video frame.
///
/// For Cascade mode, overrides the plan strength with frame-based progression.
/// For other modes, picks the plan of the nearest integer bin.
pub fn cycle_frame_plan(
    mode: SpectralEffectMode,
    plans: &[BinEffectPlan],
    current_bin_f: f64,
    frame_index: u32,
    total_frames: u32,
) -> BinEffectPlan {
    match mode {
        SpectralEffectMode::Default => BinEffectPlan::raw(),
        SpectralEffectMode::Cascade => {
            let strength = frame_index as f64 / total_frames.max(1) as f64;
            BinEffectPlan {
                strength,
                bloom_radius_scale: 1.0,
                enabled_effects: EnabledEffects::all(),
                custom_config: None,
            }
        }
        _ => {
            let wrapped = ((current_bin_f % NUM_BINS as f64) + NUM_BINS as f64) % NUM_BINS as f64;
            let nearest = (wrapped.round() as usize).min(NUM_BINS - 1);
            plans[nearest].clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mode enum ──────────────────────────────────────────────────

    #[test]
    fn test_all_returns_ten_modes() {
        assert_eq!(SpectralEffectMode::ALL.len(), 10);
    }

    #[test]
    fn test_default_is_first() {
        assert_eq!(SpectralEffectMode::ALL[0], SpectralEffectMode::Default);
    }

    #[test]
    fn test_dir_names_unique_and_nonempty() {
        let names: Vec<&str> = SpectralEffectMode::ALL.iter().map(|m| m.dir_name()).collect();
        for name in &names {
            assert!(!name.is_empty());
        }
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "dir_name() must be unique across modes");
    }

    #[test]
    fn test_display_names_unique() {
        let names: Vec<&str> = SpectralEffectMode::ALL.iter().map(|m| m.display_name()).collect();
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn test_needs_rng_correct() {
        assert!(!SpectralEffectMode::Default.needs_rng());
        assert!(!SpectralEffectMode::Dispersion.needs_rng());
        assert!(SpectralEffectMode::Resonance.needs_rng());
        assert!(!SpectralEffectMode::Weathering.needs_rng());
        assert!(SpectralEffectMode::Fingerprint.needs_rng());
        assert!(!SpectralEffectMode::Interference.needs_rng());
        assert!(!SpectralEffectMode::Cascade.needs_rng());
        assert!(!SpectralEffectMode::Masking.needs_rng());
        assert!(SpectralEffectMode::Gravity.needs_rng());
        assert!(SpectralEffectMode::Chaos.needs_rng());
    }

    // ── EnabledEffects ─────────────────────────────────────────────

    #[test]
    fn test_enabled_effects_all_count() {
        assert_eq!(EnabledEffects::all().count(), EnabledEffects::TOTAL_EFFECTS);
    }

    #[test]
    fn test_enabled_effects_none_count() {
        assert_eq!(EnabledEffects::none().count(), 0);
    }

    #[test]
    fn test_enabled_effects_set_by_index_roundtrip() {
        for idx in 0..EnabledEffects::TOTAL_EFFECTS {
            let mut fx = EnabledEffects::none();
            fx.set_by_index(idx, true);
            assert_eq!(fx.count(), 1, "set_by_index({idx}) should enable exactly one effect");
        }
    }

    #[test]
    fn test_gaseous_has_expected_effects() {
        let g = EnabledEffects::gaseous();
        assert!(g.atmospheric_depth);
        assert!(g.perceptual_blur);
        assert!(g.glow);
        assert!(g.bloom);
        assert!(!g.champleve);
        assert!(!g.aether);
    }

    #[test]
    fn test_metallic_has_expected_effects() {
        let m = EnabledEffects::metallic();
        assert!(m.champleve);
        assert!(m.fine_texture);
        assert!(m.micro_contrast);
        assert!(m.edge_luminance);
        assert!(!m.bloom);
        assert!(!m.opalescence);
    }

    #[test]
    fn test_luminous_has_expected_effects() {
        let l = EnabledEffects::luminous();
        assert!(l.aether);
        assert!(l.opalescence);
        assert!(l.chromatic_bloom);
        assert!(l.dog_bloom);
        assert!(!l.fine_texture);
        assert!(!l.perceptual_blur);
    }

    // ── BinEffectPlan ──────────────────────────────────────────────

    #[test]
    fn test_raw_plan_is_zero_strength() {
        let p = BinEffectPlan::raw();
        assert_eq!(p.strength, 0.0);
        assert_eq!(p.enabled_effects.count(), 0);
    }

    #[test]
    fn test_full_plan_is_max_strength() {
        let p = BinEffectPlan::full();
        assert_eq!(p.strength, 1.0);
        assert_eq!(p.enabled_effects.count(), EnabledEffects::TOTAL_EFFECTS);
    }

    // ── Plan: Default ──────────────────────────────────────────────

    #[test]
    fn test_plan_default_length() {
        let plans = plan_default();
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_default_all_raw() {
        for plan in plan_default() {
            assert_eq!(plan.strength, 0.0);
            assert_eq!(plan.enabled_effects.count(), 0);
        }
    }

    // ── Plan: Dispersion ───────────────────────────────────────────

    #[test]
    fn test_plan_dispersion_length() {
        let plans = plan_dispersion();
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_dispersion_bloom_monotonically_decreasing() {
        let plans = plan_dispersion();
        for i in 1..NUM_BINS {
            assert!(
                plans[i].bloom_radius_scale <= plans[i - 1].bloom_radius_scale + 1e-10,
                "bloom_radius_scale should decrease: bin {} ({}) > bin {} ({})",
                i,
                plans[i].bloom_radius_scale,
                i - 1,
                plans[i - 1].bloom_radius_scale
            );
        }
    }

    #[test]
    fn test_plan_dispersion_all_full_strength() {
        for plan in plan_dispersion() {
            assert_eq!(plan.strength, 1.0);
            assert_eq!(plan.enabled_effects.count(), EnabledEffects::TOTAL_EFFECTS);
        }
    }

    #[test]
    fn test_plan_dispersion_bloom_range() {
        let plans = plan_dispersion();
        assert!(plans[0].bloom_radius_scale > 2.5, "first bin should have large bloom");
        assert!(plans[NUM_BINS - 1].bloom_radius_scale < 1.5, "last bin should have small bloom");
    }

    // ── Plan: Resonance ────────────────────────────────────────────

    fn make_rng() -> Sha3RandomByteStream {
        Sha3RandomByteStream::new(&[0xCA, 0xFE], 100.0, 300.0, 300.0, 1.0)
    }

    #[test]
    fn test_plan_resonance_length() {
        let plans = plan_resonance(&mut make_rng());
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_resonance_correct_count() {
        let plans = plan_resonance(&mut make_rng());
        let full_count = plans.iter().filter(|p| p.strength == 1.0).count();
        assert!(
            (constants::SPECTRAL_RESONANCE_MIN_BINS..=constants::SPECTRAL_RESONANCE_MAX_BINS)
                .contains(&full_count),
            "expected {}-{} resonance bins, got {full_count}",
            constants::SPECTRAL_RESONANCE_MIN_BINS,
            constants::SPECTRAL_RESONANCE_MAX_BINS
        );
    }

    #[test]
    fn test_plan_resonance_non_resonance_low_strength() {
        let plans = plan_resonance(&mut make_rng());
        for plan in &plans {
            if plan.strength < 1.0 {
                assert!(
                    (plan.strength - 0.15).abs() < 1e-9,
                    "non-resonance bins should have strength 0.15"
                );
            }
        }
    }

    #[test]
    fn test_plan_resonance_deterministic() {
        let plans_a = plan_resonance(&mut make_rng());
        let plans_b = plan_resonance(&mut make_rng());
        for (a, b) in plans_a.iter().zip(plans_b.iter()) {
            assert_eq!(a.strength, b.strength);
            assert_eq!(a.enabled_effects, b.enabled_effects);
        }
    }

    // ── Plan: Weathering ───────────────────────────────────────────

    #[test]
    fn test_plan_weathering_length() {
        let plans = plan_weathering();
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_weathering_zone_effects() {
        let plans = plan_weathering();
        let [z1, z2] = constants::SPECTRAL_WEATHERING_ZONE_BOUNDARIES;
        for (i, plan) in plans.iter().enumerate() {
            assert_eq!(plan.strength, 1.0);
            if i < z1 {
                assert!(plan.enabled_effects.atmospheric_depth, "bin {i} should be gaseous");
                assert!(plan.enabled_effects.glow, "bin {i} should be gaseous");
            } else if i < z2 {
                assert!(plan.enabled_effects.champleve, "bin {i} should be metallic");
                assert!(plan.enabled_effects.micro_contrast, "bin {i} should be metallic");
            } else {
                assert!(plan.enabled_effects.aether, "bin {i} should be luminous");
                assert!(plan.enabled_effects.opalescence, "bin {i} should be luminous");
            }
        }
    }

    // ── Plan: Fingerprint ──────────────────────────────────────────

    #[test]
    fn test_plan_fingerprint_length() {
        let plans = plan_fingerprint(&mut make_rng());
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_fingerprint_effect_counts() {
        let plans = plan_fingerprint(&mut make_rng());
        for (i, plan) in plans.iter().enumerate() {
            let count = plan.enabled_effects.count();
            assert!(
                (constants::SPECTRAL_FINGERPRINT_MIN_EFFECTS
                    ..=constants::SPECTRAL_FINGERPRINT_MAX_EFFECTS)
                    .contains(&count),
                "bin {i} has {count} effects, expected {}-{}",
                constants::SPECTRAL_FINGERPRINT_MIN_EFFECTS,
                constants::SPECTRAL_FINGERPRINT_MAX_EFFECTS
            );
        }
    }

    #[test]
    fn test_plan_fingerprint_uniform_strength() {
        for plan in plan_fingerprint(&mut make_rng()) {
            assert!((plan.strength - 0.85).abs() < 1e-9);
        }
    }

    #[test]
    fn test_plan_fingerprint_deterministic() {
        let a = plan_fingerprint(&mut make_rng());
        let b = plan_fingerprint(&mut make_rng());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.enabled_effects, pb.enabled_effects);
        }
    }

    // ── Plan: Interference ─────────────────────────────────────────

    #[test]
    fn test_plan_interference_length() {
        let plans = plan_interference();
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_interference_strength_range() {
        for plan in plan_interference() {
            assert!(
                plan.strength >= 0.0 && plan.strength <= 1.0,
                "strength {} out of range",
                plan.strength
            );
        }
    }

    #[test]
    fn test_plan_interference_has_peaks_and_troughs() {
        let plans = plan_interference();
        let min = plans.iter().map(|p| p.strength).fold(f64::MAX, f64::min);
        let max = plans.iter().map(|p| p.strength).fold(f64::MIN, f64::max);
        assert!(min < 0.1, "should have near-zero troughs, min={min}");
        assert!(max > 0.9, "should have near-one peaks, max={max}");
    }

    #[test]
    fn test_plan_interference_all_effects_enabled() {
        for plan in plan_interference() {
            assert_eq!(plan.enabled_effects.count(), EnabledEffects::TOTAL_EFFECTS);
        }
    }

    // ── Plan: Cascade ──────────────────────────────────────────────

    #[test]
    fn test_plan_cascade_length() {
        let plans = plan_cascade();
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_cascade_monotonically_increasing() {
        let plans = plan_cascade();
        for i in 1..NUM_BINS {
            assert!(
                plans[i].strength >= plans[i - 1].strength - 1e-10,
                "cascade strength should increase: bin {} < bin {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_plan_cascade_endpoints() {
        let plans = plan_cascade();
        assert!(plans[0].strength < 0.02, "first bin should be near-zero");
        assert!((plans[NUM_BINS - 1].strength - 1.0).abs() < 1e-9, "last bin should be 1.0");
    }

    // ── Plan: Masking ──────────────────────────────────────────────

    #[test]
    fn test_plan_masking_length() {
        let plans = plan_masking();
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_masking_all_full_strength() {
        for plan in plan_masking() {
            assert_eq!(plan.strength, 1.0);
            assert_eq!(plan.enabled_effects.count(), EnabledEffects::TOTAL_EFFECTS);
        }
    }

    // ── Plan: Gravity ──────────────────────────────────────────────

    #[test]
    fn test_plan_gravity_length() {
        let plans = plan_gravity(&mut make_rng());
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_gravity_strength_range() {
        for plan in plan_gravity(&mut make_rng()) {
            assert!(plan.strength >= 0.0 && plan.strength <= 1.0);
        }
    }

    #[test]
    fn test_plan_gravity_has_peak() {
        let plans = plan_gravity(&mut make_rng());
        let max_strength = plans.iter().map(|p| p.strength).fold(f64::MIN, f64::max);
        assert!((max_strength - 1.0).abs() < 1e-9, "attractor bin should have strength 1.0");
    }

    #[test]
    fn test_plan_gravity_falls_off_from_peak() {
        let plans = plan_gravity(&mut make_rng());
        let peak_idx = plans
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Strength should decrease as we move away from peak
        if peak_idx >= 2 {
            assert!(plans[peak_idx - 2].strength < plans[peak_idx].strength);
        }
        if peak_idx + 2 < NUM_BINS {
            assert!(plans[peak_idx + 2].strength < plans[peak_idx].strength);
        }
    }

    #[test]
    fn test_plan_gravity_deterministic() {
        let a = plan_gravity(&mut make_rng());
        let b = plan_gravity(&mut make_rng());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.strength.to_bits(), pb.strength.to_bits());
        }
    }

    // ── plan_for_mode ──────────────────────────────────────────────

    fn make_base_config() -> EffectConfig {
        use super::super::effects::DogBloomConfig;
        EffectConfig {
            bloom_mode: "dog".to_string(),
            blur_radius_px: 10,
            blur_strength: 0.5,
            blur_core_brightness: 1.0,
            dog_config: DogBloomConfig::default(),
            perceptual_blur_enabled: true,
            perceptual_blur_config: Some(Default::default()),
            color_grade_enabled: true,
            color_grade_params: Default::default(),
            gradient_map_enabled: true,
            gradient_map_config: Default::default(),
            champleve_enabled: true,
            champleve_config: Default::default(),
            aether_enabled: true,
            aether_config: Default::default(),
            chromatic_bloom_enabled: true,
            chromatic_bloom_config: Default::default(),
            opalescence_enabled: true,
            opalescence_config: Default::default(),
            edge_luminance_enabled: true,
            edge_luminance_config: Default::default(),
            micro_contrast_enabled: true,
            micro_contrast_config: Default::default(),
            glow_enhancement_enabled: true,
            glow_enhancement_config: Default::default(),
            atmospheric_depth_enabled: true,
            atmospheric_depth_config: Default::default(),
            fine_texture_enabled: true,
            fine_texture_config: Default::default(),
        }
    }

    #[test]
    fn test_plan_for_mode_all_return_64() {
        let mut rng = make_rng();
        let base = make_base_config();
        for &mode in SpectralEffectMode::ALL {
            let plans = plan_for_mode(mode, &mut rng, Some(&base));
            assert_eq!(plans.len(), NUM_BINS, "mode {:?} should return {NUM_BINS} plans", mode);
        }
    }

    #[test]
    fn test_plan_for_mode_strengths_in_range() {
        let mut rng = make_rng();
        let base = make_base_config();
        for &mode in SpectralEffectMode::ALL {
            for (i, plan) in plan_for_mode(mode, &mut rng, Some(&base)).iter().enumerate() {
                assert!(
                    plan.strength >= 0.0 && plan.strength <= 1.0,
                    "mode {:?} bin {i}: strength {} out of range",
                    mode,
                    plan.strength
                );
                assert!(
                    plan.bloom_radius_scale > 0.0,
                    "mode {:?} bin {i}: bloom_radius_scale must be positive",
                    mode,
                );
            }
        }
    }

    // ── build_bin_effect_config ─────────────────────────────────────

    #[test]
    fn test_build_bin_effect_config_disables_effects() {
        use super::super::effects::DogBloomConfig;
        let base = EffectConfig {
            bloom_mode: "dog".to_string(),
            blur_radius_px: 10,
            blur_strength: 0.5,
            blur_core_brightness: 1.0,
            dog_config: DogBloomConfig::default(),
            perceptual_blur_enabled: true,
            perceptual_blur_config: None,
            color_grade_enabled: true,
            color_grade_params: Default::default(),
            gradient_map_enabled: true,
            gradient_map_config: Default::default(),
            champleve_enabled: true,
            champleve_config: Default::default(),
            aether_enabled: true,
            aether_config: Default::default(),
            chromatic_bloom_enabled: true,
            chromatic_bloom_config: Default::default(),
            opalescence_enabled: true,
            opalescence_config: Default::default(),
            edge_luminance_enabled: true,
            edge_luminance_config: Default::default(),
            micro_contrast_enabled: true,
            micro_contrast_config: Default::default(),
            glow_enhancement_enabled: true,
            glow_enhancement_config: Default::default(),
            atmospheric_depth_enabled: true,
            atmospheric_depth_config: Default::default(),
            fine_texture_enabled: true,
            fine_texture_config: Default::default(),
        };

        let plan = BinEffectPlan {
            strength: 1.0,
            bloom_radius_scale: 1.0,
            enabled_effects: EnabledEffects {
                bloom: false,
                dog_bloom: false,
                glow: true,
                chromatic_bloom: false,
                perceptual_blur: false,
                micro_contrast: true,
                gradient_map: false,
                color_grade: false,
                opalescence: false,
                champleve: false,
                aether: false,
                edge_luminance: true,
                atmospheric_depth: false,
                fine_texture: false,
            },
            custom_config: None,
        };

        let cfg = build_bin_effect_config(&base, &plan);
        assert_eq!(cfg.blur_radius_px, 0);
        assert_eq!(cfg.bloom_mode, "none");
        assert!(cfg.glow_enhancement_enabled);
        assert!(!cfg.chromatic_bloom_enabled);
        assert!(!cfg.perceptual_blur_enabled);
        assert!(cfg.micro_contrast_enabled);
        assert!(!cfg.gradient_map_enabled);
        assert!(!cfg.color_grade_enabled);
        assert!(!cfg.opalescence_enabled);
        assert!(!cfg.champleve_enabled);
        assert!(!cfg.aether_enabled);
        assert!(cfg.edge_luminance_enabled);
        assert!(!cfg.atmospheric_depth_enabled);
        assert!(!cfg.fine_texture_enabled);
    }

    #[test]
    fn test_build_bin_effect_config_scales_bloom() {
        use super::super::effects::DogBloomConfig;
        let base = EffectConfig {
            bloom_mode: "dog".to_string(),
            blur_radius_px: 10,
            blur_strength: 0.5,
            blur_core_brightness: 1.0,
            dog_config: DogBloomConfig { inner_sigma: 6.0, ..Default::default() },
            perceptual_blur_enabled: false,
            perceptual_blur_config: None,
            color_grade_enabled: false,
            color_grade_params: Default::default(),
            gradient_map_enabled: false,
            gradient_map_config: Default::default(),
            champleve_enabled: false,
            champleve_config: Default::default(),
            aether_enabled: false,
            aether_config: Default::default(),
            chromatic_bloom_enabled: false,
            chromatic_bloom_config: Default::default(),
            opalescence_enabled: false,
            opalescence_config: Default::default(),
            edge_luminance_enabled: false,
            edge_luminance_config: Default::default(),
            micro_contrast_enabled: false,
            micro_contrast_config: Default::default(),
            glow_enhancement_enabled: false,
            glow_enhancement_config: Default::default(),
            atmospheric_depth_enabled: false,
            atmospheric_depth_config: Default::default(),
            fine_texture_enabled: false,
            fine_texture_config: Default::default(),
        };

        let plan = BinEffectPlan {
            strength: 1.0,
            bloom_radius_scale: 2.5,
            enabled_effects: EnabledEffects::all(),
            custom_config: None,
        };
        let cfg = build_bin_effect_config(&base, &plan);
        assert_eq!(cfg.blur_radius_px, 25);
        assert!((cfg.dog_config.inner_sigma - 15.0).abs() < 1e-9);
    }

    // ── Conversion round-trip ──────────────────────────────────────

    #[test]
    fn test_bin_to_pixel_buffer_and_back() {
        let input = vec![[0.5f32, 0.3, 0.8], [0.0, 1.0, 0.0]];
        let buf = bin_to_pixel_buffer(&input);
        assert_eq!(buf.len(), 2);
        let output = pixel_buffer_to_bin(&buf);
        for (inp, out) in input.iter().zip(output.iter()) {
            assert!((inp[0] - out[0]).abs() < 1e-5);
            assert!((inp[1] - out[1]).abs() < 1e-5);
            assert!((inp[2] - out[2]).abs() < 1e-5);
        }
    }

    // ── Masking blend ──────────────────────────────────────────────

    #[test]
    fn test_masking_blend_white_complement_returns_raw() {
        let raw = vec![[0.5f32, 0.5, 0.5]; 4];
        let processed = vec![[1.0f32, 0.0, 0.0]; 4];
        let complement = vec![[1.0f32, 1.0, 1.0]; 4]; // bright → mask ≈ 1 → output ≈ raw
        let result = apply_masking_blend(&raw, &processed, &complement);
        for p in &result {
            assert!((p[0] - 0.5).abs() < 0.02);
            assert!((p[1] - 0.5).abs() < 0.02);
            assert!((p[2] - 0.5).abs() < 0.02);
        }
    }

    #[test]
    fn test_masking_blend_black_complement_returns_processed() {
        let raw = vec![[0.5f32, 0.5, 0.5]; 4];
        let processed = vec![[1.0f32, 0.0, 0.0]; 4];
        let complement = vec![[0.0f32, 0.0, 0.0]; 4]; // dark → mask ≈ 0 → output ≈ processed
        let result = apply_masking_blend(&raw, &processed, &complement);
        for p in &result {
            assert!((p[0] - 1.0).abs() < 0.02);
            assert!((p[1] - 0.0).abs() < 0.02);
        }
    }

    // ── cycle_frame_plan ───────────────────────────────────────────

    #[test]
    fn test_cycle_frame_plan_default_is_raw() {
        let plans = plan_default();
        let p = cycle_frame_plan(SpectralEffectMode::Default, &plans, 10.0, 50, 100);
        assert_eq!(p.strength, 0.0);
    }

    #[test]
    fn test_cycle_frame_plan_cascade_scales_with_frame() {
        let plans = plan_cascade();
        let p0 = cycle_frame_plan(SpectralEffectMode::Cascade, &plans, 0.0, 0, 100);
        let p50 = cycle_frame_plan(SpectralEffectMode::Cascade, &plans, 32.0, 50, 100);
        let p99 = cycle_frame_plan(SpectralEffectMode::Cascade, &plans, 63.0, 99, 100);
        assert!(p0.strength < 0.01);
        assert!((p50.strength - 0.5).abs() < 0.01);
        assert!(p99.strength > 0.98);
    }

    #[test]
    fn test_cycle_frame_plan_picks_nearest_bin() {
        let mut plans = vec![BinEffectPlan::raw(); NUM_BINS];
        plans[10] = BinEffectPlan::full();
        let p = cycle_frame_plan(SpectralEffectMode::Dispersion, &plans, 10.3, 5, 100);
        assert_eq!(p.strength, 1.0);
    }

    // ── Plan: Chaos ────────────────────────────────────────────────

    #[test]
    fn test_plan_chaos_length() {
        let base = make_base_config();
        let plans = plan_chaos(&mut make_rng(), &base);
        assert_eq!(plans.len(), NUM_BINS);
    }

    #[test]
    fn test_plan_chaos_strength_range() {
        let base = make_base_config();
        for (i, plan) in plan_chaos(&mut make_rng(), &base).iter().enumerate() {
            assert!(
                plan.strength >= constants::SPECTRAL_CHAOS_STRENGTH_MIN
                    && plan.strength <= constants::SPECTRAL_CHAOS_STRENGTH_MAX,
                "bin {i}: strength {} out of [{}, {}]",
                plan.strength,
                constants::SPECTRAL_CHAOS_STRENGTH_MIN,
                constants::SPECTRAL_CHAOS_STRENGTH_MAX
            );
        }
    }

    #[test]
    fn test_plan_chaos_bloom_scale_range() {
        let base = make_base_config();
        for (i, plan) in plan_chaos(&mut make_rng(), &base).iter().enumerate() {
            assert!(
                plan.bloom_radius_scale >= constants::SPECTRAL_CHAOS_BLOOM_SCALE_MIN
                    && plan.bloom_radius_scale <= constants::SPECTRAL_CHAOS_BLOOM_SCALE_MAX,
                "bin {i}: bloom_radius_scale {} out of [{}, {}]",
                plan.bloom_radius_scale,
                constants::SPECTRAL_CHAOS_BLOOM_SCALE_MIN,
                constants::SPECTRAL_CHAOS_BLOOM_SCALE_MAX
            );
        }
    }

    #[test]
    fn test_plan_chaos_all_have_custom_config() {
        let base = make_base_config();
        for (i, plan) in plan_chaos(&mut make_rng(), &base).iter().enumerate() {
            assert!(plan.custom_config.is_some(), "bin {i} should have custom_config");
        }
    }

    #[test]
    fn test_plan_chaos_has_mixed_enables() {
        let base = make_base_config();
        let plans = plan_chaos(&mut make_rng(), &base);
        let total_enabled: usize = plans.iter().map(|p| p.enabled_effects.count()).sum();
        let total_possible = NUM_BINS * EnabledEffects::TOTAL_EFFECTS;
        assert!(
            total_enabled > total_possible / 8,
            "chaos should enable a reasonable fraction of effects: {total_enabled}/{total_possible}"
        );
        assert!(
            total_enabled < total_possible * 7 / 8,
            "chaos should not enable nearly all effects: {total_enabled}/{total_possible}"
        );
    }

    #[test]
    fn test_plan_chaos_deterministic() {
        let base = make_base_config();
        let a = plan_chaos(&mut make_rng(), &base);
        let b = plan_chaos(&mut make_rng(), &base);
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                pa.strength.to_bits(),
                pb.strength.to_bits(),
                "bin {i}: strength not deterministic"
            );
            assert_eq!(pa.enabled_effects, pb.enabled_effects, "bin {i}: effects differ");
            assert!(
                pa.custom_config.is_some() && pb.custom_config.is_some(),
                "bin {i}: custom_config mismatch"
            );
        }
    }

    #[test]
    fn test_plan_chaos_custom_config_respects_enables() {
        let base = make_base_config();
        let plans = plan_chaos(&mut make_rng(), &base);
        for (i, plan) in plans.iter().enumerate() {
            let cfg = plan.custom_config.as_ref().unwrap();
            if !plan.enabled_effects.bloom {
                assert_eq!(cfg.blur_radius_px, 0, "bin {i}: bloom disabled but radius > 0");
            }
            if !plan.enabled_effects.dog_bloom {
                assert_eq!(cfg.bloom_mode, "none", "bin {i}: dog disabled but mode not none");
            }
            assert_eq!(
                cfg.glow_enhancement_enabled, plan.enabled_effects.glow,
                "bin {i}: glow mismatch"
            );
            assert_eq!(
                cfg.chromatic_bloom_enabled, plan.enabled_effects.chromatic_bloom,
                "bin {i}: chromatic_bloom mismatch"
            );
            assert_eq!(
                cfg.micro_contrast_enabled, plan.enabled_effects.micro_contrast,
                "bin {i}: micro_contrast mismatch"
            );
            assert_eq!(
                cfg.opalescence_enabled, plan.enabled_effects.opalescence,
                "bin {i}: opalescence mismatch"
            );
            assert_eq!(
                cfg.champleve_enabled, plan.enabled_effects.champleve,
                "bin {i}: champleve mismatch"
            );
            assert_eq!(cfg.aether_enabled, plan.enabled_effects.aether, "bin {i}: aether mismatch");
            assert_eq!(
                cfg.edge_luminance_enabled, plan.enabled_effects.edge_luminance,
                "bin {i}: edge_luminance mismatch"
            );
            assert_eq!(
                cfg.atmospheric_depth_enabled, plan.enabled_effects.atmospheric_depth,
                "bin {i}: atmospheric_depth mismatch"
            );
            assert_eq!(
                cfg.fine_texture_enabled, plan.enabled_effects.fine_texture,
                "bin {i}: fine_texture mismatch"
            );
        }
    }

    #[test]
    fn test_plan_chaos_randomized_params_differ() {
        let base = make_base_config();
        let plans = plan_chaos(&mut make_rng(), &base);
        let mut micro_strengths = std::collections::HashSet::new();
        for plan in &plans {
            if plan.enabled_effects.micro_contrast {
                let cfg = plan.custom_config.as_ref().unwrap();
                micro_strengths.insert(cfg.micro_contrast_config.strength.to_bits());
            }
        }
        assert!(micro_strengths.len() > 1, "chaos should produce varied micro_contrast strengths");
    }

    #[test]
    fn test_build_bin_effect_config_uses_custom_config() {
        let base = make_base_config();
        let mut custom = base.clone();
        custom.blur_radius_px = 999;
        custom.glow_enhancement_enabled = false;

        let plan = BinEffectPlan {
            strength: 1.0,
            bloom_radius_scale: 1.0,
            enabled_effects: EnabledEffects::all(),
            custom_config: Some(Box::new(custom)),
        };

        let cfg = build_bin_effect_config(&base, &plan);
        assert_eq!(cfg.blur_radius_px, 999, "should use custom_config directly");
        assert!(!cfg.glow_enhancement_enabled, "should use custom_config directly");
    }

    #[test]
    fn test_plan_chaos_dog_params_randomized() {
        let base = make_base_config();
        let plans = plan_chaos(&mut make_rng(), &base);
        let mut sigmas = std::collections::HashSet::new();
        for plan in &plans {
            if plan.enabled_effects.dog_bloom {
                let cfg = plan.custom_config.as_ref().unwrap();
                sigmas.insert(cfg.dog_config.inner_sigma.to_bits());
            }
        }
        assert!(sigmas.len() > 1, "chaos should produce varied DoG inner_sigma");
    }

    #[test]
    fn test_plan_chaos_gradient_palettes_vary() {
        let base = make_base_config();
        let plans = plan_chaos(&mut make_rng(), &base);
        let mut strengths = std::collections::HashSet::new();
        for plan in &plans {
            if plan.enabled_effects.gradient_map {
                let cfg = plan.custom_config.as_ref().unwrap();
                strengths.insert(cfg.gradient_map_config.strength.to_bits());
            }
        }
        assert!(strengths.len() > 1, "chaos should produce varied gradient map strengths");
    }
}
