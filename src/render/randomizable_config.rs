//! Randomizable configuration for all effect parameters.
//!
//! This module defines the complete parameter space for effect configuration,
//! with support for explicit user values or random generation.

use super::effect_randomizer::{EffectRandomizer, RandomizationLog, RandomizationRecord};
use super::parameter_descriptors as pd;
use crate::sim::Sha3RandomByteStream;

/// Complete configuration for all randomizable effect parameters.
///
/// Each field is `Option<T>`: `None` means "randomize this", `Some(T)` means "use explicit value".
#[derive(Clone, Debug, Default)]
pub struct RandomizableEffectConfig {
    /// Whether to enable the bloom effect.
    pub enable_bloom: Option<bool>,
    /// Whether to enable the glow effect.
    pub enable_glow: Option<bool>,
    /// Whether to enable chromatic bloom.
    pub enable_chromatic_bloom: Option<bool>,
    /// Whether to enable perceptual blur.
    pub enable_perceptual_blur: Option<bool>,
    /// Whether to enable micro contrast enhancement.
    pub enable_micro_contrast: Option<bool>,
    /// Whether to enable gradient map color mapping.
    pub enable_gradient_map: Option<bool>,
    /// Whether to enable color grading.
    pub enable_color_grade: Option<bool>,
    /// Whether to enable the champlevé material effect.
    pub enable_champleve: Option<bool>,
    /// Whether to enable the aether material effect.
    pub enable_aether: Option<bool>,
    /// Whether to enable the opalescence material effect.
    pub enable_opalescence: Option<bool>,
    /// Whether to enable edge luminance enhancement.
    pub enable_edge_luminance: Option<bool>,
    /// Whether to enable atmospheric depth simulation.
    pub enable_atmospheric_depth: Option<bool>,
    /// Whether to enable fine texture overlay.
    pub enable_fine_texture: Option<bool>,

    /// Bloom blur strength (iteration count).
    pub blur_strength: Option<f64>,
    /// Bloom blur radius as a fraction of image size.
    pub blur_radius_scale: Option<f64>,
    /// Core brightness retention during bloom blur.
    pub blur_core_brightness: Option<f64>,
    /// Difference-of-Gaussians edge enhancement strength.
    pub dog_strength: Option<f64>,
    /// `DoG` sigma as a fraction of image size.
    pub dog_sigma_scale: Option<f64>,
    /// `DoG` ratio between the two Gaussian widths.
    pub dog_ratio: Option<f64>,
    /// Glow effect strength.
    pub glow_strength: Option<f64>,
    /// Luminance threshold for glow activation.
    pub glow_threshold: Option<f64>,
    /// Glow radius as a fraction of image size.
    pub glow_radius_scale: Option<f64>,
    /// Glow falloff sharpness.
    pub glow_sharpness: Option<f64>,
    /// Saturation boost applied in glow regions.
    pub glow_saturation_boost: Option<f64>,

    /// Chromatic bloom intensity.
    pub chromatic_bloom_strength: Option<f64>,
    /// Chromatic bloom radius as a fraction of image size.
    pub chromatic_bloom_radius_scale: Option<f64>,
    /// Channel separation distance for chromatic bloom.
    pub chromatic_bloom_separation_scale: Option<f64>,
    /// Luminance threshold for chromatic bloom.
    pub chromatic_bloom_threshold: Option<f64>,

    /// Perceptual blur strength.
    pub perceptual_blur_strength: Option<f64>,

    /// Overall color grading intensity.
    pub color_grade_strength: Option<f64>,
    /// Vignette darkening strength at image edges.
    pub vignette_strength: Option<f64>,
    /// Vignette gradient softness.
    pub vignette_softness: Option<f64>,
    /// Color vibrance adjustment multiplier.
    pub vibrance: Option<f64>,
    /// Clarity (local contrast) strength.
    pub clarity_strength: Option<f64>,
    /// Tone curve S-curve strength.
    pub tone_curve_strength: Option<f64>,

    /// Gradient map blending strength.
    pub gradient_map_strength: Option<f64>,
    /// How much original hue is preserved in gradient mapping.
    pub gradient_map_hue_preservation: Option<f64>,
    /// Palette index for gradient mapping (0–14).
    pub gradient_map_palette: Option<usize>,

    /// Opalescence effect strength.
    pub opalescence_strength: Option<f64>,
    /// Opalescence pattern scale.
    pub opalescence_scale: Option<f64>,
    /// Number of opalescence interference layers.
    pub opalescence_layers: Option<usize>,

    /// Champlevé flow alignment with simulation field.
    pub champleve_flow_alignment: Option<f64>,
    /// Champlevé interference pattern amplitude.
    pub champleve_interference_amplitude: Option<f64>,
    /// Champlevé rim highlight intensity.
    pub champleve_rim_intensity: Option<f64>,
    /// Champlevé rim warmth (cool-to-warm shift).
    pub champleve_rim_warmth: Option<f64>,
    /// Champlevé interior brightness lift.
    pub champleve_interior_lift: Option<f64>,

    /// Aether flow alignment with simulation field.
    pub aether_flow_alignment: Option<f64>,
    /// Aether volumetric scattering strength.
    pub aether_scattering_strength: Option<f64>,
    /// Aether iridescence color-shift amplitude.
    pub aether_iridescence_amplitude: Option<f64>,
    /// Aether caustic light pattern strength.
    pub aether_caustic_strength: Option<f64>,

    /// Micro contrast enhancement strength.
    pub micro_contrast_strength: Option<f64>,
    /// Micro contrast sampling radius in pixels.
    pub micro_contrast_radius: Option<usize>,
    /// Edge luminance enhancement strength.
    pub edge_luminance_strength: Option<f64>,
    /// Edge detection threshold for luminance enhancement.
    pub edge_luminance_threshold: Option<f64>,
    /// Brightness boost applied to detected edges.
    pub edge_luminance_brightness_boost: Option<f64>,

    /// Atmospheric depth effect strength.
    pub atmospheric_depth_strength: Option<f64>,
    /// Atmospheric desaturation amount with depth.
    pub atmospheric_desaturation: Option<f64>,
    /// Atmospheric darkening amount with depth.
    pub atmospheric_darkening: Option<f64>,
    /// Fog color red component.
    pub atmospheric_fog_color_r: Option<f64>,
    /// Fog color green component.
    pub atmospheric_fog_color_g: Option<f64>,
    /// Fog color blue component.
    pub atmospheric_fog_color_b: Option<f64>,
    /// Fine texture overlay strength.
    pub fine_texture_strength: Option<f64>,
    /// Fine texture pattern scale.
    pub fine_texture_scale: Option<f64>,
    /// Fine texture contrast.
    pub fine_texture_contrast: Option<f64>,

    /// HDR intensity scaling factor.
    pub hdr_scale: Option<f64>,

    /// Black point clipping threshold.
    pub clip_black: Option<f64>,
    /// White point clipping threshold.
    pub clip_white: Option<f64>,
}

impl RandomizableEffectConfig {
    /// Resolve all `Option<T>` values: use explicit values or randomize.
    ///
    /// Returns a fully resolved configuration and a log of randomization decisions.
    pub fn resolve(
        &self,
        rng: &mut Sha3RandomByteStream,
        width: u32,
        height: u32,
    ) -> (ResolvedEffectConfig, RandomizationLog) {
        let mut randomizer = EffectRandomizer::new(rng);
        let mut log = RandomizationLog::new();
        let mut resolved = ResolvedEffectConfig { width, height, ..Default::default() };

        self.resolve_enable_flags(&mut resolved, &mut randomizer, &mut log);
        self.resolve_bloom_glow_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_chromatic_bloom_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_color_grade_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_material_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_detail_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_atmospheric_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_hdr_params(&mut resolved, &mut randomizer, &mut log);
        self.resolve_clip_params(&mut resolved, &mut randomizer, &mut log);

        let resolved = apply_conflict_detection(resolved, &mut log);
        (resolved, log)
    }

    fn resolve_enable_flags(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.enable_bloom = Self::resolve_enable(
            "bloom",
            self.enable_bloom,
            pd::ENABLE_PROB_BLOOM,
            randomizer,
            log,
        );
        resolved.enable_glow =
            Self::resolve_enable("glow", self.enable_glow, pd::ENABLE_PROB_GLOW, randomizer, log);
        resolved.enable_chromatic_bloom = Self::resolve_enable(
            "chromatic_bloom",
            self.enable_chromatic_bloom,
            pd::ENABLE_PROB_CHROMATIC_BLOOM,
            randomizer,
            log,
        );
        resolved.enable_perceptual_blur = Self::resolve_enable(
            "perceptual_blur",
            self.enable_perceptual_blur,
            pd::ENABLE_PROB_PERCEPTUAL_BLUR,
            randomizer,
            log,
        );
        resolved.enable_micro_contrast = Self::resolve_enable(
            "micro_contrast",
            self.enable_micro_contrast,
            pd::ENABLE_PROB_MICRO_CONTRAST,
            randomizer,
            log,
        );
        resolved.enable_gradient_map = Self::resolve_enable(
            "gradient_map",
            self.enable_gradient_map,
            pd::ENABLE_PROB_GRADIENT_MAP,
            randomizer,
            log,
        );
        resolved.enable_color_grade = Self::resolve_enable(
            "color_grade",
            self.enable_color_grade,
            pd::ENABLE_PROB_COLOR_GRADE,
            randomizer,
            log,
        );
        resolved.enable_champleve = Self::resolve_enable(
            "champleve",
            self.enable_champleve,
            pd::ENABLE_PROB_CHAMPLEVE,
            randomizer,
            log,
        );
        resolved.enable_aether = Self::resolve_enable(
            "aether",
            self.enable_aether,
            pd::ENABLE_PROB_AETHER,
            randomizer,
            log,
        );
        resolved.enable_opalescence = Self::resolve_enable(
            "opalescence",
            self.enable_opalescence,
            pd::ENABLE_PROB_OPALESCENCE,
            randomizer,
            log,
        );
        resolved.enable_edge_luminance = Self::resolve_enable(
            "edge_luminance",
            self.enable_edge_luminance,
            pd::ENABLE_PROB_EDGE_LUMINANCE,
            randomizer,
            log,
        );
        resolved.enable_atmospheric_depth = Self::resolve_enable(
            "atmospheric_depth",
            self.enable_atmospheric_depth,
            pd::ENABLE_PROB_ATMOSPHERIC_DEPTH,
            randomizer,
            log,
        );
        resolved.enable_fine_texture = Self::resolve_enable(
            "fine_texture",
            self.enable_fine_texture,
            pd::ENABLE_PROB_FINE_TEXTURE,
            randomizer,
            log,
        );
    }

    fn resolve_bloom_glow_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.blur_strength = Self::resolve_float(
            "blur_strength",
            self.blur_strength,
            &pd::BLUR_STRENGTH,
            randomizer,
            log,
        );
        resolved.blur_radius_scale = Self::resolve_float(
            "blur_radius_scale",
            self.blur_radius_scale,
            &pd::BLUR_RADIUS_SCALE,
            randomizer,
            log,
        );
        resolved.blur_core_brightness = Self::resolve_float(
            "blur_core_brightness",
            self.blur_core_brightness,
            &pd::BLUR_CORE_BRIGHTNESS,
            randomizer,
            log,
        );
        resolved.dog_strength = Self::resolve_float(
            "dog_strength",
            self.dog_strength,
            &pd::DOG_STRENGTH,
            randomizer,
            log,
        );
        resolved.dog_sigma_scale = Self::resolve_float(
            "dog_sigma_scale",
            self.dog_sigma_scale,
            &pd::DOG_SIGMA_SCALE,
            randomizer,
            log,
        );
        resolved.dog_ratio =
            Self::resolve_float("dog_ratio", self.dog_ratio, &pd::DOG_RATIO, randomizer, log);
        resolved.glow_strength = Self::resolve_float(
            "glow_strength",
            self.glow_strength,
            &pd::GLOW_STRENGTH,
            randomizer,
            log,
        );
        resolved.glow_threshold = Self::resolve_float(
            "glow_threshold",
            self.glow_threshold,
            &pd::GLOW_THRESHOLD,
            randomizer,
            log,
        );
        resolved.glow_radius_scale = Self::resolve_float(
            "glow_radius_scale",
            self.glow_radius_scale,
            &pd::GLOW_RADIUS_SCALE,
            randomizer,
            log,
        );
        resolved.glow_sharpness = Self::resolve_float(
            "glow_sharpness",
            self.glow_sharpness,
            &pd::GLOW_SHARPNESS,
            randomizer,
            log,
        );
        resolved.glow_saturation_boost = Self::resolve_float(
            "glow_saturation_boost",
            self.glow_saturation_boost,
            &pd::GLOW_SATURATION_BOOST,
            randomizer,
            log,
        );
    }

    fn resolve_chromatic_bloom_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.chromatic_bloom_strength = Self::resolve_float(
            "chromatic_bloom_strength",
            self.chromatic_bloom_strength,
            &pd::CHROMATIC_BLOOM_STRENGTH,
            randomizer,
            log,
        );
        resolved.chromatic_bloom_radius_scale = Self::resolve_float(
            "chromatic_bloom_radius_scale",
            self.chromatic_bloom_radius_scale,
            &pd::CHROMATIC_BLOOM_RADIUS_SCALE,
            randomizer,
            log,
        );
        resolved.chromatic_bloom_separation_scale = Self::resolve_float(
            "chromatic_bloom_separation_scale",
            self.chromatic_bloom_separation_scale,
            &pd::CHROMATIC_BLOOM_SEPARATION_SCALE,
            randomizer,
            log,
        );
        resolved.chromatic_bloom_threshold = Self::resolve_float(
            "chromatic_bloom_threshold",
            self.chromatic_bloom_threshold,
            &pd::CHROMATIC_BLOOM_THRESHOLD,
            randomizer,
            log,
        );
    }

    fn resolve_color_grade_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.perceptual_blur_strength = Self::resolve_float(
            "perceptual_blur_strength",
            self.perceptual_blur_strength,
            &pd::PERCEPTUAL_BLUR_STRENGTH,
            randomizer,
            log,
        );
        resolved.color_grade_strength = Self::resolve_float(
            "color_grade_strength",
            self.color_grade_strength,
            &pd::COLOR_GRADE_STRENGTH,
            randomizer,
            log,
        );
        resolved.vignette_strength = Self::resolve_float(
            "vignette_strength",
            self.vignette_strength,
            &pd::VIGNETTE_STRENGTH,
            randomizer,
            log,
        );
        resolved.vignette_softness = Self::resolve_float(
            "vignette_softness",
            self.vignette_softness,
            &pd::VIGNETTE_SOFTNESS,
            randomizer,
            log,
        );
        resolved.vibrance =
            Self::resolve_float("vibrance", self.vibrance, &pd::VIBRANCE, randomizer, log);
        resolved.clarity_strength = Self::resolve_float(
            "clarity_strength",
            self.clarity_strength,
            &pd::CLARITY_STRENGTH,
            randomizer,
            log,
        );
        resolved.tone_curve_strength = Self::resolve_float(
            "tone_curve_strength",
            self.tone_curve_strength,
            &pd::TONE_CURVE_STRENGTH,
            randomizer,
            log,
        );
        resolved.gradient_map_strength = Self::resolve_float(
            "gradient_map_strength",
            self.gradient_map_strength,
            &pd::GRADIENT_MAP_STRENGTH,
            randomizer,
            log,
        );
        resolved.gradient_map_hue_preservation = Self::resolve_float(
            "gradient_map_hue_preservation",
            self.gradient_map_hue_preservation,
            &pd::GRADIENT_MAP_HUE_PRESERVATION,
            randomizer,
            log,
        );
        resolved.gradient_map_palette = Self::resolve_int(
            "gradient_map_palette",
            self.gradient_map_palette,
            &pd::GRADIENT_MAP_PALETTE,
            randomizer,
            log,
        );
    }

    fn resolve_material_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.opalescence_strength = Self::resolve_float(
            "opalescence_strength",
            self.opalescence_strength,
            &pd::OPALESCENCE_STRENGTH,
            randomizer,
            log,
        );
        resolved.opalescence_scale = Self::resolve_float(
            "opalescence_scale",
            self.opalescence_scale,
            &pd::OPALESCENCE_SCALE,
            randomizer,
            log,
        );
        resolved.opalescence_layers = Self::resolve_int(
            "opalescence_layers",
            self.opalescence_layers,
            &pd::OPALESCENCE_LAYERS,
            randomizer,
            log,
        );
        resolved.champleve_flow_alignment = Self::resolve_float(
            "champleve_flow_alignment",
            self.champleve_flow_alignment,
            &pd::CHAMPLEVE_FLOW_ALIGNMENT,
            randomizer,
            log,
        );
        resolved.champleve_interference_amplitude = Self::resolve_float(
            "champleve_interference_amplitude",
            self.champleve_interference_amplitude,
            &pd::CHAMPLEVE_INTERFERENCE_AMPLITUDE,
            randomizer,
            log,
        );
        resolved.champleve_rim_intensity = Self::resolve_float(
            "champleve_rim_intensity",
            self.champleve_rim_intensity,
            &pd::CHAMPLEVE_RIM_INTENSITY,
            randomizer,
            log,
        );
        resolved.champleve_rim_warmth = Self::resolve_float(
            "champleve_rim_warmth",
            self.champleve_rim_warmth,
            &pd::CHAMPLEVE_RIM_WARMTH,
            randomizer,
            log,
        );
        resolved.champleve_interior_lift = Self::resolve_float(
            "champleve_interior_lift",
            self.champleve_interior_lift,
            &pd::CHAMPLEVE_INTERIOR_LIFT,
            randomizer,
            log,
        );
        resolved.aether_flow_alignment = Self::resolve_float(
            "aether_flow_alignment",
            self.aether_flow_alignment,
            &pd::AETHER_FLOW_ALIGNMENT,
            randomizer,
            log,
        );
        resolved.aether_scattering_strength = Self::resolve_float(
            "aether_scattering_strength",
            self.aether_scattering_strength,
            &pd::AETHER_SCATTERING_STRENGTH,
            randomizer,
            log,
        );
        resolved.aether_iridescence_amplitude = Self::resolve_float(
            "aether_iridescence_amplitude",
            self.aether_iridescence_amplitude,
            &pd::AETHER_IRIDESCENCE_AMPLITUDE,
            randomizer,
            log,
        );
        resolved.aether_caustic_strength = Self::resolve_float(
            "aether_caustic_strength",
            self.aether_caustic_strength,
            &pd::AETHER_CAUSTIC_STRENGTH,
            randomizer,
            log,
        );
    }

    fn resolve_detail_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.micro_contrast_strength = Self::resolve_float(
            "micro_contrast_strength",
            self.micro_contrast_strength,
            &pd::MICRO_CONTRAST_STRENGTH,
            randomizer,
            log,
        );
        resolved.micro_contrast_radius = Self::resolve_int(
            "micro_contrast_radius",
            self.micro_contrast_radius,
            &pd::MICRO_CONTRAST_RADIUS,
            randomizer,
            log,
        );
        resolved.edge_luminance_strength = Self::resolve_float(
            "edge_luminance_strength",
            self.edge_luminance_strength,
            &pd::EDGE_LUMINANCE_STRENGTH,
            randomizer,
            log,
        );
        resolved.edge_luminance_threshold = Self::resolve_float(
            "edge_luminance_threshold",
            self.edge_luminance_threshold,
            &pd::EDGE_LUMINANCE_THRESHOLD,
            randomizer,
            log,
        );
        resolved.edge_luminance_brightness_boost = Self::resolve_float(
            "edge_luminance_brightness_boost",
            self.edge_luminance_brightness_boost,
            &pd::EDGE_LUMINANCE_BRIGHTNESS_BOOST,
            randomizer,
            log,
        );
    }

    fn resolve_atmospheric_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.atmospheric_depth_strength = Self::resolve_float(
            "atmospheric_depth_strength",
            self.atmospheric_depth_strength,
            &pd::ATMOSPHERIC_DEPTH_STRENGTH,
            randomizer,
            log,
        );
        resolved.atmospheric_desaturation = Self::resolve_float(
            "atmospheric_desaturation",
            self.atmospheric_desaturation,
            &pd::ATMOSPHERIC_DESATURATION,
            randomizer,
            log,
        );
        resolved.atmospheric_darkening = Self::resolve_float(
            "atmospheric_darkening",
            self.atmospheric_darkening,
            &pd::ATMOSPHERIC_DARKENING,
            randomizer,
            log,
        );
        resolved.atmospheric_fog_color_r = Self::resolve_float(
            "atmospheric_fog_color_r",
            self.atmospheric_fog_color_r,
            &pd::ATMOSPHERIC_FOG_COLOR_R,
            randomizer,
            log,
        );
        resolved.atmospheric_fog_color_g = Self::resolve_float(
            "atmospheric_fog_color_g",
            self.atmospheric_fog_color_g,
            &pd::ATMOSPHERIC_FOG_COLOR_G,
            randomizer,
            log,
        );
        resolved.atmospheric_fog_color_b = Self::resolve_float(
            "atmospheric_fog_color_b",
            self.atmospheric_fog_color_b,
            &pd::ATMOSPHERIC_FOG_COLOR_B,
            randomizer,
            log,
        );
        resolved.fine_texture_strength = Self::resolve_float(
            "fine_texture_strength",
            self.fine_texture_strength,
            &pd::FINE_TEXTURE_STRENGTH,
            randomizer,
            log,
        );
        resolved.fine_texture_scale = Self::resolve_float(
            "fine_texture_scale",
            self.fine_texture_scale,
            &pd::FINE_TEXTURE_SCALE,
            randomizer,
            log,
        );
        resolved.fine_texture_contrast = Self::resolve_float(
            "fine_texture_contrast",
            self.fine_texture_contrast,
            &pd::FINE_TEXTURE_CONTRAST,
            randomizer,
            log,
        );
    }

    fn resolve_hdr_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        resolved.hdr_scale =
            Self::resolve_float("hdr_scale", self.hdr_scale, &pd::HDR_SCALE, randomizer, log);
    }

    fn resolve_clip_params(
        &self,
        resolved: &mut ResolvedEffectConfig,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) {
        let (clip_black, clip_white) =
            if let (Some(black), Some(white)) = (self.clip_black, self.clip_white) {
                if black < white { (black, white) } else { (white, black) }
            } else if let Some(black) = self.clip_black {
                let white = randomizer.randomize_float(&pd::CLIP_WHITE);
                if black < white { (black, white) } else { (white, black) }
            } else if let Some(white) = self.clip_white {
                let black = randomizer.randomize_float(&pd::CLIP_BLACK);
                if black < white { (black, white) } else { (white, black) }
            } else {
                randomizer.randomize_ordered_pair(&pd::CLIP_BLACK, &pd::CLIP_WHITE)
            };

        resolved.clip_black = clip_black;
        resolved.clip_white = clip_white;

        let mut clip_record = RandomizationRecord::new("clipping".to_string(), true, false);
        clip_record.add_float(
            "clip_black".to_string(),
            clip_black,
            self.clip_black.is_none(),
            (pd::CLIP_BLACK.min, pd::CLIP_BLACK.max),
        );
        clip_record.add_float(
            "clip_white".to_string(),
            clip_white,
            self.clip_white.is_none(),
            (pd::CLIP_WHITE.min, pd::CLIP_WHITE.max),
        );
        log.add_record(clip_record);
    }

    fn resolve_enable(
        name: &str,
        value: Option<bool>,
        probability: f64,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) -> bool {
        let (enabled, was_randomized) = match value {
            Some(v) => (v, false),
            None => (randomizer.randomize_enable(probability), true),
        };

        log.add_record(RandomizationRecord::new(name.to_string(), enabled, was_randomized));

        enabled
    }

    fn resolve_float(
        name: &str,
        value: Option<f64>,
        descriptor: &pd::FloatParamDescriptor,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) -> f64 {
        let (resolved, was_randomized) = match value {
            Some(v) => (v, false),
            None => (randomizer.randomize_float(descriptor), true),
        };

        // Find or create record for this parameter's effect group
        let effect_name = Self::effect_group_name(name);
        let record_idx = log.effects.iter().position(|r| r.effect_name == effect_name);

        let range = (descriptor.min, descriptor.max);
        if let Some(idx) = record_idx {
            log.effects[idx].add_float(name.to_string(), resolved, was_randomized, range);
        } else {
            let mut record = RandomizationRecord::new(effect_name, true, false);
            record.add_float(name.to_string(), resolved, was_randomized, range);
            log.add_record(record);
        }

        resolved
    }

    fn resolve_int(
        name: &str,
        value: Option<usize>,
        descriptor: &pd::IntParamDescriptor,
        randomizer: &mut EffectRandomizer,
        log: &mut RandomizationLog,
    ) -> usize {
        let (resolved, was_randomized) = match value {
            Some(v) => (v, false),
            None => (randomizer.randomize_int(descriptor), true),
        };

        let effect_name = Self::effect_group_name(name);
        let record_idx = log.effects.iter().position(|r| r.effect_name == effect_name);

        let range = (descriptor.min, descriptor.max);
        if let Some(idx) = record_idx {
            log.effects[idx].add_int(name.to_string(), resolved, was_randomized, range);
        } else {
            let mut record = RandomizationRecord::new(effect_name, true, false);
            record.add_int(name.to_string(), resolved, was_randomized, range);
            log.add_record(record);
        }

        resolved
    }

    /// Extract effect group name from parameter name (e.g., "`glow_strength`" -> "glow")
    fn effect_group_name(param_name: &str) -> String {
        if param_name.starts_with("atmospheric_") {
            return "atmospheric_depth".to_string();
        }

        const MULTI_WORD_PREFIXES: &[&str] = &[
            "chromatic_bloom",
            "gradient_map",
            "fine_texture",
            "micro_contrast",
            "edge_luminance",
            "color_grade",
            "tone_curve",
        ];
        for prefix in MULTI_WORD_PREFIXES {
            if param_name.starts_with(prefix) {
                return (*prefix).to_string();
            }
        }
        param_name.split('_').next().unwrap_or(param_name).to_string()
    }
}

/// Fully resolved effect configuration with all parameters determined.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedEffectConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,

    /// Whether bloom is enabled.
    pub enable_bloom: bool,
    /// Whether glow is enabled.
    pub enable_glow: bool,
    /// Whether chromatic bloom is enabled.
    pub enable_chromatic_bloom: bool,
    /// Whether perceptual blur is enabled.
    pub enable_perceptual_blur: bool,
    /// Whether micro contrast is enabled.
    pub enable_micro_contrast: bool,
    /// Whether gradient map is enabled.
    pub enable_gradient_map: bool,
    /// Whether color grading is enabled.
    pub enable_color_grade: bool,
    /// Whether champlevé is enabled.
    pub enable_champleve: bool,
    /// Whether aether is enabled.
    pub enable_aether: bool,
    /// Whether opalescence is enabled.
    pub enable_opalescence: bool,
    /// Whether edge luminance is enabled.
    pub enable_edge_luminance: bool,
    /// Whether atmospheric depth is enabled.
    pub enable_atmospheric_depth: bool,
    /// Whether fine texture is enabled.
    pub enable_fine_texture: bool,

    /// Resolved bloom blur strength.
    pub blur_strength: f64,
    /// Resolved bloom blur radius scale.
    pub blur_radius_scale: f64,
    /// Resolved bloom core brightness.
    pub blur_core_brightness: f64,
    /// Resolved `DoG` edge enhancement strength.
    pub dog_strength: f64,
    /// Resolved `DoG` sigma scale.
    pub dog_sigma_scale: f64,
    /// Resolved `DoG` Gaussian ratio.
    pub dog_ratio: f64,
    /// Resolved glow strength.
    pub glow_strength: f64,
    /// Resolved glow luminance threshold.
    pub glow_threshold: f64,
    /// Resolved glow radius scale.
    pub glow_radius_scale: f64,
    /// Resolved glow falloff sharpness.
    pub glow_sharpness: f64,
    /// Resolved glow saturation boost.
    pub glow_saturation_boost: f64,
    /// Resolved chromatic bloom strength.
    pub chromatic_bloom_strength: f64,
    /// Resolved chromatic bloom radius scale.
    pub chromatic_bloom_radius_scale: f64,
    /// Resolved chromatic bloom channel separation scale.
    pub chromatic_bloom_separation_scale: f64,
    /// Resolved chromatic bloom threshold.
    pub chromatic_bloom_threshold: f64,
    /// Resolved perceptual blur strength.
    pub perceptual_blur_strength: f64,
    /// Resolved color grading strength.
    pub color_grade_strength: f64,
    /// Resolved vignette strength.
    pub vignette_strength: f64,
    /// Resolved vignette softness.
    pub vignette_softness: f64,
    /// Resolved vibrance multiplier.
    pub vibrance: f64,
    /// Resolved clarity strength.
    pub clarity_strength: f64,
    /// Resolved tone curve strength.
    pub tone_curve_strength: f64,
    /// Resolved gradient map strength.
    pub gradient_map_strength: f64,
    /// Resolved gradient map hue preservation.
    pub gradient_map_hue_preservation: f64,
    /// Palette index for gradient mapping (0–14).
    pub gradient_map_palette: usize,
    /// Resolved opalescence strength.
    pub opalescence_strength: f64,
    /// Resolved opalescence scale.
    pub opalescence_scale: f64,
    /// Resolved number of opalescence layers.
    pub opalescence_layers: usize,
    /// Resolved champlevé flow alignment.
    pub champleve_flow_alignment: f64,
    /// Resolved champlevé interference amplitude.
    pub champleve_interference_amplitude: f64,
    /// Resolved champlevé rim intensity.
    pub champleve_rim_intensity: f64,
    /// Resolved champlevé rim warmth.
    pub champleve_rim_warmth: f64,
    /// Resolved champlevé interior lift.
    pub champleve_interior_lift: f64,
    /// Resolved aether flow alignment.
    pub aether_flow_alignment: f64,
    /// Resolved aether scattering strength.
    pub aether_scattering_strength: f64,
    /// Resolved aether iridescence amplitude.
    pub aether_iridescence_amplitude: f64,
    /// Resolved aether caustic strength.
    pub aether_caustic_strength: f64,
    /// Resolved micro contrast strength.
    pub micro_contrast_strength: f64,
    /// Resolved micro contrast radius.
    pub micro_contrast_radius: usize,
    /// Resolved edge luminance strength.
    pub edge_luminance_strength: f64,
    /// Resolved edge luminance threshold.
    pub edge_luminance_threshold: f64,
    /// Resolved edge luminance brightness boost.
    pub edge_luminance_brightness_boost: f64,
    /// Resolved atmospheric depth strength.
    pub atmospheric_depth_strength: f64,
    /// Resolved atmospheric desaturation.
    pub atmospheric_desaturation: f64,
    /// Resolved atmospheric darkening.
    pub atmospheric_darkening: f64,
    /// Resolved fog color red component.
    pub atmospheric_fog_color_r: f64,
    /// Resolved fog color green component.
    pub atmospheric_fog_color_g: f64,
    /// Resolved fog color blue component.
    pub atmospheric_fog_color_b: f64,
    /// Resolved fine texture strength.
    pub fine_texture_strength: f64,
    /// Resolved fine texture scale.
    pub fine_texture_scale: f64,
    /// Resolved fine texture contrast.
    pub fine_texture_contrast: f64,
    /// Resolved HDR scaling factor.
    pub hdr_scale: f64,
    /// Resolved black point clipping threshold.
    pub clip_black: f64,
    /// Resolved white point clipping threshold.
    pub clip_white: f64,
}

fn softness_stack_score(config: &ResolvedEffectConfig) -> f64 {
    let mut score = 0.0;
    if config.enable_bloom {
        score += 1.0;
    }
    if config.enable_chromatic_bloom {
        score += 0.95;
    }
    if config.enable_perceptual_blur {
        score += 0.85;
    }
    if config.enable_glow {
        score += 0.55;
    }
    if config.enable_atmospheric_depth {
        score += 0.35;
    }
    score
}

fn heavy_softness_count(config: &ResolvedEffectConfig) -> usize {
    usize::from(config.enable_bloom)
        + usize::from(config.enable_chromatic_bloom)
        + usize::from(config.enable_perceptual_blur)
}

fn cap_expensive_opalescence(config: &mut ResolvedEffectConfig, adjustments: &mut Vec<String>) {
    if config.enable_opalescence
        && config.opalescence_layers > 5
        && config.opalescence_strength > 0.30
    {
        let original_layers = config.opalescence_layers;
        config.opalescence_layers = 5;
        adjustments.push(format!(
            "Performance guard: Capped opalescence_layers ({} -> 5) at high strength ({:.2}) to prevent exponential cost",
            original_layers, config.opalescence_strength
        ));
    }
}

fn disable_isolated_chromatic_bloom(
    config: &mut ResolvedEffectConfig,
    adjustments: &mut Vec<String>,
) {
    if config.enable_chromatic_bloom && !config.enable_bloom {
        config.enable_chromatic_bloom = false;
        adjustments.push(
            "Quality guard: Disabled chromatic_bloom because base bloom was off, avoiding isolated prismatic haze"
                .to_string(),
        );
    }
}

fn soften_unsupported_atmospheric_depth(
    config: &mut ResolvedEffectConfig,
    adjustments: &mut Vec<String>,
) {
    if !config.enable_atmospheric_depth || config.enable_color_grade || config.enable_aether {
        return;
    }

    let original_strength = config.atmospheric_depth_strength;
    let original_desaturation = config.atmospheric_desaturation;

    config.atmospheric_depth_strength = config.atmospheric_depth_strength.min(0.12);
    config.atmospheric_desaturation = config.atmospheric_desaturation.min(0.16);

    if (config.atmospheric_depth_strength - original_strength).abs() > f64::EPSILON
        || (config.atmospheric_desaturation - original_desaturation).abs() > f64::EPSILON
    {
        adjustments.push(format!(
            "Quality guard: Softened atmospheric depth without color support (strength: {:.3} -> {:.3}, desaturation: {:.3} -> {:.3})",
            original_strength,
            config.atmospheric_depth_strength,
            original_desaturation,
            config.atmospheric_desaturation
        ));
    }
}

fn rebalance_gradient_map_without_color_grade(
    config: &mut ResolvedEffectConfig,
    adjustments: &mut Vec<String>,
) {
    if !config.enable_gradient_map || config.enable_color_grade {
        return;
    }

    let original_strength = config.gradient_map_strength;
    let original_hue_preservation = config.gradient_map_hue_preservation;

    config.gradient_map_strength = config.gradient_map_strength.min(0.35);
    config.gradient_map_hue_preservation = config.gradient_map_hue_preservation.max(0.50);

    if (config.gradient_map_strength - original_strength).abs() > f64::EPSILON
        || (config.gradient_map_hue_preservation - original_hue_preservation).abs() > f64::EPSILON
    {
        adjustments.push(format!(
            "Quality guard: Rebalanced gradient map toward source hues (strength: {:.3} -> {:.3}, hue preservation: {:.3} -> {:.3})",
            original_strength,
            config.gradient_map_strength,
            original_hue_preservation,
            config.gradient_map_hue_preservation
        ));
    }
}

fn break_heavy_softness_stack(config: &mut ResolvedEffectConfig, adjustments: &mut Vec<String>) {
    while heavy_softness_count(config) >= 2 && softness_stack_score(config) >= 2.0 {
        let score_before = softness_stack_score(config);
        if config.enable_perceptual_blur {
            config.enable_perceptual_blur = false;
            adjustments.push(format!(
                "Quality guard: Disabled perceptual_blur to break softness stack (score: {score_before:.2})"
            ));
        } else if config.enable_chromatic_bloom {
            config.enable_chromatic_bloom = false;
            adjustments.push(format!(
                "Quality guard: Disabled chromatic_bloom to break softness stack (score: {score_before:.2})"
            ));
        } else {
            break;
        }
    }
}

fn rescue_detail_for_softness_stack(
    config: &mut ResolvedEffectConfig,
    softness_score: f64,
    adjustments: &mut Vec<String>,
) {
    if softness_score < 2.0 {
        return;
    }

    let original = config.clone();

    config.enable_micro_contrast = true;
    config.enable_edge_luminance = true;
    config.dog_strength = config.dog_strength.min(0.28);
    config.dog_sigma_scale = config.dog_sigma_scale.min(0.0054);
    config.glow_strength = config.glow_strength.min(0.28);
    config.glow_radius_scale = config.glow_radius_scale.min(0.0034);
    config.chromatic_bloom_strength = config.chromatic_bloom_strength.min(0.36);
    config.chromatic_bloom_radius_scale = config.chromatic_bloom_radius_scale.min(0.0046);
    config.chromatic_bloom_separation_scale = config.chromatic_bloom_separation_scale.min(0.0010);
    config.perceptual_blur_strength = config.perceptual_blur_strength.min(0.44);
    config.micro_contrast_strength = config.micro_contrast_strength.max(0.24);
    config.edge_luminance_strength = config.edge_luminance_strength.max(0.18);
    config.edge_luminance_threshold = config.edge_luminance_threshold.min(0.20);
    config.edge_luminance_brightness_boost = config.edge_luminance_brightness_boost.max(0.28);

    if *config != original {
        adjustments.push(format!(
            "Quality guard: Tightened softness stack and enabled detail rescue (score: {:.2}, micro_contrast: {} -> {}, edge_luminance: {} -> {})",
            softness_score,
            original.enable_micro_contrast,
            config.enable_micro_contrast,
            original.enable_edge_luminance,
            config.enable_edge_luminance
        ));
    }
}

fn remove_extreme_softness_blur(
    config: &mut ResolvedEffectConfig,
    softness_score: f64,
    adjustments: &mut Vec<String>,
) {
    if softness_score >= 2.6 && config.enable_chromatic_bloom && config.enable_perceptual_blur {
        config.enable_perceptual_blur = false;
        adjustments.push(format!(
            "Quality guard: Disabled perceptual_blur inside an extreme softness stack (score: {softness_score:.2})"
        ));
    }
}

fn log_constraint_adjustments(log: &mut RandomizationLog, adjustments: Vec<String>) {
    if adjustments.is_empty() {
        return;
    }

    let mut adjustment_record =
        RandomizationRecord::new("render_constraints".to_string(), true, false);

    for adjustment in adjustments {
        adjustment_record.parameters.push(crate::render::effect_randomizer::RandomizedParameter {
            name: "constraint".to_string(),
            value: adjustment,
            was_randomized: false,
            range_used: "N/A".to_string(),
        });
    }

    log.add_record(adjustment_record);
}

/// Apply render constraints to prevent pathological runtime and low-quality effect combinations.
///
/// Philosophy: Maximum exploration with minimum intervention.
/// - Keep the generative space broad while blocking combinations that predictably fail QA
/// - Prefer soft caps over hard disables unless a combination is consistently harmful
/// - Preserve deterministic resolution and logging for every adjustment
fn apply_conflict_detection(
    mut config: ResolvedEffectConfig,
    log: &mut RandomizationLog,
) -> ResolvedEffectConfig {
    let mut adjustments = Vec::new();
    cap_expensive_opalescence(&mut config, &mut adjustments);
    disable_isolated_chromatic_bloom(&mut config, &mut adjustments);
    soften_unsupported_atmospheric_depth(&mut config, &mut adjustments);
    rebalance_gradient_map_without_color_grade(&mut config, &mut adjustments);
    break_heavy_softness_stack(&mut config, &mut adjustments);

    let softness_score = softness_stack_score(&config);
    rescue_detail_for_softness_stack(&mut config, softness_score, &mut adjustments);
    remove_extreme_softness_blur(&mut config, softness_score, &mut adjustments);
    log_constraint_adjustments(log, adjustments);

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sha3RandomByteStream;

    fn baseline_resolved_config() -> ResolvedEffectConfig {
        ResolvedEffectConfig {
            width: 1920,
            height: 1080,
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
            blur_radius_scale: 0.02,
            blur_strength: 10.0,
            blur_core_brightness: 10.0,
            dog_strength: 0.3,
            dog_sigma_scale: 0.006,
            dog_ratio: 2.5,
            glow_strength: 0.4,
            glow_threshold: 0.65,
            glow_radius_scale: 0.007,
            glow_sharpness: 2.5,
            glow_saturation_boost: 0.2,
            chromatic_bloom_strength: 0.6,
            chromatic_bloom_radius_scale: 0.012,
            chromatic_bloom_separation_scale: 0.002,
            chromatic_bloom_threshold: 0.15,
            perceptual_blur_strength: 0.65,
            color_grade_strength: 0.5,
            vignette_strength: 0.4,
            vignette_softness: 2.5,
            vibrance: 1.1,
            clarity_strength: 0.25,
            tone_curve_strength: 0.5,
            gradient_map_strength: 0.7,
            gradient_map_hue_preservation: 0.2,
            gradient_map_palette: 0,
            opalescence_strength: 0.15,
            opalescence_scale: 0.01,
            opalescence_layers: 3,
            champleve_flow_alignment: 0.6,
            champleve_interference_amplitude: 0.5,
            champleve_rim_intensity: 1.8,
            champleve_rim_warmth: 0.6,
            champleve_interior_lift: 0.65,
            aether_flow_alignment: 0.7,
            aether_scattering_strength: 0.9,
            aether_iridescence_amplitude: 0.6,
            aether_caustic_strength: 0.3,
            micro_contrast_strength: 0.25,
            micro_contrast_radius: 5,
            edge_luminance_strength: 0.2,
            edge_luminance_threshold: 0.18,
            edge_luminance_brightness_boost: 0.3,
            atmospheric_depth_strength: 0.25,
            atmospheric_desaturation: 0.35,
            atmospheric_darkening: 0.15,
            atmospheric_fog_color_r: 0.08,
            atmospheric_fog_color_g: 0.12,
            atmospheric_fog_color_b: 0.22,
            fine_texture_strength: 0.12,
            fine_texture_scale: 0.0018,
            fine_texture_contrast: 0.35,
            hdr_scale: 0.12,
            clip_black: 0.01,
            clip_white: 0.99,
        }
    }

    /// Test that extreme opalescence layers are capped at high strength
    #[test]
    fn test_opalescence_layers_performance_guard() {
        let config = ResolvedEffectConfig {
            enable_opalescence: true,
            opalescence_layers: 6,
            opalescence_strength: 0.35,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config.clone(), &mut log);

        // Verify layers were capped at 5
        assert_eq!(result.opalescence_layers, 5, "Opalescence layers should be capped at 5");

        // Verify adjustment was logged
        assert!(!log.effects.is_empty(), "Performance adjustment should be logged");
    }

    /// Test that opalescence layers below threshold or low strength are NOT capped
    #[test]
    fn test_opalescence_below_threshold_not_affected() {
        let config = ResolvedEffectConfig {
            enable_opalescence: true,
            opalescence_layers: 6,
            opalescence_strength: 0.25,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config.clone(), &mut log);

        // Verify layers are NOT capped (strength too low)
        assert_eq!(
            result.opalescence_layers, 6,
            "Opalescence layers should not be capped at low strength"
        );

        // Verify no adjustment was logged
        assert!(log.effects.is_empty(), "No adjustment should be logged for safe parameters");
    }

    #[test]
    fn test_chromatic_bloom_is_disabled_without_base_bloom() {
        let config =
            ResolvedEffectConfig { enable_chromatic_bloom: true, ..baseline_resolved_config() };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert!(
            !result.enable_chromatic_bloom,
            "chromatic bloom should be disabled when base bloom is off"
        );
        assert!(log.effects.iter().any(|record| {
            record.effect_name == "render_constraints"
                && record
                    .parameters
                    .iter()
                    .any(|parameter| parameter.value.contains("Disabled chromatic_bloom"))
        }));
    }

    #[test]
    fn test_atmospheric_depth_is_soft_capped_without_color_support() {
        let config = ResolvedEffectConfig {
            enable_atmospheric_depth: true,
            atmospheric_depth_strength: 0.17,
            atmospheric_desaturation: 0.31,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert_eq!(result.atmospheric_depth_strength, 0.12);
        assert_eq!(result.atmospheric_desaturation, 0.16);
        assert!(log.effects.iter().any(|record| {
            record.effect_name == "render_constraints"
                && record
                    .parameters
                    .iter()
                    .any(|parameter| parameter.value.contains("Softened atmospheric depth"))
        }));
    }

    #[test]
    fn test_atmospheric_fog_parameters_share_atmospheric_depth_group() {
        assert_eq!(
            RandomizableEffectConfig::effect_group_name("atmospheric_depth_strength"),
            "atmospheric_depth"
        );
        assert_eq!(
            RandomizableEffectConfig::effect_group_name("atmospheric_fog_color_r"),
            "atmospheric_depth"
        );
        assert_eq!(
            RandomizableEffectConfig::effect_group_name("atmospheric_fog_color_b"),
            "atmospheric_depth"
        );
    }

    #[test]
    fn test_gradient_map_is_rebalanced_without_color_grade() {
        let config = ResolvedEffectConfig {
            enable_gradient_map: true,
            gradient_map_strength: 0.62,
            gradient_map_hue_preservation: 0.18,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert_eq!(result.gradient_map_strength, 0.35);
        assert_eq!(result.gradient_map_hue_preservation, 0.50);
        assert!(log.effects.iter().any(|record| {
            record.effect_name == "render_constraints"
                && record
                    .parameters
                    .iter()
                    .any(|parameter| parameter.value.contains("Rebalanced gradient map"))
        }));
    }

    /// Test that parameter randomization works end-to-end with new wider ranges
    #[test]
    fn test_wide_range_randomization() {
        let config = RandomizableEffectConfig::default();

        let seed = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (resolved, log) = config.resolve(&mut rng, 1920, 1080);

        // Verify all parameters are within their curated museum-quality ranges
        assert!(resolved.blur_strength >= 3.0 && resolved.blur_strength <= 5.5);
        assert!(resolved.blur_radius_scale >= 0.004 && resolved.blur_radius_scale <= 0.010);
        assert!(resolved.glow_strength >= 0.18 && resolved.glow_strength <= 0.40);
        assert!(resolved.dog_strength >= 0.20 && resolved.dog_strength <= 0.36);
        assert!(
            resolved.chromatic_bloom_strength >= 0.24 && resolved.chromatic_bloom_strength <= 0.55
        );
        assert!(
            resolved.vibrance >= 1.10 && resolved.vibrance <= 1.35,
            "vibrance {} outside [1.10, 1.35]",
            resolved.vibrance
        );
        assert!(resolved.opalescence_layers >= 1 && resolved.opalescence_layers <= 6);
        assert!(resolved.gradient_map_palette <= 14, "Palette index should be 0-14");
        assert!(
            resolved.atmospheric_fog_color_r >= 0.02 && resolved.atmospheric_fog_color_r <= 0.10
        );
        assert!(
            resolved.atmospheric_fog_color_g >= 0.04 && resolved.atmospheric_fog_color_g <= 0.12
        );
        assert!(
            resolved.atmospheric_fog_color_b >= 0.08 && resolved.atmospheric_fog_color_b <= 0.18
        );

        // Verify log contains all randomized parameters
        assert!(!log.effects.is_empty(), "Should have randomization log");
    }

    /// Test that gradient palette randomization produces valid palette indices
    #[test]
    fn test_gradient_palette_randomization() {
        let config = RandomizableEffectConfig::default();

        // Test multiple random seeds to ensure variety
        for seed_val in 1..20 {
            let seed = [seed_val, 2, 3, 4, 5, 6, 7, 8];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let (resolved, _log) = config.resolve(&mut rng, 1920, 1080);

            // Verify palette index is always within valid range
            assert!(
                resolved.gradient_map_palette <= 14,
                "Palette index {} exceeds maximum (14)",
                resolved.gradient_map_palette
            );
        }
    }

    /// Test that atmospheric fog color randomization produces valid RGB values
    #[test]
    fn test_atmospheric_fog_color_randomization() {
        let config = RandomizableEffectConfig::default();

        // Test multiple random seeds to ensure variety
        for seed_val in 1..20 {
            let seed = [seed_val, 10, 20, 30, 40, 50, 60, 70];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let (resolved, _log) = config.resolve(&mut rng, 1920, 1080);

            // Verify fog colors are within valid dark tone range
            assert!(
                resolved.atmospheric_fog_color_r >= 0.02
                    && resolved.atmospheric_fog_color_r <= 0.10,
                "Fog color R {} out of range",
                resolved.atmospheric_fog_color_r
            );
            assert!(
                resolved.atmospheric_fog_color_g >= 0.04
                    && resolved.atmospheric_fog_color_g <= 0.12,
                "Fog color G {} out of range",
                resolved.atmospheric_fog_color_g
            );
            assert!(
                resolved.atmospheric_fog_color_b >= 0.08
                    && resolved.atmospheric_fog_color_b <= 0.18,
                "Fog color B {} out of range",
                resolved.atmospheric_fog_color_b
            );
        }
    }

    /// Test that per-effect enable probabilities produce statistically correct distributions.
    /// Runs many resolutions and verifies each effect's enable rate matches its probability.
    #[test]
    fn test_effect_enable_probabilities_statistical() {
        let n = 500;
        let mut counts = std::collections::HashMap::<&str, usize>::new();

        for seed_val in 0..n {
            let seed = [(seed_val & 0xFF) as u8, ((seed_val >> 8) & 0xFF) as u8, 3, 4, 5, 6, 7, 8];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let config = RandomizableEffectConfig::default();
            let (resolved, _) = config.resolve(&mut rng, 1920, 1080);

            if resolved.enable_bloom {
                *counts.entry("bloom").or_default() += 1;
            }
            if resolved.enable_glow {
                *counts.entry("glow").or_default() += 1;
            }
            if resolved.enable_chromatic_bloom {
                *counts.entry("chromatic_bloom").or_default() += 1;
            }
            if resolved.enable_perceptual_blur {
                *counts.entry("perceptual_blur").or_default() += 1;
            }
            if resolved.enable_micro_contrast {
                *counts.entry("micro_contrast").or_default() += 1;
            }
            if resolved.enable_gradient_map {
                *counts.entry("gradient_map").or_default() += 1;
            }
            if resolved.enable_color_grade {
                *counts.entry("color_grade").or_default() += 1;
            }
            if resolved.enable_champleve {
                *counts.entry("champleve").or_default() += 1;
            }
            if resolved.enable_aether {
                *counts.entry("aether").or_default() += 1;
            }
            if resolved.enable_opalescence {
                *counts.entry("opalescence").or_default() += 1;
            }
            if resolved.enable_edge_luminance {
                *counts.entry("edge_luminance").or_default() += 1;
            }
            if resolved.enable_atmospheric_depth {
                *counts.entry("atmospheric_depth").or_default() += 1;
            }
            if resolved.enable_fine_texture {
                *counts.entry("fine_texture").or_default() += 1;
            }
        }

        let check = |name: &str, expected_prob: f64, tolerance: f64| {
            let count = *counts.get(name).unwrap_or(&0) as f64;
            let rate = count / f64::from(n);
            assert!(
                (rate - expected_prob).abs() < tolerance,
                "{name}: rate {rate:.3} deviates from expected {expected_prob:.2} by more than {tolerance:.2}",
            );
        };

        let default_tolerance = 0.12; // generous tolerance for 500 samples
        check("bloom", pd::ENABLE_PROB_BLOOM, default_tolerance);
        check("glow", pd::ENABLE_PROB_GLOW, default_tolerance);
        check("chromatic_bloom", pd::ENABLE_PROB_CHROMATIC_BLOOM * pd::ENABLE_PROB_BLOOM, 0.06);
        check("perceptual_blur", pd::ENABLE_PROB_PERCEPTUAL_BLUR, default_tolerance);
        check("micro_contrast", pd::ENABLE_PROB_MICRO_CONTRAST, default_tolerance);
        check("gradient_map", pd::ENABLE_PROB_GRADIENT_MAP, default_tolerance);
        check("color_grade", pd::ENABLE_PROB_COLOR_GRADE, default_tolerance);
        check("champleve", pd::ENABLE_PROB_CHAMPLEVE, default_tolerance);
        check("aether", pd::ENABLE_PROB_AETHER, default_tolerance);
        check("opalescence", pd::ENABLE_PROB_OPALESCENCE, default_tolerance);
        check("edge_luminance", pd::ENABLE_PROB_EDGE_LUMINANCE, default_tolerance);
        check("atmospheric_depth", pd::ENABLE_PROB_ATMOSPHERIC_DEPTH, default_tolerance);
        check("fine_texture", pd::ENABLE_PROB_FINE_TEXTURE, default_tolerance);
    }

    /// Test that explicit enable values always override the probability.
    #[test]
    fn test_explicit_enable_overrides_probability() {
        let config = RandomizableEffectConfig {
            enable_perceptual_blur: Some(true),
            enable_chromatic_bloom: Some(false),
            ..Default::default()
        };

        let seed = [42, 43, 44, 45, 46, 47, 48, 49];
        let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (resolved, _) = config.resolve(&mut rng, 1920, 1080);

        assert!(
            resolved.enable_perceptual_blur,
            "explicit Some(true) must override 5% probability"
        );
        assert!(
            !resolved.enable_chromatic_bloom,
            "explicit Some(false) must override 20% probability"
        );
    }

    #[test]
    fn test_heavy_softness_stack_disables_weakest_blurs() {
        // All three heavy blurs on: bloom + chromatic + perceptual.
        // The stack-cap guard should cascade, disabling perceptual first then
        // chromatic, leaving only bloom as the remaining heavy blur.
        let config = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert!(result.enable_bloom, "bloom should remain on (never disabled by stack-cap)");
        assert!(!result.enable_perceptual_blur, "perceptual blur should be disabled first");
        assert!(!result.enable_chromatic_bloom, "chromatic bloom should be disabled second");

        let records: Vec<_> = log
            .effects
            .iter()
            .filter(|record| record.effect_name == "render_constraints")
            .flat_map(|record| record.parameters.iter())
            .map(|parameter| parameter.value.as_str())
            .collect();

        assert!(
            records.iter().any(|value| value.contains("Disabled perceptual_blur")
                && value.contains("break softness stack")),
            "stack-cap log entry for perceptual_blur missing: {records:?}"
        );
        assert!(
            records.iter().any(|value| value.contains("Disabled chromatic_bloom")
                && value.contains("break softness stack")),
            "stack-cap log entry for chromatic_bloom missing: {records:?}"
        );
    }

    #[test]
    fn test_stack_cap_preserves_single_heavy_blur() {
        // Only one heavy blur on (bloom) — stack-cap must not fire.
        let config = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_atmospheric_depth: true,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert!(result.enable_bloom);
        assert!(result.enable_glow);
        assert!(result.enable_atmospheric_depth);

        let stack_cap_fired = log.effects.iter().any(|record| {
            record
                .parameters
                .iter()
                .any(|parameter| parameter.value.contains("break softness stack"))
        });
        assert!(!stack_cap_fired, "stack-cap guard should not fire with only one heavy blur");
    }

    #[test]
    fn test_stack_cap_disables_chromatic_when_perceptual_off() {
        // bloom + chromatic + glow (no perceptual): score 2.50, heavy_count 2.
        // Stack-cap should disable chromatic (never bloom), since perceptual
        // is off and cannot be the first-choice victim.
        let config = ResolvedEffectConfig {
            enable_bloom: true,
            enable_chromatic_bloom: true,
            enable_glow: true,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert!(result.enable_bloom);
        assert!(!result.enable_chromatic_bloom);
        assert!(result.enable_glow);
        assert!(log.effects.iter().any(|record| {
            record
                .parameters
                .iter()
                .any(|parameter| parameter.value.contains("Disabled chromatic_bloom"))
        }));
    }

    #[test]
    fn test_stack_cap_preserves_pure_bloom_plus_chromatic() {
        // bloom + chromatic alone (score 1.95) is preserved — it's only a
        // mild softening that we treat as a rare variety case, not a stack.
        let config = ResolvedEffectConfig {
            enable_bloom: true,
            enable_chromatic_bloom: true,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert!(result.enable_bloom);
        assert!(result.enable_chromatic_bloom);
        let stack_cap_fired = log.effects.iter().any(|record| {
            record
                .parameters
                .iter()
                .any(|parameter| parameter.value.contains("break softness stack"))
        });
        assert!(!stack_cap_fired);
    }

    #[test]
    fn test_extreme_softness_stack_disables_perceptual_blur() {
        let config = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            enable_atmospheric_depth: true,
            ..baseline_resolved_config()
        };

        let mut log = RandomizationLog::new();
        let result = apply_conflict_detection(config, &mut log);

        assert!(
            !result.enable_perceptual_blur,
            "perceptual blur should be disabled in extreme softness stacks"
        );
        assert!(log.effects.iter().any(|record| {
            record.effect_name == "render_constraints"
                && record
                    .parameters
                    .iter()
                    .any(|parameter| parameter.value.contains("Disabled perceptual_blur"))
        }));
    }

    /// Test that vibrance always falls within the curated range [1.10, 1.35].
    #[test]
    fn test_vibrance_raised_floor() {
        for seed_val in 0u16..200 {
            let seed = [(seed_val & 0xFF) as u8, ((seed_val >> 8) & 0xFF) as u8, 99, 4, 5, 6, 7, 8];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let config = RandomizableEffectConfig::default();
            let (resolved, _) = config.resolve(&mut rng, 1920, 1080);

            assert!(
                resolved.vibrance >= 1.10,
                "seed {} produced vibrance {} below 1.10 floor",
                seed_val,
                resolved.vibrance
            );
            assert!(
                resolved.vibrance <= 1.35,
                "seed {} produced vibrance {} above 1.35 ceiling",
                seed_val,
                resolved.vibrance
            );
        }
    }

    /// Test that `LuxuryPalette::from_index` correctly maps all valid indices
    #[test]
    fn test_luxury_palette_from_index() {
        use crate::post_effects::LuxuryPalette;

        // Test all valid indices
        for i in 0..=14 {
            let _palette = LuxuryPalette::from_index(i);
        }

        // Test that modulo wrapping works (indices > 14)
        let _palette_15 = LuxuryPalette::from_index(15);
        let _palette_0 = LuxuryPalette::from_index(0);
    }
}
