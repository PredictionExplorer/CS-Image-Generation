//! Parameter descriptors for post-processing effects.
//!
//! Each descriptor defines the curated range for museum-quality output.
//! Ranges are derived from empirical analysis of visually pleasing results.

/// Descriptor for a floating-point parameter with bounded range.
#[derive(Clone, Debug)]
pub struct FloatParamDescriptor {
    /// Machine-readable parameter name used for logging and serialization.
    pub name: &'static str,
    /// Minimum allowed value (inclusive).
    pub min: f64,
    /// Maximum allowed value (inclusive).
    pub max: f64,
    /// Human-readable description of what this parameter controls.
    pub description: &'static str,
}

/// Descriptor for an integer parameter with bounded range.
#[derive(Clone, Debug)]
pub struct IntParamDescriptor {
    /// Machine-readable parameter name used for logging and serialization.
    pub name: &'static str,
    /// Minimum allowed value (inclusive).
    pub min: usize,
    /// Maximum allowed value (inclusive).
    pub max: usize,
    /// Human-readable description of what this parameter controls.
    pub description: &'static str,
}

// ---------------------------------------------------------------------------
// Borda orbit selection weights
// ---------------------------------------------------------------------------

/// Equilateralness-to-chaos Borda weight ratio descriptor.
///
/// Sampled log-uniformly with a moderate bias toward equilateralness.
/// Ratio range: 1/5 to 125 (median ~5.0).
/// At ratio < 1 chaos dominates; at ratio > 1 equilateralness dominates.
pub const EQUIL_CHAOS_RATIO: FloatParamDescriptor = FloatParamDescriptor {
    name: "equil_chaos_ratio",
    min: 0.2,
    max: 125.0,
    description: "Equilateralness-to-chaos Borda weight ratio (log-uniform, 1/5x to 125x)",
};

// ---------------------------------------------------------------------------
// Effect enable probabilities
// ---------------------------------------------------------------------------

/// Probability that the Gaussian bloom effect is enabled during randomization.
pub const ENABLE_PROB_BLOOM: f64 = 0.28;
/// Probability that the tight glow enhancement is enabled during randomization.
pub const ENABLE_PROB_GLOW: f64 = 0.55;
/// Probability that prismatic chromatic bloom is enabled during randomization.
pub const ENABLE_PROB_CHROMATIC_BLOOM: f64 = 0.20;
/// Probability that `OKLab` perceptual blur is enabled during randomization.
pub const ENABLE_PROB_PERCEPTUAL_BLUR: f64 = 0.18;
/// Probability that micro-contrast enhancement is enabled during randomization.
pub const ENABLE_PROB_MICRO_CONTRAST: f64 = 0.85;
/// Probability that luxury gradient mapping is enabled during randomization.
pub const ENABLE_PROB_GRADIENT_MAP: f64 = 0.18;
/// Probability that cinematic color grading is enabled during randomization.
pub const ENABLE_PROB_COLOR_GRADE: f64 = 0.60;
/// Probability that the champlevé enamel effect is enabled during randomization.
pub const ENABLE_PROB_CHAMPLEVE: f64 = 0.25;
/// Probability that the aether filament effect is enabled during randomization.
pub const ENABLE_PROB_AETHER: f64 = 0.35;
/// Probability that opalescence shimmer is enabled during randomization.
pub const ENABLE_PROB_OPALESCENCE: f64 = 0.25;
/// Probability that edge luminance brightening is enabled during randomization.
pub const ENABLE_PROB_EDGE_LUMINANCE: f64 = 0.55;
/// Probability that atmospheric depth perspective is enabled during randomization.
pub const ENABLE_PROB_ATMOSPHERIC_DEPTH: f64 = 0.18;
/// Probability that fine canvas texture is enabled during randomization.
pub const ENABLE_PROB_FINE_TEXTURE: f64 = 0.45;

// ---------------------------------------------------------------------------
// Bloom & glow parameters
// ---------------------------------------------------------------------------

/// Strength of the Gaussian blur bloom effect.
pub const BLUR_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "blur_strength",
    min: 3.0,
    max: 5.5,
    description: "Strength of Gaussian blur bloom effect",
};

/// Blur radius scale, expressed relative to the output resolution.
pub const BLUR_RADIUS_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "blur_radius_scale",
    min: 0.004,
    max: 0.010,
    description: "Radius scale for blur (relative to resolution)",
};

/// Brightness preservation factor in the blur core.
pub const BLUR_CORE_BRIGHTNESS: FloatParamDescriptor = FloatParamDescriptor {
    name: "blur_core_brightness",
    min: 9.0,
    max: 14.0,
    description: "Brightness preservation in blur core",
};

/// Difference-of-Gaussians bloom strength.
pub const DOG_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "dog_strength",
    min: 0.22,
    max: 0.36,
    description: "Difference-of-Gaussians bloom strength",
};

/// `DoG` inner sigma scale, relative to the output resolution.
pub const DOG_SIGMA_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "dog_sigma_scale",
    min: 0.0038,
    max: 0.0070,
    description: "DoG inner sigma scale (relative to resolution)",
};

/// Ratio of the outer to inner sigma in the `DoG` kernel.
pub const DOG_RATIO: FloatParamDescriptor = FloatParamDescriptor {
    name: "dog_ratio",
    min: 2.2,
    max: 3.0,
    description: "DoG outer/inner sigma ratio",
};

/// Tight glow enhancement strength.
pub const GLOW_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "glow_strength",
    min: 0.18,
    max: 0.40,
    description: "Tight glow enhancement strength",
};

/// Luminance threshold above which glow is activated.
pub const GLOW_THRESHOLD: FloatParamDescriptor = FloatParamDescriptor {
    name: "glow_threshold",
    min: 0.62,
    max: 0.78,
    description: "Luminance threshold for glow activation",
};

/// Glow radius scale, relative to the output resolution.
pub const GLOW_RADIUS_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "glow_radius_scale",
    min: 0.0025,
    max: 0.0045,
    description: "Glow radius scale (relative to resolution)",
};

/// Glow falloff sharpness exponent.
pub const GLOW_SHARPNESS: FloatParamDescriptor = FloatParamDescriptor {
    name: "glow_sharpness",
    min: 2.2,
    max: 3.4,
    description: "Glow falloff sharpness",
};

/// Color saturation boost applied within glow regions.
pub const GLOW_SATURATION_BOOST: FloatParamDescriptor = FloatParamDescriptor {
    name: "glow_saturation_boost",
    min: 0.15,
    max: 0.35,
    description: "Color saturation boost in glows",
};

// ---------------------------------------------------------------------------
// Chromatic effects
// ---------------------------------------------------------------------------

/// Prismatic color separation strength for chromatic bloom.
pub const CHROMATIC_BLOOM_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "chromatic_bloom_strength",
    min: 0.30,
    max: 0.55,
    description: "Prismatic color separation strength",
};

/// Chromatic bloom blur radius scale.
pub const CHROMATIC_BLOOM_RADIUS_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "chromatic_bloom_radius_scale",
    min: 0.0035,
    max: 0.0065,
    description: "Chromatic bloom radius scale",
};

/// RGB channel separation distance scale for chromatic bloom.
pub const CHROMATIC_BLOOM_SEPARATION_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "chromatic_bloom_separation_scale",
    min: 0.0008,
    max: 0.0016,
    description: "RGB channel separation distance scale",
};

/// Luminance threshold for chromatic bloom activation.
pub const CHROMATIC_BLOOM_THRESHOLD: FloatParamDescriptor = FloatParamDescriptor {
    name: "chromatic_bloom_threshold",
    min: 0.18,
    max: 0.28,
    description: "Luminance threshold for chromatic bloom",
};

// ---------------------------------------------------------------------------
// Perceptual blur
// ---------------------------------------------------------------------------

/// `OKLab`-space perceptual blur strength.
pub const PERCEPTUAL_BLUR_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "perceptual_blur_strength",
    min: 0.40,
    max: 0.60,
    description: "OKLab perceptual blur strength",
};

// ---------------------------------------------------------------------------
// Color grading
// ---------------------------------------------------------------------------

/// Overall cinematic color grading strength.
pub const COLOR_GRADE_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "color_grade_strength",
    min: 0.45,
    max: 0.72,
    description: "Overall cinematic color grading strength",
};

/// Vignette darkness strength.
pub const VIGNETTE_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "vignette_strength",
    min: 0.25,
    max: 0.55,
    description: "Vignette darkness strength",
};

/// Vignette edge softness exponent.
pub const VIGNETTE_SOFTNESS: FloatParamDescriptor = FloatParamDescriptor {
    name: "vignette_softness",
    min: 2.2,
    max: 3.0,
    description: "Vignette edge softness exponent",
};

/// Color vibrance multiplier (values > 1 boost saturation).
pub const VIBRANCE: FloatParamDescriptor = FloatParamDescriptor {
    name: "vibrance",
    min: 1.10,
    max: 1.35,
    description: "Color vibrance multiplier",
};

/// High-pass contrast clarity boost strength.
pub const CLARITY_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "clarity_strength",
    min: 0.25,
    max: 0.45,
    description: "High-pass contrast clarity boost",
};

/// Midtone contrast curve strength.
pub const TONE_CURVE_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "tone_curve_strength",
    min: 0.45,
    max: 0.75,
    description: "Midtone contrast curve strength",
};

// ---------------------------------------------------------------------------
// Gradient mapping
// ---------------------------------------------------------------------------

/// Luxury palette gradient mapping strength.
pub const GRADIENT_MAP_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "gradient_map_strength",
    min: 0.18,
    max: 0.40,
    description: "Luxury palette gradient strength",
};

/// Factor controlling how much of the original hue is preserved during gradient mapping.
pub const GRADIENT_MAP_HUE_PRESERVATION: FloatParamDescriptor = FloatParamDescriptor {
    name: "gradient_map_hue_preservation",
    min: 0.40,
    max: 0.65,
    description: "Original hue preservation factor",
};

/// Luxury palette index (0--14) for gradient mapping.
pub const GRADIENT_MAP_PALETTE: IntParamDescriptor = IntParamDescriptor {
    name: "gradient_map_palette",
    min: 0,
    max: 14,
    description: "Luxury palette selection (0-14)",
};

// ---------------------------------------------------------------------------
// Material effects
// ---------------------------------------------------------------------------

/// Gem-like iridescent shimmer strength for opalescence.
pub const OPALESCENCE_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "opalescence_strength",
    min: 0.04,
    max: 0.14,
    description: "Gem-like iridescent shimmer strength",
};

/// Opalescence interference pattern scale.
pub const OPALESCENCE_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "opalescence_scale",
    min: 0.007,
    max: 0.012,
    description: "Opalescence pattern scale",
};

/// Number of interference layers in the opalescence effect.
pub const OPALESCENCE_LAYERS: IntParamDescriptor = IntParamDescriptor {
    name: "opalescence_layers",
    min: 2,
    max: 3,
    description: "Number of interference layers",
};

/// Champlevé enamel flow alignment strength.
pub const CHAMPLEVE_FLOW_ALIGNMENT: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_flow_alignment",
    min: 0.45,
    max: 0.75,
    description: "Champlevé flow alignment strength",
};

/// Iridescent interference amplitude for the champlevé effect.
pub const CHAMPLEVE_INTERFERENCE_AMPLITUDE: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_interference_amplitude",
    min: 0.35,
    max: 0.70,
    description: "Iridescent interference amplitude",
};

/// Metallic rim brightness multiplier for champlevé.
pub const CHAMPLEVE_RIM_INTENSITY: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_rim_intensity",
    min: 1.2,
    max: 2.5,
    description: "Metallic rim brightness multiplier",
};

/// Rim warmth (gold tint) blend factor for champlevé.
pub const CHAMPLEVE_RIM_WARMTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_rim_warmth",
    min: 0.40,
    max: 0.80,
    description: "Rim warmth (gold tint) blend factor",
};

/// Interior opaline glow lift for champlevé.
pub const CHAMPLEVE_INTERIOR_LIFT: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_interior_lift",
    min: 0.45,
    max: 0.80,
    description: "Interior opaline glow lift",
};

/// Aether filament flow alignment strength.
pub const AETHER_FLOW_ALIGNMENT: FloatParamDescriptor = FloatParamDescriptor {
    name: "aether_flow_alignment",
    min: 0.55,
    max: 0.90,
    description: "Aether filament flow alignment",
};

/// Volumetric scattering intensity for the aether effect.
pub const AETHER_SCATTERING_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "aether_scattering_strength",
    min: 0.60,
    max: 1.20,
    description: "Volumetric scattering intensity",
};

/// Aether iridescent color shift amplitude.
pub const AETHER_IRIDESCENCE_AMPLITUDE: FloatParamDescriptor = FloatParamDescriptor {
    name: "aether_iridescence_amplitude",
    min: 0.40,
    max: 0.75,
    description: "Aether iridescent color shift amplitude",
};

/// Negative-space caustics intensity for the aether effect.
pub const AETHER_CAUSTIC_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "aether_caustic_strength",
    min: 0.15,
    max: 0.45,
    description: "Negative space caustics intensity",
};

// ---------------------------------------------------------------------------
// Detail & clarity
// ---------------------------------------------------------------------------

/// Local micro-contrast enhancement strength.
pub const MICRO_CONTRAST_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "micro_contrast_strength",
    min: 0.15,
    max: 0.35,
    description: "Local contrast enhancement strength",
};

/// Micro-contrast neighborhood radius in pixels.
pub const MICRO_CONTRAST_RADIUS: IntParamDescriptor = IntParamDescriptor {
    name: "micro_contrast_radius",
    min: 3,
    max: 6,
    description: "Micro-contrast neighborhood radius",
};

/// Edge luminance brightening strength.
pub const EDGE_LUMINANCE_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "edge_luminance_strength",
    min: 0.12,
    max: 0.30,
    description: "Edge brightening strength",
};

/// Edge detection sensitivity threshold for luminance brightening.
pub const EDGE_LUMINANCE_THRESHOLD: FloatParamDescriptor = FloatParamDescriptor {
    name: "edge_luminance_threshold",
    min: 0.15,
    max: 0.25,
    description: "Edge detection sensitivity threshold",
};

/// Edge brightness multiplier.
pub const EDGE_LUMINANCE_BRIGHTNESS_BOOST: FloatParamDescriptor = FloatParamDescriptor {
    name: "edge_luminance_brightness_boost",
    min: 0.20,
    max: 0.40,
    description: "Edge brightness multiplier",
};

// ---------------------------------------------------------------------------
// Atmospheric
// ---------------------------------------------------------------------------

/// Atmospheric perspective strength.
pub const ATMOSPHERIC_DEPTH_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "atmospheric_depth_strength",
    min: 0.06,
    max: 0.18,
    description: "Atmospheric perspective strength",
};

/// Depth-based desaturation amount.
pub const ATMOSPHERIC_DESATURATION: FloatParamDescriptor = FloatParamDescriptor {
    name: "atmospheric_desaturation",
    min: 0.08,
    max: 0.22,
    description: "Depth-based desaturation",
};

/// Depth-based darkening amount.
pub const ATMOSPHERIC_DARKENING: FloatParamDescriptor = FloatParamDescriptor {
    name: "atmospheric_darkening",
    min: 0.04,
    max: 0.12,
    description: "Depth-based darkening",
};

/// Atmospheric fog color red component (dark tones).
pub const ATMOSPHERIC_FOG_COLOR_R: FloatParamDescriptor = FloatParamDescriptor {
    name: "atmospheric_fog_color_r",
    min: 0.02,
    max: 0.10,
    description: "Atmospheric fog color red component (dark tones)",
};

/// Atmospheric fog color green component (dark tones).
pub const ATMOSPHERIC_FOG_COLOR_G: FloatParamDescriptor = FloatParamDescriptor {
    name: "atmospheric_fog_color_g",
    min: 0.04,
    max: 0.12,
    description: "Atmospheric fog color green component (dark tones)",
};

/// Atmospheric fog color blue component (dark tones).
pub const ATMOSPHERIC_FOG_COLOR_B: FloatParamDescriptor = FloatParamDescriptor {
    name: "atmospheric_fog_color_b",
    min: 0.08,
    max: 0.18,
    description: "Atmospheric fog color blue component (dark tones)",
};

/// Canvas/surface texture strength.
pub const FINE_TEXTURE_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "fine_texture_strength",
    min: 0.06,
    max: 0.18,
    description: "Canvas/surface texture strength",
};

/// Texture feature scale.
pub const FINE_TEXTURE_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "fine_texture_scale",
    min: 0.0012,
    max: 0.0022,
    description: "Texture feature scale",
};

/// Texture contrast intensity.
pub const FINE_TEXTURE_CONTRAST: FloatParamDescriptor = FloatParamDescriptor {
    name: "fine_texture_contrast",
    min: 0.25,
    max: 0.42,
    description: "Texture contrast intensity",
};

// ---------------------------------------------------------------------------
// HDR & exposure
// ---------------------------------------------------------------------------

/// HDR line-alpha scale multiplier.
pub const HDR_SCALE: FloatParamDescriptor = FloatParamDescriptor {
    name: "hdr_scale",
    min: 0.08,
    max: 0.18,
    description: "HDR line alpha scale multiplier",
};

// ---------------------------------------------------------------------------
// Clipping
// ---------------------------------------------------------------------------

/// Black-point percentile clipping threshold.
pub const CLIP_BLACK: FloatParamDescriptor = FloatParamDescriptor {
    name: "clip_black",
    min: 0.008,
    max: 0.015,
    description: "Black point percentile clipping",
};

/// White-point percentile clipping threshold.
pub const CLIP_WHITE: FloatParamDescriptor = FloatParamDescriptor {
    name: "clip_white",
    min: 0.985,
    max: 0.995,
    description: "White point percentile clipping",
};

// ---------------------------------------------------------------------------
// Nebula
// ---------------------------------------------------------------------------

/// Nebula cloud background opacity.
///
/// Chosen so even the highest value keeps the nebula firmly in the
/// background (never brighter than the trajectory subject) while the
/// lower bound guarantees a visible cosmic atmosphere.
pub const NEBULA_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "nebula_strength",
    min: 0.04,
    max: 0.22,
    description: "Nebula cloud background opacity",
};

/// Nebula noise detail octaves.
pub const NEBULA_OCTAVES: IntParamDescriptor = IntParamDescriptor {
    name: "nebula_octaves",
    min: 3,
    max: 5,
    description: "Nebula noise detail octaves",
};

/// Nebula noise base frequency.
pub const NEBULA_BASE_FREQUENCY: FloatParamDescriptor = FloatParamDescriptor {
    name: "nebula_base_frequency",
    min: 0.0010,
    max: 0.0028,
    description: "Nebula noise base frequency",
};

// ---------------------------------------------------------------------------
// Composition / vignette offset
// ---------------------------------------------------------------------------

/// Vignette focal-point horizontal offset (fraction of half-width).
pub const VIGNETTE_OFFSET_X: FloatParamDescriptor = FloatParamDescriptor {
    name: "vignette_offset_x",
    min: -0.18,
    max: 0.18,
    description: "Vignette focal-point horizontal offset",
};

/// Vignette focal-point vertical offset (fraction of half-height).
pub const VIGNETTE_OFFSET_Y: FloatParamDescriptor = FloatParamDescriptor {
    name: "vignette_offset_y",
    min: -0.12,
    max: 0.12,
    description: "Vignette focal-point vertical offset",
};

// ---------------------------------------------------------------------------
// Fine texture material params
// ---------------------------------------------------------------------------

/// Fine-texture orientation (radians).
pub const FINE_TEXTURE_ANGLE: FloatParamDescriptor = FloatParamDescriptor {
    name: "fine_texture_angle",
    min: 0.0,
    max: std::f64::consts::PI,
    description: "Fine texture orientation angle (radians)",
};

/// Fine-texture anisotropy (1.0 = isotropic).
pub const FINE_TEXTURE_ANISOTROPY: FloatParamDescriptor = FloatParamDescriptor {
    name: "fine_texture_anisotropy",
    min: 1.0,
    max: 2.8,
    description: "Fine texture anisotropy ratio",
};

// ---------------------------------------------------------------------------
// Champleve material params
// ---------------------------------------------------------------------------

/// Champlevé cell density multiplier.
pub const CHAMPLEVE_CELL_DENSITY: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_cell_density",
    min: 0.7,
    max: 2.1,
    description: "Champlevé cell density",
};

/// Champlevé rim sharpness (0..1).
pub const CHAMPLEVE_RIM_SHARPNESS: FloatParamDescriptor = FloatParamDescriptor {
    name: "champleve_rim_sharpness",
    min: 0.25,
    max: 0.90,
    description: "Champlevé rim sharpness",
};

// ---------------------------------------------------------------------------
// Aether material params
// ---------------------------------------------------------------------------

/// Aether filament density.
pub const AETHER_FILAMENT_DENSITY: FloatParamDescriptor = FloatParamDescriptor {
    name: "aether_filament_density",
    min: 0.6,
    max: 2.0,
    description: "Aether filament density",
};

/// Aether iridescence frequency (cycles across frame).
pub const AETHER_IRIDESCENCE_FREQUENCY: FloatParamDescriptor = FloatParamDescriptor {
    name: "aether_iridescence_frequency",
    min: 0.6,
    max: 2.4,
    description: "Aether iridescence frequency",
};

// ---------------------------------------------------------------------------
// Opalescence material params
// ---------------------------------------------------------------------------

/// Opalescence pearl-sheen amount (0..1).
pub const OPALESCENCE_PEARL_SHEEN: FloatParamDescriptor = FloatParamDescriptor {
    name: "opalescence_pearl_sheen",
    min: 0.20,
    max: 0.80,
    description: "Opalescence pearl sheen",
};

/// Opalescence chromatic-shift (0..1).
pub const OPALESCENCE_CHROMATIC_SHIFT: FloatParamDescriptor = FloatParamDescriptor {
    name: "opalescence_chromatic_shift",
    min: 0.10,
    max: 0.50,
    description: "Opalescence chromatic shift",
};

// ---------------------------------------------------------------------------
// Chromatic dispersion
// ---------------------------------------------------------------------------

/// Radial chromatic dispersion strength (0..1).
pub const DISPERSION_STRENGTH: FloatParamDescriptor = FloatParamDescriptor {
    name: "dispersion_strength",
    min: 0.0,
    max: 0.35,
    description: "Radial chromatic dispersion strength",
};

/// Probability that a style-influenced bloom mode is selected (vs
/// trusting the style-bundle default). Currently unused because the
/// bloom mode is chosen deterministically from the style — kept here
/// for symmetry with other enable probabilities in case randomization
/// is re-introduced.
pub const ENABLE_PROB_BLOOM_MODE_OVERRIDE: f64 = 0.0;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const ALL_FLOAT_DESCRIPTORS: &[&FloatParamDescriptor] = &[
        &EQUIL_CHAOS_RATIO,
        &BLUR_STRENGTH,
        &BLUR_RADIUS_SCALE,
        &BLUR_CORE_BRIGHTNESS,
        &DOG_STRENGTH,
        &DOG_SIGMA_SCALE,
        &DOG_RATIO,
        &GLOW_STRENGTH,
        &GLOW_THRESHOLD,
        &GLOW_RADIUS_SCALE,
        &GLOW_SHARPNESS,
        &GLOW_SATURATION_BOOST,
        &CHROMATIC_BLOOM_STRENGTH,
        &CHROMATIC_BLOOM_RADIUS_SCALE,
        &CHROMATIC_BLOOM_SEPARATION_SCALE,
        &CHROMATIC_BLOOM_THRESHOLD,
        &PERCEPTUAL_BLUR_STRENGTH,
        &COLOR_GRADE_STRENGTH,
        &VIGNETTE_STRENGTH,
        &VIGNETTE_SOFTNESS,
        &VIBRANCE,
        &CLARITY_STRENGTH,
        &TONE_CURVE_STRENGTH,
        &GRADIENT_MAP_STRENGTH,
        &GRADIENT_MAP_HUE_PRESERVATION,
        &OPALESCENCE_STRENGTH,
        &OPALESCENCE_SCALE,
        &CHAMPLEVE_FLOW_ALIGNMENT,
        &CHAMPLEVE_INTERFERENCE_AMPLITUDE,
        &CHAMPLEVE_RIM_INTENSITY,
        &CHAMPLEVE_RIM_WARMTH,
        &CHAMPLEVE_INTERIOR_LIFT,
        &AETHER_FLOW_ALIGNMENT,
        &AETHER_SCATTERING_STRENGTH,
        &AETHER_IRIDESCENCE_AMPLITUDE,
        &AETHER_CAUSTIC_STRENGTH,
        &MICRO_CONTRAST_STRENGTH,
        &EDGE_LUMINANCE_STRENGTH,
        &EDGE_LUMINANCE_THRESHOLD,
        &EDGE_LUMINANCE_BRIGHTNESS_BOOST,
        &ATMOSPHERIC_DEPTH_STRENGTH,
        &ATMOSPHERIC_DESATURATION,
        &ATMOSPHERIC_DARKENING,
        &ATMOSPHERIC_FOG_COLOR_R,
        &ATMOSPHERIC_FOG_COLOR_G,
        &ATMOSPHERIC_FOG_COLOR_B,
        &FINE_TEXTURE_STRENGTH,
        &FINE_TEXTURE_SCALE,
        &FINE_TEXTURE_CONTRAST,
        &HDR_SCALE,
        &CLIP_BLACK,
        &CLIP_WHITE,
        &NEBULA_STRENGTH,
        &NEBULA_BASE_FREQUENCY,
        &VIGNETTE_OFFSET_X,
        &VIGNETTE_OFFSET_Y,
        &FINE_TEXTURE_ANGLE,
        &FINE_TEXTURE_ANISOTROPY,
        &CHAMPLEVE_CELL_DENSITY,
        &CHAMPLEVE_RIM_SHARPNESS,
        &AETHER_FILAMENT_DENSITY,
        &AETHER_IRIDESCENCE_FREQUENCY,
        &OPALESCENCE_PEARL_SHEEN,
        &OPALESCENCE_CHROMATIC_SHIFT,
        &DISPERSION_STRENGTH,
    ];

    const ALL_INT_DESCRIPTORS: &[&IntParamDescriptor] =
        &[&GRADIENT_MAP_PALETTE, &OPALESCENCE_LAYERS, &MICRO_CONTRAST_RADIUS, &NEBULA_OCTAVES];

    const ALL_ENABLE_PROBS: &[(&str, f64)] = &[
        ("bloom", ENABLE_PROB_BLOOM),
        ("glow", ENABLE_PROB_GLOW),
        ("chromatic_bloom", ENABLE_PROB_CHROMATIC_BLOOM),
        ("perceptual_blur", ENABLE_PROB_PERCEPTUAL_BLUR),
        ("micro_contrast", ENABLE_PROB_MICRO_CONTRAST),
        ("gradient_map", ENABLE_PROB_GRADIENT_MAP),
        ("color_grade", ENABLE_PROB_COLOR_GRADE),
        ("champleve", ENABLE_PROB_CHAMPLEVE),
        ("aether", ENABLE_PROB_AETHER),
        ("opalescence", ENABLE_PROB_OPALESCENCE),
        ("edge_luminance", ENABLE_PROB_EDGE_LUMINANCE),
        ("atmospheric_depth", ENABLE_PROB_ATMOSPHERIC_DEPTH),
        ("fine_texture", ENABLE_PROB_FINE_TEXTURE),
    ];

    #[test]
    fn test_float_descriptors_have_valid_ranges() {
        for desc in ALL_FLOAT_DESCRIPTORS {
            assert!(desc.min.is_finite(), "{}: min is not finite", desc.name);
            assert!(desc.max.is_finite(), "{}: max is not finite", desc.name);
            assert!(desc.min <= desc.max, "{}: min ({}) > max ({})", desc.name, desc.min, desc.max);
        }
    }

    #[test]
    fn test_int_descriptors_have_valid_ranges() {
        for desc in ALL_INT_DESCRIPTORS {
            assert!(desc.min <= desc.max, "{}: min ({}) > max ({})", desc.name, desc.min, desc.max);
        }
    }

    #[test]
    fn test_enable_probabilities_in_unit_range() {
        for &(name, prob) in ALL_ENABLE_PROBS {
            assert!((0.0..=1.0).contains(&prob), "{name}: probability {prob} not in [0, 1]");
        }
    }

    #[test]
    fn test_descriptor_names_non_empty() {
        for desc in ALL_FLOAT_DESCRIPTORS {
            assert!(!desc.name.is_empty(), "Float descriptor has empty name");
            assert!(!desc.description.is_empty(), "{}: description is empty", desc.name);
        }
        for desc in ALL_INT_DESCRIPTORS {
            assert!(!desc.name.is_empty(), "Int descriptor has empty name");
            assert!(!desc.description.is_empty(), "{}: description is empty", desc.name);
        }
    }

    #[test]
    fn test_descriptor_names_are_unique() {
        let mut seen = HashSet::new();
        for desc in ALL_FLOAT_DESCRIPTORS {
            assert!(seen.insert(desc.name), "Duplicate float descriptor name: {}", desc.name);
        }
        for desc in ALL_INT_DESCRIPTORS {
            assert!(seen.insert(desc.name), "Duplicate int descriptor name: {}", desc.name);
        }
    }
}
