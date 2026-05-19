//! Data types for randomizable effect configuration.

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
    /// Palette index for gradient mapping (0-14).
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
    /// Resolved champlevé interior brightness lift.
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
    /// Art-directed framing zoom (>1 zooms in, <1 adds negative space).
    pub composition_zoom: f64,
    /// Horizontal art-directed framing offset.
    pub composition_offset_x: f64,
    /// Vertical art-directed framing offset.
    pub composition_offset_y: f64,
    /// Resolved black point clipping threshold.
    pub clip_black: f64,
    /// Resolved white point clipping threshold.
    pub clip_white: f64,
}
