//! Build concrete post-effect pipeline configuration from resolved render settings.

use super::effects::{DogBloomConfig, EffectConfig};
use super::randomizable_config::ResolvedEffectConfig;
use super::{BloomMode, FinishOutputMode, RenderConfig, constants};
use crate::post_effects::{
    AetherConfig, AtmosphericDepthConfig, ChampleveConfig, ChromaticBloomConfig,
    CrystalFacetConfig, EdgeLuminanceConfig, FineTextureConfig, GradientMapConfig, InkCutConfig,
    LuxuryPalette, MicroContrastConfig, OpalescenceConfig, PerceptualBlurConfig,
    PrismaticSparkleConfig,
};
use crate::utils::f64_to_usize_saturating;

/// Derive the perceptual-blur radius (in pixels) after accounting for the combined
/// softness of all enabled blur/bloom effects. Returns `None` when blur is disabled.
#[must_use]
pub fn compute_softness_radius(
    resolved: &ResolvedEffectConfig,
    bloom_mode: BloomMode,
) -> Option<usize> {
    if !resolved.enable_perceptual_blur {
        return None;
    }

    let use_gaussian_bloom = bloom_mode == BloomMode::Gaussian && resolved.enable_bloom;
    let softness_stack_score = (if use_gaussian_bloom { 1.0 } else { 0.0 })
        + if resolved.enable_chromatic_bloom { 0.8 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.85 } else { 0.0 }
        + if resolved.enable_glow { 0.55 } else { 0.0 }
        + if resolved.enable_atmospheric_depth { 0.35 } else { 0.0 };
    let radius_scale = if softness_stack_score >= 2.0 { 0.0030 } else { 0.0036 };
    let min_dim = resolved.width.min(resolved.height);

    Some(f64_to_usize_saturating((radius_scale * f64::from(min_dim)).round().max(1.0)))
}

fn build_dog_config(resolved: &ResolvedEffectConfig, min_dim: usize) -> DogBloomConfig {
    let dog_inner_sigma = resolved.dog_sigma_scale * min_dim as f64;
    let dog_threshold = (0.012_f64
        + if resolved.enable_glow { 0.003_f64 } else { 0.0 }
        + if resolved.enable_chromatic_bloom { 0.004_f64 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.004_f64 } else { 0.0 })
    .min(0.028_f64);

    DogBloomConfig {
        inner_sigma: dog_inner_sigma,
        outer_ratio: resolved.dog_ratio,
        strength: resolved.dog_strength,
        threshold: dog_threshold,
    }
}

fn build_perceptual_blur_config(
    resolved: &ResolvedEffectConfig,
    bloom_mode: BloomMode,
) -> Option<PerceptualBlurConfig> {
    use crate::oklab::GamutMapMode;
    compute_softness_radius(resolved, bloom_mode).map(|radius| PerceptualBlurConfig {
        radius,
        strength: resolved.perceptual_blur_strength,
        gamut_mode: GamutMapMode::PreserveHue,
    })
}

fn build_chromatic_bloom_config(
    resolved: &ResolvedEffectConfig,
    min_dim: usize,
) -> ChromaticBloomConfig {
    let radius =
        f64_to_usize_saturating((resolved.chromatic_bloom_radius_scale * min_dim as f64).round());
    let separation = resolved.chromatic_bloom_separation_scale * min_dim as f64;
    ChromaticBloomConfig {
        radius,
        strength: resolved.chromatic_bloom_strength,
        separation,
        threshold: resolved.chromatic_bloom_threshold,
    }
}

fn mix_rgb(base: (f64, f64, f64), accent: (f64, f64, f64), amount: f64) -> (f64, f64, f64) {
    let amount = amount.clamp(0.0, 1.0);
    (
        base.0 * (1.0 - amount) + accent.0 * amount,
        base.1 * (1.0 - amount) + accent.1 * amount,
        base.2 * (1.0 - amount) + accent.2 * amount,
    )
}

fn build_color_grade_params(
    resolved: &ResolvedEffectConfig,
    min_dim: usize,
) -> crate::post_effects::ColorGradeParams {
    let palette = LuxuryPalette::from_index(resolved.gradient_map_palette);
    let (shadow_tint, highlight_tint) = palette.color_grade_tints();
    crate::post_effects::ColorGradeParams {
        strength: resolved.color_grade_strength,
        vignette_strength: resolved.vignette_strength,
        vignette_softness: resolved.vignette_softness,
        vibrance: resolved.vibrance,
        clarity_strength: resolved.clarity_strength,
        clarity_radius: (0.0028 * min_dim as f64).round().max(1.0) as usize,
        tone_curve: resolved.tone_curve_strength,
        shadow_tint,
        highlight_tint,
        palette_wave_strength: if resolved.enable_gradient_map { 0.18 } else { 0.24 },
    }
}

fn build_glow_config(
    resolved: &ResolvedEffectConfig,
    min_dim: usize,
) -> crate::post_effects::GlowEnhancementConfig {
    let glow_radius = (resolved.glow_radius_scale * min_dim as f64).round() as usize;
    crate::post_effects::GlowEnhancementConfig {
        strength: resolved.glow_strength,
        threshold: resolved.glow_threshold,
        radius: glow_radius,
        sharpness: resolved.glow_sharpness,
        saturation_boost: resolved.glow_saturation_boost,
    }
}

fn build_champleve_config(resolved: &ResolvedEffectConfig) -> ChampleveConfig {
    ChampleveConfig {
        cell_density: constants::DEFAULT_CHAMPLEVE_CELL_DENSITY,
        flow_alignment: resolved.champleve_flow_alignment,
        interference_amplitude: resolved.champleve_interference_amplitude,
        interference_frequency: constants::DEFAULT_CHAMPLEVE_INTERFERENCE_FREQUENCY,
        rim_intensity: resolved.champleve_rim_intensity,
        rim_warmth: resolved.champleve_rim_warmth,
        rim_sharpness: constants::DEFAULT_CHAMPLEVE_RIM_SHARPNESS,
        interior_lift: resolved.champleve_interior_lift,
        anisotropy: constants::DEFAULT_CHAMPLEVE_ANISOTROPY,
        cell_softness: constants::DEFAULT_CHAMPLEVE_CELL_SOFTNESS,
    }
}

fn build_aether_config(resolved: &ResolvedEffectConfig) -> AetherConfig {
    AetherConfig {
        filament_density: constants::DEFAULT_AETHER_FILAMENT_DENSITY,
        flow_alignment: resolved.aether_flow_alignment,
        scattering_strength: resolved.aether_scattering_strength,
        scattering_falloff: constants::DEFAULT_AETHER_SCATTERING_FALLOFF,
        iridescence_amplitude: resolved.aether_iridescence_amplitude,
        iridescence_frequency: constants::DEFAULT_AETHER_IRIDESCENCE_FREQUENCY,
        caustic_strength: resolved.aether_caustic_strength,
        caustic_softness: constants::DEFAULT_AETHER_CAUSTIC_SOFTNESS,
        luxury_mode: true,
    }
}

fn build_opalescence_config(
    resolved: &ResolvedEffectConfig,
    width: usize,
    height: usize,
) -> OpalescenceConfig {
    let scale_abs = resolved.opalescence_scale * ((width * height) as f64).sqrt();
    OpalescenceConfig {
        strength: resolved.opalescence_strength,
        scale: scale_abs,
        layers: resolved.opalescence_layers,
        chromatic_shift: 0.5,
        angle_sensitivity: 0.8,
        pearl_sheen: 0.3,
    }
}

fn build_edge_luminance_config(resolved: &ResolvedEffectConfig) -> EdgeLuminanceConfig {
    EdgeLuminanceConfig {
        strength: resolved.edge_luminance_strength,
        threshold: resolved.edge_luminance_threshold,
        brightness_boost: resolved.edge_luminance_brightness_boost,
        bright_edges_only: true,
        min_luminance: 0.2,
    }
}

fn build_micro_contrast_config(resolved: &ResolvedEffectConfig) -> MicroContrastConfig {
    MicroContrastConfig {
        strength: resolved.micro_contrast_strength,
        radius: resolved.micro_contrast_radius,
        edge_threshold: 0.15,
        luminance_weight: 0.7,
    }
}

fn build_atmospheric_depth_config(resolved: &ResolvedEffectConfig) -> AtmosphericDepthConfig {
    let palette = LuxuryPalette::from_index(resolved.gradient_map_palette);
    let randomized_fog = (
        resolved.atmospheric_fog_color_r,
        resolved.atmospheric_fog_color_g,
        resolved.atmospheric_fog_color_b,
    );
    let palette_fog = palette.atmospheric_fog_color();
    let fog_color = mix_rgb(
        randomized_fog,
        palette_fog,
        if resolved.enable_gradient_map { 0.70 } else { 0.45 },
    );
    AtmosphericDepthConfig {
        strength: resolved.atmospheric_depth_strength,
        fog_color,
        density_threshold: 0.15,
        desaturation: resolved.atmospheric_desaturation,
        darkening: resolved.atmospheric_darkening,
        density_radius: 3,
    }
}

fn build_fine_texture_config(
    resolved: &ResolvedEffectConfig,
    output_mode: FinishOutputMode,
) -> (bool, FineTextureConfig) {
    let width = resolved.width as usize;
    let height = resolved.height as usize;
    let scale_abs = resolved.fine_texture_scale * ((width * height) as f64).sqrt();
    let min_dim = resolved.width.min(resolved.height);
    let enabled = resolved.enable_fine_texture && min_dim >= 720;
    let strength_scale = if output_mode == FinishOutputMode::Video { 0.6 } else { 1.0 };
    (
        enabled,
        FineTextureConfig {
            strength: resolved.fine_texture_strength * strength_scale,
            scale: scale_abs,
            contrast: resolved.fine_texture_contrast,
            anisotropy: 0.3,
            angle: 0.0,
        },
    )
}

fn build_prismatic_sparkle_config(
    resolved: &ResolvedEffectConfig,
    min_dim: usize,
) -> (bool, PrismaticSparkleConfig) {
    let enabled =
        min_dim >= 720 && (resolved.enable_edge_luminance || resolved.enable_micro_contrast);
    let radius = f64_to_usize_saturating((0.0014 * min_dim as f64).round()).clamp(1, 3);
    let strength = (resolved.edge_luminance_strength * 0.34
        + resolved.micro_contrast_strength * 0.18)
        .clamp(0.08, 0.16);
    let density = if resolved.enable_glow { 0.014 } else { 0.008 };

    (enabled, PrismaticSparkleConfig { strength, threshold: 0.82, density, radius })
}

fn build_crystal_facet_config(
    resolved: &ResolvedEffectConfig,
    min_dim: usize,
) -> (bool, CrystalFacetConfig) {
    let enabled = false;
    let cell_size = f64_to_usize_saturating((0.013 * min_dim as f64).round()).clamp(10, 28);
    let strength =
        (0.04 + resolved.clarity_strength * 0.08 + resolved.micro_contrast_strength * 0.10)
            .clamp(0.06, 0.13);

    (enabled, CrystalFacetConfig { strength, cell_size, threshold: 0.16, highlight_gain: 0.04 })
}

fn build_ink_cut_config(resolved: &ResolvedEffectConfig, _min_dim: usize) -> (bool, InkCutConfig) {
    let enabled = false;
    let strength = (resolved.edge_luminance_strength * 0.42
        + resolved.micro_contrast_strength * 0.16)
        .clamp(0.12, 0.26);
    let threshold = (resolved.edge_luminance_threshold * 0.85).clamp(0.10, 0.20);

    (enabled, InkCutConfig { strength, threshold, darken: 0.18, glint: 0.16 })
}

/// Build a fully populated [`EffectConfig`] from resolved parameters and render settings.
#[must_use]
pub fn build_effect_config_from_resolved(
    resolved: &ResolvedEffectConfig,
    render_config: &RenderConfig,
    output_mode: FinishOutputMode,
) -> EffectConfig {
    let width = resolved.width as usize;
    let height = resolved.height as usize;
    let min_dim = width.min(height);

    let bloom_mode = if resolved.enable_bloom { render_config.bloom_mode } else { BloomMode::None };
    let use_gaussian_bloom = bloom_mode == BloomMode::Gaussian;

    let blur_radius_px = if use_gaussian_bloom {
        (resolved.blur_radius_scale * min_dim as f64).round() as usize
    } else {
        0
    };
    let (fine_texture_enabled, fine_texture_config) =
        build_fine_texture_config(resolved, output_mode);
    let (prismatic_sparkle_enabled, prismatic_sparkle_config) =
        build_prismatic_sparkle_config(resolved, min_dim);
    let (crystal_facets_enabled, crystal_facet_config) =
        build_crystal_facet_config(resolved, min_dim);
    let (ink_cut_edges_enabled, ink_cut_config) = build_ink_cut_config(resolved, min_dim);

    EffectConfig {
        bloom_mode,
        blur_radius_px,
        blur_strength: resolved.blur_strength,
        blur_core_brightness: resolved.blur_core_brightness,
        dog_config: build_dog_config(resolved, min_dim),
        perceptual_blur_enabled: resolved.enable_perceptual_blur,
        perceptual_blur_config: build_perceptual_blur_config(resolved, bloom_mode),

        color_grade_enabled: resolved.enable_color_grade,
        color_grade_params: build_color_grade_params(resolved, min_dim),
        gradient_map_enabled: resolved.enable_gradient_map,
        gradient_map_config: GradientMapConfig {
            palette: LuxuryPalette::from_index(resolved.gradient_map_palette),
            strength: resolved.gradient_map_strength,
            hue_preservation: resolved.gradient_map_hue_preservation,
        },

        champleve_enabled: resolved.enable_champleve,
        champleve_config: build_champleve_config(resolved),
        aether_enabled: resolved.enable_aether,
        aether_config: build_aether_config(resolved),
        chromatic_bloom_enabled: resolved.enable_chromatic_bloom,
        chromatic_bloom_config: build_chromatic_bloom_config(resolved, min_dim),
        opalescence_enabled: resolved.enable_opalescence,
        opalescence_config: build_opalescence_config(resolved, width, height),

        edge_luminance_enabled: resolved.enable_edge_luminance,
        edge_luminance_config: build_edge_luminance_config(resolved),
        micro_contrast_enabled: resolved.enable_micro_contrast,
        micro_contrast_config: build_micro_contrast_config(resolved),
        glow_enhancement_enabled: resolved.enable_glow,
        glow_enhancement_config: build_glow_config(resolved, min_dim),

        atmospheric_depth_enabled: resolved.enable_atmospheric_depth,
        atmospheric_depth_config: build_atmospheric_depth_config(resolved),
        fine_texture_enabled,
        fine_texture_config,
        crystal_facets_enabled,
        crystal_facet_config,
        ink_cut_edges_enabled,
        ink_cut_config,
        prismatic_sparkle_enabled,
        prismatic_sparkle_config,
    }
}
