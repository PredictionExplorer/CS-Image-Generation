//! Rendering module: histogram passes, color mapping, line drawing, and output
//!
//! This module provides a complete rendering pipeline for the three-body problem visualization,
//! including coordinate transformations, line drawing, post-processing effects, and video output.

use crate::post_effects::{
    ChromaticBloomConfig, GradientMapConfig, LuxuryPalette, NebulaCloudConfig, NebulaClouds,
    PerceptualBlurConfig,
};
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info};

pub static ACES_TWEAK_ENABLED: AtomicBool = AtomicBool::new(true);

// Module declarations
pub mod batch_drawing;
pub mod buffer_pool;
pub mod color;
pub mod constants;
pub mod context;
pub mod drawing;
pub mod effect_randomizer;
pub mod effects;
pub mod error;
pub mod histogram;
pub mod parameter_descriptors;
pub mod randomizable_config;
pub mod simd_tonemap;
pub mod types;
pub mod velocity_hdr;
pub mod video;

// Import from our submodules
use self::batch_drawing::{draw_triangle_batch_spectral, prepare_triangle_vertices};
use self::context::{PixelBuffer, RenderContext};
use self::effects::{EffectConfig, FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use self::error::{RenderError, Result};
use self::histogram::HistogramData;

// Re-export core types and functions for public API compatibility
pub use color::{OklabColor, generate_body_color_sequences};
#[allow(unused_imports)]
pub use drawing::{draw_line_segment_aa_spectral_with_dispersion, parallel_blur_2d_rgba};
pub use effects::{DogBloomConfig, apply_dog_bloom};
#[allow(unused_imports)] // Public API re-export for library consumers
pub use histogram::compute_black_white_gamma;
// Re-export all types as part of public library API (not used internally, but part of API contract)
#[allow(unused_imports)] // Public API re-exports for library consumers
pub use types::{
    BloomConfig, BlurConfig, ChannelLevels, HdrConfig, PerceptualBlurSettings, Resolution,
    SceneData, ToneMappingControls,
};
pub use video::{VideoEncodingOptions, create_video_from_frames_singlepass};

// Re-export types from dependencies used in public API
pub use image::{DynamicImage, ImageBuffer, Rgb};

/// Rendering configuration parameters
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BloomMode {
    #[default]
    Dog,
    Gaussian,
    None,
}

impl BloomMode {
    pub fn from_arg(value: &str) -> Self {
        match value {
            v if v.eq_ignore_ascii_case("gaussian") => Self::Gaussian,
            v if v.eq_ignore_ascii_case("none") => Self::None,
            _ => Self::Dog,
        }
    }

    fn as_effect_mode(self) -> &'static str {
        match self {
            Self::Dog => "dog",
            Self::Gaussian => "gaussian",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub hdr_scale: f64,
    pub bloom_mode: BloomMode,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { hdr_scale: constants::DEFAULT_HDR_SCALE, bloom_mode: BloomMode::Dog }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum FinishOutputMode {
    #[default]
    Still,
    Video,
}

#[inline]
fn compress_display_highlights(rgb: [f64; 3], paper_white: f64, rolloff: f64) -> [f64; 3] {
    let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if luminance <= paper_white || luminance <= 1e-10 {
        return rgb;
    }

    let shoulder_span = (1.0 - paper_white).max(1e-6);
    let excess = luminance - paper_white;
    let compressed_luminance =
        paper_white + shoulder_span * (1.0 - (-(excess * rolloff) / shoulder_span).exp());
    let scale = compressed_luminance / luminance;

    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Core tonemapping function (shared logic for both 8-bit and 16-bit)
/// Returns final RGB channels in 0.0-1.0 range
/// Upgraded to AgX for superior color rendition without hue-skewing in extreme highlights
#[inline]
fn tonemap_core(fr: f64, fg: f64, fb: f64, fa: f64, levels: &ChannelLevels) -> [f64; 3] {
    let alpha = fa.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let source = [fr.max(0.0), fg.max(0.0), fb.max(0.0)];
    let premult = [source[0] * alpha, source[1] * alpha, source[2] * alpha];
    if premult[0] <= 0.0 && premult[1] <= 0.0 && premult[2] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let mut leveled = [0.0; 3];
    for i in 0..3 {
        leveled[i] =
            (((premult[i] - levels.black[i]).max(0.0)) / levels.range[i]) * levels.exposure_scale;
    }

    // 0. Matrix Inset (AgX color space)
    let r = leveled[0];
    let g = leveled[1];
    let b = leveled[2];

    let r_in = 0.842479062253094 * r + 0.0784335999999992 * g + 0.0792237451477643 * b;
    let g_in = 0.0423282422610123 * r + 0.878468636469772 * g + 0.0791661274605434 * b;
    let b_in = 0.0423756549057051 * r + 0.0784336000000000 * g + 0.877456439033405 * b;

    // 1. Log2 Allocation
    let min_ev = -10.0;
    let max_ev = 2.5;
    let range = max_ev - min_ev;

    let allocate = |v: f64| -> f64 {
        let val = v.max(1e-10).log2();
        ((val - min_ev) / range).clamp(0.0, 1.0)
    };

    let r_alloc = allocate(r_in);
    let g_alloc = allocate(g_in);
    let b_alloc = allocate(b_in);

    // 2. Spline (approximated with a sigmoid polynomial)
    let spline = |x: f64| -> f64 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x5 * x;
        // High quality fit for AgX base curve
        12.0625 * x6 - 36.3262 * x5 + 39.5298 * x4 - 17.6534 * x3 + 3.0135 * x2 + 0.3707 * x
    };

    let r_spline = spline(r_alloc);
    let g_spline = spline(g_alloc);
    let b_spline = spline(b_alloc);

    // 3. Matrix Outset (AgX Punchy if ACES_TWEAK_ENABLED, else Default)
    let is_punchy = ACES_TWEAK_ENABLED.load(Ordering::Relaxed);

    let (r_out, g_out, b_out) = if is_punchy {
        // AgX Punchy Outset (more contrast, better for generative art)
        (
            1.133276 * r_spline - 0.117109 * g_spline - 0.016167 * b_spline,
            -0.097008 * r_spline + 1.148151 * g_spline - 0.051143 * b_spline,
            -0.008107 * r_spline - 0.031776 * g_spline + 1.039883 * b_spline,
        )
    } else {
        // AgX Default Outset
        (
            1.0987524 * r_spline - 0.0880758 * g_spline - 0.0106766 * b_spline,
            -0.0729567 * r_spline + 1.1114562 * g_spline - 0.0384995 * b_spline,
            -0.0060957 * r_spline - 0.0238959 * g_spline + 1.0299916 * b_spline,
        )
    };

    let compressed = compress_display_highlights(
        [r_out.max(0.0), g_out.max(0.0), b_out.max(0.0)],
        levels.paper_white,
        levels.highlight_rolloff,
    );

    [compressed[0].clamp(0.0, 1.0), compressed[1].clamp(0.0, 1.0), compressed[2].clamp(0.0, 1.0)]
}

/// Tonemap to 8-bit (for legacy support, not currently used)
#[allow(dead_code)]
#[inline]
fn tonemap_to_8bit(fr: f64, fg: f64, fb: f64, fa: f64, levels: &ChannelLevels) -> [u8; 3] {
    let channels = tonemap_core(fr, fg, fb, fa, levels);
    [
        (channels[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (channels[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (channels[2] * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Tonemap to 16-bit (primary output format for maximum precision)
#[cfg(test)]
#[inline]
fn tonemap_to_16bit(fr: f64, fg: f64, fb: f64, fa: f64, levels: &ChannelLevels) -> [u16; 3] {
    let channels = tonemap_core(fr, fg, fb, fa, levels);
    [
        (channels[0] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        (channels[1] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        (channels[2] * 65535.0).round().clamp(0.0, 65535.0) as u16,
    ]
}

/// Save 16-bit image as PNG
///
/// TODO: Add explicit sRGB ICC profile chunk via the `png` crate for strict
/// color-managed viewers. The `image` crate's encoder omits the sRGB chunk,
/// but most viewers assume sRGB for untagged PNGs, so this is cosmetic.
pub fn save_image_as_png_16bit(
    rgb_img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    path: &str,
) -> Result<()> {
    let dyn_img = DynamicImage::ImageRgb16(rgb_img.clone());
    dyn_img.save(path).map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
    info!("   Saved 16-bit PNG (sRGB assumed) => {path}");
    Ok(())
}

fn tonemap_to_display_buffer(pixels: &PixelBuffer, levels: &ChannelLevels) -> PixelBuffer {
    pixels
        .par_iter()
        .map(|&(fr, fg, fb, fa)| {
            let mapped = tonemap_core(fr, fg, fb, fa, levels);
            (mapped[0], mapped[1], mapped[2], fa.clamp(0.0, 1.0))
        })
        .collect()
}

fn quantize_display_buffer_to_16bit(pixels: &PixelBuffer) -> Vec<u16> {
    let mut buf_16bit = vec![0u16; pixels.len() * 3];
    buf_16bit.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
    });
    buf_16bit
}

// ====================== HELPER FUNCTIONS ===========================

/// Generate nebula background buffer (separate from trajectories)
fn generate_nebula_background(
    width: usize,
    height: usize,
    frame_number: usize,
    config: &NebulaCloudConfig,
) -> PixelBuffer {
    // Start with empty buffer (black background)
    let background = vec![(0.0, 0.0, 0.0, 0.0); width * height];

    // Apply nebula effect (which adds color without needing alpha tricks)
    let nebula = NebulaClouds::new(config.clone());
    nebula
        .process_with_time(&background, width, height, frame_number)
        .expect("Failed to generate nebula background")
}

/// Composite background and foreground buffers using a standard premultiplied "over" operator.
/// Background goes first (underneath), then foreground on top
/// Note: Background is in straight alpha format (RGB + coverage alpha)
///       Foreground is in premultiplied alpha format (RGB * alpha + alpha)
fn composite_buffers(background: &PixelBuffer, foreground: &PixelBuffer) -> PixelBuffer {
    background
        .par_iter()
        .zip(foreground.par_iter())
        .map(|(&(br, bg, bb, ba), &(fr, fg, fb, fa))| {
            if fa >= 1.0 {
                (fr, fg, fb, fa)
            } else if fa <= 0.0 {
                (br * ba, bg * ba, bb * ba, ba)
            } else {
                let alpha_out = fa + ba * (1.0 - fa);
                (
                    fr + (br * ba) * (1.0 - fa),
                    fg + (bg * ba) * (1.0 - fa),
                    fb + (bb * ba) * (1.0 - fa),
                    alpha_out,
                )
            }
        })
        .collect()
}

/// Build effect configuration from resolved randomizable config
///
/// Creates a fully configured EffectConfig from a ResolvedEffectConfig with all
/// parameters determined (either explicitly set or randomized).
fn build_effect_config_from_resolved(
    resolved: &randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    output_mode: FinishOutputMode,
) -> EffectConfig {
    use crate::oklab::GamutMapMode;
    use crate::post_effects::{
        AetherConfig, AtmosphericDepthConfig, ChampleveConfig, ColorGradeParams,
        EdgeLuminanceConfig, FineTextureConfig, GlowEnhancementConfig, MicroContrastConfig,
        OpalescenceConfig, fine_texture::TextureType,
    };

    let width = resolved.width as usize;
    let height = resolved.height as usize;
    let min_dim = width.min(height);

    let use_gaussian_bloom =
        resolved.enable_bloom && matches!(render_config.bloom_mode, BloomMode::Gaussian);
    let use_dog_bloom = resolved.enable_bloom && matches!(render_config.bloom_mode, BloomMode::Dog);
    let softness_stack_score = if resolved.enable_bloom { 1.0 } else { 0.0 }
        + if resolved.enable_chromatic_bloom { 0.95 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.85 } else { 0.0 }
        + if resolved.enable_glow { 0.55 } else { 0.0 }
        + if resolved.enable_atmospheric_depth { 0.35 } else { 0.0 };

    let blur_radius_px = if use_gaussian_bloom {
        (resolved.blur_radius_scale * min_dim as f64).round() as usize
    } else {
        0
    };
    let dog_inner_sigma = resolved.dog_sigma_scale * min_dim as f64;
    let glow_radius = (resolved.glow_radius_scale * min_dim as f64).round() as usize;
    let chromatic_bloom_radius =
        (resolved.chromatic_bloom_radius_scale * min_dim as f64).round() as usize;
    let chromatic_bloom_separation = resolved.chromatic_bloom_separation_scale * min_dim as f64;
    let opalescence_scale_abs = resolved.opalescence_scale * ((width * height) as f64).sqrt();
    let fine_texture_scale_abs = resolved.fine_texture_scale * ((width * height) as f64).sqrt();
    let dog_threshold = (0.012_f64
        + if resolved.enable_glow { 0.003_f64 } else { 0.0 }
        + if resolved.enable_chromatic_bloom { 0.004_f64 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.004_f64 } else { 0.0 })
    .min(0.028_f64);

    // Build DoG config from resolved parameters
    let dog_config = DogBloomConfig {
        inner_sigma: dog_inner_sigma,
        outer_ratio: resolved.dog_ratio,
        strength: resolved.dog_strength,
        threshold: dog_threshold,
    };

    // Build perceptual blur config if enabled
    let perceptual_blur_config = if resolved.enable_perceptual_blur {
        let perceptual_radius_scale = if softness_stack_score >= 2.0 { 0.0030 } else { 0.0036 };
        let perceptual_radius =
            (perceptual_radius_scale * min_dim as f64).round().max(1.0) as usize;
        Some(PerceptualBlurConfig {
            radius: perceptual_radius,
            strength: resolved.perceptual_blur_strength,
            gamut_mode: GamutMapMode::PreserveHue,
        })
    } else {
        None
    };

    let gradient_map_enabled = resolved.enable_gradient_map;
    let gradient_map_config = GradientMapConfig {
        palette: LuxuryPalette::from_index(resolved.gradient_map_palette),
        strength: resolved.gradient_map_strength,
        hue_preservation: resolved.gradient_map_hue_preservation,
    };
    let texture_min_dim = resolved.width.min(resolved.height);
    let texture_enabled = resolved.enable_fine_texture && texture_min_dim >= 720;
    let texture_strength_scale = if output_mode == FinishOutputMode::Video { 0.6 } else { 1.0 };

    EffectConfig {
        // Core bloom and blur
        bloom_mode: if use_dog_bloom {
            BloomMode::Dog.as_effect_mode().to_string()
        } else if use_gaussian_bloom {
            BloomMode::Gaussian.as_effect_mode().to_string()
        } else {
            BloomMode::None.as_effect_mode().to_string()
        },
        blur_radius_px,
        blur_strength: resolved.blur_strength,
        blur_core_brightness: resolved.blur_core_brightness,
        dog_config,
        perceptual_blur_enabled: resolved.enable_perceptual_blur,
        perceptual_blur_config,

        // Color manipulation
        color_grade_enabled: resolved.enable_color_grade,
        color_grade_params: ColorGradeParams {
            strength: resolved.color_grade_strength,
            vignette_strength: resolved.vignette_strength,
            vignette_softness: resolved.vignette_softness,
            vibrance: resolved.vibrance,
            clarity_strength: resolved.clarity_strength,
            clarity_radius: (0.0028 * min_dim as f64).round().max(1.0) as usize,
            tone_curve: resolved.tone_curve_strength,
            shadow_tint: constants::DEFAULT_COLOR_GRADE_SHADOW_TINT,
            highlight_tint: constants::DEFAULT_COLOR_GRADE_HIGHLIGHT_TINT,
            palette_wave_strength: 0.25,
        },
        gradient_map_enabled,
        gradient_map_config,

        // Material and iridescence
        champleve_enabled: resolved.enable_champleve,
        champleve_config: ChampleveConfig {
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
        },
        aether_enabled: resolved.enable_aether,
        aether_config: AetherConfig {
            filament_density: constants::DEFAULT_AETHER_FILAMENT_DENSITY,
            flow_alignment: resolved.aether_flow_alignment,
            scattering_strength: resolved.aether_scattering_strength,
            scattering_falloff: constants::DEFAULT_AETHER_SCATTERING_FALLOFF,
            iridescence_amplitude: resolved.aether_iridescence_amplitude,
            iridescence_frequency: constants::DEFAULT_AETHER_IRIDESCENCE_FREQUENCY,
            caustic_strength: resolved.aether_caustic_strength,
            caustic_softness: constants::DEFAULT_AETHER_CAUSTIC_SOFTNESS,
            luxury_mode: true,
        },
        chromatic_bloom_enabled: resolved.enable_chromatic_bloom,
        chromatic_bloom_config: ChromaticBloomConfig {
            radius: chromatic_bloom_radius,
            strength: resolved.chromatic_bloom_strength,
            separation: chromatic_bloom_separation,
            threshold: resolved.chromatic_bloom_threshold,
        },
        opalescence_enabled: resolved.enable_opalescence,
        opalescence_config: OpalescenceConfig {
            strength: resolved.opalescence_strength,
            scale: opalescence_scale_abs,
            layers: resolved.opalescence_layers,
            chromatic_shift: 0.5,   // Fixed
            angle_sensitivity: 0.8, // Fixed
            pearl_sheen: 0.3,       // Fixed
        },

        // Detail and clarity
        edge_luminance_enabled: resolved.enable_edge_luminance,
        edge_luminance_config: EdgeLuminanceConfig {
            strength: resolved.edge_luminance_strength,
            threshold: resolved.edge_luminance_threshold,
            brightness_boost: resolved.edge_luminance_brightness_boost,
            bright_edges_only: true, // Fixed
            min_luminance: 0.2,      // Fixed
        },
        micro_contrast_enabled: resolved.enable_micro_contrast,
        micro_contrast_config: MicroContrastConfig {
            strength: resolved.micro_contrast_strength,
            radius: resolved.micro_contrast_radius,
            edge_threshold: 0.15,  // Fixed
            luminance_weight: 0.7, // Fixed
        },
        glow_enhancement_enabled: resolved.enable_glow,
        glow_enhancement_config: GlowEnhancementConfig {
            strength: resolved.glow_strength,
            threshold: resolved.glow_threshold,
            radius: glow_radius,
            sharpness: resolved.glow_sharpness,
            saturation_boost: resolved.glow_saturation_boost,
        },

        // Atmospheric and surface
        atmospheric_depth_enabled: resolved.enable_atmospheric_depth,
        atmospheric_depth_config: AtmosphericDepthConfig {
            strength: resolved.atmospheric_depth_strength,
            fog_color: (
                resolved.atmospheric_fog_color_r,
                resolved.atmospheric_fog_color_g,
                resolved.atmospheric_fog_color_b,
            ),
            density_threshold: 0.15, // Fixed
            desaturation: resolved.atmospheric_desaturation,
            darkening: resolved.atmospheric_darkening,
            density_radius: 3, // Fixed
        },
        fine_texture_enabled: texture_enabled,
        fine_texture_config: FineTextureConfig {
            texture_type: TextureType::Canvas, // Fixed
            strength: resolved.fine_texture_strength * texture_strength_scale,
            scale: fine_texture_scale_abs,
            contrast: resolved.fine_texture_contrast,
            anisotropy: 0.3, // Fixed
            angle: 0.0,      // Fixed
        },
    }
}

/// Apply energy density wavelength shift to spectral buffer
/// Hot regions (high energy) shift toward red, cool regions stay blue
fn apply_energy_density_shift(accum_spd: &mut [[f64; NUM_BINS]]) {
    use constants::{ENERGY_DENSITY_SHIFT_STRENGTH, ENERGY_DENSITY_SHIFT_THRESHOLD};

    accum_spd.par_iter_mut().for_each(|spd| {
        // Calculate total energy in this pixel
        let total_energy: f64 = spd.iter().sum();

        // If energy is below threshold, no shift needed
        if total_energy < ENERGY_DENSITY_SHIFT_THRESHOLD {
            return;
        }

        // Calculate shift amount (excess energy above threshold)
        let excess_energy = total_energy - ENERGY_DENSITY_SHIFT_THRESHOLD;
        let shift_amount = (excess_energy * ENERGY_DENSITY_SHIFT_STRENGTH).min(1.0);

        // Apply redshift: move energy from lower bins (blue) to higher bins (red)
        // We blur the spectrum toward the red end
        let mut shifted_spd = *spd;
        for i in (1..NUM_BINS).rev() {
            // Each bin receives energy from the bin below it (blueshift → redshift)
            shifted_spd[i] = spd[i] * (1.0 - shift_amount) + spd[i - 1] * shift_amount;
        }
        // First bin only loses energy
        shifted_spd[0] = spd[0] * (1.0 - shift_amount);

        *spd = shifted_spd;
    });
}

// ====================== PASS 1 (SPECTRAL) ===========================
/// Pass 1: gather global histogram for final color leveling (spectral)
#[allow(clippy::too_many_arguments)] // Low-level rendering primitive requires all parameters
pub fn pass_1_build_histogram_spectral(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64],
    resolved_config: &randomizable_config::ResolvedEffectConfig,
    frame_interval: usize,
    _noise_seed: i32,
    render_config: &RenderConfig,
    aspect_correction: bool,
) -> HistogramData {
    let width = resolved_config.width;
    let height = resolved_config.height;
    // Create render context
    let ctx = RenderContext::new(width, height, positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    // Create histogram storage
    let mut histogram = HistogramData::with_capacity(ctx.pixel_count() * 10);

    let total_steps = positions[0].len();
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;

    // Create velocity HDR calculator for efficient multiplier computation
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(positions, dt);

    for step in 0..total_steps {
        if step % chunk_line == 0 {
            let pct = (step as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 1, mode = "spectral", "Histogram pass progress");
        }

        // Prepare triangle vertices with batched data access (better cache locality)
        let vertices = prepare_triangle_vertices(
            positions,
            colors,
            &[body_alphas[0], body_alphas[1], body_alphas[2]],
            step,
            &ctx,
        );

        // Compute velocity-based HDR multipliers using the calculator
        let hdr_mult_01 = velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = velocity_calc.compute_segment_multiplier(step, 2, 0);

        // Draw entire triangle in batch (10-20% faster than individual calls)
        draw_triangle_batch_spectral(
            &mut accum_spd,
            width,
            height,
            vertices[0],
            vertices[1],
            vertices[2],
            hdr_mult_01,
            hdr_mult_12,
            hdr_mult_20,
            render_config.hdr_scale,
        );

        let is_final = step == total_steps - 1;
        if (step > 0 && step % frame_interval == 0) || is_final {
            apply_energy_density_shift(&mut accum_spd);
            convert_spd_buffer_to_rgba(
                &accum_spd,
                &mut accum_rgba,
                width as usize,
                height as usize,
            );

            let frame_params = FrameParams { _frame_number: step / frame_interval, _density: None };
            let rgba_buffer = std::mem::take(&mut accum_rgba);
            let trajectory_proxy = finish_pipeline
                .process_trajectory(rgba_buffer, width as usize, height as usize, &frame_params)
                .expect("Failed to process frame during spectral histogram pass");
            accum_rgba.clear();
            accum_rgba.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));

            // Collect histogram data efficiently
            histogram.reserve(ctx.pixel_count());
            for &(r, g, b, a) in &trajectory_proxy {
                histogram.push(r * a, g * a, b * a);
            }
        }
    }

    info!("   pass 1 (spectral histogram): 100% done");
    histogram
}

// ====================== PASS 2 (SPECTRAL) ===========================
/// Pass 2: final frames => color mapping => write frames (spectral, 16-bit output)
#[allow(clippy::too_many_arguments)] // Low-level rendering primitive requires all parameters
pub fn pass_2_write_frames_spectral(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64],
    resolved_config: &randomizable_config::ResolvedEffectConfig,
    frame_interval: usize,
    black_r: f64,
    white_r: f64,
    black_g: f64,
    white_g: f64,
    black_b: f64,
    white_b: f64,
    noise_seed: i32,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
    last_frame_out: &mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    render_config: &RenderConfig,
    aspect_correction: bool,
    enable_temporal_smoothing: bool,
) -> Result<()> {
    let width = resolved_config.width;
    let height = resolved_config.height;
    // Create render context
    let ctx = RenderContext::new(width, height, positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    // Build effect configuration from resolved config
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    // Create nebula configuration (rendered separately, not in effect chain)
    let nebula_config = NebulaCloudConfig {
        strength: resolved_config.nebula_strength,
        octaves: resolved_config.nebula_octaves,
        base_frequency: resolved_config.nebula_base_frequency,
        lacunarity: 2.0,  // Fixed
        persistence: 0.5, // Fixed
        noise_seed: noise_seed as i64,
        colors: [
            [0.08, 0.12, 0.22], // Deep blue
            [0.15, 0.08, 0.25], // Purple
            [0.25, 0.12, 0.18], // Magenta
            [0.12, 0.15, 0.28], // Blue-violet
        ],
        time_scale: 1.0, // Fixed
        edge_fade: 0.3,  // Fixed
    };

    let total_steps = positions[0].len();
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;

    // Create velocity HDR calculator for efficient multiplier computation
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(positions, dt);

    // Pre-allocate empty background buffer for reuse (optimization: saves 60× 2MB allocations)
    let empty_background = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let levels = ChannelLevels::new(black_r, white_r, black_g, white_g, black_b, white_b);

    // Temporal smoothing for video frame blending (stateful across frames)
    use crate::post_effects::{TemporalSmoothing, TemporalSmoothingConfig};
    let temporal_smoother = if enable_temporal_smoothing {
        Some(TemporalSmoothing::new(TemporalSmoothingConfig {
            blend_factor: 0.10,
            alpha_threshold: 0.01,
        }))
    } else {
        None
    };

    for step in 0..total_steps {
        if step % chunk_line == 0 {
            let pct = (step as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 2, mode = "spectral", "Render pass progress");
        }
        // Prepare triangle vertices with batched data access (better cache locality)
        let vertices = prepare_triangle_vertices(
            positions,
            colors,
            &[body_alphas[0], body_alphas[1], body_alphas[2]],
            step,
            &ctx,
        );

        // Compute velocity-based HDR multipliers using the calculator
        let hdr_mult_01 = velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = velocity_calc.compute_segment_multiplier(step, 2, 0);

        // Draw entire triangle in batch (10-20% faster than individual calls)
        draw_triangle_batch_spectral(
            &mut accum_spd,
            width,
            height,
            vertices[0],
            vertices[1],
            vertices[2],
            hdr_mult_01,
            hdr_mult_12,
            hdr_mult_20,
            render_config.hdr_scale,
        );

        let is_final = step == total_steps - 1;
        if (step > 0 && step % frame_interval == 0) || is_final {
            apply_energy_density_shift(&mut accum_spd);
            convert_spd_buffer_to_rgba(
                &accum_spd,
                &mut accum_rgba,
                width as usize,
                height as usize,
            );

            // Process trajectory finish in linear space before tonemapping
            let frame_params = FrameParams { _frame_number: step / frame_interval, _density: None };

            // Take ownership of accum_rgba to avoid clone
            let rgba_buffer = std::mem::take(&mut accum_rgba);
            let trajectory_pixels = finish_pipeline
                .process_trajectory(rgba_buffer, width as usize, height as usize, &frame_params)
                .expect("Failed to process frame during spectral render pass");
            // Reuse the buffer instead of reallocating - clear and resize to avoid allocation
            accum_rgba.clear();
            accum_rgba.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));

            // Generate nebula background separately (with zero-overhead check)
            let nebula_background = if resolved_config.nebula_strength > 0.0 {
                generate_nebula_background(
                    width as usize,
                    height as usize,
                    step / frame_interval,
                    &nebula_config,
                )
            } else {
                empty_background.clone() // Reuse pre-allocated empty buffer (zero overhead)
            };

            // Composite nebula background UNDER trajectory foreground
            let composited = composite_buffers(&nebula_background, &trajectory_pixels);

            let display_buffer = tonemap_to_display_buffer(&composited, &levels);

            // Apply temporal smoothing in display space before final image finish
            let smoothed_display = match &temporal_smoother {
                Some(smoother) => smoother.process_frame(display_buffer),
                None => display_buffer,
            };

            let final_display = finish_pipeline
                .process_image(smoothed_display, width as usize, height as usize, &frame_params)
                .expect("Failed to process final image finish during spectral render pass");
            let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

            // Convert u16 buffer to bytes for FFmpeg (little-endian rgb48le format)
            let buf_bytes = unsafe {
                std::slice::from_raw_parts(buf_16bit.as_ptr() as *const u8, buf_16bit.len() * 2)
            };

            frame_sink(buf_bytes)?;
            if is_final {
                *last_frame_out = ImageBuffer::from_raw(width, height, buf_16bit);
            }
        }
    }
    info!("   pass 2 (spectral render): 100% done");
    Ok(())
}

/// Render the fully accumulated final frame without writing intermediate video frames.
///
/// This is the correct preview path for still-image QA because it matches the final
/// accumulated composition instead of an early timeline slice.
#[allow(clippy::too_many_arguments)] // Low-level rendering primitive requires all parameters
pub fn render_final_frame_spectral(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64],
    resolved_config: &randomizable_config::ResolvedEffectConfig,
    black_r: f64,
    white_r: f64,
    black_g: f64,
    white_g: f64,
    black_b: f64,
    white_b: f64,
    noise_seed: i32,
    render_config: &RenderConfig,
    aspect_correction: bool,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    info!("   Rendering final accumulated frame (preview mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let nebula_config = NebulaCloudConfig {
        strength: resolved_config.nebula_strength,
        octaves: resolved_config.nebula_octaves,
        base_frequency: resolved_config.nebula_base_frequency,
        lacunarity: 2.0,
        persistence: 0.5,
        noise_seed: noise_seed as i64,
        colors: [[0.08, 0.12, 0.22], [0.15, 0.08, 0.25], [0.25, 0.12, 0.18], [0.12, 0.15, 0.28]],
        time_scale: 1.0,
        edge_fade: 0.3,
    };

    let total_steps = positions[0].len();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(positions, dt);
    let empty_background = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
    let levels = ChannelLevels::new(black_r, white_r, black_g, white_g, black_b, white_b);

    for step in 0..total_steps {
        let vertices = prepare_triangle_vertices(
            positions,
            colors,
            &[body_alphas[0], body_alphas[1], body_alphas[2]],
            step,
            &ctx,
        );

        let hdr_mult_01 = velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = velocity_calc.compute_segment_multiplier(step, 2, 0);

        draw_triangle_batch_spectral(
            &mut accum_spd,
            width,
            height,
            vertices[0],
            vertices[1],
            vertices[2],
            hdr_mult_01,
            hdr_mult_12,
            hdr_mult_20,
            render_config.hdr_scale,
        );
    }

    apply_energy_density_shift(&mut accum_spd);
    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let preview_frame_number = total_steps.saturating_sub(1) / frame_interval;
    let frame_params = FrameParams { _frame_number: preview_frame_number, _density: None };
    let trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .expect("Failed to process final preview frame");

    let nebula_background = if resolved_config.nebula_strength > 0.0 {
        generate_nebula_background(
            width as usize,
            height as usize,
            preview_frame_number,
            &nebula_config,
        )
    } else {
        empty_background
    };

    let composited = composite_buffers(&nebula_background, &trajectory_pixels);
    let display_buffer = tonemap_to_display_buffer(&composited, &levels);
    let final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .expect("Failed to process final image finish for preview frame");
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        RenderError::ImageEncoding("Failed to create 16-bit image buffer".to_string())
    })
}

// ====================== SINGLE FRAME RENDERING ===========================
/// Render the first timeline slice only (legacy quick preview path).
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)] // Low-level rendering primitive requires all parameters
pub fn render_single_frame_spectral(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64],
    resolved_config: &randomizable_config::ResolvedEffectConfig,
    black_r: f64,
    white_r: f64,
    black_g: f64,
    white_g: f64,
    black_b: f64,
    white_b: f64,
    noise_seed: i32,
    render_config: &RenderConfig,
    aspect_correction: bool,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    info!("   Rendering first timeline slice only (legacy test mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    // Create render context
    let ctx = RenderContext::new(width, height, positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    // Build effect configuration from resolved config
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    // Create nebula configuration
    let nebula_config = NebulaCloudConfig {
        strength: resolved_config.nebula_strength,
        octaves: resolved_config.nebula_octaves,
        base_frequency: resolved_config.nebula_base_frequency,
        lacunarity: 2.0,  // Fixed
        persistence: 0.5, // Fixed
        noise_seed: noise_seed as i64,
        colors: [
            [0.08, 0.12, 0.22], // Deep blue
            [0.15, 0.08, 0.25], // Purple
            [0.25, 0.12, 0.18], // Magenta
            [0.12, 0.15, 0.28], // Blue-violet
        ],
        time_scale: 1.0, // Fixed
        edge_fade: 0.3,  // Fixed
    };

    let total_steps = positions[0].len();
    let dt = constants::DEFAULT_DT;

    // Create velocity HDR calculator for efficient multiplier computation
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(positions, dt);

    // Pre-allocate empty background buffer for reuse (optimization)
    let empty_background = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let levels = ChannelLevels::new(black_r, white_r, black_g, white_g, black_b, white_b);

    // Render all trajectory steps up to and including the first output frame interval
    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let first_frame_step = frame_interval;

    for step in 0..=first_frame_step {
        let p0 = positions[0][step];
        let p1 = positions[1][step];
        let p2 = positions[2][step];

        let c0 = colors[0][step];
        let c1 = colors[1][step];
        let c2 = colors[2][step];

        let a0 = body_alphas[0];
        let a1 = body_alphas[1];
        let a2 = body_alphas[2];

        let (x0, y0) = ctx.to_pixel(p0[0], p0[1]);
        let (x1, y1) = ctx.to_pixel(p1[0], p1[1]);
        let (x2, y2) = ctx.to_pixel(p2[0], p2[1]);

        let z0 = p0[2] as f32;
        let z1 = p1[2] as f32;
        let z2 = p2[2] as f32;

        // Compute velocity-based HDR multipliers using the calculator
        let hdr_mult_01 = velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = velocity_calc.compute_segment_multiplier(step, 2, 0);

        draw_line_segment_aa_spectral_with_dispersion(
            &mut accum_spd,
            width,
            height,
            x0,
            y0,
            z0,
            x1,
            y1,
            z1,
            c0,
            c1,
            a0,
            a1,
            render_config.hdr_scale * hdr_mult_01,
            true,
        );
        draw_line_segment_aa_spectral_with_dispersion(
            &mut accum_spd,
            width,
            height,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            c1,
            c2,
            a1,
            a2,
            render_config.hdr_scale * hdr_mult_12,
            true,
        );
        draw_line_segment_aa_spectral_with_dispersion(
            &mut accum_spd,
            width,
            height,
            x2,
            y2,
            z2,
            x0,
            y0,
            z0,
            c2,
            c0,
            a2,
            a0,
            render_config.hdr_scale * hdr_mult_20,
            true,
        );
    }

    // Process the accumulated frame
    apply_energy_density_shift(&mut accum_spd);
    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_params = FrameParams { _frame_number: 0, _density: None };
    let trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .expect("Failed to process test frame");

    // Generate nebula background for frame 0 (with zero-overhead check)
    let nebula_background = if resolved_config.nebula_strength > 0.0 {
        generate_nebula_background(width as usize, height as usize, 0, &nebula_config)
    } else {
        empty_background // Reuse pre-allocated empty buffer (zero overhead)
    };

    // Composite nebula under trajectories
    let composited = composite_buffers(&nebula_background, &trajectory_pixels);
    let display_buffer = tonemap_to_display_buffer(&composited, &levels);
    let final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .expect("Failed to process final image finish");

    // Quantize display buffer to 16-bit
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    // Create ImageBuffer and return
    let image = ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        RenderError::ImageEncoding("Failed to create 16-bit image buffer".to_string())
    })?;

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::randomizable_config::ResolvedEffectConfig;
    use nalgebra::Vector3;

    fn default_levels() -> ChannelLevels {
        ChannelLevels::new(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    }

    fn baseline_resolved_config(width: u32, height: u32) -> ResolvedEffectConfig {
        ResolvedEffectConfig {
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
            blur_strength: 4.0,
            blur_radius_scale: 0.006,
            blur_core_brightness: 10.0,
            dog_strength: 0.3,
            dog_sigma_scale: 0.005,
            dog_ratio: 2.6,
            glow_strength: 0.25,
            glow_threshold: 0.7,
            glow_radius_scale: 0.003,
            glow_sharpness: 2.6,
            glow_saturation_boost: 0.2,
            chromatic_bloom_strength: 0.4,
            chromatic_bloom_radius_scale: 0.005,
            chromatic_bloom_separation_scale: 0.001,
            chromatic_bloom_threshold: 0.2,
            perceptual_blur_strength: 0.45,
            color_grade_strength: 0.55,
            vignette_strength: 0.35,
            vignette_softness: 2.5,
            vibrance: 1.2,
            clarity_strength: 0.3,
            tone_curve_strength: 0.6,
            gradient_map_strength: 0.25,
            gradient_map_hue_preservation: 0.6,
            gradient_map_palette: 0,
            opalescence_strength: 0.08,
            opalescence_scale: 0.01,
            opalescence_layers: 2,
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
            micro_contrast_radius: 4,
            edge_luminance_strength: 0.3,
            edge_luminance_threshold: 0.2,
            edge_luminance_brightness_boost: 0.4,
            atmospheric_depth_strength: 0.1,
            atmospheric_desaturation: 0.12,
            atmospheric_darkening: 0.06,
            atmospheric_fog_color_r: 0.04,
            atmospheric_fog_color_g: 0.07,
            atmospheric_fog_color_b: 0.12,
            fine_texture_strength: 0.12,
            fine_texture_scale: 0.0018,
            fine_texture_contrast: 0.35,
            hdr_scale: 0.12,
            clip_black: 0.01,
            clip_white: 0.99,
            nebula_strength: 0.0,
            nebula_octaves: 4,
            nebula_base_frequency: 0.0015,
        }
    }

    fn image_energy(image: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> u64 {
        image.as_raw().iter().map(|&channel| channel as u64).sum()
    }

    #[test]
    fn test_tonemap_black_produces_black() {
        let result = tonemap_core(0.0, 0.0, 0.0, 0.0, &default_levels());
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tonemap_produces_valid_range() {
        let levels = default_levels();
        for alpha in [0.1, 0.5, 1.0] {
            let result = tonemap_core(0.5, 0.3, 0.8, alpha, &levels);
            for ch in result {
                assert!(ch >= 0.0, "channel {ch} should be non-negative at alpha {alpha}");
                assert!(ch < 2.0, "channel {ch} unreasonably large at alpha {alpha}");
            }
        }
    }

    #[test]
    fn test_tonemap_reserves_paper_white_headroom() {
        let levels = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            crate::render::types::ToneMappingControls {
                exposure_scale: 1.0,
                paper_white: 0.9,
                highlight_rolloff: 2.5,
            },
        );
        let result = tonemap_core(8.0, 8.0, 8.0, 1.0, &levels);

        assert!(result[0] < 1.0);
        assert!(result[1] < 1.0);
        assert!(result[2] < 1.0);
        assert!(result[0] > 0.85, "should still look bright after compression");
    }

    #[test]
    fn test_tonemap_exposure_scale_reduces_hot_input() {
        let unity = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            crate::render::types::ToneMappingControls {
                exposure_scale: 1.0,
                paper_white: 0.92,
                highlight_rolloff: 2.25,
            },
        );
        let reduced = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            crate::render::types::ToneMappingControls {
                exposure_scale: 0.6,
                paper_white: 0.92,
                highlight_rolloff: 2.25,
            },
        );

        let unity_out = tonemap_core(2.0, 2.0, 2.0, 1.0, &unity);
        let reduced_out = tonemap_core(2.0, 2.0, 2.0, 1.0, &reduced);

        assert!(reduced_out[0] < unity_out[0]);
        assert!(reduced_out[1] < unity_out[1]);
        assert!(reduced_out[2] < unity_out[2]);
    }

    #[test]
    fn test_agx_tweak_changes_output() {
        let levels = default_levels();

        ACES_TWEAK_ENABLED.store(true, Ordering::Relaxed);
        let tweaked = tonemap_core(0.5, 0.3, 0.7, 0.8, &levels);

        ACES_TWEAK_ENABLED.store(false, Ordering::Relaxed);
        let original = tonemap_core(0.5, 0.3, 0.7, 0.8, &levels);

        ACES_TWEAK_ENABLED.store(true, Ordering::Relaxed);

        let diff = (tweaked[0] - original[0]).abs()
            + (tweaked[1] - original[1]).abs()
            + (tweaked[2] - original[2]).abs();
        assert!(diff > 1e-6, "AgX punchy tweak should produce different tonemapping");
    }

    #[test]
    fn test_tonemap_16bit_range() {
        let levels = default_levels();
        let result = tonemap_to_16bit(0.5, 0.4, 0.6, 0.9, &levels);
        for ch in result {
            assert!((ch as u32) <= 65535, "16-bit channel {ch} out of range");
        }
    }

    #[test]
    fn test_build_effect_config_uses_dog_bloom_exclusively() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: true, ..baseline_resolved_config(640, 360) };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);

        assert_eq!(effect_config.bloom_mode, "dog");
        assert_eq!(effect_config.blur_radius_px, 0);
        assert!(effect_config.dog_config.inner_sigma > 0.0);
    }

    #[test]
    fn test_build_effect_config_uses_gaussian_bloom_exclusively() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: true, ..baseline_resolved_config(640, 360) };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Gaussian };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);

        assert_eq!(effect_config.bloom_mode, "gaussian");
        assert!(effect_config.blur_radius_px > 0);
    }

    #[test]
    fn test_build_effect_config_disables_texture_for_proxy_resolution() {
        let resolved = ResolvedEffectConfig {
            enable_fine_texture: true,
            ..baseline_resolved_config(640, 360)
        };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);

        assert!(!effect_config.fine_texture_enabled, "proxy-sized renders should skip texture");
    }

    #[test]
    fn test_build_effect_config_scales_texture_for_video() {
        let resolved = ResolvedEffectConfig {
            enable_fine_texture: true,
            fine_texture_strength: 0.2,
            ..baseline_resolved_config(1920, 1080)
        };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let still_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);
        let video_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Video);

        assert!(still_config.fine_texture_enabled);
        assert!(video_config.fine_texture_enabled);
        assert!(
            (video_config.fine_texture_config.strength
                - still_config.fine_texture_config.strength * 0.6)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_build_effect_config_tightens_softness_stack_settings() {
        let resolved = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            ..baseline_resolved_config(1920, 1080)
        };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);
        let perceptual =
            effect_config.perceptual_blur_config.expect("perceptual blur should remain configured");

        assert!(
            effect_config.dog_config.threshold > 0.012,
            "softness stacks should raise the DoG threshold"
        );
        assert!(
            perceptual.radius < (0.0036_f64 * 1080.0).round() as usize,
            "softness stacks should tighten perceptual blur radius"
        );
    }

    #[test]
    fn test_histogram_pass_is_finish_aware() {
        let positions = vec![
            vec![Vector3::new(0.1, 0.1, 0.0), Vector3::new(0.2, 0.2, 0.0)],
            vec![Vector3::new(0.9, 0.1, 0.0), Vector3::new(0.8, 0.2, 0.0)],
            vec![Vector3::new(0.5, 0.9, 0.0), Vector3::new(0.5, 0.8, 0.0)],
        ];
        let colors = vec![
            vec![(0.7, 0.2, 0.1), (0.72, 0.18, 0.12)],
            vec![(0.68, -0.15, 0.2), (0.70, -0.12, 0.18)],
            vec![(0.65, 0.04, -0.18), (0.67, 0.05, -0.16)],
        ];
        let body_alphas = vec![0.8, 0.9, 1.0];
        let render_config = RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::Dog };

        let clean = baseline_resolved_config(48, 48);
        let stylized = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            enable_micro_contrast: true,
            enable_gradient_map: true,
            enable_color_grade: true,
            enable_champleve: true,
            enable_aether: true,
            enable_opalescence: true,
            enable_edge_luminance: true,
            enable_atmospheric_depth: true,
            enable_fine_texture: true,
            nebula_strength: 0.25,
            ..baseline_resolved_config(48, 48)
        };

        let clean_hist = pass_1_build_histogram_spectral(
            &positions,
            &colors,
            &body_alphas,
            &clean,
            1,
            17,
            &render_config,
            false,
        );

        let styled_hist = pass_1_build_histogram_spectral(
            &positions,
            &colors,
            &body_alphas,
            &stylized,
            1,
            999,
            &render_config,
            false,
        );

        assert_ne!(clean_hist.data(), styled_hist.data());
    }

    #[test]
    fn test_render_final_frame_accumulates_late_color_steps() {
        let resolved = baseline_resolved_config(48, 48);
        let render_config = RenderConfig { hdr_scale: 6.0, bloom_mode: BloomMode::None };
        let positions = vec![
            vec![
                Vector3::new(0.1, 0.1, 0.0),
                Vector3::new(0.1, 0.1, 0.0),
                Vector3::new(0.1, 0.1, 0.0),
                Vector3::new(0.1, 0.1, 0.0),
            ],
            vec![
                Vector3::new(0.9, 0.1, 0.0),
                Vector3::new(0.9, 0.1, 0.0),
                Vector3::new(0.9, 0.1, 0.0),
                Vector3::new(0.9, 0.1, 0.0),
            ],
            vec![
                Vector3::new(0.5, 0.9, 0.0),
                Vector3::new(0.5, 0.9, 0.0),
                Vector3::new(0.5, 0.9, 0.0),
                Vector3::new(0.5, 0.9, 0.0),
            ],
        ];
        let colors = vec![
            vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.82, 0.22, 0.08), (0.82, 0.22, 0.08)],
            vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.78, -0.18, 0.15), (0.78, -0.18, 0.15)],
            vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.74, 0.05, -0.22), (0.74, 0.05, -0.22)],
        ];
        let body_alphas = vec![1.0, 1.0, 1.0];

        let single_frame = render_single_frame_spectral(
            &positions,
            &colors,
            &body_alphas,
            &resolved,
            0.0,
            0.05,
            0.0,
            0.05,
            0.0,
            0.05,
            7,
            &render_config,
            false,
        )
        .expect("legacy single-frame preview should render");
        let final_frame = render_final_frame_spectral(
            &positions,
            &colors,
            &body_alphas,
            &resolved,
            0.0,
            0.05,
            0.0,
            0.05,
            0.0,
            0.05,
            7,
            &render_config,
            false,
        )
        .expect("final preview should render");

        let single_energy = image_energy(&single_frame);
        let final_energy = image_energy(&final_frame);

        assert!(final_energy > 0, "final preview should contain visible energy");
        assert!(
            final_energy > single_energy.saturating_mul(20),
            "final preview should retain much more energy than the legacy early-slice preview (single={single_energy}, final={final_energy})"
        );
    }
}
