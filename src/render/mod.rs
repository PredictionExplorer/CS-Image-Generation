//! Rendering module: histogram passes, color mapping, line drawing, and output
//!
//! This module provides a complete rendering pipeline for the three-body problem visualization,
//! including coordinate transformations, line drawing, post-processing effects, and video output.

use crate::post_effects::{
    ChromaticBloomConfig, GradientMapConfig, LuxuryPalette, PerceptualBlurConfig,
};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info};

pub static ACES_TWEAK_ENABLED: AtomicBool = AtomicBool::new(true);

// Module declarations
pub mod batch_drawing;
pub mod camera;
pub mod color;
pub mod constants;
pub mod context;
pub mod drawing;
pub mod effect_randomizer;
pub mod effects;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod histogram;
pub mod parameter_descriptors;
pub mod randomizable_config;
pub mod types;
pub mod velocity_hdr;
pub mod video;

// Import from our submodules
use self::batch_drawing::{draw_triangle_batch_spectral_rows, prepare_triangle_vertices};
use self::context::{PixelBuffer, RenderContext};
use self::effects::{EffectConfig, FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use self::error::{RenderError, Result};
use self::histogram::HistogramData;

// Re-export core types and functions for public API compatibility
pub use color::{OklabColor, generate_body_color_sequences};
pub use drawing::{
    LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral, parallel_blur_2d_rgba,
};
pub use effects::{DogBloomConfig, apply_dog_bloom};
pub use types::{ChannelLevels, ToneMappingControls};
pub use video::{VideoEncodingOptions, create_video_from_frames_singlepass};

// Re-export types from dependencies used in public API
pub use crate::spectrum::NUM_BINS;
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

    pub fn as_str(self) -> &'static str {
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
pub(crate) enum FinishOutputMode {
    #[default]
    Still,
    Video,
}

#[derive(Clone, Copy)]
pub struct SpectralScene<'a> {
    pub positions: &'a [Vec<Vector3<f64>>],
    pub colors: &'a [Vec<OklabColor>],
    pub body_alphas: &'a [f64],
}

impl<'a> SpectralScene<'a> {
    pub fn new(
        positions: &'a [Vec<Vector3<f64>>],
        colors: &'a [Vec<OklabColor>],
        body_alphas: &'a [f64],
    ) -> Self {
        Self { positions, colors, body_alphas }
    }

    #[inline]
    pub fn step_count(self) -> usize {
        self.positions[0].len()
    }

    #[inline]
    pub fn triangle_alphas(self) -> [f64; 3] {
        [self.body_alphas[0], self.body_alphas[1], self.body_alphas[2]]
    }
}

#[derive(Clone, Copy)]
pub struct SpectralRenderSettings<'a> {
    pub resolved_config: &'a randomizable_config::ResolvedEffectConfig,
    pub render_config: &'a RenderConfig,
    pub aspect_correction: bool,
}

impl<'a> SpectralRenderSettings<'a> {
    pub fn new(
        resolved_config: &'a randomizable_config::ResolvedEffectConfig,
        render_config: &'a RenderConfig,
        aspect_correction: bool,
    ) -> Self {
        Self { resolved_config, render_config, aspect_correction }
    }
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
#[inline(always)]
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

pub(crate) fn tonemap_to_display_buffer(pixels: &PixelBuffer, levels: &ChannelLevels) -> PixelBuffer {
    pixels
        .par_iter()
        .map(|&(fr, fg, fb, fa)| {
            let mapped = tonemap_core(fr, fg, fb, fa, levels);
            (mapped[0], mapped[1], mapped[2], fa.clamp(0.0, 1.0))
        })
        .collect()
}

pub(crate) fn tonemap_to_display_buffer_into(
    pixels: &PixelBuffer,
    levels: &ChannelLevels,
    dest: &mut PixelBuffer,
) {
    dest.resize(pixels.len(), (0.0, 0.0, 0.0, 0.0));
    dest.par_iter_mut().zip(pixels.par_iter()).for_each(|(out, &(fr, fg, fb, fa))| {
        let mapped = tonemap_core(fr, fg, fb, fa, levels);
        *out = (mapped[0], mapped[1], mapped[2], fa.clamp(0.0, 1.0));
    });
}

pub(crate) fn quantize_display_buffer_to_16bit(pixels: &PixelBuffer) -> Vec<u16> {
    let mut buf_16bit = vec![0u16; pixels.len() * 3];
    buf_16bit.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
    });
    buf_16bit
}

pub(crate) fn quantize_display_buffer_to_16bit_into(pixels: &PixelBuffer, dest: &mut Vec<u16>) {
    dest.resize(pixels.len() * 3, 0u16);
    dest.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
    });
}

/// Fused tonemap + quantize to 16-bit, eliminating the intermediate display buffer.
#[allow(dead_code)]
pub(crate) fn tonemap_and_quantize_to_16bit(
    pixels: &PixelBuffer,
    levels: &ChannelLevels,
    dest: &mut Vec<u16>,
) {
    dest.resize(pixels.len() * 3, 0u16);
    dest.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(fr, fg, fb, fa))| {
        let mapped = tonemap_core(fr, fg, fb, fa, levels);
        chunk[0] = (mapped[0].clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[1] = (mapped[1].clamp(0.0, 1.0) * 65535.0).round() as u16;
        chunk[2] = (mapped[2].clamp(0.0, 1.0) * 65535.0).round() as u16;
    });
}

// ====================== HELPER FUNCTIONS ===========================

/// Build effect configuration from resolved randomizable config
///
/// Creates a fully configured EffectConfig from a ResolvedEffectConfig with all
/// parameters determined (either explicitly set or randomized).
pub(crate) fn build_effect_config_from_resolved(
    resolved: &randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    output_mode: FinishOutputMode,
) -> EffectConfig {
    use crate::oklab::GamutMapMode;
    use crate::post_effects::{
        AetherConfig, AtmosphericDepthConfig, ChampleveConfig, ColorGradeParams,
        EdgeLuminanceConfig, FineTextureConfig, GlowEnhancementConfig, MicroContrastConfig,
        OpalescenceConfig,
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
            BloomMode::Dog.as_str().to_string()
        } else if use_gaussian_bloom {
            BloomMode::Gaussian.as_str().to_string()
        } else {
            BloomMode::None.as_str().to_string()
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
pub fn apply_energy_density_shift(accum_spd: &mut [[f64; NUM_BINS]]) {
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

/// Maximum number of step-chunks.
///
/// On machines with plenty of memory each chunk allocates a full-image SPD
/// buffer (~1012 MB at 1920x1080x64 bins).  We allow up to 64 chunks so that
/// a 64-core server can keep all cores busy during the histogram and
/// final-frame passes where the step count is very large (1M+).
const MAX_STEP_CHUNKS: usize = 64;

/// Minimum step range to justify the overhead of step-chunked parallelism.
/// Set above the video-pass per-checkpoint count (~556) so the video loop uses
/// the zero-overhead scanline-band path, while the histogram pass (~4167) and
/// final-frame render (~1M) still benefit from step-chunking.
const MIN_STEPS_FOR_CHUNKING: usize = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccumulationBackend {
    ParallelScanlines,
    #[cfg(test)]
    SerialReference,
}

#[inline]
pub fn default_accumulation_backend() -> AccumulationBackend {
    AccumulationBackend::ParallelScanlines
}

pub(crate) fn checkpoint_steps(total_steps: usize, frame_interval: usize) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }

    let mut checkpoints = Vec::new();
    let mut checkpoint = frame_interval;
    while checkpoint < total_steps {
        if checkpoint > 0 {
            checkpoints.push(checkpoint);
        }
        checkpoint += frame_interval;
    }

    let final_step = total_steps - 1;
    if checkpoints.last().copied() != Some(final_step) {
        checkpoints.push(final_step);
    }

    checkpoints
}

#[allow(clippy::too_many_arguments)]
fn accumulate_spectral_steps_into_rows(
    accum_spd: &mut [[f64; NUM_BINS]],
    depth_weight: &mut [f64],
    scene: SpectralScene<'_>,
    ctx: &RenderContext,
    velocity_calc: &velocity_hdr::VelocityHdrCalculator<'_>,
    step_start: usize,
    step_end: usize,
    row_start: usize,
    row_end: usize,
    hdr_scale: f64,
) {
    if step_start >= step_end || row_start >= row_end {
        return;
    }

    let triangle_alphas = scene.triangle_alphas();
    for step in step_start..step_end {
        let vertices =
            prepare_triangle_vertices(scene.positions, scene.colors, &triangle_alphas, step, ctx);

        let hdr_mult_01 = velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = velocity_calc.compute_segment_multiplier(step, 2, 0);

        draw_triangle_batch_spectral_rows(
            accum_spd,
            depth_weight,
            ctx.width,
            ctx.height,
            row_start,
            row_end,
            vertices,
            [hdr_mult_01, hdr_mult_12, hdr_mult_20],
            hdr_scale,
        );
    }
}

/// Merge a partial SPD buffer into the main accumulator using parallel iteration.
fn merge_partial_into(dest: &mut [[f64; NUM_BINS]], src: &[[f64; NUM_BINS]]) {
    dest.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
        for bin in 0..NUM_BINS {
            d[bin] += s[bin];
        }
    });
}

/// Merge a partial depth-weight buffer into the main accumulator.
#[allow(dead_code)]
fn merge_depth_into(dest: &mut [f64], src: &[f64]) {
    dest.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
        *d += *s;
    });
}

/// Accumulate steps using scanline-band parallelism (original approach).
/// Used as the inner kernel for both the legacy path and within each step-chunk.
#[allow(clippy::too_many_arguments)]
fn accumulate_with_scanline_bands(
    accum_spd: &mut [[f64; NUM_BINS]],
    depth_weight: Option<&mut [f64]>,
    scene: SpectralScene<'_>,
    ctx: &RenderContext,
    velocity_calc: &velocity_hdr::VelocityHdrCalculator<'_>,
    step_start: usize,
    step_end: usize,
    hdr_scale: f64,
) {
    let band_count = ctx.height_usize;
    if band_count <= 1 {
        accumulate_spectral_steps_into_rows(
            accum_spd, depth_weight.unwrap_or(&mut []),
            scene, ctx, velocity_calc, step_start, step_end, 0, ctx.height_usize,
            hdr_scale,
        );
        return;
    }

    let rows_per_band = ctx.height_usize.div_ceil(band_count);
    let pixels_per_band = ctx.width_usize * rows_per_band;

    match depth_weight {
        Some(dw) => {
            accum_spd
                .par_chunks_mut(pixels_per_band)
                .zip(dw.par_chunks_mut(pixels_per_band))
                .enumerate()
                .for_each(|(band_idx, (spd_band, dw_band))| {
                    let row_start = band_idx * rows_per_band;
                    let row_end = row_start + spd_band.len() / ctx.width_usize;
                    accumulate_spectral_steps_into_rows(
                        spd_band, dw_band, scene, ctx, velocity_calc,
                        step_start, step_end, row_start, row_end, hdr_scale,
                    );
                });
        }
        None => {
            accum_spd.par_chunks_mut(pixels_per_band).enumerate().for_each(|(band_idx, band)| {
                let row_start = band_idx * rows_per_band;
                let row_end = row_start + band.len() / ctx.width_usize;
                accumulate_spectral_steps_into_rows(
                    band, &mut [], scene, ctx, velocity_calc,
                    step_start, step_end, row_start, row_end, hdr_scale,
                );
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn accumulate_spectral_steps(
    accum_spd: &mut [[f64; NUM_BINS]],
    depth_weight: Option<&mut [f64]>,
    scene: SpectralScene<'_>,
    ctx: &RenderContext,
    velocity_calc: &velocity_hdr::VelocityHdrCalculator<'_>,
    step_start: usize,
    step_end: usize,
    hdr_scale: f64,
    backend: AccumulationBackend,
) {
    if step_start >= step_end || accum_spd.is_empty() {
        return;
    }

    match backend {
        AccumulationBackend::ParallelScanlines => {
            let step_count = step_end - step_start;
            let num_threads = rayon::current_num_threads().max(1);

            let use_chunking = step_count >= MIN_STEPS_FOR_CHUNKING
                && num_threads > 1
                && depth_weight.is_none();

            if use_chunking {
                let num_chunks = num_threads.min(MAX_STEP_CHUNKS).min(step_count);
                let steps_per_chunk = step_count.div_ceil(num_chunks);
                let pixel_count = accum_spd.len();

                let partials: Vec<Vec<[f64; NUM_BINS]>> = (0..num_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let cs = step_start + chunk_idx * steps_per_chunk;
                        let ce = (cs + steps_per_chunk).min(step_end);
                        if cs >= ce {
                            return Vec::new();
                        }
                        let mut local = vec![[0.0f64; NUM_BINS]; pixel_count];
                        accumulate_spectral_steps_into_rows(
                            &mut local, &mut [], scene, ctx, velocity_calc, cs, ce, 0,
                            ctx.height_usize, hdr_scale,
                        );
                        local
                    })
                    .collect();

                for partial in &partials {
                    if !partial.is_empty() {
                        merge_partial_into(accum_spd, partial);
                    }
                }
            } else {
                accumulate_with_scanline_bands(
                    accum_spd, depth_weight, scene, ctx, velocity_calc,
                    step_start, step_end, hdr_scale,
                );
            }
        }
        #[cfg(test)]
        AccumulationBackend::SerialReference => {
            accumulate_spectral_steps_into_rows(
                accum_spd,
                depth_weight.unwrap_or(&mut []),
                scene,
                ctx,
                velocity_calc,
                step_start,
                step_end,
                0,
                ctx.height_usize,
                hdr_scale,
            );
        }
    }
}

fn pass_1_build_histogram_spectral_with_backend(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> HistogramData {
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction, .. } = settings;
    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);
    let mut histogram = HistogramData::with_capacity(ctx.pixel_count() * 10);

    let total_steps = scene.step_count();
    let checkpoints = checkpoint_steps(total_steps, frame_interval);
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);
    let mut step_start = 0;

    for &checkpoint_step in &checkpoints {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 1, mode = "spectral", "Histogram pass progress");
        }

        accumulate_spectral_steps(
            &mut accum_spd,
            None,
            scene,
            &ctx,
            &velocity_calc,
            step_start,
            checkpoint_step + 1,
            render_config.hdr_scale,
            backend,
        );

        convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        let rgba_input = std::mem::take(&mut accum_rgba);
        let trajectory_proxy = finish_pipeline
            .process_trajectory(rgba_input, width as usize, height as usize, &frame_params)
            .expect("Failed to process frame during spectral histogram pass");

        let new_samples: Vec<[f64; 3]> = trajectory_proxy
            .par_iter()
            .map(|&(r, g, b, a)| [r * a, g * a, b * a])
            .collect();
        histogram.extend_from_slice(&new_samples);

        // Reclaim the buffer returned by process_trajectory to avoid re-allocating
        accum_rgba = trajectory_proxy;
        accum_rgba.par_iter_mut().for_each(|px| *px = (0.0, 0.0, 0.0, 0.0));

        step_start = checkpoint_step + 1;
    }

    info!("   pass 1 (spectral histogram): 100% done");
    histogram
}

// ====================== PASS 1 (SPECTRAL) ===========================
/// Pass 1: gather global histogram for final color leveling (spectral)
pub fn pass_1_build_histogram_spectral(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
) -> HistogramData {
    pass_1_build_histogram_spectral_with_backend(
        scene,
        frame_interval,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn pass_1_build_histogram_spectral_serial_reference(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
) -> HistogramData {
    pass_1_build_histogram_spectral_with_backend(
        scene,
        frame_interval,
        settings,
        AccumulationBackend::SerialReference,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass_2_write_frames_spectral_serial_reference(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
    last_frame_out: &mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    spd_out: &mut Option<Vec<[f64; NUM_BINS]>>,
    enable_temporal_smoothing: bool,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(
        scene,
        frame_interval,
        levels,
        settings,
        frame_sink,
        last_frame_out,
        spd_out,
        enable_temporal_smoothing,
        AccumulationBackend::SerialReference,
        None,
        None,
        #[cfg(feature = "gpu")]
        None,
    )
}

// ====================== PASS 2 (SPECTRAL) ===========================
/// Pass 2: final frames => color mapping => write frames (spectral, 16-bit output)
///
/// When `camera` and `original_positions` are provided, uses Option A: for each
/// frame, projects ALL accumulated geometry through the current camera and
/// re-rasterizes from scratch, giving true 3D-coherent camera movement.
#[allow(clippy::too_many_arguments)]
fn pass_2_write_frames_spectral_with_backend(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
    last_frame_out: &mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    spd_out: &mut Option<Vec<[f64; NUM_BINS]>>,
    enable_temporal_smoothing: bool,
    backend: AccumulationBackend,
    camera: Option<&camera::Camera3D>,
    original_positions: Option<&[Vec<Vector3<f64>>]>,
    #[cfg(feature = "gpu")] mut gpu_context: Option<&mut gpu::GpuContext>,
) -> Result<()> {
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction } = settings;
    let width = resolved_config.width;
    let height = resolved_config.height;

    let total_steps = scene.step_count();
    let checkpoints = checkpoint_steps(total_steps, frame_interval);

    let use_3d_camera = camera.is_some() && original_positions.is_some();

    let global_bounds = if let (Some(cam), Some(orig)) = (camera, original_positions) {
        Some(cam.compute_global_bounds(orig, &checkpoints))
    } else {
        None
    };

    let ctx = if let Some(ref gb) = global_bounds {
        let mut bounds = *gb;
        if aspect_correction {
            bounds.apply_aspect_correction(width, height);
        }
        RenderContext::new_with_bounds(width, height, bounds)
    } else {
        RenderContext::new(width, height, scene.positions, aspect_correction)
    };

    let pixel_count = ctx.pixel_count();
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; pixel_count];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); pixel_count];
    let mut spare_rgba = vec![(0.0, 0.0, 0.0, 0.0); pixel_count];

    #[cfg(feature = "gpu")]
    if let (Some(gpu), Some(orig)) = (gpu_context.as_deref_mut(), original_positions) {
        gpu.prepare(
            orig,
            scene.colors,
            scene.body_alphas,
            total_steps,
            render_config.hdr_scale as f64,
            width,
            height,
            ctx.bounds(),
            constants::DEFAULT_DT,
        );
    }

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;

    use crate::post_effects::{TemporalSmoothing, TemporalSmoothingConfig};
    let temporal_smoother = if enable_temporal_smoothing {
        Some(TemporalSmoothing::new(TemporalSmoothingConfig {
            blend_factor: 0.10,
            alpha_threshold: 0.01,
        }))
    } else {
        None
    };
    let mut step_start = 0;
    let mut display_buf: PixelBuffer = Vec::with_capacity(pixel_count);
    let mut quant_buf: Vec<u16> = Vec::with_capacity(pixel_count * 3);

    for &checkpoint_step in &checkpoints {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 2, mode = "spectral", "Render pass progress");
        }

        #[cfg(feature = "gpu")]
        let used_gpu = if let (Some(ref gpu), Some(cam), Some(orig)) =
            (gpu_context.as_deref(), camera, original_positions)
        {
            let frame_spd = gpu.render_frame(cam, orig, checkpoint_step, checkpoint_step + 1);
            accum_spd.copy_from_slice(&frame_spd);
            true
        } else {
            false
        };

        #[cfg(not(feature = "gpu"))]
        let used_gpu = false;

        if !used_gpu && use_3d_camera {
            let cam = camera.unwrap();
            let orig = original_positions.unwrap();

            let frame_positions = cam.project_all_positions_at_step(orig, checkpoint_step);

            let frame_scene = SpectralScene::new(
                &frame_positions,
                scene.colors,
                scene.body_alphas,
            );
            let frame_velocity = velocity_hdr::VelocityHdrCalculator::new(
                &frame_positions, dt,
            );

            accum_spd.iter_mut().for_each(|px| *px = [0.0; NUM_BINS]);

            accumulate_spectral_steps(
                &mut accum_spd,
                None,
                frame_scene,
                &ctx,
                &frame_velocity,
                0,
                checkpoint_step + 1,
                render_config.hdr_scale,
                backend,
            );
        } else if !used_gpu {
            let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);
            accumulate_spectral_steps(
                &mut accum_spd,
                None,
                scene,
                &ctx,
                &velocity_calc,
                step_start,
                checkpoint_step + 1,
                render_config.hdr_scale,
                backend,
            );
        }

        convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        std::mem::swap(&mut accum_rgba, &mut spare_rgba);
        let trajectory_pixels = finish_pipeline
            .process_trajectory(spare_rgba, width as usize, height as usize, &frame_params)
            .expect("Failed to process frame during spectral render pass");
        accum_rgba.par_iter_mut().for_each(|px| *px = (0.0, 0.0, 0.0, 0.0));
        spare_rgba = trajectory_pixels;

        tonemap_to_display_buffer_into(&spare_rgba, levels, &mut display_buf);

        let smoothed_display = match &temporal_smoother {
            Some(smoother) => smoother.process_frame(std::mem::take(&mut display_buf)),
            None => std::mem::take(&mut display_buf),
        };

        let final_display = finish_pipeline
            .process_image(smoothed_display, width as usize, height as usize, &frame_params)
            .expect("Failed to process final image finish during spectral render pass");
        quantize_display_buffer_to_16bit_into(&final_display, &mut quant_buf);
        let buf_bytes = crate::utils::u16_slice_as_bytes(&quant_buf);

        frame_sink(buf_bytes)?;
        if checkpoint_step + 1 == total_steps {
            *last_frame_out = ImageBuffer::from_raw(width, height, quant_buf.clone());
        }

        display_buf = final_display;
        step_start = checkpoint_step + 1;
    }

    info!("   pass 2 (spectral render): 100% done");
    *spd_out = Some(accum_spd);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn pass_2_write_frames_spectral(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
    last_frame_out: &mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    spd_out: &mut Option<Vec<[f64; NUM_BINS]>>,
    enable_temporal_smoothing: bool,
    camera: Option<&camera::Camera3D>,
    original_positions: Option<&[Vec<Vector3<f64>>]>,
    #[cfg(feature = "gpu")] gpu_context: Option<&mut gpu::GpuContext>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(
        scene,
        frame_interval,
        levels,
        settings,
        &mut frame_sink,
        last_frame_out,
        spd_out,
        enable_temporal_smoothing,
        default_accumulation_backend(),
        camera,
        original_positions,
        #[cfg(feature = "gpu")]
        gpu_context,
    )
}

/// Render the fully accumulated final frame without writing intermediate video frames.
///
/// This is the correct preview path for still-image QA because it matches the final
/// accumulated composition instead of an early timeline slice.
pub fn render_final_frame_spectral(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_final_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn render_final_frame_spectral_serial_reference(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_final_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        AccumulationBackend::SerialReference,
    )
}

fn render_final_frame_spectral_with_backend(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction } = settings;
    info!("   Rendering final accumulated frame (preview mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        total_steps,
        render_config.hdr_scale,
        backend,
    );

    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let preview_frame_number = total_steps.saturating_sub(1) / frame_interval;
    let frame_params = FrameParams { frame_number: preview_frame_number, density: None };
    let trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .expect("Failed to process final preview frame");

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
    let final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .expect("Failed to process final image finish for preview frame");
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        RenderError::ImageEncoding("Failed to create 16-bit image buffer".to_string())
    })
}

// ====================== SINGLE FRAME RENDERING ===========================
/// Render the first timeline slice only for tests.
#[cfg(test)]
pub(crate) fn render_single_frame_spectral(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_single_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn render_single_frame_spectral_serial_reference(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_single_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        AccumulationBackend::SerialReference,
    )
}

#[cfg(test)]
fn render_single_frame_spectral_with_backend(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction } = settings;
    info!("   Rendering first timeline slice only (legacy test mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let first_frame_step = frame_interval;

    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        first_frame_step + 1,
        render_config.hdr_scale,
        backend,
    );

    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_params = FrameParams { frame_number: 0, density: None };
    let trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .expect("Failed to process test frame");

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
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
    use rayon::ThreadPoolBuilder;

    type SceneData = (Vec<Vec<Vector3<f64>>>, Vec<Vec<OklabColor>>, Vec<f64>);
    type FrameCaptureResult = (Vec<u8>, Option<ImageBuffer<Rgb<u16>, Vec<u16>>>);

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
        }
    }

    fn image_energy(image: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> u64 {
        image.as_raw().iter().map(|&channel| channel as u64).sum()
    }

    fn assert_frame_bytes_eq(actual: &[u8], expected: &[u8], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: frame byte lengths differ");
        if actual != expected {
            let first = actual.iter().zip(expected).position(|(a, b)| a != b).unwrap();
            panic!(
                "{label}: frame bytes differ at index {first} ({} vs {})",
                actual[first], expected[first],
            );
        }
    }

    fn assert_spd_buffers_bits_eq(
        actual: &[[f64; NUM_BINS]],
        expected: &[[f64; NUM_BINS]],
        label: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label}: SPD buffer lengths differ");
        for (pixel_idx, (lhs, rhs)) in actual.iter().zip(expected).enumerate() {
            for (bin_idx, (&lhs_bin, &rhs_bin)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    lhs_bin.to_bits(),
                    rhs_bin.to_bits(),
                    "{label}: pixel {pixel_idx} bin {bin_idx} differed ({lhs_bin} vs {rhs_bin})"
                );
            }
        }
    }

    fn assert_histogram_bits_eq(actual: &HistogramData, expected: &HistogramData, label: &str) {
        assert_eq!(actual.data().len(), expected.data().len(), "{label}: histogram lengths differ");
        for (sample_idx, (lhs, rhs)) in actual.data().iter().zip(expected.data()).enumerate() {
            for channel_idx in 0..3 {
                assert_eq!(
                    lhs[channel_idx].to_bits(),
                    rhs[channel_idx].to_bits(),
                    "{label}: sample {sample_idx} channel {channel_idx} differed"
                );
            }
        }
    }

    fn assert_image_bits_eq(
        actual: &ImageBuffer<Rgb<u16>, Vec<u16>>,
        expected: &ImageBuffer<Rgb<u16>, Vec<u16>>,
        label: &str,
    ) {
        assert_eq!(actual.as_raw(), expected.as_raw(), "{label}: 16-bit image buffers differed");
    }

    fn sample_scene() -> SceneData {
        let positions = vec![
            vec![
                Vector3::new(0.10, 0.10, -0.30),
                Vector3::new(0.16, 0.14, -0.15),
                Vector3::new(0.24, 0.22, 0.05),
                Vector3::new(0.32, 0.28, 0.18),
                Vector3::new(0.38, 0.32, 0.30),
            ],
            vec![
                Vector3::new(0.86, 0.12, 0.24),
                Vector3::new(0.80, 0.18, 0.15),
                Vector3::new(0.72, 0.26, 0.02),
                Vector3::new(0.64, 0.34, -0.12),
                Vector3::new(0.58, 0.42, -0.24),
            ],
            vec![
                Vector3::new(0.45, 0.88, -0.18),
                Vector3::new(0.48, 0.80, -0.08),
                Vector3::new(0.52, 0.72, 0.00),
                Vector3::new(0.56, 0.64, 0.14),
                Vector3::new(0.60, 0.56, 0.26),
            ],
        ];
        let colors = vec![
            vec![
                (0.72, 0.22, 0.08),
                (0.74, 0.21, 0.10),
                (0.76, 0.19, 0.11),
                (0.78, 0.18, 0.12),
                (0.80, 0.17, 0.13),
            ],
            vec![
                (0.70, -0.18, 0.18),
                (0.72, -0.16, 0.17),
                (0.74, -0.14, 0.16),
                (0.76, -0.12, 0.15),
                (0.78, -0.10, 0.14),
            ],
            vec![
                (0.68, 0.04, -0.20),
                (0.70, 0.05, -0.18),
                (0.72, 0.06, -0.16),
                (0.74, 0.07, -0.14),
                (0.76, 0.08, -0.12),
            ],
        ];
        let body_alphas = vec![0.65, 0.85, 0.95];
        (positions, colors, body_alphas)
    }

    fn stylized_resolved_config(width: u32, height: u32) -> ResolvedEffectConfig {
        ResolvedEffectConfig {
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
            enable_fine_texture: false,
            ..baseline_resolved_config(width, height)
        }
    }

    fn derived_levels_from_serial_histogram(
        scene: SpectralScene<'_>,
        frame_interval: usize,
        settings: SpectralRenderSettings<'_>,
    ) -> ChannelLevels {
        let histogram =
            pass_1_build_histogram_spectral_serial_reference(scene, frame_interval, settings);
        let analysis = histogram::analyze_tonemapping(
            histogram.data(),
            settings.resolved_config.clip_black,
            settings.resolved_config.clip_white,
        );
        ChannelLevels::with_tone_mapping(
            analysis.black_r,
            analysis.white_r,
            analysis.black_g,
            analysis.white_g,
            analysis.black_b,
            analysis.white_b,
            ToneMappingControls {
                exposure_scale: analysis.exposure_scale,
                paper_white: constants::DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        )
    }

    fn capture_frame_bytes_with_pool(
        scene: SpectralScene<'_>,
        frame_interval: usize,
        levels: &ChannelLevels,
        settings: SpectralRenderSettings<'_>,
        enable_temporal_smoothing: bool,
        serial_reference: bool,
        thread_count: usize,
    ) -> FrameCaptureResult {
        let mut frame_bytes = Vec::new();
        let mut last_frame = None;

        ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("thread pool should build")
            .install(|| {
                let frame_sink = |bytes: &[u8]| {
                    frame_bytes.extend_from_slice(bytes);
                    Ok(())
                };

                if serial_reference {
                    pass_2_write_frames_spectral_serial_reference(
                        scene,
                        frame_interval,
                        levels,
                        settings,
                        frame_sink,
                        &mut last_frame,
                        &mut None,
                        enable_temporal_smoothing,
                    )
                } else {
                    pass_2_write_frames_spectral(
                        scene,
                        frame_interval,
                        levels,
                        settings,
                        frame_sink,
                        &mut last_frame,
                        &mut None,
                        enable_temporal_smoothing,
                        None,
                        None,
                        #[cfg(feature = "gpu")]
                        None,
                    )
                }
            })
            .expect("frame rendering should succeed");

        (frame_bytes, last_frame)
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
    fn test_scanline_accumulation_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let render_config = RenderConfig { hdr_scale: 3.5, bloom_mode: BloomMode::None };
        let ctx = RenderContext::new(9, 7, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut serial,
            None,
            scene,
            &ctx,
            &velocity_calc,
            0,
            scene.step_count(),
            render_config.hdr_scale,
            AccumulationBackend::SerialReference,
        );

        for thread_count in [1usize, 2, 3, ctx.height_usize] {
            let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    accumulate_spectral_steps(
                        &mut parallel,
                        None,
                        scene,
                        &ctx,
                        &velocity_calc,
                        0,
                        scene.step_count(),
                        render_config.hdr_scale,
                        AccumulationBackend::ParallelScanlines,
                    );
                });
            assert_spd_buffers_bits_eq(
                &parallel,
                &serial,
                &format!("accumulation/threads={thread_count}"),
            );
        }
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
            ..baseline_resolved_config(48, 48)
        };

        let clean_hist = pass_1_build_histogram_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            1,
            SpectralRenderSettings::new(&clean, &render_config, false),
        );

        let styled_hist = pass_1_build_histogram_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            1,
            SpectralRenderSettings::new(&stylized, &render_config, false),
        );

        assert_ne!(clean_hist.data(), styled_hist.data());
    }

    #[test]
    fn test_histogram_pass_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 2.8, bloom_mode: BloomMode::Dog };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let serial = pass_1_build_histogram_spectral_serial_reference(scene, 2, settings);

        for thread_count in [1usize, 2, 3, resolved.height as usize] {
            let parallel = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| pass_1_build_histogram_spectral(scene, 2, settings));
            assert_histogram_bits_eq(
                &parallel,
                &serial,
                &format!("histogram/threads={thread_count}"),
            );
        }
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
        let levels = ChannelLevels::new(0.0, 0.05, 0.0, 0.05, 0.0, 0.05);

        let single_frame = render_single_frame_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            SpectralRenderSettings::new(&resolved, &render_config, false),
        )
        .expect("legacy single-frame preview should render");
        let final_frame = render_final_frame_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            SpectralRenderSettings::new(&resolved, &render_config, false),
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

    #[test]
    fn test_render_previews_parallel_match_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 3.2, bloom_mode: BloomMode::None };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let levels = ChannelLevels::new(0.0, 0.12, 0.0, 0.12, 0.0, 0.12);

        let serial_single =
            render_single_frame_spectral_serial_reference(scene, &levels, settings).unwrap();
        let serial_final =
            render_final_frame_spectral_serial_reference(scene, &levels, settings).unwrap();

        for thread_count in [1usize, 2, 3, resolved.height as usize] {
            let (parallel_single, parallel_final) = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    (
                        render_single_frame_spectral(scene, &levels, settings).unwrap(),
                        render_final_frame_spectral(scene, &levels, settings).unwrap(),
                    )
                });
            assert_image_bits_eq(
                &parallel_single,
                &serial_single,
                &format!("single-preview/threads={thread_count}"),
            );
            assert_image_bits_eq(
                &parallel_final,
                &serial_final,
                &format!("final-preview/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_video_frame_stream_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::None };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let frame_interval = 1usize;
        let levels = derived_levels_from_serial_histogram(scene, frame_interval, settings);

        let (serial_frames, serial_last_frame) =
            capture_frame_bytes_with_pool(scene, frame_interval, &levels, settings, false, true, 1);
        let serial_last_frame = serial_last_frame.expect("serial path should capture last frame");

        for thread_count in [1usize, 2, 3, 4] {
            let (parallel_frames, parallel_last_frame) = capture_frame_bytes_with_pool(
                scene,
                frame_interval,
                &levels,
                settings,
                false,
                false,
                thread_count,
            );
            let parallel_last_frame =
                parallel_last_frame.expect("parallel path should capture last frame");
            assert_frame_bytes_eq(
                &parallel_frames,
                &serial_frames,
                &format!("video-frames/plain/threads={thread_count}"),
            );
            assert_image_bits_eq(
                &parallel_last_frame,
                &serial_last_frame,
                &format!("video-last-frame/plain/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_stylized_video_frame_stream_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = stylized_resolved_config(96, 72);
        let render_config = RenderConfig { hdr_scale: 4.2, bloom_mode: BloomMode::Dog };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let frame_interval = 2usize;
        let levels = derived_levels_from_serial_histogram(scene, frame_interval, settings);

        let (serial_frames, serial_last_frame) =
            capture_frame_bytes_with_pool(scene, frame_interval, &levels, settings, true, true, 1);
        let serial_last_frame = serial_last_frame.expect("serial path should capture last frame");

        for thread_count in [1usize, 2, 3, 4] {
            let (parallel_frames, parallel_last_frame) = capture_frame_bytes_with_pool(
                scene,
                frame_interval,
                &levels,
                settings,
                true,
                false,
                thread_count,
            );
            let parallel_last_frame =
                parallel_last_frame.expect("parallel path should capture last frame");
            assert_frame_bytes_eq(
                &parallel_frames,
                &serial_frames,
                &format!("video-frames/stylized/threads={thread_count}"),
            );
            assert_image_bits_eq(
                &parallel_last_frame,
                &serial_last_frame,
                &format!("video-last-frame/stylized/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_step_chunked_accumulation_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let render_config = RenderConfig { hdr_scale: 3.5, bloom_mode: BloomMode::None };
        let ctx = RenderContext::new(9, 7, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut serial,
            None,
            scene,
            &ctx,
            &velocity_calc,
            0,
            scene.step_count(),
            render_config.hdr_scale,
            AccumulationBackend::SerialReference,
        );

        for thread_count in [1usize, 2, 4, 8] {
            let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    accumulate_spectral_steps(
                        &mut parallel,
                        None,
                        scene,
                        &ctx,
                        &velocity_calc,
                        0,
                        scene.step_count(),
                        render_config.hdr_scale,
                        AccumulationBackend::ParallelScanlines,
                    );
                });
            assert_spd_buffers_bits_eq(
                &parallel,
                &serial,
                &format!("step-chunked/threads={thread_count}"),
            );
        }
    }

    fn make_circular_scene(n_steps: usize) -> SceneData {
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n_steps)
                    .map(|s| {
                        let t = s as f64 / n_steps as f64 * std::f64::consts::TAU;
                        let phase = body as f64 * std::f64::consts::TAU / 3.0;
                        Vector3::new((t + phase).cos() * 0.4, (t + phase).sin() * 0.4, 0.0)
                    })
                    .collect()
            })
            .collect();
        let colors: Vec<Vec<OklabColor>> = (0..3)
            .map(|body| {
                (0..n_steps)
                    .map(|_| match body {
                        0 => (0.72, 0.15, 0.08),
                        1 => (0.68, -0.12, 0.14),
                        _ => (0.65, 0.04, -0.18),
                    })
                    .collect()
            })
            .collect();
        let body_alphas = vec![0.7, 0.8, 0.9];
        (positions, colors, body_alphas)
    }

    fn assert_spd_buffers_approx_eq(
        actual: &[[f64; NUM_BINS]],
        expected: &[[f64; NUM_BINS]],
        label: &str,
        max_relative_err: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label}: buffer lengths differ");
        for (pixel_idx, (lhs, rhs)) in actual.iter().zip(expected).enumerate() {
            for (bin_idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
                if a == b {
                    continue;
                }
                let denom = a.abs().max(b.abs()).max(1e-30);
                let rel = (a - b).abs() / denom;
                assert!(
                    rel <= max_relative_err,
                    "{label}: pixel {pixel_idx} bin {bin_idx}: {a} vs {b} (rel err {rel:.2e})"
                );
            }
        }
    }

    #[test]
    fn test_step_chunked_large_range_close_to_serial() {
        let n_steps = 500;
        let (positions, colors, body_alphas) = make_circular_scene(n_steps);
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(32, 24, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut serial, None, scene, &ctx, &velocity_calc, 0, n_steps, 2.5,
            AccumulationBackend::SerialReference,
        );

        for thread_count in [2usize, 4] {
            let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    accumulate_spectral_steps(
                        &mut parallel, None, scene, &ctx, &velocity_calc, 0, n_steps, 2.5,
                        AccumulationBackend::ParallelScanlines,
                    );
                });
            assert_spd_buffers_approx_eq(
                &parallel, &serial,
                &format!("step-chunked-large-approx/threads={thread_count}"),
                1e-12,
            );
        }
    }

    #[test]
    fn test_step_chunked_deterministic_across_thread_counts() {
        let n_steps = 500;
        let (positions, colors, body_alphas) = make_circular_scene(n_steps);
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(32, 24, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut reference = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                accumulate_spectral_steps(
                    &mut reference, None, scene, &ctx, &velocity_calc, 0, n_steps, 2.5,
                    AccumulationBackend::ParallelScanlines,
                );
            });

        for _ in 0..3 {
            let mut repeated = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(|| {
                    accumulate_spectral_steps(
                        &mut repeated, None, scene, &ctx, &velocity_calc, 0, n_steps, 2.5,
                        AccumulationBackend::ParallelScanlines,
                    );
                });
            assert_spd_buffers_bits_eq(&repeated, &reference, "chunked-determinism");
        }
    }

    #[test]
    fn test_histogram_parallel_collection_matches_serial() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 2.8, bloom_mode: BloomMode::Dog };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);

        let serial = pass_1_build_histogram_spectral_serial_reference(scene, 2, settings);

        for thread_count in [1usize, 2, 4] {
            let parallel = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| pass_1_build_histogram_spectral(scene, 2, settings));
            assert_histogram_bits_eq(
                &parallel,
                &serial,
                &format!("histogram-parallel/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_merge_partial_is_additive() {
        let mut dest = vec![[1.0f64; NUM_BINS]; 10];
        let src = vec![[2.0f64; NUM_BINS]; 10];
        merge_partial_into(&mut dest, &src);
        for pixel in &dest {
            for &bin in pixel {
                assert_eq!(bin, 3.0);
            }
        }
    }

    #[test]
    fn test_step_chunked_auto_selection_threshold() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(9, 7, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut small_range = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut small_range,
            None,
            scene,
            &ctx,
            &velocity_calc,
            0,
            2,
            1.0,
            AccumulationBackend::ParallelScanlines,
        );

        let mut serial_ref = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut serial_ref,
            None,
            scene,
            &ctx,
            &velocity_calc,
            0,
            2,
            1.0,
            AccumulationBackend::SerialReference,
        );
        assert_spd_buffers_bits_eq(&small_range, &serial_ref, "auto-fallback-small-range");
    }

    #[test]
    fn test_tonemap_into_matches_allocating() {
        let pixels: PixelBuffer = (0..200)
            .map(|i| {
                let t = i as f64 / 200.0;
                (t * 0.8, (1.0 - t) * 0.6, 0.3, 0.9)
            })
            .collect();
        let levels = default_levels();

        let allocating = tonemap_to_display_buffer(&pixels, &levels);
        let mut reused = Vec::new();
        tonemap_to_display_buffer_into(&pixels, &levels, &mut reused);

        assert_eq!(allocating.len(), reused.len());
        for (i, (a, b)) in allocating.iter().zip(reused.iter()).enumerate() {
            assert_eq!(a.0.to_bits(), b.0.to_bits(), "R at {i}");
            assert_eq!(a.1.to_bits(), b.1.to_bits(), "G at {i}");
            assert_eq!(a.2.to_bits(), b.2.to_bits(), "B at {i}");
            assert_eq!(a.3.to_bits(), b.3.to_bits(), "A at {i}");
        }
    }

    #[test]
    fn test_quantize_into_matches_allocating() {
        let pixels: PixelBuffer = (0..200)
            .map(|i| {
                let t = i as f64 / 200.0;
                (t, 1.0 - t, 0.5, 1.0)
            })
            .collect();

        let allocating = quantize_display_buffer_to_16bit(&pixels);
        let mut reused = Vec::new();
        quantize_display_buffer_to_16bit_into(&pixels, &mut reused);

        assert_eq!(allocating, reused);
    }

    #[test]
    fn test_small_step_range_uses_scanline_bands_bitwise() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(9, 7, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut serial, None, scene, &ctx, &velocity_calc, 0, scene.step_count(), 2.0,
            AccumulationBackend::SerialReference,
        );

        let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut parallel, None, scene, &ctx, &velocity_calc, 0, scene.step_count(), 2.0,
            AccumulationBackend::ParallelScanlines,
        );

        assert_spd_buffers_bits_eq(&parallel, &serial, "small-step-scanline-bands");
    }

    #[test]
    fn test_large_step_range_uses_step_chunking() {
        let n_steps = 3000;
        let (positions, colors, body_alphas) = make_circular_scene(n_steps);
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(32, 24, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut serial, None, scene, &ctx, &velocity_calc, 0, n_steps, 2.5,
            AccumulationBackend::SerialReference,
        );

        let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut parallel, None, scene, &ctx, &velocity_calc, 0, n_steps, 2.5,
            AccumulationBackend::ParallelScanlines,
        );

        assert_spd_buffers_approx_eq(
            &parallel, &serial,
            "large-step-chunking",
            1e-12,
        );
    }

    #[test]
    fn test_tonemap_monotonicity() {
        let levels = default_levels();
        let mut prev_lum = 0.0_f64;
        for i in 1..=20 {
            let v = i as f64 * 0.05;
            let result = tonemap_core(v, v, v, 1.0, &levels);
            let lum = 0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2];
            assert!(
                lum >= prev_lum - 1e-10,
                "tonemap should be monotonic: v={v} lum={lum} < prev={prev_lum}"
            );
            prev_lum = lum;
        }
    }

    #[test]
    fn test_tonemap_and_quantize_matches_two_step() {
        let pixels: PixelBuffer = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                (t * 0.8, (1.0 - t) * 0.6, 0.3, 0.9)
            })
            .collect();
        let levels = default_levels();

        let display = tonemap_to_display_buffer(&pixels, &levels);
        let two_step = quantize_display_buffer_to_16bit(&display);

        let mut fused = Vec::new();
        tonemap_and_quantize_to_16bit(&pixels, &levels, &mut fused);

        assert_eq!(two_step, fused, "fused tonemap+quantize should match two-step");
    }

    #[test]
    fn test_histogram_all_identical_samples() {
        let scene_data = sample_scene();
        let resolved = baseline_resolved_config(16, 16);
        let render_config = RenderConfig { hdr_scale: 1.0, bloom_mode: BloomMode::None };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let scene = SpectralScene::new(&scene_data.0, &scene_data.1, &scene_data.2);
        let histogram = pass_1_build_histogram_spectral(scene, 2, settings);
        assert!(!histogram.data().is_empty(), "histogram should have samples even for tiny scene");
    }

    #[test]
    fn test_effect_chain_disabled_effects_are_identity() {
        let config = EffectConfig {
            bloom_mode: "none".to_string(),
            blur_radius_px: 0,
            blur_strength: 0.0,
            blur_core_brightness: 1.0,
            dog_config: DogBloomConfig::default(),
            perceptual_blur_enabled: false,
            perceptual_blur_config: None,
            color_grade_enabled: false,
            color_grade_params: crate::post_effects::ColorGradeParams::default(),
            gradient_map_enabled: false,
            gradient_map_config: crate::post_effects::GradientMapConfig::default(),
            champleve_enabled: false,
            champleve_config: crate::post_effects::ChampleveConfig::default(),
            aether_enabled: false,
            aether_config: crate::post_effects::aether::AetherConfig::default(),
            chromatic_bloom_enabled: false,
            chromatic_bloom_config: crate::post_effects::ChromaticBloomConfig::default(),
            opalescence_enabled: false,
            opalescence_config: crate::post_effects::OpalescenceConfig::default(),
            edge_luminance_enabled: false,
            edge_luminance_config: crate::post_effects::EdgeLuminanceConfig::default(),
            micro_contrast_enabled: false,
            micro_contrast_config: crate::post_effects::MicroContrastConfig::default(),
            glow_enhancement_enabled: false,
            glow_enhancement_config: crate::post_effects::GlowEnhancementConfig::default(),
            atmospheric_depth_enabled: false,
            atmospheric_depth_config: crate::post_effects::AtmosphericDepthConfig::default(),
            fine_texture_enabled: false,
            fine_texture_config: crate::post_effects::FineTextureConfig::default(),
        };
        let pipeline = FinishEffectPipeline::new(config);
        assert_eq!(pipeline.trajectory_len(), 0);
        assert_eq!(pipeline.image_len(), 0);

        let input: PixelBuffer = (0..64)
            .map(|i| (i as f64 / 64.0, 0.5, 0.3, 0.9))
            .collect();
        let frame_params = FrameParams { frame_number: 0, density: None };
        let output = pipeline.process_trajectory(input.clone(), 8, 8, &frame_params).unwrap();
        assert_eq!(input, output, "empty chain should be identity");
    }

    #[test]
    fn test_merge_partial_is_commutative() {
        let a = vec![[1.0f64; NUM_BINS]; 5];
        let b = vec![[2.0f64; NUM_BINS]; 5];
        let c = vec![[3.0f64; NUM_BINS]; 5];

        let mut path1 = a.clone();
        merge_partial_into(&mut path1, &b);
        merge_partial_into(&mut path1, &c);

        let mut path2 = a.clone();
        merge_partial_into(&mut path2, &c);
        merge_partial_into(&mut path2, &b);

        for (i, (p1, p2)) in path1.iter().zip(path2.iter()).enumerate() {
            for bin in 0..NUM_BINS {
                assert_eq!(
                    p1[bin].to_bits(),
                    p2[bin].to_bits(),
                    "pixel {i} bin {bin}: merge order should not matter"
                );
            }
        }
    }

    #[test]
    fn test_merge_partial_preserves_zeros() {
        let original = vec![[7.5f64; NUM_BINS]; 4];
        let zeros = vec![[0.0f64; NUM_BINS]; 4];
        let mut dest = original.clone();
        merge_partial_into(&mut dest, &zeros);

        for (i, (got, expected)) in dest.iter().zip(original.iter()).enumerate() {
            for bin in 0..NUM_BINS {
                assert_eq!(
                    got[bin].to_bits(),
                    expected[bin].to_bits(),
                    "pixel {i} bin {bin}: merging zeros should be identity"
                );
            }
        }
    }

    #[test]
    fn test_energy_density_shift_idempotent_below_threshold() {
        use constants::ENERGY_DENSITY_SHIFT_THRESHOLD;
        let energy_per_bin = ENERGY_DENSITY_SHIFT_THRESHOLD / (NUM_BINS as f64 * 2.0);
        let original = vec![[energy_per_bin; NUM_BINS]; 4];

        let mut once = original.clone();
        apply_energy_density_shift(&mut once);

        let mut twice = original.clone();
        apply_energy_density_shift(&mut twice);
        apply_energy_density_shift(&mut twice);

        for (i, (o, t)) in once.iter().zip(twice.iter()).enumerate() {
            for bin in 0..NUM_BINS {
                assert_eq!(
                    o[bin].to_bits(),
                    t[bin].to_bits(),
                    "pixel {i} bin {bin}: below-threshold shift should be idempotent"
                );
            }
        }
    }

    // ── Option A camera tests ──────────────────────────────────

    #[test]
    fn test_option_a_single_camera_projection_is_finite() {
        let scene_data = make_circular_scene(500);
        let cam = camera::Camera3D::new(
            &camera::Camera3DConfig::default(),
            &scene_data.0,
        );
        let projected = cam.project_all_positions_at_step(&scene_data.0, 250);
        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0].len(), 500);
        for body in &projected {
            for v in body {
                assert!(v[0].is_finite(), "proj_x not finite: {:?}", v);
                assert!(v[1].is_finite(), "proj_y not finite: {:?}", v);
                assert!(v[2].is_finite(), "depth not finite: {:?}", v);
            }
        }
    }

    #[test]
    fn test_option_a_different_cameras_produce_different_projections() {
        let scene_data = make_circular_scene(500);
        let cam = camera::Camera3D::new(
            &camera::Camera3DConfig::default(),
            &scene_data.0,
        );
        let proj_early = cam.project_all_positions_at_step(&scene_data.0, 0);
        let proj_late = cam.project_all_positions_at_step(&scene_data.0, 499);

        let diff: f64 = proj_early[0]
            .iter()
            .zip(proj_late[0].iter())
            .map(|(a, b)| (a - b).norm())
            .sum();

        assert!(
            diff > 0.1,
            "early and late camera projections should differ significantly (diff={diff})"
        );
    }

    #[test]
    fn test_option_a_global_bounds_covers_all_cameras() {
        let scene_data = make_circular_scene(500);
        let cam = camera::Camera3D::new(
            &camera::Camera3DConfig::default(),
            &scene_data.0,
        );
        let checkpoints: Vec<usize> = (0..500).step_by(50).collect();
        let bounds = cam.compute_global_bounds(&scene_data.0, &checkpoints);

        for &step in &checkpoints {
            let proj = cam.project_all_positions_at_step(&scene_data.0, step);
            for body in &proj {
                for v in body {
                    assert!(
                        v[0] >= bounds.min_x && v[0] <= bounds.max_x,
                        "x={} outside bounds [{}, {}] at step {step}",
                        v[0], bounds.min_x, bounds.max_x
                    );
                    assert!(
                        v[1] >= bounds.min_y && v[1] <= bounds.max_y,
                        "y={} outside bounds [{}, {}] at step {step}",
                        v[1], bounds.min_y, bounds.max_y
                    );
                }
            }
        }
    }

    #[test]
    fn test_option_a_accumulation_produces_nonzero_spd() {
        let (positions, colors, body_alphas) = make_circular_scene(200);
        let cam = camera::Camera3D::new(
            &camera::Camera3DConfig::default(),
            &positions,
        );
        let checkpoints = vec![199usize];
        let bounds = cam.compute_global_bounds(&positions, &checkpoints);
        let frame_positions = cam.project_all_positions_at_step(&positions, 199);

        let w = 64u32;
        let h = 36u32;
        let ctx = context::RenderContext::new_with_bounds(w, h, bounds);
        let pixel_count = ctx.pixel_count();
        let mut accum = vec![[0.0f64; NUM_BINS]; pixel_count];

        let scene = SpectralScene::new(&frame_positions, &colors, &body_alphas);
        let dt = constants::DEFAULT_DT;
        let velocity = velocity_hdr::VelocityHdrCalculator::new(&frame_positions, dt);

        accumulate_spectral_steps(
            &mut accum,
            None,
            scene,
            &ctx,
            &velocity,
            0,
            200,
            3.0,
            AccumulationBackend::ParallelScanlines,
        );

        let total_energy: f64 = accum.iter().flat_map(|px| px.iter()).sum();
        assert!(
            total_energy > 0.0,
            "Option A accumulation should produce non-zero energy"
        );
    }
}
