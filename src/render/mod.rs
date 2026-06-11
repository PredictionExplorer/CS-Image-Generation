//! Rendering module: histogram passes, color mapping, line drawing, and output
//!
//! This module provides a complete rendering pipeline for the three-body problem visualization,
//! including coordinate transformations, line drawing, post-processing effects, and video output.

use crate::post_effects::{
    AetherConfig, AtmosphericDepthConfig, ChampleveConfig, ChromaticBloomConfig,
    EdgeLuminanceConfig, FineTextureConfig, GradientMapConfig, LuxuryPalette, MicroContrastConfig,
    OpalescenceConfig, PerceptualBlurConfig,
};
use crate::spectrum::{NUM_BINS, linear_rec2020_to_display_p3};
use crate::utils::f64_to_usize_saturating;
use nalgebra::Vector3;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info};

/// When true, tonemapping uses the `AgX` punchy output matrix instead of default `AgX`.
pub static ACES_TWEAK_ENABLED: AtomicBool = AtomicBool::new(true);

// Module declarations
pub mod aesthetic_score;
pub mod batch_drawing;
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
pub mod spectral_output;
pub mod types;
pub mod velocity_hdr;
pub mod video;
pub mod visual_profile;

// Import from our submodules
use self::batch_drawing::{
    BatchDrawParams, draw_segment_rows_symmetric, draw_spoke_segments_rows,
    draw_triangle_batch_spectral_rows, interpolate_triangle_vertices, interpolate_vertex,
    max_triangle_vertex_motion_px, prepare_triangle_vertices,
};
use self::context::{PixelBuffer, RenderContext};
use self::effects::{EffectConfig, FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use self::error::{RenderError, Result};
use self::histogram::HistogramData;

// Re-export core types and functions for public API compatibility
pub use color::{OklabColor, generate_body_color_sequences};
pub use drawing::{
    LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral, parallel_blur_2d_rgba,
};
pub use effects::{DogBloomConfig, apply_diffraction_spikes, apply_dog_bloom};
pub use types::{ChannelLevels, ToneMappingControls};
pub use video::{VideoEncodingOptions, create_video_from_frames_singlepass};
pub use visual_profile::{
    LayerStack, ProjectionMode, SceneTraits, StackLayer, StructureMode, SymmetryOp,
};

// Re-export types from dependencies used in public API
pub use image::{DynamicImage, ImageBuffer, Rgb};

/// Which bloom algorithm to apply during post-processing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BloomMode {
    /// Difference-of-Gaussians bloom (sharper, default).
    #[default]
    Dog,
    /// Classical Gaussian blur bloom (softer glow).
    Gaussian,
    /// Bloom disabled.
    None,
}

impl BloomMode {
    /// Parse a bloom mode from a CLI argument string (case-insensitive).
    #[must_use]
    pub fn from_arg(value: &str) -> Self {
        match value {
            v if v.eq_ignore_ascii_case("gaussian") => Self::Gaussian,
            v if v.eq_ignore_ascii_case("none") => Self::None,
            _ => Self::Dog,
        }
    }

    /// Return the canonical lowercase string representation of this bloom mode.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dog => "dog",
            Self::Gaussian => "gaussian",
            Self::None => "none",
        }
    }
}

/// Top-level rendering parameters that apply across the entire pipeline.
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    /// Multiplier applied to spectral accumulation values before tone-mapping.
    pub hdr_scale: f64,
    /// Bloom algorithm selection.
    pub bloom_mode: BloomMode,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { hdr_scale: constants::DEFAULT_HDR_SCALE, bloom_mode: BloomMode::Dog }
    }
}

/// Whether the finish pipeline targets a single still or a video sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FinishOutputMode {
    /// Single accumulated frame (still export and histogram-style passes).
    #[default]
    Still,
    /// Multi-frame output (video encoding and temporal options).
    Video,
}

/// Borrowed view of a scene's trajectory data needed for spectral rendering.
#[derive(Clone, Copy)]
pub struct SpectralScene<'a> {
    /// Per-body position trajectories (indexed `[body][step]`).
    pub positions: &'a [Vec<Vector3<f64>>],
    /// Per-body Oklab colour sequences (indexed `[body][step]`).
    pub colors: &'a [Vec<OklabColor>],
    /// Overall opacity weight for each body's lines.
    pub body_alphas: &'a [f64],
}

impl std::fmt::Debug for SpectralScene<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralScene")
            .field("num_bodies", &self.positions.len())
            .field("num_steps", &self.positions.first().map_or(0, std::vec::Vec::len))
            .field("num_body_alphas", &self.body_alphas.len())
            .finish()
    }
}

impl<'a> SpectralScene<'a> {
    /// Bundle trajectory positions, colours, and alpha weights into a scene view.
    #[must_use]
    pub fn new(
        positions: &'a [Vec<Vector3<f64>>],
        colors: &'a [Vec<OklabColor>],
        body_alphas: &'a [f64],
    ) -> Self {
        Self { positions, colors, body_alphas }
    }

    /// Number of simulation timesteps recorded for each body.
    #[must_use]
    #[inline]
    pub fn step_count(self) -> usize {
        self.positions[0].len()
    }

    /// Extract the three body alphas into a fixed-size array for triangle drawing.
    #[must_use]
    #[inline]
    pub fn triangle_alphas(self) -> [f64; 3] {
        [self.body_alphas[0], self.body_alphas[1], self.body_alphas[2]]
    }
}

/// Aggregated settings for a spectral render pass (effect config + render config).
#[derive(Clone, Copy)]
pub struct SpectralRenderSettings<'a> {
    /// Fully-resolved effect parameters (randomised values already picked).
    pub resolved_config: &'a randomizable_config::ResolvedEffectConfig,
    /// Core render parameters (HDR scale, bloom mode).
    pub render_config: &'a RenderConfig,
    /// Whether to correct for non-square pixel aspect ratios.
    pub aspect_correction: bool,
    /// Seed-resolved scene traits (structure mode, line weight, age ramp).
    pub traits: SceneTraits,
}

impl std::fmt::Debug for SpectralRenderSettings<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralRenderSettings")
            .field("render_config", &self.render_config)
            .field("aspect_correction", &self.aspect_correction)
            .finish_non_exhaustive()
    }
}

impl<'a> SpectralRenderSettings<'a> {
    /// Bundle all spectral render inputs into a single settings struct.
    ///
    /// Uses the classic [`SceneTraits::default`] (triangle web, unit line
    /// weight); call [`Self::with_traits`] to attach seed-resolved traits.
    #[must_use]
    pub fn new(
        resolved_config: &'a randomizable_config::ResolvedEffectConfig,
        render_config: &'a RenderConfig,
        aspect_correction: bool,
    ) -> Self {
        Self { resolved_config, render_config, aspect_correction, traits: SceneTraits::default() }
    }

    /// Attach seed-resolved scene traits (structure mode, line weight, age ramp).
    #[must_use]
    pub fn with_traits(mut self, traits: SceneTraits) -> Self {
        self.traits = traits;
        self
    }
}

#[inline]
fn compress_display_highlights(rgb: [f64; 3], paper_white: f64, rolloff: f64) -> [f64; 3] {
    let luminance = constants::rec709_luminance(rgb[0], rgb[1], rgb[2]);
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
/// Upgraded to `AgX` for superior color rendition without hue-skewing in extreme highlights
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

/// Tonemap to 16-bit (primary output format for maximum precision)
#[cfg(test)]
#[inline]
fn tonemap_to_16bit(fr: f64, fg: f64, fb: f64, fa: f64, levels: &ChannelLevels) -> [u16; 3] {
    let channels = tonemap_core(fr, fg, fb, fa, levels);
    [
        crate::utils::f64_to_u16_saturating((channels[0] * constants::U16_MAX_F64).round()),
        crate::utils::f64_to_u16_saturating((channels[1] * constants::U16_MAX_F64).round()),
        crate::utils::f64_to_u16_saturating((channels[2] * constants::U16_MAX_F64).round()),
    ]
}

/// Save a 16-bit Display P3 image as PNG with explicit color metadata.
pub fn save_image_as_png_16bit(
    rgb_img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    path: &str,
) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| RenderError::ImageEncoding { reason: format!("failed to create PNG: {e}") })?;
    let writer = BufWriter::new(file);

    let mut info = png::Info::with_size(rgb_img.width(), rgb_img.height());
    info.color_type = png::ColorType::Rgb;
    info.bit_depth = png::BitDepth::Sixteen;
    info.source_gamma = Some(png::ScaledFloat::new(0.45455));
    info.source_chromaticities = Some(png::SourceChromaticities::new(
        (0.3127, 0.3290),
        (0.6800, 0.3200),
        (0.2650, 0.6900),
        (0.1500, 0.0600),
    ));
    info.coding_independent_code_points = Some(png::CodingIndependentCodePoints {
        color_primaries: 12,
        transfer_function: 13,
        matrix_coefficients: 0,
        is_video_full_range_image: true,
    });

    let mut bytes = Vec::with_capacity(rgb_img.as_raw().len() * 2);
    for &sample in rgb_img.as_raw() {
        bytes.extend_from_slice(&sample.to_be_bytes());
    }

    let encoder = png::Encoder::with_info(writer, info)
        .map_err(|e| RenderError::ImageEncoding { reason: e.to_string() })?;
    let mut encoder =
        encoder.write_header().map_err(|e| RenderError::ImageEncoding { reason: e.to_string() })?;
    encoder
        .write_image_data(&bytes)
        .map_err(|e| RenderError::ImageEncoding { reason: e.to_string() })?;
    encoder.finish().map_err(|e| RenderError::ImageEncoding { reason: e.to_string() })?;

    info!("   Saved 16-bit Display P3 PNG => {path}");
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
        let (p3_r, p3_g, p3_b) = linear_rec2020_to_display_p3(r, g, b);
        chunk[0] = (p3_r.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[1] = (p3_g.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[2] = (p3_b.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
    });
    buf_16bit
}

/// Estimate bytes needed for a full-frame 64-bin SPD buffer.
#[must_use]
pub fn estimate_full_spd_bytes(width: u32, height: u32) -> u128 {
    u128::from(width) * u128::from(height) * NUM_BINS as u128 * std::mem::size_of::<f64>() as u128
}

// ====================== HELPER FUNCTIONS ===========================

/// Derive the perceptual-blur radius (in pixels) after accounting for the combined
/// softness of all enabled blur/bloom effects. Returns `None` when blur is disabled.
#[must_use]
pub fn compute_softness_radius(
    resolved: &randomizable_config::ResolvedEffectConfig,
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

fn build_dog_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    min_dim: usize,
) -> DogBloomConfig {
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
    resolved: &randomizable_config::ResolvedEffectConfig,
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
    resolved: &randomizable_config::ResolvedEffectConfig,
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

fn build_color_grade_params(
    resolved: &randomizable_config::ResolvedEffectConfig,
    min_dim: usize,
) -> crate::post_effects::ColorGradeParams {
    crate::post_effects::ColorGradeParams {
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
    }
}

fn build_glow_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
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

fn build_champleve_config(resolved: &randomizable_config::ResolvedEffectConfig) -> ChampleveConfig {
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

fn build_aether_config(resolved: &randomizable_config::ResolvedEffectConfig) -> AetherConfig {
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
    resolved: &randomizable_config::ResolvedEffectConfig,
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

fn build_edge_luminance_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
) -> EdgeLuminanceConfig {
    EdgeLuminanceConfig {
        strength: resolved.edge_luminance_strength,
        threshold: resolved.edge_luminance_threshold,
        brightness_boost: resolved.edge_luminance_brightness_boost,
        bright_edges_only: true,
        min_luminance: 0.2,
    }
}

fn build_micro_contrast_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
) -> MicroContrastConfig {
    MicroContrastConfig {
        strength: resolved.micro_contrast_strength,
        radius: resolved.micro_contrast_radius,
        edge_threshold: 0.15,
        luminance_weight: 0.7,
    }
}

fn build_atmospheric_depth_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
) -> AtmosphericDepthConfig {
    AtmosphericDepthConfig {
        strength: resolved.atmospheric_depth_strength,
        fog_color: (
            resolved.atmospheric_fog_color_r,
            resolved.atmospheric_fog_color_g,
            resolved.atmospheric_fog_color_b,
        ),
        density_threshold: 0.15,
        desaturation: resolved.atmospheric_desaturation,
        darkening: resolved.atmospheric_darkening,
        density_radius: 3,
    }
}

fn build_fine_texture_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
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

/// Build a fully populated [`EffectConfig`] from resolved parameters and render settings.
#[must_use]
pub fn build_effect_config_from_resolved(
    resolved: &randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    output_mode: FinishOutputMode,
) -> EffectConfig {
    let width = resolved.width as usize;
    let height = resolved.height as usize;
    let min_dim = width.min(height);

    let use_gaussian_bloom =
        resolved.enable_bloom && matches!(render_config.bloom_mode, BloomMode::Gaussian);
    let use_dog_bloom = resolved.enable_bloom && matches!(render_config.bloom_mode, BloomMode::Dog);

    let blur_radius_px = if use_gaussian_bloom {
        (resolved.blur_radius_scale * min_dim as f64).round() as usize
    } else {
        0
    };
    let (fine_texture_enabled, fine_texture_config) =
        build_fine_texture_config(resolved, output_mode);

    EffectConfig {
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
        dog_config: build_dog_config(resolved, min_dim),
        perceptual_blur_enabled: resolved.enable_perceptual_blur,
        perceptual_blur_config: build_perceptual_blur_config(resolved, render_config.bloom_mode),

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
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AccumulationBackend {
    ParallelScanlines,
    #[cfg(test)]
    SerialReference,
}

#[inline]
fn default_accumulation_backend() -> AccumulationBackend {
    AccumulationBackend::ParallelScanlines
}

fn checkpoint_steps(total_steps: usize, frame_interval: usize) -> Vec<usize> {
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

struct AccumulationParams<'a> {
    scene: SpectralScene<'a>,
    ctx: &'a RenderContext,
    velocity_calc: &'a velocity_hdr::VelocityHdrCalculator<'a>,
    step_start: usize,
    step_end: usize,
    hdr_scale: f64,
    traits: SceneTraits,
}

/// Apply the rare diffraction-spike finish to a trajectory buffer in place.
///
/// Runs after the trajectory effect chain in every render path (histogram,
/// video frames, final stills) so exposure analysis sees the spiked image.
#[inline]
fn apply_spike_finish(buffer: &mut PixelBuffer, width: usize, height: usize, traits: &SceneTraits) {
    if traits.spikes.enabled() {
        effects::apply_diffraction_spikes(buffer, width, height, &traits.spikes);
    }
}

/// Timeline taper for vocabularies that rule sheets between instantaneous
/// geometry (webs, spokes, chords, veils, weaves): without it the first and
/// last fans burn in as hard sheet edges.
fn sheet_taper_for_vocabulary(vocabulary: StructureMode, t: f64) -> f64 {
    let is_sheet_prone = matches!(
        vocabulary,
        StructureMode::TriangleWeb
            | StructureMode::Duet { .. }
            | StructureMode::Spokes
            | StructureMode::TimeChords
            | StructureMode::NebulaVeil
            | StructureMode::HarmonicWeave
    );
    if !is_sheet_prone {
        return 1.0;
    }
    let middle_emphasis = (std::f64::consts::PI * t.clamp(0.0, 1.0)).sin().max(0.0).powf(0.65);
    0.58 + 0.42 * middle_emphasis
}

impl AccumulationParams<'_> {
    /// Trail-age exposure factor for `step`; positive ramps brighten late steps.
    #[inline]
    fn age_factor(&self, step: usize) -> f64 {
        let total = self.scene.step_count().max(1);
        let t = step as f64 / total as f64;
        1.0 + self.traits.age_ramp * (t - 0.5) * 2.0
    }

    /// Normalized timeline position of `step` in [0, 1].
    #[inline]
    fn timeline_t(&self, step: usize) -> f64 {
        let total = self.scene.step_count().max(1);
        step as f64 / total as f64
    }

    /// Per-edge alpha weights for a web-style layer.
    ///
    /// Combines the duet edge mask with the seed's `edge_energy` asymmetry,
    /// which gently emphasises one edge of the triangle over its opposite.
    #[inline]
    fn layer_edge_weights(&self, vocabulary: StructureMode) -> [f64; 3] {
        let base = match vocabulary {
            StructureMode::Duet { dropped_edge } => {
                let mut weights = [1.0; 3];
                weights[usize::from(dropped_edge.min(2))] = 0.0;
                weights
            }
            _ => [1.0; 3],
        };
        let energy = self.traits.edge_energy;
        let asymmetry = [energy, 1.0, 2.0 - energy];
        [base[0] * asymmetry[0], base[1] * asymmetry[1], base[2] * asymmetry[2]]
    }

    /// Time lag, in simulation steps, used by chord and echo strokes.
    #[inline]
    fn chord_lag_steps(&self) -> usize {
        let total = self.scene.step_count();
        if total < 2 {
            return 1;
        }
        // usize→f64 precision loss is irrelevant for lag computation.
        ((total as f64 * self.traits.chord_lag_fraction).round() as usize).clamp(1, total - 1)
    }

    /// Time pitch, in simulation steps, between stipple dots.
    #[inline]
    fn stipple_pitch_steps(&self) -> usize {
        let total = self.scene.step_count();
        // usize→f64 precision loss is irrelevant for pitch computation.
        ((total as f64 * self.traits.stipple_pitch_fraction).round() as usize)
            .max(constants::STIPPLE_MIN_PITCH_STEPS)
    }
}

/// Draw the triangle-web stroke set for one step, with motion interpolation.
#[allow(clippy::too_many_arguments)]
fn accumulate_web_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    next_vertices: Option<[batch_drawing::TriangleVertex; 3]>,
    edge_weights: [f64; 3],
) {
    let edge_dynamics = [
        params.velocity_calc.segment_dynamics(step, 0, 1),
        params.velocity_calc.segment_dynamics(step, 1, 2),
        params.velocity_calc.segment_dynamics(step, 2, 0),
    ];

    let Some(next_vertices) = next_vertices else {
        draw_triangle_batch_spectral_rows(
            accum_spd,
            &BatchDrawParams {
                width: params.ctx.width,
                height: params.ctx.height,
                row_start,
                row_end,
                vertices,
                edge_dynamics,
                edge_weights,
                line_weight: params.traits.line_weight,
                hdr_scale: step_hdr_scale,
                symmetry: params.traits.symmetry,
            },
        );
        return;
    };

    let max_motion_px = max_triangle_vertex_motion_px(vertices, next_vertices);
    let substeps = constants::crisp_line_interpolation_substeps(
        params.ctx.width,
        params.ctx.height,
        max_motion_px,
    );
    let substep_hdr_scale = step_hdr_scale / substeps as f64;

    for substep in 0..substeps {
        let sample_vertices = if substep == 0 {
            vertices
        } else {
            let t = substep as f32 / substeps as f32;
            interpolate_triangle_vertices(vertices, next_vertices, t)
        };
        draw_triangle_batch_spectral_rows(
            accum_spd,
            &BatchDrawParams {
                width: params.ctx.width,
                height: params.ctx.height,
                row_start,
                row_end,
                vertices: sample_vertices,
                edge_dynamics,
                edge_weights,
                line_weight: params.traits.line_weight,
                hdr_scale: substep_hdr_scale,
                symmetry: params.traits.symmetry,
            },
        );
    }
}

/// Velocity dynamics for each body's own trail at `step`.
#[inline]
fn body_dynamics_at(
    params: &AccumulationParams<'_>,
    step: usize,
) -> [velocity_hdr::SegmentDynamics; 3] {
    [
        params.velocity_calc.body_dynamics(step, 0),
        params.velocity_calc.body_dynamics(step, 1),
        params.velocity_calc.body_dynamics(step, 2),
    ]
}

/// Draw the three per-body trail strokes (`step -> step + 1`) for ribbon modes.
///
/// `energy_scale` lets callers layer the ribbon as a reduced-alpha underlay
/// (chord modes) without a separate drawing path.
#[allow(clippy::too_many_arguments)]
fn accumulate_ribbon_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    next_vertices: Option<[batch_drawing::TriangleVertex; 3]>,
    energy_scale: f64,
) {
    let Some(next_vertices) = next_vertices else {
        return;
    };

    let draw_params = BatchDrawParams {
        width: params.ctx.width,
        height: params.ctx.height,
        row_start,
        row_end,
        vertices,
        edge_dynamics: body_dynamics_at(params, step),
        edge_weights: [1.0; 3],
        line_weight: params.traits.line_weight,
        hdr_scale: step_hdr_scale,
        symmetry: params.traits.symmetry,
    };
    for body in 0..3 {
        batch_drawing::draw_body_trail_segment_rows(
            accum_spd,
            &draw_params,
            body,
            next_vertices,
            energy_scale,
        );
    }
}

/// Draw three time-lagged chord strokes (`step -> step + lag`), one per body.
///
/// Chords connect each body to its own future position, ruling luminous sheets
/// between successive loop windings (string-art bands). Like ribbon trails,
/// chords intentionally reference geometry beyond the current video chunk so
/// the accumulated still is seamless across checkpoints.
#[allow(clippy::too_many_arguments)]
fn accumulate_chord_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    lag_steps: usize,
    energy_scale: f64,
) {
    if energy_scale <= 0.0 {
        return;
    }
    let target_step = step + lag_steps;
    if target_step >= params.scene.step_count() {
        return;
    }

    let triangle_alphas = params.scene.triangle_alphas();
    let lagged_vertices = prepare_triangle_vertices(
        params.scene.positions,
        params.scene.colors,
        &triangle_alphas,
        target_step,
        params.ctx,
    );
    let draw_params = BatchDrawParams {
        width: params.ctx.width,
        height: params.ctx.height,
        row_start,
        row_end,
        vertices,
        edge_dynamics: body_dynamics_at(params, step),
        edge_weights: [1.0; 3],
        line_weight: params.traits.line_weight,
        hdr_scale: step_hdr_scale,
        symmetry: params.traits.symmetry,
    };
    for body in 0..3 {
        batch_drawing::draw_body_trail_segment_rows(
            accum_spd,
            &draw_params,
            body,
            lagged_vertices,
            energy_scale,
        );
    }
}

/// Draw the body-to-centroid spokes for one step, with motion interpolation.
///
/// Spokes are instantaneous geometry like the web edges, so they use the same
/// substep interpolation; without it, fast passages leave visible striping
/// between consecutive spoke fans.
#[allow(clippy::too_many_arguments)]
fn accumulate_spokes_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    next_vertices: Option<[batch_drawing::TriangleVertex; 3]>,
    energy_scale: f64,
) {
    if energy_scale <= 0.0 {
        return;
    }
    let body_dynamics = body_dynamics_at(params, step);
    let make_params = |sample_vertices, hdr_scale| BatchDrawParams {
        width: params.ctx.width,
        height: params.ctx.height,
        row_start,
        row_end,
        vertices: sample_vertices,
        edge_dynamics: body_dynamics,
        edge_weights: [1.0; 3],
        line_weight: params.traits.line_weight,
        hdr_scale,
        symmetry: params.traits.symmetry,
    };

    let Some(next_vertices) = next_vertices else {
        draw_spoke_segments_rows(accum_spd, &make_params(vertices, step_hdr_scale * energy_scale));
        return;
    };

    let max_motion_px = max_triangle_vertex_motion_px(vertices, next_vertices);
    let substeps = constants::crisp_line_interpolation_substeps(
        params.ctx.width,
        params.ctx.height,
        max_motion_px,
    );
    let substep_hdr_scale = step_hdr_scale * energy_scale / substeps as f64;

    for substep in 0..substeps {
        let sample_vertices = if substep == 0 {
            vertices
        } else {
            let t = substep as f32 / substeps as f32;
            interpolate_triangle_vertices(vertices, next_vertices, t)
        };
        draw_spoke_segments_rows(accum_spd, &make_params(sample_vertices, substep_hdr_scale));
    }
}

/// Sweep the triangle interior with interpolated fill lines (`NebulaVeil`).
///
/// Each fill line connects matching parametric points on the two edges that
/// share the pivot vertex, so colors blend smoothly across the gauze. The
/// pivot rotates with the absolute step index to avoid directional bias, and
/// the per-line energy is normalized by the fill count and stride so a veil
/// layer deposits ink comparable to a single web edge per step.
#[allow(clippy::too_many_arguments)]
fn accumulate_veil_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    energy_scale: f64,
) {
    if energy_scale <= 0.0 || !step.is_multiple_of(constants::VEIL_STEP_STRIDE) {
        return;
    }
    let fill_lines = usize::from(params.traits.veil_fill_lines.max(2));
    let pivot = step % 3;
    let a = vertices[pivot];
    let b = vertices[(pivot + 1) % 3];
    let c = vertices[(pivot + 2) % 3];
    let dynamics = params.velocity_calc.body_dynamics(step, pivot);

    // usize→f64: stride and fill counts are tiny.
    let line_energy = step_hdr_scale * energy_scale * constants::VEIL_STEP_STRIDE as f64
        / fill_lines as f64
        * dynamics.hdr_multiplier;

    for k in 1..=fill_lines {
        // usize→f32: parametric position along the edges.
        let t = k as f32 / (fill_lines + 1) as f32;
        let start = interpolate_vertex(a, b, t);
        let end = interpolate_vertex(a, c, t);
        draw_segment_rows_symmetric(
            accum_spd,
            params.ctx.width,
            params.ctx.height,
            row_start,
            row_end,
            SpectralLineSegment {
                start,
                end,
                hdr_scale: line_energy,
                thickness_factor: dynamics.thickness_factor * params.traits.line_weight,
            },
            params.traits.symmetry,
        );
    }
}

/// Evaluate a quadratic Bezier between two triangle vertices in pixel space.
#[inline]
fn weave_bezier_point(
    from: batch_drawing::TriangleVertex,
    control: (f32, f32, f32),
    to: batch_drawing::TriangleVertex,
    t: f32,
) -> batch_drawing::TriangleVertex {
    let u = 1.0 - t;
    let mut point = interpolate_vertex(from, to, t);
    point.x = u * u * from.x + 2.0 * u * t * control.0 + t * t * to.x;
    point.y = u * u * from.y + 2.0 * u * t * control.1 + t * t * to.y;
    point.z = u * u * from.z + 2.0 * u * t * control.2 + t * t * to.z;
    point
}

/// Draw curved Bezier chords between bodies (`HarmonicWeave`).
///
/// Each inter-body chord bows toward or away from the third body with a
/// seeded bow factor that breathes slowly over the timeline, tessellated into
/// short straight segments fed through the standard spectral splatter.
#[allow(clippy::too_many_arguments)]
fn accumulate_weave_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    edge_weights: [f64; 3],
    energy_scale: f64,
) {
    if energy_scale <= 0.0 || !step.is_multiple_of(constants::WEAVE_STEP_STRIDE) {
        return;
    }
    let centroid = (
        (vertices[0].x + vertices[1].x + vertices[2].x) / 3.0,
        (vertices[0].y + vertices[1].y + vertices[2].y) / 3.0,
        (vertices[0].z + vertices[1].z + vertices[2].z) / 3.0,
    );
    let t_norm = params.timeline_t(step);
    let edge_dynamics = [
        params.velocity_calc.segment_dynamics(step, 0, 1),
        params.velocity_calc.segment_dynamics(step, 1, 2),
        params.velocity_calc.segment_dynamics(step, 2, 0),
    ];

    // usize→f64: stride and tessellation counts are tiny.
    let segment_energy = step_hdr_scale * energy_scale * constants::WEAVE_STEP_STRIDE as f64
        / constants::WEAVE_SEGMENTS as f64;

    for (edge_idx, (i, j, third)) in
        [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)].into_iter().enumerate()
    {
        let weight = edge_weights[edge_idx];
        if weight <= 0.0 {
            continue;
        }
        // The bow breathes over the timeline with a per-edge phase so the
        // three curve families interleave instead of moving in lockstep.
        let wobble = 1.0 - constants::WEAVE_BOW_WOBBLE
            + constants::WEAVE_BOW_WOBBLE
                * (std::f64::consts::TAU * (t_norm * 2.0 + edge_idx as f64 / 3.0)).sin();
        // f64→f32 precision loss is irrelevant at raster scale.
        let bow = (params.traits.weave_bow * wobble) as f32;
        let control = (
            centroid.0 + (centroid.0 - vertices[third].x) * bow,
            centroid.1 + (centroid.1 - vertices[third].y) * bow,
            centroid.2 + (centroid.2 - vertices[third].z) * bow,
        );

        let dynamics = edge_dynamics[edge_idx];
        let mut prev = vertices[i];
        for s in 1..=constants::WEAVE_SEGMENTS {
            // usize→f32: tessellation parameter.
            let t = s as f32 / constants::WEAVE_SEGMENTS as f32;
            let point = weave_bezier_point(vertices[i], control, vertices[j], t);
            draw_segment_rows_symmetric(
                accum_spd,
                params.ctx.width,
                params.ctx.height,
                row_start,
                row_end,
                SpectralLineSegment {
                    start: prev,
                    end: point,
                    hdr_scale: segment_energy * dynamics.hdr_multiplier * weight,
                    thickness_factor: dynamics.thickness_factor * params.traits.line_weight,
                },
                params.traits.symmetry,
            );
            prev = point;
        }
    }
}

/// Splat pointillist dots at a fixed time pitch (`StippleConstellation`).
///
/// Time-uniform sampling makes orbital speed visible: slow passages cluster
/// into dense bead curtains while fast whips scatter sparse sparks. Every Nth
/// dot is a brighter, larger pearl. Dot energy compensates for the skipped
/// steps so a stipple layer deposits ink comparable to a ribbon layer.
#[allow(clippy::too_many_arguments)]
fn accumulate_stipple_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    energy_scale: f64,
) {
    if energy_scale <= 0.0 {
        return;
    }
    let pitch = params.stipple_pitch_steps();
    if !step.is_multiple_of(pitch) {
        return;
    }
    let dot_index = step / pitch;
    let pearl = dot_index.is_multiple_of(usize::from(params.traits.stipple_pearl_every.max(2)));
    let (thickness, energy_mult) = if pearl {
        (constants::STIPPLE_PEARL_THICKNESS, constants::STIPPLE_PEARL_ENERGY)
    } else {
        (constants::STIPPLE_DOT_THICKNESS, 1.0)
    };

    // usize→f64: pitch is bounded by the step count.
    let dot_energy = step_hdr_scale
        * energy_scale
        * pitch as f64
        * constants::STIPPLE_ENERGY_FACTOR
        * energy_mult;

    for (body, vertex) in vertices.into_iter().enumerate() {
        let dynamics = params.velocity_calc.body_dynamics(step, body);
        draw_segment_rows_symmetric(
            accum_spd,
            params.ctx.width,
            params.ctx.height,
            row_start,
            row_end,
            SpectralLineSegment {
                start: vertex,
                end: vertex,
                hdr_scale: dot_energy * dynamics.hdr_multiplier,
                thickness_factor: thickness * params.traits.line_weight,
            },
            params.traits.symmetry,
        );
    }
}

/// Draw velocity tangent segments centered on each body (`TangentCaustics`).
///
/// The orbit is never traced directly; it emerges as the envelope of its own
/// tangents. Per-segment energy is normalized by the realized length so long
/// fast tangents do not flood the frame.
#[allow(clippy::too_many_arguments)]
fn accumulate_tangent_step(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
    step: usize,
    step_hdr_scale: f64,
    vertices: [batch_drawing::TriangleVertex; 3],
    next_vertices: Option<[batch_drawing::TriangleVertex; 3]>,
    energy_scale: f64,
) {
    if energy_scale <= 0.0 {
        return;
    }
    let Some(next_vertices) = next_vertices else {
        return;
    };
    // u32→f32 precision loss is irrelevant at raster scale.
    let min_dim = params.ctx.width.min(params.ctx.height) as f32;

    for body in 0..3 {
        let vertex = vertices[body];
        let next = next_vertices[body];
        let dx = next.x - vertex.x;
        let dy = next.y - vertex.y;
        let speed_px = (dx * dx + dy * dy).sqrt();
        if speed_px < 1e-6 || !speed_px.is_finite() {
            continue;
        }
        // f64→f32 precision loss is irrelevant at raster scale.
        let half_length = (speed_px * constants::TANGENT_VELOCITY_GAIN as f32).clamp(
            min_dim * constants::TANGENT_MIN_LEN_FRAC as f32,
            min_dim * constants::TANGENT_MAX_LEN_FRAC as f32,
        ) * params.traits.tangent_length as f32
            * 0.5;
        let ux = dx / speed_px;
        let uy = dy / speed_px;

        let mut start = vertex;
        start.x = vertex.x - ux * half_length;
        start.y = vertex.y - uy * half_length;
        let mut end = vertex;
        end.x = vertex.x + ux * half_length;
        end.y = vertex.y + uy * half_length;

        // Length normalization keeps total per-step ink stable as tangents
        // stretch with speed.
        let reference_len = f64::from(min_dim) * constants::TANGENT_REFERENCE_LEN_FRAC;
        let length_norm = (reference_len / f64::from(half_length * 2.0)).clamp(0.2, 3.0);
        let dynamics = params.velocity_calc.body_dynamics(step, body);
        draw_segment_rows_symmetric(
            accum_spd,
            params.ctx.width,
            params.ctx.height,
            row_start,
            row_end,
            SpectralLineSegment {
                start,
                end,
                hdr_scale: step_hdr_scale * energy_scale * length_norm * dynamics.hdr_multiplier,
                thickness_factor: dynamics.thickness_factor * params.traits.line_weight,
            },
            params.traits.symmetry,
        );
    }
}

/// `SplitMix64`: tiny deterministic stream for stardust placement.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn splitmix_unit(state: &mut u64) -> f64 {
    // u64→f64 may lose precision for large values; acceptable for placement jitter.
    splitmix64(state) as f64 / u64::MAX as f64
}

/// Splat the faint seeded stardust field once per accumulation pass.
///
/// Dot placement is a pure function of the trait seed, and the row-banded
/// line splatter clips each dot to the owned band, so parallel scanline
/// accumulation stays bit-identical to the serial reference.
fn splat_stardust_rows(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
) {
    let dust = params.traits.stardust;
    if !dust.enabled() {
        return;
    }

    let body_alphas = params.scene.body_alphas;
    // usize→f64: body counts and step counts are well within f64 precision.
    let mean_alpha = body_alphas.iter().sum::<f64>() / body_alphas.len().max(1) as f64;
    let base_energy = dust.brightness
        * mean_alpha
        * params.scene.step_count() as f64
        * params.hdr_scale
        * constants::STARDUST_ENERGY_FACTOR;
    if base_energy <= 0.0 {
        return;
    }

    // u32→f32/f64 precision loss is irrelevant at raster scale.
    let width = f64::from(params.ctx.width);
    let height = f64::from(params.ctx.height);
    let mut state = dust.seed | 1;

    for _ in 0..dust.count {
        let x = splitmix_unit(&mut state) * width;
        let y = splitmix_unit(&mut state) * height;
        let size = 0.45 + splitmix_unit(&mut state) * 1.15;
        let glow = splitmix_unit(&mut state);
        let hue = splitmix_unit(&mut state) * 360.0;
        let twinkle = if splitmix_unit(&mut state) < 0.06 { 3.0 } else { 1.0 };

        let lightness = (dust.lightness + (glow - 0.5) * 0.12).clamp(0.30, 0.92);
        let color = crate::oklab::oklch_to_oklab(lightness, dust.chroma, hue);
        let vertex = LineVertex {
            // f64→f32 precision loss is irrelevant at raster scale.
            x: x as f32,
            y: y as f32,
            z: 0.0,
            color,
            alpha: base_energy * (0.4 + 0.6 * glow) * twinkle,
        };
        draw_line_segment_aa_spectral_rows_local(
            accum_spd,
            params.ctx.width,
            params.ctx.height,
            row_start,
            row_end,
            SpectralLineSegment {
                start: vertex,
                end: vertex,
                hdr_scale: 1.0,
                thickness_factor: size,
            },
        );
    }
}

/// Thin local alias so stardust uses the row-banded splatter directly
/// (stardust is a uniform field: symmetry replication would be invisible).
#[inline]
fn draw_line_segment_aa_spectral_rows_local(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    row_start: usize,
    row_end: usize,
    segment: SpectralLineSegment,
) {
    drawing::draw_line_segment_aa_spectral_rows(accum, width, height, row_start, row_end, segment);
}

fn accumulate_spectral_steps_into_rows(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
) {
    if params.step_start >= params.step_end || row_start >= row_end {
        return;
    }

    // The stardust field belongs to the scene, not the timeline: splat it
    // exactly once per accumulation pass, with the first step chunk.
    if params.step_start == 0 {
        splat_stardust_rows(accum_spd, params, row_start, row_end);
    }

    let triangle_alphas = params.scene.triangle_alphas();
    let layers = params.traits.stack.layers();
    let layer_edge_weights: SmallVec<[[f64; 3]; 3]> =
        layers.iter().map(|layer| params.layer_edge_weights(layer.vocabulary)).collect();

    for step in params.step_start..params.step_end {
        let vertices = prepare_triangle_vertices(
            params.scene.positions,
            params.scene.colors,
            &triangle_alphas,
            step,
            params.ctx,
        );

        // Next-step vertices exist whenever the simulation continues. The web
        // path additionally restricts interpolation to the current chunk so
        // video frames never leak future motion; ribbon trails must cross the
        // chunk boundary or every checkpoint would leave a one-step gap.
        let next_vertices = if step + 1 < params.scene.step_count() {
            Some(prepare_triangle_vertices(
                params.scene.positions,
                params.scene.colors,
                &triangle_alphas,
                step + 1,
                params.ctx,
            ))
        } else {
            None
        };
        let next_in_chunk = if step + 1 < params.step_end { next_vertices } else { None };

        let age_hdr_scale = params.hdr_scale * params.age_factor(step);
        let t_norm = params.timeline_t(step);

        for (layer, edge_weights) in layers.iter().zip(layer_edge_weights.iter()) {
            let step_hdr_scale =
                age_hdr_scale * layer.alpha * sheet_taper_for_vocabulary(layer.vocabulary, t_norm);

            match layer.vocabulary {
                StructureMode::TriangleWeb | StructureMode::Duet { .. } => {
                    accumulate_web_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        next_in_chunk,
                        *edge_weights,
                    );
                }
                StructureMode::OrbitRibbons => {
                    accumulate_ribbon_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        next_vertices,
                        1.0,
                    );
                    // Optional decaying time-lagged echoes: rule soft bands
                    // between loop windings (comet tails at higher counts).
                    if params.traits.ribbon_echo_alpha > 0.0 {
                        let lag = params.chord_lag_steps();
                        let echo_layers = usize::from(params.traits.echo_layers.clamp(1, 3));
                        for (echo_idx, decay) in
                            constants::COMET_ECHO_DECAY.iter().take(echo_layers).enumerate()
                        {
                            accumulate_chord_step(
                                accum_spd,
                                params,
                                row_start,
                                row_end,
                                step,
                                step_hdr_scale,
                                vertices,
                                lag * (echo_idx + 1),
                                params.traits.ribbon_echo_alpha * decay,
                            );
                        }
                    }
                }
                StructureMode::Spokes => {
                    accumulate_spokes_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        next_in_chunk,
                        1.0,
                    );
                }
                StructureMode::TimeChords => {
                    accumulate_ribbon_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        next_vertices,
                        constants::CHORD_RIBBON_UNDERLAY_ALPHA,
                    );
                    accumulate_chord_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        params.chord_lag_steps(),
                        1.0,
                    );
                }
                StructureMode::NebulaVeil => {
                    accumulate_veil_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        1.0,
                    );
                }
                StructureMode::HarmonicWeave => {
                    accumulate_weave_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        *edge_weights,
                        1.0,
                    );
                }
                StructureMode::StippleConstellation => {
                    accumulate_stipple_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        1.0,
                    );
                }
                StructureMode::TangentCaustics => {
                    accumulate_tangent_step(
                        accum_spd,
                        params,
                        row_start,
                        row_end,
                        step,
                        step_hdr_scale,
                        vertices,
                        next_vertices,
                        1.0,
                    );
                }
            }
        }
    }
}

fn accumulate_spectral_steps(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    backend: AccumulationBackend,
) {
    if params.step_start >= params.step_end || accum_spd.is_empty() {
        return;
    }

    match backend {
        AccumulationBackend::ParallelScanlines => {
            let ctx = params.ctx;
            let band_count = ctx.height_usize.min(rayon::current_num_threads().max(1));
            if band_count <= 1 {
                accumulate_spectral_steps_into_rows(accum_spd, params, 0, ctx.height_usize);
                return;
            }

            let rows_per_band = ctx.height_usize.div_ceil(band_count);
            let pixels_per_band = ctx.width_usize * rows_per_band;
            accum_spd.par_chunks_mut(pixels_per_band).enumerate().for_each(|(band_idx, band)| {
                let row_start = band_idx * rows_per_band;
                let row_end = row_start + band.len() / ctx.width_usize;
                accumulate_spectral_steps_into_rows(band, params, row_start, row_end);
            });
        }
        #[cfg(test)]
        AccumulationBackend::SerialReference => {
            accumulate_spectral_steps_into_rows(accum_spd, params, 0, params.ctx.height_usize);
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
            &AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start,
                step_end: checkpoint_step + 1,
                hdr_scale: render_config.hdr_scale,
                traits: settings.traits,
            },
            backend,
        );

        convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let mut trajectory_proxy = finish_pipeline
            .process_trajectory(rgba_buffer, width as usize, height as usize, &frame_params)
            .expect("effect chain invariant: histogram-pass trajectory processing must not fail");
        apply_spike_finish(
            &mut trajectory_proxy,
            width as usize,
            height as usize,
            &settings.traits,
        );
        accum_rgba.clear();
        accum_rgba.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));

        histogram.reserve(ctx.pixel_count());
        for &(r, g, b, a) in &trajectory_proxy {
            histogram.push(r * a, g * a, b * a);
        }

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

/// Bundled parameters for `pass_2_write_frames_spectral` and its backend variants.
///
/// Groups the non-closure state that every pass-2 call site must supply,
/// keeping the function signatures under clippy's argument-count threshold.
pub struct Pass2Params<'a> {
    /// Trajectory positions, colours, and per-body alphas to draw.
    pub scene: SpectralScene<'a>,
    /// Simulation steps between emitted video frames.
    pub frame_interval: usize,
    /// Per-channel black/white levels from pass 1 histogram analysis.
    pub levels: &'a ChannelLevels,
    /// Resolved effects, HDR scale, bloom, noise seed, and aspect handling.
    pub settings: SpectralRenderSettings<'a>,
    /// Receives the final frame as 16-bit RGB when the pass completes.
    pub last_frame_out: &'a mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    /// When true, blends consecutive display frames to reduce temporal noise.
    pub enable_temporal_smoothing: bool,
    /// Scratch buffer for per-pixel spectral power distributions (reused across checkpoints).
    pub accum_spd: &'a mut Vec<[f64; NUM_BINS]>,
}

#[cfg(test)]
pub(crate) fn pass_2_write_frames_spectral_serial_reference(
    params: Pass2Params<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(
        params,
        frame_sink,
        AccumulationBackend::SerialReference,
    )
}

// ====================== PASS 2 (SPECTRAL) ===========================
/// Pass 2: final frames => color mapping => write frames (spectral, 16-bit output)
///
/// The caller-provided `accum_spd` buffer is populated incrementally and contains
/// the fully accumulated spectral data when this function returns.
fn pass_2_write_frames_spectral_with_backend(
    params: Pass2Params<'_>,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
    backend: AccumulationBackend,
) -> Result<()> {
    let Pass2Params {
        scene,
        frame_interval,
        levels,
        settings,
        last_frame_out,
        enable_temporal_smoothing,
        accum_spd,
    } = params;
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction, .. } = settings;
    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    accum_spd.resize(ctx.pixel_count(), [0.0f64; NUM_BINS]);
    for s in accum_spd.iter_mut() {
        *s = [0.0; NUM_BINS];
    }
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let checkpoints = checkpoint_steps(total_steps, frame_interval);
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

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

    for &checkpoint_step in &checkpoints {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 2, mode = "spectral", "Render pass progress");
        }

        accumulate_spectral_steps(
            accum_spd,
            &AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start,
                step_end: checkpoint_step + 1,
                hdr_scale: render_config.hdr_scale,
                traits: settings.traits,
            },
            backend,
        );

        convert_spd_buffer_to_rgba(accum_spd, &mut accum_rgba, width as usize, height as usize);

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let mut trajectory_pixels = finish_pipeline
            .process_trajectory(rgba_buffer, width as usize, height as usize, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "trajectory_chain".into(),
                reason: e.to_string(),
            })?;
        apply_spike_finish(
            &mut trajectory_pixels,
            width as usize,
            height as usize,
            &settings.traits,
        );

        let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);

        // Reclaim the trajectory buffer's allocation back into accum_rgba.
        // It will be fully overwritten by convert_spd_buffer_to_rgba next iteration,
        // so we just need the capacity -- no need to clear or resize.
        trajectory_pixels.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));
        accum_rgba = trajectory_pixels;

        let smoothed_display = match &temporal_smoother {
            Some(smoother) => smoother.process_frame(display_buffer),
            None => display_buffer,
        };

        let final_display = finish_pipeline
            .process_image(smoothed_display, width as usize, height as usize, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "image_chain".into(),
                reason: e.to_string(),
            })?;
        let buf_16bit = quantize_display_buffer_to_16bit(&final_display);
        let buf_bytes: &[u8] = bytemuck::cast_slice(&buf_16bit);

        frame_sink(buf_bytes)?;
        if checkpoint_step + 1 == total_steps {
            *last_frame_out = ImageBuffer::from_raw(width, height, buf_16bit);
        }

        step_start = checkpoint_step + 1;
    }

    info!("   pass 2 (spectral render): 100% done");
    Ok(())
}

/// Pass 2: render frames with spectral accumulation and feed 16-bit bytes to `frame_sink`.
pub fn pass_2_write_frames_spectral(
    params: Pass2Params<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(params, frame_sink, default_accumulation_backend())
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
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction, .. } = settings;
    info!("   Rendering final accumulated frame (preview mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let pixel_count = ctx.pixel_count();
    if pixel_count > constants::HIGH_RES_TILED_PIXEL_THRESHOLD
        && !resolved_config.any_legacy_effect_enabled()
        && backend == AccumulationBackend::ParallelScanlines
    {
        return render_final_frame_spectral_tiled(scene, levels, settings, &ctx);
    }

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
        &AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: total_steps,
            hdr_scale: render_config.hdr_scale,
            traits: settings.traits,
        },
        backend,
    );

    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let preview_frame_number = total_steps.saturating_sub(1) / frame_interval;
    let frame_params = FrameParams { frame_number: preview_frame_number, density: None };
    let mut trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "trajectory_chain".into(),
            reason: e.to_string(),
        })?;
    apply_spike_finish(&mut trajectory_pixels, width as usize, height as usize, &settings.traits);

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
    let final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "image_chain".into(),
            reason: e.to_string(),
        })?;
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| RenderError::ImageEncoding {
        reason: "Failed to create 16-bit image buffer".into(),
    })
}

fn render_final_frame_spectral_tiled(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    ctx: &RenderContext,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction: _, .. } =
        settings;
    let full_spd_gib =
        estimate_full_spd_bytes(ctx.width, ctx.height) as f64 / (1024.0 * 1024.0 * 1024.0);
    info!(
        "   Rendering final still in crisp tiled mode: {}x{} ({:.2} GiB full SPD avoided)",
        ctx.width, ctx.height, full_spd_gib
    );

    let tile_rows = constants::HIGH_RES_TILE_ROWS.max(1);
    let guard_rows = constants::crisp_tiled_guard_rows(ctx.width, ctx.height);
    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);
    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let preview_frame_number = total_steps.saturating_sub(1) / frame_interval;
    let frame_params = FrameParams { frame_number: preview_frame_number, density: None };
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);
    let mut full_output = vec![0u16; ctx.pixel_count() * 3];

    for core_start in (0..ctx.height_usize).step_by(tile_rows) {
        let core_end = (core_start + tile_rows).min(ctx.height_usize);
        let guard_start = core_start.saturating_sub(guard_rows);
        let guard_end = (core_end + guard_rows).min(ctx.height_usize);
        let guard_height = guard_end - guard_start;
        let mut tile_spd = vec![[0.0f64; NUM_BINS]; ctx.width_usize * guard_height];
        let mut tile_rgba = vec![(0.0, 0.0, 0.0, 0.0); tile_spd.len()];

        accumulate_spectral_steps_into_rows(
            &mut tile_spd,
            &AccumulationParams {
                scene,
                ctx,
                velocity_calc: &velocity_calc,
                step_start: 0,
                step_end: total_steps,
                hdr_scale: render_config.hdr_scale,
                traits: settings.traits,
            },
            guard_start,
            guard_end,
        );
        convert_spd_buffer_to_rgba(&tile_spd, &mut tile_rgba, ctx.width_usize, guard_height);

        let mut trajectory_pixels = finish_pipeline
            .process_trajectory(tile_rgba, ctx.width_usize, guard_height, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "trajectory_chain".into(),
                reason: e.to_string(),
            })?;
        // Spikes run band-locally in tiled mode (sources outside the guard
        // rows cannot contribute); production resolutions never tile.
        apply_spike_finish(&mut trajectory_pixels, ctx.width_usize, guard_height, &settings.traits);
        let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
        let final_display = finish_pipeline
            .process_image(display_buffer, ctx.width_usize, guard_height, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "image_chain".into(),
                reason: e.to_string(),
            })?;
        let tile_u16 = quantize_display_buffer_to_16bit(&final_display);

        let core_offset = core_start - guard_start;
        for global_y in core_start..core_end {
            let local_y = global_y - guard_start;
            let src = local_y * ctx.width_usize * 3;
            let dst = global_y * ctx.width_usize * 3;
            let len = ctx.width_usize * 3;
            debug_assert!(local_y >= core_offset);
            full_output[dst..dst + len].copy_from_slice(&tile_u16[src..src + len]);
        }
    }

    ImageBuffer::from_raw(ctx.width, ctx.height, full_output).ok_or_else(|| {
        RenderError::ImageEncoding { reason: "Failed to create tiled 16-bit image buffer".into() }
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
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction, .. } = settings;
    info!("   Rendering first timeline slice only (legacy test mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    // Create render context
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    // Build effect configuration from resolved config
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;

    // Create velocity HDR calculator for efficient multiplier computation
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    // Render all trajectory steps up to and including the first output frame interval
    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let first_frame_step = frame_interval;

    accumulate_spectral_steps(
        &mut accum_spd,
        &AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: first_frame_step + 1,
            hdr_scale: render_config.hdr_scale,
            traits: settings.traits,
        },
        backend,
    );

    // Process the accumulated frame
    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_params = FrameParams { frame_number: 0, density: None };
    let mut trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "trajectory_chain".into(),
            reason: e.to_string(),
        })?;
    apply_spike_finish(&mut trajectory_pixels, width as usize, height as usize, &settings.traits);

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
    let final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "image_chain".into(),
            reason: e.to_string(),
        })?;

    // Quantize display buffer to 16-bit
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    // Create ImageBuffer and return
    let image = ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        RenderError::ImageEncoding { reason: "Failed to create 16-bit image buffer".into() }
    })?;

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::randomizable_config::ResolvedEffectConfig;
    use nalgebra::Vector3;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn convert_spd_buffer_is_spectrally_neutral_across_energy() {
        // After removing the energy-density redshift, the production SPD->RGBA
        // conversion must map a fixed spectrum to a fixed hue regardless of total
        // energy. Brightness may change with energy; hue must not drift toward red.
        let bin = NUM_BINS / 2; // mid-spectrum (green)
        let mut low = [0.0f64; NUM_BINS];
        let mut high = [0.0f64; NUM_BINS];
        low[bin] = 0.02;
        high[bin] = 8.0;

        let src = vec![low, high];
        let mut dest = vec![(0.0, 0.0, 0.0, 0.0); 2];
        convert_spd_buffer_to_rgba(&src, &mut dest, 2, 1);

        let hue = |p: (f64, f64, f64, f64)| {
            let (r, g, b, a) = p;
            let inv = 1.0 / a.max(1e-9);
            let (_, oa, ob) = crate::oklab::linear_rec2020_to_oklab(r * inv, g * inv, b * inv);
            ob.atan2(oa).to_degrees().rem_euclid(360.0)
        };
        let low_hue = hue(dest[0]);
        let high_hue = hue(dest[1]);
        let delta = {
            let d = (low_hue - high_hue).abs();
            d.min(360.0 - d)
        };
        assert!(
            delta < 8.0,
            "conversion hue drifted with energy (low={low_hue:.2}, high={high_hue:.2}); an energy-density redshift may have been reintroduced"
        );
    }

    #[test]
    fn sheet_taper_reduces_fill_modes_at_timeline_edges() {
        let early = sheet_taper_for_vocabulary(StructureMode::Spokes, 0.0);
        let mid = sheet_taper_for_vocabulary(StructureMode::Spokes, 0.5);
        let ribbon = sheet_taper_for_vocabulary(StructureMode::OrbitRibbons, 0.0);
        let veil = sheet_taper_for_vocabulary(StructureMode::NebulaVeil, 0.0);
        let stipple = sheet_taper_for_vocabulary(StructureMode::StippleConstellation, 0.0);

        assert!(early < mid, "sheet-prone modes should taper at timeline edges");
        assert!((mid - 1.0).abs() < 1e-12, "middle of sheet taper should preserve energy");
        assert_eq!(ribbon, 1.0, "ribbon modes should not use sheet taper");
        assert!(veil < 1.0, "veil sweeps sheets and should taper");
        assert_eq!(stipple, 1.0, "stipple dots should not use sheet taper");
    }

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
        image.as_raw().iter().map(|&channel| u64::from(channel)).sum()
    }

    type SceneData = (Vec<Vec<Vector3<f64>>>, Vec<Vec<OklabColor>>, Vec<f64>);
    type CapturedFrameResult = (Vec<u8>, Option<ImageBuffer<Rgb<u16>, Vec<u16>>>);

    fn assert_frame_bytes_eq(actual: &[u8], expected: &[u8], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: frame byte lengths differ");
        if actual != expected {
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .expect("expected differing byte position");
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
    ) -> CapturedFrameResult {
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
                let mut spd_buf = Vec::new();

                let params = Pass2Params {
                    scene,
                    frame_interval,
                    levels,
                    settings,
                    last_frame_out: &mut last_frame,
                    enable_temporal_smoothing,
                    accum_spd: &mut spd_buf,
                };
                if serial_reference {
                    pass_2_write_frames_spectral_serial_reference(params, frame_sink)
                } else {
                    pass_2_write_frames_spectral(params, frame_sink)
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
            assert!(u32::from(ch) <= 65535, "16-bit channel {ch} out of range");
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

    /// Trait variants exercising every vocabulary, stacking, symmetry, and
    /// the stardust field. Shared by the equivalence and robustness tests.
    fn trait_variants() -> Vec<(&'static str, SceneTraits)> {
        use visual_profile::{LayerStack, StackLayer, StardustTraits, SymmetryOp};

        let solo = |vocabulary| SceneTraits {
            stack: LayerStack::solo(vocabulary),
            ..SceneTraits::default()
        };
        let mut variants = vec![
            ("web", solo(StructureMode::TriangleWeb)),
            ("ribbons", solo(StructureMode::OrbitRibbons)),
            ("duet", solo(StructureMode::Duet { dropped_edge: 1 })),
            ("spokes", solo(StructureMode::Spokes)),
            ("chords", solo(StructureMode::TimeChords)),
            ("veil", solo(StructureMode::NebulaVeil)),
            ("weave", solo(StructureMode::HarmonicWeave)),
            ("stipple", solo(StructureMode::StippleConstellation)),
            ("tangent", solo(StructureMode::TangentCaustics)),
        ];
        variants.push((
            "ribbons_with_echo",
            SceneTraits {
                stack: LayerStack::solo(StructureMode::OrbitRibbons),
                ribbon_echo_alpha: 0.35,
                echo_layers: 3,
                ..SceneTraits::default()
            },
        ));
        variants.push((
            "stacked_web_veil_stipple",
            SceneTraits {
                stack: LayerStack {
                    primary: StructureMode::TriangleWeb,
                    underlay: Some(StackLayer {
                        vocabulary: StructureMode::NebulaVeil,
                        alpha: 0.30,
                    }),
                    accent: Some(StackLayer {
                        vocabulary: StructureMode::StippleConstellation,
                        alpha: 0.10,
                    }),
                },
                ..SceneTraits::default()
            },
        ));
        variants.push((
            "rotational_symmetry",
            SceneTraits { symmetry: SymmetryOp::Rotational { k: 4 }, ..SceneTraits::default() },
        ));
        variants.push((
            "dihedral_symmetry",
            SceneTraits {
                stack: LayerStack::solo(StructureMode::HarmonicWeave),
                symmetry: SymmetryOp::Dihedral { k: 3 },
                ..SceneTraits::default()
            },
        ));
        variants.push((
            "stardust",
            SceneTraits {
                stardust: StardustTraits {
                    count: 64,
                    brightness: 1.0,
                    seed: 0x5EED_CAFE,
                    lightness: 0.72,
                    chroma: 0.03,
                },
                ..SceneTraits::default()
            },
        ));
        variants
    }

    #[test]
    fn test_all_trait_variants_parallel_match_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(24, 18, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        for (label, traits) in trait_variants() {
            let accum_params = AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start: 0,
                step_end: scene.step_count(),
                hdr_scale: 3.5,
                traits,
            };

            let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            accumulate_spectral_steps(
                &mut serial,
                &accum_params,
                AccumulationBackend::SerialReference,
            );

            for thread_count in [2usize, 3] {
                let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
                ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .expect("thread pool should build")
                    .install(|| {
                        accumulate_spectral_steps(
                            &mut parallel,
                            &accum_params,
                            AccumulationBackend::ParallelScanlines,
                        );
                    });
                assert_spd_buffers_bits_eq(
                    &parallel,
                    &serial,
                    &format!("variant={label}/threads={thread_count}"),
                );
            }
        }
    }

    #[test]
    fn test_every_vocabulary_deposits_finite_nonzero_energy() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(24, 18, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        for (label, traits) in trait_variants() {
            let mut accum = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            accumulate_spectral_steps(
                &mut accum,
                &AccumulationParams {
                    scene,
                    ctx: &ctx,
                    velocity_calc: &velocity_calc,
                    step_start: 0,
                    step_end: scene.step_count(),
                    hdr_scale: 3.5,
                    traits,
                },
                AccumulationBackend::SerialReference,
            );

            let total: f64 = accum.iter().flat_map(|bins| bins.iter()).sum();
            assert!(
                total.is_finite() && total > 0.0,
                "variant {label} deposited invalid energy: {total}"
            );
            for (pixel, bins) in accum.iter().enumerate() {
                for (bin, value) in bins.iter().enumerate() {
                    assert!(
                        value.is_finite() && *value >= 0.0,
                        "variant {label} pixel {pixel} bin {bin} invalid: {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_degenerate_collinear_geometry_stays_finite_for_all_vocabularies() {
        // Three collinear, slowly drifting bodies: degenerate triangles must
        // not produce NaN energy in any vocabulary (veil fill lines collapse,
        // tangent directions shrink, weave controls coincide).
        let steps = 6usize;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..steps)
                    .map(|step| {
                        let t = step as f64 * 0.01;
                        Vector3::new(f64::from(body as u32) * 0.2 + t, 0.5, 0.0)
                    })
                    .collect()
            })
            .collect();
        let colors: Vec<Vec<OklabColor>> = (0..3).map(|_| vec![(0.7, 0.1, 0.05); steps]).collect();
        let body_alphas = vec![0.5, 0.5, 0.5];
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(16, 12, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        for (label, traits) in trait_variants() {
            let mut accum = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            accumulate_spectral_steps(
                &mut accum,
                &AccumulationParams {
                    scene,
                    ctx: &ctx,
                    velocity_calc: &velocity_calc,
                    step_start: 0,
                    step_end: steps,
                    hdr_scale: 2.0,
                    traits,
                },
                AccumulationBackend::SerialReference,
            );
            let total: f64 = accum.iter().flat_map(|bins| bins.iter()).sum();
            assert!(total.is_finite(), "variant {label} produced non-finite energy");
        }
    }

    #[test]
    fn test_stardust_splats_exactly_once_across_chunked_accumulation() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let ctx = RenderContext::new(20, 14, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);
        let traits = trait_variants()
            .into_iter()
            .find(|(label, _)| *label == "stardust")
            .expect("stardust variant exists")
            .1;

        let make_params = |step_start: usize, step_end: usize| AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start,
            step_end,
            hdr_scale: 3.0,
            traits,
        };

        // Single full-range accumulation.
        let mut single = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut single,
            &make_params(0, scene.step_count()),
            AccumulationBackend::SerialReference,
        );

        // Chunked accumulation (as the checkpointed video passes run it).
        let mut chunked = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut chunked,
            &make_params(0, 2),
            AccumulationBackend::SerialReference,
        );
        accumulate_spectral_steps(
            &mut chunked,
            &make_params(2, scene.step_count()),
            AccumulationBackend::SerialReference,
        );

        assert_spd_buffers_bits_eq(&chunked, &single, "stardust/chunked-vs-single");
    }

    #[test]
    fn test_scanline_accumulation_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let render_config = RenderConfig { hdr_scale: 3.5, bloom_mode: BloomMode::None };
        let ctx = RenderContext::new(9, 7, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let accum_params = AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: scene.step_count(),
            hdr_scale: render_config.hdr_scale,
            traits: SceneTraits::default(),
        };

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(&mut serial, &accum_params, AccumulationBackend::SerialReference);

        for thread_count in [1usize, 2, 3, ctx.height_usize] {
            let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    accumulate_spectral_steps(
                        &mut parallel,
                        &accum_params,
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
            final_energy > single_energy.saturating_mul(2),
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

        let serial_single = render_single_frame_spectral_serial_reference(scene, &levels, settings)
            .expect("serial single frame render should succeed");
        let serial_final = render_final_frame_spectral_serial_reference(scene, &levels, settings)
            .expect("serial final frame render should succeed");

        for thread_count in [1usize, 2, 3, resolved.height as usize] {
            let (parallel_single, parallel_final) = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    (
                        render_single_frame_spectral(scene, &levels, settings)
                            .expect("parallel single frame render should succeed"),
                        render_final_frame_spectral(scene, &levels, settings)
                            .expect("parallel final frame render should succeed"),
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
    fn test_rayon_pool_respects_custom_stack_size() {
        let pool = ThreadPoolBuilder::new()
            .stack_size(constants::THREAD_STACK_SIZE)
            .num_threads(2)
            .build()
            .expect("pool with THREAD_STACK_SIZE should build");

        let result = pool.install(|| {
            let mut v = vec![0u64; 1024];
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = i as u64;
            }
            v.iter().sum::<u64>()
        });
        assert_eq!(result, (0..1024u64).sum::<u64>());
    }

    #[test]
    fn test_compute_softness_radius_disabled() {
        let mut cfg = baseline_resolved_config(1920, 1080);
        cfg.enable_perceptual_blur = false;
        assert!(compute_softness_radius(&cfg, BloomMode::Dog).is_none());
    }

    #[test]
    fn test_compute_softness_radius_enabled_returns_some() {
        let mut cfg = baseline_resolved_config(1920, 1080);
        cfg.enable_perceptual_blur = true;
        let radius = compute_softness_radius(&cfg, BloomMode::Dog);
        assert!(radius.is_some());
        assert!(radius.expect("softness radius should be some") >= 1);
    }

    #[test]
    fn test_compute_softness_radius_high_softness_uses_smaller_scale() {
        let mut low = baseline_resolved_config(1920, 1080);
        low.enable_perceptual_blur = true;

        let mut high = low.clone();
        high.enable_chromatic_bloom = true;
        high.enable_glow = true;
        high.enable_atmospheric_depth = true;

        let r_low = compute_softness_radius(&low, BloomMode::Dog)
            .expect("low softness radius should resolve");
        let r_high = compute_softness_radius(&high, BloomMode::Dog)
            .expect("high softness radius should resolve");
        assert!(
            r_high <= r_low,
            "higher softness stack should produce equal or smaller radius: {r_high} vs {r_low}",
        );
    }
}
