//! Rendering module: histogram passes, color mapping, line drawing, and output
//!
//! This module provides a complete rendering pipeline for the three-body problem visualization,
//! including coordinate transformations, line drawing, post-processing effects, and video output.

use nalgebra::Vector3;
use std::io::BufWriter;
use std::mem::size_of;
use std::path::Path;
use tracing::info;

// Module declarations
pub mod batch_drawing;
pub mod color;
pub mod constants;
pub mod context;
pub mod drawing;
mod effect_config;
pub mod effect_randomizer;
pub mod effects;
pub mod error;
pub mod histogram;
pub mod parameter_descriptors;
pub mod randomizable_config;
pub mod spectral_output;
mod spectral_passes;
mod tonemapping;
pub mod types;
pub mod velocity_hdr;
pub mod video;

// Import from our submodules
use self::error::{RenderError, Result};
#[cfg(test)]
use self::tonemapping::{tonemap_core, tonemap_to_16bit};

// Re-export core types and functions for public API compatibility
pub use color::{OklabColor, generate_body_color_sequences};
pub use drawing::{
    LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral, parallel_blur_2d_rgba,
};
pub use effect_config::{build_effect_config_from_resolved, compute_softness_radius};
pub use effects::{DogBloomConfig, apply_dog_bloom, try_apply_dog_bloom};
#[cfg(test)]
pub(crate) use spectral_passes::{
    AccumulationBackend, AccumulationParams, accumulate_spectral_steps,
    pass_1_build_histogram_spectral_serial_reference,
    pass_2_write_frames_spectral_serial_reference, render_final_frame_spectral_serial_reference,
    render_single_frame_spectral, render_single_frame_spectral_serial_reference,
};
pub use spectral_passes::{
    Pass2Params, pass_1_build_histogram_spectral, pass_2_write_frames_spectral,
    render_final_frame_spectral,
};
pub use types::{ChannelLevels, ToneMappingControls};
pub use video::{VideoEncodingOptions, create_video_from_frames_singlepass};

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
    /// Whether spectral-to-RGBA conversion uses the enhanced saturation matrix.
    pub sat_boost: bool,
    /// Whether tonemapping uses the punchy `AgX` output matrix.
    pub aces_tweak: bool,
    /// Whether radial spectral dispersion uses the stronger boosted profile.
    pub dispersion_boost: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            hdr_scale: constants::DEFAULT_HDR_SCALE,
            bloom_mode: BloomMode::Dog,
            sat_boost: true,
            aces_tweak: true,
            dispersion_boost: true,
        }
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

    /// Validate the scene's buffer shapes before render passes index into them.
    ///
    /// # Errors
    ///
    /// Returns an error when the scene does not contain exactly three bodies,
    /// contains no steps, or has mismatched color/position/alpha lengths.
    pub fn validate(self) -> Result<()> {
        if self.positions.len() != 3 {
            return Err(RenderError::InvalidScene {
                reason: format!("expected 3 position tracks, got {}", self.positions.len()),
            });
        }
        if self.colors.len() != 3 {
            return Err(RenderError::InvalidScene {
                reason: format!("expected 3 color tracks, got {}", self.colors.len()),
            });
        }
        if self.body_alphas.len() != 3 {
            return Err(RenderError::InvalidScene {
                reason: format!("expected 3 body alphas, got {}", self.body_alphas.len()),
            });
        }

        let step_count = self.positions[0].len();
        if step_count == 0 {
            return Err(RenderError::InvalidScene {
                reason: "position tracks must contain at least one step".to_string(),
            });
        }

        for (body_idx, positions) in self.positions.iter().enumerate() {
            if positions.len() != step_count {
                return Err(RenderError::InvalidScene {
                    reason: format!(
                        "position track {body_idx} has {} steps, expected {step_count}",
                        positions.len()
                    ),
                });
            }
        }
        for (body_idx, colors) in self.colors.iter().enumerate() {
            if colors.len() != step_count {
                return Err(RenderError::InvalidScene {
                    reason: format!(
                        "color track {body_idx} has {} steps, expected {step_count}",
                        colors.len()
                    ),
                });
            }
        }
        for (body_idx, alpha) in self.body_alphas.iter().enumerate() {
            if !alpha.is_finite() {
                return Err(RenderError::InvalidScene {
                    reason: format!("body alpha {body_idx} must be finite"),
                });
            }
        }

        Ok(())
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
    #[must_use]
    pub fn new(
        resolved_config: &'a randomizable_config::ResolvedEffectConfig,
        render_config: &'a RenderConfig,
        aspect_correction: bool,
    ) -> Self {
        Self { resolved_config, render_config, aspect_correction }
    }
}

fn rgb16_to_png_bytes(rgb_img: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> Vec<u8> {
    let samples = rgb_img.as_raw();
    let mut data = Vec::with_capacity(samples.len() * size_of::<u16>());
    for &sample in samples {
        data.extend_from_slice(&sample.to_be_bytes());
    }
    data
}

/// Save a 16-bit RGB image as a color-managed sRGB PNG.
///
/// # Errors
///
/// Returns an error if the image cannot be encoded or written to `path`.
pub fn save_image_as_png_16bit(
    rgb_img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();

    let file = std::fs::File::create(path).map_err(|e| RenderError::ImageEncoding {
        reason: format!("Failed to create {}: {e}", path.display()),
    })?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, rgb_img.width(), rgb_img.height());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);

    let mut png_writer = encoder.write_header().map_err(|e| RenderError::ImageEncoding {
        reason: format!("Failed to write PNG header for {}: {e}", path.display()),
    })?;
    png_writer.write_image_data(&rgb16_to_png_bytes(rgb_img)).map_err(|e| {
        RenderError::ImageEncoding {
            reason: format!("Failed to write PNG data for {}: {e}", path.display()),
        }
    })?;
    png_writer.finish().map_err(|e| RenderError::ImageEncoding {
        reason: format!("Failed to finish PNG {}: {e}", path.display()),
    })?;

    info!("   Saved 16-bit PNG (sRGB tagged) => {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::context::RenderContext;
    use crate::render::histogram::HistogramData;
    use crate::render::randomizable_config::ResolvedEffectConfig;
    use crate::spectrum::NUM_BINS;
    use nalgebra::Vector3;
    use rayon::ThreadPoolBuilder;

    fn default_levels() -> ChannelLevels {
        ChannelLevels::new(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    }

    #[test]
    fn test_save_image_as_png_16bit_writes_srgb_chunk() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let path = tmp.path().join("tagged.png");
        let img = ImageBuffer::from_raw(1, 1, vec![u16::MAX, 0, 0])
            .expect("test image buffer should be valid");

        save_image_as_png_16bit(&img, &path).expect("PNG save should succeed");

        let bytes = std::fs::read(&path).expect("PNG should be readable");
        assert!(bytes.windows(4).any(|chunk| chunk == b"sRGB"));
        image::open(&path).expect("tagged PNG should still decode");
    }

    #[test]
    fn test_spectral_scene_validation_rejects_shape_mismatch() {
        let positions = vec![vec![Vector3::new(0.0, 0.0, 0.0)]; 2];
        let colors = vec![vec![(0.5, 0.0, 0.0)]; 3];
        let body_alphas = vec![1.0, 1.0, 1.0];

        let err = SpectralScene::new(&positions, &colors, &body_alphas)
            .validate()
            .expect_err("two position tracks should be invalid");
        assert!(matches!(err, RenderError::InvalidScene { .. }));
    }

    #[test]
    fn test_spectral_scene_validation_rejects_empty_steps() {
        let positions = vec![Vec::new(), Vec::new(), Vec::new()];
        let colors = vec![Vec::new(), Vec::new(), Vec::new()];
        let body_alphas = vec![1.0, 1.0, 1.0];

        assert!(SpectralScene::new(&positions, &colors, &body_alphas).validate().is_err());
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
            pass_1_build_histogram_spectral_serial_reference(scene, frame_interval, settings)
                .expect("serial histogram pass should succeed");
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
        let result = tonemap_core(0.0, 0.0, 0.0, 0.0, &default_levels(), true);
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tonemap_produces_valid_range() {
        let levels = default_levels();
        for alpha in [0.1, 0.5, 1.0] {
            let result = tonemap_core(0.5, 0.3, 0.8, alpha, &levels, true);
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
        let result = tonemap_core(8.0, 8.0, 8.0, 1.0, &levels, true);

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

        let unity_out = tonemap_core(2.0, 2.0, 2.0, 1.0, &unity, true);
        let reduced_out = tonemap_core(2.0, 2.0, 2.0, 1.0, &reduced, true);

        assert!(reduced_out[0] < unity_out[0]);
        assert!(reduced_out[1] < unity_out[1]);
        assert!(reduced_out[2] < unity_out[2]);
    }

    #[test]
    fn test_agx_tweak_changes_output() {
        let levels = default_levels();

        let tweaked = tonemap_core(0.5, 0.3, 0.7, 0.8, &levels, true);
        let original = tonemap_core(0.5, 0.3, 0.7, 0.8, &levels, false);

        let diff = (tweaked[0] - original[0]).abs()
            + (tweaked[1] - original[1]).abs()
            + (tweaked[2] - original[2]).abs();
        assert!(diff > 1e-6, "AgX punchy tweak should produce different tonemapping");
    }

    #[test]
    fn test_tonemap_16bit_range() {
        let levels = default_levels();
        let result = tonemap_to_16bit(0.5, 0.4, 0.6, 0.9, &levels, true);
        for ch in result {
            assert!(u32::from(ch) <= 65535, "16-bit channel {ch} out of range");
        }
    }

    #[test]
    fn test_build_effect_config_uses_dog_bloom_exclusively() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: true, ..baseline_resolved_config(640, 360) };
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);

        assert_eq!(effect_config.bloom_mode, BloomMode::Dog);
        assert_eq!(effect_config.blur_radius_px, 0);
        assert!(effect_config.dog_config.inner_sigma > 0.0);
    }

    #[test]
    fn test_build_effect_config_uses_gaussian_bloom_exclusively() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: true, ..baseline_resolved_config(640, 360) };
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Gaussian,
            ..Default::default()
        };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);

        assert_eq!(effect_config.bloom_mode, BloomMode::Gaussian);
        assert!(effect_config.blur_radius_px > 0);
    }

    #[test]
    fn test_build_effect_config_disables_bloom_mode_when_bloom_is_off() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: false, ..baseline_resolved_config(640, 360) };
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);

        assert_eq!(effect_config.bloom_mode, BloomMode::None);
        assert_eq!(effect_config.blur_radius_px, 0);
    }

    #[test]
    fn test_build_effect_config_disables_texture_for_proxy_resolution() {
        let resolved = ResolvedEffectConfig {
            enable_fine_texture: true,
            ..baseline_resolved_config(640, 360)
        };
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

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
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

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
    fn test_build_effect_config_routes_crisp_finishes_to_still_and_video() {
        let resolved = ResolvedEffectConfig {
            enable_micro_contrast: true,
            enable_edge_luminance: true,
            enable_color_grade: true,
            ..baseline_resolved_config(1920, 1080)
        };
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

        let still_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);
        let video_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Video);

        for effect_config in [&still_config, &video_config] {
            assert!(effect_config.prismatic_sparkle_enabled);
            assert!(effect_config.prismatic_sparkle_config.radius <= 3);
            assert!(effect_config.prismatic_sparkle_config.strength > 0.0);
            assert!(effect_config.crystal_facets_enabled);
            assert!(effect_config.crystal_facet_config.cell_size >= 10);
            assert!(effect_config.ink_cut_edges_enabled);
            assert!(effect_config.ink_cut_config.strength > 0.0);
        }
        assert_eq!(
            still_config.prismatic_sparkle_config.radius,
            video_config.prismatic_sparkle_config.radius,
        );
        assert_eq!(
            still_config.crystal_facet_config.cell_size,
            video_config.crystal_facet_config.cell_size,
        );
        assert_eq!(still_config.ink_cut_config.threshold, video_config.ink_cut_config.threshold);
    }

    #[test]
    fn test_crisp_luxury_image_effects_skip_proxy_resolution() {
        let resolved = ResolvedEffectConfig {
            enable_micro_contrast: true,
            enable_edge_luminance: true,
            enable_color_grade: true,
            ..baseline_resolved_config(640, 360)
        };
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

        let effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Still);
        let video_effect_config =
            build_effect_config_from_resolved(&resolved, &render_config, FinishOutputMode::Video);

        for effect_config in [&effect_config, &video_effect_config] {
            assert!(!effect_config.crystal_facets_enabled);
            assert!(!effect_config.ink_cut_edges_enabled);
            assert!(!effect_config.prismatic_sparkle_enabled);
        }
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
        let render_config = RenderConfig {
            hdr_scale: resolved.hdr_scale,
            bloom_mode: BloomMode::Dog,
            ..Default::default()
        };

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
        let render_config =
            RenderConfig { hdr_scale: 3.5, bloom_mode: BloomMode::None, ..Default::default() };
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
        let render_config =
            RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::Dog, ..Default::default() };

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
        )
        .expect("clean histogram pass should succeed");

        let styled_hist = pass_1_build_histogram_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            1,
            SpectralRenderSettings::new(&stylized, &render_config, false),
        )
        .expect("stylized histogram pass should succeed");

        assert_ne!(clean_hist.data(), styled_hist.data());
    }

    #[test]
    fn test_histogram_pass_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config =
            RenderConfig { hdr_scale: 2.8, bloom_mode: BloomMode::Dog, ..Default::default() };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let serial = pass_1_build_histogram_spectral_serial_reference(scene, 2, settings)
            .expect("serial histogram pass should succeed");

        for thread_count in [1usize, 2, 3, resolved.height as usize] {
            let parallel = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    pass_1_build_histogram_spectral(scene, 2, settings)
                        .expect("parallel histogram pass should succeed")
                });
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
        let render_config =
            RenderConfig { hdr_scale: 6.0, bloom_mode: BloomMode::None, ..Default::default() };
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
    fn test_final_still_is_independent_from_temporal_smoothing() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config =
            RenderConfig { hdr_scale: 3.2, bloom_mode: BloomMode::None, ..Default::default() };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let levels = ChannelLevels::new(0.0, 0.12, 0.0, 0.12, 0.0, 0.12);

        let (_, unsmoothed_last) =
            capture_frame_bytes_with_pool(scene, 1, &levels, settings, false, false, 2);
        let (_, smoothed_last) =
            capture_frame_bytes_with_pool(scene, 1, &levels, settings, true, false, 2);
        let still = render_final_frame_spectral(scene, &levels, settings)
            .expect("dedicated still render should succeed");

        let unsmoothed_last = unsmoothed_last.expect("pass 2 should capture final frame");
        let smoothed_last = smoothed_last.expect("pass 2 should capture final frame");

        assert_ne!(
            smoothed_last.as_raw(),
            unsmoothed_last.as_raw(),
            "fixture should exercise temporal smoothing",
        );
        assert_ne!(
            smoothed_last.as_raw(),
            still.as_raw(),
            "dedicated still render should not inherit temporal smoothing",
        );
    }

    #[test]
    fn test_render_previews_parallel_match_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config =
            RenderConfig { hdr_scale: 3.2, bloom_mode: BloomMode::None, ..Default::default() };
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
        let render_config =
            RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::None, ..Default::default() };
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
        let render_config =
            RenderConfig { hdr_scale: 4.2, bloom_mode: BloomMode::Dog, ..Default::default() };
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
