//! Application orchestration and workflow management
//!
//! This module breaks down the main application flow into clean, focused functions,
//! each with a single responsibility. This improves testability, readability, and
//! maintainability.

use crate::drift::parse_drift_mode;
use crate::drift_config::{ResolvedDriftConfig, resolve_drift_config};
use crate::error::{ConfigError, Result};
use crate::generation_log::{
    DriftConfig, GenerationLogger, GenerationRecord, LoggedRenderConfig, OrbitInfo,
    SimulationConfig,
};
use crate::render::{
    self, ChannelLevels, RenderConfig, SpectralRenderSettings, SpectralScene, ToneMappingControls,
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
    generate_body_color_sequences, pass_1_build_histogram_spectral, pass_2_write_frames_spectral,
    save_image_as_png_16bit,
};
use crate::sim::{self, Body, Sha3RandomByteStream, TrajectoryResult};
use image::{ImageBuffer, Rgb};
use nalgebra::Vector3;
use std::fs;
use tracing::{info, warn};

/// Museum-quality enhancement flags (all default to true / enabled).
#[derive(Clone, Debug)]
pub struct Enhancements {
    pub chroma_boost: bool,
    pub sat_boost: bool,
    pub aces_tweak: bool,
    pub alpha_variation: bool,
    pub aspect_correction: bool,
    pub dispersion_boost: bool,
}

impl Default for Enhancements {
    fn default() -> Self {
        Self {
            chroma_boost: true,
            sat_boost: true,
            aces_tweak: true,
            alpha_variation: true,
            aspect_correction: false,
            dispersion_boost: true,
        }
    }
}

/// Configuration recorded for generation logging.
pub struct GenerationLogConfig {
    pub num_steps_sim: usize,
    pub width: u32,
    pub height: u32,
    pub clip_black: f64,
    pub clip_white: f64,
    pub alpha_denom: usize,
    pub alpha_compress: f64,
    pub escape_threshold: f64,
    pub drift_mode: String,
    pub bloom_mode: String,
    pub dog_strength: f64,
    pub dog_sigma: Option<f64>,
    pub dog_ratio: f64,
    pub hdr_mode: String,
    pub hdr_scale: f64,
    pub perceptual_blur: String,
    pub perceptual_blur_radius: Option<usize>,
    pub perceptual_blur_strength: f64,
    pub perceptual_gamut_mode: String,
    pub min_mass: f64,
    pub max_mass: f64,
    pub location: f64,
    pub velocity: f64,
    pub chaos_weight: f64,
    pub equil_weight: f64,
}

/// Initialize application directories
pub fn setup_directories() -> Result<()> {
    fs::create_dir_all("pics").map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: "pics".to_string(),
        error: e,
    })?;

    fs::create_dir_all("vids").map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: "vids".to_string(),
        error: e,
    })?;

    Ok(())
}

/// Parse and validate hex seed
pub fn parse_seed(seed: &str) -> Result<Vec<u8>> {
    let hex_seed = seed.strip_prefix("0x").unwrap_or(seed);

    hex::decode(hex_seed)
        .map_err(|e| ConfigError::InvalidSeed { seed: seed.to_string(), error: e }.into())
}

/// Derive noise seed from simulation seed for nebula generation
pub fn derive_noise_seed(seed_bytes: &[u8]) -> i32 {
    let get_or_zero = |idx| seed_bytes.get(idx).copied().unwrap_or(0);
    i32::from_le_bytes([get_or_zero(0), get_or_zero(1), get_or_zero(2), get_or_zero(3)])
}

/// Run Borda selection to find the best orbit
pub fn run_borda_selection(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    num_steps_sim: usize,
    chaos_weight: f64,
    equil_weight: f64,
    escape_threshold: f64,
) -> Result<(Vec<Body>, TrajectoryResult)> {
    info!("STAGE 1/7: Borda search over {} random orbits...", num_sims);

    sim::select_best_trajectory(
        rng,
        num_sims,
        num_steps_sim,
        chaos_weight,
        equil_weight,
        escape_threshold,
    )
}

/// Re-run the best orbit to get full trajectory
pub fn simulate_best_orbit(best_bodies: Vec<Body>, num_steps_sim: usize) -> Vec<Vec<Vector3<f64>>> {
    info!("STAGE 2/7: Re-running best orbit for {} steps...", num_steps_sim);
    let sim_result = sim::get_positions(best_bodies, num_steps_sim);
    info!("   => Done.");
    sim_result.positions
}

/// Apply drift transformation to positions
pub fn apply_drift_transformation(
    positions: &mut [Vec<Vector3<f64>>],
    drift_mode: &str,
    drift_scale: Option<f64>,
    drift_arc_fraction: Option<f64>,
    drift_orbit_eccentricity: Option<f64>,
    rng: &mut Sha3RandomByteStream,
) -> Option<ResolvedDriftConfig> {
    info!("STAGE 2.5/7: Resolving drift configuration...");

    let resolved =
        resolve_drift_config(drift_scale, drift_arc_fraction, drift_orbit_eccentricity, rng);

    info!("Applying {} drift...", drift_mode);
    let num_steps = positions[0].len();
    let drift_params = resolved.to_drift_parameters();

    if crate::utils::is_zero(drift_params.arc_fraction)
        && drift_mode.to_lowercase().starts_with("ell")
    {
        warn!("Elliptical drift requested with zero arc fraction; skipping motion");
    }

    let mut drift_transform = parse_drift_mode(drift_mode, rng, drift_params, num_steps);
    drift_transform.apply(positions, constants::DEFAULT_DT);

    info!("   => Drift applied successfully");
    Some(resolved)
}

/// Generate color sequences and alpha values for bodies
pub fn generate_colors(
    rng: &mut Sha3RandomByteStream,
    num_steps_sim: usize,
    alpha_denom: usize,
    enhancements: &Enhancements,
) -> (Vec<Vec<render::OklabColor>>, Vec<f64>) {
    info!("STAGE 3/7: Generating color sequences + alpha...");
    generate_body_color_sequences(
        rng,
        num_steps_sim,
        alpha_denom,
        enhancements.chroma_boost,
        enhancements.alpha_variation,
    )
}

/// Build histogram and determine color levels
pub fn build_histogram_and_levels(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<render::OklabColor>],
    body_alphas: &[f64],
    resolved_config: &render::randomizable_config::ResolvedEffectConfig,
    noise_seed: i32,
    render_config: &RenderConfig,
    aspect_correction: bool,
) -> Result<ChannelLevels> {
    info!("STAGE 5/7: PASS 1 => building global histogram...");

    let target_frames = constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES;
    let frame_interval = (positions[0].len() / target_frames as usize).max(1);

    let histogram = pass_1_build_histogram_spectral(
        SpectralScene::new(positions, colors, body_alphas),
        frame_interval,
        SpectralRenderSettings::new(resolved_config, render_config, noise_seed, aspect_correction),
    );

    info!("STAGE 6/7: Determine global black/white/gamma...");
    let analysis = render::histogram::analyze_tonemapping(
        histogram.data(),
        resolved_config.clip_black,
        resolved_config.clip_white,
    );

    info!(
        "   => R:[{:.3e},{:.3e}] G:[{:.3e},{:.3e}] B:[{:.3e},{:.3e}] exposure={:.3} near_clip={:.3}%",
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        analysis.exposure_scale,
        analysis.near_clip_ratio * constants::PERCENT_FACTOR
    );

    Ok(ChannelLevels::with_tone_mapping(
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
    ))
}

/// Render full video
pub fn render_video(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_vid: &str,
    output_png: &str,
    fast_encode: bool,
    enable_temporal_smoothing: bool,
) -> Result<()> {
    if fast_encode {
        info!("STAGE 7/7: PASS 2 => final frames => video (FAST ENCODE MODE)...");
    } else {
        info!("STAGE 7/7: PASS 2 => final frames => video (HIGH QUALITY MODE)...");
    }

    let frame_rate = constants::DEFAULT_VIDEO_FPS;
    let target_frames = constants::DEFAULT_TARGET_FRAMES;
    let frame_interval = (scene.step_count() / target_frames as usize).max(1);

    let mut last_frame_png: Option<ImageBuffer<Rgb<u16>, Vec<u16>>> = None;
    let video_options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    create_video_from_frames_singlepass(
        settings.resolved_config.width,
        settings.resolved_config.height,
        frame_rate,
        |out| {
            pass_2_write_frames_spectral(
                scene,
                frame_interval,
                levels,
                settings,
                |buf_8bit| {
                    out.write_all(buf_8bit).map_err(render::error::RenderError::VideoEncoding)?;
                    Ok(())
                },
                &mut last_frame_png,
                enable_temporal_smoothing,
            )?;
            Ok(())
        },
        output_vid,
        &video_options,
    )?;

    // Save final frame
    if let Some(last_frame) = last_frame_png {
        info!("Attempting to save 16-bit PNG to: {}", output_png);
        save_image_as_png_16bit(&last_frame, output_png)?;
    } else {
        warn!("Warning: No final frame was generated to save as PNG.");
    }

    Ok(())
}

/// Log generation parameters for reproducibility
pub fn log_generation(
    config: &GenerationLogConfig,
    file_name: &str,
    seed: &str,
    drift_config: &Option<ResolvedDriftConfig>,
    num_sims: usize,
    best_info: &TrajectoryResult,
    randomization_log: Option<&render::effect_randomizer::RandomizationLog>,
) {
    let logger = GenerationLogger::new();

    let mut record = GenerationRecord::new(file_name.to_string(), format!("0x{}", seed));

    record.render_config = LoggedRenderConfig {
        width: config.width,
        height: config.height,
        clip_black: config.clip_black,
        clip_white: config.clip_white,
        alpha_denom: config.alpha_denom,
        alpha_compress: config.alpha_compress,
        bloom_mode: config.bloom_mode.clone(),
        dog_strength: config.dog_strength,
        dog_sigma: config.dog_sigma,
        dog_ratio: config.dog_ratio,
        hdr_mode: config.hdr_mode.clone(),
        hdr_scale: config.hdr_scale,
        perceptual_blur: config.perceptual_blur.clone(),
        perceptual_blur_radius: config.perceptual_blur_radius,
        perceptual_blur_strength: config.perceptual_blur_strength,
        perceptual_gamut_mode: config.perceptual_gamut_mode.clone(),
    };

    record.drift_config = if let Some(drift) = drift_config {
        DriftConfig {
            enabled: true,
            mode: config.drift_mode.clone(),
            scale: drift.scale,
            arc_fraction: drift.arc_fraction,
            orbit_eccentricity: drift.orbit_eccentricity,
            randomized: drift.was_randomized,
        }
    } else {
        DriftConfig {
            enabled: false,
            mode: "none".to_string(),
            scale: 0.0,
            arc_fraction: 0.0,
            orbit_eccentricity: 0.0,
            randomized: false,
        }
    };

    record.simulation_config = SimulationConfig {
        num_sims,
        num_steps_sim: config.num_steps_sim,
        location: config.location,
        velocity: config.velocity,
        min_mass: config.min_mass,
        max_mass: config.max_mass,
        chaos_weight: config.chaos_weight,
        equil_weight: config.equil_weight,
        escape_threshold: config.escape_threshold,
    };

    record.orbit_info = OrbitInfo {
        selected_index: best_info.selected_index,
        weighted_score: best_info.total_score_weighted,
        total_candidates: num_sims,
        discarded_count: best_info.discarded_count,
    };

    // Include randomization log if provided
    record.randomization_log = randomization_log.cloned();

    logger.log_generation(record);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_seed_valid() {
        let result = parse_seed("0x100033");
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert_eq!(bytes, vec![0x10, 0x00, 0x33]);
    }

    #[test]
    fn test_parse_seed_no_prefix() {
        let result = parse_seed("100033");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_seed_invalid() {
        let result = parse_seed("0xZZZ");
        assert!(result.is_err());
    }

    #[test]
    fn test_derive_noise_seed() {
        let seed = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let noise = derive_noise_seed(&seed);
        assert_eq!(noise, i32::from_le_bytes([0x01, 0x02, 0x03, 0x04]));
    }

    #[test]
    fn test_enhancements_default_quality_profile() {
        let e = Enhancements::default();
        assert!(e.chroma_boost);
        assert!(e.sat_boost);
        assert!(e.aces_tweak);
        assert!(e.alpha_variation);
        assert!(!e.aspect_correction);
        assert!(e.dispersion_boost);
    }

    #[test]
    fn test_enhancements_selective_disable() {
        let e = Enhancements {
            chroma_boost: false,
            sat_boost: true,
            aces_tweak: false,
            alpha_variation: true,
            aspect_correction: false,
            dispersion_boost: true,
        };
        assert!(!e.chroma_boost);
        assert!(e.sat_boost);
        assert!(!e.aces_tweak);
        assert!(e.alpha_variation);
        assert!(!e.aspect_correction);
        assert!(e.dispersion_boost);
    }

    #[test]
    fn test_generate_colors_with_enhancements() {
        use crate::sim::Sha3RandomByteStream;
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let enhancements = Enhancements::default();
        let (colors, alphas) = generate_colors(&mut rng, 100, 15_000_000, &enhancements);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for body_colors in &colors {
            assert_eq!(body_colors.len(), 100);
        }
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "default enhancements should enable alpha variation");
    }

    #[test]
    fn test_generate_colors_no_enhancements() {
        use crate::sim::Sha3RandomByteStream;
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let enhancements =
            Enhancements { alpha_variation: false, chroma_boost: false, ..Enhancements::default() };
        let (colors, alphas) = generate_colors(&mut rng, 100, 15_000_000, &enhancements);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas[0], alphas[1]);
        assert_eq!(alphas[1], alphas[2]);
    }

    /// Run the full seed-to-pixels pipeline at minimal scale and return the
    /// raw 16-bit pixel buffer.  Two calls with the same seed MUST return
    /// bitwise-identical buffers on the same architecture.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PipelineRenderMode {
        DefaultParallel,
        SerialReference,
    }

    fn assert_pixel_buffers_eq(actual: &[u16], expected: &[u16], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: pixel buffer lengths differ");

        if actual != expected {
            let diff_count = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            let first = actual.iter().zip(expected).position(|(a, b)| a != b).unwrap();
            panic!(
                "{label}: pixel buffers differ: {diff_count} of {} values, \
                 first at index {first} ({} vs {})",
                actual.len(),
                actual[first],
                expected[first],
            );
        }
    }

    fn run_full_pipeline(seed: &[u8], mode: PipelineRenderMode) -> Vec<u16> {
        use crate::sim::Sha3RandomByteStream;

        let width = 64u32;
        let height = 36u32;
        let num_sims = 20;
        let num_steps = 5_000;

        let noise_seed = derive_noise_seed(seed);
        let mut rng = Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0);

        let config = render::randomizable_config::RandomizableEffectConfig {
            enable_bloom: Some(false),
            enable_glow: Some(false),
            enable_chromatic_bloom: Some(false),
            enable_perceptual_blur: Some(false),
            enable_micro_contrast: Some(false),
            enable_gradient_map: Some(false),
            enable_color_grade: Some(false),
            enable_champleve: Some(false),
            enable_aether: Some(false),
            enable_opalescence: Some(false),
            enable_edge_luminance: Some(false),
            enable_atmospheric_depth: Some(false),
            enable_fine_texture: Some(false),
            nebula_strength: Some(0.0),
            ..Default::default()
        };
        let (resolved, _) = config.resolve(&mut rng, width, height);

        let (best_bodies, _) =
            crate::sim::select_best_trajectory(&mut rng, num_sims, num_steps, 0.75, 11.0, -0.3)
                .expect("Borda search should find at least one valid orbit");

        let mut positions = simulate_best_orbit(best_bodies, num_steps);

        apply_drift_transformation(&mut positions, "elliptical", None, None, None, &mut rng);

        let enhancements = Enhancements {
            chroma_boost: false,
            sat_boost: false,
            aces_tweak: false,
            alpha_variation: false,
            aspect_correction: false,
            dispersion_boost: false,
        };
        let (colors, body_alphas) = generate_colors(&mut rng, num_steps, 15_000_000, &enhancements);

        let render_config =
            render::RenderConfig { hdr_scale: resolved.hdr_scale, ..Default::default() };
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let settings = SpectralRenderSettings::new(&resolved, &render_config, noise_seed, false);
        let frame_interval = (scene.step_count()
            / render::constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES as usize)
            .max(1);
        let histogram = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::pass_1_build_histogram_spectral(scene, frame_interval, settings)
            }
            PipelineRenderMode::SerialReference => {
                render::pass_1_build_histogram_spectral_serial_reference(
                    scene,
                    frame_interval,
                    settings,
                )
            }
        };
        let analysis = render::histogram::analyze_tonemapping(
            histogram.data(),
            resolved.clip_black,
            resolved.clip_white,
        );
        let levels = render::ChannelLevels::with_tone_mapping(
            analysis.black_r,
            analysis.white_r,
            analysis.black_g,
            analysis.white_g,
            analysis.black_b,
            analysis.white_b,
            render::ToneMappingControls {
                exposure_scale: analysis.exposure_scale,
                paper_white: render::constants::DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: render::constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        );

        let image = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::render_single_frame_spectral(scene, &levels, settings)
            }
            PipelineRenderMode::SerialReference => {
                render::render_single_frame_spectral_serial_reference(scene, &levels, settings)
            }
        }
        .expect("render should succeed");

        image.into_raw()
    }

    #[test]
    fn test_end_to_end_pipeline_determinism() {
        for seed in [[0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
            let pixels_a = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            let pixels_b = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            assert_pixel_buffers_eq(&pixels_a, &pixels_b, &format!("default_parallel/{seed:02X?}"));
        }
    }

    #[test]
    fn test_end_to_end_pipeline_parallel_matches_serial_reference() {
        for seed in [[0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
            let parallel_pixels = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            let serial_pixels = run_full_pipeline(&seed, PipelineRenderMode::SerialReference);
            assert_pixel_buffers_eq(
                &parallel_pixels,
                &serial_pixels,
                &format!("parallel_vs_serial_reference/{seed:02X?}"),
            );
        }
    }
}
