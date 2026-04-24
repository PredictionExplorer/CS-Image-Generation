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
    VideoEncodingOptions, constants, generate_body_color_sequences,
    pass_1_build_histogram_spectral, pass_2_write_frames_spectral, save_image_as_png_16bit,
    video::{VideoEncoder, create_video_from_frames_singlepass_with_encoder},
};
use crate::sim::{self, Body, BordaWeights, Sha3RandomByteStream, TrajectoryResult};
use image::{ImageBuffer, Rgb};
use nalgebra::Vector3;
use std::fs;
use tracing::{info, warn};

/// Museum-quality enhancement flags (all default to true / enabled).
#[derive(Clone, Debug)]
pub struct Enhancements {
    /// Enable chroma boosting for richer color saturation.
    pub chroma_boost: bool,
    /// Enable perceptual saturation boost.
    pub sat_boost: bool,
    /// Enable ACES-inspired tone-mapping tweak.
    pub aces_tweak: bool,
    /// Enable per-body alpha variation for visual depth.
    pub alpha_variation: bool,
    /// Enable aspect-ratio correction for non-square outputs.
    pub aspect_correction: bool,
    /// Enable spectral dispersion boost.
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
    /// Number of simulation time-steps.
    pub num_steps_sim: usize,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Black-point clipping percentile.
    pub clip_black: f64,
    /// White-point clipping percentile.
    pub clip_white: f64,
    /// Alpha denominator controlling base trail opacity.
    pub alpha_denom: usize,
    /// Alpha compression factor.
    pub alpha_compress: f64,
    /// Escape threshold for orbit rejection.
    pub escape_threshold: f64,
    /// Drift mode identifier (e.g. `"elliptical"`, `"none"`).
    pub drift_mode: String,
    /// Bloom post-processing mode.
    pub bloom_mode: String,
    /// Difference-of-Gaussians edge-enhancement strength.
    pub dog_strength: f64,
    /// Optional sigma for the `DoG` narrow Gaussian.
    pub dog_sigma: Option<f64>,
    /// Ratio between the two `DoG` Gaussian widths.
    pub dog_ratio: f64,
    /// HDR tone-mapping mode.
    pub hdr_mode: String,
    /// HDR intensity scale factor.
    pub hdr_scale: f64,
    /// Perceptual blur mode identifier.
    pub perceptual_blur: String,
    /// Optional explicit radius for perceptual blur.
    pub perceptual_blur_radius: Option<usize>,
    /// Perceptual blur strength multiplier.
    pub perceptual_blur_strength: f64,
    /// Perceptual gamut-mapping mode.
    pub perceptual_gamut_mode: String,
    /// Minimum body mass for simulation.
    pub min_mass: f64,
    /// Maximum body mass for simulation.
    pub max_mass: f64,
    /// Initial location spread parameter.
    pub location: f64,
    /// Initial velocity spread parameter.
    pub velocity: f64,
    /// Weight for chaos metric in Borda scoring.
    pub chaos_weight: f64,
    /// Weight for equilibrium metric in Borda scoring.
    pub equil_weight: f64,
    /// Weight for curvature-entropy metric in Borda scoring.
    pub curvature_weight: f64,
    /// Weight for permutation-entropy metric in Borda scoring.
    pub permutation_weight: f64,
    /// Whether any Borda weight was randomized (rather than CLI-provided).
    pub weights_randomized: bool,
}

/// Initialize per-seed output directory structure:
///   output/{seed}/
///   output/{seed}/spectral/
///
/// Rejects output names containing path separators or `..` to prevent directory traversal.
///
/// # Errors
///
/// Returns an error when the output name is unsafe or the output directories
/// cannot be created.
pub fn setup_seed_directory(seed: &str) -> Result<String> {
    validate_output_name(seed)?;

    let seed_dir = format!("output/{seed}");
    let spectral_dir = format!("{seed_dir}/spectral");

    fs::create_dir_all(&seed_dir).map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: seed_dir.clone(),
        error: e,
    })?;

    fs::create_dir_all(&spectral_dir).map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: spectral_dir,
        error: e,
    })?;

    Ok(seed_dir)
}

/// Validate an output directory name without creating it.
///
/// # Errors
///
/// Returns an error when the name is empty, all whitespace, contains path
/// separators, contains `..`, or includes control characters.
pub fn validate_output_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(ConfigError::InvalidOutputName {
            name: name.to_string(),
            reason: "must not be empty".to_string(),
        }
        .into());
    }

    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err(ConfigError::InvalidOutputName {
            name: name.to_string(),
            reason: "must not contain path separators or '..'".to_string(),
        }
        .into());
    }

    if name.chars().any(char::is_control) {
        return Err(ConfigError::InvalidOutputName {
            name: name.to_string(),
            reason: "must not contain control characters".to_string(),
        }
        .into());
    }

    Ok(())
}

/// Parse and validate hex seed.
///
/// # Errors
///
/// Returns an error when `seed` is not valid even-length hexadecimal.
pub fn parse_seed(seed: &str) -> Result<Vec<u8>> {
    let hex_seed = seed.strip_prefix("0x").or_else(|| seed.strip_prefix("0X")).unwrap_or(seed);

    if hex_seed.is_empty() {
        return Err(ConfigError::EmptySeed { seed: seed.to_string() }.into());
    }

    hex::decode(hex_seed)
        .map_err(|e| ConfigError::InvalidSeed { seed: seed.to_string(), error: e }.into())
}

/// Run Borda selection to find the best orbit.
///
/// `weights` controls how the four metric rank points are combined into the
/// final weighted score — see [`BordaWeights`] for semantics.
///
/// # Errors
///
/// Returns an error when no valid orbit can be selected.
pub fn run_borda_selection(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    num_steps_sim: usize,
    weights: BordaWeights,
    escape_threshold: f64,
) -> Result<(Vec<Body>, TrajectoryResult)> {
    info!("STAGE 1/7: Borda search over {} random orbits...", num_sims);

    sim::select_best_trajectory(rng, num_sims, num_steps_sim, weights, escape_threshold)
}

/// Re-run the best orbit to get full trajectory
pub fn simulate_best_orbit(best_bodies: Vec<Body>, num_steps_sim: usize) -> Vec<Vec<Vector3<f64>>> {
    info!("STAGE 2/7: Re-running best orbit for {} steps...", num_steps_sim);
    let sim_result = sim::get_positions(best_bodies, num_steps_sim);
    info!("   => Done.");
    sim_result.positions
}

/// Apply drift transformation to positions.
///
/// # Errors
///
/// Returns an error when the drift configuration or drift mode is invalid.
pub fn apply_drift_transformation(
    positions: &mut [Vec<Vector3<f64>>],
    drift_mode: &str,
    drift_scale: Option<f64>,
    drift_arc_fraction: Option<f64>,
    drift_orbit_eccentricity: Option<f64>,
    rng: &mut Sha3RandomByteStream,
) -> Result<Option<ResolvedDriftConfig>> {
    info!("STAGE 2.5/7: Resolving drift configuration...");

    let resolved =
        resolve_drift_config(drift_scale, drift_arc_fraction, drift_orbit_eccentricity, rng)?;

    info!("Applying {} drift...", drift_mode);
    let num_steps = positions[0].len();
    let drift_params = resolved.to_drift_parameters();

    if crate::utils::is_zero(drift_params.arc_fraction)
        && drift_mode.to_lowercase().starts_with("ell")
    {
        warn!("Elliptical drift requested with zero arc fraction; skipping motion");
    }

    let mut drift_transform = parse_drift_mode(drift_mode, rng, drift_params, num_steps)?;
    drift_transform.apply(positions, constants::DEFAULT_DT);

    info!("   => Drift applied successfully");
    Ok(Some(resolved))
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

/// Build histogram and determine color levels.
///
/// # Errors
///
/// Returns an error if the spectral histogram render pass fails.
pub fn build_histogram_and_levels(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<render::OklabColor>],
    body_alphas: &[f64],
    resolved_config: &render::randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    aspect_correction: bool,
) -> Result<ChannelLevels> {
    info!("STAGE 5/7: PASS 1 => building global histogram...");

    let target_frames = constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES;
    let frame_interval = (positions[0].len() / target_frames as usize).max(1);

    let histogram = pass_1_build_histogram_spectral(
        SpectralScene::new(positions, colors, body_alphas),
        frame_interval,
        SpectralRenderSettings::new(resolved_config, render_config, aspect_correction),
    )?;

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

/// Render full video, returning the fully accumulated SPD buffer for spectral outputs.
///
/// # Errors
///
/// Returns an error if frame rendering, image encoding, or video encoding fails.
pub fn render_video(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_vid: &str,
    output_png: &str,
    fast_encode: bool,
    enable_temporal_smoothing: bool,
) -> Result<Vec<[f64; crate::spectrum::NUM_BINS]>> {
    render_video_with_encoder(
        scene,
        levels,
        settings,
        output_vid,
        output_png,
        fast_encode,
        enable_temporal_smoothing,
        &render::video::FfmpegVideoEncoder,
    )
}

pub(crate) fn render_video_with_encoder(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_vid: &str,
    output_png: &str,
    fast_encode: bool,
    enable_temporal_smoothing: bool,
    video_encoder: &dyn VideoEncoder,
) -> Result<Vec<[f64; crate::spectrum::NUM_BINS]>> {
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

    let mut accum_spd = Vec::new();

    let mut write_frames = |out: &mut dyn std::io::Write| {
        pass_2_write_frames_spectral(
            render::Pass2Params {
                scene,
                frame_interval,
                levels,
                settings,
                last_frame_out: &mut last_frame_png,
                enable_temporal_smoothing,
                accum_spd: &mut accum_spd,
            },
            |buf_8bit| {
                out.write_all(buf_8bit).map_err(render::error::RenderError::VideoEncoding)?;
                Ok(())
            },
        )?;
        Ok(())
    };

    create_video_from_frames_singlepass_with_encoder(
        settings.resolved_config.width,
        settings.resolved_config.height,
        frame_rate,
        &mut write_frames,
        output_vid,
        &video_options,
        video_encoder,
    )?;

    // Save final frame
    if let Some(last_frame) = last_frame_png {
        info!("Attempting to save 16-bit PNG to: {}", output_png);
        save_image_as_png_16bit(&last_frame, output_png)?;
    } else {
        warn!("Warning: No final frame was generated to save as PNG.");
    }

    Ok(accum_spd)
}

/// Generate the spectral gallery: 64 per-bin 16-bit PNGs in `spectral_dir`.
///
/// # Errors
///
/// Returns an error if the gallery directory cannot be created or a bin PNG
/// cannot be encoded.
pub fn generate_spectral_gallery(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    spectral_dir: &str,
) -> Result<()> {
    Ok(render::spectral_output::generate_spectral_gallery(accum_spd, width, height, spectral_dir)?)
}

/// Generate the spectral sweep video (violet-to-red cycle) at `output_path`.
///
/// # Errors
///
/// Returns an error if sweep frame generation or video encoding fails.
pub fn generate_spectral_sweep_video(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> Result<()> {
    generate_spectral_sweep_video_with_encoder(
        accum_spd,
        width,
        height,
        output_path,
        fast_encode,
        &render::video::FfmpegVideoEncoder,
    )
}

pub(crate) fn generate_spectral_sweep_video_with_encoder(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
    video_encoder: &dyn VideoEncoder,
) -> Result<()> {
    Ok(render::spectral_output::generate_spectral_sweep_video_with_encoder(
        accum_spd,
        width,
        height,
        output_path,
        fast_encode,
        video_encoder,
    )?)
}

/// Log generation parameters for reproducibility.
#[must_use]
pub fn build_generation_record(
    config: &GenerationLogConfig,
    file_name: &str,
    seed: &str,
    drift_config: Option<&ResolvedDriftConfig>,
    num_sims: usize,
    best_info: &TrajectoryResult,
    randomization_log: Option<&render::effect_randomizer::RandomizationLog>,
) -> GenerationRecord {
    let mut record = GenerationRecord::new(file_name.to_string(), format!("0x{seed}"));

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
        curvature_weight: config.curvature_weight,
        permutation_weight: config.permutation_weight,
        escape_threshold: config.escape_threshold,
        weights_randomized: config.weights_randomized,
    };

    record.orbit_info = OrbitInfo {
        selected_index: best_info.selected_index,
        weighted_score: best_info.total_score_weighted,
        total_candidates: num_sims,
        discarded_count: best_info.discarded_count,
        chaos: best_info.chaos,
        equilateralness: best_info.equilateralness,
        curvature_entropy: best_info.curvature_entropy,
        permutation_entropy: best_info.permutation_entropy,
    };

    record.randomization_log = randomization_log.cloned();

    record
}

/// Log generation parameters for reproducibility.
///
/// # Errors
///
/// Returns an error if the generation log cannot be written.
pub fn log_generation(
    config: &GenerationLogConfig,
    file_name: &str,
    seed: &str,
    drift_config: Option<&ResolvedDriftConfig>,
    num_sims: usize,
    best_info: &TrajectoryResult,
    randomization_log: Option<&render::effect_randomizer::RandomizationLog>,
) -> Result<()> {
    let logger = GenerationLogger::new();
    let record = build_generation_record(
        config,
        file_name,
        seed,
        drift_config,
        num_sims,
        best_info,
        randomization_log,
    );
    logger.log_generation(&record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_seed_valid() {
        let result = parse_seed("0x100033");
        assert!(result.is_ok());

        let bytes = result.expect("hex bytes should parse");
        assert_eq!(bytes, vec![0x10, 0x00, 0x33]);
    }

    #[test]
    fn test_parse_seed_no_prefix() {
        let result = parse_seed("100033");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_seed_uppercase_prefix() {
        let result = parse_seed("0X100033");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_seed_empty_rejected() {
        let result = parse_seed("0x");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_seed_invalid() {
        let result = parse_seed("0xZZZ");
        assert!(result.is_err());
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

    fn sample_generation_log_config() -> GenerationLogConfig {
        GenerationLogConfig {
            num_steps_sim: 1234,
            width: 320,
            height: 180,
            clip_black: 0.02,
            clip_white: 0.98,
            alpha_denom: 42_000,
            alpha_compress: 5.5,
            escape_threshold: -0.25,
            drift_mode: "elliptical".to_string(),
            bloom_mode: "dog".to_string(),
            dog_strength: 0.31,
            dog_sigma: Some(1.25),
            dog_ratio: 2.7,
            hdr_mode: "auto".to_string(),
            hdr_scale: 0.12,
            perceptual_blur: "on".to_string(),
            perceptual_blur_radius: Some(3),
            perceptual_blur_strength: 0.44,
            perceptual_gamut_mode: "preserve-hue".to_string(),
            min_mass: 100.0,
            max_mass: 300.0,
            location: 300.0,
            velocity: 1.0,
            chaos_weight: 1.0,
            equil_weight: 2.0,
            curvature_weight: 3.0,
            permutation_weight: 4.0,
            weights_randomized: true,
        }
    }

    fn sample_trajectory_result() -> TrajectoryResult {
        TrajectoryResult {
            chaos: 0.11,
            equilateralness: 0.22,
            curvature_entropy: 0.33,
            permutation_entropy: 0.44,
            chaos_pts: 9,
            equil_pts: 8,
            curvature_pts: 7,
            permutation_pts: 6,
            total_score: 30,
            total_score_weighted: 12.5,
            selected_index: 5,
            discarded_count: 2,
        }
    }

    #[test]
    fn build_generation_record_maps_drift_and_simulation_fields() {
        let config = sample_generation_log_config();
        let drift = ResolvedDriftConfig::from_values(1.5, 0.25, 0.45);
        let record = build_generation_record(
            &config,
            "gallery",
            "cafe",
            Some(&drift),
            99,
            &sample_trajectory_result(),
            None,
        );

        assert_eq!(record.file_name, "gallery");
        assert_eq!(record.seed, "0xcafe");
        assert_eq!(record.render_config.width, 320);
        assert_eq!(record.render_config.dog_sigma, Some(1.25));
        assert!(record.drift_config.enabled);
        assert_eq!(record.drift_config.mode, "elliptical");
        assert_eq!(record.drift_config.scale, 1.5);
        assert_eq!(record.simulation_config.num_sims, 99);
        assert_eq!(record.simulation_config.num_steps_sim, 1234);
        assert_eq!(record.simulation_config.chaos_weight, 1.0);
        assert!(record.simulation_config.weights_randomized);
        assert_eq!(record.orbit_info.selected_index, 5);
        assert_eq!(record.orbit_info.weighted_score, 12.5);
        assert!(record.randomization_log.is_none());
    }

    #[test]
    fn build_generation_record_uses_disabled_drift_defaults_without_drift() {
        let config = sample_generation_log_config();
        let record = build_generation_record(
            &config,
            "still",
            "deadbeef",
            None,
            7,
            &sample_trajectory_result(),
            None,
        );

        assert!(!record.drift_config.enabled);
        assert_eq!(record.drift_config.mode, "none");
        assert_eq!(record.drift_config.scale, 0.0);
        assert!(!record.drift_config.randomized);
    }

    #[test]
    fn build_generation_record_clones_randomization_log() {
        let mut randomization_log = render::effect_randomizer::RandomizationLog::new();
        let mut record = render::effect_randomizer::RandomizationRecord::new("glow", true, true);
        record.add_float("strength", 0.5, true, (0.1, 0.9));
        randomization_log.add_record(record);

        let generation_record = build_generation_record(
            &sample_generation_log_config(),
            "randomized",
            "abcd",
            None,
            3,
            &sample_trajectory_result(),
            Some(&randomization_log),
        );

        let logged = generation_record
            .randomization_log
            .expect("record should include cloned randomization log");
        assert_eq!(logged.effects.len(), 1);
        assert_eq!(logged.effects[0].effect_name, "glow");
        assert_eq!(logged.effects[0].parameters[0].name, "strength");
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
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .expect("expected differing pixel position");
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
            ..Default::default()
        };
        let (resolved, _) = config.resolve(&mut rng, width, height);

        let test_weights = BordaWeights::new(0.75, 11.0, 2.5, 1.5);
        let (best_bodies, _) =
            crate::sim::select_best_trajectory(&mut rng, num_sims, num_steps, test_weights, -0.3)
                .expect("Borda search should find at least one valid orbit");

        let mut positions = simulate_best_orbit(best_bodies, num_steps);

        apply_drift_transformation(&mut positions, "elliptical", None, None, None, &mut rng)
            .expect("drift config resolution should succeed with all-None args");

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
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let frame_interval = (scene.step_count()
            / render::constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES as usize)
            .max(1);
        let histogram = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::pass_1_build_histogram_spectral(scene, frame_interval, settings)
                    .expect("parallel histogram pass should succeed")
            }
            PipelineRenderMode::SerialReference => {
                render::pass_1_build_histogram_spectral_serial_reference(
                    scene,
                    frame_interval,
                    settings,
                )
                .expect("serial histogram pass should succeed")
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
    fn test_setup_seed_directory_returns_correct_path() {
        let result = setup_seed_directory("test_seed_42");
        assert!(result.is_ok());
        let seed_dir = result.expect("seed directory setup should succeed");
        assert_eq!(seed_dir, "output/test_seed_42");
        assert!(std::path::Path::new("output/test_seed_42").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/spectral").is_dir());
        let _ = fs::remove_dir_all("output/test_seed_42");
    }

    #[test]
    fn test_setup_seed_directory_idempotent() {
        let r1 = setup_seed_directory("seed_idem");
        let r2 = setup_seed_directory("seed_idem");
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        let _ = fs::remove_dir_all("output/seed_idem");
    }

    #[test]
    fn test_validate_output_name_rejects_traversal_and_empty_names() {
        assert!(validate_output_name("../escape").is_err());
        assert!(validate_output_name("nested/path").is_err());
        assert!(validate_output_name("   ").is_err());
        assert!(validate_output_name("safe-name_01").is_ok());
    }

    proptest::proptest! {
        #[test]
        fn proptest_parse_seed_never_panics(input in "\\PC*") {
            let _ = parse_seed(&input);
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
