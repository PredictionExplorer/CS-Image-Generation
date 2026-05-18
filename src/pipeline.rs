//! Top-level generation pipeline.
//!
//! This module owns the seed-to-output orchestration used by the binary.  Keeping
//! the workflow here makes the CLI a thin argument parser and gives tests a
//! stable boundary for exercising generation without spawning external tools.

use crate::app::{self, Enhancements, GenerationLogConfig};
use crate::drift_config::ResolvedDriftConfig;
use crate::error::{self, Result};
use crate::render::{
    self, RenderConfig, SpectralRenderSettings, SpectralScene,
    effect_randomizer::RandomizationLog,
    randomizable_config::{RandomizableEffectConfig, ResolvedEffectConfig},
    video::{FfmpegVideoEncoder, VideoEncoder},
};
use crate::sim::{BordaWeights, Sha3RandomByteStream, TrajectoryResult};
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;
use rayon::ThreadPoolBuilder;
use std::path::Path;
use std::sync::OnceLock;
use tracing::{info, warn};

/// Default output directory stem when the CLI does not provide one.
pub const DEFAULT_OUTPUT_NAME: &str = "output";
/// Default number of candidate simulations.
pub const DEFAULT_NUM_SIMS: usize = 100_000;
/// Default number of integration steps per simulation.
pub const DEFAULT_NUM_STEPS: usize = 1_000_000;
/// Maximum accepted candidate simulation count.
pub const MAX_NUM_SIMS: usize = 10_000_000;
/// Maximum accepted integration step count.
pub const MAX_NUM_STEPS: usize = 100_000_000;
/// Default output resolution.
pub const DEFAULT_RESOLUTION: &str = "1920x1080";
/// Default tracing filter used by the binary.
pub const DEFAULT_LOG_LEVEL: &str = "info";
/// Initial position spread for sampled bodies.
pub const DEFAULT_LOCATION: f64 = 300.0;
/// Initial velocity spread for sampled bodies.
pub const DEFAULT_VELOCITY: f64 = 1.0;
/// Minimum sampled body mass.
pub const DEFAULT_MIN_MASS: f64 = 100.0;
/// Maximum sampled body mass.
pub const DEFAULT_MAX_MASS: f64 = 300.0;
/// Base trail opacity denominator.
pub const DEFAULT_ALPHA_DENOM: usize = 15_000_000;
/// Alpha compression curve strength.
pub const DEFAULT_ALPHA_COMPRESS: f64 = 6.0;
/// Orbit rejection threshold for escaping trajectories.
pub const DEFAULT_ESCAPE_THRESHOLD: f64 = -0.3;
/// Logged HDR mode string for current automatic tone-mapping.
pub const DEFAULT_HDR_MODE: &str = "auto";
/// Logged gamut mode string for perceptual blur.
pub const DEFAULT_PERCEPTUAL_GAMUT_MODE: &str = "preserve-hue";

static THREAD_POOL_INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Drift mode selected for a generation request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GenerationDriftMode {
    /// Do not apply camera drift.
    None,
    /// Apply linear drift.
    Linear,
    /// Apply Brownian drift.
    Brownian,
    /// Apply elliptical drift.
    #[default]
    Elliptical,
}

impl GenerationDriftMode {
    /// Return the canonical lowercase string representation.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Brownian => "brownian",
            Self::Elliptical => "elliptical",
        }
    }
}

/// Optional Borda weights supplied by a caller.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BordaWeightOptions {
    /// Optional chaos metric weight.
    pub chaos: Option<f64>,
    /// Optional equilateralness metric weight.
    pub equil: Option<f64>,
    /// Optional curvature-entropy metric weight.
    pub curvature: Option<f64>,
    /// Optional permutation-entropy metric weight.
    pub permutation: Option<f64>,
}

impl BordaWeightOptions {
    /// Validate explicit weights supplied by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error when any explicit weight is not finite or is less than
    /// or equal to zero.
    pub fn validate(self) -> Result<()> {
        for (name, value) in [
            ("chaos_weight", self.chaos),
            ("equil_weight", self.equil),
            ("curvature_weight", self.curvature),
            ("permutation_weight", self.permutation),
        ] {
            if let Some(weight) = value
                && (!weight.is_finite() || weight <= 0.0)
            {
                return Err(error::ConfigError::InvalidParameter {
                    parameter: name.to_string(),
                    reason: "must be finite and greater than zero".to_string(),
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Resolved Borda weights used for the current generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedBordaWeights {
    /// Chaos metric rank-point weight.
    pub chaos_weight: f64,
    /// Equilateralness metric rank-point weight.
    pub equil_weight: f64,
    /// Curvature-entropy metric rank-point weight.
    pub curvature_weight: f64,
    /// Permutation-entropy metric rank-point weight.
    pub permutation_weight: f64,
    /// Whether at least one weight was randomized.
    pub was_randomized: bool,
}

impl ResolvedBordaWeights {
    fn to_borda_weights(self) -> BordaWeights {
        BordaWeights::new(
            self.chaos_weight,
            self.equil_weight,
            self.curvature_weight,
            self.permutation_weight,
        )
    }
}

/// Complete generation request after CLI defaults have been resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    /// Hex seed string, with or without `0x` prefix.
    pub seed: String,
    /// User-facing output name; becomes `output/{output}`.
    pub output: String,
    /// Number of candidate simulations to evaluate.
    pub sims: usize,
    /// Number of integration steps per simulation.
    pub steps: usize,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Drift mode applied after the best orbit is re-simulated.
    pub drift_mode: GenerationDriftMode,
    /// Use fast video encoding options when true.
    pub fast_encode: bool,
    /// Optional Borda weights; missing values are randomized from curated ranges.
    pub borda_weights: BordaWeightOptions,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            seed: "0x100033".to_string(),
            output: DEFAULT_OUTPUT_NAME.to_string(),
            sims: DEFAULT_NUM_SIMS,
            steps: DEFAULT_NUM_STEPS,
            width: 1920,
            height: 1080,
            drift_mode: GenerationDriftMode::Elliptical,
            fast_encode: false,
            borda_weights: BordaWeightOptions::default(),
        }
    }
}

impl GenerationRequest {
    /// Validate all caller-provided request fields before generation starts.
    ///
    /// # Errors
    ///
    /// Returns an error when seed, output name, simulation counts, dimensions,
    /// or explicit Borda weights are invalid.
    pub fn validate(&self) -> Result<()> {
        app::parse_seed(&self.seed)?;
        app::validate_output_name(&self.output)?;
        error::validation::validate_dimensions(self.width, self.height)?;

        if self.sims == 0 || self.sims > MAX_NUM_SIMS {
            return Err(error::ConfigError::InvalidParameter {
                parameter: "sims".to_string(),
                reason: format!("must be between 1 and {MAX_NUM_SIMS}"),
            }
            .into());
        }

        if self.steps == 0 || self.steps > MAX_NUM_STEPS {
            return Err(error::ConfigError::InvalidParameter {
                parameter: "steps".to_string(),
                reason: format!("must be between 1 and {MAX_NUM_STEPS}"),
            }
            .into());
        }

        self.borda_weights.validate()
    }
}

/// Paths produced by a generation run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationOutputs {
    /// Per-seed output directory.
    pub seed_dir: String,
    /// Final 16-bit PNG path.
    pub image_png: String,
    /// Main video path.
    pub video_mp4: String,
    /// Directory containing per-bin spectral PNGs.
    pub spectral_dir: String,
    /// Spectral sweep video path.
    pub spectral_sweep_mp4: String,
}

impl GenerationOutputs {
    fn for_seed_dir(seed_dir: impl AsRef<Path>) -> Self {
        let seed_path = seed_dir.as_ref();
        Self {
            seed_dir: app::path_to_string(seed_path),
            image_png: app::path_to_string(&seed_path.join(app::IMAGE_FILE_NAME)),
            video_mp4: app::path_to_string(&seed_path.join(app::VIDEO_FILE_NAME)),
            spectral_dir: app::path_to_string(&seed_path.join(app::SPECTRAL_DIR_NAME)),
            spectral_sweep_mp4: app::path_to_string(&seed_path.join(app::SPECTRAL_SWEEP_FILE_NAME)),
        }
    }

    /// Per-seed output directory as a typed path.
    #[must_use]
    pub fn seed_dir_path(&self) -> &Path {
        Path::new(&self.seed_dir)
    }

    /// Final 16-bit PNG output path.
    #[must_use]
    pub fn image_png_path(&self) -> &Path {
        Path::new(&self.image_png)
    }

    /// Main video output path.
    #[must_use]
    pub fn video_mp4_path(&self) -> &Path {
        Path::new(&self.video_mp4)
    }

    /// Directory containing per-bin spectral PNGs.
    #[must_use]
    pub fn spectral_dir_path(&self) -> &Path {
        Path::new(&self.spectral_dir)
    }

    /// Spectral sweep video output path.
    #[must_use]
    pub fn spectral_sweep_mp4_path(&self) -> &Path {
        Path::new(&self.spectral_sweep_mp4)
    }
}

/// Human-scale summary of a completed generation.
#[derive(Clone, Debug)]
pub struct GenerationSummary {
    /// Output paths created by the run.
    pub outputs: GenerationOutputs,
    /// Final selected trajectory score.
    pub best_score: f64,
    /// Index of the chosen orbit in the ranking list.
    pub selected_index: usize,
    /// Resolved Borda weights used by the selector.
    pub borda_weights: ResolvedBordaWeights,
    /// Drift configuration when drift was enabled.
    pub drift_config: Option<ResolvedDriftConfig>,
}

/// Run a full generation using the production `FFmpeg` encoder.
///
/// # Errors
///
/// Returns an error if configuration validation, simulation, rendering, output
/// writing, or video encoding fails.
pub fn run_generation(request: &GenerationRequest) -> Result<GenerationSummary> {
    run_generation_with_video_encoder(request, &FfmpegVideoEncoder)
}

pub(crate) fn run_generation_with_video_encoder(
    request: &GenerationRequest,
    video_encoder: &dyn VideoEncoder,
) -> Result<GenerationSummary> {
    request.validate()?;
    install_global_thread_pool()?;

    let enhancements = Enhancements::default();

    let seed_bytes = app::parse_seed(&request.seed)?;
    let hex_seed = seed_hex(&request.seed);
    let seed_dir = app::setup_output_directory_path(&request.output)?;
    let outputs = GenerationOutputs::for_seed_dir(&seed_dir);
    let mut rng = Sha3RandomByteStream::new(
        &seed_bytes,
        DEFAULT_MIN_MASS,
        DEFAULT_MAX_MASS,
        DEFAULT_LOCATION,
        DEFAULT_VELOCITY,
    );

    let (resolved_effect_config, randomization_log) =
        resolve_randomized_effects(&mut rng, request.width, request.height);
    let borda_weights = resolve_borda_weights(request.borda_weights, &mut rng);

    let (best_bodies, best_info) = app::run_borda_selection(
        &mut rng,
        request.sims,
        request.steps,
        borda_weights.to_borda_weights(),
        DEFAULT_ESCAPE_THRESHOLD,
    )?;

    let mut positions = app::simulate_best_orbit(best_bodies, request.steps);
    let drift_config = apply_optional_drift(request, &mut positions, &mut rng)?;
    let (colors, body_alphas) =
        app::generate_colors(&mut rng, request.steps, DEFAULT_ALPHA_DENOM, &enhancements);

    let render_config = RenderConfig {
        hdr_scale: resolved_effect_config.hdr_scale,
        bloom_mode: render::BloomMode::Dog,
        sat_boost: enhancements.sat_boost,
        aces_tweak: enhancements.aces_tweak,
        dispersion_boost: enhancements.dispersion_boost,
    };
    render_outputs(
        RenderOutputInputs {
            request,
            outputs: &outputs,
            positions: &positions,
            colors: &colors,
            body_alphas: &body_alphas,
            resolved_effect_config: &resolved_effect_config,
            render_config: &render_config,
            enhancements: &enhancements,
        },
        video_encoder,
    )?;

    info!(
        "Done! Best orbit => Weighted Borda = {:.3}\nHave a nice day!",
        best_info.total_score_weighted
    );

    log_generation_result(
        request,
        hex_seed,
        drift_config.as_ref(),
        &best_info,
        &randomization_log,
        &resolved_effect_config,
        &render_config,
        &borda_weights,
    );

    Ok(GenerationSummary {
        outputs,
        best_score: best_info.total_score_weighted,
        selected_index: best_info.selected_index,
        borda_weights,
        drift_config,
    })
}

fn install_global_thread_pool() -> Result<()> {
    let init = THREAD_POOL_INIT.get_or_init(|| {
        ThreadPoolBuilder::new()
            .stack_size(render::constants::THREAD_STACK_SIZE)
            .build_global()
            .map_err(|e| e.to_string())
    });

    match init {
        Ok(()) => Ok(()),
        Err(message) if message.contains("already") && message.contains("initialized") => Ok(()),
        Err(message) => Err(std::io::Error::other(message.clone()).into()),
    }
}

fn seed_hex(seed: &str) -> &str {
    seed.strip_prefix("0x").or_else(|| seed.strip_prefix("0X")).unwrap_or(seed)
}

/// Log-uniform sample of a single Borda weight.
fn resolve_single_weight(user_value: Option<f64>, rng: &mut Sha3RandomByteStream) -> (f64, bool) {
    use render::parameter_descriptors::EQUIL_CHAOS_RATIO;
    if let Some(v) = user_value {
        return (v, false);
    }
    let log_min = EQUIL_CHAOS_RATIO.min.ln();
    let log_max = EQUIL_CHAOS_RATIO.max.ln();
    let value = (log_min + rng.next_f64() * (log_max - log_min)).exp();
    (value, true)
}

/// Resolve all four Borda weights for this run.
pub fn resolve_borda_weights(
    options: BordaWeightOptions,
    rng: &mut Sha3RandomByteStream,
) -> ResolvedBordaWeights {
    let (chaos_weight, chaos_rand) = resolve_single_weight(options.chaos, rng);
    let (equil_weight, equil_rand) = resolve_single_weight(options.equil, rng);
    let (curvature_weight, curve_rand) = resolve_single_weight(options.curvature, rng);
    let (permutation_weight, perm_rand) = resolve_single_weight(options.permutation, rng);
    let was_randomized = chaos_rand || equil_rand || curve_rand || perm_rand;

    let dominant = [
        ("chaos", chaos_weight),
        ("equil", equil_weight),
        ("curvature", curvature_weight),
        ("permutation", permutation_weight),
    ]
    .into_iter()
    .max_by(|a, b| a.1.total_cmp(&b.1))
    .map_or("chaos", |(name, _)| name);

    info!(
        "Borda weights: chaos={chaos_weight:.3}, equil={equil_weight:.3}, \
         curvature={curvature_weight:.3}, permutation={permutation_weight:.3} \
         (dominant: {dominant}){randomized}",
        randomized = if was_randomized { " [randomized]" } else { " [explicit]" },
    );

    ResolvedBordaWeights {
        chaos_weight,
        equil_weight,
        curvature_weight,
        permutation_weight,
        was_randomized,
    }
}

fn resolve_randomized_effects(
    rng: &mut Sha3RandomByteStream,
    width: u32,
    height: u32,
) -> (ResolvedEffectConfig, RandomizationLog) {
    info!("Resolving effect configuration...");
    let randomizable_config = RandomizableEffectConfig::default();
    let (resolved_effect_config, randomization_log) =
        randomizable_config.resolve(rng, width, height);

    let num_randomized = randomization_log
        .effects
        .iter()
        .map(|effect| effect.parameters.iter().filter(|param| param.was_randomized).count())
        .sum::<usize>();
    let num_parameters =
        randomization_log.effects.iter().map(|effect| effect.parameters.len()).sum::<usize>();

    info!(
        "   => Resolved {} effects ({} parameters randomized, {} explicit)",
        randomization_log.effects.len(),
        num_randomized,
        num_parameters - num_randomized
    );

    (resolved_effect_config, randomization_log)
}

fn apply_optional_drift(
    request: &GenerationRequest,
    positions: &mut [Vec<Vector3<f64>>],
    rng: &mut Sha3RandomByteStream,
) -> Result<Option<ResolvedDriftConfig>> {
    if request.drift_mode == GenerationDriftMode::None {
        info!("STAGE 2.5/7: Drift disabled");
        return Ok(None);
    }

    app::apply_drift_transformation(positions, request.drift_mode.as_str(), None, None, None, rng)
}

#[derive(Clone, Copy)]
struct RenderOutputInputs<'a> {
    request: &'a GenerationRequest,
    outputs: &'a GenerationOutputs,
    positions: &'a [Vec<Vector3<f64>>],
    colors: &'a [Vec<render::OklabColor>],
    body_alphas: &'a [f64],
    resolved_effect_config: &'a ResolvedEffectConfig,
    render_config: &'a RenderConfig,
    enhancements: &'a Enhancements,
}

fn render_outputs(
    inputs: RenderOutputInputs<'_>,
    video_encoder: &dyn VideoEncoder,
) -> Result<Vec<[f64; NUM_BINS]>> {
    let RenderOutputInputs {
        request,
        outputs,
        positions,
        colors,
        body_alphas,
        resolved_effect_config,
        render_config,
        enhancements,
    } = inputs;
    info!("   => Using OKLab color space for accumulation");
    info!("STAGE 4/7: Determining bounding box...");
    let render_ctx = render::context::RenderContext::try_new(
        request.width,
        request.height,
        positions,
        enhancements.aspect_correction,
    )?;
    let bbox = render_ctx.bounds();
    info!(
        "   => X: [{:.3}, {:.3}], Y: [{:.3}, {:.3}]",
        bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
    );

    let levels = app::build_histogram_and_levels(
        positions,
        colors,
        body_alphas,
        resolved_effect_config,
        render_config,
        enhancements.aspect_correction,
    )?;

    let accum_spd = app::render_video_with_encoder(
        SpectralScene::new(positions, colors, body_alphas),
        &levels,
        SpectralRenderSettings::new(
            resolved_effect_config,
            render_config,
            enhancements.aspect_correction,
        ),
        outputs.video_mp4_path(),
        outputs.image_png_path(),
        request.fast_encode,
        true,
        video_encoder,
    )?;

    app::generate_spectral_gallery(
        &accum_spd,
        request.width,
        request.height,
        outputs.spectral_dir_path(),
    )?;
    app::generate_spectral_sweep_video_with_encoder(
        &accum_spd,
        request.width,
        request.height,
        outputs.spectral_sweep_mp4_path(),
        request.fast_encode,
        video_encoder,
    )?;

    Ok(accum_spd)
}

fn build_generation_log_config(
    request: &GenerationRequest,
    resolved: &ResolvedEffectConfig,
    render_config: &RenderConfig,
    borda_weights: &ResolvedBordaWeights,
) -> GenerationLogConfig {
    let min_dim = resolved.width.min(resolved.height);
    let bloom_mode = if resolved.enable_bloom {
        render_config.bloom_mode.as_str()
    } else {
        render::BloomMode::None.as_str()
    };

    GenerationLogConfig {
        num_steps_sim: request.steps,
        width: resolved.width,
        height: resolved.height,
        clip_black: resolved.clip_black,
        clip_white: resolved.clip_white,
        alpha_denom: DEFAULT_ALPHA_DENOM,
        alpha_compress: DEFAULT_ALPHA_COMPRESS,
        escape_threshold: DEFAULT_ESCAPE_THRESHOLD,
        drift_mode: request.drift_mode.as_str().to_string(),
        bloom_mode: bloom_mode.to_string(),
        dog_strength: resolved.dog_strength,
        dog_sigma: Some(resolved.dog_sigma_scale * f64::from(min_dim)),
        dog_ratio: resolved.dog_ratio,
        hdr_mode: DEFAULT_HDR_MODE.to_string(),
        hdr_scale: render_config.hdr_scale,
        sat_boost: render_config.sat_boost,
        aces_tweak: render_config.aces_tweak,
        dispersion_boost: render_config.dispersion_boost,
        perceptual_blur: if resolved.enable_perceptual_blur {
            "on".to_string()
        } else {
            "off".to_string()
        },
        perceptual_blur_radius: render::compute_softness_radius(resolved, render_config.bloom_mode),
        perceptual_blur_strength: resolved.perceptual_blur_strength,
        perceptual_gamut_mode: DEFAULT_PERCEPTUAL_GAMUT_MODE.to_string(),
        min_mass: DEFAULT_MIN_MASS,
        max_mass: DEFAULT_MAX_MASS,
        location: DEFAULT_LOCATION,
        velocity: DEFAULT_VELOCITY,
        chaos_weight: borda_weights.chaos_weight,
        equil_weight: borda_weights.equil_weight,
        curvature_weight: borda_weights.curvature_weight,
        permutation_weight: borda_weights.permutation_weight,
        weights_randomized: borda_weights.was_randomized,
    }
}

fn log_generation_result(
    request: &GenerationRequest,
    hex_seed: &str,
    drift_config: Option<&ResolvedDriftConfig>,
    best_info: &TrajectoryResult,
    randomization_log: &RandomizationLog,
    resolved_effect_config: &ResolvedEffectConfig,
    render_config: &RenderConfig,
    borda_weights: &ResolvedBordaWeights,
) {
    let generation_log_config =
        build_generation_log_config(request, resolved_effect_config, render_config, borda_weights);
    if let Err(e) = app::log_generation(
        &generation_log_config,
        &request.output,
        hex_seed,
        drift_config,
        request.sims,
        best_info,
        Some(randomization_log),
    ) {
        warn!("Generation logging failed (non-fatal): {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::VideoEncodingOptions;
    use crate::render::video::VideoEncoder;
    use std::cell::RefCell;
    use std::error::Error;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static CWD_LOCK: Mutex<()> = Mutex::new(());

    struct CurrentDirGuard {
        original: PathBuf,
    }

    impl CurrentDirGuard {
        fn enter(path: &Path) -> Self {
            let original = std::env::current_dir().expect("current dir should be readable");
            std::env::set_current_dir(path).expect("test cwd should be set");
            Self { original }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    fn fresh_rng() -> Sha3RandomByteStream {
        Sha3RandomByteStream::new(&[0x42; 32], 100.0, 300.0, 300.0, 1.0)
    }

    fn assert_in_range(label: &str, value: f64) {
        assert!((0.2..=125.0).contains(&value), "{label} weight {value} outside [0.2, 125.0]");
    }

    #[test]
    fn resolve_borda_weights_all_none_randomizes_in_range() {
        let mut rng = fresh_rng();
        let w = resolve_borda_weights(BordaWeightOptions::default(), &mut rng);
        assert!(w.was_randomized);
        assert_in_range("chaos", w.chaos_weight);
        assert_in_range("equil", w.equil_weight);
        assert_in_range("curvature", w.curvature_weight);
        assert_in_range("permutation", w.permutation_weight);
    }

    #[test]
    fn borda_weight_options_rejects_non_finite_or_non_positive_values() {
        for options in [
            BordaWeightOptions { chaos: Some(0.0), ..Default::default() },
            BordaWeightOptions { equil: Some(-1.0), ..Default::default() },
            BordaWeightOptions { curvature: Some(f64::NAN), ..Default::default() },
            BordaWeightOptions { permutation: Some(f64::INFINITY), ..Default::default() },
        ] {
            assert!(options.validate().is_err());
        }
    }

    #[test]
    fn generation_request_validate_rejects_invalid_boundary_values() {
        let invalid_requests = [
            GenerationRequest { seed: "0x".to_string(), ..Default::default() },
            GenerationRequest { output: "../escape".to_string(), ..Default::default() },
            GenerationRequest { sims: 0, ..Default::default() },
            GenerationRequest { sims: MAX_NUM_SIMS + 1, ..Default::default() },
            GenerationRequest { steps: 0, ..Default::default() },
            GenerationRequest { steps: MAX_NUM_STEPS + 1, ..Default::default() },
            GenerationRequest { width: 0, ..Default::default() },
            GenerationRequest {
                borda_weights: BordaWeightOptions { chaos: Some(f64::NAN), ..Default::default() },
                ..Default::default()
            },
        ];

        for request in invalid_requests {
            assert!(request.validate().is_err(), "request should be rejected: {request:?}");
        }
    }

    #[test]
    fn generation_request_validate_accepts_uppercase_seed_prefix() {
        let request = GenerationRequest { seed: "0X100033".to_string(), ..Default::default() };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn resolve_borda_weights_all_some_uses_explicit_values() {
        let mut rng = fresh_rng();
        let w = resolve_borda_weights(
            BordaWeightOptions {
                chaos: Some(1.0),
                equil: Some(10.0),
                curvature: Some(2.5),
                permutation: Some(0.3),
            },
            &mut rng,
        );
        assert!(!w.was_randomized);
        assert_eq!(w.chaos_weight, 1.0);
        assert_eq!(w.equil_weight, 10.0);
        assert_eq!(w.curvature_weight, 2.5);
        assert_eq!(w.permutation_weight, 0.3);
    }

    #[test]
    fn resolve_borda_weights_partial_some_mixes() {
        let mut rng = fresh_rng();
        let w = resolve_borda_weights(
            BordaWeightOptions {
                chaos: Some(1.0),
                equil: None,
                curvature: Some(2.5),
                permutation: None,
            },
            &mut rng,
        );
        assert!(w.was_randomized);
        assert_eq!(w.chaos_weight, 1.0);
        assert_eq!(w.curvature_weight, 2.5);
        assert_in_range("equil", w.equil_weight);
        assert_in_range("permutation", w.permutation_weight);
    }

    #[test]
    fn resolve_borda_weights_is_deterministic_under_seed() {
        let mut rng1 = fresh_rng();
        let mut rng2 = fresh_rng();
        let w1 = resolve_borda_weights(BordaWeightOptions::default(), &mut rng1);
        let w2 = resolve_borda_weights(BordaWeightOptions::default(), &mut rng2);
        assert_eq!(w1.chaos_weight.to_bits(), w2.chaos_weight.to_bits());
        assert_eq!(w1.equil_weight.to_bits(), w2.equil_weight.to_bits());
        assert_eq!(w1.curvature_weight.to_bits(), w2.curvature_weight.to_bits());
        assert_eq!(w1.permutation_weight.to_bits(), w2.permutation_weight.to_bits());
    }

    #[test]
    fn resolve_borda_weights_range_coverage_all_seeds() {
        for seed_byte in 0u8..=255 {
            let seed = [seed_byte; 32];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let w = resolve_borda_weights(BordaWeightOptions::default(), &mut rng);
            for (label, value) in [
                ("chaos", w.chaos_weight),
                ("equil", w.equil_weight),
                ("curvature", w.curvature_weight),
                ("permutation", w.permutation_weight),
            ] {
                assert!(
                    (0.2..=125.0).contains(&value),
                    "seed {seed_byte}: {label} = {value} outside [0.2, 125.0]",
                );
            }
        }
    }

    #[test]
    fn resolved_borda_weights_converts_to_sim_struct() {
        let resolved = ResolvedBordaWeights {
            chaos_weight: 1.0,
            equil_weight: 2.0,
            curvature_weight: 3.0,
            permutation_weight: 4.0,
            was_randomized: false,
        };
        let bw = resolved.to_borda_weights();
        assert_eq!(bw.chaos, 1.0);
        assert_eq!(bw.equil, 2.0);
        assert_eq!(bw.curvature, 3.0);
        assert_eq!(bw.permutation, 4.0);
    }

    #[test]
    fn generation_outputs_are_derived_from_seed_directory() {
        let seed_dir = Path::new("output").join("example");
        let outputs = GenerationOutputs::for_seed_dir(&seed_dir);

        assert_eq!(outputs.seed_dir, app::path_to_string(&seed_dir));
        assert_eq!(outputs.seed_dir_path(), seed_dir.as_path());
        assert_eq!(outputs.image_png_path(), seed_dir.join(app::IMAGE_FILE_NAME).as_path());
        assert_eq!(outputs.video_mp4_path(), seed_dir.join(app::VIDEO_FILE_NAME).as_path());
        assert_eq!(outputs.spectral_dir_path(), seed_dir.join(app::SPECTRAL_DIR_NAME).as_path());
        assert_eq!(
            outputs.spectral_sweep_mp4_path(),
            seed_dir.join(app::SPECTRAL_SWEEP_FILE_NAME).as_path()
        );
    }

    #[derive(Default)]
    struct FakeVideoEncoder {
        calls: RefCell<Vec<(u32, u32, u32, String, usize)>>,
    }

    impl VideoEncoder for FakeVideoEncoder {
        fn encode(
            &self,
            width: u32,
            height: u32,
            frame_rate: u32,
            frames_iter: &mut dyn FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
            output_file: &Path,
            _options: &VideoEncodingOptions,
        ) -> render::error::Result<()> {
            let mut sink = Vec::new();
            frames_iter(&mut sink).map_err(|e| {
                render::error::RenderError::VideoEncoding(std::io::Error::other(e.to_string()))
            })?;
            self.calls.borrow_mut().push((
                width,
                height,
                frame_rate,
                app::path_to_string(output_file),
                sink.len(),
            ));
            Ok(())
        }
    }

    fn tiny_generation_request(
        output: String,
        drift_mode: GenerationDriftMode,
    ) -> GenerationRequest {
        GenerationRequest {
            seed: "0xcafe".to_string(),
            output,
            sims: 6,
            steps: 64,
            width: 8,
            height: 6,
            drift_mode,
            fast_encode: true,
            borda_weights: BordaWeightOptions {
                chaos: Some(1.0),
                equil: Some(2.0),
                curvature: Some(3.0),
                permutation: Some(4.0),
            },
        }
    }

    fn png_channel_energy(path: &Path) -> u64 {
        let image = image::open(path).expect("PNG should decode").to_rgb16();
        image.as_raw().iter().map(|&channel| u64::from(channel)).sum()
    }

    #[test]
    fn pipeline_smoke_test_uses_video_encoder_seam() {
        let _lock = CWD_LOCK.lock().expect("cwd test lock should not be poisoned");
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let output_name = tmp
            .path()
            .file_name()
            .expect("tempdir should have a final component")
            .to_string_lossy()
            .to_string();
        let output_parent =
            tmp.path().parent().expect("tempdir should have a parent").to_path_buf();
        let _cwd = CurrentDirGuard::enter(&output_parent);

        let encoder = FakeVideoEncoder::default();
        let request = tiny_generation_request(output_name.clone(), GenerationDriftMode::None);

        let result = run_generation_with_video_encoder(&request, &encoder);
        let summary = result.expect("tiny generation should complete with fake encoder");

        assert!(summary.best_score.is_finite());
        assert_eq!(
            summary.outputs.seed_dir,
            app::path_to_string(&std::path::Path::new("output").join(output_name))
        );
        assert_eq!(encoder.calls.borrow().len(), 2, "main and spectral sweep videos are encoded");
        assert!(
            encoder.calls.borrow().iter().all(|(_, _, _, _, bytes)| *bytes > 0),
            "fake encoder should receive non-empty frame streams"
        );
    }

    #[test]
    fn pipeline_low_density_smoke_test_outputs_nonblack_assets() {
        let _lock = CWD_LOCK.lock().expect("cwd test lock should not be poisoned");
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let output_name = tmp
            .path()
            .file_name()
            .expect("tempdir should have a final component")
            .to_string_lossy()
            .to_string();
        let output_parent =
            tmp.path().parent().expect("tempdir should have a parent").to_path_buf();
        let _cwd = CurrentDirGuard::enter(&output_parent);

        let encoder = FakeVideoEncoder::default();
        let request = GenerationRequest {
            width: 256,
            height: 256,
            ..tiny_generation_request(output_name, GenerationDriftMode::Elliptical)
        };

        let summary = run_generation_with_video_encoder(&request, &encoder)
            .expect("low-density smoke generation should complete");

        assert!(
            png_channel_energy(summary.outputs.image_png_path()) > 0,
            "final image.png must contain visible non-black pixels",
        );
        assert!(
            summary.outputs.spectral_dir_path().join("32_542nm.png").is_file(),
            "spectral gallery should still be generated",
        );
        assert_eq!(encoder.calls.borrow().len(), 2, "main and spectral sweep videos are encoded");
        let main_video_bytes = encoder.calls.borrow()[0].4;
        assert!(main_video_bytes > 0, "main video encoder should receive frame bytes");
    }

    #[test]
    fn pipeline_smoke_test_covers_all_drift_modes() {
        let _lock = CWD_LOCK.lock().expect("cwd test lock should not be poisoned");
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let output_parent =
            tmp.path().parent().expect("tempdir should have a parent").to_path_buf();
        let _cwd = CurrentDirGuard::enter(&output_parent);

        for drift_mode in [
            GenerationDriftMode::Linear,
            GenerationDriftMode::Brownian,
            GenerationDriftMode::Elliptical,
        ] {
            let output_name = format!(
                "{}-{}",
                tmp.path()
                    .file_name()
                    .expect("tempdir should have a final component")
                    .to_string_lossy(),
                drift_mode.as_str()
            );
            let encoder = FakeVideoEncoder::default();
            let request = tiny_generation_request(output_name, drift_mode);

            let summary = run_generation_with_video_encoder(&request, &encoder)
                .expect("tiny drift generation should complete with fake encoder");

            assert!(summary.best_score.is_finite());
            assert_eq!(
                encoder.calls.borrow().len(),
                2,
                "{drift_mode:?} should encode main and spectral sweep videos"
            );
        }
    }
}
