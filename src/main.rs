//! CLI front-end for the three-body problem visualization generator.
//!
//! # Exit status
//!
//! - `0`: the package is complete (with the ember edition unless `--no-ember`).
//! - `1`: any other failure, including an invalid `--seed` (e.g. an odd number of hex digits) or
//!   a resolution above 16,384 pixels per side; the package is incomplete.
//! - `2`: rejected by the argument parser (`clap`): an unknown flag or a malformed value.
//! - `3` (`EXIT_EMBER_FAILED`): the package is complete except for the ember edition, whose
//!   preflight or render failed.
//!
//! A process ended by a signal (an abort under the release profile's `panic = "abort"`, an
//! out-of-memory kill) has no exit status; shells report it as 128 plus the signal number.
//!
//! # Stale ember files
//!
//! Every run except `--metadata-only` first removes the ember files (`app::EMBER_OUTPUT_PATHS`)
//! that an earlier run left in the same output directory, so that a package never holds ember
//! files its `metadata/assets.json` does not list, whatever the flags of this run.

use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rayon::ThreadPoolBuilder;
use three_body_problem::{
    app,
    drift::AppliedDrift,
    ember::{EmberConfig, View},
    error::{self, AppError, Result},
    nft_traits,
    render::{self, RenderConfig},
    sim::{Body, Sha3RandomByteStream},
    spectrum_simd,
};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Exit status of a run whose package is complete except for the ember edition.
///
/// The ember preflight or the ember stage failed: the run logs the error, removes every ember
/// output (`app::EMBER_OUTPUT_PATHS`), and still writes the rest of the package exactly as a
/// `--no-ember` run does, including `metadata/assets.json` (without ember entries),
/// `metadata/generation.json` and `metadata/nft_traits.json`. `run.py` uploads such a package
/// and retries the ember edition later as a backfill.
const EXIT_EMBER_FAILED: u8 = 3;

/// The exit statuses, for `--help` (the crate docs hold the same table).
const EXIT_STATUS_HELP: &str = "Exit status:
  0  the package is complete
  1  any other failure, including an invalid --seed or a resolution above 16,384 per side; the
     package is incomplete
  2  rejected by the argument parser (unknown flag or malformed value)
  3  the package is complete except for the ember edition, which failed (its partial files are
     removed; the rest of the package, metadata included, is written as with --no-ember)";

const DEFAULT_OUTPUT_NAME: &str = "output";
const DEFAULT_NUM_SIMS: usize = 100_000;
const DEFAULT_NUM_STEPS: usize = 1_000_000;
const MAX_NUM_SIMS: usize = 10_000_000;
const MAX_NUM_STEPS: usize = 100_000_000;
const DEFAULT_RESOLUTION: &str = "3456x2234";
const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_LOCATION: f64 = 300.0;
const DEFAULT_VELOCITY: f64 = 1.0;
const DEFAULT_MIN_MASS: f64 = 100.0;
const DEFAULT_MAX_MASS: f64 = 300.0;
const DEFAULT_ALPHA_DENOM: usize = 15_000_000;
const DEFAULT_ALPHA_COMPRESS: f64 = 6.0;
const DEFAULT_ESCAPE_THRESHOLD: f64 = -0.3;
const DEFAULT_HDR_MODE: &str = "auto";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OutputResolution {
    width: u32,
    height: u32,
}

fn parse_resolution(value: &str) -> std::result::Result<OutputResolution, String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| "resolution must use WIDTHxHEIGHT format".to_string())?;
    let width = width
        .parse::<u32>()
        .map_err(|_| "resolution width must be a positive integer".to_string())?;
    let height = height
        .parse::<u32>()
        .map_err(|_| "resolution height must be a positive integer".to_string())?;

    if width == 0 || height == 0 {
        return Err("resolution dimensions must be greater than zero".to_string());
    }

    Ok(OutputResolution { width, height })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DriftModeArg {
    None,
    Linear,
    Brownian,
    Elliptical,
}

impl DriftModeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Brownian => "brownian",
            Self::Elliptical => "elliptical",
        }
    }
}

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate a curated three-body image and video from a seed.",
    after_help = EXIT_STATUS_HELP
)]
struct Args {
    #[arg(long, default_value = "0x100033")]
    seed: String,

    #[arg(short, long, default_value = DEFAULT_OUTPUT_NAME)]
    output: String,

    #[arg(long, default_value_t = DEFAULT_NUM_SIMS, value_parser = parse_bounded_sims)]
    sims: usize,

    #[arg(long, default_value_t = DEFAULT_NUM_STEPS, value_parser = parse_bounded_steps)]
    steps: usize,

    #[arg(short = 'r', long, default_value = DEFAULT_RESOLUTION, value_parser = parse_resolution)]
    resolution: OutputResolution,

    #[arg(long, value_enum, default_value_t = DriftModeArg::Elliptical)]
    drift: DriftModeArg,

    #[arg(long, default_value_t = false)]
    fast_encode: bool,

    /// Render only the stills and their WebP derivatives: the master and,
    /// unless `--no-ember`, the ember still with its certificate (skip all
    /// videos, the spectral gallery, and the sweep video).
    #[arg(long, default_value_t = false)]
    image_only: bool,

    /// Skip the ember edition (the orbit drawn in sumi ink by the fluid it
    /// stirs): no ember still, WebP images, videos, or `metadata/ember.json`
    /// (those of an earlier run into the same output directory are removed).
    /// Use it for orbits the edition rejects, e.g. short test runs
    /// (`--steps 20000`), which otherwise end with exit status 3.
    #[arg(long, default_value_t = false)]
    no_ember: bool,

    /// Skip all rendering and media encoding; run the simulation, selection,
    /// and trait analyses, then write only `metadata/generation.json` and
    /// `metadata/nft_traits.json`. Takes precedence over `--image-only`.
    /// Trait values are identical to a full render (everything is derived
    /// from the seed before rendering begins).
    #[arg(long, default_value_t = false)]
    metadata_only: bool,

    #[arg(long, default_value = DEFAULT_LOG_LEVEL)]
    log_level: String,

    /// Borda weight for chaos (FFT regularity) rank points.
    /// Omit to randomize from a curated range.
    #[arg(long)]
    chaos_weight: Option<f64>,

    /// Borda weight for equilateralness (triangle balance) rank points.
    /// Omit to randomize from a curated range.
    #[arg(long)]
    equil_weight: Option<f64>,

    /// Print the id of the ember look this generator renders (e.g. `ember-v3`,
    /// the `algorithm` of every `metadata/ember.json` it writes) and exit.
    /// The sync loop (run.py) renders the published editions of every older
    /// look again.
    #[arg(long, default_value_t = false, exclusive = true)]
    ember_algorithm: bool,
}

fn parse_bounded_sims(value: &str) -> std::result::Result<usize, String> {
    let n: usize = value.parse().map_err(|_| "sims must be a positive integer".to_string())?;
    if n == 0 || n > MAX_NUM_SIMS {
        return Err(format!("sims must be between 1 and {MAX_NUM_SIMS}"));
    }
    Ok(n)
}

fn parse_bounded_steps(value: &str) -> std::result::Result<usize, String> {
    let n: usize = value.parse().map_err(|_| "steps must be a positive integer".to_string())?;
    if n == 0 || n > MAX_NUM_STEPS {
        return Err(format!("steps must be between 1 and {MAX_NUM_STEPS}"));
    }
    Ok(n)
}

fn setup_logging(level: &str) {
    let env_filter =
        EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_LEVEL));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_thread_ids(false)
        .init();
}

struct ResolvedBordaWeights {
    chaos_weight: f64,
    equil_weight: f64,
    was_randomized: bool,
}

/// Draw a log-uniform equil/chaos ratio and derive weights from it.
///
/// Log-uniform sampling ensures that chaos-dominant ratios (e.g. 1/20)
/// and equil-dominant ratios (e.g. 20) are equally likely.  The ratio
/// is expressed as `equil_weight` / `chaos_weight`.
fn resolve_borda_weights(
    chaos_opt: Option<f64>,
    equil_opt: Option<f64>,
    rng: &mut Sha3RandomByteStream,
) -> ResolvedBordaWeights {
    use render::parameter_descriptors::EQUIL_CHAOS_RATIO;

    let (chaos_weight, equil_weight, was_randomized) = match (chaos_opt, equil_opt) {
        (Some(cw), Some(ew)) => (cw, ew, false),
        (None, None) => {
            let log_min = EQUIL_CHAOS_RATIO.min.ln();
            let log_max = EQUIL_CHAOS_RATIO.max.ln();
            let ratio = (log_min + rng.next_f64() * (log_max - log_min)).exp();
            (1.0, ratio, true)
        }
        (Some(cw), None) => {
            let log_min = EQUIL_CHAOS_RATIO.min.ln();
            let log_max = EQUIL_CHAOS_RATIO.max.ln();
            let ratio = (log_min + rng.next_f64() * (log_max - log_min)).exp();
            (cw, cw * ratio, true)
        }
        (None, Some(ew)) => {
            let log_min = EQUIL_CHAOS_RATIO.min.ln();
            let log_max = EQUIL_CHAOS_RATIO.max.ln();
            let ratio = (log_min + rng.next_f64() * (log_max - log_min)).exp();
            (ew / ratio, ew, true)
        }
    };

    let ratio = equil_weight / chaos_weight;
    let label = if ratio >= 1.0 {
        format!("equil {ratio:.1}x")
    } else {
        format!("chaos {:.1}x", 1.0 / ratio)
    };
    info!(
        "Borda weights: chaos={:.3}, equil={:.3} ({}){}",
        chaos_weight,
        equil_weight,
        label,
        if was_randomized { " [randomized]" } else { " [explicit]" }
    );

    ResolvedBordaWeights { chaos_weight, equil_weight, was_randomized }
}

fn build_generation_log_config(
    args: &Args,
    resolved: &render::randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    borda_weights: &ResolvedBordaWeights,
) -> app::GenerationLogConfig {
    let bloom_mode = if resolved.enable_bloom {
        render_config.bloom_mode.as_str()
    } else {
        render::BloomMode::None.as_str()
    };
    let (palette_fingerprint, palette_gate) = render::color::current_palette_metadata();

    app::GenerationLogConfig {
        num_steps_sim: args.steps,
        width: resolved.width,
        height: resolved.height,
        clip_black: resolved.clip_black,
        clip_white: resolved.clip_white,
        alpha_denom: DEFAULT_ALPHA_DENOM,
        alpha_compress: DEFAULT_ALPHA_COMPRESS,
        escape_threshold: DEFAULT_ESCAPE_THRESHOLD,
        drift_mode: args.drift.as_str().to_string(),
        visual_profile: render::visual_profile::COSMIC_SIGNATURE_PROFILE_NAME.to_string(),
        post_effects_enabled: resolved.any_legacy_effect_enabled(),
        bloom_mode: bloom_mode.to_string(),
        hdr_mode: DEFAULT_HDR_MODE.to_string(),
        hdr_scale: render_config.hdr_scale,
        dispersion_strength: render::constants::SPECTRAL_DISPERSION_STRENGTH,
        dispersion_mode: "crisp_off".to_string(),
        palette_fingerprint: palette_fingerprint.clone(),
        palette_gate: palette_gate.clone(),
        min_mass: DEFAULT_MIN_MASS,
        max_mass: DEFAULT_MAX_MASS,
        location: DEFAULT_LOCATION,
        velocity: DEFAULT_VELOCITY,
        chaos_weight: borda_weights.chaos_weight,
        equil_weight: borda_weights.equil_weight,
        weights_randomized: borda_weights.was_randomized,
    }
}

/// Checks, before the main render, that the ember edition can render the selected orbit: the
/// ember stage runs last, so an orbit it rejects (typically a short test run whose orbit lasts
/// too little fluid time) is known in seconds and the stage is skipped instead of failing after
/// the whole main render and its encodes.
fn preflight_ember_stage(
    args: &Args,
    bodies: &[Body],
    view: &View,
    config: &EmberConfig,
) -> Result<()> {
    info!("STAGE EMBER (preflight): checking the ember edition can render this orbit...");
    app::preflight_ember_edition(
        bodies,
        view,
        args.steps,
        args.resolution.width,
        args.resolution.height,
        config,
    )
    .map(|_| ())
}

/// `error` followed by its sources, `": "`-separated, skipping a source whose message the text
/// already ends with (wrappers such as `AppError::Ember` repeat their source's message, while
/// `RenderError::VideoEncoding` keeps its details only in the source).
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        let message = inner.to_string();
        if !text.ends_with(&message) {
            text = format!("{text}: {message}");
        }
        source = inner.source();
    }
    text
}

/// Logs that the ember edition `what` (e.g. "failed") with `error`, and what happens next.
fn warn_ember_failed(what: &str, error: &AppError) {
    warn!("The ember edition {what}: {}", error_chain(error));
    warn!(
        "Continuing without the ember edition: the rest of the package is written as with \
         --no-ember and the run exits with status {EXIT_EMBER_FAILED}. Pass --no-ember to skip \
         the edition deliberately (e.g. for short test runs, whose orbits are too short for it); \
         run.py uploads such a package and retries the ember edition later as a backfill."
    );
}

/// Renders the ember edition of the selected orbit into the package (after the main outputs,
/// before the asset manifest) and returns what the manifest records about it.
fn render_ember_stage(
    args: &Args,
    seed_dir: &str,
    hex_seed: &str,
    seed_bytes: &[u8],
    bodies: &[Body],
    view: &View,
    config: &EmberConfig,
) -> Result<app::EmberManifest> {
    let path = |relative: &str| format!("{seed_dir}/{relative}");
    let [still_png, full_webp, preview_webp, web_video, hq_video, certificate] = [
        app::EMBER_STILL_PATH,
        app::EMBER_FULL_WEBP_PATH,
        app::EMBER_PREVIEW_WEBP_PATH,
        app::EMBER_WEB_VIDEO_PATH,
        app::EMBER_HQ_VIDEO_PATH,
        app::EMBER_CERTIFICATE_PATH,
    ]
    .map(path);
    let slow_web_videos = app::EMBER_SLOW_FILMS.map(|film| path(film.path));
    let summary = app::render_ember_edition(&app::EmberEditionRequest {
        seed_hex: hex_seed,
        seed_bytes,
        bodies,
        view,
        steps: args.steps,
        width: args.resolution.width,
        height: args.resolution.height,
        image_only: args.image_only,
        fast_encode: args.fast_encode,
        config,
        paths: app::EmberOutputPaths {
            still_png: &still_png,
            full_webp: &full_webp,
            preview_webp: &preview_webp,
            web_video: &web_video,
            slow_web_videos: slow_web_videos.each_ref().map(String::as_str),
            hq_video: &hq_video,
            certificate: &certificate,
        },
    })?;
    Ok(app::EmberManifest::from_summary(&summary, args.fast_encode))
}

/// Runs the generator; see the crate docs for the exit statuses.
fn main() -> Result<ExitCode> {
    ThreadPoolBuilder::new()
        .stack_size(render::constants::THREAD_STACK_SIZE)
        .build_global()
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let args = Args::parse();
    if args.ember_algorithm {
        println!("{}", three_body_problem::ember::certificate::ALGORITHM_VERSION);
        return Ok(ExitCode::SUCCESS);
    }

    setup_logging(&args.log_level);

    let enhancements = app::Enhancements::default();
    spectrum_simd::SAT_BOOST_ENABLED
        .store(enhancements.sat_boost, std::sync::atomic::Ordering::Relaxed);
    render::ACES_TWEAK_ENABLED.store(enhancements.aces_tweak, std::sync::atomic::Ordering::Relaxed);
    render::drawing::DISPERSION_BOOST_ENABLED
        .store(enhancements.dispersion_boost, std::sync::atomic::Ordering::Relaxed);

    error::validation::validate_dimensions(args.resolution.width, args.resolution.height)?;

    let seed_bytes = app::parse_seed(&args.seed)?;
    let hex_seed = if args.seed.starts_with("0x") { &args.seed[2..] } else { &args.seed };

    let seed_dir = app::setup_seed_directory(&args.output)?;

    let mut rng = Sha3RandomByteStream::new(
        &seed_bytes,
        DEFAULT_MIN_MASS,
        DEFAULT_MAX_MASS,
        DEFAULT_LOCATION,
        DEFAULT_VELOCITY,
    );

    info!("Resolving CosmicSignature visual profile...");
    let visual_profile = render::visual_profile::ResolvedVisualProfile::cosmic_signature(
        &mut rng,
        args.resolution.width,
        args.resolution.height,
    );

    let borda_weights = resolve_borda_weights(args.chaos_weight, args.equil_weight, &mut rng);

    // Borda search ranks physics quality; the aesthetic pass proxy-renders the
    // top candidates with this seed's layer stack (in its projection space)
    // and keeps the most paintable orbit (with its trajectory, so no
    // re-simulation is needed).
    let mut selection = app::run_borda_selection_with_aesthetics(
        &mut rng,
        args.sims,
        args.steps,
        borda_weights.chaos_weight,
        borda_weights.equil_weight,
        DEFAULT_ESCAPE_THRESHOLD,
        visual_profile.parameters.stack,
        visual_profile.parameters.projection,
    )?;
    // Moved out, not copied: nothing reads `selection.positions` afterwards (the ember edition
    // and the traits re-simulate the raw orbit from `selection.bodies`; the ember edition then
    // re-applies the view captured below), so dropping `positions` further down really frees
    // the trajectory.
    let mut positions = std::mem::take(&mut selection.positions);
    let visual_profile = visual_profile.with_stack(selection.stack);
    let resolved_effect_config = visual_profile.effect_config.clone();
    let randomization_log = visual_profile.randomization_log.clone();

    let num_randomized = randomization_log
        .effects
        .iter()
        .map(|effect| effect.parameters.iter().filter(|param| param.was_randomized).count())
        .sum::<usize>();

    info!(
        "   => Resolved {} visual profile record(s) ({} parameters randomized, {} explicit); preferred stack={} chosen stack={} projection={} symmetry={}",
        randomization_log.effects.len(),
        num_randomized,
        randomization_log.effects.iter().map(|effect| effect.parameters.len()).sum::<usize>()
            - num_randomized,
        visual_profile.preferred_stack().label(),
        visual_profile.parameters.stack.label(),
        visual_profile.parameters.projection.label(),
        visual_profile.parameters.symmetry.label(),
    );

    // Seeded viewing orientation: photograph the 3D orbit from the
    // best-composed of several candidate angles.
    let view_rotation = app::apply_view_orientation(&mut positions, &rng, selection.stack);

    let (drift_config, applied_drift) = if args.drift == DriftModeArg::None {
        info!("STAGE 2.5/7: Drift disabled");
        (None, AppliedDrift::None)
    } else {
        let drift = app::apply_drift_transformation(
            &mut positions,
            args.drift.as_str(),
            None,
            None,
            None,
            &mut rng,
        )?;
        (Some(drift.config), drift.applied)
    };

    let (colors, body_alphas) = app::generate_colors(
        &mut rng,
        args.steps,
        DEFAULT_ALPHA_DENOM,
        &enhancements,
        visual_profile.parameters.palette_phase,
    );

    info!("   => Using OKLab color space for accumulation");
    info!("STAGE 4/7: Determining bounding box...");
    let render_ctx = render::context::RenderContext::new(
        args.resolution.width,
        args.resolution.height,
        &positions,
        enhancements.aspect_correction,
    );
    let bbox = render_ctx.bounds();
    info!(
        "   => X: [{:.3}, {:.3}], Y: [{:.3}, {:.3}]",
        bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
    );
    let spd_gib = render::estimate_full_spd_bytes(args.resolution.width, args.resolution.height)
        as f64
        / (1024.0 * 1024.0 * 1024.0);
    info!("   => Full-frame SPD memory estimate: {spd_gib:.2} GiB");

    let render_config = RenderConfig {
        hdr_scale: resolved_effect_config.hdr_scale,
        bloom_mode: render::BloomMode::Dog,
    };

    let scene_traits = visual_profile.parameters.scene_traits();
    info!(
        "   => Scene traits: stack={} symmetry={} line_weight={:.3} age_ramp={:+.3} exposure_key={:.3} halation={:.3} spikes={:.3} stardust={}",
        scene_traits.stack.label(),
        scene_traits.symmetry.label(),
        scene_traits.line_weight,
        scene_traits.age_ramp,
        visual_profile.parameters.exposure_key,
        visual_profile.parameters.halation_strength,
        scene_traits.spikes.strength,
        scene_traits.stardust.count,
    );

    // The ember edition's look and simulation parameters (recorded in its certificate).
    let ember_config = EmberConfig::default();
    // The main edition's view of the orbit, which the ember edition's bodies follow: the
    // projection space, the viewing rotation, the drift and the frame resolved above.
    let ember_view = app::ember_view(
        visual_profile.parameters.projection,
        &view_rotation,
        applied_drift,
        bbox,
        scene_traits.symmetry,
        (args.resolution.width, args.resolution.height),
    );
    // Set once the ember edition has failed (its view, its preflight or its stage): the package
    // is then completed without it, and the run exits with `EXIT_EMBER_FAILED`.
    let mut ember_failed = false;

    if args.metadata_only {
        info!("METADATA-ONLY MODE: skipping histogram, rendering, and asset manifest");
        // The preflight costs a fraction of a second and logs what the ember edition would cost
        // for this orbit; nothing is rendered and the exit status does not depend on it.
        if !args.no_ember
            && let Err(error) = ember_view.and_then(|view| {
                preflight_ember_stage(&args, &selection.bodies, &view, &ember_config)
            })
        {
            warn!("The ember edition cannot render this orbit: {}", error_chain(&error));
        }
    } else {
        // Ember files an earlier run left in this directory would survive a run that writes
        // fewer of them (`--no-ember`, `--image-only`, a failed stage), unlisted in the manifest
        // this run writes. Removed up front, before anything is rendered; unremovable files fail
        // the run, since a package must never ship ember files its manifest does not describe.
        app::remove_ember_outputs(&seed_dir)?;
        // `Some` while the ember edition is to be rendered: its view passed the preflight.
        let ember_view = if args.no_ember {
            None
        } else {
            let checked = ember_view.and_then(|view| {
                preflight_ember_stage(&args, &selection.bodies, &view, &ember_config).map(|()| view)
            });
            match checked {
                Ok(view) => Some(view),
                Err(error) => {
                    warn_ember_failed("cannot render this orbit (preflight)", &error);
                    ember_failed = true;
                    None
                }
            }
        };
        {
            // Main still, videos, spectral gallery and sweep. `levels`, the frame buffers and the
            // full-frame SPD accumulation (GiBs at full size) are freed at the end of this block,
            // before the ember edition allocates its own buffers.
            let levels = app::build_histogram_and_levels(
                &positions,
                &colors,
                &body_alphas,
                &resolved_effect_config,
                &render_config,
                enhancements.aspect_correction,
                &visual_profile.parameters,
            )?;

            let image_master_png = format!("{seed_dir}/images/source/master.png");
            let image_full_webp = format!("{seed_dir}/images/web/full.webp");
            let image_preview_webp = format!("{seed_dir}/images/web/preview.webp");
            let image_outputs = app::ImageOutputPaths {
                master_png: &image_master_png,
                full_webp: &image_full_webp,
                preview_webp: &image_preview_webp,
            };
            let main_web_video = format!("{seed_dir}/videos/web/main.mp4");
            let main_hq_video = format!("{seed_dir}/videos/hq/main.mp4");
            let main_video_outputs =
                app::VideoOutputPaths { web: &main_web_video, high_quality: &main_hq_video };

            let spectral_settings = render::SpectralRenderSettings::new(
                &resolved_effect_config,
                &render_config,
                enhancements.aspect_correction,
            )
            .with_traits(scene_traits);

            if args.image_only {
                app::render_still_image(
                    render::SpectralScene::new(&positions, &colors, &body_alphas),
                    &levels,
                    spectral_settings,
                    image_outputs,
                )?;
            } else {
                let accum_spd = app::render_video(
                    render::SpectralScene::new(&positions, &colors, &body_alphas),
                    &levels,
                    spectral_settings,
                    main_video_outputs,
                    image_outputs,
                    args.fast_encode,
                )?;

                let spectral_dir = format!("{seed_dir}/spectral");
                let spectral_sweep_web_path = format!("{seed_dir}/videos/web/spectral_sweep.mp4");
                let spectral_sweep_hq_path = format!("{seed_dir}/videos/hq/spectral_sweep.mp4");
                let spectral_sweep_outputs = app::VideoOutputPaths {
                    web: &spectral_sweep_web_path,
                    high_quality: &spectral_sweep_hq_path,
                };

                app::generate_spectral_gallery(
                    &accum_spd,
                    args.resolution.width,
                    args.resolution.height,
                    &spectral_dir,
                )?;

                app::generate_spectral_sweep_video(
                    &accum_spd,
                    args.resolution.width,
                    args.resolution.height,
                    spectral_sweep_outputs,
                    args.fast_encode,
                )?;
            }
        }
        // Free the projected trajectory (moved out of `selection` above) and its colours before
        // the ember edition allocates its buffers.
        drop((positions, colors, body_alphas));

        let ember = if args.no_ember {
            info!("STAGE EMBER: skipped (--no-ember)");
            None
        } else if let Some(view) = &ember_view {
            match render_ember_stage(
                &args,
                &seed_dir,
                hex_seed,
                &seed_bytes,
                &selection.bodies,
                view,
                &ember_config,
            ) {
                Ok(manifest) => Some(manifest),
                Err(error) => {
                    warn_ember_failed("failed", &error);
                    ember_failed = true;
                    None
                }
            }
        } else {
            info!("STAGE EMBER: skipped (the preflight rejected this orbit)");
            None
        };
        if ember_failed {
            // A failed stage can leave partial files (a PNG whose WebPs failed, truncated
            // videos, a partial certificate) that the manifest will not list. Unremovable files
            // fail the run, as above.
            app::remove_ember_outputs(&seed_dir)?;
        }

        app::write_asset_manifest(
            &seed_dir,
            args.resolution.width,
            args.resolution.height,
            args.steps,
            args.image_only,
            ember.as_ref(),
        )?;
    }

    info!(
        "Done! Best orbit => Weighted Borda = {:.3}\nHave a nice day!",
        selection.result.total_score_weighted
    );

    let generation_log_config =
        build_generation_log_config(&args, &resolved_effect_config, &render_config, &borda_weights);
    if let Err(e) = app::log_generation(
        &generation_log_config,
        &args.output,
        hex_seed,
        &drift_config,
        args.sims,
        &selection,
        Some(&randomization_log),
        Some(&format!("{seed_dir}/metadata/generation.json")),
    ) {
        warn!("Generation logging failed (non-fatal): {e}");
    }

    // The public trait file is a required package artifact: fail hard rather
    // than upload a package without it.
    nft_traits::compute_and_write(
        &seed_dir,
        &nft_traits::NftTraitsInputs {
            seed_hex: hex_seed,
            parameters: &visual_profile.parameters,
            selection: &selection,
            drift: drift_config.as_ref(),
            drift_mode: args.drift.as_str(),
            num_sims: args.sims,
            num_steps: args.steps,
            chaos_weight: borda_weights.chaos_weight,
            equil_weight: borda_weights.equil_weight,
            weights_randomized: borda_weights.was_randomized,
            escape_threshold: DEFAULT_ESCAPE_THRESHOLD,
            width: args.resolution.width,
            height: args.resolution.height,
        },
    )?;

    if ember_failed {
        warn!(
            "The package is complete except for the ember edition (see the warnings above); \
             exiting with status {EXIT_EMBER_FAILED}."
        );
        return Ok(ExitCode::from(EXIT_EMBER_FAILED));
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_defaults() {
        let args = Args::parse_from(["three_body_problem"]);

        assert_eq!(args.output, DEFAULT_OUTPUT_NAME);
        assert_eq!(args.sims, DEFAULT_NUM_SIMS);
        assert_eq!(args.steps, DEFAULT_NUM_STEPS);
        assert_eq!(args.resolution, OutputResolution { width: 3456, height: 2234 });
        assert_eq!(args.drift, DriftModeArg::Elliptical);
        assert!(!args.fast_encode);
        assert!(!args.image_only);
        assert!(!args.metadata_only);
        assert!(!args.no_ember, "the ember edition is part of the default package");
        assert_eq!(args.log_level, DEFAULT_LOG_LEVEL);
        assert!(args.chaos_weight.is_none());
        assert!(args.equil_weight.is_none());
        assert!(!args.ember_algorithm);
    }

    #[test]
    fn test_ember_algorithm_stands_alone() {
        assert!(Args::parse_from(["three_body_problem", "--ember-algorithm"]).ember_algorithm);
        assert!(
            Args::try_parse_from(["three_body_problem", "--ember-algorithm", "--no-ember"])
                .is_err()
        );
    }

    #[test]
    fn test_parse_output_mode_flags() {
        let args = Args::parse_from(["three_body_problem", "--no-ember"]);
        assert!(args.no_ember);
        assert!(!args.image_only);

        let args = Args::parse_from(["three_body_problem", "--image-only", "--no-ember"]);
        assert!(args.image_only && args.no_ember);

        let args = Args::parse_from(["three_body_problem", "--image-only", "--metadata-only"]);
        assert!(args.image_only && args.metadata_only && !args.no_ember);

        assert!(Args::try_parse_from(["three_body_problem", "--no-ember=maybe"]).is_err());
    }

    #[test]
    fn test_parse_custom_resolution_and_drift() {
        let args = Args::parse_from([
            "three_body_problem",
            "--output",
            "gallery-piece",
            "--resolution",
            "1280x720",
            "--drift",
            "none",
            "--sims",
            "5000",
            "--steps",
            "12000",
            "--fast-encode",
        ]);

        assert_eq!(args.output, "gallery-piece");
        assert_eq!(args.resolution, OutputResolution { width: 1280, height: 720 });
        assert_eq!(args.drift, DriftModeArg::None);
        assert_eq!(args.sims, 5000);
        assert_eq!(args.steps, 12000);
        assert!(args.fast_encode);
    }

    #[test]
    fn test_parse_explicit_borda_weights() {
        let args = Args::parse_from([
            "three_body_problem",
            "--chaos-weight",
            "1.5",
            "--equil-weight",
            "8.0",
        ]);
        assert_eq!(args.chaos_weight, Some(1.5));
        assert_eq!(args.equil_weight, Some(8.0));
    }

    #[test]
    fn test_resolve_borda_weights_randomized() {
        let mut rng = Sha3RandomByteStream::new(&[0x42; 32], 100.0, 300.0, 300.0, 1.0);
        let w = resolve_borda_weights(None, None, &mut rng);
        assert!(w.was_randomized);
        assert_eq!(w.chaos_weight, 1.0);
        let ratio = w.equil_weight / w.chaos_weight;
        assert!((0.2..=50.0).contains(&ratio), "ratio {ratio} outside [0.2, 50.0]");
    }

    #[test]
    fn test_resolve_borda_weights_explicit() {
        let mut rng = Sha3RandomByteStream::new(&[0x42; 32], 100.0, 300.0, 300.0, 1.0);
        let w = resolve_borda_weights(Some(1.0), Some(10.0), &mut rng);
        assert!(!w.was_randomized);
        assert_eq!(w.chaos_weight, 1.0);
        assert_eq!(w.equil_weight, 10.0);
    }

    #[test]
    fn test_resolve_borda_weights_partial_chaos_explicit() {
        let mut rng = Sha3RandomByteStream::new(&[0x42; 32], 100.0, 300.0, 300.0, 1.0);
        let w = resolve_borda_weights(Some(0.5), None, &mut rng);
        assert!(w.was_randomized);
        assert_eq!(w.chaos_weight, 0.5);
        let ratio = w.equil_weight / w.chaos_weight;
        assert!((0.2..=50.0).contains(&ratio), "ratio {ratio} outside [0.2, 50.0]");
    }

    #[test]
    fn test_resolve_borda_weights_partial_equil_explicit() {
        let mut rng = Sha3RandomByteStream::new(&[0x42; 32], 100.0, 300.0, 300.0, 1.0);
        let w = resolve_borda_weights(None, Some(5.0), &mut rng);
        assert!(w.was_randomized);
        assert_eq!(w.equil_weight, 5.0);
        let ratio = w.equil_weight / w.chaos_weight;
        assert!((0.2..=50.0).contains(&ratio), "ratio {ratio} outside [0.2, 50.0]");
    }

    #[test]
    fn test_resolve_borda_weights_range_coverage() {
        for seed_byte in 0u8..=255 {
            let seed = [seed_byte; 32];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let w = resolve_borda_weights(None, None, &mut rng);
            assert_eq!(w.chaos_weight, 1.0);
            let ratio = w.equil_weight / w.chaos_weight;
            assert!(
                (0.2..=50.0).contains(&ratio),
                "seed {seed_byte} produced ratio {ratio} outside [0.2, 50.0]"
            );
        }
    }

    #[test]
    fn test_error_chain_names_the_cause_once() {
        use three_body_problem::{ember::EmberError, render::error::RenderError};

        let sink = AppError::from(EmberError::Sink("video encoder pipe: Broken pipe".into()));
        assert_eq!(
            error_chain(&sink),
            "Ember edition error: frame sink failed: video encoder pipe: Broken pipe"
        );
        let encode = AppError::from(RenderError::VideoEncoding(std::io::Error::other(
            "FFmpeg failed for hq/ember.mp4 with exit status: 1",
        )));
        assert_eq!(
            error_chain(&encode),
            "Video encoding failed: FFmpeg failed for hq/ember.mp4 with exit status: 1"
        );
    }

    #[test]
    fn test_help_documents_the_exit_statuses() {
        assert_eq!(EXIT_EMBER_FAILED, 3, "run.py and the README rely on this value");
        assert!(EXIT_STATUS_HELP.contains(&format!("\n  {EXIT_EMBER_FAILED}  the package is")));
        // The contract run.py and the docs state (tests/cli.rs checks the binary keeps it).
        assert!(EXIT_STATUS_HELP.contains("\n  1  any other failure, including an invalid --seed"));
        assert!(EXIT_STATUS_HELP.contains("\n  2  rejected by the argument parser"));
        let help = <Args as clap::CommandFactory>::command().render_long_help().to_string();
        assert!(help.contains("Exit status:"), "{help}");
        assert!(help.contains("rejected by the argument parser"), "{help}");
    }

    #[test]
    fn test_reject_invalid_resolution() {
        let result = Args::try_parse_from(["three_body_problem", "--resolution", "wide-by-tall"]);
        assert!(result.is_err());
    }
}
