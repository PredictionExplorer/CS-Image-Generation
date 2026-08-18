//! CLI front-end for the three-body problem visualization generator.

use clap::{Parser, ValueEnum};
use rayon::ThreadPoolBuilder;
use three_body_problem::{
    app,
    error::{self, Result},
    render::{self, RenderConfig},
    sim::Sha3RandomByteStream,
    spectrum_simd, viz,
};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum VizQualityArg {
    Draft,
    #[default]
    Final,
}

impl VizQualityArg {
    fn to_viz(self) -> viz::context::VizQuality {
        match self {
            Self::Draft => viz::context::VizQuality::Draft,
            Self::Final => viz::context::VizQuality::Final,
        }
    }
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
#[command(author, version, about = "Generate a curated three-body image and video from a seed.")]
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

    /// Render only the master still and WebP derivatives (skip videos,
    /// spectral gallery, and sweep video).
    #[arg(long, default_value_t = false)]
    image_only: bool,

    /// Render an experimental 360-degree orbit (turntable) video of the
    /// finished sculpture to videos/web/orbit.mp4 and videos/hq/orbit.mp4.
    #[arg(long, default_value_t = false)]
    orbit_video: bool,

    /// Duration of the full 360-degree orbit sweep in seconds.
    #[arg(long, default_value_t = 12.0)]
    orbit_seconds: f64,

    /// Orbit video frame rate in frames per second.
    #[arg(long, default_value_t = 30)]
    orbit_fps: u32,

    /// Camera elevation above the sculpture's equator during the orbit,
    /// in degrees (avoids fully edge-on passes for near-planar orbits).
    #[arg(long, default_value_t = 18.0)]
    orbit_tilt_deg: f64,

    /// Keep every Nth simulation step for orbit frames (energy-compensated;
    /// higher values render faster with slightly coarser strokes).
    #[arg(long, default_value_t = 2, value_parser = parse_bounded_stride)]
    orbit_step_stride: usize,

    /// Orbit video resolution as `WIDTHxHEIGHT` (default: half of
    /// --resolution, rounded down to even dimensions).
    #[arg(long, value_parser = parse_resolution)]
    orbit_resolution: Option<OutputResolution>,

    /// Visualization modes to render after the main outputs. Accepts mode
    /// flags, category names, or `all`; repeatable and comma-separable
    /// (see `--viz-list` and `docs/VIZ_MASTER_PLAN.md`).
    #[arg(long, action = clap::ArgAction::Append)]
    viz: Vec<String>,

    /// Print the visualization mode catalog and exit.
    #[arg(long, default_value_t = false)]
    viz_list: bool,

    /// Artifact quality for visualization modes.
    #[arg(long, value_enum, default_value_t = VizQualityArg::Final)]
    viz_quality: VizQualityArg,

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

const MAX_ORBIT_STEP_STRIDE: usize = 1024;

fn parse_bounded_stride(value: &str) -> std::result::Result<usize, String> {
    let n: usize =
        value.parse().map_err(|_| "orbit step stride must be a positive integer".to_string())?;
    if n == 0 || n > MAX_ORBIT_STEP_STRIDE {
        return Err(format!("orbit step stride must be between 1 and {MAX_ORBIT_STEP_STRIDE}"));
    }
    Ok(n)
}

/// Round a dimension down to the nearest even value (required by yuv420p),
/// clamped to a sane minimum.
fn even_dimension(value: u32) -> u32 {
    (value & !1).max(16)
}

/// Resolve the orbit output resolution: explicit value (rounded to even with
/// a warning if needed) or half of the main resolution.
fn resolve_orbit_resolution(
    explicit: Option<OutputResolution>,
    main_resolution: OutputResolution,
) -> OutputResolution {
    let raw = explicit.unwrap_or(OutputResolution {
        width: main_resolution.width / 2,
        height: main_resolution.height / 2,
    });
    let even =
        OutputResolution { width: even_dimension(raw.width), height: even_dimension(raw.height) };
    if explicit.is_some() && (even.width != raw.width || even.height != raw.height) {
        warn!(
            "orbit resolution {}x{} adjusted to {}x{} (even dimensions required by yuv420p)",
            raw.width, raw.height, even.width, even.height
        );
    }
    even
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

fn main() -> Result<()> {
    ThreadPoolBuilder::new()
        .stack_size(render::constants::THREAD_STACK_SIZE)
        .build_global()
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let args = Args::parse();

    setup_logging(&args.log_level);

    if args.viz_list {
        print!("{}", viz::catalog::render_list());
        return Ok(());
    }
    // Validate viz selection before any heavy work so typos fail fast.
    let viz_selection = viz::VizSelection::resolve(&args.viz)?;

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
    let selection = app::run_borda_selection_with_aesthetics(
        &mut rng,
        args.sims,
        args.steps,
        borda_weights.chaos_weight,
        borda_weights.equil_weight,
        DEFAULT_ESCAPE_THRESHOLD,
        visual_profile.parameters.stack,
        visual_profile.parameters.projection,
    )?;
    let mut positions = selection.positions.clone();
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
    app::apply_view_orientation(&mut positions, &rng, selection.stack);

    let drift_config = if args.drift == DriftModeArg::None {
        info!("STAGE 2.5/7: Drift disabled");
        None
    } else {
        app::apply_drift_transformation(
            &mut positions,
            args.drift.as_str(),
            None,
            None,
            None,
            &mut rng,
        )?
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

    let mut viz_tap_collector = if viz_selection.needs_frame_tap() && !args.image_only {
        let centroid = viz::context::trajectory_centroid_px(&positions, &render_ctx);
        Some(viz::context::FrameTapCollector::new(
            args.resolution.width,
            args.resolution.height,
            centroid,
        ))
    } else {
        None
    };

    let mut viz_state = viz::VizStageState::new();
    let mut retained_energy: Option<Vec<f32>> = None;

    if args.image_only {
        app::render_still_image(
            render::SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            spectral_settings,
            image_outputs,
        )?;
        let skipped = viz_selection.phase_flags(viz::VizPhase::Spd);
        if !skipped.is_empty() {
            warn!(
                "viz SPD-phase modes skipped under --image-only (no SPD buffer): {}",
                skipped.join(", ")
            );
        }
    } else {
        let mut tap_observe = |frame: &[u8]| {
            if let Some(collector) = viz_tap_collector.as_mut() {
                collector.observe(frame);
            }
        };
        let accum_spd = app::render_video(
            render::SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            spectral_settings,
            main_video_outputs,
            image_outputs,
            args.fast_encode,
            viz_selection.needs_frame_tap().then_some(&mut tap_observe),
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

        // Retain the compact energy field for trajectory-phase modes that
        // need it after the SPD buffer is dropped.
        if viz_selection.needs_energy_field() {
            retained_energy = Some(viz::common::spd::energy_field(&accum_spd));
        }

        // SPD phase: modes that need the accumulated spectral buffer run
        // here, while it is still alive. Failures are collected, not raised.
        if viz_selection.has_phase(viz::VizPhase::Spd) {
            let viz_ctx = viz::context::VizContext::new(
                &positions,
                &colors,
                &body_alphas,
                &selection.bodies,
                &levels,
                spectral_settings,
                args.resolution.width,
                args.resolution.height,
                hex_seed,
                &seed_dir,
                args.viz_quality.to_viz(),
                args.fast_encode,
                None,
                Some(&accum_spd),
                None,
                &rng,
            );
            viz_state.run_phase(&viz_ctx, &viz_selection, viz::VizPhase::Spd);
        }
    }

    if args.orbit_video {
        let orbit_resolution = resolve_orbit_resolution(args.orbit_resolution, args.resolution);
        error::validation::validate_dimensions(orbit_resolution.width, orbit_resolution.height)?;
        let orbit_config = render::orbit::OrbitVideoConfig {
            width: orbit_resolution.width,
            height: orbit_resolution.height,
            fps: args.orbit_fps,
            seconds: args.orbit_seconds,
            tilt_deg: args.orbit_tilt_deg,
            step_stride: args.orbit_step_stride,
        };
        let orbit_web_video = format!("{seed_dir}/videos/web/orbit.mp4");
        let orbit_hq_video = format!("{seed_dir}/videos/hq/orbit.mp4");
        app::render_orbit_video(
            render::SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            spectral_settings,
            &orbit_config,
            app::VideoOutputPaths { web: &orbit_web_video, high_quality: &orbit_hq_video },
            args.fast_encode,
        )?;
    }

    app::write_asset_manifest(
        &seed_dir,
        args.resolution.width,
        args.resolution.height,
        args.image_only,
    )?;

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

    // Trajectory-phase viz modes run last: the core package above is
    // complete regardless of any visualization failure, which is still
    // reported via the exit code.
    if !viz_selection.is_empty() {
        let tap_data = viz_tap_collector.map(viz::context::FrameTapCollector::finish);
        let viz_ctx = viz::context::VizContext::new(
            &positions,
            &colors,
            &body_alphas,
            &selection.bodies,
            &levels,
            spectral_settings,
            args.resolution.width,
            args.resolution.height,
            hex_seed,
            &seed_dir,
            args.viz_quality.to_viz(),
            args.fast_encode,
            tap_data,
            None,
            retained_energy,
            &rng,
        );
        viz_state.run_phase(&viz_ctx, &viz_selection, viz::VizPhase::Trajectory);
        viz_state.finish(&seed_dir)?;
    }

    Ok(())
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
        assert_eq!(args.log_level, DEFAULT_LOG_LEVEL);
        assert!(args.chaos_weight.is_none());
        assert!(args.equil_weight.is_none());
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
    fn test_reject_invalid_resolution() {
        let result = Args::try_parse_from(["three_body_problem", "--resolution", "wide-by-tall"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_orbit_defaults() {
        let args = Args::parse_from(["three_body_problem"]);
        assert!(!args.orbit_video);
        assert_eq!(args.orbit_seconds, 12.0);
        assert_eq!(args.orbit_fps, 30);
        assert_eq!(args.orbit_tilt_deg, 18.0);
        assert_eq!(args.orbit_step_stride, 2);
        assert!(args.orbit_resolution.is_none());
    }

    #[test]
    fn test_parse_orbit_flags() {
        let args = Args::parse_from([
            "three_body_problem",
            "--orbit-video",
            "--orbit-seconds",
            "8",
            "--orbit-fps",
            "24",
            "--orbit-tilt-deg",
            "25.5",
            "--orbit-step-stride",
            "4",
            "--orbit-resolution",
            "1280x828",
        ]);
        assert!(args.orbit_video);
        assert_eq!(args.orbit_seconds, 8.0);
        assert_eq!(args.orbit_fps, 24);
        assert_eq!(args.orbit_tilt_deg, 25.5);
        assert_eq!(args.orbit_step_stride, 4);
        assert_eq!(args.orbit_resolution, Some(OutputResolution { width: 1280, height: 828 }));
    }

    #[test]
    fn test_reject_zero_orbit_stride() {
        let result = Args::try_parse_from(["three_body_problem", "--orbit-step-stride", "0"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_orbit_resolution_defaults_to_even_half() {
        let resolved =
            resolve_orbit_resolution(None, OutputResolution { width: 3456, height: 2234 });
        assert_eq!(resolved, OutputResolution { width: 1728, height: 1116 });
    }

    #[test]
    fn test_resolve_orbit_resolution_rounds_explicit_odd_dimensions() {
        let resolved = resolve_orbit_resolution(
            Some(OutputResolution { width: 1281, height: 829 }),
            OutputResolution { width: 3456, height: 2234 },
        );
        assert_eq!(resolved, OutputResolution { width: 1280, height: 828 });
    }
}
