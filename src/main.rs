use clap::{Parser, ValueEnum};
use three_body_problem::{
    app,
    error::{self, Result},
    render::{self, RenderConfig},
    sim::Sha3RandomByteStream,
    spectrum_simd,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

const DEFAULT_OUTPUT_NAME: &str = "output";
const DEFAULT_NUM_SIMS: usize = 100_000;
const DEFAULT_NUM_STEPS: usize = 1_000_000;
const DEFAULT_RESOLUTION: &str = "1920x1080";
const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_LOCATION: f64 = 300.0;
const DEFAULT_VELOCITY: f64 = 1.0;
const DEFAULT_MIN_MASS: f64 = 100.0;
const DEFAULT_MAX_MASS: f64 = 300.0;
const DEFAULT_CHAOS_WEIGHT: f64 = 0.75;
const DEFAULT_EQUIL_WEIGHT: f64 = 11.0;
const DEFAULT_ALPHA_DENOM: usize = 15_000_000;
const DEFAULT_ALPHA_COMPRESS: f64 = 6.0;
const DEFAULT_ESCAPE_THRESHOLD: f64 = -0.3;
const DEFAULT_HDR_MODE: &str = "auto";
const DEFAULT_PERCEPTUAL_GAMUT_MODE: &str = "preserve-hue";

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
#[command(author, version, about = "Generate a curated three-body image and video from a seed.")]
struct Args {
    #[arg(long, default_value = "0x100033")]
    seed: String,

    #[arg(short, long, default_value = DEFAULT_OUTPUT_NAME)]
    output: String,

    #[arg(long, default_value_t = DEFAULT_NUM_SIMS)]
    sims: usize,

    #[arg(long, default_value_t = DEFAULT_NUM_STEPS)]
    steps: usize,

    #[arg(short = 'r', long, default_value = DEFAULT_RESOLUTION, value_parser = parse_resolution)]
    resolution: OutputResolution,

    #[arg(long, value_enum, default_value_t = DriftModeArg::Elliptical)]
    drift: DriftModeArg,

    #[arg(long, default_value_t = false)]
    fast_encode: bool,

    #[arg(long, default_value = DEFAULT_LOG_LEVEL)]
    log_level: String,
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

fn resolved_perceptual_blur_radius(
    resolved: &render::randomizable_config::ResolvedEffectConfig,
    bloom_mode: render::BloomMode,
) -> Option<usize> {
    if !resolved.enable_perceptual_blur {
        return None;
    }

    let use_gaussian_bloom = bloom_mode == render::BloomMode::Gaussian && resolved.enable_bloom;
    let softness_stack_score = (if use_gaussian_bloom { 1.0 } else { 0.0 })
        + if resolved.enable_chromatic_bloom { 0.8 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.85 } else { 0.0 }
        + if resolved.enable_glow { 0.55 } else { 0.0 }
        + if resolved.enable_atmospheric_depth { 0.35 } else { 0.0 };
    let radius_scale = if softness_stack_score >= 2.0 { 0.0030 } else { 0.0036 };
    let min_dim = resolved.width.min(resolved.height);

    Some((radius_scale * min_dim as f64).round().max(1.0) as usize)
}

fn build_generation_log_config(
    args: &Args,
    resolved: &render::randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
) -> app::GenerationLogConfig {
    let min_dim = resolved.width.min(resolved.height);
    let bloom_mode = if resolved.enable_bloom {
        render_config.bloom_mode.as_str()
    } else {
        render::BloomMode::None.as_str()
    };

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
        bloom_mode: bloom_mode.to_string(),
        dog_strength: resolved.dog_strength,
        dog_sigma: Some(resolved.dog_sigma_scale * min_dim as f64),
        dog_ratio: resolved.dog_ratio,
        hdr_mode: DEFAULT_HDR_MODE.to_string(),
        hdr_scale: render_config.hdr_scale,
        perceptual_blur: if resolved.enable_perceptual_blur {
            "on".to_string()
        } else {
            "off".to_string()
        },
        perceptual_blur_radius: resolved_perceptual_blur_radius(resolved, render_config.bloom_mode),
        perceptual_blur_strength: resolved.perceptual_blur_strength,
        perceptual_gamut_mode: DEFAULT_PERCEPTUAL_GAMUT_MODE.to_string(),
        min_mass: DEFAULT_MIN_MASS,
        max_mass: DEFAULT_MAX_MASS,
        location: DEFAULT_LOCATION,
        velocity: DEFAULT_VELOCITY,
        chaos_weight: DEFAULT_CHAOS_WEIGHT,
        equil_weight: DEFAULT_EQUIL_WEIGHT,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging(&args.log_level);

    let enhancements = app::Enhancements::default();
    spectrum_simd::SAT_BOOST_ENABLED
        .store(enhancements.sat_boost, std::sync::atomic::Ordering::Relaxed);
    render::ACES_TWEAK_ENABLED.store(enhancements.aces_tweak, std::sync::atomic::Ordering::Relaxed);
    render::drawing::DISPERSION_BOOST_ENABLED
        .store(enhancements.dispersion_boost, std::sync::atomic::Ordering::Relaxed);

    app::setup_directories()?;
    error::validation::validate_dimensions(args.resolution.width, args.resolution.height)?;

    let seed_bytes = app::parse_seed(&args.seed)?;
    let hex_seed = if args.seed.starts_with("0x") { &args.seed[2..] } else { &args.seed };
    let noise_seed = app::derive_noise_seed(&seed_bytes);

    let mut rng = Sha3RandomByteStream::new(
        &seed_bytes,
        DEFAULT_MIN_MASS,
        DEFAULT_MAX_MASS,
        DEFAULT_LOCATION,
        DEFAULT_VELOCITY,
    );

    info!("Resolving effect configuration...");
    let randomizable_config = render::randomizable_config::RandomizableEffectConfig::default();
    let (resolved_effect_config, randomization_log) =
        randomizable_config.resolve(&mut rng, args.resolution.width, args.resolution.height);

    let num_randomized = randomization_log
        .effects
        .iter()
        .map(|effect| effect.parameters.iter().filter(|param| param.was_randomized).count())
        .sum::<usize>();

    info!(
        "   => Resolved {} effects ({} parameters randomized, {} explicit)",
        randomization_log.effects.len(),
        num_randomized,
        randomization_log.effects.iter().map(|effect| effect.parameters.len()).sum::<usize>()
            - num_randomized
    );

    let (best_bodies, best_info) = app::run_borda_selection(
        &mut rng,
        args.sims,
        args.steps,
        DEFAULT_CHAOS_WEIGHT,
        DEFAULT_EQUIL_WEIGHT,
        DEFAULT_ESCAPE_THRESHOLD,
    )?;

    let mut positions = app::simulate_best_orbit(best_bodies, args.steps);

    let drift_config = if args.drift != DriftModeArg::None {
        app::apply_drift_transformation(
            &mut positions,
            args.drift.as_str(),
            None,
            None,
            None,
            &mut rng,
        )
    } else {
        info!("STAGE 2.5/7: Drift disabled");
        None
    };

    let (colors, body_alphas) =
        app::generate_colors(&mut rng, args.steps, DEFAULT_ALPHA_DENOM, &enhancements);

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

    let render_config = RenderConfig {
        hdr_scale: resolved_effect_config.hdr_scale,
        bloom_mode: render::BloomMode::Dog,
    };

    let levels = app::build_histogram_and_levels(
        &positions,
        &colors,
        &body_alphas,
        &resolved_effect_config,
        noise_seed,
        &render_config,
        enhancements.aspect_correction,
    )?;

    let output_png = format!("pics/{}.png", args.output);
    let output_vid = format!("vids/{}.mp4", args.output);

    app::render_video(
        &positions,
        &colors,
        &body_alphas,
        &resolved_effect_config,
        &levels,
        noise_seed,
        &render_config,
        &output_vid,
        &output_png,
        args.fast_encode,
        enhancements.aspect_correction,
        true,
    )?;

    info!(
        "Done! Best orbit => Weighted Borda = {:.3}\nHave a nice day!",
        best_info.total_score_weighted
    );

    let generation_log_config =
        build_generation_log_config(&args, &resolved_effect_config, &render_config);
    app::log_generation(
        &generation_log_config,
        &args.output,
        hex_seed,
        &drift_config,
        args.sims,
        &best_info,
        Some(&randomization_log),
    );

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
        assert_eq!(args.resolution, OutputResolution { width: 1920, height: 1080 });
        assert_eq!(args.drift, DriftModeArg::Elliptical);
        assert!(!args.fast_encode);
        assert_eq!(args.log_level, DEFAULT_LOG_LEVEL);
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
    fn test_reject_invalid_resolution() {
        let result = Args::try_parse_from(["three_body_problem", "--resolution", "wide-by-tall"]);
        assert!(result.is_err());
    }
}
