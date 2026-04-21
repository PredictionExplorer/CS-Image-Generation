//! CLI front-end for the three-body problem visualization generator.

use clap::{Parser, ValueEnum};
use rayon::ThreadPoolBuilder;
use three_body_problem::{
    app,
    error::{self, Result},
    render::{self, RenderConfig},
    sim::Sha3RandomByteStream,
    spectrum_simd,
};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

const DEFAULT_OUTPUT_NAME: &str = "output";
const DEFAULT_NUM_SIMS: usize = 100_000;
const DEFAULT_NUM_STEPS: usize = 1_000_000;
const MAX_NUM_SIMS: usize = 10_000_000;
const MAX_NUM_STEPS: usize = 100_000_000;
const DEFAULT_RESOLUTION: &str = "1920x1080";
const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_LOCATION: f64 = 300.0;
const DEFAULT_VELOCITY: f64 = 1.0;
const DEFAULT_MIN_MASS: f64 = 100.0;
const DEFAULT_MAX_MASS: f64 = 300.0;
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
    Spiral,
}

impl DriftModeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Brownian => "brownian",
            Self::Elliptical => "elliptical",
            Self::Spiral => "spiral",
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
        dog_sigma: Some(resolved.dog_sigma_scale * f64::from(min_dim)),
        dog_ratio: resolved.dog_ratio,
        hdr_mode: DEFAULT_HDR_MODE.to_string(),
        hdr_scale: render_config.hdr_scale,
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
        weights_randomized: borda_weights.was_randomized,
        art_style: Some(resolved.art_style.name().to_string()),
        nebula_palette: Some(resolved.nebula_palette.name().to_string()),
        grade_preset: Some(resolved.grade_preset.name().to_string()),
        hue_palette_mode: Some(resolved.hue_palette_mode.name().to_string()),
        bloom_mode_choice: Some(resolved.bloom_mode_choice.as_str().to_string()),
        drift_character: Some(resolved.drift_character.name().to_string()),
        framing_zoom: Some(resolved.framing_zoom),
        starfield_enabled: Some(resolved.enable_starfield),
        lens_flare_enabled: Some(resolved.enable_lens_flare),
    }
}

fn main() -> Result<()> {
    ThreadPoolBuilder::new()
        .stack_size(render::constants::THREAD_STACK_SIZE)
        .build_global()
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let args = Args::parse();

    setup_logging(&args.log_level);

    // Always pad the orbital bounding box to the target aspect ratio. This
    // preserves the orbit's natural 2D proportions (no asymmetric x/y
    // stretching) and keeps the trajectory centroid at the canvas center for
    // every output resolution -- the foundation of museum-quality framing.
    let enhancements = app::Enhancements { aspect_correction: true, ..Default::default() };
    spectrum_simd::SAT_BOOST_ENABLED
        .store(enhancements.sat_boost, std::sync::atomic::Ordering::Relaxed);
    render::ACES_TWEAK_ENABLED.store(enhancements.aces_tweak, std::sync::atomic::Ordering::Relaxed);
    render::drawing::DISPERSION_BOOST_ENABLED
        .store(enhancements.dispersion_boost, std::sync::atomic::Ordering::Relaxed);

    error::validation::validate_dimensions(args.resolution.width, args.resolution.height)?;

    let seed_bytes = app::parse_seed(&args.seed)?;
    let hex_seed = if args.seed.starts_with("0x") { &args.seed[2..] } else { &args.seed };

    let seed_dir = app::setup_seed_directory(&args.output)?;
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

    let borda_weights = resolve_borda_weights(args.chaos_weight, args.equil_weight, &mut rng);

    let (best_bodies, best_info) = app::run_borda_selection(
        &mut rng,
        args.sims,
        args.steps,
        borda_weights.chaos_weight,
        borda_weights.equil_weight,
        DEFAULT_ESCAPE_THRESHOLD,
    )?;

    let mut positions = app::simulate_best_orbit(best_bodies, args.steps);

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

    let (colors, body_alphas) = app::generate_colors_with_mode(
        &mut rng,
        args.steps,
        DEFAULT_ALPHA_DENOM,
        &enhancements,
        Some(resolved_effect_config.hue_palette_mode),
    );

    info!("   => Using OKLab color space for accumulation");
    info!("STAGE 4/7: Determining bounding box...");
    let render_ctx = render::context::RenderContext::new_with_framing(
        args.resolution.width,
        args.resolution.height,
        &positions,
        enhancements.aspect_correction,
        resolved_effect_config.framing_zoom,
    );
    let bbox = render_ctx.bounds();
    info!(
        "   => X: [{:.3}, {:.3}], Y: [{:.3}, {:.3}]",
        bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
    );

    let render_config = RenderConfig {
        hdr_scale: resolved_effect_config.hdr_scale,
        bloom_mode: resolved_effect_config.bloom_mode_choice,
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

    let output_png = format!("{seed_dir}/image.png");
    let output_vid = format!("{seed_dir}/video.mp4");

    let accum_spd = app::render_video(
        render::SpectralScene::new(&positions, &colors, &body_alphas),
        &levels,
        render::SpectralRenderSettings::new(
            &resolved_effect_config,
            &render_config,
            noise_seed,
            enhancements.aspect_correction,
        ),
        &output_vid,
        &output_png,
        args.fast_encode,
        true,
    )?;

    let spectral_dir = format!("{seed_dir}/spectral");
    let spectral_sweep_path = format!("{seed_dir}/spectral_sweep.mp4");

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
        &spectral_sweep_path,
        args.fast_encode,
    )?;

    info!(
        "Done! Best orbit => Weighted Borda = {:.3}\nHave a nice day!",
        best_info.total_score_weighted
    );

    let generation_log_config =
        build_generation_log_config(&args, &resolved_effect_config, &render_config, &borda_weights);
    if let Err(e) = app::log_generation(
        &generation_log_config,
        &args.output,
        hex_seed,
        &drift_config,
        args.sims,
        &best_info,
        Some(&randomization_log),
    ) {
        warn!("Generation logging failed (non-fatal): {e}");
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
        assert_eq!(args.resolution, OutputResolution { width: 1920, height: 1080 });
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
        assert!((0.2..=125.0).contains(&ratio), "ratio {ratio} outside [0.2, 125.0]");
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
        assert!((0.2..=125.0).contains(&ratio), "ratio {ratio} outside [0.2, 125.0]");
    }

    #[test]
    fn test_resolve_borda_weights_partial_equil_explicit() {
        let mut rng = Sha3RandomByteStream::new(&[0x42; 32], 100.0, 300.0, 300.0, 1.0);
        let w = resolve_borda_weights(None, Some(5.0), &mut rng);
        assert!(w.was_randomized);
        assert_eq!(w.equil_weight, 5.0);
        let ratio = w.equil_weight / w.chaos_weight;
        assert!((0.2..=125.0).contains(&ratio), "ratio {ratio} outside [0.2, 125.0]");
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
                (0.2..=125.0).contains(&ratio),
                "seed {seed_byte} produced ratio {ratio} outside [0.2, 125.0]"
            );
        }
    }

    #[test]
    fn test_reject_invalid_resolution() {
        let result = Args::try_parse_from(["three_body_problem", "--resolution", "wide-by-tall"]);
        assert!(result.is_err());
    }
}
