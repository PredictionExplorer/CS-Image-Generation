use clap::{Parser, ValueEnum};
use std::time::Instant;
use three_body_problem::{
    app,
    error::{self, Result},
    extra_outputs,
    perf::{PerformanceProfile, StageTimer},
    render::{self, RenderConfig},
    sim::Sha3RandomByteStream,
    spectrum_simd,
};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

const DEFAULT_OUTPUT_NAME: &str = "output";
const DEFAULT_NUM_SIMS: usize = 100_000;
const DEFAULT_NUM_STEPS: usize = 1_000_000;
const DEFAULT_RESOLUTION: &str = "1920x1080";
const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_LOCATION: f64 = 300.0;
const DEFAULT_VELOCITY: f64 = 1.0;
const DEFAULT_MIN_MASS: f64 = 1000.0;
const DEFAULT_MAX_MASS: f64 = 3000.0;
const DEFAULT_CHAOS_WEIGHT: f64 = 0.75;
const DEFAULT_EQUIL_WEIGHT: f64 = 11.0;
const DEFAULT_ALPHA_DENOM: usize = 15_000_000;
const DEFAULT_ALPHA_COMPRESS: f64 = 6.0;
const DEFAULT_ESCAPE_THRESHOLD: f64 = -3.0;
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

    #[arg(long, value_enum, default_value_t = DriftModeArg::None)]
    drift: DriftModeArg,

    #[arg(long, default_value_t = false)]
    fast_encode: bool,

    #[arg(long, default_value_t = false, help = "Also generate extra outputs (loop clip, genesis burst, phase portrait, avatar, wallpapers, stats card, reveal video, 3D model)")]
    extras: bool,

    #[arg(long, default_value_t = false, help = "Skip wallpaper pack when generating extras")]
    no_wallpapers: bool,

    #[arg(long, default_value_t = false, help = "Skip museum-grade 8K print renders (very slow)")]
    no_museum_prints: bool,

    #[arg(long, default_value_t = false, help = "Skip cinematic zoom video (renders 4x hi-res base frame)")]
    no_cinematic_zoom: bool,

    #[arg(long, default_value_t = false, help = "Disable the 3D perspective camera and use flat 2D orthographic projection")]
    no_camera_3d: bool,

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
    let pipeline_start = Instant::now();
    let args = Args::parse();

    setup_logging(&args.log_level);

    let mut profile = PerformanceProfile::new();

    let enhancements = app::Enhancements { camera_3d: !args.no_camera_3d, ..Default::default() };
    spectrum_simd::SAT_BOOST_ENABLED
        .store(enhancements.sat_boost, std::sync::atomic::Ordering::Relaxed);
    render::ACES_TWEAK_ENABLED.store(enhancements.aces_tweak, std::sync::atomic::Ordering::Relaxed);
    render::drawing::DISPERSION_BOOST_ENABLED
        .store(enhancements.dispersion_boost, std::sync::atomic::Ordering::Relaxed);

    let output_base = format!("output/{}", args.output);
    app::setup_output_directories(&output_base)?;
    error::validation::validate_dimensions(args.resolution.width, args.resolution.height)?;

    let seed_bytes = app::parse_seed(&args.seed)?;
    let hex_seed = if args.seed.starts_with("0x") { &args.seed[2..] } else { &args.seed };
    let mut rng = Sha3RandomByteStream::new(
        &seed_bytes,
        DEFAULT_MIN_MASS,
        DEFAULT_MAX_MASS,
        DEFAULT_LOCATION,
        DEFAULT_VELOCITY,
    );

    // ── Effect Resolution ──
    let timer = StageTimer::start("Effect Resolution");
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
    profile.push(timer.finish_with_throughput(Some(format!(
        "{} effects",
        randomization_log.effects.len()
    ))));

    // ── Borda Search ──
    let timer = StageTimer::start("Borda Search");
    let (best_bodies, best_info) = app::run_borda_selection(
        &mut rng,
        args.sims,
        args.steps,
        DEFAULT_CHAOS_WEIGHT,
        DEFAULT_EQUIL_WEIGHT,
        DEFAULT_ESCAPE_THRESHOLD,
    )?;
    profile.push(timer.finish_with_throughput(Some(format!(
        "{} candidates, {} steps",
        args.sims, args.steps
    ))));

    let body_masses = [best_bodies[0].mass, best_bodies[1].mass, best_bodies[2].mass];

    // ── Final Simulation ──
    let timer = StageTimer::start("Final Simulation");
    let (mut positions, velocities) = app::simulate_best_orbit(best_bodies, args.steps);
    profile.push(timer.finish_with_throughput(Some(format!("{} steps", args.steps))));

    // ── Drift Transform ──
    let timer = StageTimer::start("Drift Transform");
    let drift_config = if enhancements.camera_3d {
        if args.drift != DriftModeArg::None {
            warn!("STAGE 2.5/7: Drift skipped (incompatible with 3D camera)");
        } else {
            info!("STAGE 2.5/7: Drift disabled");
        }
        None
    } else if args.drift != DriftModeArg::None {
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
    profile.push(timer.finish_with_throughput(Some(
        if enhancements.camera_3d && args.drift != DriftModeArg::None {
            "skipped (3D camera)"
        } else {
            args.drift.as_str()
        },
    )));

    // ── 3D Camera Projection ──
    let projected_storage;
    let render_positions: &[Vec<nalgebra::Vector3<f64>>] = if enhancements.camera_3d {
        let timer = StageTimer::start("3D Camera Projection");
        let camera = render::camera::Camera3D::new(
            &render::camera::Camera3DConfig::default(),
            &positions,
        );
        projected_storage = camera.project_all_positions(&positions);
        profile.push(timer.finish_with_throughput(Some(format!("{} steps", args.steps))));
        &projected_storage
    } else {
        projected_storage = vec![];
        let _ = &projected_storage;
        &positions
    };

    // ── Color Generation ──
    let timer = StageTimer::start("Color Generation");
    let (colors, body_alphas) =
        app::generate_colors(&mut rng, args.steps, DEFAULT_ALPHA_DENOM, &enhancements);
    info!("   => Using OKLab color space for accumulation");
    profile.push(timer.finish_with_throughput(Some(format!("{} steps", args.steps))));

    // ── Bounding Box ──
    info!("STAGE 4/7: Determining bounding box...");
    let render_ctx = render::context::RenderContext::new(
        args.resolution.width,
        args.resolution.height,
        render_positions,
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

    // ── Histogram Pass ──
    let timer = StageTimer::start("Histogram Pass");
    let levels = app::build_histogram_and_levels(
        render_positions,
        &colors,
        &body_alphas,
        &resolved_effect_config,
        &render_config,
        enhancements.aspect_correction,
    )?;
    profile.push(timer.finish_with_throughput(Some(format!(
        "{}x{}",
        args.resolution.width, args.resolution.height
    ))));

    // ── Video Render ──
    let output_png = format!("{}/still.png", output_base);
    let output_vid = format!("{}/video.mp4", output_base);

    let mut spd_buffer: Option<Vec<[f64; render::NUM_BINS]>> = None;
    let timer = StageTimer::start("Video Render");
    app::render_video(
        render::SpectralScene::new(render_positions, &colors, &body_alphas),
        &levels,
        render::SpectralRenderSettings::new(
            &resolved_effect_config,
            &render_config,
            enhancements.aspect_correction,
        ),
        &output_vid,
        &output_png,
        args.fast_encode,
        true,
        &mut spd_buffer,
    )?;
    profile.push(timer.finish_with_throughput(Some(format!(
        "{} target frames at {}x{}",
        render::constants::DEFAULT_TARGET_FRAMES,
        args.resolution.width,
        args.resolution.height
    ))));

    // ── Sonification (gallery-grade with orbit character) ──
    let timer = StageTimer::start("Sonification");
    let audio_ctx = extra_outputs::sonification::GalleryAudioContext {
        chaos: best_info.chaos,
        equilateralness: best_info.equilateralness,
        masses: body_masses,
    };
    if let Err(e) = extra_outputs::sonification::generate_gallery_sonification(
        &positions, 30.0, &output_vid, &audio_ctx,
    ) {
        warn!("Sonification failed: {}", e);
    }
    profile.push(timer.finish());

    // ── Spectral Gallery (always runs, reuses video render buffer when available) ──
    let timer = StageTimer::start("Spectral Gallery");
    let spectral_dir = format!("{}/spectral", output_base);
    let gallery_result = if let Some(spd) = spd_buffer.take() {
        extra_outputs::spectral_gallery::render_spectral_gallery_from_buffer(
            spd,
            args.resolution.width,
            args.resolution.height,
            &spectral_dir,
        )
    } else {
        extra_outputs::spectral_gallery::render_spectral_gallery(
            render::SpectralScene::new(render_positions, &colors, &body_alphas),
            render::SpectralRenderSettings::new(&resolved_effect_config, &render_config, false),
            &spectral_dir,
        )
    };
    if let Err(e) = gallery_result {
        warn!("Spectral gallery failed: {}", e);
    }
    profile.push(timer.finish());

    // ── Extra Outputs ──
    if args.extras {
        app::setup_extras_directories(&output_base)?;
        let timer = StageTimer::start("Extra Outputs");
        let extra_timings = generate_extras(
            &positions,
            &velocities,
            &colors,
            &body_alphas,
            &levels,
            &resolved_effect_config,
            &render_config,
            &args,
            hex_seed,
            &best_info,
            &output_base,
        );
        let mut extras_stage = timer.finish_with_throughput(Some(format!(
            "{} tasks",
            extra_timings.len()
        )));
        extras_stage.sub_stages = Some(extra_timings);
        profile.push(extras_stage);
    }

    // ── Finalize ──
    profile.set_total(pipeline_start.elapsed().as_secs_f64());
    profile.log_summary();

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
        Some(profile),
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_extras(
    positions: &[Vec<nalgebra::Vector3<f64>>],
    velocities: &[Vec<nalgebra::Vector3<f64>>],
    colors: &[Vec<render::OklabColor>],
    body_alphas: &[f64],
    levels: &render::ChannelLevels,
    resolved: &render::randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    args: &Args,
    seed: &str,
    best_info: &three_body_problem::sim::TrajectoryResult,
    base: &str,
) -> Vec<three_body_problem::perf::StageTiming> {
    use std::sync::Mutex;
    use three_body_problem::perf::StageTiming;

    info!("=== Generating extra outputs (parallel) ===");

    let scene = render::SpectralScene::new(positions, colors, body_alphas);
    let settings =
        render::SpectralRenderSettings::new(resolved, render_config, false);

    let fast_encode = args.fast_encode;
    let no_wallpapers = args.no_wallpapers;
    let no_museum_prints = args.no_museum_prints;
    let no_cinematic_zoom = args.no_cinematic_zoom;
    let num_sims = args.sims;
    let num_steps = args.steps;
    let output_name = &args.output;

    let timings: Mutex<Vec<StageTiming>> = Mutex::new(Vec::new());

    macro_rules! timed_spawn {
        ($scope:ident, $name:expr, $body:expr) => {
            $scope.spawn(|_| {
                let t = StageTimer::start($name);
                let result = $body;
                timings.lock().unwrap().push(t.finish());
                result
            });
        };
    }

    rayon::scope(|s| {
        timed_spawn!(s, "genesis_burst", {
            if let Err(e) = extra_outputs::genesis_burst::render_genesis_burst(
                scene,
                levels,
                settings,
                &format!("{base}/stills/genesis.png"),
            ) {
                warn!("Genesis burst failed: {}", e);
            }
        });

        timed_spawn!(s, "loop_clip", {
            if let Err(e) = extra_outputs::loop_clip::render_loop_clip(
                scene,
                levels,
                settings,
                &format!("{base}/videos/loop.mp4"),
                Some(&format!("{base}/videos/loop.webm")),
                fast_encode,
            ) {
                warn!("Loop clip failed: {}", e);
            }
        });

        timed_spawn!(s, "phase_portrait", {
            if let Err(e) = extra_outputs::phase_portrait::render_phase_portrait(
                positions,
                velocities,
                colors,
                body_alphas,
                levels,
                settings,
                &format!("{base}/stills/phase.png"),
            ) {
                warn!("Phase portrait failed: {}", e);
            }
        });


        timed_spawn!(s, "avatar", {
            if let Err(e) = extra_outputs::avatar::render_animated_avatar(
                scene,
                levels,
                settings,
                &format!("{base}/stills/avatar.webp"),
            ) {
                warn!("Animated avatar failed: {}", e);
            }
        });

        if !no_wallpapers {
            timed_spawn!(s, "wallpaper_pack", {
                if let Err(e) = extra_outputs::wallpaper::render_wallpaper_pack(
                    scene,
                    levels,
                    settings,
                    &format!("{base}/wallpapers"),
                ) {
                    warn!("Wallpaper pack failed: {}", e);
                }
            });
        }

        timed_spawn!(s, "stats_card", {
            if let Err(e) = extra_outputs::stats_card::generate_stats_card(
                &extra_outputs::stats_card::StatsCardData {
                    seed,
                    result: best_info,
                    config: resolved,
                    num_sims,
                    num_steps,
                },
                &format!("{base}/data/stats.svg"),
            ) {
                warn!("Stats card failed: {}", e);
            }
        });

        timed_spawn!(s, "reveal_video", {
            if let Err(e) = extra_outputs::reveal_video::render_reveal_video(
                scene,
                levels,
                settings,
                &format!("{base}/videos/reveal.mp4"),
                fast_encode,
            ) {
                warn!("Reveal video failed: {}", e);
            }
        });

        timed_spawn!(s, "3d_and_ar_export", {
            if let Err(e) = extra_outputs::gltf_export::export_gltf(
                positions,
                colors,
                &format!("{base}/3d/model.glb"),
            ) {
                warn!("3D export failed: {}", e);
            }

            if let Err(e) = extra_outputs::ar_export::export_usdz(
                &format!("{base}/3d/model.glb"),
                &format!("{base}/3d/model.usdz"),
            ) {
                warn!("AR export failed: {}", e);
            }
        });

        timed_spawn!(s, "spectral_fingerprint", {
            if let Err(e) = extra_outputs::spectral_fingerprint::generate_spectral_fingerprint(
                scene,
                settings,
                seed,
                &format!("{base}/data/fingerprint.svg"),
            ) {
                warn!("Spectral fingerprint failed: {}", e);
            }
        });

        timed_spawn!(s, "color_palette", {
            if let Err(e) = extra_outputs::color_palette::generate_color_palette(
                scene,
                levels,
                settings,
                seed,
                &format!("{base}/data/palette.svg"),
            ) {
                warn!("Color palette failed: {}", e);
            }
        });

        timed_spawn!(s, "light_variant", {
            if let Err(e) = extra_outputs::gallery_variants::render_light_variant(
                scene,
                levels,
                settings,
                &format!("{base}/stills/light.png"),
            ) {
                warn!("Light variant failed: {}", e);
            }
        });

        timed_spawn!(s, "timelapse", {
            if let Err(e) = extra_outputs::timelapse::render_timelapse(
                scene,
                levels,
                settings,
                &format!("{base}/videos/timelapse.mp4"),
                fast_encode,
            ) {
                warn!("Timelapse failed: {}", e);
            }
        });

        timed_spawn!(s, "cinemagraph", {
            if let Err(e) = extra_outputs::cinemagraph::render_cinemagraph(
                scene,
                levels,
                settings,
                &format!("{base}/videos/living.webm"),
                fast_encode,
            ) {
                warn!("Cinemagraph failed: {}", e);
            }
        });

        timed_spawn!(s, "social_kit", {
            if let Err(e) = extra_outputs::social_kit::generate_social_kit(
                scene,
                levels,
                settings,
                &format!("{base}/social"),
            ) {
                warn!("Social media kit failed: {}", e);
            }
        });

        timed_spawn!(s, "rarity_report", {
            if let Err(e) = extra_outputs::rarity_report::generate_rarity_report(
                &extra_outputs::rarity_report::RarityReportData {
                    seed,
                    result: best_info,
                    config: resolved,
                    num_sims,
                },
                &format!("{base}/data/rarity.svg"),
            ) {
                warn!("Rarity report failed: {}", e);
            }
        });

        timed_spawn!(s, "orbit_comparison", {
            if let Err(e) = extra_outputs::orbit_comparison::render_orbit_comparison(
                scene,
                levels,
                settings,
                best_info,
                seed,
                &format!("{base}/stills/comparison.png"),
            ) {
                warn!("Orbit comparison failed: {}", e);
            }
        });

        timed_spawn!(s, "exhibition_page", {
            if let Err(e) = extra_outputs::exhibition_page::generate_exhibition_page(
                &extra_outputs::exhibition_page::ExhibitionPageData {
                    seed,
                    output_name,
                    result: best_info,
                    config: resolved,
                    num_sims,
                    num_steps,
                },
                &format!("{base}/web/exhibition.html"),
            ) {
                warn!("Exhibition page failed: {}", e);
            }
        });

        timed_spawn!(s, "dossier", {
            if let Err(e) = extra_outputs::dossier::generate_dossier(
                &extra_outputs::dossier::DossierData {
                    seed,
                    output_name,
                    result: best_info,
                    config: resolved,
                    num_sims,
                    num_steps,
                },
                &format!("{base}/web"),
            ) {
                warn!("Dossier failed: {}", e);
            }
        });

        if !no_museum_prints {
            timed_spawn!(s, "museum_prints", {
                if let Err(e) = extra_outputs::museum_print::render_museum_prints(
                    scene,
                    levels,
                    settings,
                    &format!("{base}/print"),
                ) {
                    warn!("Museum prints failed: {}", e);
                }
            });
        }

        if !no_cinematic_zoom {
            timed_spawn!(s, "cinematic_zoom", {
                if let Err(e) = extra_outputs::cinematic_zoom::render_cinematic_zoom(
                    scene,
                    levels,
                    settings,
                    &format!("{base}/videos/zoom.mp4"),
                    fast_encode,
                ) {
                    warn!("Cinematic zoom failed: {}", e);
                }
            });
        }
    });

    info!("=== Extra outputs complete ===");

    let mut result = timings.into_inner().unwrap();
    result.sort_by(|a, b| b.wall_clock_secs.partial_cmp(&a.wall_clock_secs).unwrap());
    result
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
        assert!(!args.extras);
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

    #[test]
    fn test_extras_disabled_by_default() {
        let args = Args::parse_from(["three_body_problem"]);
        assert!(!args.extras, "extras should be off by default");
    }

    #[test]
    fn test_extras_flag_enables_extras() {
        let args = Args::parse_from(["three_body_problem", "--extras"]);
        assert!(args.extras, "--extras should enable extras");
    }

    #[test]
    fn test_extras_combined_with_other_flags() {
        let args = Args::parse_from([
            "three_body_problem",
            "--extras",
            "--fast-encode",
            "--no-wallpapers",
        ]);
        assert!(args.extras);
        assert!(args.fast_encode);
        assert!(args.no_wallpapers);
    }
}
