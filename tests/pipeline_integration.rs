//! End-to-end smoke test for the public rendering pipeline.

use three_body_problem::app::{self, Enhancements};
use three_body_problem::render::randomizable_config::RandomizableEffectConfig;
use three_body_problem::render::{self, BloomMode, RenderConfig, SpectralRenderSettings, SpectralScene};
use three_body_problem::sim::{self, Sha3RandomByteStream};

const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;
const NUM_SIMS: usize = 20;
const NUM_STEPS: usize = 5_000;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 36;

fn render_final_pixels(seed: &str) -> Vec<u16> {
    let seed_bytes = app::parse_seed(seed).expect("seed should parse");
    let noise_seed = app::derive_noise_seed(&seed_bytes);
    let mut rng = Sha3RandomByteStream::new(&seed_bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY);

    let config = RandomizableEffectConfig {
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
    let (resolved, _) = config.resolve(&mut rng, WIDTH, HEIGHT);

    let (best_bodies, _) =
        sim::select_best_trajectory(&mut rng, NUM_SIMS, NUM_STEPS, 0.75, 11.0, -0.3)
            .expect("Borda search should find a valid orbit");
    let mut positions = app::simulate_best_orbit(best_bodies, NUM_STEPS);
    app::apply_drift_transformation(&mut positions, "elliptical", None, None, None, &mut rng)
        .expect("drift should resolve");

    let enhancements = Enhancements {
        chroma_boost: false,
        sat_boost: false,
        aces_tweak: false,
        alpha_variation: false,
        aspect_correction: false,
        dispersion_boost: false,
    };
    let (colors, body_alphas) = app::generate_colors(&mut rng, NUM_STEPS, 15_000_000, &enhancements);

    let render_config = RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };
    let levels = app::build_histogram_and_levels(
        &positions,
        &colors,
        &body_alphas,
        &resolved,
        noise_seed,
        &render_config,
        false,
    )
    .expect("histogram pass should succeed");

    render::render_final_frame_spectral(
        SpectralScene::new(&positions, &colors, &body_alphas),
        &levels,
        SpectralRenderSettings::new(&resolved, &render_config, noise_seed, false),
    )
    .expect("final frame render should succeed")
    .into_raw()
}

#[test]
fn problematic_seed_final_frame_is_not_black() {
    let pixels = render_final_pixels("0x870a93725533");
    assert!(pixels.iter().any(|&channel| channel > 0), "rendered frame should contain lit pixels");

    let non_zero_channels = pixels.iter().filter(|&&channel| channel > 0).count();
    assert!(
        non_zero_channels > pixels.len() / 200,
        "rendered frame should contain more than a handful of lit channels, got {non_zero_channels}/{}",
        pixels.len(),
    );
}
