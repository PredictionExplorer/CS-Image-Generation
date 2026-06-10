//! Image-quality invariants for the `CosmicSignature` default profile.

use nalgebra::Vector3;
use three_body_problem::post_effects::{GaussianBloom, PostEffect};
use three_body_problem::render::effects::FinishEffectPipeline;
use three_body_problem::render::visual_profile::ResolvedVisualProfile;
use three_body_problem::render::{
    self, BloomMode, ChannelLevels, FinishOutputMode, RenderConfig, SpectralRenderSettings,
    SpectralScene, constants,
};
use three_body_problem::sim::Sha3RandomByteStream;

type Pixel = (f64, f64, f64, f64);

const WIDTH: usize = 33;
const HEIGHT: usize = 33;

fn make_rng(seed: &[u8]) -> Sha3RandomByteStream {
    Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0)
}

fn sharp_line_buffer() -> Vec<Pixel> {
    let mut buffer = vec![(0.0, 0.0, 0.0, 0.0); WIDTH * HEIGHT];
    for i in 4..(WIDTH - 4) {
        let idx = (HEIGHT / 2) * WIDTH + i;
        buffer[idx] = (0.9, 0.85, 1.0, 1.0);
    }
    buffer
}

fn off_line_energy(buffer: &[Pixel]) -> f64 {
    buffer
        .iter()
        .enumerate()
        .filter(|(idx, _)| idx / WIDTH != HEIGHT / 2)
        .map(|(_, &(r, g, b, _))| r + g + b)
        .sum()
}

fn laplacian_energy(buffer: &[Pixel]) -> f64 {
    let luma = |p: Pixel| 0.2126 * p.0 + 0.7152 * p.1 + 0.0722 * p.2;
    let mut total = 0.0;
    for y in 1..(HEIGHT - 1) {
        for x in 1..(WIDTH - 1) {
            let center = luma(buffer[y * WIDTH + x]) * 4.0;
            let neighbors = luma(buffer[y * WIDTH + x - 1])
                + luma(buffer[y * WIDTH + x + 1])
                + luma(buffer[(y - 1) * WIDTH + x])
                + luma(buffer[(y + 1) * WIDTH + x]);
            let response = center - neighbors;
            total += response * response;
        }
    }
    total
}

fn rendered_edge_score(image: &image::ImageBuffer<image::Rgb<u16>, Vec<u16>>) -> f64 {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let raw = image.as_raw();
    let mut edge = 0.0;
    let mut energy = 0.0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = (y * width + x) * 3;
            let luma = |i: usize| -> f64 {
                0.2126 * f64::from(raw[i])
                    + 0.7152 * f64::from(raw[i + 1])
                    + 0.0722 * f64::from(raw[i + 2])
            };
            let center = luma(idx);
            let left = luma(idx - 3);
            let right = luma(idx + 3);
            let up = luma(idx - width * 3);
            let down = luma(idx + width * 3);
            edge += ((center * 4.0) - left - right - up - down).abs();
            energy += center;
        }
    }

    edge / energy.max(1.0)
}

fn crisp_scene(
    step_count: usize,
) -> (Vec<Vec<Vector3<f64>>>, Vec<Vec<render::OklabColor>>, Vec<f64>) {
    let mut positions = vec![
        Vec::with_capacity(step_count),
        Vec::with_capacity(step_count),
        Vec::with_capacity(step_count),
    ];
    for step in 0..step_count {
        let t = step as f64 / (step_count - 1).max(1) as f64;
        positions[0].push(Vector3::new(-0.85 + 1.7 * t, -0.38, 0.0));
        positions[1].push(Vector3::new(-0.65 + 1.3 * t, 0.44, 0.0));
        positions[2].push(Vector3::new(
            -0.15 + 0.3 * (t * std::f64::consts::TAU).sin(),
            -0.08,
            0.0,
        ));
    }

    let colors = vec![
        vec![(0.72, 0.25, 0.02); step_count],
        vec![(0.72, -0.08, 0.24); step_count],
        vec![(0.72, -0.16, -0.12); step_count],
    ];
    let alphas = vec![0.05, 0.05, 0.05];

    (positions, colors, alphas)
}

/// Find a seed whose profile resolves with no finish passes at all
/// (no halation, no rare traits), so the pipeline must be a strict no-op.
fn clean_profile(width: u32, height: u32) -> ResolvedVisualProfile {
    for seed in 0u8..=255 {
        let mut rng = make_rng(&[seed, 0x00, 0x33]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, width, height);
        let p = profile.parameters;
        if p.halation_strength == 0.0 && p.prism_strength == 0.0 && p.nebula_whisper_strength == 0.0
        {
            return profile;
        }
    }
    panic!("no finish-free seed found in 256 candidates");
}

#[test]
fn cosmic_signature_pipeline_preserves_sharp_line_without_haze() {
    let profile = clean_profile(WIDTH as u32, HEIGHT as u32);
    let render_config =
        RenderConfig { hdr_scale: profile.effect_config.hdr_scale, bloom_mode: BloomMode::Dog };
    let effect_config = render::build_effect_config_from_resolved(
        &profile.effect_config,
        &render_config,
        FinishOutputMode::Still,
    );
    let pipeline = FinishEffectPipeline::new(effect_config);

    let input = sharp_line_buffer();
    let output = pipeline
        .process_trajectory(
            input.clone(),
            WIDTH,
            HEIGHT,
            &render::effects::FrameParams { frame_number: 0, density: None },
        )
        .expect("cosmic signature pipeline should process");

    assert_eq!(output, input);
    assert_eq!(off_line_energy(&output), 0.0);
    assert_eq!(laplacian_energy(&output), laplacian_energy(&input));
}

#[test]
fn gaussian_bloom_fixture_demonstrates_haze_metric_sensitivity() {
    let input = sharp_line_buffer();
    let bloom = GaussianBloom::new(4, 0.7, 12.0);
    let output = bloom.process(&input, WIDTH, HEIGHT).expect("bloom should process");

    assert!(off_line_energy(&output) > off_line_energy(&input));
    assert!(
        laplacian_energy(&output) >= laplacian_energy(&input) * 0.8,
        "bloom should not destroy the base line in this fixture"
    );
}

#[test]
fn cosmic_signature_distinct_seeds_keep_no_effects_invariant() {
    for seed in [[0x01, 0x02], [0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
        let mut rng = make_rng(&seed);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
        // Only the curated signature finishes (halation plus the rare prism /
        // nebula-whisper traits) may ever be enabled; every other legacy
        // effect must stay off for all seeds.
        assert!(
            !profile.effect_config.any_effect_beyond_signature_traits_enabled(),
            "seed {seed:02X?} enabled a legacy effect outside the signature trait set"
        );
        assert_eq!(
            profile.effect_config.enable_bloom,
            profile.parameters.halation_strength > 0.0,
            "seed {seed:02X?} bloom flag must mirror the halation trait"
        );
        assert_eq!(
            profile.effect_config.enable_chromatic_bloom,
            profile.parameters.prism_strength > 0.0,
            "seed {seed:02X?} chromatic bloom flag must mirror the prism trait"
        );
        assert_eq!(
            profile.effect_config.nebula_strength, profile.parameters.nebula_whisper_strength,
            "seed {seed:02X?} nebula strength must mirror the whisper trait"
        );
    }
}

#[test]
fn cosmic_signature_crisp_mode_disables_all_softening_sources() {
    let mut rng = make_rng(&[0x5A, 0xA5]);
    let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 1024, 576);
    let config = profile.effect_config;

    assert!(
        !config.any_effect_beyond_signature_traits_enabled(),
        "post-effects outside the signature trait set must stay disabled"
    );
    assert_eq!(config.blur_strength, 0.0);
    assert_eq!(config.glow_strength, 0.0);
    assert_eq!(config.chromatic_bloom_strength, profile.parameters.prism_strength);
    assert_eq!(config.perceptual_blur_strength, 0.0);
    assert_eq!(constants::CRISP_DISPERSION_STRENGTH, 0.0);
    assert_eq!(constants::SPECTRAL_DISPERSION_STRENGTH, 0.0);
    assert_eq!(constants::SPECTRAL_DISPERSION_STRENGTH_BOOSTED, 0.0);
    const {
        assert!(constants::CRISP_LINE_MIN_THICKNESS >= 0.30);
        assert!(constants::CRISP_LINE_FALLOFF_EXPONENT <= 2.1);
        assert!(constants::CRISP_LINE_ENERGY_CUTOFF >= 0.004);
    }
    assert!((constants::crisp_line_resolution_scale(3456, 2234) - 1.0).abs() < 0.001);
    assert!(constants::crisp_line_resolution_scale(10_000, 6_460) > 1.0);
    assert_eq!(constants::crisp_line_interpolation_substeps(3456, 2234, 0.75), 1);
    assert_eq!(constants::crisp_line_interpolation_substeps(3456, 2234, 8.0), 1);
    assert!(constants::crisp_line_interpolation_substeps(3456, 2234, 50.0) <= 8);
    assert_eq!(constants::CRISP_LINE_SUBPIXEL_GRID, 2);
    assert!(
        constants::crisp_tiled_guard_rows(100_000, 64_640) > constants::HIGH_RES_TILE_GUARD_ROWS
    );
    assert_eq!(constants::SWEEP_BLOOM_RADIUS, 0);
    assert_eq!(constants::SWEEP_BLOOM_STRENGTH, 0.0);
    const { assert!(constants::SWEEP_GAUSSIAN_SIGMA <= 0.75) };
}

#[test]
fn full_spd_memory_estimate_scales_for_extreme_resolutions() {
    let normal = render::estimate_full_spd_bytes(1920, 1080);
    let extreme = render::estimate_full_spd_bytes(50_000, 28_125);

    assert!(extreme > normal);
    assert_eq!(normal, 1920_u128 * 1080_u128 * 64_u128 * 8_u128);
    assert_eq!(extreme, 50_000_u128 * 28_125_u128 * 64_u128 * 8_u128);
}

#[test]
fn crisp_render_edge_score_survives_resolution_scaling() {
    let (positions, colors, alphas) = crisp_scene(64);
    let levels = ChannelLevels::new(0.0, 0.004, 0.0, 0.004, 0.0, 0.004);
    let render_config = RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::None };

    let render_at = |width: u32, height: u32| {
        // A finish-free profile keeps this a pure crisp-line scaling test
        // (halation / rare traits are covered elsewhere).
        let profile = clean_profile(width, height);
        render::render_final_frame_spectral(
            SpectralScene::new(&positions, &colors, &alphas),
            &levels,
            SpectralRenderSettings::new(&profile.effect_config, &render_config, 0, false),
        )
        .expect("crisp fixture should render")
    };

    let low = render_at(96, 54);
    let high = render_at(192, 108);
    let low_score = rendered_edge_score(&low);
    let high_score = rendered_edge_score(&high);

    assert!(low_score > 0.02, "low-res edge score too soft: {low_score}");
    assert!(
        high_score > low_score * 0.35,
        "edge score collapsed when scaling: low={low_score} high={high_score}"
    );
}
