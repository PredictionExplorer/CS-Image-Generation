//! Image-quality invariants for the `CosmicSignature` default profile.

use three_body_problem::post_effects::{GaussianBloom, PostEffect};
use three_body_problem::render::effects::FinishEffectPipeline;
use three_body_problem::render::visual_profile::ResolvedVisualProfile;
use three_body_problem::render::{self, BloomMode, FinishOutputMode, RenderConfig};
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

#[test]
fn cosmic_signature_pipeline_preserves_sharp_line_without_haze() {
    let mut rng = make_rng(&[0x10, 0x00, 0x33]);
    let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, WIDTH as u32, HEIGHT as u32);
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
        assert!(
            !profile.effect_config.any_legacy_effect_enabled(),
            "seed {seed:02X?} enabled a legacy effect"
        );
    }
}
