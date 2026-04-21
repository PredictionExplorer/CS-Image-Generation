//! Shared helpers for integration and render-quality tests.
#![allow(dead_code)]

use std::path::Path;

use image::ImageBuffer;
use three_body_problem::app::{self, Enhancements};
use three_body_problem::render::randomizable_config::{RandomizableEffectConfig, ResolvedEffectConfig};
use three_body_problem::render::{
    self, BloomMode, RenderConfig, SpectralRenderSettings, SpectralScene,
};
use three_body_problem::sim::{self, Sha3RandomByteStream};

pub(crate) const MIN_MASS: f64 = 100.0;
pub(crate) const MAX_MASS: f64 = 300.0;
pub(crate) const LOCATION: f64 = 300.0;
pub(crate) const VELOCITY: f64 = 1.0;
pub(crate) const NUM_SIMS: usize = 18;
pub(crate) const NUM_STEPS: usize = 3_000;
pub(crate) const WIDTH: u32 = 80;
pub(crate) const HEIGHT: u32 = 45;

#[derive(Clone)]
pub(crate) struct RenderSample {
    pub(crate) image: ImageBuffer<image::Rgb<u16>, Vec<u16>>,
    pub(crate) pixels: Vec<u16>,
    pub(crate) resolved: ResolvedEffectConfig,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImageMetrics {
    pub(crate) mean_luma: f64,
    pub(crate) nonzero_fraction: f64,
    pub(crate) near_white_fraction: f64,
    pub(crate) contrast_span: f64,
    pub(crate) detail_energy: f64,
}

pub(crate) fn render_with_config(seed: &str, config: RandomizableEffectConfig) -> RenderSample {
    let seed_bytes = app::parse_seed(seed).expect("seed should parse");
    let mut rng = Sha3RandomByteStream::new(&seed_bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY);
    let (resolved, _) = config.resolve(&mut rng, WIDTH, HEIGHT);

    let (best_bodies, _) =
        sim::select_best_trajectory(&mut rng, NUM_SIMS, NUM_STEPS, 0.75, 11.0, -0.3)
            .expect("Borda search should find a valid orbit");
    let mut positions = app::simulate_best_orbit(best_bodies, NUM_STEPS);
    app::apply_drift_transformation(
        &mut positions,
        "elliptical",
        None,
        None,
        None,
        resolved.drift_scale_bias,
        resolved.drift_arc_bias,
        resolved.drift_eccentricity_bias,
        &mut rng,
    )
    .expect("drift should resolve");

    let enhancements = Enhancements::default();
    let (colors, body_alphas) = app::generate_colors(&mut rng, NUM_STEPS, 15_000_000, &enhancements);

    let render_config = RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };
    let levels = app::build_histogram_and_levels(
        &positions,
        &colors,
        &body_alphas,
        &resolved,
        &render_config,
        enhancements.aspect_correction,
    )
    .expect("histogram pass should succeed");

    let image = render::render_final_frame_spectral(
        SpectralScene::new(&positions, &colors, &body_alphas),
        &levels,
        SpectralRenderSettings::new(&resolved, &render_config, enhancements.aspect_correction),
    )
    .expect("final frame render should succeed");

    RenderSample { pixels: image.as_raw().clone(), image, resolved }
}

pub(crate) fn render_default(seed: &str) -> RenderSample {
    render_with_config(seed, RandomizableEffectConfig::default())
}

pub(crate) fn hash_u16_buffer(buf: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &value in buf {
        for byte in value.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

pub(crate) fn save_and_reload_pixels(
    image: &ImageBuffer<image::Rgb<u16>, Vec<u16>>,
    path: &Path,
) -> Vec<u16> {
    render::save_image_as_png_16bit(image, path.to_str().expect("path should be utf-8"))
        .expect("PNG save should succeed");
    image::open(path)
        .expect("saved PNG should reopen")
        .to_rgb16()
        .into_raw()
}

pub(crate) fn compute_metrics(pixels: &[u16], width: u32, height: u32) -> ImageMetrics {
    let pixel_count = (width as usize) * (height as usize);
    assert_eq!(pixels.len(), pixel_count * 3);
    let mut lumas = Vec::with_capacity(pixel_count);
    let mut nonzero = 0usize;
    let mut near_white = 0usize;

    for chunk in pixels.chunks_exact(3) {
        let r = f64::from(chunk[0]) / 65535.0;
        let g = f64::from(chunk[1]) / 65535.0;
        let b = f64::from(chunk[2]) / 65535.0;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        lumas.push(luma);
        if chunk[0] > 0 || chunk[1] > 0 || chunk[2] > 0 {
            nonzero += 1;
        }
        if r >= 0.985 && g >= 0.985 && b >= 0.985 {
            near_white += 1;
        }
    }

    let mean_luma = lumas.iter().sum::<f64>() / (lumas.len() as f64).max(1.0);
    let mut sorted_lumas = lumas.clone();
    sorted_lumas.sort_by(|a, b| a.partial_cmp(b).expect("finite luma"));
    let p05 = sorted_lumas[((sorted_lumas.len() as f64 * 0.05).floor() as usize)
        .min(sorted_lumas.len() - 1)];
    let p95 = sorted_lumas[((sorted_lumas.len() as f64 * 0.95).floor() as usize)
        .min(sorted_lumas.len() - 1)];

    let width = width as usize;
    let height = height as usize;
    let mut edge_sum = 0.0;
    let mut edge_count = 0usize;
    let luma_at = |x: usize, y: usize| -> f64 { lumas[y * width + x] };
    for y in 0..height {
        for x in 0..width {
            if x + 1 < width {
                edge_sum += (luma_at(x, y) - luma_at(x + 1, y)).abs();
                edge_count += 1;
            }
            if y + 1 < height {
                edge_sum += (luma_at(x, y) - luma_at(x, y + 1)).abs();
                edge_count += 1;
            }
        }
    }

    ImageMetrics {
        mean_luma,
        nonzero_fraction: nonzero as f64 / pixel_count as f64,
        near_white_fraction: near_white as f64 / pixel_count as f64,
        contrast_span: (p95 - p05).max(0.0),
        detail_energy: edge_sum / (edge_count as f64).max(1.0),
    }
}
