//! Benchmarks for post-processing effects.
//!
//! Measures individual effect throughput on representative image buffers.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use three_body_problem::post_effects::{
    EdgeLuminance, EdgeLuminanceConfig, GlowEnhancement, GlowEnhancementConfig, MicroContrast,
    MicroContrastConfig, PostEffect,
};

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn make_test_buffer() -> Vec<(f64, f64, f64, f64)> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let t = i as f64 / (WIDTH * HEIGHT) as f64;
            (t * 0.8, t * 0.6, t * 0.4, 1.0)
        })
        .collect()
}

fn bench_micro_contrast(c: &mut Criterion) {
    let buffer = make_test_buffer();
    let config = MicroContrastConfig {
        strength: 0.25,
        radius: 4,
        edge_threshold: 0.15,
        luminance_weight: 0.5,
    };
    let effect = MicroContrast::new(config);

    c.bench_function("post_effects/micro_contrast_512x512", |b| {
        b.iter(|| {
            let buf = buffer.clone();
            black_box(effect.process(&buf, WIDTH, HEIGHT)).expect("process should succeed");
        });
    });
}

fn bench_edge_luminance(c: &mut Criterion) {
    let buffer = make_test_buffer();
    let config = EdgeLuminanceConfig {
        strength: 0.2,
        threshold: 0.2,
        brightness_boost: 0.3,
        bright_edges_only: true,
        min_luminance: 0.1,
    };
    let effect = EdgeLuminance::new(config);

    c.bench_function("post_effects/edge_luminance_512x512", |b| {
        b.iter(|| {
            let buf = buffer.clone();
            black_box(effect.process(&buf, WIDTH, HEIGHT)).expect("process should succeed");
        });
    });
}

fn bench_glow_enhancement(c: &mut Criterion) {
    let buffer = make_test_buffer();
    let config = GlowEnhancementConfig {
        strength: 0.3,
        threshold: 0.65,
        radius: 8,
        sharpness: 2.5,
        saturation_boost: 0.25,
    };
    let effect = GlowEnhancement::new(config);

    c.bench_function("post_effects/glow_enhancement_512x512", |b| {
        b.iter(|| {
            let buf = buffer.clone();
            black_box(effect.process(&buf, WIDTH, HEIGHT)).expect("process should succeed");
        });
    });
}

criterion_group!(benches, bench_micro_contrast, bench_edge_luminance, bench_glow_enhancement,);
criterion_main!(benches);
