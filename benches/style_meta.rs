//! Benchmarks for new style-meta components and post-effects added in the
//! museum-quality overhaul.
//!
//! These benches measure throughput and latency of:
//! - `ArtStyle::pick` over a Sha3-seeded RNG.
//! - `Starfield::process` on a 1080p-class buffer.
//! - `LensFlare::process` on a 1080p-class buffer.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use three_body_problem::post_effects::{
    LensFlare, LensFlareConfig, PostEffect, Starfield, StarfieldConfig,
};
use three_body_problem::render::art_style::ArtStyle;
use three_body_problem::sim::Sha3RandomByteStream;

const WIDTH: usize = 960;
const HEIGHT: usize = 540;

fn make_dim_buffer() -> Vec<(f64, f64, f64, f64)> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let t = i as f64 / (WIDTH * HEIGHT) as f64;
            (t * 0.12, t * 0.08, t * 0.06, 0.2)
        })
        .collect()
}

fn make_flare_buffer() -> Vec<(f64, f64, f64, f64)> {
    let mut buf = vec![(0.02, 0.02, 0.03, 0.1); WIDTH * HEIGHT];
    let cx = WIDTH / 2;
    let cy = HEIGHT / 2;
    let r = 40isize;
    for y in (cy as isize - r)..=(cy as isize + r) {
        for x in (cx as isize - r)..=(cx as isize + r) {
            let dx = (x - cx as isize) as f64;
            let dy = (y - cy as isize) as f64;
            let d2 = dx * dx + dy * dy;
            if d2 <= (r * r) as f64 {
                let idx = y as usize * WIDTH + x as usize;
                let falloff = (-d2 / ((r * r) as f64 * 0.25)).exp();
                buf[idx] = (falloff * 1.4, falloff * 1.3, falloff * 1.1, 1.0);
            }
        }
    }
    buf
}

fn bench_art_style_pick(c: &mut Criterion) {
    let mut rng = Sha3RandomByteStream::new(&[0xA1, 0xB2, 0xC3, 0xD4], 100.0, 300.0, 300.0, 1.0);
    c.bench_function("style/art_style_pick", |b| {
        b.iter(|| {
            let s = ArtStyle::pick(&mut rng);
            black_box(s);
        });
    });
}

fn bench_starfield_1080p(c: &mut Criterion) {
    let buffer = make_dim_buffer();
    let cfg = StarfieldConfig {
        strength: 0.45,
        density: 350.0,
        max_radius: 1.8,
        seed: 0xC0FF_EE42,
        ..Default::default()
    };
    let effect = Starfield::new(cfg);

    c.bench_function("post_effects/starfield_960x540", |b| {
        b.iter(|| {
            black_box(effect.process(&buffer, WIDTH, HEIGHT)).expect("starfield should succeed");
        });
    });
}

fn bench_lens_flare_1080p(c: &mut Criterion) {
    let buffer = make_flare_buffer();
    let cfg = LensFlareConfig { strength: 0.6, ..Default::default() };
    let effect = LensFlare::new(cfg);

    c.bench_function("post_effects/lens_flare_960x540", |b| {
        b.iter(|| {
            black_box(effect.process(&buffer, WIDTH, HEIGHT)).expect("lens flare should succeed");
        });
    });
}

criterion_group!(benches, bench_art_style_pick, bench_starfield_1080p, bench_lens_flare_1080p,);
criterion_main!(benches);
