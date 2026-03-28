//! Criterion benchmarks for the performance-critical paths.
//!
//! Run with: cargo bench
//! Results in: target/criterion/

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nalgebra::Vector3;
use three_body_problem::render::drawing::{LineVertex, SpectralLineSegment};
use three_body_problem::render::{self};
use three_body_problem::sim::{Body, Sha3RandomByteStream, get_positions};
use three_body_problem::spectrum::NUM_BINS;

fn make_bodies() -> Vec<Body> {
    vec![
        Body::new(
            200.0,
            Vector3::new(100.0, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.0),
        ),
        Body::new(
            150.0,
            Vector3::new(-50.0, 86.0, 0.0),
            Vector3::new(-0.3, -0.2, 0.0),
        ),
        Body::new(
            250.0,
            Vector3::new(-50.0, -86.0, 0.0),
            Vector3::new(0.3, -0.3, 0.0),
        ),
    ]
}

fn bench_simulation(c: &mut Criterion) {
    let bodies = make_bodies();

    c.bench_function("get_positions/1000_steps", |b| {
        b.iter(|| get_positions(black_box(bodies.clone()), 1000))
    });

    c.bench_function("get_positions/5000_steps", |b| {
        b.iter(|| get_positions(black_box(bodies.clone()), 5000))
    });
}

fn bench_spd_to_rgba(c: &mut Criterion) {
    use three_body_problem::spectrum::spd_to_rgba;

    let spd = [0.1, 0.3, 0.6, 0.9, 1.2, 0.8, 0.5, 0.3, 0.1, 0.4, 0.7, 0.9, 0.6, 0.2, 0.05, 0.0];

    c.bench_function("spd_to_rgba/single", |b| {
        b.iter(|| spd_to_rgba(black_box(&spd)))
    });

    let batch: Vec<[f64; NUM_BINS]> = (0..10000)
        .map(|i| {
            let scale = (i as f64) / 10000.0;
            let mut s = [0.0; 16];
            for (j, v) in s.iter_mut().enumerate() {
                *v = scale * (j as f64 + 1.0) / 16.0;
            }
            s
        })
        .collect();

    c.bench_function("spd_to_rgba/10k_batch", |b| {
        b.iter(|| {
            let mut total = (0.0, 0.0, 0.0, 0.0);
            for spd in &batch {
                let r = spd_to_rgba(black_box(spd));
                total.0 += r.0;
                total.1 += r.1;
                total.2 += r.2;
                total.3 += r.3;
            }
            total
        })
    });
}

fn bench_line_segment(c: &mut Criterion) {
    let segment = SpectralLineSegment {
        start: LineVertex {
            x: 10.0,
            y: 10.0,
            z: 0.0,
            color: (0.7, 0.15, 0.08),
            alpha: 0.85,
        },
        end: LineVertex {
            x: 200.0,
            y: 150.0,
            z: -0.3,
            color: (0.65, -0.12, 0.14),
            alpha: 0.55,
        },
        hdr_scale: 2.0,
    };

    let width = 320;
    let height = 240;
    let mut accum = vec![[0.0f64; NUM_BINS]; width * height];

    c.bench_function("draw_line_segment/320x240", |b| {
        b.iter(|| {
            for pixel in accum.iter_mut() {
                *pixel = [0.0; NUM_BINS];
            }
            render::drawing::draw_line_segment_aa_spectral(
                black_box(&mut accum),
                width as u32,
                height as u32,
                segment,
            )
        })
    });
}

fn bench_blur(c: &mut Criterion) {
    let width = 320;
    let height = 240;
    let mut buffer: Vec<(f64, f64, f64, f64)> = (0..width * height)
        .map(|i| {
            let t = i as f64 / (width * height) as f64;
            (t * 0.8, (1.0 - t) * 0.6, 0.3 + t * 0.2, 0.9)
        })
        .collect();

    c.bench_function("parallel_blur_2d/r=3/320x240", |b| {
        b.iter(|| {
            render::drawing::parallel_blur_2d_rgba(
                black_box(&mut buffer),
                width,
                height,
                3,
            )
        })
    });
}

fn bench_rng(c: &mut Criterion) {
    c.bench_function("sha3_rng/1000_f64", |b| {
        b.iter(|| {
            let mut rng =
                Sha3RandomByteStream::new(&[0x10, 0x00, 0x33], 100.0, 300.0, 300.0, 1.0);
            let mut sum = 0.0;
            for _ in 0..1000 {
                sum += rng.next_f64();
            }
            sum
        })
    });
}

fn bench_histogram_analysis(c: &mut Criterion) {
    use three_body_problem::render::histogram;

    let samples: Vec<[f64; 3]> = (0..100_000)
        .map(|i| {
            let t = i as f64 / 100_000.0;
            [t * 0.8, (1.0 - t) * 0.6, 0.3 + t * 0.4]
        })
        .collect();

    c.bench_function("histogram_analysis/100k_samples", |b| {
        b.iter(|| histogram::analyze_tonemapping(black_box(&samples), 0.01, 0.99))
    });
}

fn bench_simd_dispatch(c: &mut Criterion) {
    use three_body_problem::spectrum_simd;

    let spd = [0.1, 0.3, 0.6, 0.9, 1.2, 0.8, 0.5, 0.3, 0.1, 0.4, 0.7, 0.9, 0.6, 0.2, 0.05, 0.0];

    c.bench_function("simd_dispatch/single", |b| {
        b.iter(|| spectrum_simd::spd_to_rgba_simd(black_box(&spd)))
    });

    c.bench_function("simd_dispatch/path_name", |b| {
        b.iter(|| spectrum_simd::detected_simd_path_name())
    });
}

fn bench_parallel_spd_conversion(c: &mut Criterion) {
    use rayon::prelude::*;
    use three_body_problem::spectrum::spd_to_rgba;

    let batch: Vec<[f64; NUM_BINS]> = (0..10000)
        .map(|i| {
            let scale = (i as f64) / 10000.0;
            let mut s = [0.0; 16];
            for (j, v) in s.iter_mut().enumerate() {
                *v = scale * (j as f64 + 1.0) / 16.0;
            }
            s
        })
        .collect();

    c.bench_function("spd_to_rgba/10k_sequential", |b| {
        b.iter(|| {
            let total: (f64, f64, f64, f64) = batch.iter().fold((0.0, 0.0, 0.0, 0.0), |acc, spd| {
                let r = spd_to_rgba(black_box(spd));
                (acc.0 + r.0, acc.1 + r.1, acc.2 + r.2, acc.3 + r.3)
            });
            total
        })
    });

    c.bench_function("spd_to_rgba/10k_parallel", |b| {
        b.iter(|| {
            batch
                .par_iter()
                .map(|spd| spd_to_rgba(black_box(spd)))
                .reduce(|| (0.0, 0.0, 0.0, 0.0), |a, b| {
                    (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
                })
        })
    });
}

fn bench_u16_as_bytes(c: &mut Criterion) {
    use three_body_problem::utils::u16_slice_as_bytes;

    let frame: Vec<u16> = (0..1920 * 1080 * 3).map(|i| (i % 65536) as u16).collect();

    c.bench_function("u16_slice_as_bytes/1080p_frame", |b| {
        b.iter(|| {
            let bytes = u16_slice_as_bytes(black_box(&frame));
            black_box(bytes.len())
        })
    });
}

criterion_group!(
    benches,
    bench_simulation,
    bench_spd_to_rgba,
    bench_line_segment,
    bench_blur,
    bench_rng,
    bench_histogram_analysis,
    bench_simd_dispatch,
    bench_parallel_spd_conversion,
    bench_u16_as_bytes,
);
criterion_main!(benches);
