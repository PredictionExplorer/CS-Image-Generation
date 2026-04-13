//! Benchmarks for spectral power distribution to RGBA conversion.
//!
//! Measures throughput of the SIMD-accelerated SPD conversion which is the
//! innermost hot loop of the rendering pipeline.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use three_body_problem::spectrum::NUM_BINS;
use three_body_problem::spectrum_simd::spd_to_rgba_simd;

fn make_uniform_spd() -> [f64; NUM_BINS] {
    [0.5; NUM_BINS]
}

fn make_peaked_spd() -> [f64; NUM_BINS] {
    let mut spd = [0.0; NUM_BINS];
    for (i, val) in spd.iter_mut().enumerate() {
        let center = NUM_BINS / 2;
        let dist = (i as f64 - center as f64).abs();
        *val = (-dist * 0.1).exp();
    }
    spd
}

fn make_sparse_spd() -> [f64; NUM_BINS] {
    let mut spd = [0.0; NUM_BINS];
    for i in (0..NUM_BINS).step_by(8) {
        spd[i] = 0.8;
    }
    spd
}

fn bench_spd_to_rgba_uniform(c: &mut Criterion) {
    let spd = make_uniform_spd();
    c.bench_function("spd_to_rgba_simd/uniform", |b| {
        b.iter(|| black_box(spd_to_rgba_simd(black_box(&spd))));
    });
}

fn bench_spd_to_rgba_peaked(c: &mut Criterion) {
    let spd = make_peaked_spd();
    c.bench_function("spd_to_rgba_simd/peaked", |b| {
        b.iter(|| black_box(spd_to_rgba_simd(black_box(&spd))));
    });
}

fn bench_spd_to_rgba_sparse(c: &mut Criterion) {
    let spd = make_sparse_spd();
    c.bench_function("spd_to_rgba_simd/sparse", |b| {
        b.iter(|| black_box(spd_to_rgba_simd(black_box(&spd))));
    });
}

fn bench_spd_to_rgba_batch(c: &mut Criterion) {
    let spds: Vec<[f64; NUM_BINS]> = (0..1024)
        .map(|i| {
            let mut spd = [0.0; NUM_BINS];
            for (j, val) in spd.iter_mut().enumerate() {
                *val = ((i * 17 + j * 31) % 100) as f64 / 100.0;
            }
            spd
        })
        .collect();

    c.bench_function("spd_to_rgba_simd/batch_1024", |b| {
        b.iter(|| {
            for spd in &spds {
                black_box(spd_to_rgba_simd(black_box(spd)));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_spd_to_rgba_uniform,
    bench_spd_to_rgba_peaked,
    bench_spd_to_rgba_sparse,
    bench_spd_to_rgba_batch,
);
criterion_main!(benches);
