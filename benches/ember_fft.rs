//! Benchmarks of the ember edition's deterministic real two-dimensional FFT.
//!
//! One iteration is a forward plus an inverse transform of a pseudo-random field on the
//! production fluid grid (1440×1024, the default for a 3456×2234 output) and on the half-size
//! grid. The fluid solver runs 26 such transforms per time step (23 of them band-limited).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use three_body_problem::ember::fft_bench::RoundTrip;

fn bench_rfft2d_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("ember_rfft2d");
    group.sample_size(30);
    for (nx, ny) in [(1440, 1024), (720, 512)] {
        let mut round_trip = RoundTrip::new(nx, ny).expect("5-smooth even sizes");
        group.bench_function(format!("forward_inverse_{nx}x{ny}"), |b| {
            b.iter(|| black_box(round_trip.run()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rfft2d_round_trip);
criterion_main!(benches);
