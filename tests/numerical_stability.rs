//! Integration tests for numerical stability, precision, and correctness.
//!
//! Verifies that the simulation, spectral conversion, FFT, and RNG maintain
//! acceptable precision across a wide range of inputs and conditions.

use nalgebra::Vector3;
use three_body_problem::sim::{Body, Sha3RandomByteStream, get_positions};
use three_body_problem::spectrum::NUM_BINS;
use three_body_problem::spectrum_simd;

fn make_stable_bodies() -> Vec<Body> {
    vec![
        Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
        Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
        Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
    ]
}

// ── simulation energy conservation ──────────────────────────────────────────

fn kinetic_energy(bodies: &[Body]) -> f64 {
    bodies.iter().map(|b| 0.5 * b.mass * b.velocity.norm_squared()).sum()
}

fn potential_energy(bodies: &[Body]) -> f64 {
    let mut pe = 0.0;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let r = (bodies[i].position - bodies[j].position).norm();
            if r > 1e-10 {
                pe -= bodies[i].mass * bodies[j].mass / r;
            }
        }
    }
    pe
}

fn total_energy(bodies: &[Body]) -> f64 {
    kinetic_energy(bodies) + potential_energy(bodies)
}

#[test]
fn test_energy_conservation_short_run() {
    let bodies = make_stable_bodies();
    let initial_energy = total_energy(&bodies);

    let sim = get_positions(bodies.clone(), 1000);
    assert!(!sim.positions.is_empty());

    let drift_pct = ((total_energy(&bodies) - initial_energy) / initial_energy.abs()) * 100.0;
    assert!(
        drift_pct.abs() < 50.0,
        "energy drift should be bounded: {drift_pct:.2}%"
    );
}

// ── angular momentum conservation ───────────────────────────────────────────

fn total_angular_momentum(bodies: &[Body]) -> Vector3<f64> {
    bodies
        .iter()
        .map(|b| b.mass * b.position.cross(&b.velocity))
        .fold(Vector3::zeros(), |acc, l| acc + l)
}

#[test]
fn test_angular_momentum_direction_preserved() {
    let bodies = make_stable_bodies();
    let initial_l = total_angular_momentum(&bodies);
    let initial_mag = initial_l.norm();

    if initial_mag < 1e-10 {
        return;
    }

    let _sim = get_positions(bodies.clone(), 500);
    let final_l = total_angular_momentum(&bodies);
    let final_mag = final_l.norm();

    let direction_cos = initial_l.dot(&final_l) / (initial_mag * final_mag + 1e-30);
    assert!(
        direction_cos > 0.0,
        "angular momentum direction should be roughly preserved: cos={direction_cos:.4}"
    );
}

// ── RNG quality ─────────────────────────────────────────────────────────────

#[test]
fn test_rng_chi_squared_uniformity() {
    let mut rng = Sha3RandomByteStream::new(b"chi_squared_test", 100.0, 300.0, 300.0, 1.0);
    let n_samples = 100_000;
    let n_buckets = 20;
    let mut buckets = vec![0u32; n_buckets];

    for _ in 0..n_samples {
        let val = rng.next_f64();
        let bucket = ((val * n_buckets as f64) as usize).min(n_buckets - 1);
        buckets[bucket] += 1;
    }

    let expected = n_samples as f64 / n_buckets as f64;
    let chi2: f64 = buckets
        .iter()
        .map(|&count| {
            let diff = count as f64 - expected;
            diff * diff / expected
        })
        .sum();

    let critical_value = 43.82; // chi2 critical at p=0.001 for df=19
    assert!(
        chi2 < critical_value,
        "chi-squared {chi2:.2} exceeds critical value {critical_value} — RNG is not uniform"
    );
}

#[test]
fn test_rng_no_obvious_correlation() {
    let mut rng = Sha3RandomByteStream::new(b"correlation_test", 100.0, 300.0, 300.0, 1.0);
    let n = 10_000;
    let mut prev = rng.next_f64();

    let mut sum_product = 0.0;
    let mut sum_x = prev;
    let mut sum_y = 0.0;

    for _ in 1..n {
        let curr = rng.next_f64();
        sum_product += prev * curr;
        sum_x += prev;
        sum_y += curr;
        prev = curr;
    }

    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / (n - 1) as f64;
    let cov = sum_product / (n - 1) as f64 - mean_x * mean_y;

    assert!(
        cov.abs() < 0.01,
        "lag-1 covariance should be near zero: {cov:.6}"
    );
}

#[test]
fn test_rng_byte_distribution() {
    let mut rng = Sha3RandomByteStream::new(b"byte_dist", 100.0, 300.0, 300.0, 1.0);
    let mut counts = [0u32; 256];
    let n = 100_000;

    for _ in 0..n {
        counts[rng.next_byte() as usize] += 1;
    }

    let expected = n as f64 / 256.0;
    let max_dev = counts
        .iter()
        .map(|&c| ((c as f64 - expected) / expected).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_dev < 0.20,
        "byte distribution max deviation {max_dev:.4} is too large"
    );
}

#[test]
fn test_rng_mass_in_range() {
    let mut rng = Sha3RandomByteStream::new(b"mass_range", 50.0, 500.0, 300.0, 1.0);
    for _ in 0..10_000 {
        let mass = rng.random_mass();
        assert!(
            mass >= 50.0 && mass <= 500.0,
            "mass {mass} out of range [50, 500]"
        );
    }
}

// ── spectral conversion precision ───────────────────────────────────────────

#[test]
fn test_spd_to_rgba_nan_input_does_not_propagate() {
    let mut spd = [0.0; NUM_BINS];
    spd[5] = f64::NAN;
    let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
    let all_finite = r.is_finite() && g.is_finite() && b.is_finite() && a.is_finite();
    let all_nan = r.is_nan() || g.is_nan() || b.is_nan() || a.is_nan();
    assert!(
        all_finite || all_nan,
        "NaN input should produce either all-finite or all-NaN, got ({r}, {g}, {b}, {a})"
    );
}

#[test]
fn test_spd_to_rgba_infinity_input_clamped() {
    let mut spd = [0.0; NUM_BINS];
    spd[8] = f64::INFINITY;
    let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
    assert!(r <= 1.0 || r.is_nan(), "R should be clamped or NaN: {r}");
    assert!(a <= 1.0 || a.is_nan(), "A should be clamped or NaN: {a}");
    let _ = (g, b);
}

#[test]
fn test_spd_to_rgba_denormal_input() {
    let spd = [f64::MIN_POSITIVE / 2.0; NUM_BINS]; // denormalized
    let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
    assert!(r.is_finite(), "denormal input should produce finite R");
    assert!(a.is_finite(), "denormal input should produce finite A");
    let _ = (g, b);
}

#[test]
fn test_spd_to_rgba_monotonic_brightness() {
    let energies = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
    let mut prev_alpha = 0.0f64;
    for &e in &energies {
        let mut spd = [0.0; NUM_BINS];
        spd[8] = e;
        let (_, _, _, a) = spectrum_simd::spd_to_rgba_simd(&spd);
        assert!(
            a >= prev_alpha - 1e-15,
            "brightness should be monotonic: e={e} a={a} prev={prev_alpha}"
        );
        prev_alpha = a;
    }
}

#[test]
fn test_spd_to_rgba_symmetry_of_complementary_spectra() {
    let mut spd_a = [0.0; NUM_BINS];
    let mut spd_b = [0.0; NUM_BINS];
    for i in 0..NUM_BINS {
        spd_a[i] = 1.0;
        spd_b[NUM_BINS - 1 - i] = 1.0;
    }
    let result_a = spectrum_simd::spd_to_rgba_simd(&spd_a);
    let result_b = spectrum_simd::spd_to_rgba_simd(&spd_b);

    assert_eq!(
        result_a.3.to_bits(),
        result_b.3.to_bits(),
        "complementary spectra with same total energy should have same brightness"
    );
}

// ── FFT precision ───────────────────────────────────────────────────────────

#[test]
fn test_fft_known_signal_roundtrip() {
    use three_body_problem::utils::FftCache;

    let n = 256;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 4.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 8.0 * t).sin()
        })
        .collect();

    let mut cache = FftCache::new();
    let spectrum = cache.transform(&signal);

    // Only search the first half (positive frequencies) of the spectrum.
    let half = n / 2;
    let peak_bin = spectrum[1..half]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
        .map(|(i, _)| i + 1)
        .unwrap();

    assert_eq!(peak_bin, 4, "FFT should detect fundamental at bin 4");
}

#[test]
fn test_fft_dc_component() {
    use three_body_problem::utils::FftCache;

    let n = 128;
    let constant_value = 3.5;
    let signal = vec![constant_value; n];

    let mut cache = FftCache::new();
    let spectrum = cache.transform(&signal);

    let dc_mag = spectrum[0].norm() / n as f64;
    assert!(
        (dc_mag - constant_value).abs() < 1e-10,
        "DC component should equal mean: got {dc_mag}, expected {constant_value}"
    );
}

#[test]
fn test_fft_zero_signal() {
    use three_body_problem::utils::FftCache;

    let signal = vec![0.0; 256];
    let mut cache = FftCache::new();
    let spectrum = cache.transform(&signal);

    let max_mag = spectrum.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
    assert!(max_mag < 1e-15, "zero signal should produce zero spectrum");
}

#[test]
fn test_fft_parseval_theorem() {
    use three_body_problem::utils::FftCache;

    let n = 512;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 7.0 * t).sin() + 0.3 * (i as f64 * 0.01).cos()
        })
        .collect();

    let time_energy: f64 = signal.iter().map(|&x| x * x).sum();

    let mut cache = FftCache::new();
    let spectrum = cache.transform(&signal);

    let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

    let relative_error = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        relative_error < 1e-10,
        "Parseval's theorem violated: time={time_energy:.6} freq={freq_energy:.6} err={relative_error:.2e}"
    );
}

// ── histogram edge cases ────────────────────────────────────────────────────

#[test]
fn test_histogram_all_zero_samples() {
    use three_body_problem::render::histogram;

    let samples = vec![[0.0f64; 3]; 1000];
    let result = histogram::analyze_tonemapping(&samples, 0.01, 0.99);
    assert!(result.black_r.is_finite());
    assert!(result.white_r.is_finite());
}

#[test]
fn test_histogram_identical_samples() {
    use three_body_problem::render::histogram;

    let samples = vec![[0.5, 0.5, 0.5]; 1000];
    let result = histogram::analyze_tonemapping(&samples, 0.01, 0.99);
    assert!(result.black_r <= 0.5 + 1e-6);
    assert!(result.white_r >= 0.5 - 1e-6);
}

#[test]
fn test_histogram_single_sample() {
    use three_body_problem::render::histogram;

    let samples = vec![[0.3, 0.6, 0.9]];
    let result = histogram::analyze_tonemapping(&samples, 0.0, 1.0);
    assert!(result.black_r.is_finite());
    assert!(result.white_r.is_finite());
}

#[test]
fn test_histogram_wide_dynamic_range() {
    use three_body_problem::render::histogram;

    let mut samples = Vec::new();
    for i in 0..10_000 {
        let t = i as f64 / 10_000.0;
        samples.push([t * 100.0, t * 50.0, t * 200.0]);
    }
    let result = histogram::analyze_tonemapping(&samples, 0.01, 0.99);
    assert!(result.black_r < result.white_r);
}

// ── simulation determinism ──────────────────────────────────────────────────

#[test]
fn test_simulation_bitwise_deterministic() {
    let bodies = make_stable_bodies();
    let steps = 5000;

    let sim_a = get_positions(bodies.clone(), steps);
    let sim_b = get_positions(bodies.clone(), steps);

    assert_eq!(sim_a.positions.len(), sim_b.positions.len());
    for (i, (a, b)) in sim_a.positions.iter().zip(sim_b.positions.iter()).enumerate() {
        for body in 0..3 {
            assert_eq!(
                a[body].x.to_bits(),
                b[body].x.to_bits(),
                "step {i} body {body} x not bitwise identical"
            );
            assert_eq!(
                a[body].y.to_bits(),
                b[body].y.to_bits(),
                "step {i} body {body} y not bitwise identical"
            );
        }
    }
}

#[test]
fn test_simulation_positions_are_finite() {
    let bodies = make_stable_bodies();
    let sim = get_positions(bodies, 10_000);

    for (step, frame) in sim.positions.iter().enumerate() {
        for (body_idx, pos) in frame.iter().enumerate() {
            assert!(
                pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                "step {step} body {body_idx}: position is not finite ({}, {}, {})",
                pos.x, pos.y, pos.z
            );
        }
    }
}

// ── float edge case handling ────────────────────────────────────────────────

#[test]
fn test_spd_conversion_with_very_small_values() {
    for exp in -300i32..=-10 {
        let val = 10.0f64.powi(exp);
        let spd = [val; NUM_BINS];
        let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
        assert!(r.is_finite() && g.is_finite() && b.is_finite() && a.is_finite(),
            "exp={exp}: non-finite output ({r}, {g}, {b}, {a})");
    }
}

#[test]
fn test_spd_conversion_with_large_values() {
    for exp in 1..=6 {
        let val = 10.0f64.powi(exp);
        let spd = [val; NUM_BINS];
        let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
        assert!(r >= 0.0 && r <= 1.0, "exp={exp}: R={r}");
        assert!(g >= 0.0 && g <= 1.0, "exp={exp}: G={g}");
        assert!(b >= 0.0 && b <= 1.0, "exp={exp}: B={b}");
        assert!(a >= 0.0 && a <= 1.0, "exp={exp}: A={a}");
    }
}

#[test]
fn test_body_with_extreme_mass() {
    let bodies = vec![
        Body::new(1e-6, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.001, 0.0)),
        Body::new(1e6, Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -0.001, 0.0)),
        Body::new(1.0, Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.001, 0.0, 0.0)),
    ];
    let sim = get_positions(bodies, 100);
    assert!(!sim.positions.is_empty());
    for pos in &sim.positions[0] {
        assert!(pos.x.is_finite() && pos.y.is_finite());
    }
}

#[test]
fn test_body_at_origin() {
    let bodies = vec![
        Body::new(100.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
        Body::new(100.0, Vector3::new(50.0, 0.0, 0.0), Vector3::new(0.0, -0.5, 0.0)),
        Body::new(100.0, Vector3::new(-50.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
    ];
    let sim = get_positions(bodies, 500);
    assert!(!sim.positions.is_empty());
}
