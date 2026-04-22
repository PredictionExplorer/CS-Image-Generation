//! Orbit quality metrics: energy, angular momentum, chaoticness, and equilateralness.

use crate::sim::{Body, G};
use crate::utils::fourier_transform;
use nalgebra::Vector3;

fn sample_std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }

    let mut mean = 0.0;
    let mut sum_sq = 0.0;
    for (index, value) in values.iter().copied().enumerate() {
        let count = (index + 1) as f64;
        let delta = value - mean;
        mean += delta / count;
        let delta2 = value - mean;
        sum_sq += delta * delta2;
    }

    (sum_sq / (values.len() as f64 - 1.0)).sqrt()
}

/// Total energy: kinetic + potential
#[must_use]
pub fn calculate_total_energy(bodies: &[Body]) -> f64 {
    let mut kin = 0.0;
    let mut pot = 0.0;
    for b in bodies {
        kin += crate::render::constants::KINETIC_ENERGY_FACTOR * b.mass * b.velocity.norm_squared();
    }
    let n = bodies.len(); // Cache length to avoid repeated calls
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (bodies[i].position - bodies[j].position).norm();
            if r > crate::utils::FLOAT_EPSILON {
                pot += -G * bodies[i].mass * bodies[j].mass / r;
            }
        }
    }
    kin + pot
}

/// Total angular momentum vector
#[must_use]
pub fn calculate_total_angular_momentum(bodies: &[Body]) -> Vector3<f64> {
    let mut total_l = Vector3::zeros();
    for b in bodies {
        total_l += b.mass * b.position.cross(&b.velocity);
    }
    total_l
}

/// Distance from each body to the centre-of-mass of the other two.
///
/// Returns `(r1, r2, r3)` where `r_i[step]` is the distance from body `i`
/// to the COM of the other two bodies at timestep `step`.
fn compute_com_distances(
    m1: f64,
    m2: f64,
    m3: f64,
    positions: &[Vec<Vector3<f64>>],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = positions[0].len();
    let mut r1 = vec![0.0; len];
    let mut r2 = vec![0.0; len];
    let mut r3 = vec![0.0; len];
    for i in 0..len {
        let p1 = positions[0][i];
        let p2 = positions[1][i];
        let p3 = positions[2][i];
        let cm1 = (m2 * p2 + m3 * p3) / (m2 + m3);
        let cm2 = (m1 * p1 + m3 * p3) / (m1 + m3);
        let cm3 = (m1 * p1 + m2 * p2) / (m1 + m2);
        r1[i] = (p1 - cm1).norm();
        r2[i] = (p2 - cm2).norm();
        r3[i] = (p3 - cm3).norm();
    }
    (r1, r2, r3)
}

/// Spectral entropy of an FFT magnitude sequence.
///
/// Computes the Shannon entropy of the normalised power spectrum over the
/// positive-frequency bins (DC excluded), divided by `ln(N_bins)`. Result
/// ∈ [0, 1]: `1.0` means power is spread perfectly uniformly across bins
/// (broadband, maximally chaotic); `0.0` means all power is in one bin
/// (pure tone, maximally regular).
///
/// We previously used Wiener spectral flatness (`geo_mean / arith_mean`) but
/// the geometric mean is dominated by the noise floor for FFT sizes in the
/// hundreds of thousands, collapsing the metric to ~10⁻⁹ for every realistic
/// orbit. Shannon entropy of the normalised PSD discriminates much better
/// across the typical 0.3–0.95 range.
fn spectral_entropy_from_magnitudes(mags: &[f64]) -> f64 {
    let n = mags.len();
    let half = n / 2;
    if half <= 1 {
        return 0.0;
    }
    let mut total_power = 0.0;
    for bin in mags.iter().take(half).skip(1) {
        total_power += bin * bin;
    }
    if total_power <= 1e-30 {
        return 0.0;
    }
    let inv_total = 1.0 / total_power;
    let mut entropy = 0.0;
    for bin in mags.iter().take(half).skip(1) {
        let p = bin * bin * inv_total;
        if p > 1e-30 {
            entropy -= p * p.ln();
        }
    }
    let max_entropy = ((half - 1) as f64).ln();
    if max_entropy < 1e-14 {
        return 0.0;
    }
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Permutation entropy (Bandt–Pompe) of a single time series, embedding
/// dimension `m = 4`, lag `τ = 1`.
///
/// Slides a 4-wide window across the series, converts each window into the
/// rank-order permutation of its values, counts pattern frequencies, and
/// returns the Shannon entropy divided by `ln(4!)`. Result ∈ [0, 1]: `1.0`
/// means all 24 ordinal patterns are equally likely (maximally chaotic),
/// `0.0` means the series is perfectly monotone.
fn single_series_permutation_entropy(series: &[f64]) -> f64 {
    const EMBED_DIM: usize = 4;
    const NUM_PATTERNS: usize = 24; // 4!
    let n = series.len();
    if n < EMBED_DIM {
        return 0.0;
    }
    // Encode each permutation of [0..4) as a u8 by packing 2-bit indices.
    // Max encoded value = 3 | (3<<2) | (3<<4) | (3<<6) = 255. Only 24 slots
    // ever populate (distinct indices), rest stay zero.
    let mut counts = [0_u32; 256];
    let mut total = 0_u32;
    let mut indices = [0_usize; EMBED_DIM];
    for i in 0..=n - EMBED_DIM {
        for (j, idx) in indices.iter_mut().enumerate() {
            *idx = j;
        }
        indices.sort_by(|&a, &b| series[i + a].total_cmp(&series[i + b]));
        let key = indices[0] | (indices[1] << 2) | (indices[2] << 4) | (indices[3] << 6);
        counts[key] = counts[key].saturating_add(1);
        total = total.saturating_add(1);
    }
    if total == 0 {
        return 0.0;
    }
    let total_f = f64::from(total);
    let mut entropy = 0.0_f64;
    for &c in &counts {
        if c == 0 {
            continue;
        }
        let p = f64::from(c) / total_f;
        entropy -= p * p.ln();
    }
    let max_entropy = (NUM_PATTERNS as f64).ln();
    if max_entropy < 1e-14 {
        return 0.0;
    }
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Spatial-extent stability across the trajectory.
///
/// Splits the trajectory into five equal windows along the time axis. In each
/// window, measures the largest instantaneous distance from any body to the
/// three-body centroid. Returns `min(extents) / max(extents)` — `1.0` means
/// the bodies stayed within a stable extent throughout (bounded), and values
/// approaching `0.0` mean one or more bodies drifted far away in some window
/// (slow escape, fly-off, or hierarchical separation).
///
/// Used as a hard viability gate so we never select orbits where a body flies
/// off and the final frame loses its composition.
#[must_use]
pub fn boundedness_score(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let len = positions[0].len();
    const NUM_WINDOWS: usize = 5;
    if len < NUM_WINDOWS * 4 {
        return 0.0;
    }
    let mut extents = [0.0_f64; NUM_WINDOWS];
    for (w, extent) in extents.iter_mut().enumerate() {
        let start = (w * len) / NUM_WINDOWS;
        let end = ((w + 1) * len) / NUM_WINDOWS;
        let mut max_d = 0.0_f64;
        let iter = positions[0][start..end]
            .iter()
            .zip(positions[1][start..end].iter())
            .zip(positions[2][start..end].iter());
        for ((&p0, &p1), &p2) in iter {
            let centroid = (p0 + p1 + p2) / 3.0;
            max_d = max_d.max((p0 - centroid).norm());
            max_d = max_d.max((p1 - centroid).norm());
            max_d = max_d.max((p2 - centroid).norm());
        }
        *extent = max_d;
    }
    let min_e = extents.iter().copied().fold(f64::INFINITY, f64::min);
    let max_e = extents.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_e < 1e-14 {
        return 0.0;
    }
    (min_e / max_e).clamp(0.0, 1.0)
}

/// Bundle of chaos metrics computed in a single pass over the trajectory.
///
/// All three metrics share the three FFTs over the body-to-COM distance
/// signals, so this is much cheaper than calling three separate pipelines.
#[derive(Clone, Copy, Debug)]
pub struct ChaosMetrics {
    /// Legacy FFT-magnitude std-dev (smaller => more chaotic).
    pub non_chaoticness: f64,
    /// Shannon entropy of the normalised power spectrum
    /// (higher => more broadband / more chaotic, ∈ [0, 1]).
    pub spectral_entropy: f64,
    /// Bandt–Pompe ordinal-pattern entropy (higher => more chaotic, ∈ [0, 1]).
    pub permutation_entropy: f64,
}

/// Compute all chaos metrics in one pass, sharing the three FFTs.
///
/// The trajectory's body-to-COM distance signals are Fourier-transformed
/// once; the transforms then feed into both the FFT-magnitude std-dev
/// (`non_chaoticness`) and the spectral-entropy metric. Permutation entropy
/// is computed directly on the time series without FFT.
#[must_use]
pub fn compute_chaos_metrics(
    m1: f64,
    m2: f64,
    m3: f64,
    positions: &[Vec<Vector3<f64>>],
) -> ChaosMetrics {
    let len = positions[0].len();
    if len < 4 {
        return ChaosMetrics {
            non_chaoticness: 0.0,
            spectral_entropy: 0.0,
            permutation_entropy: 0.0,
        };
    }
    let (r1, r2, r3) = compute_com_distances(m1, m2, m3, positions);

    let abs1: Vec<f64> = fourier_transform(&r1).iter().map(|c| c.norm()).collect();
    let abs2: Vec<f64> = fourier_transform(&r2).iter().map(|c| c.norm()).collect();
    let abs3: Vec<f64> = fourier_transform(&r3).iter().map(|c| c.norm()).collect();

    let non_chaoticness =
        (sample_std_dev(&abs1) + sample_std_dev(&abs2) + sample_std_dev(&abs3)) / 3.0;
    let spectral_entropy = (spectral_entropy_from_magnitudes(&abs1)
        + spectral_entropy_from_magnitudes(&abs2)
        + spectral_entropy_from_magnitudes(&abs3))
        / 3.0;
    let permutation_entropy = (single_series_permutation_entropy(&r1)
        + single_series_permutation_entropy(&r2)
        + single_series_permutation_entropy(&r3))
        / 3.0;

    ChaosMetrics { non_chaoticness, spectral_entropy, permutation_entropy }
}

/// Shannon entropy of the 2D dwell distribution across a coarse 32×32 grid
/// fitted to the trajectory's bounding box.
///
/// Returns a value in `[0, 1]`: `1.0` means body positions are perfectly
/// uniform across all grid cells (the trajectory fills the canvas);
/// `0.0` means all dwell concentrates in a single cell (a tight blob).
///
/// This is a direct visual-quality signal: orbits where bodies execute
/// tiny tight motions and rare excursions — Lagrangian-triangle rotations,
/// concentrated near-collisions, etc. — score very low because most of
/// the dwell is in a few cells. Genuinely chaotic dances that sweep the
/// frame score high.
#[must_use]
pub fn dwell_entropy_score(positions: &[Vec<Vector3<f64>>]) -> f64 {
    const GRID: usize = 32;
    if positions.is_empty() || positions[0].is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for body in positions {
        for p in body {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }
    let w = (max_x - min_x).max(1e-12);
    let h = (max_y - min_y).max(1e-12);

    let mut counts = vec![0_u64; GRID * GRID];
    let mut total = 0_u64;
    let grid_f = GRID as f64;
    let max_idx = (GRID - 1) as f64;
    for body in positions {
        for p in body {
            let fx = ((p.x - min_x) / w * grid_f).clamp(0.0, max_idx);
            let fy = ((p.y - min_y) / h * grid_f).clamp(0.0, max_idx);
            let gx = fx as usize;
            let gy = fy as usize;
            counts[gy * GRID + gx] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut entropy = 0.0_f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total_f;
            entropy -= p * p.ln();
        }
    }
    let max_entropy = ((GRID * GRID) as f64).ln();
    if max_entropy < 1e-14 {
        return 0.0;
    }
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Score how "equilateral" the 3-body triangle is over time
#[must_use]
pub fn equilateralness_score(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let n = positions[0].len();
    if n < 1 {
        return 0.0;
    }
    let mut sum = 0.0;
    for ((&p0, &p1), &p2) in
        positions[0].iter().zip(positions[1].iter()).zip(positions[2].iter()).take(n)
    {
        let l01 = (p0 - p1).norm();
        let l12 = (p1 - p2).norm();
        let l20 = (p2 - p0).norm();
        let mn = l01.min(l12).min(l20);
        if mn < 1e-14 {
            continue;
        }
        let mx = l01.max(l12).max(l20);
        sum += 1.0 / (mx / mn);
    }
    sum / (n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_std_dev_matches_known_value() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let expected = (5.0_f64 / 3.0).sqrt();
        let actual = sample_std_dev(&values);

        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_sample_std_dev_short_input_is_nan() {
        assert!(sample_std_dev(&[]).is_nan());
        assert!(sample_std_dev(&[42.0]).is_nan());
    }

    #[test]
    fn test_calculate_total_energy_stationary_bodies() {
        let bodies = vec![
            Body::new(1.0, Vector3::new(1.0, 0.0, 0.0), Vector3::zeros()),
            Body::new(1.0, Vector3::new(-1.0, 0.0, 0.0), Vector3::zeros()),
        ];
        let energy = calculate_total_energy(&bodies);
        assert!(
            energy < 0.0,
            "Stationary bodies should have negative total energy (pure potential)"
        );
    }

    #[test]
    fn test_calculate_total_energy_single_body() {
        let bodies = vec![Body::new(1.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0))];
        let energy = calculate_total_energy(&bodies);
        assert!(energy > 0.0, "Single moving body should have positive kinetic energy");
    }

    #[test]
    fn test_angular_momentum_zero_velocity() {
        let bodies = vec![
            Body::new(1.0, Vector3::new(1.0, 0.0, 0.0), Vector3::zeros()),
            Body::new(1.0, Vector3::new(0.0, 1.0, 0.0), Vector3::zeros()),
        ];
        let l = calculate_total_angular_momentum(&bodies);
        assert!(l.norm() < 1e-12, "Stationary bodies should have zero angular momentum");
    }

    #[test]
    fn test_angular_momentum_circular_orbit() {
        let bodies = vec![
            Body::new(1.0, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            Body::new(1.0, Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
        ];
        let l = calculate_total_angular_momentum(&bodies);
        assert!(l[2].abs() > 0.1, "Circular orbit should have non-zero z-angular momentum");
    }

    #[test]
    fn test_chaos_metrics_empty_returns_zero() {
        let positions = vec![vec![], vec![], vec![]];
        let m = compute_chaos_metrics(1.0, 1.0, 1.0, &positions);
        assert_eq!(m.non_chaoticness, 0.0);
        assert_eq!(m.spectral_entropy, 0.0);
        assert_eq!(m.permutation_entropy, 0.0);
    }

    #[test]
    fn test_chaos_metrics_finite_on_periodic_orbit() {
        let n: i32 = 256;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = f64::from(step) * 0.1;
                        let angle = t + f64::from(body) * std::f64::consts::TAU / 3.0;
                        Vector3::new(angle.cos(), angle.sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let m = compute_chaos_metrics(1.0, 1.0, 1.0, &positions);
        assert!(m.non_chaoticness.is_finite() && m.non_chaoticness >= 0.0);
        assert!((0.0..=1.0).contains(&m.spectral_entropy));
        assert!((0.0..=1.0).contains(&m.permutation_entropy));
    }

    #[test]
    fn test_equilateralness_equilateral_triangle() {
        let n = 50;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                let angle = f64::from(body) * std::f64::consts::TAU / 3.0;
                vec![Vector3::new(angle.cos(), angle.sin(), 0.0); n]
            })
            .collect();
        let score = equilateralness_score(&positions);
        assert!((score - 1.0).abs() < 1e-6, "Perfect equilateral should score ~1.0, got {score}");
    }

    #[test]
    fn test_equilateralness_degenerate() {
        let n = 10;
        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0); n],
            vec![Vector3::new(100.0, 0.0, 0.0); n],
            vec![Vector3::new(100.001, 0.0, 0.0); n],
        ];
        let score = equilateralness_score(&positions);
        assert!(score < 0.05, "Near-degenerate triangle should score near 0, got {score}");
    }

    #[test]
    fn test_equilateralness_empty_returns_zero() {
        let positions = vec![vec![], vec![], vec![]];
        assert_eq!(equilateralness_score(&positions), 0.0);
    }

    // --- New chaos metrics --------------------------------------------------

    fn periodic_positions(num_steps: i32) -> Vec<Vec<Vector3<f64>>> {
        // Asymmetric circular orbits: different radii and angular velocities
        // per body so the body-to-COM distance signals actually oscillate
        // (non-constant → peaked FFT → low spectral flatness). A perfectly
        // symmetric rotating triangle would collapse to constant distances.
        const RADIUS: [f64; 3] = [1.0, 1.4, 0.8];
        const OMEGA: [f64; 3] = [1.0, 1.3, 0.7];
        (0..3)
            .map(|body| {
                let idx = body as usize;
                (0..num_steps)
                    .map(|step| {
                        let t = f64::from(step) * 0.05;
                        let angle = OMEGA[idx] * t + f64::from(body) * std::f64::consts::TAU / 3.0;
                        Vector3::new(RADIUS[idx] * angle.cos(), RADIUS[idx] * angle.sin(), 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    fn pseudo_random_positions(num_steps: i32) -> Vec<Vec<Vector3<f64>>> {
        // Deterministic LCG — gives a broadband spectrum so we can confirm
        // the chaos metrics return values close to their noise-limit without
        // pulling a real RNG into the analysis tests.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut step = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let high = (state >> 33) as u32;
            f64::from(high) / f64::from(u32::MAX)
        };
        (0..3)
            .map(|_| {
                (0..num_steps)
                    .map(|_| {
                        let x = step() * 2.0 - 1.0;
                        let y = step() * 2.0 - 1.0;
                        let z = step() * 2.0 - 1.0;
                        Vector3::new(x, y, z)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn spectral_entropy_rewards_broadband_over_periodic() {
        let regular = periodic_positions(2048);
        let chaotic = pseudo_random_positions(2048);
        let se_r = compute_chaos_metrics(1.0, 1.0, 1.0, &regular).spectral_entropy;
        let se_c = compute_chaos_metrics(1.0, 1.0, 1.0, &chaotic).spectral_entropy;
        assert!(
            (0.0..=1.0).contains(&se_r) && (0.0..=1.0).contains(&se_c),
            "spectral entropy out of [0, 1]: regular={se_r}, chaotic={se_c}"
        );
        assert!(
            se_c > se_r,
            "chaotic spectrum ({se_c}) should have higher entropy than regular ({se_r})"
        );
    }

    #[test]
    fn permutation_entropy_rewards_disorder_over_periodic() {
        let regular = periodic_positions(2048);
        let chaotic = pseudo_random_positions(2048);
        let pe_r = compute_chaos_metrics(1.0, 1.0, 1.0, &regular).permutation_entropy;
        let pe_c = compute_chaos_metrics(1.0, 1.0, 1.0, &chaotic).permutation_entropy;
        assert!(
            (0.0..=1.0).contains(&pe_r) && (0.0..=1.0).contains(&pe_c),
            "permutation entropy out of [0, 1]: regular={pe_r}, chaotic={pe_c}"
        );
        assert!(
            pe_c > pe_r,
            "chaotic orbit ({pe_c}) should score higher than periodic orbit ({pe_r})"
        );
    }

    #[test]
    fn permutation_entropy_monotone_is_zero() {
        // A strictly monotone series only ever produces the identity pattern
        // (0, 1, 2, 3), so the entropy should be ~0.
        let n: i32 = 256;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = f64::from(step) + f64::from(body);
                        Vector3::new(t, 0.0, 0.0)
                    })
                    .collect()
            })
            .collect();
        let pe = compute_chaos_metrics(1.0, 1.0, 1.0, &positions).permutation_entropy;
        assert!(pe < 1e-9, "monotone series should have near-zero permutation entropy, got {pe}");
    }

    #[test]
    fn boundedness_of_static_triangle_is_one() {
        let n = 200;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                let angle = f64::from(body) * std::f64::consts::TAU / 3.0;
                vec![Vector3::new(angle.cos(), angle.sin(), 0.0); n]
            })
            .collect();
        let b = boundedness_score(&positions);
        assert!((b - 1.0).abs() < 1e-9, "static triangle should score 1.0, got {b}");
    }

    #[test]
    fn boundedness_penalises_drifting_body() {
        // Two bodies stay put; one drifts outward along +x.
        let n: i32 = 500;
        let cap = n as usize;
        let mut positions: Vec<Vec<Vector3<f64>>> =
            (0..3).map(|_| Vec::with_capacity(cap)).collect();
        for step in 0..n {
            let frac = f64::from(step) / f64::from(n);
            positions[0].push(Vector3::new(-1.0, 0.0, 0.0));
            positions[1].push(Vector3::new(0.0, 1.0, 0.0));
            positions[2].push(Vector3::new(1.0 + 20.0 * frac, 0.0, 0.0));
        }
        let b = boundedness_score(&positions);
        assert!(b < 0.4, "drifting body should push boundedness well below 0.4, got {b}");
    }

    #[test]
    fn boundedness_short_trajectory_is_zero() {
        let positions: Vec<Vec<Vector3<f64>>> = vec![vec![Vector3::zeros(); 3]; 3];
        assert_eq!(boundedness_score(&positions), 0.0);
    }

    #[test]
    fn dwell_entropy_zero_for_static_orbit() {
        // All bodies sit at one location → all dwell in one cell → entropy 0.
        let n = 200;
        let positions: Vec<Vec<Vector3<f64>>> =
            (0..3).map(|_| vec![Vector3::new(1.0, 1.0, 0.0); n]).collect();
        let d = dwell_entropy_score(&positions);
        assert!(d < 1e-9, "static orbit should have ~0 dwell entropy, got {d}");
    }

    #[test]
    fn dwell_entropy_higher_for_distributed_than_clustered() {
        // Cluster: 90% of dwell at one point, 10% along a short arc.
        let n: i32 = 500;
        let cap = n as usize;
        let mut clustered: Vec<Vec<Vector3<f64>>> =
            (0..3).map(|_| Vec::with_capacity(cap)).collect();
        for step in 0..n {
            let frac = f64::from(step) / f64::from(n);
            let in_cluster = frac < 0.9;
            for (body, body_positions) in clustered.iter_mut().enumerate() {
                let p = if in_cluster {
                    Vector3::new(0.0, 0.0, 0.0)
                } else {
                    let theta = (frac - 0.9) * std::f64::consts::TAU * 4.0 + body as f64;
                    Vector3::new(theta.cos(), theta.sin(), 0.0)
                };
                body_positions.push(p);
            }
        }
        // Distributed: bodies sweep wide circles continuously.
        let distributed: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = f64::from(step) / f64::from(n) * std::f64::consts::TAU * 5.0;
                        let phase = f64::from(body) * std::f64::consts::TAU / 3.0;
                        let r = 1.0 + 0.3 * (t * 0.7).sin();
                        Vector3::new(r * (t + phase).cos(), r * (t + phase).sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let d_cluster = dwell_entropy_score(&clustered);
        let d_dist = dwell_entropy_score(&distributed);
        assert!(
            d_dist > d_cluster,
            "distributed dwell ({d_dist}) should beat clustered dwell ({d_cluster})"
        );
    }

    #[test]
    fn dwell_entropy_empty_returns_zero() {
        let positions: Vec<Vec<Vector3<f64>>> = vec![vec![], vec![], vec![]];
        assert_eq!(dwell_entropy_score(&positions), 0.0);
    }
}
