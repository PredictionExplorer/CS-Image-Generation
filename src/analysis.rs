//! Orbit quality metrics used by the Borda selection stage.
//!
//! Gatekeeping signals (not ranked):
//! - [`calculate_total_energy`] — kinetic + potential energy of the initial
//!   condition, filtered in [`crate::sim::select_best_trajectory`].
//! - [`calculate_total_angular_momentum`] — magnitude used as a pre-rank filter.
//!
//! Borda-ranked signals (all "higher raw value = more desirable" except the
//! non-chaoticness score, which is ranked ascending so that *chaotic* orbits win):
//! - [`non_chaoticness_from_series`] — peakedness of each body's
//!   distance-to-opposite-pair-COM FFT spectrum (std-dev of magnitudes).
//! - [`equilateralness_score`] — time-averaged triangle balance.
//! - [`curvature_entropy`] — Shannon entropy of binned turning angles,
//!   rewarding trajectories with a varied mix of curvature regimes.
//! - [`permutation_entropy_from_series`] — Bandt-Pompe ordinal entropy of the
//!   distance-to-opposite-pair-COM series, rewarding orbits with rich
//!   temporal complexity.
//!
//! For the FFT-based non-chaoticness and Bandt-Pompe permutation signals, we
//! share the same distance series via
//! [`compute_com_distance_series`] and the [`compute_orbit_quality`] combined
//! entry point — the selection hot loop calls the latter to avoid recomputing
//! the series three times per candidate.

use crate::sim::{Body, G};
use crate::utils::{FLOAT_EPSILON, fourier_transform};
use nalgebra::Vector3;

/// Number of histogram bins used by [`curvature_entropy`].
///
/// Chosen so that `ln(BIN_COUNT) ≈ 3.466` provides a comfortable dynamic
/// range without oversampling `n`-small trajectories.
const CURVATURE_BIN_COUNT: usize = 32;

/// Embedding dimension `m` for [`permutation_entropy_from_series`] (Bandt-Pompe 2002).
///
/// `m = 4` yields `m! = 24` ordinal patterns — enough resolution to separate
/// periodic, quasi-periodic, and chaotic regimes while keeping per-candidate
/// cost trivial (one 4-element sort per step).
const PERM_EMBEDDING_DIM: usize = 4;

/// Time delay `τ` between consecutive samples inside a permutation window.
const PERM_DELAY: usize = 1;

/// Number of distinct ordinal patterns for [`PERM_EMBEDDING_DIM`], i.e. `m!`.
const PERM_FACTORIAL: usize = 24;

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

/// Distance from each body to the center-of-mass of the *other two* bodies,
/// evaluated at every step of the recorded trajectory.
///
/// Shared input for [`non_chaoticness_from_series`] and
/// [`permutation_entropy_from_series`] so that the Borda hot loop can compute
/// both metrics from a single pass (via [`compute_orbit_quality`]).
///
/// # Panics
///
/// Expects `positions.len() == 3` and all three sub-vectors to share the same
/// length.  Both hold for the 3-body simulator output by construction; we
/// therefore index without bounds checks for throughput.
#[must_use]
pub fn compute_com_distance_series(
    m1: f64,
    m2: f64,
    m3: f64,
    positions: &[Vec<Vector3<f64>>],
) -> [Vec<f64>; 3] {
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
    [r1, r2, r3]
}

/// Non-chaoticness computed from pre-extracted distance series.
///
/// Smaller values indicate richer chaotic motion; the Borda selection ranks
/// this score ascending.
#[must_use]
pub fn non_chaoticness_from_series(series: &[Vec<f64>; 3]) -> f64 {
    if series[0].is_empty() {
        return 0.0;
    }
    let sd1 = sample_std_dev_of_fft_magnitudes(&series[0]);
    let sd2 = sample_std_dev_of_fft_magnitudes(&series[1]);
    let sd3 = sample_std_dev_of_fft_magnitudes(&series[2]);
    (sd1 + sd2 + sd3) / 3.0
}

fn sample_std_dev_of_fft_magnitudes(signal: &[f64]) -> f64 {
    let spectrum = fourier_transform(signal);
    let mags: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
    sample_std_dev(&mags)
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

/// Shannon entropy of binned turning angles, averaged across the three bodies.
///
/// For each body, the turning angle at step `i` is the angle between successive
/// displacement vectors `(p_i - p_{i-1})` and `(p_{i+1} - p_i)`.  Samples where
/// either displacement has magnitude below [`FLOAT_EPSILON`] are skipped.
///
/// Angles are binned into [`CURVATURE_BIN_COUNT`] equal bins spanning the
/// *observed* `[min, max]` turning-angle range for each body — so the metric
/// measures how evenly *shape variety* is distributed rather than whether
/// per-step angles happen to fall in a particular absolute range.  Typical
/// gravitational orbits produce tiny per-step turning angles (well below
/// `π/32`); a fixed `[0, π]` binning would collapse every such orbit into a
/// single bin regardless of how varied its curvature actually is.
///
/// The entropy is computed over the resulting histogram and averaged across
/// the three bodies.  Returns `0.0` for trajectories shorter than three steps,
/// when every segment is degenerate, or when the observed angle range is
/// below the noise floor (no meaningful variation).
///
/// A perfectly circular orbit collapses to a single bin (entropy `0`); a
/// trajectory that mixes sharp turns, gentle curves, and near-straight
/// segments spreads mass across many bins (entropy approaches `ln(32) ≈ 3.466`).
#[must_use]
pub fn curvature_entropy(positions: &[Vec<Vector3<f64>>]) -> f64 {
    if positions.is_empty() || positions[0].len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for body_traj in positions {
        sum += curvature_entropy_of_body(body_traj);
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Relative-variation floor below which the observed turning-angle range is
/// considered indistinguishable from numerical noise (RK4 roundoff, etc.).
/// Ratios below this trigger an entropy of `0` — no meaningful variety.
const CURVATURE_SPAN_NOISE_FLOOR: f64 = 1e-8;

fn curvature_entropy_of_body(traj: &[Vector3<f64>]) -> f64 {
    if traj.len() < 3 {
        return 0.0;
    }
    // First pass: collect the turning angles that survive the degeneracy filter.
    // Pre-allocating to the maximum possible length keeps the hot path
    // allocation-free after the initial reservation.
    let mut angles: Vec<f64> = Vec::with_capacity(traj.len() - 2);
    for i in 1..traj.len() - 1 {
        let v1 = traj[i] - traj[i - 1];
        let v2 = traj[i + 1] - traj[i];
        let n1 = v1.norm();
        let n2 = v2.norm();
        if n1 < FLOAT_EPSILON || n2 < FLOAT_EPSILON {
            continue;
        }
        let cos_theta = (v1.dot(&v2) / (n1 * n2)).clamp(-1.0, 1.0);
        angles.push(cos_theta.acos());
    }
    if angles.is_empty() {
        return 0.0;
    }

    let mut min_angle = f64::INFINITY;
    let mut max_angle = f64::NEG_INFINITY;
    for &a in &angles {
        if a < min_angle {
            min_angle = a;
        }
        if a > max_angle {
            max_angle = a;
        }
    }
    let span = max_angle - min_angle;
    // Absolute and relative noise-floor checks: we demand both that the raw
    // span is not at machine-epsilon scale *and* that it represents a real
    // fraction of the typical angle magnitude — this keeps integration noise
    // from being mistaken for curvature variety on near-circular orbits.
    let scale = max_angle.abs().max(FLOAT_EPSILON);
    if span < FLOAT_EPSILON || span < CURVATURE_SPAN_NOISE_FLOOR * scale {
        return 0.0;
    }

    // Second pass: normalize angles to `[0, 1]` and drop them into
    // [`CURVATURE_BIN_COUNT`] equal bins over that range.
    let mut counts = [0u32; CURVATURE_BIN_COUNT];
    let mut total = 0u32;
    let bin_scale = (CURVATURE_BIN_COUNT as f64) / span;
    for angle in angles {
        let raw = (angle - min_angle) * bin_scale;
        // usize saturating: theta ∈ [min_angle, max_angle] so raw ∈ [0, BIN_COUNT];
        // clamp the right-endpoint sample into the last bin.
        let bin = (raw as usize).min(CURVATURE_BIN_COUNT - 1);
        counts[bin] += 1;
        total += 1;
    }
    shannon_entropy(&counts, total)
}

fn shannon_entropy<const N: usize>(counts: &[u32; N], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = f64::from(total);
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = f64::from(c) / total_f;
            h -= p * p.ln();
        }
    }
    h
}

/// Normalized Bandt-Pompe permutation entropy of the COM-distance series,
/// averaged across the three bodies.
///
/// For each body's distance series `r_k`, count how often each ordinal pattern
/// of a length-[`PERM_EMBEDDING_DIM`] window appears, then compute Shannon
/// entropy normalized by `ln(PERM_FACTORIAL)` so the result lies in `[0, 1]`.
///
/// A strictly monotonic series produces only one pattern (entropy `0`); an
/// i.i.d. series produces every pattern with near-equal frequency (entropy
/// approaches `1`).  Periodic series visit a small subset of patterns and sit
/// somewhere in between.
///
/// Returns `0.0` if a series is shorter than [`PERM_EMBEDDING_DIM`].
///
/// Computed from pre-extracted distance series so the selection hot loop can
/// share [`compute_com_distance_series`] with the FFT-based score.
#[must_use]
pub fn permutation_entropy_from_series(series: &[Vec<f64>; 3]) -> f64 {
    let h1 = permutation_entropy_of_series(&series[0]);
    let h2 = permutation_entropy_of_series(&series[1]);
    let h3 = permutation_entropy_of_series(&series[2]);
    (h1 + h2 + h3) / 3.0
}

fn permutation_entropy_of_series(series: &[f64]) -> f64 {
    if series.len() < PERM_EMBEDDING_DIM {
        return 0.0;
    }
    let mut counts = [0u32; PERM_FACTORIAL];
    let mut total = 0u32;
    let last_start = series.len() - (PERM_EMBEDDING_DIM - 1) * PERM_DELAY;
    for i in 0..last_start {
        let window = [
            series[i],
            series[i + PERM_DELAY],
            series[i + 2 * PERM_DELAY],
            series[i + 3 * PERM_DELAY],
        ];
        let idx = ordinal_pattern_m4(&window);
        counts[idx] += 1;
        total += 1;
    }
    let h = shannon_entropy(&counts, total);
    // usize→f64: PERM_FACTORIAL is a small compile-time constant
    h / (PERM_FACTORIAL as f64).ln()
}

/// Map a length-4 window to its ordinal pattern index in `[0, 24)`.
///
/// Uses the Lehmer code of the sort-permutation: sort indices `[0, 1, 2, 3]`
/// by the corresponding window values, then encode the result via factorial
/// base.  `total_cmp` ensures a total order that handles `NaN` without
/// panicking (rare but possible with degenerate inputs).
fn ordinal_pattern_m4(window: &[f64; PERM_EMBEDDING_DIM]) -> usize {
    let mut indices: [u8; PERM_EMBEDDING_DIM] = [0, 1, 2, 3];
    indices.sort_unstable_by(|&a, &b| window[a as usize].total_cmp(&window[b as usize]));
    const FACTORIALS: [usize; PERM_EMBEDDING_DIM - 1] = [6, 2, 1]; // 3!, 2!, 1!
    let mut rank = 0usize;
    for (k, factorial) in FACTORIALS.iter().enumerate() {
        let mut inversions = 0usize;
        for &later in &indices[k + 1..] {
            if later < indices[k] {
                inversions += 1;
            }
        }
        rank += inversions * factorial;
    }
    rank
}

/// All four Borda-ranked orbit-quality metrics, computed in a single pass.
///
/// Reuses the shared COM-distance series across [`non_chaoticness_from_series`] and
/// [`permutation_entropy_from_series`] to avoid recomputing it per candidate inside
/// the Borda hot loop. For isolated metric lookups in tests or diagnostics, prefer
/// the individual `pub` functions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrbitQualityMetrics {
    /// [`non_chaoticness_from_series`] score — FFT-magnitude std-dev, higher = more regular.
    pub non_chaoticness: f64,
    /// [`equilateralness_score`] — time-averaged triangle balance, higher = better.
    pub equilateralness: f64,
    /// [`curvature_entropy`] — turning-angle diversity, higher = more varied.
    pub curvature_entropy: f64,
    /// [`permutation_entropy_from_series`] — Bandt-Pompe ordinal entropy, higher = more complex.
    pub permutation_entropy: f64,
}

/// Compute all four Borda-ranked quality metrics for a recorded orbit.
#[must_use]
pub fn compute_orbit_quality(
    m1: f64,
    m2: f64,
    m3: f64,
    positions: &[Vec<Vector3<f64>>],
) -> OrbitQualityMetrics {
    if positions.is_empty() || positions[0].is_empty() {
        return OrbitQualityMetrics::default();
    }
    let series = compute_com_distance_series(m1, m2, m3, positions);
    OrbitQualityMetrics {
        non_chaoticness: non_chaoticness_from_series(&series),
        equilateralness: equilateralness_score(positions),
        curvature_entropy: curvature_entropy(positions),
        permutation_entropy: permutation_entropy_from_series(&series),
    }
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
    fn test_non_chaoticness_empty_returns_zero() {
        let positions = vec![vec![], vec![], vec![]];
        let series = compute_com_distance_series(1.0, 1.0, 1.0, &positions);
        let result = non_chaoticness_from_series(&series);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_non_chaoticness_returns_finite() {
        let n = 100;
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
        let series = compute_com_distance_series(1.0, 1.0, 1.0, &positions);
        let result = non_chaoticness_from_series(&series);
        assert!(result.is_finite(), "non_chaoticness should return a finite value");
        assert!(result >= 0.0);
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

    // -----------------------------------------------------------------------
    // curvature_entropy
    // -----------------------------------------------------------------------

    fn constant_trajectory_for_3(body_traj: Vec<Vector3<f64>>) -> Vec<Vec<Vector3<f64>>> {
        vec![body_traj.clone(), body_traj.clone(), body_traj]
    }

    #[test]
    fn curvature_entropy_empty_returns_zero() {
        let positions: Vec<Vec<Vector3<f64>>> = vec![vec![], vec![], vec![]];
        assert_eq!(curvature_entropy(&positions), 0.0);
    }

    #[test]
    fn curvature_entropy_too_short_returns_zero() {
        let two_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let positions = constant_trajectory_for_3(two_points);
        assert_eq!(curvature_entropy(&positions), 0.0);
    }

    #[test]
    fn curvature_entropy_straight_line_is_zero() {
        let traj: Vec<Vector3<f64>> =
            (0..100).map(|k| Vector3::new(f64::from(k), 0.0, 0.0)).collect();
        let positions = constant_trajectory_for_3(traj);
        let h = curvature_entropy(&positions);
        assert!(h.abs() < 1e-12, "straight line should yield zero curvature entropy, got {h}");
    }

    #[test]
    fn curvature_entropy_degenerate_zero_velocity_returns_zero() {
        let traj = vec![Vector3::new(2.0, 3.0, 4.0); 50];
        let positions = constant_trajectory_for_3(traj);
        let h = curvature_entropy(&positions);
        assert_eq!(h, 0.0, "stationary body should yield zero curvature entropy");
    }

    #[test]
    fn curvature_entropy_uniform_circle_is_low() {
        let n = 400usize;
        let traj: Vec<Vector3<f64>> = (0..n)
            .map(|k| {
                // usize→f64: test input is a small fixed constant
                let theta = std::f64::consts::TAU * (k as f64) / (n as f64);
                Vector3::new(theta.cos(), theta.sin(), 0.0)
            })
            .collect();
        let positions = constant_trajectory_for_3(traj);
        let h = curvature_entropy(&positions);
        assert!(h < 1e-6, "uniform circle should concentrate in a single bin, got H = {h}");
    }

    #[test]
    fn curvature_entropy_chaotic_is_high() {
        // Deterministic xorshift pseudo-random walk.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let next_unit = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            // u64→f64 via as-cast: we accept the rounding; result is used only for random geometry.
            (*s as f64) / (u64::MAX as f64)
        };
        let n = 4000usize;
        let traj: Vec<Vector3<f64>> = (0..n)
            .scan(Vector3::zeros(), |pos, _| {
                let dx = next_unit(&mut state) - 0.5;
                let dy = next_unit(&mut state) - 0.5;
                let dz = next_unit(&mut state) - 0.5;
                *pos += Vector3::new(dx, dy, dz);
                Some(*pos)
            })
            .collect();
        let positions = constant_trajectory_for_3(traj);
        let h = curvature_entropy(&positions);
        let h_max = (CURVATURE_BIN_COUNT as f64).ln();
        assert!(
            h > 0.7 * h_max,
            "chaotic walk should spread mass across bins: H = {h}, max = {h_max}"
        );
    }

    #[test]
    fn curvature_entropy_is_finite_on_realistic_input() {
        let n = 500usize;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = f64::from(u16::try_from(step).unwrap()) * 0.01;
                        let phase = f64::from(body) * std::f64::consts::TAU / 3.0;
                        Vector3::new((t + phase).cos(), (t + phase).sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let h = curvature_entropy(&positions);
        assert!(h.is_finite(), "curvature_entropy should always return a finite value");
        assert!(h >= 0.0);
    }

    // -----------------------------------------------------------------------
    // permutation_entropy
    // -----------------------------------------------------------------------

    fn xorshift_unit_series(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed | 1; // xorshift requires non-zero state
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // u64→f64 via as-cast: acceptable rounding for a test RNG.
                (state as f64) / (u64::MAX as f64)
            })
            .collect()
    }

    #[test]
    fn permutation_entropy_empty_public_api_returns_zero() {
        let positions: Vec<Vec<Vector3<f64>>> = vec![vec![], vec![], vec![]];
        let series = compute_com_distance_series(1.0, 1.0, 1.0, &positions);
        assert_eq!(permutation_entropy_from_series(&series), 0.0);
    }

    #[test]
    fn permutation_entropy_of_series_empty_returns_zero() {
        assert_eq!(permutation_entropy_of_series(&[]), 0.0);
    }

    #[test]
    fn permutation_entropy_of_series_too_short_returns_zero() {
        assert_eq!(permutation_entropy_of_series(&[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn permutation_entropy_monotonic_series_is_zero() {
        let series: Vec<f64> = (0..200).map(f64::from).collect();
        let h = permutation_entropy_of_series(&series);
        assert!(h.abs() < 1e-12, "strictly monotonic series should give H = 0, got {h}");
    }

    #[test]
    fn permutation_entropy_random_series_is_near_one() {
        let series = xorshift_unit_series(0xDEAD_BEEF_CAFE_F00D, 5000);
        let h = permutation_entropy_of_series(&series);
        assert!(h > 0.95, "i.i.d. series should approach H = 1, got {h}");
        assert!(h <= 1.0 + 1e-12, "permutation entropy must not exceed 1, got {h}");
    }

    #[test]
    fn permutation_entropy_periodic_series_is_low() {
        let period = 8.0;
        let series: Vec<f64> =
            (0..500).map(|k| (std::f64::consts::TAU * f64::from(k) / period).sin()).collect();
        let h = permutation_entropy_of_series(&series);
        assert!(h < 0.65, "periodic series should give moderate-to-low H, got {h}");
    }

    #[test]
    fn permutation_entropy_is_in_unit_interval_across_seeds() {
        for seed in [1u64, 42, 0xC0FFEE, 0xBAD_F00D, 9876543210] {
            let series = xorshift_unit_series(seed, 500);
            let h = permutation_entropy_of_series(&series);
            assert!(
                (0.0..=1.0 + 1e-12).contains(&h),
                "permutation entropy out of [0, 1]: H = {h} (seed = {seed})"
            );
        }
    }

    #[test]
    fn permutation_entropy_public_api_is_finite_and_bounded() {
        // Full public API: three-body trajectory → three distance series → avg H.
        let n = 400usize;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = f64::from(u16::try_from(step).unwrap()) * 0.05;
                        let phase = f64::from(body) * std::f64::consts::TAU / 3.0;
                        Vector3::new((t + phase).cos(), (t + phase).sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let series = compute_com_distance_series(1.0, 2.0, 3.0, &positions);
        let h = permutation_entropy_from_series(&series);
        assert!(h.is_finite(), "permutation_entropy should be finite");
        assert!(
            (0.0..=1.0 + 1e-12).contains(&h),
            "permutation_entropy must lie in [0, 1], got {h}"
        );
    }

    // -----------------------------------------------------------------------
    // ordinal_pattern_m4 bijection
    // -----------------------------------------------------------------------

    #[test]
    fn ordinal_pattern_m4_covers_all_24_patterns_uniquely() {
        // Enumerate all 4! = 24 permutations of [0, 1, 2, 3] and verify that
        // each one yields a distinct pattern index in [0, 24).
        let mut perms: Vec<[usize; 4]> = Vec::with_capacity(24);
        let digits = [0usize, 1, 2, 3];
        for a in 0..4 {
            for b in 0..4 {
                if b == a {
                    continue;
                }
                for c in 0..4 {
                    if c == a || c == b {
                        continue;
                    }
                    for d in 0..4 {
                        if d == a || d == b || d == c {
                            continue;
                        }
                        perms.push([digits[a], digits[b], digits[c], digits[d]]);
                    }
                }
            }
        }
        assert_eq!(perms.len(), 24);

        let mut seen = [false; 24];
        for perm in &perms {
            // Build a window where sorting indices by value reproduces `perm`.
            // Assign increasing values along the permutation: window[perm[r]] = r.
            let mut window = [0.0f64; 4];
            for (rank, &idx) in perm.iter().enumerate() {
                // usize→f64: rank is in [0, 4)
                window[idx] = rank as f64;
            }
            let pattern = ordinal_pattern_m4(&window);
            assert!(pattern < 24, "pattern index out of range: {pattern}");
            assert!(!seen[pattern], "collision at pattern index {pattern} for {perm:?}");
            seen[pattern] = true;
        }
        assert!(seen.iter().all(|&s| s), "some pattern indices were never produced");
    }

    // -----------------------------------------------------------------------
    // compute_orbit_quality
    // -----------------------------------------------------------------------

    #[test]
    fn compute_orbit_quality_matches_individual_functions() {
        let n = 300usize;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = f64::from(u16::try_from(step).unwrap()) * 0.05;
                        let phase = f64::from(body) * std::f64::consts::TAU / 3.0;
                        Vector3::new((t + phase).cos(), (t + phase).sin(), 0.2 * t.sin())
                    })
                    .collect()
            })
            .collect();

        let metrics = compute_orbit_quality(1.0, 2.0, 3.0, &positions);
        let series = compute_com_distance_series(1.0, 2.0, 3.0, &positions);
        let nc = non_chaoticness_from_series(&series);
        let eq = equilateralness_score(&positions);
        let ce = curvature_entropy(&positions);
        let pe = permutation_entropy_from_series(&series);

        assert_eq!(metrics.non_chaoticness.to_bits(), nc.to_bits());
        assert_eq!(metrics.equilateralness.to_bits(), eq.to_bits());
        assert_eq!(metrics.curvature_entropy.to_bits(), ce.to_bits());
        assert_eq!(metrics.permutation_entropy.to_bits(), pe.to_bits());
    }

    #[test]
    fn compute_orbit_quality_empty_returns_defaults() {
        let positions: Vec<Vec<Vector3<f64>>> = vec![vec![], vec![], vec![]];
        let metrics = compute_orbit_quality(1.0, 1.0, 1.0, &positions);
        assert_eq!(metrics, OrbitQualityMetrics::default());
    }
}
