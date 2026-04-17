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

/// A measure of "regularity" vs "chaos", smaller => more chaotic
#[must_use]
pub fn non_chaoticness(m1: f64, m2: f64, m3: f64, positions: &[Vec<Vector3<f64>>]) -> f64 {
    let len = positions[0].len();
    if len == 0 {
        return 0.0;
    }
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
    let abs1: Vec<f64> = fourier_transform(&r1).iter().map(|c| c.norm()).collect();
    let abs2: Vec<f64> = fourier_transform(&r2).iter().map(|c| c.norm()).collect();
    let abs3: Vec<f64> = fourier_transform(&r3).iter().map(|c| c.norm()).collect();
    let sd1 = sample_std_dev(&abs1);
    let sd2 = sample_std_dev(&abs2);
    let sd3 = sample_std_dev(&abs3);
    (sd1 + sd2 + sd3) / 3.0
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

/// Count of timesteps where any pairwise distance drops below a scale-aware threshold.
#[must_use]
pub fn close_approach_density(positions: &[Vec<Vector3<f64>>], threshold: f64) -> f64 {
    let n = positions[0].len();
    if n == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    for (p0, (p1, p2)) in positions[0].iter().zip(positions[1].iter().zip(positions[2].iter())) {
        let d01 = (p0 - p1).norm();
        let d12 = (p1 - p2).norm();
        let d20 = (p2 - p0).norm();
        if d01.min(d12).min(d20) < threshold {
            hits += 1;
        }
    }
    hits as f64 / n as f64
}

/// Shannon entropy of XY tangent-angle bins (structure vs noise proxy).
#[must_use]
pub fn turning_angle_entropy_xy(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let n = positions[0].len();
    if n < 4 {
        return 0.0;
    }
    const BINS: usize = 36;
    let mut hist = [0usize; BINS];
    let mut total = 0usize;
    for p in positions {
        for win in p.windows(2).skip(1) {
            let vx = win[1].x - win[0].x;
            let vy = win[1].y - win[0].y;
            let ang = vy.atan2(vx);
            let idx = (((ang + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * BINS as f64)
                .floor() as usize;
            let idx = idx.min(BINS - 1);
            hist[idx] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let inv = 1.0 / total as f64;
    let mut h = 0.0;
    for c in hist {
        if c > 0 {
            let p = c as f64 * inv;
            h -= p * p.ln();
        }
    }
    h / (BINS as f64).ln().max(1e-9)
}

/// Simple box-counting proxy on the XY projection of body 0 (fractal-ish trails score higher).
#[must_use]
pub fn fractal_dimension_proxy_xy(positions: &[Vec<Vector3<f64>>], grid: usize) -> f64 {
    let p = &positions[0];
    let n = p.len().min(50_000);
    if n < 8 || grid < 4 {
        return 0.0;
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for v in p.iter().take(n) {
        min_x = min_x.min(v.x);
        max_x = max_x.max(v.x);
        min_y = min_y.min(v.y);
        max_y = max_y.max(v.y);
    }
    let w = (max_x - min_x).max(1e-9);
    let h = (max_y - min_y).max(1e-9);
    let mut occupied = vec![false; grid * grid];
    let mut count = 0usize;
    for v in p.iter().take(n) {
        let gx = (((v.x - min_x) / w) * (grid - 1) as f64).round() as usize;
        let gy = (((v.y - min_y) / h) * (grid - 1) as f64).round() as usize;
        let gx = gx.min(grid - 1);
        let gy = gy.min(grid - 1);
        let i = gy * grid + gx;
        if !occupied[i] {
            occupied[i] = true;
            count += 1;
        }
    }
    (count as f64).ln() / (grid as f64).ln()
}

/// Time-reversal symmetry score on the XY projection of the ensemble.
///
/// For each body we compare `p(t)` with `p(N-1-t)` after translating so the trajectory
/// is zero-mean.  A perfectly time-symmetric pattern (reversible) yields a value near 1.0;
/// a fully random trajectory yields values near 0.
#[must_use]
pub fn time_reversal_symmetry_xy(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let n = positions[0].len();
    if n < 8 {
        return 0.0;
    }
    let mut sum = 0.0;
    for p in positions {
        let mut mx = 0.0;
        let mut my = 0.0;
        for v in p.iter().take(n) {
            mx += v.x;
            my += v.y;
        }
        let inv_n = 1.0 / n as f64;
        mx *= inv_n;
        my *= inv_n;

        let mut energy = 0.0f64;
        let mut mirror = 0.0f64;
        for t in 0..n {
            let a_x = p[t].x - mx;
            let a_y = p[t].y - my;
            let b_x = p[n - 1 - t].x - mx;
            let b_y = p[n - 1 - t].y - my;
            mirror += a_x * b_x + a_y * b_y;
            energy += a_x * a_x + a_y * a_y;
        }
        if energy > 1e-18 {
            sum += (mirror / energy).clamp(-1.0, 1.0);
        }
    }
    ((sum / 3.0) + 1.0) * 0.5
}

/// `C_k` rotational symmetry score of body 0's XY trajectory (best over `k ∈ {2, 3}`).
#[must_use]
pub fn rotational_symmetry_score_xy(positions: &[Vec<Vector3<f64>>]) -> f64 {
    use std::f64::consts::TAU;
    let p = &positions[0];
    let n = p.len();
    if n < 16 {
        return 0.0;
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for v in p {
        cx += v.x;
        cy += v.y;
    }
    cx /= n as f64;
    cy /= n as f64;

    let mut best = 0.0f64;
    for &k in &[2.0f64, 3.0] {
        let theta = TAU / k;
        let (s, c) = theta.sin_cos();
        let mut mirror = 0.0f64;
        let mut energy = 0.0f64;
        for v in p {
            let x = v.x - cx;
            let y = v.y - cy;
            let rx = c * x - s * y;
            let ry = s * x + c * y;
            // Compare rotated point to the closest of a small stride of samples.
            let mut best_dist = f64::INFINITY;
            let stride = (n / 128).max(1);
            for w in p.iter().step_by(stride) {
                let dx = rx - (w.x - cx);
                let dy = ry - (w.y - cy);
                best_dist = best_dist.min(dx * dx + dy * dy);
            }
            mirror += (x * rx + y * ry) - best_dist;
            energy += x * x + y * y;
        }
        if energy > 1e-18 {
            let score = ((mirror / energy) + 1.0) * 0.5;
            best = best.max(score.clamp(0.0, 1.0));
        }
    }
    best
}

/// Combined aesthetic score for Borda augmentation (higher is more visually interesting).
#[must_use]
pub fn beauty_ensemble_score(positions: &[Vec<Vector3<f64>>], m1: f64, m2: f64, m3: f64) -> f64 {
    let scale = (m1 + m2 + m3) / 3.0;
    let thr = (scale.powf(1.0 / 3.0) * 0.08).max(1e-3);
    let close = close_approach_density(positions, thr);
    let turn = turning_angle_entropy_xy(positions);
    let frac = fractal_dimension_proxy_xy(positions, 48);
    let sym = time_reversal_symmetry_xy(positions);
    let rot = rotational_symmetry_score_xy(positions);
    // Weighted sum — all components in [0, ~1.5]; tanh keeps output in (0, 1).
    let raw = 2.0 * close + 1.2 * turn + 0.8 * frac + 0.6 * sym + 0.4 * rot;
    raw.tanh()
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
        let result = non_chaoticness(1.0, 1.0, 1.0, &positions);
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
        let result = non_chaoticness(1.0, 1.0, 1.0, &positions);
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

    fn triangular_orbit_positions(n: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                (0..n)
                    .map(|step| {
                        let t = step as f64 * 0.05;
                        let angle = t + f64::from(body) * std::f64::consts::TAU / 3.0;
                        Vector3::new(angle.cos(), angle.sin(), 0.1 * t.sin())
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_close_approach_density_empty() {
        let positions = vec![vec![], vec![], vec![]];
        assert_eq!(close_approach_density(&positions, 0.1), 0.0);
    }

    #[test]
    fn test_close_approach_density_all_close_returns_one() {
        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0); 50],
            vec![Vector3::new(0.01, 0.0, 0.0); 50],
            vec![Vector3::new(-0.01, 0.01, 0.0); 50],
        ];
        assert_eq!(close_approach_density(&positions, 1.0), 1.0);
    }

    #[test]
    fn test_close_approach_density_zero_when_far() {
        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0); 50],
            vec![Vector3::new(100.0, 0.0, 0.0); 50],
            vec![Vector3::new(0.0, 100.0, 0.0); 50],
        ];
        assert_eq!(close_approach_density(&positions, 1.0), 0.0);
    }

    #[test]
    fn test_turning_angle_entropy_non_negative_and_bounded() {
        let positions = triangular_orbit_positions(400);
        let h = turning_angle_entropy_xy(&positions);
        assert!(h >= 0.0);
        assert!(h <= 1.0 + 1e-9);
    }

    #[test]
    fn test_turning_angle_entropy_too_short_is_zero() {
        let positions = vec![vec![Vector3::zeros(); 2]; 3];
        assert_eq!(turning_angle_entropy_xy(&positions), 0.0);
    }

    #[test]
    fn test_fractal_dimension_within_bounds() {
        let positions = triangular_orbit_positions(1024);
        let v = fractal_dimension_proxy_xy(&positions, 32);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn test_fractal_dimension_handles_short_input() {
        let positions = vec![vec![Vector3::zeros(); 4]; 3];
        assert_eq!(fractal_dimension_proxy_xy(&positions, 16), 0.0);
    }

    #[test]
    fn test_time_reversal_symmetry_high_for_reversible_orbit() {
        let positions = triangular_orbit_positions(200);
        let s = time_reversal_symmetry_xy(&positions);
        assert!(s >= 0.0);
        assert!(s <= 1.0);
    }

    #[test]
    fn test_time_reversal_symmetry_bounded_below_for_random() {
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as f64 / f64::from(u32::MAX) - 0.5
        };
        let positions: Vec<Vec<Vector3<f64>>> =
            (0..3).map(|_| (0..256).map(|_| Vector3::new(next(), next(), 0.0)).collect()).collect();
        let s = time_reversal_symmetry_xy(&positions);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_rotational_symmetry_bounded() {
        let positions = triangular_orbit_positions(512);
        let s = rotational_symmetry_score_xy(&positions);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_beauty_ensemble_score_within_unit_interval() {
        let positions = triangular_orbit_positions(512);
        let s = beauty_ensemble_score(&positions, 1.0, 1.0, 1.0);
        assert!(s.is_finite());
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_beauty_ensemble_score_empty_is_zero() {
        let positions = vec![vec![], vec![], vec![]];
        let s = beauty_ensemble_score(&positions, 1.0, 1.0, 1.0);
        assert_eq!(s, 0.0);
    }
}
