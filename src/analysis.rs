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
                        let t = step as f64 * 0.1;
                        let angle = t + body as f64 * std::f64::consts::TAU / 3.0;
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
                let angle = body as f64 * std::f64::consts::TAU / 3.0;
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
}
