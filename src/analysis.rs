use crate::sim::{Body, G};
use crate::utils::FftCache;
#[cfg(test)]
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
            if r > 1e-10 {
                pot += -G * bodies[i].mass * bodies[j].mass / r;
            }
        }
    }
    kin + pot
}

/// Total angular momentum vector
pub fn calculate_total_angular_momentum(bodies: &[Body]) -> Vector3<f64> {
    let mut total_l = Vector3::zeros();
    for b in bodies {
        total_l += b.mass * b.position.cross(&b.velocity);
    }
    total_l
}

fn compute_jacobi_radii(
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

/// A measure of "regularity" vs "chaos", smaller => more chaotic.
/// Uses a thread-local FFT planner cache for efficiency in parallel contexts.
pub fn non_chaoticness_cached(
    m1: f64,
    m2: f64,
    m3: f64,
    positions: &[Vec<Vector3<f64>>],
    fft_cache: &mut FftCache,
) -> f64 {
    if positions[0].is_empty() {
        return 0.0;
    }
    let (r1, r2, r3) = compute_jacobi_radii(m1, m2, m3, positions);
    let abs1: Vec<f64> = fft_cache.transform(&r1).iter().map(|c| c.norm()).collect();
    let abs2: Vec<f64> = fft_cache.transform(&r2).iter().map(|c| c.norm()).collect();
    let abs3: Vec<f64> = fft_cache.transform(&r3).iter().map(|c| c.norm()).collect();
    (sample_std_dev(&abs1) + sample_std_dev(&abs2) + sample_std_dev(&abs3)) / 3.0
}

#[cfg(test)]
pub fn non_chaoticness(m1: f64, m2: f64, m3: f64, positions: &[Vec<Vector3<f64>>]) -> f64 {
    if positions[0].is_empty() {
        return 0.0;
    }
    let (r1, r2, r3) = compute_jacobi_radii(m1, m2, m3, positions);
    let abs1: Vec<f64> = fourier_transform(&r1).iter().map(|c| c.norm()).collect();
    let abs2: Vec<f64> = fourier_transform(&r2).iter().map(|c| c.norm()).collect();
    let abs3: Vec<f64> = fourier_transform(&r3).iter().map(|c| c.norm()).collect();
    (sample_std_dev(&abs1) + sample_std_dev(&abs2) + sample_std_dev(&abs3)) / 3.0
}

/// Score how "equilateral" the 3-body triangle is over time
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

    fn triangle_positions(n: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                (0..n)
                    .map(|s| {
                        let t = s as f64 / n as f64 * std::f64::consts::TAU;
                        let phase = body as f64 * std::f64::consts::TAU / 3.0;
                        Vector3::new((t + phase).cos() * 100.0, (t + phase).sin() * 80.0, 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_non_chaoticness_cached_matches_uncached() {
        let positions = triangle_positions(512);
        let (m1, m2, m3) = (200.0, 150.0, 250.0);

        let uncached = non_chaoticness(m1, m2, m3, &positions);
        let mut cache = FftCache::new();
        let cached = non_chaoticness_cached(m1, m2, m3, &positions, &mut cache);

        assert_eq!(
            uncached.to_bits(),
            cached.to_bits(),
            "cached FFT should produce bitwise identical result"
        );
    }

    #[test]
    fn test_non_chaoticness_cached_deterministic_across_calls() {
        let positions = triangle_positions(256);
        let (m1, m2, m3) = (180.0, 220.0, 160.0);
        let mut cache = FftCache::new();

        let run1 = non_chaoticness_cached(m1, m2, m3, &positions, &mut cache);
        let run2 = non_chaoticness_cached(m1, m2, m3, &positions, &mut cache);

        assert_eq!(run1.to_bits(), run2.to_bits(), "repeated calls should be identical");
    }

    #[test]
    fn test_non_chaoticness_empty_positions() {
        let positions = vec![vec![], vec![], vec![]];
        assert_eq!(non_chaoticness(1.0, 1.0, 1.0, &positions), 0.0);

        let mut cache = FftCache::new();
        assert_eq!(non_chaoticness_cached(1.0, 1.0, 1.0, &positions, &mut cache), 0.0);
    }

    #[test]
    fn test_equilateralness_perfect_triangle() {
        let n = 100;
        let positions = triangle_positions(n);
        let score = equilateralness_score(&positions);
        assert!(score > 0.0, "equilateral triangle should have positive score");
    }

    #[test]
    fn test_calculate_total_energy_bound_system() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.3, 0.0)),
            Body::new(200.0, Vector3::new(-100.0, 0.0, 0.0), Vector3::new(0.0, -0.3, 0.0)),
            Body::new(200.0, Vector3::new(0.0, 100.0, 0.0), Vector3::new(-0.3, 0.0, 0.0)),
        ];
        let energy = calculate_total_energy(&bodies);
        assert!(energy < 0.0, "closely bound system should have negative energy, got {energy}");
    }

    #[test]
    fn test_angular_momentum_symmetric_system() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(200.0, Vector3::new(-50.0, 86.6, 0.0), Vector3::new(-0.433, -0.25, 0.0)),
            Body::new(200.0, Vector3::new(-50.0, -86.6, 0.0), Vector3::new(0.433, -0.25, 0.0)),
        ];
        let l = calculate_total_angular_momentum(&bodies);
        assert!(l.norm() > 0.0, "rotating system should have non-zero angular momentum");
    }
}
