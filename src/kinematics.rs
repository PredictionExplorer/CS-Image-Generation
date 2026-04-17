//! Finite-difference kinematics derived from recorded position trajectories.

use nalgebra::Vector3;

/// Per-timestep kinematic quantities for the three-body render path.
#[derive(Clone, Debug)]
pub struct KinematicTrajectories {
    /// Per-body velocity estimates `velocity[body][t]` (last step duplicated).
    pub velocity: Vec<Vec<Vector3<f64>>>,
    /// Per-body acceleration `acceleration[body][t]` (first step zero).
    pub acceleration: Vec<Vec<Vector3<f64>>>,
    /// Scalar curvature proxy on the XY projection (turning rate per unit step length).
    pub curvature_xy: Vec<Vec<f64>>,
    /// Minimum pairwise separation at each timestep.
    pub min_pairwise_distance: Vec<f64>,
}

/// Build kinematic series from integrated positions and simulation `dt`.
#[must_use]
pub fn compute_kinematics(positions: &[Vec<Vector3<f64>>], dt: f64) -> KinematicTrajectories {
    let n = positions[0].len();
    let mut velocity = vec![vec![Vector3::zeros(); n]; 3];
    let mut acceleration = vec![vec![Vector3::zeros(); n]; 3];
    let mut curvature_xy = vec![vec![0.0f64; n]; 3];
    let mut min_pairwise_distance = vec![0.0f64; n];

    let inv_dt = 1.0 / dt.max(1e-18);

    for body in 0..3 {
        let p = &positions[body];
        for t in 0..n {
            if t + 1 < n {
                velocity[body][t] = (p[t + 1] - p[t]) * inv_dt;
            } else if t > 0 {
                velocity[body][t] = velocity[body][t - 1];
            }
        }
        for t in 0..n {
            if t > 0 {
                acceleration[body][t] = (velocity[body][t] - velocity[body][t - 1]) * inv_dt;
            }
        }
        for t in 1..n.saturating_sub(1) {
            let v0 =
                Vector3::new(p[t][0] - p[t - 1][0], p[t][1] - p[t - 1][1], p[t][2] - p[t - 1][2]);
            let v1 =
                Vector3::new(p[t + 1][0] - p[t][0], p[t + 1][1] - p[t][1], p[t + 1][2] - p[t][2]);
            let l0 = (v0.x * v0.x + v0.y * v0.y).sqrt().max(1e-18);
            let l1 = (v1.x * v1.x + v1.y * v1.y).sqrt().max(1e-18);
            let u0x = v0.x / l0;
            let u0y = v0.y / l0;
            let u1x = v1.x / l1;
            let u1y = v1.y / l1;
            let cross = u0x * u1y - u0y * u1x;
            let _dot = (u0x * u1x + u0y * u1y).clamp(-1.0, 1.0);
            let sin_theta = cross.abs().clamp(0.0, 1.0);
            let mean_len = 0.5 * (l0 + l1);
            curvature_xy[body][t] = sin_theta / mean_len.max(1e-18);
        }
    }

    for t in 0..n {
        let p0 = positions[0][t];
        let p1 = positions[1][t];
        let p2 = positions[2][t];
        let d01 = (p0 - p1).norm();
        let d12 = (p1 - p2).norm();
        let d20 = (p2 - p0).norm();
        min_pairwise_distance[t] = d01.min(d12).min(d20);
    }

    KinematicTrajectories { velocity, acceleration, curvature_xy, min_pairwise_distance }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_line(n: usize, v: f64) -> Vec<Vec<Vector3<f64>>> {
        (0..3i32)
            .map(|body| {
                (0..n).map(|t| Vector3::new(v * t as f64 + f64::from(body), 0.0, 0.0)).collect()
            })
            .collect()
    }

    #[test]
    fn test_velocity_matches_finite_difference() {
        let positions = straight_line(8, 2.5);
        let dt = 0.1;
        let k = compute_kinematics(&positions, dt);
        for body in 0..3 {
            for t in 0..6 {
                let vx = k.velocity[body][t].x;
                assert!((vx - 2.5 / dt).abs() < 1e-9, "vx mismatch {body}/{t}: {vx}");
            }
        }
    }

    #[test]
    fn test_acceleration_zero_for_uniform_velocity() {
        let positions = straight_line(12, 1.0);
        let k = compute_kinematics(&positions, 0.05);
        for body in 0..3 {
            for t in 2..10 {
                assert!(
                    k.acceleration[body][t].norm() < 1e-6,
                    "unexpected acceleration body={body} t={t}"
                );
            }
        }
    }

    #[test]
    fn test_curvature_zero_on_straight_line() {
        let positions = straight_line(16, 1.0);
        let k = compute_kinematics(&positions, 0.1);
        for body in 0..3 {
            for t in 1..14 {
                assert!(k.curvature_xy[body][t].abs() < 1e-9, "curvature mismatch");
            }
        }
    }

    #[test]
    fn test_curvature_positive_on_circle() {
        let n = 512usize;
        let positions: Vec<Vec<Vector3<f64>>> = (0i32..3)
            .map(|body| {
                (0..n)
                    .map(|t| {
                        let ang =
                            (t as f64) * std::f64::consts::TAU / n as f64 + f64::from(body) * 0.1;
                        Vector3::new(ang.cos(), ang.sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let k = compute_kinematics(&positions, 1.0);
        let mean: f64 = k.curvature_xy[0].iter().skip(10).take(400).copied().sum::<f64>() / 400.0;
        assert!(mean > 0.0, "circle should have positive curvature, got {mean}");
    }

    #[test]
    fn test_min_pairwise_distance_matches_expected_pair() {
        let n = 5u32;
        // Body 0 sits at origin, body 1 drifts away linearly, body 2 drifts twice as fast.
        let positions: Vec<Vec<Vector3<f64>>> = vec![
            (0..n).map(|_| Vector3::zeros()).collect(),
            (0..n).map(|t| Vector3::new(f64::from(t), 0.0, 0.0)).collect(),
            (0..n).map(|t| Vector3::new(2.0 * f64::from(t), 0.0, 0.0)).collect(),
        ];
        let k = compute_kinematics(&positions, 1.0);
        for (t, distance) in k.min_pairwise_distance.iter().enumerate() {
            // Pairs: (0,1) = t, (1,2) = t, (0,2) = 2t → min is always t.
            assert!((distance - t as f64).abs() < 1e-12, "unexpected min at {t}: {distance}");
        }
    }
}
