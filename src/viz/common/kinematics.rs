//! Derived kinematic series shared by visualization modes.
//!
//! Computed once per run (lazily, via [`crate::viz::context::VizContext`])
//! from the projected trajectory: velocities and speeds by central
//! differences, pairwise separations, and the body masses of the winning
//! candidate.

use crate::render::constants::DEFAULT_DT;
use crate::sim::Body;
use nalgebra::Vector3;

/// The three unordered body pairs in canonical order: (0,1), (0,2), (1,2).
pub const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Derived per-step kinematic series for the three bodies.
pub struct Kinematics {
    /// Per-body velocity vectors (central differences, `dt = DEFAULT_DT`).
    pub velocities: Vec<Vec<Vector3<f64>>>,
    /// Per-body speed magnitudes.
    pub speeds: Vec<Vec<f64>>,
    /// Pairwise separations per step, indexed by [`PAIRS`] order.
    pub pairwise: [Vec<f64>; 3],
    /// Body masses of the selected candidate.
    pub masses: [f64; 3],
}

impl Kinematics {
    /// Compute all series from the projected trajectory and selected bodies.
    #[must_use]
    pub fn compute(positions: &[Vec<Vector3<f64>>], bodies: &[Body]) -> Self {
        let steps = positions.first().map_or(0, Vec::len);
        let dt = DEFAULT_DT;

        let velocities: Vec<Vec<Vector3<f64>>> = positions
            .iter()
            .map(|body| {
                (0..steps)
                    .map(|step| {
                        let next = (step + 1).min(steps.saturating_sub(1));
                        let prev = step.saturating_sub(1);
                        let span = ((next - prev).max(1)) as f64 * dt;
                        (body[next] - body[prev]) / span
                    })
                    .collect()
            })
            .collect();

        let speeds: Vec<Vec<f64>> =
            velocities.iter().map(|body| body.iter().map(Vector3::norm).collect()).collect();

        let pairwise = PAIRS.map(|(a, b)| {
            (0..steps).map(|step| (positions[a][step] - positions[b][step]).norm()).collect()
        });

        let mut masses = [0.0; 3];
        for (slot, body) in masses.iter_mut().zip(bodies.iter()) {
            *slot = body.mass;
        }

        Self { velocities, speeds, pairwise, masses }
    }

    /// Speed normalization window: the (low, high) quantiles of a strided
    /// pooled speed sample, mirroring the production velocity-dynamics
    /// normalization (quantiles 0.15 / 0.97).
    #[must_use]
    pub fn speed_window(&self) -> (f64, f64) {
        let mut sample: Vec<f64> = self
            .speeds
            .iter()
            .flat_map(|body| body.iter().step_by(97).copied())
            .filter(|value| value.is_finite())
            .collect();
        if sample.is_empty() {
            return (0.0, 1.0);
        }
        sample.sort_by(f64::total_cmp);
        let low = sample[((sample.len() - 1) as f64 * 0.15) as usize];
        let high = sample[((sample.len() - 1) as f64 * 0.97) as usize];
        (low, high.max(low + 1e-12))
    }

    /// Normalize a speed into `[0, 1]` against [`Self::speed_window`].
    #[must_use]
    pub fn normalized_speed(&self, window: (f64, f64), speed: f64) -> f64 {
        ((speed - window.0) / (window.1 - window.0)).clamp(0.0, 1.0)
    }
}

/// Leading eigenvector of the 2x2 covariance of all projected xy positions
/// (the trajectory's principal separation axis, unit length).
#[must_use]
pub fn principal_axis_xy(positions: &[Vec<Vector3<f64>>]) -> (f64, f64) {
    let mut mean = (0.0_f64, 0.0_f64);
    let mut count = 0.0_f64;
    for body in positions {
        for point in body.iter().step_by(97) {
            mean.0 += point.x;
            mean.1 += point.y;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return (1.0, 0.0);
    }
    mean.0 /= count;
    mean.1 /= count;

    let (mut cxx, mut cxy, mut cyy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for body in positions {
        for point in body.iter().step_by(97) {
            let dx = point.x - mean.0;
            let dy = point.y - mean.1;
            cxx += dx * dx;
            cxy += dx * dy;
            cyy += dy * dy;
        }
    }
    let trace_half = 0.5 * (cxx + cyy);
    let det = cxx * cyy - cxy * cxy;
    let lambda = trace_half + (trace_half * trace_half - det).max(0.0).sqrt();
    let (vx, vy) = if cxy.abs() > 1e-15 { (lambda - cyy, cxy) } else { (1.0, 0.0) };
    let norm = (vx * vx + vy * vy).sqrt().max(1e-15);
    (vx / norm, vy / norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let phase = f64::from(body as u32) * 2.0;
                (0..steps)
                    .map(|step| {
                        let t = step as f64 * DEFAULT_DT;
                        Vector3::new((t + phase).cos(), (t + phase).sin(), 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn unit_circle_speed_is_one() {
        let positions = circle_positions(1000);
        let bodies = vec![
            Body::new(1.0, Vector3::zeros(), Vector3::zeros()),
            Body::new(2.0, Vector3::zeros(), Vector3::zeros()),
            Body::new(3.0, Vector3::zeros(), Vector3::zeros()),
        ];
        let kin = Kinematics::compute(&positions, &bodies);
        // Angular velocity 1 rad/unit-time on a unit circle -> speed 1.
        let mid = kin.speeds[0][500];
        assert!((mid - 1.0).abs() < 1e-3, "expected ~1.0, got {mid}");
        assert_eq!(kin.masses, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn pairwise_matches_direct_distance() {
        let positions = circle_positions(64);
        let bodies = vec![
            Body::new(1.0, Vector3::zeros(), Vector3::zeros()),
            Body::new(1.0, Vector3::zeros(), Vector3::zeros()),
            Body::new(1.0, Vector3::zeros(), Vector3::zeros()),
        ];
        let kin = Kinematics::compute(&positions, &bodies);
        let direct = (positions[0][10] - positions[1][10]).norm();
        assert!((kin.pairwise[0][10] - direct).abs() < 1e-12);
    }
}
