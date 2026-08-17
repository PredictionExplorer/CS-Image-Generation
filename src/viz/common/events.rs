//! Orbit event detection: syzygies (three-body alignments), pairwise
//! periapses, and the closest triple approach.
//!
//! All detectors are deterministic functions of the trajectory. Constants
//! follow `docs/VIZ_MASTER_PLAN.md` Part II.2.

use super::kinematics::{Kinematics, PAIRS};
use nalgebra::Vector3;

/// Collinearity quality below which a configuration counts as a syzygy.
const SYZYGY_QUALITY_THRESHOLD: f64 = 0.02;
/// Minimum step separation between reported syzygies.
const SYZYGY_MIN_GAP_STEPS: usize = 500;
/// Non-maximum-suppression window for periapsis events (steps).
const PERIAPSIS_NMS_STEPS: usize = 2_500;
/// Percentile of pair distance below which minima qualify as periapses.
const PERIAPSIS_PERCENTILE: f64 = 0.10;

/// One three-body alignment event.
#[derive(Clone, Copy, Debug)]
pub struct Syzygy {
    /// Simulation step of the alignment.
    pub step: usize,
    /// Index of the body lying between the other two.
    pub middle_body: usize,
    /// Alignment sharpness in `[0, 1]` (1 = perfectly collinear).
    pub sharpness: f64,
}

/// One pairwise close-approach event.
#[derive(Clone, Copy, Debug)]
pub struct Approach {
    /// Simulation step of the local distance minimum.
    pub step: usize,
    /// The approaching pair (canonical [`PAIRS`] indices).
    pub pair: (usize, usize),
    /// Separation at the minimum.
    pub distance: f64,
}

/// Detected orbit events for one run.
pub struct Events {
    /// Alignment events in step order.
    pub syzygies: Vec<Syzygy>,
    /// Close approaches across all pairs, in step order.
    pub periapses: Vec<Approach>,
    /// Step at which the summed pairwise separation is smallest.
    pub closest_triple: usize,
}

/// Normalized triangle quality: `4*sqrt(3)*area / (a^2 + b^2 + c^2)`
/// (1 for equilateral, 0 for collinear).
fn triangle_quality(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let area = 0.5 * ab.cross(&ac).norm();
    let sum_sq = (b - a).norm_squared() + (c - b).norm_squared() + (a - c).norm_squared();
    if sum_sq <= 1e-24 {
        return 0.0;
    }
    (4.0 * 3.0_f64.sqrt() * area / sum_sq).clamp(0.0, 1.0)
}

/// Index of the body between the other two in a near-collinear configuration.
fn middle_body(points: [Vector3<f64>; 3]) -> usize {
    for candidate in 0..3 {
        let other_a = points[(candidate + 1) % 3];
        let other_b = points[(candidate + 2) % 3];
        if (other_a - points[candidate]).dot(&(other_b - points[candidate])) < 0.0 {
            return candidate;
        }
    }
    0
}

impl Events {
    /// Detect all events from the trajectory and derived kinematics.
    #[must_use]
    pub fn detect(positions: &[Vec<Vector3<f64>>], kinematics: &Kinematics) -> Self {
        let steps = positions.first().map_or(0, Vec::len);

        // --- Syzygies: local minima of triangle quality below threshold.
        let mut syzygies = Vec::new();
        let mut last_syzygy: Option<usize> = None;
        let quality: Vec<f64> = (0..steps)
            .map(|step| {
                triangle_quality(positions[0][step], positions[1][step], positions[2][step])
            })
            .collect();
        for step in 1..steps.saturating_sub(1) {
            let q = quality[step];
            if q < SYZYGY_QUALITY_THRESHOLD && q <= quality[step - 1] && q <= quality[step + 1] {
                if last_syzygy.is_some_and(|prev| step - prev < SYZYGY_MIN_GAP_STEPS) {
                    continue;
                }
                let config = [positions[0][step], positions[1][step], positions[2][step]];
                syzygies.push(Syzygy {
                    step,
                    middle_body: middle_body(config),
                    sharpness: 1.0 - q / SYZYGY_QUALITY_THRESHOLD,
                });
                last_syzygy = Some(step);
            }
        }

        // --- Periapses per pair: local minima below the 10th percentile.
        let mut periapses = Vec::new();
        for (pair_index, &pair) in PAIRS.iter().enumerate() {
            let series = &kinematics.pairwise[pair_index];
            let mut sorted: Vec<f64> = series.iter().step_by(37).copied().collect();
            if sorted.is_empty() {
                continue;
            }
            sorted.sort_by(f64::total_cmp);
            let threshold = sorted[((sorted.len() - 1) as f64 * PERIAPSIS_PERCENTILE) as usize];

            let mut last_event: Option<usize> = None;
            for step in 1..steps.saturating_sub(1) {
                let d = series[step];
                if d < threshold && d <= series[step - 1] && d <= series[step + 1] {
                    if last_event.is_some_and(|prev| step - prev < PERIAPSIS_NMS_STEPS) {
                        continue;
                    }
                    periapses.push(Approach { step, pair, distance: d });
                    last_event = Some(step);
                }
            }
        }
        periapses.sort_by_key(|event| event.step);

        // --- Closest triple approach.
        let closest_triple = (0..steps)
            .min_by(|&a, &b| {
                let sum_a: f64 = kinematics.pairwise.iter().map(|series| series[a]).sum();
                let sum_b: f64 = kinematics.pairwise.iter().map(|series| series[b]).sum();
                sum_a.total_cmp(&sum_b)
            })
            .unwrap_or(0);

        Self { syzygies, periapses, closest_triple }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_is_one_for_equilateral_and_zero_for_collinear() {
        let equilateral = triangle_quality(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 3.0_f64.sqrt() / 2.0, 0.0),
        );
        assert!((equilateral - 1.0).abs() < 1e-12);
        let collinear = triangle_quality(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        );
        assert!(collinear < 1e-12);
    }

    #[test]
    fn middle_body_is_detected() {
        let points = [
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.1, 0.01, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        assert_eq!(middle_body(points), 1);
    }
}
