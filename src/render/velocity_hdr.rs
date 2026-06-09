//! Velocity-modulated stroke dynamics for dramatic motion contrast.
//!
//! Earlier builds compared body speeds against a fixed absolute threshold.
//! Because gravitationally bound orbits in this simulation move tens of times
//! faster than that threshold, every segment saturated to the maximum boost
//! and the intended flare contrast was lost. This module instead normalizes
//! speed *within each orbit* (percentile-based), so every seed exhibits the
//! full dynamic range: slow apoapsis arcs render bold and quiet, periapsis
//! whips render as thin brilliant flares.

use nalgebra::Vector3;

use super::constants::{
    VELOCITY_FLARE_GAMMA, VELOCITY_HDR_BOOST_FACTOR, VELOCITY_NORM_HIGH_QUANTILE,
    VELOCITY_NORM_LOW_QUANTILE, VELOCITY_THICKNESS_FAST, VELOCITY_THICKNESS_SLOW,
};

/// Maximum number of strided speed samples used for percentile estimation.
const MAX_SPEED_SAMPLES_PER_BODY: usize = 4096;

/// Normalized speed assigned when an orbit has no usable speed spread.
const DEGENERATE_MOVING_NORM: f64 = 0.6;

/// Per-segment stroke dynamics derived from orbit-normalized speed.
#[derive(Clone, Copy, Debug)]
pub struct SegmentDynamics {
    /// Energy multiplier in `[1, VELOCITY_HDR_BOOST_FACTOR]`; fast segments flare.
    pub hdr_multiplier: f64,
    /// Line width multiplier; slow segments are bold, fast segments are hairline.
    pub thickness_factor: f64,
}

impl SegmentDynamics {
    /// Neutral dynamics (no flare, unit thickness).
    pub const NEUTRAL: Self = Self { hdr_multiplier: 1.0, thickness_factor: 1.0 };
}

/// Calculator for orbit-relative velocity dynamics.
pub struct VelocityHdrCalculator<'a> {
    positions: &'a [Vec<Vector3<f64>>],
    dt: f64,
    v_low: f64,
    inv_span: f64,
}

impl<'a> VelocityHdrCalculator<'a> {
    /// Build a calculator from per-body position trajectories and timestep `dt`.
    ///
    /// Scans the trajectories (strided, deterministic) to estimate the orbit's
    /// own speed distribution, then maps the low/high percentiles onto the
    /// normalized dynamics range.
    #[must_use]
    pub fn new(positions: &'a [Vec<Vector3<f64>>], dt: f64) -> Self {
        let (v_low, inv_span) = Self::speed_normalization(positions, dt);
        Self { positions, dt, v_low, inv_span }
    }

    fn speed_normalization(positions: &[Vec<Vector3<f64>>], dt: f64) -> (f64, f64) {
        let steps = positions.first().map_or(0, Vec::len);
        if steps < 2 || dt <= 0.0 {
            return (0.0, 0.0);
        }

        let stride = ((steps - 1) / MAX_SPEED_SAMPLES_PER_BODY).max(1);
        let mut speeds = Vec::with_capacity(positions.len() * ((steps - 1) / stride + 1));
        for body in positions {
            let mut step = 0;
            while step + 1 < body.len() {
                speeds.push((body[step + 1] - body[step]).norm() / dt);
                step += stride;
            }
        }

        if speeds.len() < 2 {
            return (0.0, 0.0);
        }

        speeds.sort_by(f64::total_cmp);
        let quantile = |q: f64| -> f64 {
            let idx = (q * (speeds.len() - 1) as f64).round() as usize;
            speeds[idx.min(speeds.len() - 1)]
        };
        let low = quantile(VELOCITY_NORM_LOW_QUANTILE);
        let high = quantile(VELOCITY_NORM_HIGH_QUANTILE);
        let span = high - low;
        if span <= 1e-12 || !span.is_finite() {
            return (low, 0.0);
        }

        (low, 1.0 / span)
    }

    /// Orbit-normalized speed in `[0, 1]` for `body` at `step`.
    #[must_use]
    #[inline]
    pub fn normalized_body_speed(&self, step: usize, body: usize) -> f64 {
        if step + 1 >= self.positions[body].len() || self.dt <= 0.0 {
            return 0.0;
        }
        let speed = (self.positions[body][step + 1] - self.positions[body][step]).norm() / self.dt;
        self.normalize_speed(speed)
    }

    #[inline]
    fn normalize_speed(&self, speed: f64) -> f64 {
        if self.inv_span > 0.0 {
            ((speed - self.v_low) * self.inv_span).clamp(0.0, 1.0)
        } else if speed > 1e-12 {
            DEGENERATE_MOVING_NORM
        } else {
            0.0
        }
    }

    /// Stroke dynamics for the edge between `body0` and `body1` at `step`.
    #[must_use]
    #[inline]
    pub fn segment_dynamics(&self, step: usize, body0: usize, body1: usize) -> SegmentDynamics {
        if step + 1 >= self.positions[0].len() {
            return SegmentDynamics::NEUTRAL;
        }
        let norm = 0.5
            * (self.normalized_body_speed(step, body0) + self.normalized_body_speed(step, body1));
        dynamics_from_normalized_speed(norm)
    }

    /// Stroke dynamics for a single body's own trail at `step` (ribbons/spokes).
    #[must_use]
    #[inline]
    pub fn body_dynamics(&self, step: usize, body: usize) -> SegmentDynamics {
        if step + 1 >= self.positions[body].len() {
            return SegmentDynamics::NEUTRAL;
        }
        dynamics_from_normalized_speed(self.normalized_body_speed(step, body))
    }
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Map an orbit-normalized speed in `[0, 1]` to stroke dynamics.
#[must_use]
#[inline]
pub fn dynamics_from_normalized_speed(norm: f64) -> SegmentDynamics {
    let s = smoothstep(norm);
    let hdr_multiplier = 1.0 + s.powf(VELOCITY_FLARE_GAMMA) * (VELOCITY_HDR_BOOST_FACTOR - 1.0);
    let thickness_factor =
        VELOCITY_THICKNESS_SLOW + (VELOCITY_THICKNESS_FAST - VELOCITY_THICKNESS_SLOW) * s;
    SegmentDynamics { hdr_multiplier, thickness_factor }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic two-phase orbit: slow drift for the first half, fast sweep after.
    fn two_speed_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        let make_body = |offset: f64| -> Vec<Vector3<f64>> {
            let mut x = 0.0;
            (0..steps)
                .map(|step| {
                    let speed = if step < steps / 2 { 0.001 } else { 0.05 };
                    x += speed;
                    Vector3::new(x, offset, 0.0)
                })
                .collect()
        };
        vec![make_body(0.0), make_body(1.0), make_body(2.0)]
    }

    #[test]
    fn test_slow_phase_gets_less_boost_than_fast_phase() {
        let positions = two_speed_positions(512);
        let calc = VelocityHdrCalculator::new(&positions, 0.001);

        let slow = calc.segment_dynamics(10, 0, 1);
        let fast = calc.segment_dynamics(400, 0, 1);

        assert!(
            fast.hdr_multiplier > slow.hdr_multiplier + 1.0,
            "fast phase should flare brighter: slow={} fast={}",
            slow.hdr_multiplier,
            fast.hdr_multiplier
        );
        assert!(
            fast.thickness_factor < slow.thickness_factor,
            "fast phase should draw thinner lines: slow={} fast={}",
            slow.thickness_factor,
            fast.thickness_factor
        );
    }

    #[test]
    fn test_dynamics_bounds() {
        for i in 0..=20 {
            let norm = f64::from(i) / 20.0;
            let dynamics = dynamics_from_normalized_speed(norm);
            assert!(
                (1.0..=VELOCITY_HDR_BOOST_FACTOR).contains(&dynamics.hdr_multiplier),
                "hdr multiplier out of range at norm {norm}: {}",
                dynamics.hdr_multiplier
            );
            let (lo, hi) = (
                VELOCITY_THICKNESS_FAST.min(VELOCITY_THICKNESS_SLOW),
                VELOCITY_THICKNESS_FAST.max(VELOCITY_THICKNESS_SLOW),
            );
            assert!(
                (lo..=hi).contains(&dynamics.thickness_factor),
                "thickness factor out of range at norm {norm}: {}",
                dynamics.thickness_factor
            );
        }
    }

    #[test]
    fn test_full_dynamic_range_is_used_within_an_orbit() {
        let positions = two_speed_positions(2048);
        let calc = VelocityHdrCalculator::new(&positions, 0.001);

        let mut min_mult = f64::INFINITY;
        let mut max_mult = 0.0f64;
        for step in 0..2047 {
            let d = calc.segment_dynamics(step, 0, 1);
            min_mult = min_mult.min(d.hdr_multiplier);
            max_mult = max_mult.max(d.hdr_multiplier);
        }

        assert!(min_mult < 1.5, "slow arcs should be near 1x, got {min_mult}");
        assert!(
            max_mult > VELOCITY_HDR_BOOST_FACTOR * 0.85,
            "fast whips should approach max boost, got {max_mult}"
        );
    }

    #[test]
    fn test_degenerate_constant_speed_orbit_is_stable() {
        let steps = 64u32;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..steps)
                    .map(|step| Vector3::new(f64::from(step) * 0.01, f64::from(body), 0.0))
                    .collect()
            })
            .collect();
        let calc = VelocityHdrCalculator::new(&positions, 0.001);
        let d = calc.segment_dynamics(5, 0, 1);

        assert!(d.hdr_multiplier.is_finite());
        assert!(d.hdr_multiplier >= 1.0);
        assert!(d.thickness_factor.is_finite());
    }

    #[test]
    fn test_last_step_returns_neutral_dynamics() {
        let positions = two_speed_positions(16);
        let calc = VelocityHdrCalculator::new(&positions, 0.001);
        let d = calc.segment_dynamics(15, 0, 1);
        assert_eq!(d.hdr_multiplier, 1.0);
        assert_eq!(d.thickness_factor, 1.0);
    }

    #[test]
    fn test_body_dynamics_tracks_individual_speed() {
        let steps = 256u32;
        // Body 0 crawls; body 1 races.
        let positions: Vec<Vec<Vector3<f64>>> = vec![
            (0..steps).map(|s| Vector3::new(f64::from(s) * 0.0005, 0.0, 0.0)).collect(),
            (0..steps).map(|s| Vector3::new(f64::from(s) * 0.05, 1.0, 0.0)).collect(),
            (0..steps).map(|s| Vector3::new(f64::from(s) * 0.01, 2.0, 0.0)).collect(),
        ];
        let calc = VelocityHdrCalculator::new(&positions, 0.001);

        let slow = calc.body_dynamics(100, 0);
        let fast = calc.body_dynamics(100, 1);
        assert!(
            fast.hdr_multiplier > slow.hdr_multiplier,
            "faster body should flare more: slow={} fast={}",
            slow.hdr_multiplier,
            fast.hdr_multiplier
        );
    }
}
