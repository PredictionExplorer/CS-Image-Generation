use crate::sim::Sha3RandomByteStream;
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
use tracing::warn;

/// Trait for applying drift transformations to position data
pub trait DriftTransform {
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], dt: f64);
}

/// Shared configuration for every drift strategy.
#[derive(Clone, Copy, Debug)]
pub struct DriftParameters {
    pub scale: f64,
    pub arc_fraction: f64,
    pub eccentricity: f64,
}

impl DriftParameters {
    pub fn new(scale: f64, arc_fraction: f64, eccentricity: f64) -> Self {
        let clamped_scale = scale.max(0.0);
        let clamped_arc = arc_fraction.clamp(0.0, 1.0);
        let clamped_ecc = eccentricity.clamp(0.0, 0.95);

        if (arc_fraction - clamped_arc).abs() > f64::EPSILON {
            warn!(
                original = arc_fraction,
                clamped = clamped_arc,
                "drift_arc_fraction out of range [0, 1]; clamping"
            );
        }
        if (eccentricity - clamped_ecc).abs() > f64::EPSILON {
            warn!(
                original = eccentricity,
                clamped = clamped_ecc,
                "drift_orbit_eccentricity out of range [0, 0.95]; clamping"
            );
        }

        Self { scale: clamped_scale, arc_fraction: clamped_arc, eccentricity: clamped_ecc }
    }

    #[inline]
    pub fn sweep_radians(&self) -> f64 {
        self.arc_fraction * crate::render::constants::TWO_PI
    }
}

/// No drift - positions remain unchanged
pub struct NoDrift;

impl DriftTransform for NoDrift {
    fn apply(&mut self, _positions: &mut [Vec<Vector3<f64>>], _dt: f64) {
        // Do nothing
    }
}

/// Brownian drift - random walk motion with pre-generated random values
pub struct BrownianDrift {
    displacements: Vec<Vector3<f64>>,
}

impl BrownianDrift {
    pub fn new(rng: &mut Sha3RandomByteStream, scale: f64, num_steps: usize) -> Self {
        let dt_sqrt = 0.001f64.sqrt(); // Using known dt value
        let mut displacements = Vec::with_capacity(num_steps);

        for _ in 0..num_steps {
            // Generate 3D Gaussian displacement using Box-Muller transform
            let dx = Self::gaussian_from_rng(rng) * scale * dt_sqrt;
            let dy = Self::gaussian_from_rng(rng) * scale * dt_sqrt;
            let dz = Self::gaussian_from_rng(rng) * scale * dt_sqrt;

            displacements.push(Vector3::new(dx, dy, dz));
        }

        Self { displacements }
    }

    /// Generate a Gaussian random number using Box-Muller transform
    fn gaussian_from_rng(rng: &mut Sha3RandomByteStream) -> f64 {
        // Box-Muller transform: convert two uniform [0,1] to two Gaussian N(0,1)
        let u1 = rng.next_f64();
        let u2 = rng.next_f64();

        // Avoid log(0)
        let u1 = u1.max(1e-10);

        let r = (-crate::render::constants::GAUSSIAN_TWO_FACTOR * u1.ln()).sqrt();
        let theta = crate::render::constants::TWO_PI * u2;

        r * theta.cos() // Return one of the two generated values
    }
}

impl DriftTransform for BrownianDrift {
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], _dt: f64) {
        if positions.is_empty() || positions[0].is_empty() {
            return;
        }

        let steps = positions[0].len().min(self.displacements.len());
        let mut offset = Vector3::zeros();

        // Apply Brownian motion: each step adds a random displacement
        for step in 0..steps {
            // Accumulate offset
            offset += self.displacements[step];

            // Apply the same offset to all bodies at this timestep
            for body_positions in positions.iter_mut() {
                body_positions[step] += offset;
            }
        }
    }
}

/// Linear drift - constant velocity motion
pub struct LinearDrift {
    velocity: Vector3<f64>,
}

impl LinearDrift {
    pub fn new(rng: &mut Sha3RandomByteStream, scale: f64) -> Self {
        // Random spherical coordinates
        let theta = rng.next_f64() * PI; // polar angle [0, π]
        let phi = rng.next_f64() * crate::render::constants::TWO_PI; // azimuthal angle [0, 2π]
        let speed = scale;

        let velocity = Vector3::new(
            speed * theta.sin() * phi.cos(),
            speed * theta.sin() * phi.sin(),
            speed * theta.cos(),
        );

        Self { velocity }
    }
}

impl DriftTransform for LinearDrift {
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], dt: f64) {
        if positions.is_empty() || positions[0].is_empty() {
            return;
        }

        let steps = positions[0].len();

        for step in 0..steps {
            let offset = self.velocity * (step as f64) * dt;

            // Apply the same offset to all bodies at this timestep
            for body_positions in positions.iter_mut() {
                body_positions[step] += offset;
            }
        }
    }
}

/// Elliptical drift - deterministic heliocentric-arc motion
pub struct EllipticalDrift {
    params: DriftParameters,
    rotation: Matrix3<f64>,
    initial_mean_anomaly: f64,
    sweep_radians: f64,
}

impl EllipticalDrift {
    pub fn new(rng: &mut Sha3RandomByteStream, params: DriftParameters) -> Self {
        let inclination = rng.next_f64() * PI;
        let ascending_node = rng.next_f64() * crate::render::constants::TWO_PI;
        let argument_of_periapsis = rng.next_f64() * crate::render::constants::TWO_PI;
        let rotation = build_rotation_matrix(ascending_node, inclination, argument_of_periapsis);

        let initial_mean_anomaly = rng.next_f64() * crate::render::constants::TWO_PI - PI;
        let sweep_radians = params.sweep_radians();

        Self { params, rotation, initial_mean_anomaly, sweep_radians }
    }
}

impl DriftTransform for EllipticalDrift {
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], dt: f64) {
        if positions.is_empty() || positions[0].len() < 2 {
            return;
        }

        if self.sweep_radians.abs() <= f64::EPSILON || self.params.scale <= 0.0 {
            return;
        }

        let (semi_major, semi_minor) = orbital_axes(positions, self.params);
        if semi_major <= f64::EPSILON || semi_minor <= f64::EPSILON {
            return;
        }

        let eccentricity = self.params.eccentricity;
        let total_steps = positions[0].len();
        let total_duration = dt * (total_steps.saturating_sub(1) as f64).max(dt);
        let mean_motion =
            if total_duration > 0.0 { self.sweep_radians / total_duration } else { 0.0 };

        for step in 0..total_steps {
            let time = step as f64 * dt;
            let mean_anomaly = normalize_angle(self.initial_mean_anomaly + mean_motion * time);
            let eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity);

            let x = semi_major * (eccentric_anomaly.cos() - eccentricity);
            let y = semi_minor * eccentric_anomaly.sin();
            let orbital_plane = Vector3::new(x, y, 0.0);
            let offset = self.rotation * orbital_plane;

            for body_positions in positions.iter_mut() {
                body_positions[step] += offset;
            }
        }
    }
}

/// Linearly interpolate between two position sets.
///
/// At `t = 0.0` the result equals `a`; at `t = 1.0` it equals `b`.
/// Used to control how much the 3D camera tracks drift motion
/// (`drift_camera_coupling`).
pub fn lerp_positions(
    a: &[Vec<Vector3<f64>>],
    b: &[Vec<Vector3<f64>>],
    t: f64,
) -> Vec<Vec<Vector3<f64>>> {
    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    a.iter()
        .zip(b.iter())
        .map(|(body_a, body_b)| {
            body_a
                .iter()
                .zip(body_b.iter())
                .map(|(pa, pb)| pa * one_minus_t + pb * t)
                .collect()
        })
        .collect()
}

/// Parse drift mode from string
pub fn parse_drift_mode(
    mode: &str,
    rng: &mut Sha3RandomByteStream,
    params: DriftParameters,
    num_steps: usize,
) -> Box<dyn DriftTransform> {
    match mode.to_lowercase().as_str() {
        "none" => Box::new(NoDrift),
        "brownian" => Box::new(BrownianDrift::new(rng, params.scale, num_steps)),
        "linear" => Box::new(LinearDrift::new(rng, params.scale)),
        "elliptical" | "ellipse" => Box::new(EllipticalDrift::new(rng, params)),
        _ => {
            warn!("Unknown drift mode '{}'", mode);
            Box::new(BrownianDrift::new(rng, params.scale, num_steps))
        }
    }
}

fn orbital_axes(positions: &[Vec<Vector3<f64>>], params: DriftParameters) -> (f64, f64) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    let mut min_z = f64::MAX;
    let mut max_z = f64::MIN;

    for body in positions {
        for p in body {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
            min_z = min_z.min(p.z);
            max_z = max_z.max(p.z);
        }
    }

    let span_x = (max_x - min_x).abs();
    let span_y = (max_y - min_y).abs();
    let span_z = (max_z - min_z).abs();
    let reference_span = span_x.max(span_y).max(span_z).max(1e-6);

    let semi_major = 0.5 * reference_span * params.scale.max(1e-6);
    let semi_minor = semi_major * (1.0 - params.eccentricity * params.eccentricity).sqrt();

    (semi_major, semi_minor.max(1e-6))
}

fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % crate::render::constants::TWO_PI;
    if a > PI {
        a -= crate::render::constants::TWO_PI;
    } else if a < -PI {
        a += crate::render::constants::TWO_PI;
    }
    a
}

fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    if eccentricity.abs() <= f64::EPSILON {
        return mean_anomaly;
    }

    let mut eccentric_anomaly = mean_anomaly;
    for _ in 0..8 {
        let f = eccentric_anomaly - eccentricity * eccentric_anomaly.sin() - mean_anomaly;
        let f_prime = 1.0 - eccentricity * eccentric_anomaly.cos();
        if f_prime.abs() <= f64::EPSILON {
            break;
        }
        let delta = f / f_prime;
        eccentric_anomaly -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    eccentric_anomaly
}

fn build_rotation_matrix(
    ascending_node: f64,
    inclination: f64,
    argument_periapsis: f64,
) -> Matrix3<f64> {
    let (sin_omega, cos_omega) = ascending_node.sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    let (sin_w, cos_w) = argument_periapsis.sin_cos();

    let r11 = cos_omega * cos_w - sin_omega * sin_w * cos_i;
    let r12 = -cos_omega * sin_w - sin_omega * cos_w * cos_i;
    let r13 = sin_omega * sin_i;

    let r21 = sin_omega * cos_w + cos_omega * sin_w * cos_i;
    let r22 = -sin_omega * sin_w + cos_omega * cos_w * cos_i;
    let r23 = -cos_omega * sin_i;

    let r31 = sin_w * sin_i;
    let r32 = cos_w * sin_i;
    let r33 = cos_i;

    Matrix3::new(r11, r12, r13, r21, r22, r23, r31, r32, r33)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bodies() -> Vec<Vec<Vector3<f64>>> {
        vec![
            vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
            ],
            vec![
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            vec![
                Vector3::new(-1.0, -1.0, 0.0),
                Vector3::new(-1.0, -1.0, 0.0),
                Vector3::new(-1.0, -1.0, 0.0),
            ],
        ]
    }

    fn make_rng() -> Sha3RandomByteStream {
        let seed = [0x42u8; 32];
        Sha3RandomByteStream::new(&seed, 1.0, 2.0, 1.0, 1.0)
    }

    #[test]
    fn elliptical_drift_moves_bodies() {
        let mut positions = test_bodies();
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.25, 0.1);
        let mut drift = EllipticalDrift::new(&mut rng, params);
        drift.apply(&mut positions, 0.001);

        let initial = Vector3::new(1.0, 0.0, 0.0);
        assert_ne!(positions[0][1], initial);
    }

    #[test]
    fn zero_arc_fraction_yields_no_motion() {
        let mut positions = test_bodies();
        let positions_clone = positions.clone();
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.0, 0.1);
        let mut drift = EllipticalDrift::new(&mut rng, params);
        drift.apply(&mut positions, 0.001);
        assert_eq!(positions, positions_clone);
    }

    #[test]
    fn test_drift_parameters_clamps_values() {
        let neg_scale = DriftParameters::new(-2.0, 0.5, 0.3);
        assert_eq!(neg_scale.scale, 0.0);
        assert_eq!(neg_scale.arc_fraction, 0.5);
        assert_eq!(neg_scale.eccentricity, 0.3);

        let low_arc = DriftParameters::new(1.0, -0.25, 0.2);
        assert_eq!(low_arc.arc_fraction, 0.0);

        let high_arc = DriftParameters::new(1.0, 1.5, 0.2);
        assert_eq!(high_arc.arc_fraction, 1.0);

        let low_ecc = DriftParameters::new(1.0, 0.5, -0.5);
        assert_eq!(low_ecc.eccentricity, 0.0);

        let high_ecc = DriftParameters::new(1.0, 0.5, 1.5);
        assert_eq!(high_ecc.eccentricity, 0.95);
    }

    #[test]
    fn test_brownian_drift_changes_positions() {
        let original = test_bodies();
        let mut positions = original.clone();
        let mut rng = make_rng();
        let mut drift = BrownianDrift::new(&mut rng, 1.0, 3);
        drift.apply(&mut positions, 0.001);
        assert_ne!(positions, original);
    }

    #[test]
    fn test_linear_drift_changes_positions() {
        let original = test_bodies();
        let mut positions = original.clone();
        let mut rng = make_rng();
        let mut drift = LinearDrift::new(&mut rng, 1.0);
        drift.apply(&mut positions, 0.001);
        assert_ne!(positions, original);
    }

    #[test]
    fn test_parse_drift_mode_creates_correct_type() {
        let params = DriftParameters::new(1.0, 0.25, 0.1);
        for mode in ["linear", "brownian", "elliptical"] {
            let original = test_bodies();
            let mut positions = original.clone();
            let mut rng = make_rng();
            let mut drift = parse_drift_mode(mode, &mut rng, params, 3);
            drift.apply(&mut positions, 0.001);
            assert_ne!(positions, original, "drift mode {mode} should move bodies");
        }
    }

    #[test]
    fn lerp_positions_at_zero_returns_a() {
        let a = test_bodies();
        let b = vec![
            vec![Vector3::new(10.0, 10.0, 10.0); 3],
            vec![Vector3::new(20.0, 20.0, 20.0); 3],
            vec![Vector3::new(30.0, 30.0, 30.0); 3],
        ];
        let result = super::lerp_positions(&a, &b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn lerp_positions_at_one_returns_b() {
        let a = test_bodies();
        let b = vec![
            vec![Vector3::new(10.0, 10.0, 10.0); 3],
            vec![Vector3::new(20.0, 20.0, 20.0); 3],
            vec![Vector3::new(30.0, 30.0, 30.0); 3],
        ];
        let result = super::lerp_positions(&a, &b, 1.0);
        assert_eq!(result, b);
    }

    #[test]
    fn lerp_positions_at_half_is_midpoint() {
        let a = vec![vec![Vector3::new(0.0, 0.0, 0.0); 1]];
        let b = vec![vec![Vector3::new(10.0, 20.0, 30.0); 1]];
        let result = super::lerp_positions(&a, &b, 0.5);
        let expected = Vector3::new(5.0, 10.0, 15.0);
        assert!((result[0][0] - expected).norm() < 1e-12);
    }

    #[test]
    fn lerp_positions_clamps_out_of_range() {
        let a = vec![vec![Vector3::new(0.0, 0.0, 0.0); 1]];
        let b = vec![vec![Vector3::new(10.0, 10.0, 10.0); 1]];
        let below = super::lerp_positions(&a, &b, -0.5);
        let above = super::lerp_positions(&a, &b, 1.5);
        assert_eq!(below, a, "t < 0 should clamp to a");
        assert_eq!(above, b, "t > 1 should clamp to b");
    }

    #[test]
    fn test_elliptical_drift_determinism() {
        let params = DriftParameters::new(1.0, 0.25, 0.1);

        let mut positions_a = test_bodies();
        let mut rng_a = make_rng();
        let mut drift_a = EllipticalDrift::new(&mut rng_a, params);
        drift_a.apply(&mut positions_a, 0.001);

        let mut positions_b = test_bodies();
        let mut rng_b = make_rng();
        let mut drift_b = EllipticalDrift::new(&mut rng_b, params);
        drift_b.apply(&mut positions_b, 0.001);

        assert_eq!(positions_a, positions_b);
    }
}
