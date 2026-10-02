use crate::error::ConfigError;
use crate::sim::Sha3RandomByteStream;
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
use tracing::warn;

/// Trait for applying drift transformations to position data
pub trait DriftTransform {
    /// Apply the drift transformation to all body positions and report what was added.
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], dt: f64) -> AppliedDrift;
}

/// What a [`DriftTransform`] added to the positions: the same offset for every body at each
/// step. The ember edition re-applies it from these values to follow the same motion
/// (`ember::View`), so each variant holds the resolved quantities, not the configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppliedDrift {
    /// Nothing was added: no drift, or a drift whose parameters made it a no-op.
    None,
    /// `offset[step] = (velocity * step) * dt`.
    Linear {
        /// Drift velocity.
        velocity: Vector3<f64>,
    },
    /// A random walk; its per-step offsets are not recorded.
    Brownian,
    /// `offset[step] = rotation * (a * (cos E - e), b * sin E, 0)` with `E` the eccentric
    /// anomaly of the mean anomaly `initial_mean_anomaly + mean_motion * (step * dt)`.
    Elliptical {
        /// Orientation of the drift ellipse.
        rotation: Matrix3<f64>,
        /// Mean anomaly at step 0, in radians.
        initial_mean_anomaly: f64,
        /// Mean motion in radians per unit of simulation time.
        mean_motion: f64,
        /// Orbital eccentricity `e`.
        eccentricity: f64,
        /// Semi-major axis `a`.
        semi_major: f64,
        /// Semi-minor axis `b`.
        semi_minor: f64,
    },
}

/// Maximum supported drift sweep, in fractions of a full rotation.
///
/// Values above 1 sweep more than a complete loop, producing long smeary
/// camera arcs; the cap keeps explicit user values from degenerating into
/// many indistinct revolutions.
pub const MAX_ARC_FRACTION: f64 = 1.5;

/// Shared configuration for every drift strategy.
#[derive(Clone, Copy, Debug)]
pub struct DriftParameters {
    /// Non-negative multiplier for drift magnitude and elliptical orbit size.
    pub scale: f64,
    /// Fraction of one full rotation (0–[`MAX_ARC_FRACTION`]) swept during elliptical drift.
    pub arc_fraction: f64,
    /// Orbital eccentricity for elliptical drift (0 is circular; clamped below 1).
    pub eccentricity: f64,
}

impl DriftParameters {
    /// Create drift parameters, clamping values to valid ranges.
    #[must_use]
    pub fn new(scale: f64, arc_fraction: f64, eccentricity: f64) -> Self {
        let clamped_scale = scale.max(0.0);
        let clamped_arc = arc_fraction.clamp(0.0, MAX_ARC_FRACTION);
        let clamped_ecc = eccentricity.clamp(0.0, 0.95);

        if (arc_fraction - clamped_arc).abs() > f64::EPSILON {
            warn!(
                original = arc_fraction,
                clamped = clamped_arc,
                "drift_arc_fraction out of range [0, MAX_ARC_FRACTION]; clamping"
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

    /// Convert `arc_fraction` to radians (0–2π range).
    #[must_use]
    #[inline]
    pub fn sweep_radians(&self) -> f64 {
        self.arc_fraction * crate::render::constants::TWO_PI
    }
}

/// No drift - positions remain unchanged
pub struct NoDrift;

impl DriftTransform for NoDrift {
    fn apply(&mut self, _positions: &mut [Vec<Vector3<f64>>], _dt: f64) -> AppliedDrift {
        AppliedDrift::None
    }
}

/// Brownian drift - random walk motion with pre-generated random values
pub struct BrownianDrift {
    displacements: Vec<Vector3<f64>>,
}

impl BrownianDrift {
    /// Pre-generate Brownian displacement vectors for the given number of steps.
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
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], _dt: f64) -> AppliedDrift {
        if positions.is_empty() || positions[0].is_empty() {
            return AppliedDrift::None;
        }

        let steps = positions[0].len().min(self.displacements.len());
        if self.displacements[..steps].iter().all(|step| *step == Vector3::zeros()) {
            return AppliedDrift::None;
        }
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
        AppliedDrift::Brownian
    }
}

/// Linear drift - constant velocity motion
pub struct LinearDrift {
    velocity: Vector3<f64>,
}

impl LinearDrift {
    /// Create a linear drift with a random velocity direction and the given speed.
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
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], dt: f64) -> AppliedDrift {
        if positions.is_empty() || positions[0].is_empty() {
            return AppliedDrift::None;
        }

        if self.velocity == Vector3::zeros() {
            return AppliedDrift::None;
        }
        let steps = positions[0].len();

        for step in 0..steps {
            let offset = self.velocity * (step as f64) * dt;

            // Apply the same offset to all bodies at this timestep
            for body_positions in positions.iter_mut() {
                body_positions[step] += offset;
            }
        }
        AppliedDrift::Linear { velocity: self.velocity }
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
    /// Create an elliptical drift with random orbital orientation from the given parameters.
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
    fn apply(&mut self, positions: &mut [Vec<Vector3<f64>>], dt: f64) -> AppliedDrift {
        if positions.is_empty() || positions[0].len() < 2 {
            return AppliedDrift::None;
        }

        if self.sweep_radians.abs() <= f64::EPSILON || self.params.scale <= 0.0 {
            return AppliedDrift::None;
        }

        let (semi_major, semi_minor) = orbital_axes(positions, self.params);
        if semi_major <= f64::EPSILON || semi_minor <= f64::EPSILON {
            return AppliedDrift::None;
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
        AppliedDrift::Elliptical {
            rotation: self.rotation,
            initial_mean_anomaly: self.initial_mean_anomaly,
            mean_motion,
            eccentricity,
            semi_major,
            semi_minor,
        }
    }
}

/// Parse drift mode from string, returning an error for unrecognised values.
pub fn parse_drift_mode(
    mode: &str,
    rng: &mut Sha3RandomByteStream,
    params: DriftParameters,
    num_steps: usize,
) -> std::result::Result<Box<dyn DriftTransform>, ConfigError> {
    match mode.to_lowercase().as_str() {
        "none" => Ok(Box::new(NoDrift)),
        "brownian" => Ok(Box::new(BrownianDrift::new(rng, params.scale, num_steps))),
        "linear" => Ok(Box::new(LinearDrift::new(rng, params.scale))),
        "elliptical" | "ellipse" => Ok(Box::new(EllipticalDrift::new(rng, params))),
        _ => Err(ConfigError::InvalidResolution {
            reason: format!(
                "Unknown drift mode '{mode}'. Valid modes: none, brownian, linear, elliptical"
            ),
        }),
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

    /// Every drift returns what it added to the positions, which the ember edition re-applies:
    /// `None` on each path that leaves them alone, whatever was configured.
    #[test]
    fn every_drift_reports_what_it_added() {
        let dt = 0.001;
        let original = test_bodies();
        let elliptical = |scale, arc, positions: &mut Vec<Vec<Vector3<f64>>>| {
            let params = DriftParameters::new(scale, arc, 0.3);
            EllipticalDrift::new(&mut make_rng(), params).apply(positions, dt)
        };

        // Paths that add nothing.
        let mut positions = original.clone();
        assert_eq!(NoDrift.apply(&mut positions, dt), AppliedDrift::None);
        assert_eq!(elliptical(1.0, 0.0, &mut positions), AppliedDrift::None, "no arc");
        assert_eq!(elliptical(0.0, 0.5, &mut positions), AppliedDrift::None, "no scale");
        assert_eq!(positions, original);
        let mut single: Vec<Vec<Vector3<f64>>> =
            original.iter().map(|body| body[..1].to_vec()).collect();
        assert_eq!(elliptical(1.0, 0.5, &mut single), AppliedDrift::None, "a single step");
        let mut empty: Vec<Vec<Vector3<f64>>> = Vec::new();
        assert_eq!(elliptical(1.0, 0.5, &mut empty), AppliedDrift::None);
        assert_eq!(
            LinearDrift::new(&mut make_rng(), 1.0).apply(&mut empty, dt),
            AppliedDrift::None
        );

        // A linear drift adds `(velocity·k)·dt` to every body at step `k`.
        let mut positions = original.clone();
        let AppliedDrift::Linear { velocity } =
            LinearDrift::new(&mut make_rng(), 2.0).apply(&mut positions, dt)
        else {
            panic!("a linear drift reports its velocity");
        };
        for (body, moved) in original.iter().zip(&positions) {
            for (step, (before, after)) in body.iter().zip(moved).enumerate() {
                assert_eq!(*after, before + velocity * (step as f64) * dt, "step {step}");
            }
        }

        // An elliptical drift reports the ellipse it followed; the offset is the same for the
        // three bodies and is not zero at step 0.
        let mut positions = original.clone();
        let AppliedDrift::Elliptical { mean_motion, eccentricity, semi_major, semi_minor, .. } =
            elliptical(1.0, 0.5, &mut positions)
        else {
            panic!("an elliptical drift reports its ellipse");
        };
        assert_eq!(eccentricity, 0.3);
        assert!(semi_major > semi_minor && semi_minor > 0.0, "{semi_major} {semi_minor}");
        // Half a turn over the two intervals of the three recorded steps.
        assert!((mean_motion * (2.0 * dt) - PI).abs() < 1e-9, "{mean_motion}");
        let offsets: Vec<Vector3<f64>> =
            (0..3).map(|body| positions[body][0] - original[body][0]).collect();
        assert!(offsets[0].norm() > 1e-3, "{:?}", offsets[0]);
        assert!(
            (offsets[0] - offsets[1]).norm() < 1e-12 && (offsets[0] - offsets[2]).norm() < 1e-12
        );

        let mut positions = original.clone();
        let brownian = BrownianDrift::new(&mut make_rng(), 1.0, 3).apply(&mut positions, dt);
        assert_eq!(brownian, AppliedDrift::Brownian);

        // Without a scale the linear and Brownian drifts add nothing either.
        let mut positions = original.clone();
        let still = LinearDrift::new(&mut make_rng(), 0.0).apply(&mut positions, dt);
        assert_eq!(still, AppliedDrift::None);
        let still = BrownianDrift::new(&mut make_rng(), 0.0, 3).apply(&mut positions, dt);
        assert_eq!(still, AppliedDrift::None);
        assert_eq!(positions, original);
    }

    #[test]
    fn test_parse_drift_mode_none() {
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.5, 0.1);
        let mut drift = parse_drift_mode("none", &mut rng, params, 100).expect("valid mode");
        let mut positions = test_bodies();
        let original = positions.clone();
        drift.apply(&mut positions, 0.001);
        assert_eq!(positions, original, "NoDrift should not modify positions");
    }

    #[test]
    fn test_parse_drift_mode_brownian() {
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.5, 0.1);
        let mut drift = parse_drift_mode("brownian", &mut rng, params, 3).expect("valid mode");
        let mut positions = test_bodies();
        let original = positions.clone();
        drift.apply(&mut positions, 0.001);
        assert_ne!(positions, original, "BrownianDrift should modify positions");
    }

    #[test]
    fn test_parse_drift_mode_linear() {
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.5, 0.1);
        let mut drift = parse_drift_mode("linear", &mut rng, params, 3).expect("valid mode");
        let mut positions = test_bodies();
        drift.apply(&mut positions, 0.001);
        let initial_step1 = Vector3::new(1.0, 0.0, 0.0);
        assert_ne!(positions[0][1], initial_step1, "LinearDrift should offset later steps");
    }

    #[test]
    fn test_parse_drift_mode_elliptical() {
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.25, 0.1);
        let mut drift = parse_drift_mode("elliptical", &mut rng, params, 3).expect("valid mode");
        let mut positions = test_bodies();
        drift.apply(&mut positions, 0.001);
        let initial = Vector3::new(1.0, 0.0, 0.0);
        assert_ne!(positions[0][1], initial);
    }

    #[test]
    fn test_parse_drift_mode_unknown_returns_error() {
        let mut rng = make_rng();
        let params = DriftParameters::new(1.0, 0.5, 0.1);
        let result = parse_drift_mode("unknown_xyz", &mut rng, params, 3);
        assert!(result.is_err(), "Unknown mode should return an error");
    }

    #[test]
    fn test_drift_parameters_clamping() {
        let p = DriftParameters::new(-5.0, 2.0, 1.0);
        assert_eq!(p.scale, 0.0, "Negative scale should clamp to 0");
        assert_eq!(
            p.arc_fraction, MAX_ARC_FRACTION,
            "arc_fraction beyond the cap should clamp to MAX_ARC_FRACTION"
        );
        assert_eq!(p.eccentricity, 0.95, "eccentricity > 0.95 should clamp to 0.95");

        let multi_loop = DriftParameters::new(1.0, 1.35, 0.45);
        assert_eq!(multi_loop.arc_fraction, 1.35, "sweeps beyond one loop must be preserved");
    }

    #[test]
    fn test_sweep_radians() {
        let p = DriftParameters::new(1.0, 0.5, 0.0);
        let expected = 0.5 * crate::render::constants::TWO_PI;
        assert!((p.sweep_radians() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_no_drift_leaves_positions_unchanged() {
        let mut positions = test_bodies();
        let original = positions.clone();
        NoDrift.apply(&mut positions, 0.001);
        assert_eq!(positions, original);
    }

    #[test]
    fn test_brownian_drift_accumulates_offset() {
        let mut rng = make_rng();
        let mut drift = BrownianDrift::new(&mut rng, 1.0, 3);
        let mut positions = test_bodies();
        drift.apply(&mut positions, 0.001);
        // All bodies at the same timestep get the same offset
        let offset_0 = positions[0][2] - Vector3::new(1.0, 0.0, 0.0);
        let offset_1 = positions[1][2] - Vector3::new(0.0, 1.0, 0.0);
        assert!(
            (offset_0 - offset_1).norm() < 1e-10,
            "All bodies should share the same drift offset"
        );
    }

    #[test]
    fn test_linear_drift_grows_with_step() {
        let mut rng = make_rng();
        let mut drift = LinearDrift::new(&mut rng, 1.0);
        let steps = 10;
        let mut positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0); steps],
            vec![Vector3::new(1.0, 0.0, 0.0); steps],
            vec![Vector3::new(0.0, 1.0, 0.0); steps],
        ];
        drift.apply(&mut positions, 0.001);
        let offset_early = (positions[0][1] - Vector3::zeros()).norm();
        let offset_late = (positions[0][steps - 1] - Vector3::zeros()).norm();
        assert!(offset_late > offset_early, "Linear drift should grow over time");
    }
}
