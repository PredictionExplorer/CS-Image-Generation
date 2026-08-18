//! Deterministic re-simulation on top of the production integrator
//! (master plan II.3): exact replays, extended "epilogue" runs with
//! ejection detection, perturbation grids/ensembles for basin cartography
//! and multiverse renders, and a Kabsch fit that recovers the seeded view
//! orientation so re-simulated trajectories can be dressed to match the
//! master render (the drift translation is intentionally not reproduced).

use crate::render::constants::DEFAULT_DT;
use crate::sim::{Body, G, Sha3RandomByteStream, symplectic_step};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

/// Escape-energy threshold used by the production candidate filter
/// (mirrors `main.rs`'s `DEFAULT_ESCAPE_THRESHOLD`).
pub const DEFAULT_ESCAPE_THRESHOLD: f64 = -0.3;
/// Ejection check cadence in steps (II.3).
pub const EJECTION_CHECK_INTERVAL: usize = 10_000;
/// Consecutive positive checks required to call an ejection (V23's 50k-step
/// hysteresis at the 10k check cadence).
pub const EJECTION_HYSTERESIS_CHECKS: usize = 5;

/// A detected ejection in an extended run.
#[derive(Clone, Copy, Debug)]
pub struct Ejection {
    /// Recorded-window step index at which the hysteresis confirmed.
    pub step: usize,
    /// Index of the escaping body.
    pub escaper: usize,
}

/// Result of an extended re-simulation.
pub struct Extended {
    /// Recorded positions, `steps * factor` long per body (the first
    /// `steps` reproduce the original recording exactly).
    pub positions: Vec<Vec<Vector3<f64>>>,
    /// Confirmed ejection, if any.
    pub ejection: Option<Ejection>,
    /// Body states at the end of the recording.
    pub final_bodies: Vec<Body>,
}

/// Index of the body whose two-sided energy is positive beyond `threshold`
/// (mirrors [`is_definitely_escaping`] but names the escaper).
#[must_use]
pub fn escaping_body(bodies: &[Body], threshold: f64) -> Option<usize> {
    let mut local = bodies.to_vec();
    crate::sim::shift_bodies_to_com(&mut local);
    for (index, body) in local.iter().enumerate() {
        let kinetic = crate::render::constants::KINETIC_ENERGY_FACTOR
            * body.mass
            * body.velocity.norm_squared();
        let mut potential = 0.0;
        for (other_index, other) in local.iter().enumerate() {
            if index != other_index {
                let distance = (body.position - other.position).norm();
                if distance > 1e-12 {
                    potential -= G * body.mass * other.mass / distance;
                }
            }
        }
        if kinetic + potential > threshold {
            return Some(index);
        }
    }
    None
}

/// Exact replay of the production run: warm up `steps`, record `steps`
/// (identical float path to `sim::get_positions`).
#[must_use]
pub fn rerun(bodies: &[Body], steps: usize) -> Vec<Vec<Vector3<f64>>> {
    crate::sim::get_positions(bodies.to_vec(), steps).positions
}

/// Replay with an extended recording window of `steps * factor`; the tail
/// beyond the original window is the epilogue segment. Ejections are
/// detected every [`EJECTION_CHECK_INTERVAL`] recorded steps with
/// [`EJECTION_HYSTERESIS_CHECKS`]-sample hysteresis.
#[must_use]
pub fn extended(bodies: &[Body], steps: usize, factor: usize, escape_threshold: f64) -> Extended {
    let mut state = bodies.to_vec();
    crate::sim::shift_bodies_to_com(&mut state);
    // Warm-up phase mirrors get_positions exactly.
    for _ in 0..steps {
        symplectic_step(&mut state, DEFAULT_DT);
    }
    let total = steps * factor.max(1);
    let mut positions = vec![vec![Vector3::zeros(); total]; state.len()];
    let mut ejection: Option<Ejection> = None;
    let mut consecutive = 0usize;
    for step in 0..total {
        for (body_positions, body) in positions.iter_mut().zip(state.iter()) {
            body_positions[step] = body.position;
        }
        symplectic_step(&mut state, DEFAULT_DT);
        if ejection.is_none() && step > 0 && step.is_multiple_of(EJECTION_CHECK_INTERVAL) {
            if let Some(escaper) = escaping_body(&state, escape_threshold) {
                consecutive += 1;
                if consecutive >= EJECTION_HYSTERESIS_CHECKS {
                    ejection = Some(Ejection {
                        step: step - (EJECTION_HYSTERESIS_CHECKS - 1) * EJECTION_CHECK_INTERVAL,
                        escaper,
                    });
                }
            } else {
                consecutive = 0;
            }
        }
    }
    Extended { positions, ejection, final_bodies: state }
}

/// Outcome of one capped perturbation run.
#[derive(Clone, Copy, Debug)]
pub struct CellOutcome {
    /// Escaping body index, if the run ejected within the cap.
    pub escaper: Option<u8>,
    /// Recorded step at which the ejection check first confirmed.
    pub ejection_step: Option<u32>,
}

/// Storage-free capped outcome run (warm-up included so cells live in the
/// same window as the production recording).
fn capped_outcome(
    bodies: &[Body],
    warmup: usize,
    cap: usize,
    check_interval: usize,
    escape_threshold: f64,
) -> CellOutcome {
    let mut state = bodies.to_vec();
    crate::sim::shift_bodies_to_com(&mut state);
    for step in 0..warmup + cap {
        symplectic_step(&mut state, DEFAULT_DT);
        if step > warmup
            && (step - warmup).is_multiple_of(check_interval)
            && let Some(escaper) = escaping_body(&state, escape_threshold)
        {
            return CellOutcome {
                escaper: Some(escaper as u8),
                ejection_step: Some((step - warmup) as u32),
            };
        }
    }
    CellOutcome { escaper: None, ejection_step: None }
}

/// Parameters for a perturbation lattice.
#[derive(Clone, Copy, Debug)]
pub struct GridParams {
    /// Lattice edge (cells per axis).
    pub n: usize,
    /// Half-extent of the displacement in world units.
    pub epsilon: f64,
    /// Warm-up steps before the recorded window (match the production run).
    pub warmup: usize,
    /// Capped recorded steps per cell.
    pub cap: usize,
    /// Ejection check cadence in steps.
    pub check_interval: usize,
    /// Escape threshold passed to the detector.
    pub escape_threshold: f64,
}

/// Capped outcome of arbitrary initial conditions under grid parameters
/// (the unperturbed reference for consistency checks, and the V62 terrain
/// probe).
#[must_use]
pub fn capped_fate(bodies: &[Body], params: &GridParams) -> CellOutcome {
    capped_outcome(
        bodies,
        params.warmup,
        params.cap,
        params.check_interval,
        params.escape_threshold,
    )
}

/// n x n lattice of initial conditions with body 0's position displaced in
/// the x/y plane; returns row-major outcomes (rayon-parallel).
#[must_use]
pub fn perturb_grid(bodies: &[Body], params: &GridParams) -> Vec<CellOutcome> {
    let n = params.n.max(2);
    (0..n * n)
        .into_par_iter()
        .map(|cell| {
            let row = cell / n;
            let col = cell % n;
            let fx = col as f64 / (n - 1) as f64 * 2.0 - 1.0;
            let fy = row as f64 / (n - 1) as f64 * 2.0 - 1.0;
            let mut perturbed = bodies.to_vec();
            perturbed[0].position.x += fx * params.epsilon;
            perturbed[0].position.y += fy * params.epsilon;
            capped_outcome(
                &perturbed,
                params.warmup,
                params.cap,
                params.check_interval,
                params.escape_threshold,
            )
        })
        .collect()
}

/// `count` sibling universes with isotropic position perturbations of
/// relative size `epsilon` (scaled by the mean initial pair separation).
#[must_use]
pub fn perturb_ensemble(
    bodies: &[Body],
    count: usize,
    epsilon: f64,
    rng: &mut Sha3RandomByteStream,
) -> Vec<Vec<Body>> {
    let scale = {
        let mut sum = 0.0f64;
        let mut pairs = 0.0f64;
        for i in 0..bodies.len() {
            for j in i + 1..bodies.len() {
                sum += (bodies[i].position - bodies[j].position).norm();
                pairs += 1.0;
            }
        }
        (sum / pairs.max(1.0)).max(1e-9)
    };
    (0..count)
        .map(|_| {
            let mut sibling = bodies.to_vec();
            for body in &mut sibling {
                // Uniform direction on the sphere via z/azimuth sampling.
                let z = rng.next_f64() * 2.0 - 1.0;
                let azimuth = rng.next_f64() * std::f64::consts::TAU;
                let ring = (1.0 - z * z).max(0.0).sqrt();
                let direction = Vector3::new(ring * azimuth.cos(), ring * azimuth.sin(), z);
                body.position += direction * (epsilon * scale);
            }
            sibling
        })
        .collect()
}

/// Recover the seeded view rotation `R` with `dressed ~= R * raw + d(t)`
/// via a per-step-centered Kabsch fit (centering removes the drift
/// translation and any center-of-mass wander).
#[must_use]
pub fn recover_orientation(
    raw: &[Vec<Vector3<f64>>],
    dressed: &[Vec<Vector3<f64>>],
) -> Matrix3<f64> {
    let steps = raw.first().map_or(0, Vec::len).min(dressed.first().map_or(0, Vec::len));
    if steps == 0 {
        return Matrix3::identity();
    }
    let stride = (steps / 4_000).max(1);
    let mut cross = Matrix3::<f64>::zeros();
    for step in (0..steps).step_by(stride) {
        let raw_centroid: Vector3<f64> =
            raw.iter().map(|body| body[step]).sum::<Vector3<f64>>() / raw.len() as f64;
        let dressed_centroid: Vector3<f64> =
            dressed.iter().map(|body| body[step]).sum::<Vector3<f64>>() / dressed.len() as f64;
        for (raw_body, dressed_body) in raw.iter().zip(dressed.iter()) {
            let a = raw_body[step] - raw_centroid;
            let b = dressed_body[step] - dressed_centroid;
            cross += b * a.transpose();
        }
    }
    let svd = cross.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return Matrix3::identity();
    };
    let mut d = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    u * d * v_t
}

/// Apply a rotation to every point of a trajectory set.
#[must_use]
pub fn rotate_trajectories(
    trajectories: &[Vec<Vector3<f64>>],
    rotation: &Matrix3<f64>,
) -> Vec<Vec<Vector3<f64>>> {
    trajectories.iter().map(|body| body.iter().map(|point| rotation * point).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn figure_eightish_bodies() -> Vec<Body> {
        vec![
            Body::new(1.0, Vector3::new(-0.97, 0.243, 0.0), Vector3::new(0.466, 0.432, 0.0)),
            Body::new(1.0, Vector3::new(0.97, -0.243, 0.0), Vector3::new(0.466, 0.432, 0.0)),
            Body::new(1.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(-0.932, -0.864, 0.0)),
        ]
    }

    #[test]
    fn rerun_matches_production_replay_exactly() {
        let bodies = figure_eightish_bodies();
        let steps = 512usize;
        let ours = rerun(&bodies, steps);
        let production = crate::sim::get_positions(bodies.clone(), steps).positions;
        for (a, b) in ours.iter().zip(production.iter()) {
            for (pa, pb) in a.iter().zip(b.iter()) {
                assert_eq!(pa, pb, "replay must be bit-identical");
            }
        }
    }

    #[test]
    fn extended_prefix_reproduces_the_original_window() {
        let bodies = figure_eightish_bodies();
        let steps = 256usize;
        let original = rerun(&bodies, steps);
        let ext = extended(&bodies, steps, 3, 1e9);
        assert_eq!(ext.positions[0].len(), steps * 3);
        for (extended_body, original_body) in ext.positions.iter().zip(original.iter()) {
            for (extended_point, original_point) in
                extended_body.iter().take(steps).zip(original_body.iter())
            {
                assert_eq!(
                    extended_point, original_point,
                    "extended prefix must match the original recording"
                );
            }
        }
    }

    #[test]
    fn perturb_grid_is_deterministic_and_sized() {
        let bodies = figure_eightish_bodies();
        let params = GridParams {
            n: 8,
            epsilon: 0.01,
            warmup: 64,
            cap: 256,
            check_interval: 64,
            escape_threshold: 0.5,
        };
        let first = perturb_grid(&bodies, &params);
        let second = perturb_grid(&bodies, &params);
        assert_eq!(first.len(), 64);
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.escaper, b.escaper);
            assert_eq!(a.ejection_step, b.ejection_step);
        }
    }

    #[test]
    fn ensemble_perturbations_have_the_requested_scale() {
        let bodies = figure_eightish_bodies();
        let mut rng = Sha3RandomByteStream::new(&[7, 7, 7, 7], 0.0, 1.0, 0.0, 0.0);
        let siblings = perturb_ensemble(&bodies, 4, 1e-6, &mut rng);
        assert_eq!(siblings.len(), 4);
        for sibling in &siblings {
            for (original, perturbed) in bodies.iter().zip(sibling.iter()) {
                let delta = (original.position - perturbed.position).norm();
                assert!(delta > 0.0 && delta < 1e-4, "delta {delta} out of range");
                assert_eq!(original.velocity, perturbed.velocity);
            }
        }
    }

    #[test]
    fn kabsch_recovers_a_known_rotation_under_drift() {
        let bodies = figure_eightish_bodies();
        let raw = rerun(&bodies, 200);
        let angle = 0.7f64;
        let (s, c) = angle.sin_cos();
        let rotation = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        // Dress with the rotation plus a per-step drift translation.
        let dressed: Vec<Vec<Vector3<f64>>> = raw
            .iter()
            .map(|body| {
                body.iter()
                    .enumerate()
                    .map(|(step, point)| {
                        let drift = Vector3::new(
                            (step as f64 * 0.01).sin() * 3.0,
                            step as f64 * 0.002,
                            0.0,
                        );
                        rotation * point + drift
                    })
                    .collect()
            })
            .collect();
        let recovered = recover_orientation(&raw, &dressed);
        let error = (recovered - rotation).norm();
        assert!(error < 1e-9, "Kabsch error {error}");
    }
}

#[cfg(test)]
mod rotate_tests {
    use super::*;

    #[test]
    fn rotate_trajectories_applies_the_matrix_pointwise() {
        let original: Vec<Vec<Vector3<f64>>> =
            vec![vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 2.0, 0.0)]];
        let quarter = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let rotated = rotate_trajectories(&original, &quarter);
        assert!((rotated[0][0] - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
        assert!((rotated[0][1] - Vector3::new(-2.0, 0.0, 0.0)).norm() < 1e-12);
    }
}
