//! Simulation module: 3-body orbits, RNG, integrator, and Borda search

use crate::analysis::{
    ChaosMetrics, boundedness_score, calculate_total_angular_momentum, calculate_total_energy,
    compute_chaos_metrics, dwell_entropy_score, equilateralness_score,
};
use crate::error::{Result, SimulationError};
use nalgebra::Vector3;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

/// Gravitational constant
pub const G: f64 = 9.8;

/// A custom RNG based on repeated Sha3 hashing
pub struct Sha3RandomByteStream {
    hasher: Sha3_256,
    seed: Vec<u8>,
    buffer: Vec<u8>,
    index: usize,
    min_mass: f64,
    max_mass: f64,
    location_range: f64,
    velocity_range: f64,
}

impl Sha3RandomByteStream {
    /// Initialise the stream from a byte seed and configure sampling ranges
    /// for mass, location, and velocity.
    #[must_use]
    pub fn new(seed: &[u8], min_mass: f64, max_mass: f64, location: f64, velocity: f64) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(seed);
        let buffer = hasher.clone().finalize_reset().to_vec();
        Self {
            hasher,
            seed: seed.to_vec(),
            buffer,
            index: 0,
            min_mass,
            max_mass,
            location_range: location,
            velocity_range: velocity,
        }
    }
    /// Return the next pseudorandom byte, re-hashing when the buffer is exhausted.
    pub fn next_byte(&mut self) -> u8 {
        if self.index >= self.buffer.len() {
            self.hasher.update(&self.seed);
            self.hasher.update(&self.buffer);
            self.buffer = self.hasher.finalize_reset().to_vec();
            self.index = 0;
        }
        let b = self.buffer[self.index];
        self.index += 1;
        b
    }
    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        for b in &mut bytes {
            *b = self.next_byte();
        }
        u64::from_le_bytes(bytes)
    }
    /// Return a uniform pseudorandom `f64` in [0, 1].
    pub fn next_f64(&mut self) -> f64 {
        // u64→f64 may lose precision for large values; acceptable for RNG normalization
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    fn gen_range(&mut self, min: f64, max: f64) -> f64 {
        self.next_f64() * (max - min) + min
    }
    /// Sample a mass uniformly within [`min_mass`, `max_mass`].
    pub fn random_mass(&mut self) -> f64 {
        self.gen_range(self.min_mass, self.max_mass)
    }
    /// Sample a coordinate uniformly within ±`location_range`.
    pub fn random_location(&mut self) -> f64 {
        self.gen_range(-self.location_range, self.location_range)
    }
    /// Sample a velocity component uniformly within ±`velocity_range`.
    pub fn random_velocity(&mut self) -> f64 {
        self.gen_range(-self.velocity_range, self.velocity_range)
    }
}

impl std::fmt::Debug for Sha3RandomByteStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sha3RandomByteStream")
            .field("index", &self.index)
            .field("min_mass", &self.min_mass)
            .field("max_mass", &self.max_mass)
            .finish_non_exhaustive()
    }
}

/// Single body in the three-body gravitational simulation.
#[derive(Clone, Debug)]
pub struct Body {
    /// Gravitational mass of this body.
    pub mass: f64,
    /// Current 3D position vector.
    pub position: Vector3<f64>,
    /// Current 3D velocity vector.
    pub velocity: Vector3<f64>,
    acceleration: Vector3<f64>,
}
impl Body {
    /// Construct a body with zero initial acceleration.
    #[must_use]
    pub fn new(mass: f64, pos: Vector3<f64>, vel: Vector3<f64>) -> Self {
        Self { mass, position: pos, velocity: vel, acceleration: Vector3::zeros() }
    }
    fn reset_acceleration(&mut self) {
        self.acceleration = Vector3::zeros();
    }
    fn update_acceleration(&mut self, om: f64, op: &Vector3<f64>) {
        let dir = self.position - *op;
        let d = dir.norm();
        if d > crate::utils::FLOAT_EPSILON {
            self.acceleration += -G * om * dir / d.powi(3);
        }
    }
}

/// 4th-Order Yoshida Symplectic Integrator (conserves energy infinitely)
fn symplectic_step(bodies: &mut [Body], dt: f64) {
    debug_assert_eq!(bodies.len(), 3, "Optimized for 3-body problem");
    let mut mass = [0.0; 3];
    for (i, b) in bodies.iter().enumerate().take(3) {
        mass[i] = b.mass;
    }

    const W1: f64 = 1.3512071919596578;
    const W0: f64 = -1.7024143839193153;

    const C1: f64 = W1 / 2.0;
    const C2: f64 = f64::midpoint(W0, W1);
    const C3: f64 = C2;
    const C4: f64 = C1;

    const D1: f64 = W1;
    const D2: f64 = W0;
    const D3: f64 = W1;

    // Step 1
    for b in bodies.iter_mut() {
        b.position += b.velocity * (C1 * dt);
    }
    compute_accelerations(bodies, &mass);
    for b in bodies.iter_mut() {
        b.velocity += b.acceleration * (D1 * dt);
    }

    // Step 2
    for b in bodies.iter_mut() {
        b.position += b.velocity * (C2 * dt);
    }
    compute_accelerations(bodies, &mass);
    for b in bodies.iter_mut() {
        b.velocity += b.acceleration * (D2 * dt);
    }

    // Step 3
    for b in bodies.iter_mut() {
        b.position += b.velocity * (C3 * dt);
    }
    compute_accelerations(bodies, &mass);
    for b in bodies.iter_mut() {
        b.velocity += b.acceleration * (D3 * dt);
    }

    // Step 4
    for b in bodies.iter_mut() {
        b.position += b.velocity * (C4 * dt);
    }
}

#[inline(always)]
fn compute_accelerations(bodies: &mut [Body], mass: &[f64; 3]) {
    let mut pos = [Vector3::zeros(); 3];
    for (i, b) in bodies.iter().enumerate().take(3) {
        pos[i] = b.position;
    }
    for (i, body) in bodies.iter_mut().enumerate().take(3) {
        body.reset_acceleration();
        for j in 0..3 {
            if i != j {
                body.update_acceleration(mass[j], &pos[j]);
            }
        }
    }
}

/// Complete simulation output: recorded position trajectories for each body.
pub struct FullSim {
    /// `positions[body_index][step_index]` — 3D position at each timestep.
    pub positions: Vec<Vec<Vector3<f64>>>,
}

impl std::fmt::Debug for FullSim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullSim")
            .field("num_bodies", &self.positions.len())
            .field("num_steps", &self.positions.first().map_or(0, std::vec::Vec::len))
            .finish()
    }
}

/// Run a full simulation: warm up for `steps`, then record `steps` more positions.
///
/// Bodies are shifted to the centre-of-mass frame before integration.
#[must_use]
pub fn get_positions(mut bodies: Vec<Body>, steps: usize) -> FullSim {
    // Ensure the initial state is expressed in the centre-of-mass (COM) frame
    shift_bodies_to_com(&mut bodies);
    let dt = crate::render::constants::DEFAULT_DT;
    for _ in 0..steps {
        symplectic_step(&mut bodies, dt);
    }
    let mut b2 = bodies.clone();
    let mut all = vec![vec![Vector3::zeros(); steps]; bodies.len()];
    for (step, snapshot) in std::iter::repeat_with(|| {
        let positions: Vec<_> = b2.iter().map(|body| body.position).collect();
        symplectic_step(&mut b2, dt);
        positions
    })
    .take(steps)
    .enumerate()
    {
        for (body_positions, position) in all.iter_mut().zip(snapshot) {
            body_positions[step] = position;
        }
    }
    FullSim { positions: all }
}

/// Fast trajectory simulation with early-exit for clearly bad candidates
/// Returns None if the trajectory is clearly unsuitable (saves expensive full simulation)
#[must_use]
pub fn get_positions_with_early_exit(
    mut bodies: Vec<Body>,
    steps: usize,
    escape_threshold: f64,
) -> Option<FullSim> {
    // Ensure the initial state is expressed in the centre-of-mass (COM) frame
    shift_bodies_to_com(&mut bodies);
    let dt = crate::render::constants::DEFAULT_DT;

    // Warmup phase with periodic escape checks
    const CHECK_INTERVAL: usize = 10000; // Check every 10k steps during warmup
    for step in 0..steps {
        symplectic_step(&mut bodies, dt);

        // Early-exit check: detect escaping bodies during warmup
        if step % CHECK_INTERVAL == 0
            && step > 0
            && is_definitely_escaping(&bodies, escape_threshold)
        {
            return None; // Body escaping, skip this candidate
        }
    }

    // Final escape check after warmup
    if is_definitely_escaping(&bodies, escape_threshold) {
        return None;
    }

    // Record phase - body configuration is good, record the full trajectory
    let mut b2 = bodies.clone();
    let mut all = vec![vec![Vector3::zeros(); steps]; bodies.len()];
    for (step, snapshot) in std::iter::repeat_with(|| {
        let positions: Vec<_> = b2.iter().map(|body| body.position).collect();
        symplectic_step(&mut b2, dt);
        positions
    })
    .take(steps)
    .enumerate()
    {
        for (body_positions, position) in all.iter_mut().zip(snapshot) {
            body_positions[step] = position;
        }
    }

    Some(FullSim { positions: all })
}

/// Translate positions and velocities so the centre of mass is at the origin with zero momentum.
pub fn shift_bodies_to_com(b: &mut [Body]) {
    let mt: f64 = b.iter().map(|x| x.mass).sum();
    if mt < 1e-14 {
        return;
    }
    let mut rc = Vector3::zeros();
    for x in b.iter() {
        rc += x.mass * x.position;
    }
    rc /= mt;
    let mut vc = Vector3::zeros();
    for x in b.iter() {
        vc += x.mass * x.velocity;
    }
    vc /= mt;
    for x in b.iter_mut() {
        x.position -= rc;
        x.velocity -= vc;
    }
}

/// Return `true` if any body's kinetic energy exceeds its binding potential by more than `th`.
#[must_use]
pub fn is_definitely_escaping(b: &[Body], th: f64) -> bool {
    let mut loc = b.to_vec();
    shift_bodies_to_com(&mut loc);
    for (i, bi) in loc.iter().enumerate() {
        let kin =
            crate::render::constants::KINETIC_ENERGY_FACTOR * bi.mass * bi.velocity.norm_squared();
        let mut pot = 0.0;
        for (j, bj) in loc.iter().enumerate() {
            if i != j {
                let d = (bi.position - bj.position).norm();
                if d > 1e-12 {
                    pot += -G * bi.mass * bj.mass / d;
                }
            }
        }
        if kin + pot > th {
            return true;
        }
    }
    false
}

/// Outcome of a Borda-count trajectory search over many random orbits.
#[derive(Clone)]
pub struct TrajectoryResult {
    /// Legacy FFT-magnitude std-dev (smaller => more chaotic).
    ///
    /// Retained for logging and determinism tests; no longer drives the
    /// Borda chaos ranking — see [`chaos_combined`] for that.
    ///
    /// [`chaos_combined`]: Self::chaos_combined
    pub chaos: f64,
    /// Shannon entropy of the normalised power spectrum, averaged across
    /// the three body-to-COM signals (higher => more chaotic, range `[0, 1]`).
    pub spectral_entropy: f64,
    /// Bandt–Pompe permutation entropy, averaged across the three
    /// body-to-COM signals (higher => more chaotic, range `[0, 1]`).
    pub permutation_entropy: f64,
    /// Combined chaos score used for Borda ranking —
    /// `(spectral_entropy + permutation_entropy) / 2`.
    pub chaos_combined: f64,
    /// Equilateralness score (higher = more triangular).
    pub equilateralness: f64,
    /// Spatial-extent stability across the trajectory (higher = more
    /// bounded, range `[0, 1]`). Used as a soft viability gate.
    pub boundedness: f64,
    /// Shannon entropy of the 2D dwell distribution over a 32×32 grid
    /// (higher => more spread across the canvas, range `[0, 1]`).
    /// Used as the primary visual-quality gate.
    pub dwell_entropy: f64,
    /// Borda points awarded for the combined chaos rank.
    pub chaos_pts: usize,
    /// Borda points awarded for equilateralness rank.
    pub equil_pts: usize,
    /// Sum of `chaos_pts` and `equil_pts`.
    pub total_score: usize,
    /// Weighted combination of Borda points used for final ranking.
    pub total_score_weighted: f64,
    /// Original simulation index of the selected orbit.
    pub selected_index: usize,
    /// Number of orbits discarded by quality filters.
    pub discarded_count: usize,
}

/// Minimum boundedness score an orbit must reach to be considered for ranking.
///
/// The score is `min(window_extents) / max(window_extents)` over five time
/// windows. A value of `0.15` means no window's extent may exceed `~6.7×`
/// the tightest window — loose enough to admit genuinely chaotic dances
/// (which naturally vary in scale across windows), tight enough to reject
/// gross slow escapes where the orbit grows unbounded over time.
pub const MIN_BOUNDEDNESS: f64 = 0.15;

/// Minimum 2D dwell-entropy an orbit must reach to be considered for ranking.
///
/// The score is the Shannon entropy of body positions binned into a 32×32
/// grid over the trajectory's bounding box, normalised to `[0, 1]`. A
/// threshold of `0.4` rejects orbits that pile their dwell into a tiny
/// fraction of the canvas (Lagrangian-triangle rotations, near-collision
/// blobs, single-circle figures) while admitting any orbit whose bodies
/// actually sweep through the frame in a visually-rich way.
pub const MIN_DWELL_ENTROPY: f64 = 0.4;

fn random_body(rng: &mut Sha3RandomByteStream) -> Body {
    Body::new(
        rng.random_mass(),
        Vector3::new(rng.random_location(), rng.random_location(), rng.random_location()),
        Vector3::new(rng.random_velocity(), rng.random_velocity(), rng.random_velocity()),
    )
}

/// Assign Borda-count points and sort trajectories by weighted score.
///
/// Flattens `results` (discarding `None`s), ranks each candidate by the
/// combined chaos score (descending — higher `chaos_combined` = more
/// chaotic = more points) and by equilateralness (descending), assigns
/// Borda points, and returns the list sorted by descending weighted score.
fn rank_trajectories(
    results: Vec<Option<(TrajectoryResult, usize)>>,
    cw: f64,
    ew: f64,
) -> Vec<(TrajectoryResult, usize)> {
    // Rank points: rank 0 (best) gets `n` points, rank `n-1` (worst) gets `1`.
    // `descending = true` means larger values rank higher (get more points).
    fn assign(vals: &mut [(f64, usize)], descending: bool) -> Vec<usize> {
        if descending {
            vals.sort_by(|a, b| b.0.total_cmp(&a.0));
        } else {
            vals.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        let n = vals.len();
        let mut out = vec![0; n];
        for (r, &(_, i)) in vals.iter().enumerate() {
            out[i] = n - r;
        }
        out
    }

    let mut iv: Vec<(TrajectoryResult, usize)> = results.into_iter().flatten().collect();
    if iv.is_empty() {
        return iv;
    }

    let mut cv = Vec::with_capacity(iv.len());
    let mut ev = Vec::with_capacity(iv.len());
    for (i, (t, _)) in iv.iter().enumerate() {
        cv.push((t.chaos_combined, i));
        ev.push((t.equilateralness, i));
    }
    // Both axes rank with "higher value = more points". Combined chaos is on
    // a [0, 1] scale where 1.0 = maximally broadband / ordinal-random; we want
    // those to win. Equilateralness is also "higher is better".
    let cps = assign(&mut cv, true);
    let eps = assign(&mut ev, true);
    for (i, (t, _)) in iv.iter_mut().enumerate() {
        t.chaos_pts = cps[i];
        t.equil_pts = eps[i];
        t.total_score = t.chaos_pts + t.equil_pts;
        // usize→f64: chaos_pts/equil_pts are bounded by the number of valid trajectories
        t.total_score_weighted = cw * (t.chaos_pts as f64) + ew * (t.equil_pts as f64);
    }
    iv.sort_by(|a, b| b.0.total_score_weighted.total_cmp(&a.0.total_score_weighted));
    iv
}

/// Run `num_sims` random orbits in parallel and pick the best via weighted Borda count.
///
/// Returns the winning initial conditions together with their [`TrajectoryResult`] scores.
pub fn select_best_trajectory(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    steps: usize,
    cw: f64,
    ew: f64,
    th: f64,
) -> Result<(Vec<Body>, TrajectoryResult)> {
    // Generate random triples and immediately transform them to the COM frame so
    // the total linear momentum and the COM position are exactly zero.
    let many: Vec<Vec<Body>> = (0..num_sims)
        .map(|_| {
            let mut v = vec![random_body(rng), random_body(rng), random_body(rng)];
            shift_bodies_to_com(&mut v);
            v
        })
        .collect();
    let pc = AtomicUsize::new(0);
    let cs = (num_sims / 10).max(1);
    let dc = AtomicUsize::new(0);
    let results: Vec<Option<(TrajectoryResult, usize)>> = many
        .par_iter()
        .enumerate()
        .map(|(i, b)| {
            let cnt = pc.fetch_add(1, Ordering::Relaxed) + 1;
            if cnt.is_multiple_of(cs) {
                // usize→f64: cnt and num_sims are bounded by num_sims (≤ ~10^6 in practice)
                info!(
                    "   Borda search: {:.0}% done",
                    (cnt as f64 / num_sims as f64) * crate::render::constants::PERCENT_FACTOR
                );
            }
            // Quick rejection: check energy and angular momentum first
            let e = calculate_total_energy(b);
            let ang = calculate_total_angular_momentum(b).norm();
            if e > 10.0 || ang < 10.0 {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Run simulation with early-exit checks for escaping bodies
            let Some(simr) = get_positions_with_early_exit(b.clone(), steps, th) else {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            };

            let pos = simr.positions;
            let m1 = b[0].mass;
            let m2 = b[1].mass;
            let m3 = b[2].mass;

            // Soft viability gate: catch gross slow escapes where the
            // trajectory's spatial extent grows unbounded over time.
            // Loose enough to admit genuinely chaotic dances.
            let bnd = boundedness_score(&pos);
            if bnd < MIN_BOUNDEDNESS {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Hard visual-quality gate: reject orbits whose dwell is
            // concentrated in a tiny fraction of the canvas — these
            // render as a single bright blob and look nothing like
            // museum art, no matter how "chaotic" the spectral metrics
            // claim.
            let dwell = dwell_entropy_score(&pos);
            if dwell < MIN_DWELL_ENTROPY {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Compute chaos metrics in a single pass (shares the three FFTs
            // across non_chaoticness and spectral_entropy).
            let ChaosMetrics { non_chaoticness: c, spectral_entropy: se, permutation_entropy: pe } =
                compute_chaos_metrics(m1, m2, m3, &pos);
            let chaos_combined = (se + pe) * 0.5;
            let eq = equilateralness_score(&pos);

            Some((
                TrajectoryResult {
                    chaos: c,
                    spectral_entropy: se,
                    permutation_entropy: pe,
                    chaos_combined,
                    equilateralness: eq,
                    boundedness: bnd,
                    dwell_entropy: dwell,
                    chaos_pts: 0,
                    equil_pts: 0,
                    total_score: 0,
                    total_score_weighted: 0.0,
                    selected_index: 0,
                    discarded_count: 0,
                },
                i,
            ))
        })
        .collect();
    let dtot = dc.load(Ordering::Relaxed);
    // usize→f64: dtot and num_sims are bounded by num_sims (≤ ~10^6 in practice)
    info!(
        "   => Discarded {dtot}/{num_sims} ({:.1}%) orbits due to filters or escapes.",
        crate::render::constants::PERCENT_FACTOR * dtot as f64 / num_sims as f64
    );
    let iv = rank_trajectories(results, cw, ew);
    if iv.is_empty() {
        return Err(SimulationError::NoValidOrbits {
            total_attempted: num_sims,
            discarded: dtot,
            reason: format!(
                "All orbits filtered out due to: high energy (E > 10), \
                low angular momentum (L < 10), escaping bodies (threshold: {th}), \
                unbounded spatial extent (boundedness < {MIN_BOUNDEDNESS}), \
                or concentrated dwell distribution (dwell entropy < {MIN_DWELL_ENTROPY})"
            ),
        }
        .into());
    }
    let bi = iv[0].1;
    let mut bt = iv[0].0.clone();
    bt.selected_index = bi;
    bt.discarded_count = dtot;
    info!("\n   => Chosen orbit idx {bi} with weighted score {:.3}", bt.total_score_weighted);
    Ok((many[bi].clone(), bt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_determinism_same_seed() {
        let seed = [0x10, 0x00, 0x33];
        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);

        for i in 0..500 {
            let a = rng1.next_f64();
            let b = rng2.next_f64();
            assert_eq!(a.to_bits(), b.to_bits(), "RNG diverged at step {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_rng_different_seeds_diverge() {
        let mut rng1 = Sha3RandomByteStream::new(&[1, 2, 3], 100.0, 300.0, 300.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&[4, 5, 6], 100.0, 300.0, 300.0, 1.0);

        let mut same_count = 0;
        for _ in 0..100 {
            if rng1.next_f64().to_bits() == rng2.next_f64().to_bits() {
                same_count += 1;
            }
        }
        assert!(same_count < 5, "different seeds should produce different sequences");
    }

    #[test]
    fn test_simulation_determinism() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
        ];
        let steps = 5000;

        let run1 = get_positions(bodies.clone(), steps);
        let run2 = get_positions(bodies, steps);

        assert_eq!(run1.positions.len(), run2.positions.len());
        for body in 0..run1.positions.len() {
            for step in 0..run1.positions[body].len() {
                let p1 = run1.positions[body][step];
                let p2 = run2.positions[body][step];
                assert_eq!(p1[0].to_bits(), p2[0].to_bits(), "body {body} step {step} X diverged");
                assert_eq!(p1[1].to_bits(), p2[1].to_bits(), "body {body} step {step} Y diverged");
                assert_eq!(p1[2].to_bits(), p2[2].to_bits(), "body {body} step {step} Z diverged");
            }
        }
    }

    #[test]
    fn test_body_new_zero_acceleration() {
        let b = Body::new(5.0, Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(b.mass, 5.0);
        assert_eq!(b.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(b.velocity, Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(b.acceleration, Vector3::zeros());
    }

    #[test]
    fn test_next_byte_produces_values() {
        let mut rng = Sha3RandomByteStream::new(&[42], 1.0, 2.0, 1.0, 1.0);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            seen.insert(rng.next_byte());
        }
        assert!(
            seen.len() > 200,
            "RNG should produce diverse byte values, only got {}",
            seen.len()
        );
    }

    #[test]
    fn test_next_f64_in_unit_range() {
        let mut rng = Sha3RandomByteStream::new(&[99], 1.0, 2.0, 1.0, 1.0);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..=1.0).contains(&v), "next_f64 returned {v}, expected [0, 1]");
        }
    }

    #[test]
    fn test_random_mass_in_range() {
        let mut rng = Sha3RandomByteStream::new(&[7], 100.0, 300.0, 1.0, 1.0);
        for _ in 0..1000 {
            let m = rng.random_mass();
            assert!((100.0..=300.0).contains(&m), "mass {m} not in [100, 300]");
        }
    }

    #[test]
    fn test_shift_bodies_to_com_centers_position() {
        let mut bodies = vec![
            Body::new(1.0, Vector3::new(10.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)),
            Body::new(1.0, Vector3::new(-10.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
            Body::new(1.0, Vector3::new(0.0, 20.0, 0.0), Vector3::new(0.0, -2.0, 0.0)),
        ];
        shift_bodies_to_com(&mut bodies);

        let mt: f64 = bodies.iter().map(|b| b.mass).sum();
        let com: Vector3<f64> =
            bodies.iter().map(|b| b.mass * b.position).sum::<Vector3<f64>>() / mt;
        assert!(com.norm() < 1e-10, "COM should be at origin, got {com:?}");

        let mom: Vector3<f64> = bodies.iter().map(|b| b.mass * b.velocity).sum();
        assert!(mom.norm() < 1e-10, "Total momentum should be zero, got {mom:?}");
    }

    #[test]
    fn test_is_definitely_escaping_bound_system() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.01, 0.0)),
            Body::new(200.0, Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -0.01, 0.0)),
            Body::new(200.0, Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.01, 0.0, 0.0)),
        ];
        assert!(!is_definitely_escaping(&bodies, 0.0), "Close, slow bodies should not escape");
    }

    #[test]
    fn test_is_definitely_escaping_fast_body() {
        let bodies = vec![
            Body::new(1.0, Vector3::new(1000.0, 0.0, 0.0), Vector3::new(1e6, 0.0, 0.0)),
            Body::new(1.0, Vector3::new(-1.0, 0.0, 0.0), Vector3::zeros()),
            Body::new(1.0, Vector3::new(0.0, 1.0, 0.0), Vector3::zeros()),
        ];
        assert!(is_definitely_escaping(&bodies, 0.0), "Very fast, far body should be escaping");
    }

    #[test]
    fn test_full_sim_debug() {
        let sim = FullSim { positions: vec![vec![Vector3::zeros(); 10]; 3] };
        let dbg = format!("{sim:?}");
        assert!(dbg.contains("num_bodies"));
        assert!(dbg.contains("num_steps"));
    }
}
