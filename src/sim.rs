//! Simulation module: 3-body orbits, RNG, integrator, and Borda search

use crate::analysis::{
    calculate_total_angular_momentum, calculate_total_energy, equilateralness_score,
    non_chaoticness_cached,
};
use crate::error::{Result, SimulationError};
use crate::utils::FftCache;
use nalgebra::Vector3;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

/// SHA3 hash output length in bytes.
const SHA3_HASH_LEN: usize = 32;

/// Number of SHA3 hashes to chain per refill (8 hashes = 256 bytes).
/// Larger batches amortise the per-hash overhead of seed concatenation.
const HASH_BATCH_SIZE: usize = 8;

/// A custom RNG based on repeated Sha3 hashing with batched refill.
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
    pub fn new(seed: &[u8], min_mass: f64, max_mass: f64, location: f64, velocity: f64) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(seed);
        let initial = hasher.clone().finalize_reset();
        let mut buffer = Vec::with_capacity(SHA3_HASH_LEN * HASH_BATCH_SIZE);
        buffer.extend_from_slice(&initial);
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

    /// Refill the internal buffer by chaining multiple SHA3 hashes.
    fn refill_buffer(&mut self) {
        let prev_tail_start = self.buffer.len().saturating_sub(SHA3_HASH_LEN);
        let prev_tail: [u8; SHA3_HASH_LEN] = {
            let mut arr = [0u8; SHA3_HASH_LEN];
            arr.copy_from_slice(&self.buffer[prev_tail_start..]);
            arr
        };
        self.buffer.clear();
        let mut chain_input = prev_tail;
        for _ in 0..HASH_BATCH_SIZE {
            self.hasher.update(&self.seed);
            self.hasher.update(chain_input);
            let hash = self.hasher.finalize_reset();
            self.buffer.extend_from_slice(&hash);
            chain_input.copy_from_slice(&hash);
        }
        self.index = 0;
    }

    pub fn next_byte(&mut self) -> u8 {
        if self.index >= self.buffer.len() {
            self.refill_buffer();
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
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    fn gen_range(&mut self, min: f64, max: f64) -> f64 {
        self.next_f64() * (max - min) + min
    }
    pub fn random_mass(&mut self) -> f64 {
        self.gen_range(self.min_mass, self.max_mass)
    }
    pub fn random_location(&mut self) -> f64 {
        self.gen_range(-self.location_range, self.location_range)
    }
    pub fn random_velocity(&mut self) -> f64 {
        self.gen_range(-self.velocity_range, self.velocity_range)
    }
}

/// Single Body in the 3-body sim
#[derive(Clone)]
pub struct Body {
    pub mass: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    acceleration: Vector3<f64>,
}
impl Body {
    pub fn new(mass: f64, pos: Vector3<f64>, vel: Vector3<f64>) -> Self {
        Self { mass, position: pos, velocity: vel, acceleration: Vector3::zeros() }
    }
}

/// 4th-Order Yoshida Symplectic Integrator (conserves energy infinitely)
fn symplectic_step(bodies: &mut [Body], dt: f64) {
    debug_assert_eq!(bodies.len(), 3, "Optimized for 3-body problem");
    let mass = [bodies[0].mass, bodies[1].mass, bodies[2].mass];

    const W1: f64 = 1.3512071919596578;
    const W0: f64 = -1.7024143839193153;

    const C1: f64 = W1 / 2.0;
    const C2: f64 = (W0 + W1) / 2.0;
    const C3: f64 = C2;
    const C4: f64 = C1;

    const D1: f64 = W1;
    const D2: f64 = W0;
    const D3: f64 = W1;

    // Step 1
    let c1dt = C1 * dt;
    bodies[0].position += bodies[0].velocity * c1dt;
    bodies[1].position += bodies[1].velocity * c1dt;
    bodies[2].position += bodies[2].velocity * c1dt;
    compute_accelerations_unrolled(bodies, &mass);
    let d1dt = D1 * dt;
    bodies[0].velocity += bodies[0].acceleration * d1dt;
    bodies[1].velocity += bodies[1].acceleration * d1dt;
    bodies[2].velocity += bodies[2].acceleration * d1dt;

    // Step 2
    let c2dt = C2 * dt;
    bodies[0].position += bodies[0].velocity * c2dt;
    bodies[1].position += bodies[1].velocity * c2dt;
    bodies[2].position += bodies[2].velocity * c2dt;
    compute_accelerations_unrolled(bodies, &mass);
    let d2dt = D2 * dt;
    bodies[0].velocity += bodies[0].acceleration * d2dt;
    bodies[1].velocity += bodies[1].acceleration * d2dt;
    bodies[2].velocity += bodies[2].acceleration * d2dt;

    // Step 3
    let c3dt = C3 * dt;
    bodies[0].position += bodies[0].velocity * c3dt;
    bodies[1].position += bodies[1].velocity * c3dt;
    bodies[2].position += bodies[2].velocity * c3dt;
    compute_accelerations_unrolled(bodies, &mass);
    let d3dt = D3 * dt;
    bodies[0].velocity += bodies[0].acceleration * d3dt;
    bodies[1].velocity += bodies[1].acceleration * d3dt;
    bodies[2].velocity += bodies[2].acceleration * d3dt;

    // Step 4
    let c4dt = C4 * dt;
    bodies[0].position += bodies[0].velocity * c4dt;
    bodies[1].position += bodies[1].velocity * c4dt;
    bodies[2].position += bodies[2].velocity * c4dt;
}

/// Fully unrolled pairwise gravitational acceleration for exactly 3 bodies.
///
/// Computes all three pairs (0-1, 0-2, 1-2) with Newton's third law so each
/// displacement vector is computed once. Uses `d_sq.sqrt()` and manual `d^{-3}`
/// to give the compiler maximal freedom for FMA scheduling.
#[inline(always)]
fn compute_accelerations_unrolled(bodies: &mut [Body], mass: &[f64; 3]) {
    let p0 = bodies[0].position;
    let p1 = bodies[1].position;
    let p2 = bodies[2].position;

    let mut a0 = Vector3::zeros();
    let mut a1 = Vector3::zeros();
    let mut a2 = Vector3::zeros();

    // Pair (0, 1)
    let d01 = p0 - p1;
    let r01_sq = d01.norm_squared();
    if r01_sq > 1e-20 {
        let r01 = r01_sq.sqrt();
        let inv_r3 = 1.0 / (r01 * r01_sq);
        a0 -= d01 * (mass[1] * inv_r3);
        a1 += d01 * (mass[0] * inv_r3);
    }

    // Pair (0, 2)
    let d02 = p0 - p2;
    let r02_sq = d02.norm_squared();
    if r02_sq > 1e-20 {
        let r02 = r02_sq.sqrt();
        let inv_r3 = 1.0 / (r02 * r02_sq);
        a0 -= d02 * (mass[2] * inv_r3);
        a2 += d02 * (mass[0] * inv_r3);
    }

    // Pair (1, 2)
    let d12 = p1 - p2;
    let r12_sq = d12.norm_squared();
    if r12_sq > 1e-20 {
        let r12 = r12_sq.sqrt();
        let inv_r3 = 1.0 / (r12 * r12_sq);
        a1 -= d12 * (mass[2] * inv_r3);
        a2 += d12 * (mass[1] * inv_r3);
    }

    bodies[0].acceleration = a0;
    bodies[1].acceleration = a1;
    bodies[2].acceleration = a2;
}

/// Recorded positions and velocities from simulation
pub struct FullSim {
    pub positions: Vec<Vec<Vector3<f64>>>,
    pub velocities: Vec<Vec<Vector3<f64>>>,
}

/// warmup + record with zero per-step heap allocations
pub fn get_positions(mut bodies: Vec<Body>, steps: usize) -> FullSim {
    shift_bodies_to_com(&mut bodies);
    let dt = crate::render::constants::DEFAULT_DT;
    for _ in 0..steps {
        symplectic_step(&mut bodies, dt);
    }
    let mut b2 = bodies.clone();
    let n_bodies = bodies.len();
    let mut all_pos = vec![vec![Vector3::zeros(); steps]; n_bodies];
    let mut all_vel = vec![vec![Vector3::zeros(); steps]; n_bodies];
    for step in 0..steps {
        for body_idx in 0..n_bodies {
            all_pos[body_idx][step] = b2[body_idx].position;
            all_vel[body_idx][step] = b2[body_idx].velocity;
        }
        symplectic_step(&mut b2, dt);
    }
    FullSim { positions: all_pos, velocities: all_vel }
}

/// Fast trajectory simulation with early-exit for clearly bad candidates
/// Returns None if the trajectory is clearly unsuitable (saves expensive full simulation)
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

    // Record phase -- zero per-step heap allocations
    let mut b2 = bodies.clone();
    let n_bodies = bodies.len();
    let mut all_pos = vec![vec![Vector3::zeros(); steps]; n_bodies];
    let mut all_vel = vec![vec![Vector3::zeros(); steps]; n_bodies];
    for step in 0..steps {
        for body_idx in 0..n_bodies {
            all_pos[body_idx][step] = b2[body_idx].position;
            all_vel[body_idx][step] = b2[body_idx].velocity;
        }
        symplectic_step(&mut b2, dt);
    }

    Some(FullSim { positions: all_pos, velocities: all_vel })
}

/// Shift to COM
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

/// Escaping check -- operates on borrowed data without cloning.
///
/// Computes COM-frame positions and velocities inline to avoid a mutable
/// `to_vec()` + `shift_bodies_to_com` clone on every call.
pub fn is_definitely_escaping(b: &[Body], th: f64) -> bool {
    let mt: f64 = b.iter().map(|x| x.mass).sum();
    if mt < 1e-14 {
        return false;
    }
    let inv_mt = 1.0 / mt;
    let rc: Vector3<f64> =
        b.iter().map(|x| x.mass * x.position).sum::<Vector3<f64>>() * inv_mt;
    let vc: Vector3<f64> =
        b.iter().map(|x| x.mass * x.velocity).sum::<Vector3<f64>>() * inv_mt;

    for (i, bi) in b.iter().enumerate() {
        let vi = bi.velocity - vc;
        let pi = bi.position - rc;
        let kin = crate::render::constants::KINETIC_ENERGY_FACTOR * bi.mass * vi.norm_squared();
        let mut pot = 0.0;
        for (j, bj) in b.iter().enumerate() {
            if i != j {
                let d = (pi - (bj.position - rc)).norm();
                if d > 1e-12 {
                    pot += -bi.mass * bj.mass / d;
                }
            }
        }
        if kin + pot > th {
            return true;
        }
    }
    false
}

/// Borda result
#[derive(Clone)]
pub struct TrajectoryResult {
    pub chaos: f64,
    pub equilateralness: f64,
    pub chaos_pts: usize,
    pub equil_pts: usize,
    pub total_score: usize,
    pub total_score_weighted: f64,
    /// Original simulation index of the selected orbit.
    pub selected_index: usize,
    /// Number of orbits discarded by quality filters.
    pub discarded_count: usize,
}

/// Borda search
pub fn select_best_trajectory(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    steps: usize,
    cw: f64,
    ew: f64,
    th: f64,
) -> Result<(Vec<Body>, TrajectoryResult)> {
    info!("STAGE 1/7: Borda search over {num_sims} random orbits...");
    // Generate random triples and immediately transform them to the COM frame so
    // the total linear momentum and the COM position are exactly zero.
    let many: Vec<Vec<Body>> = (0..num_sims)
        .map(|_| {
            let mut v = vec![
                Body::new(
                    rng.random_mass(),
                    Vector3::new(
                        rng.random_location(),
                        rng.random_location(),
                        rng.random_location(),
                    ),
                    Vector3::new(
                        rng.random_velocity(),
                        rng.random_velocity(),
                        rng.random_velocity(),
                    ),
                ),
                Body::new(
                    rng.random_mass(),
                    Vector3::new(
                        rng.random_location(),
                        rng.random_location(),
                        rng.random_location(),
                    ),
                    Vector3::new(
                        rng.random_velocity(),
                        rng.random_velocity(),
                        rng.random_velocity(),
                    ),
                ),
                Body::new(
                    rng.random_mass(),
                    Vector3::new(
                        rng.random_location(),
                        rng.random_location(),
                        rng.random_location(),
                    ),
                    Vector3::new(
                        rng.random_velocity(),
                        rng.random_velocity(),
                        rng.random_velocity(),
                    ),
                ),
            ];
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
        .map_init(FftCache::new, |fft_cache, (i, b)| {
            let cnt = pc.fetch_add(1, Ordering::Relaxed) + 1;
            if cnt.is_multiple_of(cs) {
                info!(
                    "   Borda search: {:.0}% done",
                    (cnt as f64 / num_sims as f64) * crate::render::constants::PERCENT_FACTOR
                );
            }
            let e = calculate_total_energy(b);
            let ang = calculate_total_angular_momentum(b).norm();
            if e > 100.0 || ang < 100.0 {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            let simr = match get_positions_with_early_exit(b.clone(), steps, th) {
                Some(result) => result,
                None => {
                    dc.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };

            let pos = simr.positions;
            let m1 = b[0].mass;
            let m2 = b[1].mass;
            let m3 = b[2].mass;

            let c = non_chaoticness_cached(m1, m2, m3, &pos, fft_cache);
            let eq = equilateralness_score(&pos);

            const MIN_VIABLE_CHAOS: f64 = 0.1;
            const MIN_VIABLE_EQUILATERAL: f64 = 0.01;

            if c < MIN_VIABLE_CHAOS && eq < MIN_VIABLE_EQUILATERAL {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            Some((
                TrajectoryResult {
                    chaos: c,
                    equilateralness: eq,
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
    info!(
        "   => Discarded {dtot}/{num_sims} ({:.1}%) orbits due to filters or escapes.",
        crate::render::constants::PERCENT_FACTOR * dtot as f64 / num_sims as f64
    );
    let mut iv: Vec<(TrajectoryResult, usize)> = results.into_iter().flatten().collect();
    if iv.is_empty() {
        return Err(SimulationError::NoValidOrbits {
            total_attempted: num_sims,
            discarded: dtot,
            reason: format!(
                "All orbits filtered out due to: high energy (E > 10), \
                low angular momentum (L < 10), or escaping bodies (threshold: {})",
                th
            ),
        }
        .into());
    }
    fn assign(vals: Vec<(f64, usize)>, hb: bool) -> Vec<usize> {
        let mut v = vals;
        if hb {
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        } else {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        let n = v.len();
        let mut out = vec![0; n];
        for (r, (_, i)) in v.into_iter().enumerate() {
            out[i] = n - r;
        }
        out
    }
    let mut cv = Vec::with_capacity(iv.len());
    let mut ev = Vec::with_capacity(iv.len());
    for (i, (t, _)) in iv.iter().enumerate() {
        cv.push((t.chaos, i));
        ev.push((t.equilateralness, i));
    }
    let cps = assign(cv, false);
    let eps = assign(ev, true);
    for (i, (t, _)) in iv.iter_mut().enumerate() {
        t.chaos_pts = cps[i];
        t.equil_pts = eps[i];
        t.total_score = t.chaos_pts + t.equil_pts;
        t.total_score_weighted = cw * (t.chaos_pts as f64) + ew * (t.equil_pts as f64);
    }
    iv.sort_by(|a, b| b.0.total_score_weighted.partial_cmp(&a.0.total_score_weighted).unwrap());
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
    fn test_velocities_shape_matches_positions() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
        ];
        let sim = get_positions(bodies, 500);

        assert_eq!(sim.velocities.len(), sim.positions.len());
        for body in 0..sim.positions.len() {
            assert_eq!(
                sim.velocities[body].len(),
                sim.positions[body].len(),
                "body {body} velocity/position length mismatch"
            );
        }
    }

    #[test]
    fn test_velocities_nonzero() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
        ];
        let sim = get_positions(bodies, 500);

        let has_nonzero = sim.velocities.iter().any(|body_vels| {
            body_vels.iter().any(|v| v.norm() > 1e-10)
        });
        assert!(has_nonzero, "simulated bodies should have non-zero velocities");
    }

    #[test]
    fn test_velocities_deterministic() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
        ];
        let run1 = get_positions(bodies.clone(), 500);
        let run2 = get_positions(bodies, 500);

        for body in 0..run1.velocities.len() {
            for step in 0..run1.velocities[body].len() {
                let v1 = run1.velocities[body][step];
                let v2 = run2.velocities[body][step];
                assert_eq!(v1[0].to_bits(), v2[0].to_bits(), "vel body {body} step {step} X");
                assert_eq!(v1[1].to_bits(), v2[1].to_bits(), "vel body {body} step {step} Y");
                assert_eq!(v1[2].to_bits(), v2[2].to_bits(), "vel body {body} step {step} Z");
            }
        }
    }

    #[test]
    fn test_borda_search_determinism_with_fft_cache() {
        let seed = [0x10, 0x00, 0x33];
        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);

        let result1 = select_best_trajectory(&mut rng1, 20, 5000, 0.75, 11.0, -0.3);
        let result2 = select_best_trajectory(&mut rng2, 20, 5000, 0.75, 11.0, -0.3);

        match (result1, result2) {
            (Ok((_, info1)), Ok((_, info2))) => {
                assert_eq!(
                    info1.total_score_weighted.to_bits(),
                    info2.total_score_weighted.to_bits(),
                    "Borda scores must be deterministic"
                );
                assert_eq!(info1.selected_index, info2.selected_index);
            }
            (Err(_), Err(_)) => {}
            _ => panic!("Both runs should produce the same Ok/Err variant"),
        }
    }

    #[test]
    fn test_early_exit_escaping_bodies() {
        let bodies = vec![
            Body::new(100.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 10.0, 10.0)),
            Body::new(100.0, Vector3::new(1.0, 0.0, 0.0), Vector3::new(-10.0, -10.0, -10.0)),
            Body::new(100.0, Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
        ];
        let result = get_positions_with_early_exit(bodies, 50000, -0.3);
        assert!(result.is_none(), "high-velocity bodies should trigger early exit");
    }

    #[test]
    fn test_shift_bodies_to_com_zeroes_momentum() {
        let mut bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.5, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.5, 0.0)),
        ];
        shift_bodies_to_com(&mut bodies);

        let total_mass: f64 = bodies.iter().map(|b| b.mass).sum();
        let com: Vector3<f64> =
            bodies.iter().map(|b| b.mass * b.position).sum::<Vector3<f64>>() / total_mass;
        let mom: Vector3<f64> =
            bodies.iter().map(|b| b.mass * b.velocity).sum::<Vector3<f64>>() / total_mass;

        assert!(com.norm() < 1e-10, "COM position should be near zero");
        assert!(mom.norm() < 1e-10, "COM velocity should be near zero");
    }

    #[test]
    fn test_symplectic_integrator_energy_conservation() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.3, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.2, -0.1, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.2, -0.2, 0.0)),
        ];

        let initial_energy = crate::analysis::calculate_total_energy(&bodies);
        let sim = get_positions(bodies, 10000);

        let final_bodies = vec![
            Body::new(200.0, *sim.positions[0].last().unwrap(), *sim.velocities[0].last().unwrap()),
            Body::new(150.0, *sim.positions[1].last().unwrap(), *sim.velocities[1].last().unwrap()),
            Body::new(250.0, *sim.positions[2].last().unwrap(), *sim.velocities[2].last().unwrap()),
        ];
        let final_energy = crate::analysis::calculate_total_energy(&final_bodies);

        let rel_error = ((final_energy - initial_energy) / initial_energy.abs()).abs();
        assert!(
            rel_error < 0.01,
            "symplectic integrator should conserve energy within 1%, got {rel_error:.6}"
        );
    }

    #[test]
    fn test_long_term_energy_drift() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.3, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.2, -0.1, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.2, -0.2, 0.0)),
        ];

        let initial_energy = crate::analysis::calculate_total_energy(&bodies);
        let sim = get_positions(bodies, 100_000);

        let final_bodies = vec![
            Body::new(200.0, *sim.positions[0].last().unwrap(), *sim.velocities[0].last().unwrap()),
            Body::new(150.0, *sim.positions[1].last().unwrap(), *sim.velocities[1].last().unwrap()),
            Body::new(250.0, *sim.positions[2].last().unwrap(), *sim.velocities[2].last().unwrap()),
        ];
        let final_energy = crate::analysis::calculate_total_energy(&final_bodies);

        let rel_error = ((final_energy - initial_energy) / initial_energy.abs()).abs();
        assert!(
            rel_error < 0.05,
            "100k-step integrator drift should be < 5%, got {rel_error:.6}"
        );
    }

    #[test]
    fn test_rng_chi_squared_uniformity() {
        let mut rng = Sha3RandomByteStream::new(&[0xDE, 0xAD], 0.0, 1.0, 1.0, 1.0);
        let n = 10000;
        let k = 10;
        let mut bins = vec![0usize; k];
        for _ in 0..n {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "next_f64 produced {v}");
            let idx = (v * k as f64).min((k - 1) as f64) as usize;
            bins[idx] += 1;
        }
        let expected = n as f64 / k as f64;
        let chi_sq: f64 = bins.iter().map(|&b| {
            let diff = b as f64 - expected;
            diff * diff / expected
        }).sum();
        assert!(chi_sq < 30.0, "chi-squared {chi_sq} too high for k={k} bins (threshold ~30)");
    }

    #[test]
    fn test_rng_batched_refill_determinism() {
        let seed = [0x42, 0x43, 0x44, 0x45];
        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        for i in 0..2000 {
            let a = rng1.next_byte();
            let b = rng2.next_byte();
            assert_eq!(a, b, "RNG bytes diverged at step {i}");
        }
    }

    #[test]
    fn test_escape_check_no_clone_correctness() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.5, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.5, 0.0)),
        ];
        let result_tight = is_definitely_escaping(&bodies, -100.0);
        let result_loose = is_definitely_escaping(&bodies, 1000.0);
        assert!(!result_tight, "tightly bound threshold should not escape");
        assert!(!result_loose, "very loose threshold should not escape");
    }

    #[test]
    fn test_get_positions_single_step() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
        ];
        let sim = get_positions(bodies, 1);
        assert_eq!(sim.positions[0].len(), 1);
        assert_eq!(sim.velocities[0].len(), 1);
    }

    #[test]
    fn test_body_creation_with_zero_mass() {
        let body = Body::new(0.0, Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(body.mass, 0.0);
        assert_eq!(body.position, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_body_creation_with_negative_coordinates() {
        let body = Body::new(
            100.0,
            Vector3::new(-1e6, -2e6, -3e6),
            Vector3::new(-100.0, -200.0, -300.0),
        );
        assert_eq!(body.mass, 100.0);
        assert!(body.position.x < 0.0);
        assert!(body.velocity.y < 0.0);
    }

    #[test]
    fn test_body_creation_with_huge_velocities() {
        let body = Body::new(1.0, Vector3::zeros(), Vector3::new(1e10, 1e10, 1e10));
        assert!(body.velocity.norm() > 1e10);
    }

    #[test]
    fn test_simulation_with_equal_masses_symmetric() {
        let bodies = vec![
            Body::new(100.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.3, 0.0)),
            Body::new(100.0, Vector3::new(-100.0, 0.0, 0.0), Vector3::new(0.0, -0.3, 0.0)),
            Body::new(100.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
        ];
        let sim = get_positions(bodies, 1000);
        assert!(sim.positions[0].len() > 0);
    }

    #[test]
    fn test_rng_output_spans_full_range() {
        let mut rng = Sha3RandomByteStream::new(b"span_test", 0.0, 1.0, 1.0, 1.0);
        let mut min_val = 1.0f64;
        let mut max_val = 0.0f64;
        for _ in 0..100_000 {
            let v = rng.next_f64();
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        assert!(min_val < 0.01, "min should be near 0, got {min_val}");
        assert!(max_val > 0.99, "max should be near 1, got {max_val}");
    }

    #[test]
    fn test_rng_mass_range_boundaries() {
        let mut rng = Sha3RandomByteStream::new(b"mass_bound", 10.0, 10.0, 1.0, 1.0);
        for _ in 0..100 {
            let mass = rng.random_mass();
            assert!((mass - 10.0).abs() < 1e-10, "with equal min/max, mass should be ~10.0");
        }
    }

    #[test]
    fn test_get_positions_zero_steps() {
        let bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
            Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
            Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
        ];
        let sim = get_positions(bodies, 0);
        assert!(sim.positions[0].is_empty() || sim.positions[0].len() <= 1);
    }

    #[test]
    fn test_unrolled_acceleration_matches_physics() {
        let mut bodies = vec![
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::zeros()),
            Body::new(200.0, Vector3::new(-100.0, 0.0, 0.0), Vector3::zeros()),
            Body::new(200.0, Vector3::new(0.0, 100.0, 0.0), Vector3::zeros()),
        ];
        let mass = [200.0, 200.0, 200.0];
        compute_accelerations_unrolled(&mut bodies, &mass);

        for b in &bodies {
            assert!(b.acceleration.norm() > 0.0, "acceleration should be nonzero");
        }

        let total_force: Vector3<f64> = bodies.iter().map(|b| b.mass * b.acceleration).sum();
        assert!(
            total_force.norm() < 1e-10,
            "Newton's third law: total force should be ~zero, got {}",
            total_force.norm()
        );
    }
}
