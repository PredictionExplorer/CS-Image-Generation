//! Simulation module: 3-body orbits, RNG, integrator, and Borda search

use crate::analysis::{
    calculate_total_angular_momentum, calculate_total_energy, equilateralness_score,
    non_chaoticness,
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
        let d2 = dir.norm_squared();
        if d2 <= crate::utils::FLOAT_EPSILON * crate::utils::FLOAT_EPSILON {
            return;
        }
        let eps2 = crate::render::pipeline_flags::sim_softening_eps2();
        let inv = if eps2 > 0.0 { 1.0 / (d2 + eps2).powf(1.5) } else { 1.0 / d2.powf(1.5) };
        self.acceleration += -G * om * dir * inv;
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
    /// Non-chaoticness score (higher = more regular orbit).
    pub chaos: f64,
    /// Equilateralness score (higher = more triangular).
    pub equilateralness: f64,
    /// Borda points awarded for non-chaoticness rank.
    pub chaos_pts: usize,
    /// Borda points awarded for equilateralness rank.
    pub equil_pts: usize,
    /// Sum of `chaos_pts` and `equil_pts`.
    pub total_score: usize,
    /// Weighted combination of Borda points used for final ranking.
    pub total_score_weighted: f64,
    /// Auxiliary aesthetic ensemble score (close approaches, motion entropy, etc.).
    pub beauty: f64,
    /// Borda points from the beauty ensemble rank.
    pub beauty_pts: usize,
    /// Original simulation index of the selected orbit.
    pub selected_index: usize,
    /// Number of orbits discarded by quality filters.
    pub discarded_count: usize,
}

fn random_body(rng: &mut Sha3RandomByteStream) -> Body {
    Body::new(
        rng.random_mass(),
        Vector3::new(rng.random_location(), rng.random_location(), rng.random_location()),
        Vector3::new(rng.random_velocity(), rng.random_velocity(), rng.random_velocity()),
    )
}

/// Assign Borda-count points and sort trajectories by weighted score.
///
/// Flattens `results` (discarding `None`s), ranks by non-chaoticness and equilateralness,
/// assigns Borda points, and returns the list sorted by descending weighted score.
fn rank_trajectories(
    results: Vec<Option<(TrajectoryResult, usize)>>,
    cw: f64,
    ew: f64,
    bw: f64,
) -> Vec<(TrajectoryResult, usize)> {
    fn assign(vals: &mut [(f64, usize)], hb: bool) -> Vec<usize> {
        if hb {
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
    let mut bv = Vec::with_capacity(iv.len());
    for (i, (t, _)) in iv.iter().enumerate() {
        cv.push((t.chaos, i));
        ev.push((t.equilateralness, i));
        bv.push((t.beauty, i));
    }
    let cps = assign(&mut cv, false);
    let eps = assign(&mut ev, true);
    let bps = assign(&mut bv, true);
    for (i, (t, _)) in iv.iter_mut().enumerate() {
        t.chaos_pts = cps[i];
        t.equil_pts = eps[i];
        t.beauty_pts = bps[i];
        t.total_score = t.chaos_pts + t.equil_pts + t.beauty_pts;
        // usize→f64: chaos_pts/equil_pts are bounded by the number of valid trajectories
        t.total_score_weighted =
            cw * (t.chaos_pts as f64) + ew * (t.equil_pts as f64) + bw * (t.beauty_pts as f64);
    }
    iv.sort_by(|a, b| b.0.total_score_weighted.total_cmp(&a.0.total_score_weighted));
    iv
}

/// Re-rank the top-K Borda candidates using a higher-fidelity re-simulation and a richer
/// aesthetic ensemble (close approaches + motion entropy + fractal dimension + symmetries).
///
/// Returns the re-ranked Borda winner with its updated `beauty` score.  The returned
/// `selected_index` is the index into the *original* `candidates` slice, which callers use
/// to look up the original initial conditions.
fn thumbnail_rerank_top_k(
    candidates: &[(Vec<Body>, TrajectoryResult)],
    k: usize,
    rerank_steps: usize,
    escape_threshold: f64,
) -> Option<(usize, TrajectoryResult)> {
    if candidates.is_empty() || k == 0 {
        return None;
    }
    let take = k.min(candidates.len()).max(1);
    let scored: Vec<(usize, f64, TrajectoryResult)> = (0..take)
        .into_par_iter()
        .filter_map(|i| {
            let (bodies, traj) = &candidates[i];
            let simr =
                get_positions_with_early_exit(bodies.clone(), rerank_steps, escape_threshold)?;
            let pos = simr.positions;
            let m1 = bodies[0].mass;
            let m2 = bodies[1].mass;
            let m3 = bodies[2].mass;
            let beauty = crate::analysis::beauty_ensemble_score(&pos, m1, m2, m3);
            // Combine stage-1 weighted Borda score with the fresh beauty ensemble.
            // 0.55 weight on the Borda stage keeps chaos/equil influence; 0.45 on beauty lets
            // the richer re-rank differentiate near-ties.
            let combined = 0.55 * traj.total_score_weighted + 0.45 * beauty * 100.0;
            let mut updated = traj.clone();
            updated.beauty = beauty;
            Some((i, combined, updated))
        })
        .collect();

    scored.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(idx, _, traj)| (idx, traj))
}

/// Outlier dominance threshold (p98 / p50 trimmed axial extent). Above
/// this ratio a single excursion of one or two bodies dwarfs the main
/// cluster, producing the "tiny cluster plus a long tail into empty
/// space" failure mode that the framing pipeline can only partially
/// compensate for. Empirically, well-composed three-body orbits sit at
/// 3–8; anything above 20 is pathological.
pub const MAX_OUTLIER_EXTENT_RATIO: f64 = 20.0;

/// Absolute floor on the dominant axial extent of any body. Below this,
/// every body is collapsing to a near-point and the resulting image
/// carries no meaningful trajectory to light up pixels. Measured in the
/// same world units as the initial-condition location scale.
pub const MIN_DOMINANT_BODY_EXTENT: f64 = 1.0;

/// Percentile-trimmed axial extent of `positions` along axis
/// `axis` (0 = x, 1 = y, 2 = z).
///
/// Returns the span between the `low`-th and `(1 - low)`-th
/// percentiles of the coordinates sampled over all bodies and
/// time-steps. `low` is expected to be small (e.g. 0.02 → 2 %–98 %
/// trim). A zero-length input returns `0.0`.
fn trimmed_axial_extent(positions: &[Vec<Vector3<f64>>], axis: usize, low: f64) -> f64 {
    let total: usize = positions.iter().map(std::vec::Vec::len).sum();
    if total == 0 {
        return 0.0;
    }
    let mut values: Vec<f64> = Vec::with_capacity(total);
    for body in positions {
        for p in body {
            values.push(match axis {
                0 => p.x,
                1 => p.y,
                _ => p.z,
            });
        }
    }
    let n = values.len();
    let lo_idx = ((low * n as f64).floor() as usize).min(n - 1);
    let hi_idx = (((1.0 - low) * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    values.select_nth_unstable_by(lo_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let lo = values[lo_idx];
    values.select_nth_unstable_by(hi_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let hi = values[hi_idx];
    (hi - lo).max(0.0)
}

/// Maximum, across `(body, axis)` pairs, of the 98th–2nd percentile
/// trimmed axial extent. Used as the "dominant body extent" signal:
/// as long as at least one body has a non-trivial range of motion on
/// at least one axis, the orbit can produce meaningful image content.
fn dominant_body_extent(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let mut best = 0.0f64;
    for body in positions {
        if body.is_empty() {
            continue;
        }
        for axis in 0..3 {
            let single = [body.clone()];
            let ext = trimmed_axial_extent(&single, axis, 0.02);
            if ext > best {
                best = ext;
            }
        }
    }
    best
}

/// Ratio between the untrimmed and 2 %-trimmed axial extents, maxed
/// over axes. Large values indicate outliers: a small, tight main
/// cluster plus one or more rare excursions that blow up the raw
/// bounding box. When the ratio exceeds
/// [`MAX_OUTLIER_EXTENT_RATIO`] the framing camera is forced to
/// choose between keeping the outlier visible (leaving the main
/// subject tiny) and zooming past it (leaving a hot splat near the
/// frame edge).
fn outlier_extent_ratio(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let mut worst = 0.0f64;
    for axis in 0..3 {
        let trimmed = trimmed_axial_extent(positions, axis, 0.02);
        let raw = trimmed_axial_extent(positions, axis, 0.0);
        if trimmed <= 1e-9 {
            continue;
        }
        let ratio = raw / trimmed;
        if ratio > worst {
            worst = ratio;
        }
    }
    worst
}

/// Reject `positions` if the orbit's projected spatial extent is
/// degenerate or outlier-dominated.
///
/// Returns `true` when the trajectory is acceptable for rendering, and
/// `false` when Borda selection should discard it. Conservative by
/// design: only catches clearly-broken orbits (every body collapsed to
/// a point, or a single body's far excursion dwarfing the rest).
/// Preserves variety — "boring but bounded" orbits are not filtered;
/// the framing / tone-mapping pipeline already handles them.
pub fn passes_spatial_extent_filter(positions: &[Vec<Vector3<f64>>]) -> bool {
    if positions.is_empty() {
        return false;
    }
    if dominant_body_extent(positions) < MIN_DOMINANT_BODY_EXTENT {
        return false;
    }
    if outlier_extent_ratio(positions) > MAX_OUTLIER_EXTENT_RATIO {
        return false;
    }
    true
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
    bw: f64,
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

            // Spatial-extent degeneracy filter. Catches the two
            // failure modes that routinely slip past chaos/equil/energy
            // gates into final renders:
            //   1. All three bodies collapse to a near-point (dominant
            //      extent < `MIN_DOMINANT_BODY_EXTENT`). The resulting
            //      image is a single bright dot with no structure.
            //   2. One body's rare long excursion dominates the
            //      bounding box (outlier ratio >
            //      `MAX_OUTLIER_EXTENT_RATIO`). The framing camera is
            //      then forced to either zoom onto a tiny cluster with
            //      a bright splat far off-centre, or frame the whole
            //      extent and leave the real subject as a few pixels.
            // The filter is conservative: empirically-observed
            // museum-quality orbits sit well inside both bounds.
            if !passes_spatial_extent_filter(&pos) {
                dc.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            let m1 = b[0].mass;
            let m2 = b[1].mass;
            let m3 = b[2].mass;

            // Compute quality metrics
            let c = non_chaoticness(m1, m2, m3, &pos);
            let eq = equilateralness_score(&pos);
            let beauty = crate::analysis::beauty_ensemble_score(&pos, m1, m2, m3);

            // Early rejection: if both metrics are terrible, skip
            // This saves time on Borda ranking for clearly unsuitable candidates
            const MIN_VIABLE_CHAOS: f64 = 0.1; // Below this, too chaotic
            const MIN_VIABLE_EQUILATERAL: f64 = 0.01; // Below this, too linear

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
                    beauty,
                    beauty_pts: 0,
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
    let iv = rank_trajectories(results, cw, ew, bw);
    if iv.is_empty() {
        return Err(SimulationError::NoValidOrbits {
            total_attempted: num_sims,
            discarded: dtot,
            reason: format!(
                "All orbits filtered out due to: high energy (E > 10), \
                low angular momentum (L < 10), or escaping bodies (threshold: {th})"
            ),
        }
        .into());
    }
    // Stage-1 winner indices and scores are in `iv`.  Build a top-K candidate list for stage-2
    // re-rank, which re-simulates at `steps` and refreshes the beauty ensemble.  This amortises
    // the full 100k stage-1 pool down to a few candidates for a high-fidelity comparison.
    const TOP_K: usize = 8;
    let top_k: Vec<(Vec<Body>, TrajectoryResult)> =
        iv.iter().take(TOP_K).map(|(t, sim_idx)| (many[*sim_idx].clone(), t.clone())).collect();

    let rerank_steps = steps.clamp(5_000, 200_000);
    let (winner_in_top_k, bt_stage2) =
        thumbnail_rerank_top_k(&top_k, TOP_K, rerank_steps, th).unwrap_or((0, iv[0].0.clone()));

    let stage1_index = iv[winner_in_top_k].1;
    let mut bt = bt_stage2;
    bt.selected_index = stage1_index;
    bt.discarded_count = dtot;
    info!(
        "\n   => Stage-1 top-{TOP_K} -> rerank winner orbit idx {stage1_index} (top-K slot {winner_in_top_k}), weighted={:.3}, beauty={:.3}",
        bt.total_score_weighted, bt.beauty
    );

    // Emit spatial-extent telemetry for the winner. We re-simulate at
    // `rerank_steps` (the same horizon used for beauty scoring) so the
    // telemetry reflects what the downstream render will see, not the
    // coarser `steps` footprint from stage-1.
    if let Some(simr) =
        get_positions_with_early_exit(many[stage1_index].clone(), rerank_steps, th)
    {
        let pos = &simr.positions;
        let dbe = dominant_body_extent(pos);
        let oer = outlier_extent_ratio(pos);
        crate::generation_log::record_telemetry(|t| {
            t.dominant_body_extent = Some(dbe);
            t.outlier_extent_ratio = Some(oer);
        });
    }

    Ok((many[stage1_index].clone(), bt))
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

    fn make_trajectory(
        chaos: f64,
        equil: f64,
        beauty: f64,
        idx: usize,
    ) -> (TrajectoryResult, usize) {
        (
            TrajectoryResult {
                chaos,
                equilateralness: equil,
                chaos_pts: 0,
                equil_pts: 0,
                total_score: 0,
                total_score_weighted: 0.0,
                beauty,
                beauty_pts: 0,
                selected_index: 0,
                discarded_count: 0,
            },
            idx,
        )
    }

    #[test]
    fn test_rank_trajectories_weights_beauty_bins() {
        let results = vec![
            Some(make_trajectory(1.0, 0.5, 0.2, 0)), // high chaos, low equil, low beauty
            Some(make_trajectory(0.1, 0.9, 0.95, 1)), // low chaos, high equil, high beauty
            Some(make_trajectory(0.5, 0.5, 0.5, 2)),
        ];
        let ranked = rank_trajectories(results, 1.0, 1.0, 5.0);
        assert_eq!(ranked.len(), 3);
        // Index 1 has high beauty and high equil; with bw=5 it must win.
        assert_eq!(ranked[0].1, 1, "beauty-weighted winner should be index 1");
    }

    #[test]
    fn test_rank_trajectories_filters_none_entries() {
        let results: Vec<Option<(TrajectoryResult, usize)>> = vec![
            Some(make_trajectory(1.0, 1.0, 1.0, 0)),
            None,
            None,
            Some(make_trajectory(0.5, 0.5, 0.5, 3)),
        ];
        let ranked = rank_trajectories(results, 1.0, 1.0, 1.0);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn test_rank_trajectories_total_score_equals_sum_of_pts() {
        let results = vec![
            Some(make_trajectory(0.3, 0.6, 0.1, 0)),
            Some(make_trajectory(0.8, 0.2, 0.4, 1)),
            Some(make_trajectory(0.1, 0.9, 0.9, 2)),
        ];
        let ranked = rank_trajectories(results, 1.0, 1.0, 1.0);
        for (t, _) in &ranked {
            assert_eq!(t.total_score, t.chaos_pts + t.equil_pts + t.beauty_pts);
        }
    }

    #[test]
    fn test_select_best_trajectory_returns_viable_orbit() {
        let mut rng = Sha3RandomByteStream::new(&[0x01, 0x02, 0x03], 100.0, 300.0, 300.0, 1.0);
        let (bodies, info) = select_best_trajectory(&mut rng, 64, 2_000, 1.0, 3.0, 0.5, -0.3)
            .expect("selection should succeed on a populated seed");
        assert_eq!(bodies.len(), 3);
        assert!(info.chaos.is_finite());
        assert!(info.equilateralness.is_finite());
        assert!(info.beauty.is_finite());
        assert!((0.0..=1.0).contains(&info.beauty));
    }

    // ------------------------------------------------------------------
    // Spatial-extent filter (Phase 4) unit + property tests
    // ------------------------------------------------------------------

    /// Build a three-body circular motion on the XY plane with the
    /// given radius. Used as a "well-composed" reference trajectory
    /// that must always pass the spatial-extent filter.
    fn circular_trajectory(radius: f64, steps: usize) -> Vec<Vec<Vector3<f64>>> {
        use std::f64::consts::TAU;
        (0..3i32)
            .map(|body_idx| {
                let phase = TAU * f64::from(body_idx) / 3.0;
                (0..steps)
                    .map(|i| {
                        let t = TAU * (i as f64) / (steps as f64);
                        Vector3::new(
                            radius * (t + phase).cos(),
                            radius * (t + phase).sin(),
                            0.0,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn spatial_extent_filter_accepts_healthy_orbit() {
        let pos = circular_trajectory(5.0, 500);
        assert!(passes_spatial_extent_filter(&pos), "healthy circular orbit must pass");
    }

    #[test]
    fn spatial_extent_filter_rejects_collapsed_orbit() {
        // All bodies glued to origin: dominant extent = 0, below floor.
        let pos: Vec<Vec<Vector3<f64>>> =
            (0..3).map(|_| (0..500).map(|_| Vector3::zeros()).collect()).collect();
        assert!(!passes_spatial_extent_filter(&pos), "collapsed orbit must be rejected");
    }

    #[test]
    fn spatial_extent_filter_rejects_outlier_dominated_orbit() {
        // Baseline tight cluster + one extreme excursion of body 0 far
        // from the origin. Raw extent is driven by the outlier; trimmed
        // extent stays inside the cluster. Ratio should blow past the
        // 20x threshold.
        let steps = 500;
        let mut pos = circular_trajectory(1.0, steps);
        pos[0][steps - 1] = Vector3::new(5000.0, 0.0, 0.0);
        pos[0][steps - 2] = Vector3::new(5000.0, 0.0, 0.0);
        assert!(
            !passes_spatial_extent_filter(&pos),
            "outlier-dominated orbit must be rejected"
        );
    }

    #[test]
    fn spatial_extent_filter_preserves_modest_asymmetry() {
        // 5x asymmetry between bodies — well within the 20x threshold.
        // This guards the conservative-by-design promise: the filter
        // must not prune legitimately-varied orbits from the pool.
        let mut pos = circular_trajectory(2.0, 500);
        for p in &mut pos[0] {
            p.x *= 4.0;
        }
        assert!(
            passes_spatial_extent_filter(&pos),
            "moderately-asymmetric orbit must pass (filter is conservative)"
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 128,
            .. proptest::prelude::ProptestConfig::default()
        })]

        // Any non-degenerate, bounded circular orbit at moderate radius
        // with moderate anisotropy must pass the spatial-extent filter.
        // This encodes the "conservative" invariant: the filter may
        // only reject clearly broken orbits, never reasonable ones.
        #[test]
        fn proptest_conservative_filter_never_rejects_reasonable_orbits(
            radius in 1.5f64..50.0,
            anisotropy in 0.3f64..3.0,
            z_amp in 0.0f64..5.0,
            steps in 200usize..1000,
        ) {
            use std::f64::consts::TAU;
            let pos: Vec<Vec<Vector3<f64>>> = (0..3i32)
                .map(|body_idx| {
                    let phase = TAU * f64::from(body_idx) / 3.0;
                    (0..steps)
                        .map(|i| {
                            let t = TAU * (i as f64) / (steps as f64);
                            Vector3::new(
                                radius * (t + phase).cos(),
                                radius * anisotropy * (t + phase).sin(),
                                z_amp * (t * 2.0 + phase).sin(),
                            )
                        })
                        .collect()
                })
                .collect();
            proptest::prop_assert!(
                passes_spatial_extent_filter(&pos),
                "rejected reasonable orbit (radius={}, anisotropy={}, z_amp={}, steps={})",
                radius, anisotropy, z_amp, steps,
            );
        }
    }
}
