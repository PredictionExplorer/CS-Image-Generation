//! Deterministic orbit analyses backing the public NFT trait file.
//!
//! Every function here is a pure function of the winning candidate's initial
//! bodies (and therefore of the on-chain seed): syzygy detection, closest
//! pairwise approach, braid-word extraction, long-term fate classification,
//! and a scale-invariant chaos index. The syzygy/braid/fate machinery is
//! ported from the `viz-master-plan` visualization branch so the production
//! metadata pipeline does not depend on the visualization stack.
//!
//! All analyses run on **raw physics-space positions** (a deterministic
//! replay of the winning candidate), never on the projected or drifted
//! render-space trajectory.

use crate::render::constants::{DEFAULT_DT, KINETIC_ENERGY_FACTOR};
use crate::sim::{Body, G, shift_bodies_to_com, symplectic_step};
use crate::utils::fourier_transform;
use nalgebra::Vector3;

/// The three unordered body pairs in canonical order: (0,1), (0,2), (1,2).
pub const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Collinearity quality below which a configuration counts as a syzygy.
const SYZYGY_QUALITY_THRESHOLD: f64 = 0.02;
/// Minimum step separation between reported syzygies.
const SYZYGY_MIN_GAP_STEPS: usize = 500;
/// Minimum step separation between recorded braid crossings of the same pair.
const BRAID_CROSSING_MERGE_STEPS: usize = 300;
/// Maximum number of Artin generators serialized into the braid word string.
const BRAID_WORD_MAX_TOKENS: usize = 48;
/// Ejection check cadence in steps for fate classification.
const EJECTION_CHECK_INTERVAL: usize = 10_000;
/// Consecutive positive checks required before an ejection is confirmed.
const EJECTION_HYSTERESIS_CHECKS: usize = 5;
/// Coefficient-of-variation ceiling for the chaos index transform: values at
/// or above this map to index 0 (metronomic regularity).
///
/// A broadband (chaotic) FFT magnitude spectrum has a small, near-Rayleigh
/// coefficient of variation (about 0.5), while a near-periodic orbit
/// concentrates energy in a few bins and produces values in the hundreds.
/// The logarithmic falloff up to this ceiling maps that multi-decade
/// continuum onto an intuitive 0-100 scale.
const CHAOS_CV_CEILING: f64 = 1_000.0;

/// One three-body alignment event.
///
/// Only the event count feeds the published traits today; the per-event
/// fields are kept for tests and future consumers of the analysis.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct Syzygy {
    /// Simulation step of the alignment.
    pub step: usize,
    /// Index of the body lying between the other two.
    pub middle_body: usize,
    /// Alignment sharpness in `[0, 1]` (1 = perfectly collinear).
    pub sharpness: f64,
}

/// Global closest pairwise approach over the recorded window.
#[derive(Clone, Copy, Debug)]
pub struct ClosestApproach {
    /// Smallest pairwise separation reached.
    pub distance: f64,
    /// The approaching pair (canonical [`PAIRS`] entry).
    pub pair: (usize, usize),
    /// Simulation step of the minimum.
    pub step: usize,
}

/// One strand crossing extracted for the braid word.
///
/// Only the generator and sign feed the serialized word today; the other
/// fields are kept for tests and future consumers of the analysis.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct BraidCrossing {
    /// Simulation step of the crossing.
    pub step: usize,
    /// The two bodies that swap order along the principal axis.
    pub bodies: (usize, usize),
    /// Artin generator index (1 or 2) at crossing time.
    pub generator: usize,
    /// True for a positive (sigma) crossing, false for its inverse.
    pub positive: bool,
}

/// Braid-word summary of the trajectory's crossing topology.
#[derive(Clone, Debug)]
pub struct BraidSummary {
    /// Serialized Artin word, e.g. `"s1 s2' s1"` (`'` marks the inverse).
    /// Truncated to [`BRAID_WORD_MAX_TOKENS`] generators for JSON sanity.
    pub word: String,
    /// Total number of crossings detected (before truncation).
    pub crossings: usize,
    /// True when `word` was truncated.
    pub truncated: bool,
}

/// Long-term dynamical outcome of the system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    /// No body exceeded the escape-energy threshold within the horizon.
    EternalDance,
    /// One body was confirmed escaping within the horizon.
    Ejection,
}

impl Fate {
    /// Stable identifier used in metadata JSON.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::EternalDance => "eternal_dance",
            Self::Ejection => "ejection",
        }
    }
}

/// Outcome of the extended fate classification run.
#[derive(Clone, Copy, Debug)]
pub struct FateOutcome {
    /// Classified long-term outcome.
    pub fate: Fate,
    /// Index of the escaping body, when [`Fate::Ejection`].
    pub escaper: Option<usize>,
    /// Extended-window step at which the ejection hysteresis confirmed.
    pub ejection_step: Option<usize>,
    /// Total extended steps checked beyond warm-up.
    pub horizon_steps: usize,
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

/// Detect syzygies (three-body alignments): local minima of the triangle
/// quality below a fixed threshold, with non-maximum suppression.
#[must_use]
pub fn detect_syzygies(positions: &[Vec<Vector3<f64>>]) -> Vec<Syzygy> {
    let steps = positions.first().map_or(0, Vec::len);
    let quality: Vec<f64> = (0..steps)
        .map(|step| triangle_quality(positions[0][step], positions[1][step], positions[2][step]))
        .collect();

    let mut syzygies = Vec::new();
    let mut last_syzygy: Option<usize> = None;
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
    syzygies
}

/// Global minimum pairwise separation over the recorded window.
#[must_use]
pub fn closest_approach(positions: &[Vec<Vector3<f64>>]) -> Option<ClosestApproach> {
    let steps = positions.first().map_or(0, Vec::len);
    if positions.len() < 3 || steps == 0 {
        return None;
    }
    let mut best: Option<ClosestApproach> = None;
    for &(a, b) in &PAIRS {
        for (step, (pos_a, pos_b)) in positions[a].iter().zip(positions[b].iter()).enumerate() {
            let distance = (pos_a - pos_b).norm();
            if best.as_ref().is_none_or(|cur| distance < cur.distance) {
                best = Some(ClosestApproach { distance, pair: (a, b), step });
            }
        }
    }
    best
}

/// Leading eigenvector of the 2x2 covariance of all xy positions
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

/// Extract the trajectory's braid word: bodies are projected onto the
/// principal separation axis; each order swap of an adjacent pair emits an
/// Artin generator whose sign is decided by z-depth (the over-strand is the
/// body closer to the viewer).
///
/// Unlike the visualization mode this operates on every simulation step (with
/// the same per-pair merge window), so the word is independent of any canvas
/// size.
#[must_use]
pub fn compute_braid(positions: &[Vec<Vector3<f64>>]) -> BraidSummary {
    let steps = positions.first().map_or(0, Vec::len);
    if positions.len() < 3 || steps < 2 {
        return BraidSummary { word: String::new(), crossings: 0, truncated: false };
    }

    let axis = principal_axis_xy(positions);
    let projected: Vec<Vec<f64>> = positions
        .iter()
        .map(|body| body.iter().map(|point| point.x * axis.0 + point.y * axis.1).collect())
        .collect();

    // Crossing detection with per-pair merge windows.
    let mut raw_crossings: Vec<(usize, usize, usize)> = Vec::new(); // (step, a, b)
    for &(a, b) in &PAIRS {
        let mut last: Option<usize> = None;
        for step in 1..steps {
            let before = projected[a][step - 1] - projected[b][step - 1];
            let after = projected[a][step] - projected[b][step];
            if before.signum() != after.signum() && before != 0.0 {
                if last.is_some_and(|prev| step - prev < BRAID_CROSSING_MERGE_STEPS) {
                    continue;
                }
                raw_crossings.push((step, a, b));
                last = Some(step);
            }
        }
    }
    raw_crossings.sort_unstable_by_key(|&(step, ..)| step);

    // Braid word: track rank order (left-to-right along the axis);
    // the over-strand is the body with greater z (closer to the viewer).
    let mut word: Vec<BraidCrossing> = Vec::new();
    for &(step, a, b) in &raw_crossings {
        let mut order: Vec<usize> = (0..3).collect();
        order.sort_by(|&lhs, &rhs| projected[lhs][step - 1].total_cmp(&projected[rhs][step - 1]));
        let rank_a = order.iter().position(|&body| body == a).unwrap_or(0);
        let rank_b = order.iter().position(|&body| body == b).unwrap_or(0);
        if rank_a.abs_diff(rank_b) != 1 {
            continue; // simultaneous multi-crossing artifact; skip
        }
        let left_body = if rank_a < rank_b { a } else { b };
        let over_body = if positions[a][step].z >= positions[b][step].z { a } else { b };
        word.push(BraidCrossing {
            step,
            bodies: (a, b),
            generator: rank_a.min(rank_b) + 1,
            positive: over_body == left_body,
        });
    }

    let truncated = word.len() > BRAID_WORD_MAX_TOKENS;
    let serialized = word
        .iter()
        .take(BRAID_WORD_MAX_TOKENS)
        .map(|crossing| {
            let mark = if crossing.positive { "" } else { "'" };
            format!("s{}{}", crossing.generator, mark)
        })
        .collect::<Vec<_>>()
        .join(" ");

    BraidSummary { word: serialized, crossings: word.len(), truncated }
}

/// Index of the body whose two-sided energy is positive beyond `threshold`
/// (mirrors `sim::is_definitely_escaping` but names the escaper).
#[must_use]
pub fn escaping_body(bodies: &[Body], threshold: f64) -> Option<usize> {
    let mut local = bodies.to_vec();
    shift_bodies_to_com(&mut local);
    for (index, body) in local.iter().enumerate() {
        let kinetic = KINETIC_ENERGY_FACTOR * body.mass * body.velocity.norm_squared();
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

/// Classify the system's long-term fate by extending the production replay
/// far past the rendered window (storage-free: no positions are recorded).
///
/// The run warms up for `steps` (mirroring `sim::get_positions`), then steps
/// through `steps * factor` more while checking for a confirmed ejection
/// every [`EJECTION_CHECK_INTERVAL`] steps with
/// [`EJECTION_HYSTERESIS_CHECKS`]-sample hysteresis.
#[must_use]
pub fn classify_fate(bodies: &[Body], steps: usize, factor: usize, threshold: f64) -> FateOutcome {
    let mut state = bodies.to_vec();
    shift_bodies_to_com(&mut state);
    for _ in 0..steps {
        symplectic_step(&mut state, DEFAULT_DT);
    }

    let horizon_steps = steps * factor.max(1);
    let mut consecutive = 0usize;
    for step in 0..horizon_steps {
        symplectic_step(&mut state, DEFAULT_DT);
        if step > 0 && step % EJECTION_CHECK_INTERVAL == 0 {
            if let Some(escaper) = escaping_body(&state, threshold) {
                consecutive += 1;
                if consecutive >= EJECTION_HYSTERESIS_CHECKS {
                    return FateOutcome {
                        fate: Fate::Ejection,
                        escaper: Some(escaper),
                        ejection_step: Some(
                            step - (EJECTION_HYSTERESIS_CHECKS - 1) * EJECTION_CHECK_INTERVAL,
                        ),
                        horizon_steps,
                    };
                }
            } else {
                consecutive = 0;
            }
        }
    }
    FateOutcome { fate: Fate::EternalDance, escaper: None, ejection_step: None, horizon_steps }
}

/// Scale-invariant chaos measure: the mean coefficient of variation
/// (std-dev / mean) of each body's FFT magnitude spectrum.
///
/// Near-periodic orbits concentrate spectral energy in a few bins (huge
/// coefficient of variation); broadband chaotic orbits spread it evenly
/// (small one). Position units cancel, so the value is comparable across
/// seeds regardless of orbit scale.
#[must_use]
pub fn chaos_coefficient_of_variation(masses: [f64; 3], positions: &[Vec<Vector3<f64>>]) -> f64 {
    let len = positions.first().map_or(0, Vec::len);
    if len < 2 {
        return 0.0;
    }
    let [m1, m2, m3] = masses;
    let mut cv_sum = 0.0;
    for body in 0..3 {
        let series: Vec<f64> = (0..len)
            .map(|step| {
                let p1 = positions[0][step];
                let p2 = positions[1][step];
                let p3 = positions[2][step];
                match body {
                    0 => (p1 - (m2 * p2 + m3 * p3) / (m2 + m3)).norm(),
                    1 => (p2 - (m1 * p1 + m3 * p3) / (m1 + m3)).norm(),
                    _ => (p3 - (m1 * p1 + m2 * p2) / (m1 + m2)).norm(),
                }
            })
            .collect();
        let magnitudes: Vec<f64> = fourier_transform(&series).iter().map(|c| c.norm()).collect();
        let mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        if mean <= 1e-24 {
            return f64::MAX;
        }
        let variance = magnitudes.iter().map(|m| (m - mean) * (m - mean)).sum::<f64>()
            / (magnitudes.len() as f64 - 1.0);
        cv_sum += variance.sqrt() / mean;
    }
    cv_sum / 3.0
}

/// Map a chaos coefficient of variation onto a 0-100 index
/// (0 = metronomic regularity, 100 = broadband chaos).
///
/// The transform `100 * (1 - ln(1 + cv) / ln(1 + CEILING))` is frozen;
/// changing it would change published trait values.
#[must_use]
pub fn chaos_index_from_cv(cv: f64) -> u32 {
    if !cv.is_finite() || cv >= CHAOS_CV_CEILING {
        return 0;
    }
    let normalized = (1.0 + cv.max(0.0)).ln() / (1.0 + CHAOS_CV_CEILING).ln();
    ((1.0 - normalized) * 100.0).round() as u32
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

    /// Body 1 sweeps through the line joining bodies 0 and 2 exactly once.
    fn single_alignment_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / (steps - 1) as f64;
                        match body {
                            0 => Vector3::new(-1.0, 0.0, 0.0),
                            1 => Vector3::new(0.0, t - 0.5, 0.0),
                            _ => Vector3::new(1.0, 0.0, 0.0),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn one_sweep_through_alignment_yields_one_syzygy() {
        let positions = single_alignment_positions(20_000);
        let syzygies = detect_syzygies(&positions);
        assert_eq!(syzygies.len(), 1, "expected exactly one alignment event");
        let syzygy = syzygies[0];
        assert_eq!(syzygy.middle_body, 1);
        assert!((syzygy.step as f64 / 20_000.0 - 0.5).abs() < 0.01);
        assert!(syzygy.sharpness > 0.9);
    }

    #[test]
    fn closest_approach_finds_the_global_minimum() {
        let positions = single_alignment_positions(1_000);
        let approach = closest_approach(&positions).expect("non-empty trajectory");
        // Body 1 passes within 1.0 of both fixed bodies at the midpoint.
        assert!((approach.distance - 1.0).abs() < 1e-3, "distance {}", approach.distance);
        assert!(approach.pair == (0, 1) || approach.pair == (1, 2));
    }

    #[test]
    fn braid_detects_strand_swaps() {
        let steps = 10_000usize;
        // Bodies 0 and 1 oscillate past each other along x; body 2 idles far
        // right so only the (0, 1) pair crosses.
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64 * std::f64::consts::TAU * 2.0;
                        match body {
                            0 => Vector3::new(t.sin(), 0.0, 1.0),
                            1 => Vector3::new(-t.sin(), 0.0, -1.0),
                            _ => Vector3::new(10.0, 0.0, 0.0),
                        }
                    })
                    .collect()
            })
            .collect();
        let braid = compute_braid(&positions);
        assert!(braid.crossings >= 3, "expected several crossings, got {}", braid.crossings);
        assert!(!braid.word.is_empty());
        assert!(
            braid.word.split_whitespace().all(|tok| tok.starts_with("s1") || tok.starts_with("s2"))
        );
    }

    #[test]
    fn braid_word_is_truncated_for_long_sequences() {
        let steps = 60_000usize;
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64 * std::f64::consts::TAU * 40.0;
                        match body {
                            0 => Vector3::new(t.sin(), 0.0, 1.0),
                            1 => Vector3::new(-t.sin(), 0.0, -1.0),
                            _ => Vector3::new(10.0, 0.0, 0.0),
                        }
                    })
                    .collect()
            })
            .collect();
        let braid = compute_braid(&positions);
        assert!(braid.crossings > BRAID_WORD_MAX_TOKENS);
        assert!(braid.truncated);
        assert_eq!(braid.word.split_whitespace().count(), BRAID_WORD_MAX_TOKENS);
    }

    fn bound_bodies() -> Vec<Body> {
        vec![
            Body::new(200.0, Vector3::new(-100.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            Body::new(200.0, Vector3::new(0.0, 150.0, 0.0), Vector3::new(0.5, 0.0, 0.0)),
        ]
    }

    #[test]
    fn fate_is_eternal_dance_under_an_unreachable_threshold() {
        let outcome = classify_fate(&bound_bodies(), 2_000, 2, 1e12);
        assert_eq!(outcome.fate, Fate::EternalDance);
        assert!(outcome.escaper.is_none());
        assert_eq!(outcome.horizon_steps, 4_000);
    }

    #[test]
    fn fate_detects_a_violent_ejection() {
        let bodies = vec![
            Body::new(100.0, Vector3::new(-300.0, 0.0, 0.0), Vector3::new(-500.0, 0.0, 0.0)),
            Body::new(100.0, Vector3::new(300.0, 0.0, 0.0), Vector3::new(500.0, 0.0, 0.0)),
            Body::new(100.0, Vector3::new(0.0, 300.0, 0.0), Vector3::new(0.0, 500.0, 0.0)),
        ];
        let outcome = classify_fate(&bodies, 1_000, 100, -0.3);
        assert_eq!(outcome.fate, Fate::Ejection);
        assert!(outcome.escaper.is_some());
        assert!(outcome.ejection_step.is_some());
    }

    #[test]
    fn escaping_body_names_the_runaway() {
        // Bodies 0 and 1 form a tightly bound close pair (mutual potential
        // dominates); body 2 is distant and unbound with modest speed so the
        // centre-of-mass correction leaves the pair bound.
        let bodies = vec![
            Body::new(100.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
            Body::new(100.0, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
            Body::new(100.0, Vector3::new(-1.0e6, 0.0, 0.0), Vector3::new(-5.0, 0.0, 0.0)),
        ];
        assert_eq!(escaping_body(&bodies, -0.3), Some(2));
    }

    #[test]
    fn chaos_cv_separates_periodic_from_noisy_series() {
        let steps = 4_096usize;
        let periodic: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                let phase = f64::from(body as u32) * 2.0;
                (0..steps)
                    .map(|step| {
                        let t = step as f64 * 0.01;
                        Vector3::new((t + phase).cos(), (t + phase).sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        // Deterministic pseudo-noise via incommensurate sinusoid products.
        let noisy: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                let phase = f64::from(body as u32) * 1.7;
                (0..steps)
                    .map(|step| {
                        let t = step as f64;
                        let x = (t * 0.7134 + phase).sin() * (t * 0.0913).cos()
                            + (t * 1.417).sin() * 0.7;
                        let y = (t * 0.9271 + phase).cos() * (t * 0.1531).sin()
                            + (t * 1.113).cos() * 0.7;
                        Vector3::new(x, y, 0.0)
                    })
                    .collect()
            })
            .collect();
        let masses = [1.0, 1.0, 1.0];
        let cv_periodic = chaos_coefficient_of_variation(masses, &periodic);
        let cv_noisy = chaos_coefficient_of_variation(masses, &noisy);
        assert!(
            cv_periodic > cv_noisy,
            "periodic cv {cv_periodic} must exceed noisy cv {cv_noisy}"
        );
        assert!(chaos_index_from_cv(cv_noisy) > chaos_index_from_cv(cv_periodic));
    }

    #[test]
    fn chaos_index_transform_is_monotonic_and_bounded() {
        let mut previous = 101u32;
        for step in 0..600 {
            let cv = f64::from(step) * 2.5;
            let index = chaos_index_from_cv(cv);
            assert!(index <= 100);
            assert!(index <= previous, "index must fall as cv grows");
            previous = index;
        }
        assert_eq!(chaos_index_from_cv(0.0), 100);
        assert_eq!(chaos_index_from_cv(1_000.0), 0);
        assert_eq!(chaos_index_from_cv(f64::INFINITY), 0);
        // A broadband (Rayleigh-like) spectrum sits near the chaotic end and
        // a realistic regular orbit (cv in the low hundreds) keeps headroom.
        assert!(chaos_index_from_cv(0.5) >= 90);
        let regular = chaos_index_from_cv(127.0);
        assert!((20..=40).contains(&regular), "regular orbit index {regular}");
    }
}
