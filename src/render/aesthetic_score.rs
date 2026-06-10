//! Geometry-space aesthetic scoring for candidate orbits and view rotations.
//!
//! The Borda search ranks orbits by physics proxies (FFT regularity and
//! triangle balance) but never looks at the picture. This module closes that
//! gap with a cheap proxy render: sampled trajectory geometry is splatted onto
//! a small per-body ink grid using the seed's structure mode, then scored on
//! image-space qualities (ink coverage, spatial balance, structural contrast,
//! per-body colour separation, and a grey-mush penalty for dense regions where
//! all three bodies pile up). Scores are deterministic functions of geometry,
//! so candidate selection stays reproducible per seed.

use super::visual_profile::StructureMode;
use nalgebra::Vector3;

/// Grid edge (pixels) used when scoring shortlisted orbit candidates.
pub const CANDIDATE_GRID_SIZE: usize = 192;
/// Grid edge (pixels) used when scoring candidate view rotations (cheaper).
pub const VIEW_GRID_SIZE: usize = 96;
/// Sample budget per body when scoring shortlisted orbit candidates.
pub const CANDIDATE_SAMPLES_PER_BODY: usize = 24_000;
/// Sample budget per body when scoring candidate view rotations.
pub const VIEW_SAMPLES_PER_BODY: usize = 6_000;

/// Ideal lower bound of the inked-area fraction (sparser images start losing points).
const COVERAGE_BAND_LOW: f64 = 0.10;
/// Ideal upper bound of the inked-area fraction (denser images start losing points).
const COVERAGE_BAND_HIGH: f64 = 0.42;
/// Coverage below this scores zero (an almost empty frame).
const COVERAGE_FLOOR: f64 = 0.01;
/// Coverage above this scores zero (a fully flooded frame).
const COVERAGE_CEIL: f64 = 0.85;
/// Cells per axis for the spatial-balance occupancy histogram.
const BALANCE_CELLS: usize = 4;
/// Share of total ink below which a body counts as visually missing.
const BODY_PRESENCE_FLOOR: f64 = 0.06;
/// Fraction of bright pixels allowed to be three-body mush before penalties.
const MUSH_TOLERANCE: f64 = 0.35;
/// Time-lag fraction used to proxy chord-style modes during scoring.
const PROXY_CHORD_LAG_FRACTION: f64 = 0.012;

/// Component metrics of a proxy-rendered aesthetic evaluation.
#[derive(Clone, Copy, Debug)]
pub struct AestheticScore {
    /// Weighted total in `[0, 1]`; higher is better.
    pub total: f64,
    /// Fraction of grid pixels carrying ink.
    pub coverage: f64,
    /// Band-mapped coverage score in `[0, 1]`.
    pub coverage_score: f64,
    /// Spatial occupancy entropy in `[0, 1]` (dead-zone penalty built in).
    pub balance: f64,
    /// Structural contrast of the ink-density distribution in `[0, 1]`.
    pub contrast: f64,
    /// Per-body colour separation / presence score in `[0, 1]`.
    pub body_mix: f64,
    /// Fraction of bright pixels where all three bodies overlap heavily.
    pub mush_fraction: f64,
}

impl AestheticScore {
    fn zero() -> Self {
        Self {
            total: 0.0,
            coverage: 0.0,
            coverage_score: 0.0,
            balance: 0.0,
            contrast: 0.0,
            body_mix: 0.0,
            mush_fraction: 0.0,
        }
    }
}

/// Parameters for a proxy-render evaluation.
#[derive(Clone, Copy, Debug)]
pub struct ProxyRenderParams {
    /// Square grid edge length in pixels.
    pub grid_size: usize,
    /// Maximum trajectory samples drawn per body.
    pub samples_per_body: usize,
    /// Structure mode used to convert geometry into strokes.
    pub structure: StructureMode,
}

impl ProxyRenderParams {
    /// Parameters tuned for scoring shortlisted orbit candidates.
    #[must_use]
    pub fn for_candidates(structure: StructureMode) -> Self {
        Self {
            grid_size: CANDIDATE_GRID_SIZE,
            samples_per_body: CANDIDATE_SAMPLES_PER_BODY,
            structure,
        }
    }

    /// Cheaper parameters tuned for ranking candidate view rotations.
    #[must_use]
    pub fn for_view_selection(structure: StructureMode) -> Self {
        Self { grid_size: VIEW_GRID_SIZE, samples_per_body: VIEW_SAMPLES_PER_BODY, structure }
    }
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Map a coverage fraction onto `[0, 1]` with a flat ideal band.
fn coverage_band_score(coverage: f64) -> f64 {
    if coverage <= COVERAGE_FLOOR || coverage >= COVERAGE_CEIL {
        return 0.0;
    }
    if coverage < COVERAGE_BAND_LOW {
        return smoothstep((coverage - COVERAGE_FLOOR) / (COVERAGE_BAND_LOW - COVERAGE_FLOOR));
    }
    if coverage > COVERAGE_BAND_HIGH {
        return smoothstep((COVERAGE_CEIL - coverage) / (COVERAGE_CEIL - COVERAGE_BAND_HIGH));
    }
    1.0
}

/// Per-body ink accumulation grid used by the proxy renderer.
struct InkGrid {
    size: usize,
    /// Per-pixel ink, one channel per body.
    ink: Vec<[f64; 3]>,
}

impl InkGrid {
    fn new(size: usize) -> Self {
        Self { size, ink: vec![[0.0; 3]; size * size] }
    }

    #[inline]
    fn deposit(&mut self, x: f64, y: f64, body: usize, amount: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let size = self.size as isize;
        let xi = x.floor() as isize;
        let yi = y.floor() as isize;
        if xi < 0 || yi < 0 || xi >= size || yi >= size {
            return;
        }
        self.ink[yi as usize * self.size + xi as usize][body] += amount;
    }

    /// Splat a line segment with a fixed-step DDA walk.
    fn draw_segment(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, body: usize, amount: f64) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let length = (dx * dx + dy * dy).sqrt();
        if !length.is_finite() {
            return;
        }
        let steps = (length.ceil() as usize).clamp(1, self.size * 2);
        // usize→f64 precision loss is irrelevant at grid scale.
        let inv = 1.0 / steps as f64;
        for i in 0..=steps {
            let t = i as f64 * inv;
            self.deposit(x0 + dx * t, y0 + dy * t, body, amount);
        }
    }
}

/// Projected (x, y) points per body, normalized to grid coordinates.
struct ProjectedTrajectories {
    /// `points[body][sample] = (grid_x, grid_y)`.
    points: Vec<Vec<(f64, f64)>>,
}

/// Project sampled positions to grid space using the padded 2D bounding box.
fn project_to_grid(
    positions: &[Vec<Vector3<f64>>],
    grid_size: usize,
    samples_per_body: usize,
) -> Option<ProjectedTrajectories> {
    let steps = positions.first().map_or(0, Vec::len);
    if positions.len() < 3 || steps < 2 {
        return None;
    }

    let stride = (steps / samples_per_body.max(1)).max(1);

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for body in positions {
        let mut idx = 0;
        while idx < body.len() {
            let p = body[idx];
            if p[0].is_finite() && p[1].is_finite() {
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
                min_y = min_y.min(p[1]);
                max_y = max_y.max(p[1]);
            }
            idx += stride;
        }
    }
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return None;
    }

    let span_x = (max_x - min_x).max(1e-12);
    let span_y = (max_y - min_y).max(1e-12);
    // 4% margin mirrors the renderer's padded composition.
    let margin = 0.04;
    let scale = grid_size as f64 * (1.0 - 2.0 * margin);
    let offset = grid_size as f64 * margin;

    let points = positions
        .iter()
        .map(|body| {
            let mut sampled = Vec::with_capacity(body.len() / stride + 2);
            let mut idx = 0;
            while idx < body.len() {
                let p = body[idx];
                let gx = offset + (p[0] - min_x) / span_x * scale;
                let gy = offset + (p[1] - min_y) / span_y * scale;
                sampled.push((gx, gy));
                idx += stride;
            }
            sampled
        })
        .collect();

    Some(ProjectedTrajectories { points })
}

/// Splat the sampled geometry according to the structure mode.
fn rasterize(projected: &ProjectedTrajectories, params: ProxyRenderParams) -> InkGrid {
    let mut grid = InkGrid::new(params.grid_size);
    let samples = projected.points[0].len();
    if samples < 2 {
        return grid;
    }
    let chord_lag =
        ((samples as f64 * PROXY_CHORD_LAG_FRACTION).round() as usize).clamp(1, samples - 1);

    let draw_edges = |grid: &mut InkGrid, skip_edge: Option<usize>, amount: f64| {
        for step in 0..samples {
            let a = projected.points[0][step];
            let b = projected.points[1][step];
            let c = projected.points[2][step];
            let edges = [(a, b), (b, c), (c, a)];
            for (edge_idx, ((x0, y0), (x1, y1))) in edges.into_iter().enumerate() {
                if skip_edge == Some(edge_idx) {
                    continue;
                }
                grid.draw_segment(x0, y0, x1, y1, edge_idx, amount);
            }
        }
    };
    let draw_ribbons = |grid: &mut InkGrid, amount: f64| {
        for body in 0..3 {
            for step in 0..samples - 1 {
                let (x0, y0) = projected.points[body][step];
                let (x1, y1) = projected.points[body][step + 1];
                grid.draw_segment(x0, y0, x1, y1, body, amount);
            }
        }
    };
    let draw_chords = |grid: &mut InkGrid, amount: f64| {
        for body in 0..3 {
            for step in 0..samples - chord_lag {
                let (x0, y0) = projected.points[body][step];
                let (x1, y1) = projected.points[body][step + chord_lag];
                grid.draw_segment(x0, y0, x1, y1, body, amount);
            }
        }
    };
    let draw_spokes = |grid: &mut InkGrid, amount: f64| {
        for step in 0..samples {
            let a = projected.points[0][step];
            let b = projected.points[1][step];
            let c = projected.points[2][step];
            let cx = (a.0 + b.0 + c.0) / 3.0;
            let cy = (a.1 + b.1 + c.1) / 3.0;
            for (body, (x, y)) in [a, b, c].into_iter().enumerate() {
                grid.draw_segment(x, y, cx, cy, body, amount);
            }
        }
    };

    match params.structure {
        StructureMode::TriangleWeb => draw_edges(&mut grid, None, 1.0),
        StructureMode::Duet { dropped_edge } => {
            draw_edges(&mut grid, Some(usize::from(dropped_edge.min(2))), 1.0);
        }
        StructureMode::OrbitRibbons => draw_ribbons(&mut grid, 1.0),
        StructureMode::WebRibbonHybrid => {
            draw_edges(&mut grid, None, 0.30);
            draw_ribbons(&mut grid, 1.0);
        }
        StructureMode::Spokes => draw_spokes(&mut grid, 1.0),
        StructureMode::TimeChords => {
            draw_ribbons(&mut grid, 0.30);
            draw_chords(&mut grid, 1.0);
        }
        StructureMode::CometRibbons => {
            draw_ribbons(&mut grid, 1.0);
            draw_chords(&mut grid, 0.45);
        }
        StructureMode::WebSpokesLace => {
            draw_edges(&mut grid, None, 0.55);
            draw_spokes(&mut grid, 0.45);
        }
    }

    grid
}

/// Compute aesthetic metrics from an accumulated ink grid.
fn score_grid(grid: &InkGrid) -> AestheticScore {
    let pixel_count = grid.ink.len();
    if pixel_count == 0 {
        return AestheticScore::zero();
    }

    let mut total_ink = [0.0f64; 3];
    let mut inked_pixels = 0usize;
    let mut densities: Vec<f64> = Vec::new();
    let mut cell_occupancy = [0.0f64; BALANCE_CELLS * BALANCE_CELLS];
    let cell_span = (grid.size as f64 / BALANCE_CELLS as f64).max(1.0);

    for (idx, ink) in grid.ink.iter().enumerate() {
        let density = ink[0] + ink[1] + ink[2];
        if density <= 0.0 {
            continue;
        }
        inked_pixels += 1;
        densities.push(density);
        for body in 0..3 {
            total_ink[body] += ink[body];
        }
        let x = idx % grid.size;
        let y = idx / grid.size;
        let cell_x = ((x as f64 / cell_span) as usize).min(BALANCE_CELLS - 1);
        let cell_y = ((y as f64 / cell_span) as usize).min(BALANCE_CELLS - 1);
        cell_occupancy[cell_y * BALANCE_CELLS + cell_x] += density;
    }

    if inked_pixels == 0 {
        return AestheticScore::zero();
    }

    let coverage = inked_pixels as f64 / pixel_count as f64;
    let coverage_score = coverage_band_score(coverage);

    // Spatial balance: normalized entropy of the per-cell ink distribution.
    let occupancy_sum: f64 = cell_occupancy.iter().sum();
    let balance = if occupancy_sum > 0.0 {
        let entropy: f64 = cell_occupancy
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / occupancy_sum;
                -p * p.ln()
            })
            .sum();
        entropy / (cell_occupancy.len() as f64).ln()
    } else {
        0.0
    };

    // Structural contrast: spread of the log-density distribution.
    densities.sort_by(f64::total_cmp);
    let percentile = |q: f64| -> f64 {
        let idx = (q * (densities.len() - 1) as f64).round() as usize;
        densities[idx.min(densities.len() - 1)]
    };
    let p50 = percentile(0.50).max(1e-12);
    let p95 = percentile(0.95).max(1e-12);
    let log_spread = (p95 / p50).ln();
    let contrast = smoothstep(log_spread / 2.3);

    // Per-body presence: every body should own a visible share of the ink.
    let ink_sum: f64 = total_ink.iter().sum();
    let min_share = total_ink.iter().fold(f64::INFINITY, |acc, &v| acc.min(v / ink_sum.max(1e-12)));
    let body_mix = smoothstep(min_share / BODY_PRESENCE_FLOOR.max(1e-12)).min(1.0);

    // Grey-mush: bright pixels where all three bodies pile up at similar weights
    // integrate toward desaturated grey in the real spectral renderer.
    let bright_threshold = percentile(0.60);
    let mut bright = 0usize;
    let mut mush = 0usize;
    for ink in &grid.ink {
        let density = ink[0] + ink[1] + ink[2];
        if density <= 0.0 || density < bright_threshold {
            continue;
        }
        bright += 1;
        let min_component = ink[0].min(ink[1]).min(ink[2]);
        if min_component / density > 0.25 {
            mush += 1;
        }
    }
    let mush_fraction = if bright > 0 { mush as f64 / bright as f64 } else { 0.0 };
    let mush_penalty = ((mush_fraction - MUSH_TOLERANCE).max(0.0) / (1.0 - MUSH_TOLERANCE)) * 0.30;

    let weighted = 0.40 * coverage_score + 0.25 * balance + 0.20 * contrast + 0.15 * body_mix;
    let total = (weighted - mush_penalty).clamp(0.0, 1.0);

    AestheticScore { total, coverage, coverage_score, balance, contrast, body_mix, mush_fraction }
}

/// Proxy-render `positions` with the seed's structure mode and score the result.
///
/// Returns a zero score for degenerate inputs (fewer than two steps, NaN
/// geometry) so callers can rank without special cases.
#[must_use]
pub fn score_trajectory(
    positions: &[Vec<Vector3<f64>>],
    params: ProxyRenderParams,
) -> AestheticScore {
    let Some(projected) = project_to_grid(positions, params.grid_size, params.samples_per_body)
    else {
        return AestheticScore::zero();
    };
    let grid = rasterize(&projected, params);
    score_grid(&grid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn looping_positions(steps: usize, loops: f64) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let phase = f64::from(body) * 2.1;
                let radius = 1.0 + f64::from(body) * 0.4;
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64 * std::f64::consts::TAU * loops;
                        Vector3::new(
                            radius * (t + phase).cos() + 0.3 * (2.7 * t + phase).sin(),
                            radius * (t + phase).sin() + 0.3 * (1.9 * t).cos(),
                            0.0,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    fn linear_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let offset = f64::from(body) * 0.001;
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64;
                        Vector3::new(t, t + offset, 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn rich_tangle_outscores_degenerate_line() {
        let rich = looping_positions(4_000, 7.0);
        let dull = linear_positions(4_000);
        let params = ProxyRenderParams::for_candidates(StructureMode::TriangleWeb);

        let rich_score = score_trajectory(&rich, params);
        let dull_score = score_trajectory(&dull, params);
        assert!(
            rich_score.total > dull_score.total,
            "tangle should outscore a line: rich={} dull={}",
            rich_score.total,
            dull_score.total
        );
    }

    #[test]
    fn score_is_deterministic() {
        let positions = looping_positions(2_000, 5.0);
        let params = ProxyRenderParams::for_candidates(StructureMode::OrbitRibbons);
        let a = score_trajectory(&positions, params);
        let b = score_trajectory(&positions, params);
        assert_eq!(a.total.to_bits(), b.total.to_bits());
    }

    #[test]
    fn empty_and_degenerate_inputs_score_zero() {
        let empty: Vec<Vec<Vector3<f64>>> = vec![Vec::new(), Vec::new(), Vec::new()];
        let params = ProxyRenderParams::for_candidates(StructureMode::TriangleWeb);
        assert_eq!(score_trajectory(&empty, params).total, 0.0);

        let nan = vec![
            vec![Vector3::new(f64::NAN, 0.0, 0.0); 8],
            vec![Vector3::new(0.0, f64::NAN, 0.0); 8],
            vec![Vector3::new(f64::NAN, f64::NAN, 0.0); 8],
        ];
        assert_eq!(score_trajectory(&nan, params).total, 0.0);
    }

    #[test]
    fn all_structure_modes_produce_finite_scores() {
        let positions = looping_positions(3_000, 6.0);
        let modes = [
            StructureMode::TriangleWeb,
            StructureMode::OrbitRibbons,
            StructureMode::WebRibbonHybrid,
            StructureMode::Duet { dropped_edge: 1 },
            StructureMode::Spokes,
            StructureMode::TimeChords,
            StructureMode::CometRibbons,
            StructureMode::WebSpokesLace,
        ];
        for mode in modes {
            let score = score_trajectory(&positions, ProxyRenderParams::for_candidates(mode));
            assert!(
                score.total.is_finite() && (0.0..=1.0).contains(&score.total),
                "mode {mode:?} produced invalid score {score:?}"
            );
            assert!(score.coverage > 0.0, "mode {mode:?} deposited no ink");
        }
    }

    #[test]
    fn coverage_band_rewards_target_density() {
        assert_eq!(coverage_band_score(0.005), 0.0);
        assert_eq!(coverage_band_score(0.9), 0.0);
        assert!(coverage_band_score(0.2) > coverage_band_score(0.03));
        assert!((coverage_band_score(0.25) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn view_params_are_cheaper_than_candidate_params() {
        let candidate = ProxyRenderParams::for_candidates(StructureMode::TriangleWeb);
        let view = ProxyRenderParams::for_view_selection(StructureMode::TriangleWeb);
        assert!(view.grid_size < candidate.grid_size);
        assert!(view.samples_per_body < candidate.samples_per_body);
    }
}
