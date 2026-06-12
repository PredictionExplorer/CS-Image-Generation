//! Geometry-space aesthetic scoring for candidate orbits and view rotations.
//!
//! The Borda search ranks orbits by physics proxies (FFT regularity and
//! triangle balance) but never looks at the picture. This module closes that
//! gap with a cheap proxy render: sampled trajectory geometry is splatted onto
//! a small per-body ink grid using the seed's full layer stack, then scored on
//! image-space qualities (ink coverage, spatial balance, structural contrast,
//! crisp line energy, interior negative space, per-body colour separation, and
//! penalties for dense mush or low-gradient veils). Stacks that intentionally
//! sweep translucent veils are scored with a relaxed veil profile so the soft
//! gauze is not mistaken for a defect. Scores are deterministic functions of
//! geometry, so candidate selection stays reproducible per seed.

use super::visual_profile::{LayerStack, StructureMode, VocabularyFamily};
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
///
/// Tightened from 0.42: sprawling compositions spread the fixed ink budget so
/// thin that production strokes render as faint hairlines.
const COVERAGE_BAND_HIGH: f64 = 0.36;
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
/// Local normalized density gradient below which lit pixels read as a flat veil.
const VEIL_GRADIENT_THRESHOLD: f64 = 0.050;
/// Local normalized density gradient above which lit pixels read as crisp line work.
const CRISP_GRADIENT_THRESHOLD: f64 = 0.180;
/// Flat-fill fraction tolerated before the score starts dropping hard.
const VEIL_TOLERANCE: f64 = 0.12;
/// Cells per axis for the negative-space occupancy grid.
const NEGATIVE_SPACE_CELLS: usize = 8;
/// Time-lag fraction used to proxy chord-style modes during scoring.
const PROXY_CHORD_LAG_FRACTION: f64 = 0.012;
/// Multiple of the median lit density above which a pixel counts as
/// revisit-rich (full-bodied accumulation rather than a one-pass scribble).
const FULLNESS_DENSITY_FACTOR: f64 = 2.5;
/// Revisit-rich fraction of lit pixels that maps to a full fullness score.
const FULLNESS_SATURATION: f64 = 0.18;

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
    /// Fraction of lit pixels whose neighbourhood has very low gradient.
    pub veil_fraction: f64,
    /// Fraction of lit pixels with strong local edge energy.
    pub crispness: f64,
    /// Coarse interior dark-space score in `[0, 1]`.
    pub negative_space: f64,
    /// Full-bodied stroke score in `[0, 1]`: saturating fraction of lit pixels
    /// whose accumulated ink is well above the median (repeated passes, slow
    /// cusps, crossings) rather than one-pass hairline deposits.
    pub fullness: f64,
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
            veil_fraction: 0.0,
            crispness: 0.0,
            negative_space: 0.0,
            fullness: 0.0,
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
    /// Layer stack used to convert geometry into strokes.
    pub stack: LayerStack,
}

impl ProxyRenderParams {
    /// Parameters tuned for scoring shortlisted orbit candidates.
    #[must_use]
    pub fn for_candidates(stack: LayerStack) -> Self {
        Self { grid_size: CANDIDATE_GRID_SIZE, samples_per_body: CANDIDATE_SAMPLES_PER_BODY, stack }
    }

    /// Cheaper parameters tuned for ranking candidate view rotations.
    #[must_use]
    pub fn for_view_selection(stack: LayerStack) -> Self {
        Self { grid_size: VIEW_GRID_SIZE, samples_per_body: VIEW_SAMPLES_PER_BODY, stack }
    }
}

/// Scoring weights, derived from the stack being evaluated.
///
/// Veil-bearing stacks intentionally sweep low-gradient gauze, so the veil
/// penalty is relaxed and the crispness weight is partially redistributed to
/// balance/contrast; otherwise soft area glow would be punished as a defect
/// and veil compositions could never win adaptive selection.
#[derive(Clone, Copy, Debug)]
struct ScoreProfile {
    coverage_weight: f64,
    balance_weight: f64,
    contrast_weight: f64,
    body_mix_weight: f64,
    crisp_weight: f64,
    negative_weight: f64,
    fullness_weight: f64,
    veil_tolerance: f64,
    veil_penalty_scale: f64,
}

impl ScoreProfile {
    fn for_stack(stack: &LayerStack) -> Self {
        if stack.contains_family(VocabularyFamily::Veil) {
            Self {
                coverage_weight: 0.26,
                balance_weight: 0.20,
                contrast_weight: 0.14,
                body_mix_weight: 0.10,
                crisp_weight: 0.18,
                negative_weight: 0.08,
                fullness_weight: 0.06,
                veil_tolerance: 0.22,
                veil_penalty_scale: 0.30,
            }
        } else {
            Self {
                coverage_weight: 0.28,
                balance_weight: 0.18,
                contrast_weight: 0.10,
                body_mix_weight: 0.10,
                crisp_weight: 0.22,
                negative_weight: 0.10,
                fullness_weight: 0.12,
                veil_tolerance: VEIL_TOLERANCE,
                veil_penalty_scale: 0.40,
            }
        }
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

fn negative_space_score(density_map: &[f64], size: usize) -> f64 {
    let mut min_x = size;
    let mut min_y = size;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;
    for (idx, &density) in density_map.iter().enumerate() {
        if density <= 0.0 {
            continue;
        }
        found = true;
        let x = idx % size;
        let y = idx / size;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    if !found || max_x <= min_x + 2 || max_y <= min_y + 2 {
        return 0.0;
    }

    let mut occupied = [false; NEGATIVE_SPACE_CELLS * NEGATIVE_SPACE_CELLS];
    for (idx, &density) in density_map.iter().enumerate() {
        if density <= 0.0 {
            continue;
        }
        let x = idx % size;
        let y = idx / size;
        if x < min_x || x > max_x || y < min_y || y > max_y {
            continue;
        }
        let nx = (x - min_x) as f64 / (max_x - min_x).max(1) as f64;
        let ny = (y - min_y) as f64 / (max_y - min_y).max(1) as f64;
        let cell_x = (nx * NEGATIVE_SPACE_CELLS as f64).floor() as usize;
        let cell_y = (ny * NEGATIVE_SPACE_CELLS as f64).floor() as usize;
        let cell_x = cell_x.min(NEGATIVE_SPACE_CELLS - 1);
        let cell_y = cell_y.min(NEGATIVE_SPACE_CELLS - 1);
        occupied[cell_y * NEGATIVE_SPACE_CELLS + cell_x] = true;
    }

    let mut interior = 0usize;
    let mut dark = 0usize;
    for y in 1..NEGATIVE_SPACE_CELLS - 1 {
        for x in 1..NEGATIVE_SPACE_CELLS - 1 {
            interior += 1;
            if !occupied[y * NEGATIVE_SPACE_CELLS + x] {
                dark += 1;
            }
        }
    }
    if interior == 0 {
        return 0.0;
    }

    let dark_ratio = dark as f64 / interior as f64;
    let enough_void = smoothstep(dark_ratio / 0.22);
    let not_empty_frame = smoothstep((0.70 - dark_ratio) / 0.35);
    let center = (NEGATIVE_SPACE_CELLS / 2) * NEGATIVE_SPACE_CELLS + (NEGATIVE_SPACE_CELLS / 2);
    let center_void_bonus = if occupied[center] { 0.0 } else { 0.65 };
    (enough_void * not_empty_frame).max(center_void_bonus).clamp(0.0, 1.0)
}

fn local_gradient(density_map: &[f64], size: usize, idx: usize) -> f64 {
    let x = idx % size;
    let y = idx / size;
    let center = density_map[idx];
    let mut gradient = 0.0f64;
    if x > 0 {
        gradient = gradient.max((center - density_map[idx - 1]).abs());
    }
    if x + 1 < size {
        gradient = gradient.max((center - density_map[idx + 1]).abs());
    }
    if y > 0 {
        gradient = gradient.max((center - density_map[idx - size]).abs());
    }
    if y + 1 < size {
        gradient = gradient.max((center - density_map[idx + size]).abs());
    }
    gradient
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

/// Splat the sampled geometry according to the layer stack.
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
    // Veil proxy: three interior fill lines approximating the swept gauze.
    let draw_veil = |grid: &mut InkGrid, amount: f64| {
        const FILL_LINES: usize = 3;
        let line_amount = amount / FILL_LINES as f64;
        for step in 0..samples {
            let pivot = step % 3;
            let a = projected.points[pivot][step];
            let b = projected.points[(pivot + 1) % 3][step];
            let c = projected.points[(pivot + 2) % 3][step];
            for k in 1..=FILL_LINES {
                let t = k as f64 / (FILL_LINES + 1) as f64;
                let x0 = a.0 + (b.0 - a.0) * t;
                let y0 = a.1 + (b.1 - a.1) * t;
                let x1 = a.0 + (c.0 - a.0) * t;
                let y1 = a.1 + (c.1 - a.1) * t;
                grid.draw_segment(x0, y0, x1, y1, pivot, line_amount);
            }
        }
    };
    // Weave proxy: bowed Bezier chords tessellated coarsely.
    let draw_weave = |grid: &mut InkGrid, amount: f64| {
        const SEGMENTS: usize = 4;
        const BOW: f64 = 0.5;
        let segment_amount = amount / SEGMENTS as f64 * 2.0;
        for step in (0..samples).step_by(2) {
            let points =
                [projected.points[0][step], projected.points[1][step], projected.points[2][step]];
            let cx = (points[0].0 + points[1].0 + points[2].0) / 3.0;
            let cy = (points[0].1 + points[1].1 + points[2].1) / 3.0;
            for (i, j, third) in [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)] {
                let ctrl = (cx + (cx - points[third].0) * BOW, cy + (cy - points[third].1) * BOW);
                let mut prev = points[i];
                for s in 1..=SEGMENTS {
                    let t = s as f64 / SEGMENTS as f64;
                    let u = 1.0 - t;
                    let x = u * u * points[i].0 + 2.0 * u * t * ctrl.0 + t * t * points[j].0;
                    let y = u * u * points[i].1 + 2.0 * u * t * ctrl.1 + t * t * points[j].1;
                    grid.draw_segment(prev.0, prev.1, x, y, i, segment_amount);
                    prev = (x, y);
                }
            }
        }
    };
    // Stipple proxy: time-pitched dots (slow passages cluster).
    let draw_stipple = |grid: &mut InkGrid, amount: f64| {
        let pitch = (samples / 400).max(1);
        let dot_amount = amount * pitch as f64 * 0.6;
        for body in 0..3 {
            for step in (0..samples).step_by(pitch) {
                let (x, y) = projected.points[body][step];
                grid.deposit(x, y, body, dot_amount);
            }
        }
    };
    // Tangent proxy: velocity-aligned segments centered on each body.
    let draw_tangent = |grid: &mut InkGrid, amount: f64| {
        let half = (params.grid_size as f64 * 0.025).max(1.0);
        for body in 0..3 {
            for step in 0..samples - 1 {
                let (x0, y0) = projected.points[body][step];
                let (x1, y1) = projected.points[body][step + 1];
                let dx = x1 - x0;
                let dy = y1 - y0;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-9 || !len.is_finite() {
                    continue;
                }
                let ux = dx / len * half;
                let uy = dy / len * half;
                grid.draw_segment(x0 - ux, y0 - uy, x0 + ux, y0 + uy, body, amount);
            }
        }
    };

    for layer in params.stack.layers() {
        let amount = layer.alpha;
        match layer.vocabulary {
            StructureMode::TriangleWeb => draw_edges(&mut grid, None, amount),
            StructureMode::Duet { dropped_edge } => {
                draw_edges(&mut grid, Some(usize::from(dropped_edge.min(2))), amount);
            }
            StructureMode::OrbitRibbons => draw_ribbons(&mut grid, amount),
            StructureMode::Spokes => draw_spokes(&mut grid, amount),
            StructureMode::TimeChords => {
                draw_ribbons(&mut grid, amount * 0.30);
                draw_chords(&mut grid, amount);
            }
            StructureMode::NebulaVeil => draw_veil(&mut grid, amount),
            StructureMode::HarmonicWeave => draw_weave(&mut grid, amount),
            StructureMode::StippleConstellation => draw_stipple(&mut grid, amount),
            StructureMode::TangentCaustics => draw_tangent(&mut grid, amount),
        }
    }

    grid
}

/// Compute aesthetic metrics from an accumulated ink grid.
fn score_grid(grid: &InkGrid, profile: ScoreProfile) -> AestheticScore {
    let pixel_count = grid.ink.len();
    if pixel_count == 0 {
        return AestheticScore::zero();
    }

    let mut total_ink = [0.0f64; 3];
    let mut inked_pixels = 0usize;
    let mut density_map = vec![0.0f64; pixel_count];
    let mut densities: Vec<f64> = Vec::new();
    let mut cell_occupancy = [0.0f64; BALANCE_CELLS * BALANCE_CELLS];
    let cell_span = (grid.size as f64 / BALANCE_CELLS as f64).max(1.0);

    for (idx, ink) in grid.ink.iter().enumerate() {
        let density = ink[0] + ink[1] + ink[2];
        density_map[idx] = density;
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
    let density_scale = p95.max(p50).max(1e-12);

    // Full-bodied strokes: lit pixels whose ink is well above the median come
    // from repeated passes (loops, slow cusps, crossings). One-pass hairline
    // scribbles keep almost every lit pixel near the single-deposit level.
    let fullness_threshold = p50 * FULLNESS_DENSITY_FACTOR;
    let revisit_rich = densities.partition_point(|&d| d < fullness_threshold);
    let revisit_fraction = (densities.len() - revisit_rich) as f64 / densities.len() as f64;
    let fullness = smoothstep(revisit_fraction / FULLNESS_SATURATION);

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

    let mut lit_for_edges = 0usize;
    let mut veiled = 0usize;
    let mut crisp = 0usize;
    for (idx, &density) in density_map.iter().enumerate() {
        if density <= 0.0 {
            continue;
        }
        lit_for_edges += 1;
        let normalized_gradient = local_gradient(&density_map, grid.size, idx) / density_scale;
        if normalized_gradient < VEIL_GRADIENT_THRESHOLD {
            veiled += 1;
        }
        if normalized_gradient > CRISP_GRADIENT_THRESHOLD {
            crisp += 1;
        }
    }
    let veil_fraction = if lit_for_edges > 0 { veiled as f64 / lit_for_edges as f64 } else { 0.0 };
    let crispness = if lit_for_edges > 0 { crisp as f64 / lit_for_edges as f64 } else { 0.0 };
    let negative_space = negative_space_score(&density_map, grid.size);
    let veil_penalty = ((veil_fraction - profile.veil_tolerance).max(0.0)
        / (1.0 - profile.veil_tolerance))
        * profile.veil_penalty_scale;

    let weighted = profile.coverage_weight * coverage_score
        + profile.balance_weight * balance
        + profile.contrast_weight * contrast
        + profile.body_mix_weight * body_mix
        + profile.crisp_weight * crispness
        + profile.negative_weight * negative_space
        + profile.fullness_weight * fullness;
    let total = (weighted - mush_penalty - veil_penalty).clamp(0.0, 1.0);

    AestheticScore {
        total,
        coverage,
        coverage_score,
        balance,
        contrast,
        body_mix,
        mush_fraction,
        veil_fraction,
        crispness,
        negative_space,
        fullness,
    }
}

/// Proxy-render `positions` with the seed's layer stack and score the result.
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
    score_grid(&grid, ScoreProfile::for_stack(&params.stack))
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

    fn flat_disc_grid(size: usize) -> InkGrid {
        let mut grid = InkGrid::new(size);
        let center = (size as f64 - 1.0) * 0.5;
        let radius = size as f64 * 0.28;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                if (dx * dx + dy * dy).sqrt() <= radius {
                    let idx = y * size + x;
                    // Dominated by two bodies so the fixture isolates the
                    // flat-veil signal instead of also saturating the
                    // three-body mush penalty.
                    grid.ink[idx] = [1.0, 0.8, 0.1];
                }
            }
        }
        grid
    }

    fn ring_grid(size: usize) -> InkGrid {
        let mut grid = InkGrid::new(size);
        let center = (size as f64 - 1.0) * 0.5;
        let radius = size as f64 * 0.30;
        for body in 0..3 {
            let phase = f64::from(body as u32) * std::f64::consts::TAU / 3.0;
            let mut prev = None;
            for i in 0..360 {
                let t = f64::from(i) / 360.0 * std::f64::consts::TAU;
                let wobble = radius * (1.0 + 0.08 * (3.0 * t + phase).sin());
                let point = (center + wobble * t.cos(), center + wobble * t.sin());
                if let Some((x0, y0)) = prev {
                    grid.draw_segment(x0, y0, point.0, point.1, body, 1.0);
                }
                prev = Some(point);
            }
        }
        grid
    }

    #[test]
    fn rich_tangle_outscores_degenerate_line() {
        let rich = looping_positions(4_000, 7.0);
        let dull = linear_positions(4_000);
        let params = ProxyRenderParams::for_candidates(LayerStack::with_underlay(
            StructureMode::OrbitRibbons,
            StructureMode::TimeChords,
            0.45,
        ));

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
        let params =
            ProxyRenderParams::for_candidates(LayerStack::solo(StructureMode::OrbitRibbons));
        let a = score_trajectory(&positions, params);
        let b = score_trajectory(&positions, params);
        assert_eq!(a.total.to_bits(), b.total.to_bits());
    }

    #[test]
    fn empty_and_degenerate_inputs_score_zero() {
        let empty: Vec<Vec<Vector3<f64>>> = vec![Vec::new(), Vec::new(), Vec::new()];
        let params =
            ProxyRenderParams::for_candidates(LayerStack::solo(StructureMode::TriangleWeb));
        assert_eq!(score_trajectory(&empty, params).total, 0.0);

        let nan = vec![
            vec![Vector3::new(f64::NAN, 0.0, 0.0); 8],
            vec![Vector3::new(0.0, f64::NAN, 0.0); 8],
            vec![Vector3::new(f64::NAN, f64::NAN, 0.0); 8],
        ];
        assert_eq!(score_trajectory(&nan, params).total, 0.0);
    }

    #[test]
    fn all_vocabularies_produce_finite_scores() {
        let positions = looping_positions(3_000, 6.0);
        let vocabularies = [
            StructureMode::TriangleWeb,
            StructureMode::OrbitRibbons,
            StructureMode::Duet { dropped_edge: 1 },
            StructureMode::Spokes,
            StructureMode::TimeChords,
            StructureMode::NebulaVeil,
            StructureMode::HarmonicWeave,
            StructureMode::StippleConstellation,
            StructureMode::TangentCaustics,
        ];
        for vocabulary in vocabularies {
            let score = score_trajectory(
                &positions,
                ProxyRenderParams::for_candidates(LayerStack::solo(vocabulary)),
            );
            assert!(
                score.total.is_finite() && (0.0..=1.0).contains(&score.total),
                "vocabulary {vocabulary:?} produced invalid score {score:?}"
            );
            assert!(score.coverage > 0.0, "vocabulary {vocabulary:?} deposited no ink");
            assert!((0.0..=1.0).contains(&score.veil_fraction));
            assert!((0.0..=1.0).contains(&score.crispness));
            assert!((0.0..=1.0).contains(&score.negative_space));
        }
    }

    #[test]
    fn stacked_layers_deposit_more_ink_than_solo_primary() {
        let positions = looping_positions(3_000, 6.0);
        let solo = score_trajectory(
            &positions,
            ProxyRenderParams::for_candidates(LayerStack::solo(StructureMode::TriangleWeb)),
        );
        let stacked = score_trajectory(
            &positions,
            ProxyRenderParams::for_candidates(LayerStack::with_underlay(
                StructureMode::TriangleWeb,
                StructureMode::NebulaVeil,
                0.40,
            )),
        );
        assert!(
            stacked.coverage > solo.coverage,
            "underlay must add ink: solo={} stacked={}",
            solo.coverage,
            stacked.coverage
        );
    }

    #[test]
    fn veil_metric_penalizes_flat_fills_for_line_stacks() {
        let line_profile = ScoreProfile::for_stack(&LayerStack::solo(StructureMode::TriangleWeb));
        let flat = score_grid(&flat_disc_grid(96), line_profile);
        let ring = score_grid(&ring_grid(96), line_profile);

        assert!(
            flat.veil_fraction > ring.veil_fraction,
            "flat fill should have more veil: flat={flat:?} ring={ring:?}"
        );
        assert!(
            ring.crispness > flat.crispness,
            "ring should have more crisp line energy: flat={flat:?} ring={ring:?}"
        );
        assert!(
            ring.total > flat.total,
            "line/ring structure should beat flat translucent film: flat={flat:?} ring={ring:?}"
        );
    }

    #[test]
    fn veil_stacks_score_soft_gauze_more_gently() {
        let line_profile = ScoreProfile::for_stack(&LayerStack::solo(StructureMode::TriangleWeb));
        let veil_profile = ScoreProfile::for_stack(&LayerStack::solo(StructureMode::NebulaVeil));
        let flat = flat_disc_grid(96);

        let as_lines = score_grid(&flat, line_profile);
        let as_veil = score_grid(&flat, veil_profile);
        assert!(
            as_veil.total > as_lines.total,
            "intentional veil stacks must not be punished for soft gauze: \
             veil={} lines={}",
            as_veil.total,
            as_lines.total
        );
    }

    #[test]
    fn negative_space_rewards_open_interior_structure() {
        let profile = ScoreProfile::for_stack(&LayerStack::solo(StructureMode::TriangleWeb));
        let flat = score_grid(&flat_disc_grid(96), profile);
        let ring = score_grid(&ring_grid(96), profile);

        assert!(
            ring.negative_space > flat.negative_space,
            "open ring should carry more negative-space score: flat={flat:?} ring={ring:?}"
        );
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
        let stack = LayerStack::solo(StructureMode::TriangleWeb);
        let candidate = ProxyRenderParams::for_candidates(stack);
        let view = ProxyRenderParams::for_view_selection(stack);
        assert!(view.grid_size < candidate.grid_size);
        assert!(view.samples_per_body < candidate.samples_per_body);
    }
}
