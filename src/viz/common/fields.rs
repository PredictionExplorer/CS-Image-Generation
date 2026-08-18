//! Gravitational field infrastructure (master plan Part II.4): pixel-aligned
//! potential grids, asinh-spaced equipotential polylines, evenly spaced
//! field-line streamlines (Jobard-Lefer), and co-rotating Roche effective
//! potentials with sub-cell L1 saddle location.
//!
//! All functions are deterministic; grids sample the plane-of-record (the
//! projected xy positions the render pipeline draws).

use crate::render::context::RenderContext;
use crate::sim::G;
use crate::viz::common::contours::{Segment, marching_squares};
use rayon::prelude::*;

/// A polyline in pixel coordinates of the sampling grid.
pub type Polyline = Vec<(f32, f32)>;

/// One point mass in world coordinates (projected xy plane).
#[derive(Clone, Copy, Debug)]
pub struct PointMass {
    /// World x coordinate.
    pub x: f64,
    /// World y coordinate.
    pub y: f64,
    /// Body mass.
    pub mass: f64,
}

/// Plummer-softened gravitational potential of a set of point masses:
/// `phi(p) = -G * sum m_i / sqrt(|p - p_i|^2 + s^2)`.
fn point_potential(masses: &[PointMass], x: f64, y: f64, softening_sq: f64) -> f64 {
    let mut phi = 0.0;
    for body in masses {
        let dx = x - body.x;
        let dy = y - body.y;
        phi -= G * body.mass / (dx * dx + dy * dy + softening_sq).sqrt();
    }
    phi
}

/// Analytic world-space gradient of [`point_potential`].
fn point_gradient(masses: &[PointMass], x: f64, y: f64, softening_sq: f64) -> (f64, f64) {
    let mut gx = 0.0;
    let mut gy = 0.0;
    for body in masses {
        let dx = x - body.x;
        let dy = y - body.y;
        let dist_sq = dx * dx + dy * dy + softening_sq;
        let inv = G * body.mass / (dist_sq * dist_sq.sqrt());
        gx += dx * inv;
        gy += dy * inv;
    }
    (gx, gy)
}

/// Gravitational potential sampled at pixel centers of a render context.
pub struct PotentialGrid {
    /// Grid width in pixels.
    pub width: usize,
    /// Grid height in pixels.
    pub height: usize,
    /// Row-major potential values at pixel centers.
    pub values: Vec<f32>,
}

impl PotentialGrid {
    /// Sample `phi` on the pixel grid of `ctx` (one value per pixel center).
    ///
    /// `softening_world` is the Plummer softening length in world units
    /// (the spec's "2 px in world units" is computed by the caller from the
    /// target context's pixel scale).
    #[must_use]
    pub fn sample(masses: &[PointMass], ctx: &RenderContext, softening_world: f64) -> Self {
        let width = ctx.width_usize;
        let height = ctx.height_usize;
        let bounds = *ctx.bounds();
        let step_x = bounds.width / width.max(1) as f64;
        let step_y = bounds.height / height.max(1) as f64;
        let softening_sq = softening_world * softening_world;

        let mut values = vec![0.0f32; width * height];
        values.par_chunks_mut(width).enumerate().for_each(|(row, out)| {
            let world_y = bounds.min_y + (row as f64 + 0.5) * step_y;
            for (col, slot) in out.iter_mut().enumerate() {
                let world_x = bounds.min_x + (col as f64 + 0.5) * step_x;
                *slot = point_potential(masses, world_x, world_y, softening_sq) as f32;
            }
        });
        Self { width, height, values }
    }

    /// Wrap precomputed row-major values (Roche grids, tests).
    #[must_use]
    pub fn from_values(values: Vec<f32>, width: usize, height: usize) -> Self {
        debug_assert_eq!(values.len(), width * height);
        Self { width, height, values }
    }

    /// Value at integer node (clamped).
    #[inline]
    fn node(&self, col: i64, row: i64) -> f64 {
        let col = col.clamp(0, self.width as i64 - 1) as usize;
        let row = row.clamp(0, self.height as i64 - 1) as usize;
        f64::from(self.values[row * self.width + col])
    }

    /// Bilinear potential value at pixel coordinates.
    #[must_use]
    pub fn value_at(&self, x: f64, y: f64) -> f64 {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let col = fx.floor() as i64;
        let row = fy.floor() as i64;
        let tx = fx - col as f64;
        let ty = fy - row as f64;
        let top = self.node(col, row) * (1.0 - tx) + self.node(col + 1, row) * tx;
        let bottom = self.node(col, row + 1) * (1.0 - tx) + self.node(col + 1, row + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// Central-difference gradient at an integer node, in pixel units.
    #[inline]
    fn node_gradient(&self, col: i64, row: i64) -> (f64, f64) {
        let gx = (self.node(col + 1, row) - self.node(col - 1, row)) * 0.5;
        let gy = (self.node(col, row + 1) - self.node(col, row - 1)) * 0.5;
        (gx, gy)
    }

    /// Bilinear pixel-space gradient of the potential at pixel coordinates.
    #[must_use]
    pub fn gradient_at(&self, x: f64, y: f64) -> (f64, f64) {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let col = fx.floor() as i64;
        let row = fy.floor() as i64;
        let tx = fx - col as f64;
        let ty = fy - row as f64;
        let g00 = self.node_gradient(col, row);
        let g10 = self.node_gradient(col + 1, row);
        let g01 = self.node_gradient(col, row + 1);
        let g11 = self.node_gradient(col + 1, row + 1);
        let top = (g00.0 * (1.0 - tx) + g10.0 * tx, g00.1 * (1.0 - tx) + g10.1 * tx);
        let bottom = (g01.0 * (1.0 - tx) + g11.0 * tx, g01.1 * (1.0 - tx) + g11.1 * tx);
        (top.0 * (1.0 - ty) + bottom.0 * ty, top.1 * (1.0 - ty) + bottom.1 * ty)
    }

    /// Contour levels evenly spaced in `asinh(phi)` between robust
    /// percentiles of the sampled values (interior spacing: no level sits at
    /// either percentile endpoint).
    #[must_use]
    pub fn asinh_levels(&self, count: usize) -> Vec<f32> {
        if self.values.is_empty() || count == 0 {
            return Vec::new();
        }
        let mut sorted: Vec<f32> =
            self.values.iter().copied().filter(|value| value.is_finite()).collect();
        sorted.sort_by(f32::total_cmp);
        if sorted.is_empty() {
            return Vec::new();
        }
        let pick = |quantile: f64| sorted[((sorted.len() - 1) as f64 * quantile) as usize];
        let low = f64::from(pick(0.01)).asinh();
        let high = f64::from(pick(0.995)).asinh();
        (1..=count)
            .map(|index| {
                let t = index as f64 / (count + 1) as f64;
                ((low + (high - low) * t).sinh()) as f32
            })
            .collect()
    }
}

/// Quantize a contour endpoint to a hash key (1/16 px lattice).
#[inline]
fn endpoint_key(point: (f32, f32)) -> (i64, i64) {
    ((f64::from(point.0) * 16.0).round() as i64, (f64::from(point.1) * 16.0).round() as i64)
}

/// Stitch marching-squares segments into polylines by shared endpoints.
#[must_use]
pub fn stitch_segments(segments: &[Segment]) -> Vec<Polyline> {
    use std::collections::HashMap;
    // Adjacency: endpoint key -> list of (segment index, end index).
    let mut adjacency: HashMap<(i64, i64), Vec<(usize, u8)>> = HashMap::new();
    for (index, segment) in segments.iter().enumerate() {
        adjacency.entry(endpoint_key(segment.0)).or_default().push((index, 0));
        adjacency.entry(endpoint_key(segment.1)).or_default().push((index, 1));
    }

    let mut used = vec![false; segments.len()];
    let mut polylines = Vec::new();
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut line: Vec<(f32, f32)> = vec![segments[start].0, segments[start].1];

        // Extend forward from the tail, then backward from the head.
        for direction in 0..2 {
            loop {
                let tip = if direction == 0 { *line.last().expect("non-empty") } else { line[0] };
                let Some(candidates) = adjacency.get(&endpoint_key(tip)) else { break };
                let next = candidates.iter().find(|&&(index, _)| !used[index]);
                let Some(&(index, end)) = next else { break };
                used[index] = true;
                let segment = segments[index];
                let other = if end == 0 { segment.1 } else { segment.0 };
                if direction == 0 {
                    line.push(other);
                } else {
                    line.insert(0, other);
                }
            }
        }
        polylines.push(line);
    }
    polylines
}

/// Marching-squares equipotentials of a grid, stitched into polylines,
/// one bundle per requested level.
#[must_use]
pub fn equipotentials(grid: &PotentialGrid, levels: &[f32]) -> Vec<Vec<Polyline>> {
    levels
        .par_iter()
        .map(|&level| {
            stitch_segments(&marching_squares(&grid.values, grid.width, grid.height, level))
        })
        .collect()
}

/// Occupancy hash for streamline separation testing.
struct SeparationMask {
    cell: f64,
    cols: usize,
    rows: usize,
    buckets: Vec<Vec<(f32, f32)>>,
}

impl SeparationMask {
    fn new(width: usize, height: usize, cell: f64) -> Self {
        let cols = (width as f64 / cell).ceil() as usize + 2;
        let rows = (height as f64 / cell).ceil() as usize + 2;
        Self { cell, cols, rows, buckets: vec![Vec::new(); cols * rows] }
    }

    fn bucket_of(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x / self.cell).floor().max(0.0) as usize).min(self.cols - 1);
        let row = ((y / self.cell).floor().max(0.0) as usize).min(self.rows - 1);
        (col, row)
    }

    fn insert(&mut self, point: (f32, f32)) {
        let (col, row) = self.bucket_of(f64::from(point.0), f64::from(point.1));
        self.buckets[row * self.cols + col].push(point);
    }

    /// Whether any registered point lies within `radius` of (x, y).
    fn occupied_within(&self, x: f64, y: f64, radius: f64) -> bool {
        let (col, row) = self.bucket_of(x, y);
        let reach = (radius / self.cell).ceil() as i64;
        let radius_sq = radius * radius;
        for delta_row in -reach..=reach {
            for delta_col in -reach..=reach {
                let bucket_col = col as i64 + delta_col;
                let bucket_row = row as i64 + delta_row;
                if bucket_col < 0
                    || bucket_row < 0
                    || bucket_col >= self.cols as i64
                    || bucket_row >= self.rows as i64
                {
                    continue;
                }
                for point in &self.buckets[bucket_row as usize * self.cols + bucket_col as usize] {
                    let dx = f64::from(point.0) - x;
                    let dy = f64::from(point.1) - y;
                    if dx * dx + dy * dy < radius_sq {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Trace one field line from a seed in both directions (RK2 midpoint on the
/// normalized `-grad(phi)` direction field). Returns `None` when the seed is
/// too close to existing lines or degenerate.
fn trace_streamline(
    grid: &PotentialGrid,
    mask: &SeparationMask,
    seed: (f64, f64),
    step: f64,
    d_test: f64,
    max_points: usize,
) -> Option<Polyline> {
    if mask.occupied_within(seed.0, seed.1, d_test) {
        return None;
    }
    let direction = |x: f64, y: f64, sign: f64| -> Option<(f64, f64)> {
        let (gx, gy) = grid.gradient_at(x, y);
        let norm = (gx * gx + gy * gy).sqrt();
        if norm < 1e-12 {
            return None;
        }
        // Field lines of gravity: follow -grad(phi).
        Some((-gx / norm * sign, -gy / norm * sign))
    };

    // Stay clear of the boundary: central differences degrade to one-sided
    // there and lines would crawl along the frame edge.
    let margin = 1.5f64;
    let in_bounds = |x: f64, y: f64| {
        x >= margin
            && y >= margin
            && x < grid.width as f64 - margin
            && y < grid.height as f64 - margin
    };

    // Self-separation: the line's own most recent points are exempt (the
    // tail cannot collide with itself within one turn radius).
    let ignore_recent = (2.0 * d_test / step).ceil() as usize + 2;

    let mut halves: [Vec<(f32, f32)>; 2] = [Vec::new(), Vec::new()];
    for (half_index, sign) in [1.0f64, -1.0].into_iter().enumerate() {
        let mut own: Vec<(f32, f32)> = Vec::new();
        let (mut x, mut y) = seed;
        for _ in 0..max_points {
            let Some((dx, dy)) = direction(x, y, sign) else { break };
            let mid_x = x + 0.5 * step * dx;
            let mid_y = y + 0.5 * step * dy;
            if !in_bounds(mid_x, mid_y) {
                break;
            }
            let Some((mid_dx, mid_dy)) = direction(mid_x, mid_y, sign) else { break };
            x += step * mid_dx;
            y += step * mid_dy;
            if !in_bounds(x, y) || mask.occupied_within(x, y, d_test) {
                break;
            }
            // Loop closure against the line's own older points.
            if own.len() > ignore_recent {
                let older = &own[..own.len() - ignore_recent];
                let collided = older.iter().any(|point| {
                    let ox = f64::from(point.0) - x;
                    let oy = f64::from(point.1) - y;
                    ox * ox + oy * oy < d_test * d_test
                });
                if collided {
                    break;
                }
            }
            own.push((x as f32, y as f32));
        }
        halves[half_index] = own;
    }

    let [forward, backward] = halves;
    let mut line: Polyline = backward.into_iter().rev().collect();
    line.push((seed.0 as f32, seed.1 as f32));
    line.extend(forward);
    if line.len() < 3 { None } else { Some(line) }
}

/// Register an accepted line: mask points at ~`d_test` arc spacing and seed
/// candidates at +/- `d_sep` along local normals at ~`d_sep` arc spacing.
fn register_streamline(
    line: &Polyline,
    d_sep: f64,
    d_test: f64,
    mask: &mut SeparationMask,
    queue: &mut std::collections::VecDeque<(f64, f64)>,
) {
    let mut since_mask = f64::INFINITY;
    let mut since_seed = f64::INFINITY;
    for window in line.windows(2) {
        let a = window[0];
        let b = window[1];
        let dx = f64::from(b.0 - a.0);
        let dy = f64::from(b.1 - a.1);
        let length = (dx * dx + dy * dy).sqrt().max(1e-9);
        since_mask += length;
        since_seed += length;
        if since_mask >= d_test * 0.9 {
            mask.insert(b);
            since_mask = 0.0;
        }
        if since_seed >= d_sep {
            let (nx, ny) = (-dy / length, dx / length);
            queue.push_back((f64::from(b.0) + nx * d_sep, f64::from(b.1) + ny * d_sep));
            queue.push_back((f64::from(b.0) - nx * d_sep, f64::from(b.1) - ny * d_sep));
            since_seed = 0.0;
        }
    }
    if let Some(&first) = line.first() {
        mask.insert(first);
    }
    if let Some(&last) = line.last() {
        mask.insert(last);
    }
}

/// Evenly spaced streamlines of `-grad(phi)` (Jobard-Lefer): new lines seed
/// at `spacing_px` from accepted lines and terminate at half spacing.
#[must_use]
pub fn streamlines(grid: &PotentialGrid, spacing_px: f64) -> Vec<Polyline> {
    let d_sep = spacing_px.max(2.0);
    let d_test = 0.5 * d_sep;
    let step = (0.25 * d_sep).clamp(0.6, 2.5);
    let max_points = ((grid.width + grid.height) as f64 * 4.0 / step) as usize;

    let mut mask = SeparationMask::new(grid.width, grid.height, d_test.max(1.0));
    let mut lines: Vec<Polyline> = Vec::new();

    // Deterministic first seed: the strongest-gradient node (first in
    // row-major order on ties).
    let mut best = (0usize, 0usize, -1.0f64);
    for row in (0..grid.height).step_by(2) {
        for col in (0..grid.width).step_by(2) {
            let (gx, gy) = grid.node_gradient(col as i64, row as i64);
            let strength = gx * gx + gy * gy;
            if strength > best.2 {
                best = (col, row, strength);
            }
        }
    }
    let mut queue: std::collections::VecDeque<(f64, f64)> = std::collections::VecDeque::new();
    queue.push_back((best.0 as f64 + 0.5, best.1 as f64 + 0.5));

    while let Some(seed) = queue.pop_front() {
        if seed.0 < 0.0
            || seed.1 < 0.0
            || seed.0 >= grid.width as f64
            || seed.1 >= grid.height as f64
        {
            continue;
        }
        if let Some(line) = trace_streamline(grid, &mask, seed, step, d_test, max_points) {
            register_streamline(&line, d_sep, d_test, &mut mask, &mut queue);
            lines.push(line);
        }
    }
    lines
}

/// Co-rotating effective potential of a body pair (analytic):
/// `phi_eff = phi_bodies + phi_perturber - 0.5 * omega^2 * rho^2`, in frame
/// coordinates where the pair lies on the x axis around its barycenter.
pub struct RochePotential {
    /// The pair, in frame coordinates (on the x axis, barycenter at origin).
    pub bodies: [PointMass; 2],
    /// Optional third body (frame coordinates) perturbing the lobes.
    pub perturber: Option<PointMass>,
    /// Squared frame rotation rate.
    pub omega_sq: f64,
    /// Plummer softening length (world units).
    pub softening: f64,
}

impl RochePotential {
    /// Two-body circular-orbit configuration: bodies on the x axis around
    /// the barycenter with `omega^2 = G (m1 + m2) / r^3` (the analytic test
    /// case and the default construction for callers).
    #[must_use]
    pub fn circular(mass_primary: f64, mass_secondary: f64, separation: f64) -> Self {
        let total = mass_primary + mass_secondary;
        Self {
            bodies: [
                PointMass { x: -separation * mass_secondary / total, y: 0.0, mass: mass_primary },
                PointMass { x: separation * mass_primary / total, y: 0.0, mass: mass_secondary },
            ],
            perturber: None,
            omega_sq: G * total / separation.powi(3),
            softening: 0.0,
        }
    }

    fn masses(&self) -> impl Iterator<Item = PointMass> + '_ {
        self.bodies.iter().copied().chain(self.perturber)
    }

    /// Effective potential value at frame coordinates.
    #[must_use]
    pub fn value(&self, x: f64, y: f64) -> f64 {
        let softening_sq = self.softening * self.softening;
        let mut phi = 0.0;
        for body in self.masses() {
            let dx = x - body.x;
            let dy = y - body.y;
            phi -= G * body.mass / (dx * dx + dy * dy + softening_sq).sqrt();
        }
        phi - 0.5 * self.omega_sq * (x * x + y * y)
    }

    /// Analytic gradient of the effective potential.
    #[must_use]
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        let softening_sq = self.softening * self.softening;
        let bodies: Vec<PointMass> = self.masses().collect();
        let (mut gx, mut gy) = point_gradient(&bodies, x, y, softening_sq);
        gx -= self.omega_sq * x;
        gy -= self.omega_sq * y;
        (gx, gy)
    }

    /// Analytic Hessian `(phi_xx, phi_xy, phi_yy)`.
    fn hessian(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let softening_sq = self.softening * self.softening;
        let mut xx = -self.omega_sq;
        let mut xy = 0.0;
        let mut yy = -self.omega_sq;
        for body in self.masses() {
            let dx = x - body.x;
            let dy = y - body.y;
            let dist_sq = dx * dx + dy * dy + softening_sq;
            let inv_3 = G * body.mass / (dist_sq * dist_sq.sqrt());
            let inv_5 = 3.0 * inv_3 / dist_sq;
            xx += inv_3 - inv_5 * dx * dx;
            xy -= inv_5 * dx * dy;
            yy += inv_3 - inv_5 * dy * dy;
        }
        (xx, xy, yy)
    }

    /// Locate the L1 saddle between the pair to sub-cell accuracy: golden
    /// section maximum of `phi_eff` along the open inter-body segment,
    /// refined by 2D Newton steps on the analytic gradient. Returns the
    /// frame position and the potential value there.
    #[must_use]
    pub fn l1(&self) -> ((f64, f64), f64) {
        let a = self.bodies[0];
        let b = self.bodies[1];
        // Golden-section maximum along the segment (excluding the ends).
        let phi_at = |t: f64| {
            let x = a.x + (b.x - a.x) * t;
            let y = a.y + (b.y - a.y) * t;
            self.value(x, y)
        };
        let golden = 0.618_033_988_749_894_8;
        let (mut lo, mut hi) = (0.02, 0.98);
        let mut t1 = hi - golden * (hi - lo);
        let mut t2 = lo + golden * (hi - lo);
        let (mut f1, mut f2) = (phi_at(t1), phi_at(t2));
        for _ in 0..90 {
            if f1 > f2 {
                hi = t2;
                t2 = t1;
                f2 = f1;
                t1 = hi - golden * (hi - lo);
                f1 = phi_at(t1);
            } else {
                lo = t1;
                t1 = t2;
                f1 = f2;
                t2 = lo + golden * (hi - lo);
                f2 = phi_at(t2);
            }
        }
        let t = 0.5 * (lo + hi);
        let mut x = a.x + (b.x - a.x) * t;
        let mut y = a.y + (b.y - a.y) * t;

        // Newton refinement on grad(phi_eff) = 0 (guarded).
        let scale = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
        for _ in 0..16 {
            let (gx, gy) = self.gradient(x, y);
            let (xx, xy, yy) = self.hessian(x, y);
            let det = xx * yy - xy * xy;
            if det.abs() < 1e-30 {
                break;
            }
            let dx = (-gx * yy + gy * xy) / det;
            let dy = (-gy * xx + gx * xy) / det;
            if !dx.is_finite() || !dy.is_finite() || dx.hypot(dy) > 0.25 * scale {
                break;
            }
            x += dx;
            y += dy;
            if dx.hypot(dy) < 1e-12 * scale.max(1.0) {
                break;
            }
        }
        ((x, y), self.value(x, y))
    }

    /// Whether the given point is a saddle (Hessian determinant < 0).
    #[must_use]
    pub fn is_saddle(&self, x: f64, y: f64) -> bool {
        let (xx, xy, yy) = self.hessian(x, y);
        xx * yy - xy * xy < 0.0
    }

    /// Sample the effective potential on the pixel grid of `ctx` (frame
    /// coordinates flow through the context's world-to-pixel mapping).
    #[must_use]
    pub fn sample_grid(&self, ctx: &RenderContext) -> PotentialGrid {
        let width = ctx.width_usize;
        let height = ctx.height_usize;
        let bounds = *ctx.bounds();
        let step_x = bounds.width / width.max(1) as f64;
        let step_y = bounds.height / height.max(1) as f64;

        let mut values = vec![0.0f32; width * height];
        values.par_chunks_mut(width).enumerate().for_each(|(row, out)| {
            let y = bounds.min_y + (row as f64 + 0.5) * step_y;
            for (col, slot) in out.iter_mut().enumerate() {
                let x = bounds.min_x + (col as f64 + 0.5) * step_x;
                *slot = self.value(x, y) as f32;
            }
        });
        PotentialGrid::from_values(values, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::context::BoundingBox;

    fn square_ctx(size: u32, half_extent: f64) -> RenderContext {
        RenderContext::with_bounds(
            size,
            size,
            BoundingBox {
                min_x: -half_extent,
                max_x: half_extent,
                min_y: -half_extent,
                max_y: half_extent,
                width: half_extent * 2.0,
                height: half_extent * 2.0,
            },
        )
    }

    #[test]
    fn potential_grid_matches_analytic_single_mass() {
        let ctx = square_ctx(64, 8.0);
        let masses = [PointMass { x: 0.0, y: 0.0, mass: 200.0 }];
        let grid = PotentialGrid::sample(&masses, &ctx, 0.1);
        // Pixel (48, 32) center -> world x = -8 + 48.5 * 0.25, y = -8 + 32.5 * 0.25.
        let world = (-8.0 + 48.5 * 0.25, -8.0 + 32.5 * 0.25);
        let expected = -G * 200.0 / (world.0 * world.0 + world.1 * world.1 + 0.01f64).sqrt();
        let sampled = f64::from(grid.values[32 * 64 + 48]);
        assert!(
            (sampled - expected).abs() < 1e-3 * expected.abs(),
            "sampled {sampled} vs analytic {expected}"
        );
    }

    #[test]
    fn asinh_levels_are_monotonic() {
        let ctx = square_ctx(96, 6.0);
        let masses = [PointMass { x: 1.0, y: -0.5, mass: 150.0 }];
        let grid = PotentialGrid::sample(&masses, &ctx, 0.05);
        let levels = grid.asinh_levels(24);
        assert_eq!(levels.len(), 24);
        for pair in levels.windows(2) {
            assert!(pair[0] < pair[1], "levels must increase: {} !< {}", pair[0], pair[1]);
        }
    }

    #[test]
    fn equipotentials_of_a_single_mass_are_circles() {
        let ctx = square_ctx(160, 10.0);
        let masses = [PointMass { x: 0.0, y: 0.0, mass: 200.0 }];
        let grid = PotentialGrid::sample(&masses, &ctx, 0.0);
        let level = grid.value_at(40.0, 80.0) as f32; // a radius-40px circle
        let bundles = equipotentials(&grid, &[level]);
        assert_eq!(bundles.len(), 1);
        let points: Vec<(f32, f32)> =
            bundles[0].iter().flat_map(|line| line.iter().copied()).collect();
        assert!(points.len() > 60, "expected a dense circle, got {} points", points.len());
        let center = 80.0f64;
        let radii: Vec<f64> = points
            .iter()
            .map(|&(x, y)| {
                ((f64::from(x) - center).powi(2) + (f64::from(y) - center).powi(2)).sqrt()
            })
            .collect();
        let mean = radii.iter().sum::<f64>() / radii.len() as f64;
        let deviation =
            (radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64).sqrt();
        assert!(
            deviation / mean < 0.03,
            "equipotential should be circular: mean {mean:.2}, sigma {deviation:.3}"
        );
    }

    #[test]
    fn stitching_joins_segments_into_long_polylines() {
        let ctx = square_ctx(128, 10.0);
        let masses = [PointMass { x: 0.0, y: 0.0, mass: 200.0 }];
        let grid = PotentialGrid::sample(&masses, &ctx, 0.0);
        let level = grid.value_at(32.0, 64.0) as f32;
        let segments = marching_squares(&grid.values, grid.width, grid.height, level);
        let polylines = stitch_segments(&segments);
        assert!(!polylines.is_empty());
        let longest = polylines.iter().map(Vec::len).max().expect("non-empty");
        assert!(
            longest > segments.len() / 2,
            "stitching should form one dominant loop: longest {longest} of {} segments",
            segments.len()
        );
    }

    #[test]
    fn streamlines_of_a_single_mass_are_radial_and_separated() {
        let ctx = square_ctx(200, 10.0);
        let masses = [PointMass { x: 0.0, y: 0.0, mass: 200.0 }];
        let grid = PotentialGrid::sample(&masses, &ctx, 0.05);
        let spacing = 14.0;
        let lines = streamlines(&grid, spacing);
        assert!(lines.len() >= 8, "expected a radial fan, got {} lines", lines.len());

        // Radiality: tangents align with the radial direction away from the core.
        let center = 100.0f64;
        let mut worst = (0.0f64, (0.0f64, 0.0f64), 0usize);
        for (line_index, line) in lines.iter().enumerate() {
            for window in line.windows(2).step_by(5) {
                let (ax, ay) = (f64::from(window[0].0), f64::from(window[0].1));
                let (bx, by) = (f64::from(window[1].0), f64::from(window[1].1));
                let radial = (ax - center, ay - center);
                let radial_norm = (radial.0 * radial.0 + radial.1 * radial.1).sqrt();
                if radial_norm < spacing {
                    continue; // the discrete field is noisy at the singular core
                }
                let tangent = (bx - ax, by - ay);
                let tangent_norm = (tangent.0 * tangent.0 + tangent.1 * tangent.1).sqrt();
                let cross = (radial.0 * tangent.1 - radial.1 * tangent.0).abs()
                    / (radial_norm * tangent_norm).max(1e-12);
                if cross > worst.0 {
                    worst = (cross, (ax, ay), line_index);
                }
            }
        }
        assert!(
            worst.0 < 0.12,
            "streamline should be radial (|sin| {:.3} at ({:.1}, {:.1}) line {} of {})",
            worst.0,
            worst.1.0,
            worst.1.1,
            worst.2,
            lines.len()
        );

        // Separation: distinct lines stay apart by a good fraction of spacing.
        let mut min_gap = f64::INFINITY;
        for (index_a, line_a) in lines.iter().enumerate() {
            for line_b in lines.iter().skip(index_a + 1) {
                for &(ax, ay) in line_a.iter().step_by(4) {
                    let (ax, ay) = (f64::from(ax), f64::from(ay));
                    if ((ax - center).powi(2) + (ay - center).powi(2)).sqrt() < spacing * 1.5 {
                        continue; // the fan legitimately converges at the core
                    }
                    for &(bx, by) in line_b.iter().step_by(4) {
                        let (bx, by) = (f64::from(bx), f64::from(by));
                        if ((bx - center).powi(2) + (by - center).powi(2)).sqrt() < spacing * 1.5 {
                            continue;
                        }
                        let gap = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                        min_gap = min_gap.min(gap);
                    }
                }
            }
        }
        assert!(
            min_gap >= spacing * 0.35,
            "evenly-spaced lines should stay separated: min gap {min_gap:.2} for spacing {spacing}"
        );
    }

    #[test]
    fn roche_l1_sits_at_midpoint_for_equal_masses() {
        let roche = RochePotential::circular(200.0, 200.0, 10.0);
        let ((x, y), value) = roche.l1();
        assert!(x.abs() < 1e-9, "equal masses put L1 at the barycenter, got x = {x}");
        assert!(y.abs() < 1e-9, "L1 must lie on the pair axis, got y = {y}");
        assert!(value < 0.0);
        let (gx, gy) = roche.gradient(x, y);
        assert!(gx.hypot(gy) < 1e-9, "gradient at L1 must vanish, got |g| = {}", gx.hypot(gy));
        assert!(roche.is_saddle(x, y), "L1 must be a saddle point");
    }

    #[test]
    fn roche_l1_matches_hill_radius_for_small_secondary() {
        let (m1, m2, r) = (300.0, 0.3, 10.0);
        let roche = RochePotential::circular(m1, m2, r);
        let ((x, y), _) = roche.l1();
        let secondary_x = roche.bodies[1].x;
        let distance = (secondary_x - x).hypot(y);
        let hill = r * (m2 / (3.0 * m1)).cbrt();
        assert!(
            ((distance - hill) / hill).abs() < 0.03,
            "L1 distance {distance:.4} should match the Hill radius {hill:.4}"
        );
        let (gx, gy) = roche.gradient(x, y);
        assert!(gx.hypot(gy) < 1e-9 * G * m1, "gradient at refined L1 must vanish");
    }

    #[test]
    fn roche_grid_sampling_matches_analytic_value() {
        let roche = RochePotential::circular(250.0, 120.0, 8.0);
        let ctx = square_ctx(80, 12.0);
        let grid = roche.sample_grid(&ctx);
        // Pixel (20, 50): world x = -12 + 20.5 * 0.3, y = -12 + 50.5 * 0.3.
        let world = (-12.0 + 20.5 * 0.3, -12.0 + 50.5 * 0.3);
        let expected = roche.value(world.0, world.1);
        let sampled = f64::from(grid.values[50 * 80 + 20]);
        assert!(
            (sampled - expected).abs() < 1e-3 * expected.abs().max(1.0),
            "sampled {sampled} vs analytic {expected}"
        );
    }
}
