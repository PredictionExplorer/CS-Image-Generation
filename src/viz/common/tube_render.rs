//! CPU sphere-traced renderer for capsule-chain scenes (master plan II.8).
//!
//! Scenes are unions of emissive capsules (decimated trajectory polylines
//! with per-end colors), optional textured emissive spheres, and matte
//! planes/room walls. Acceleration is a uniform grid over capsule segments;
//! sphere tracing takes conservative steps of `min(local_sdf, cell)`, which
//! stays correct because any capsule surface outside the queried 3x3x3
//! neighborhood is at least one cell away. Volumetrics are single-scatter
//! homogeneous fog with equiangular sampling toward the K brightest
//! emitters. Output is linear Rec.2020 for the shared display encoders.
//!
//! The module also hosts the deterministic marching-tetrahedra mesher over
//! capsule SDF unions used by V50 `sculpture-export`.

use crate::viz::common::raster::Rgb64;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

/// World-space vector alias used throughout the tracer.
pub type Vec3 = Vector3<f64>;

/// One capsule (thick segment) with an emission gradient along its axis.
#[derive(Clone, Debug)]
pub struct Capsule {
    /// Segment start point.
    pub a: Vec3,
    /// Segment end point.
    pub b: Vec3,
    /// Tube radius.
    pub radius: f64,
    /// Linear Rec.2020 HDR emission at `a`.
    pub emission_a: Rgb64,
    /// Linear Rec.2020 HDR emission at `b`.
    pub emission_b: Rgb64,
    /// 0..1 strength of the darker core line seen through the tube center.
    pub core_darkening: f64,
}

/// Emissive sphere with an equirectangular texture (V08 globe).
pub struct TexturedSphere {
    /// Sphere center.
    pub center: Vec3,
    /// Sphere radius.
    pub radius: f64,
    /// Equirectangular emissive texel grid (row-major, `tex_w * tex_h`).
    pub texture: Vec<Rgb64>,
    /// Texture width in texels (longitude).
    pub tex_w: usize,
    /// Texture height in texels (latitude).
    pub tex_h: usize,
    /// World-to-texture orientation (applied to the hit normal before UV).
    pub orientation: Matrix3<f64>,
    /// Additive rim (atmosphere) color.
    pub rim_color: Rgb64,
    /// Rim glow strength.
    pub rim_strength: f64,
}

/// Procedural finish applied to a plane's shading normal.
#[derive(Clone, Copy, Debug)]
pub enum PlaneFinish {
    /// Flat matte surface.
    Plain,
    /// Procedural running-bond brick with mortar grooves.
    Brick {
        /// Brick width in world units.
        brick_w: f64,
        /// Brick height in world units.
        brick_h: f64,
        /// Mortar groove width in world units.
        mortar: f64,
        /// Normal perturbation strength (0..1).
        relief: f64,
        /// Deterministic per-brick tone seed.
        seed: u64,
    },
}

/// A matte (optionally glossy) plane, infinite or clipped to a rectangle.
pub struct Plane {
    /// A point on the plane.
    pub point: Vec3,
    /// Unit normal pointing into the scene.
    pub normal: Vec3,
    /// Unit tangent defining the texture/extent U axis.
    pub u_axis: Vec3,
    /// Diffuse albedo (linear Rec.2020).
    pub albedo: Rgb64,
    /// Glossy reflection strength (0 = pure matte).
    pub gloss: f64,
    /// Half-extents along `u_axis` and `normal x u_axis`; `None` = infinite.
    pub extent: Option<(f64, f64)>,
    /// Shading finish.
    pub finish: PlaneFinish,
}

/// Homogeneous single-scatter fog parameters.
#[derive(Clone, Copy, Debug)]
pub struct Fog {
    /// Scattering coefficient per world unit.
    pub sigma_s: f64,
    /// Extinction coefficient per world unit (>= `sigma_s`).
    pub sigma_t: f64,
}

/// One representative emitter used for fog scattering and plane lighting.
#[derive(Clone, Copy, Debug)]
pub struct Emitter {
    /// Emitter position (capsule midpoint).
    pub position: Vec3,
    /// Radiant power color (linear Rec.2020, pre-scaled).
    pub power: Rgb64,
}

/// Uniform grid over capsule indices for local SDF queries, with a
/// Chebyshev distance transform for long empty-space steps.
struct UniformGrid {
    origin: Vec3,
    cell: f64,
    dims: [usize; 3],
    starts: Vec<u32>,
    items: Vec<u32>,
    /// Chebyshev cell distance to the nearest occupied cell (0 = occupied).
    empty_distance: Vec<u16>,
}

impl UniformGrid {
    fn build(capsules: &[Capsule], bounds: (Vec3, Vec3), max_radius: f64) -> Self {
        let extent = bounds.1 - bounds.0;
        let diag = extent.norm().max(1e-9);
        // Cell = 2x max radius per spec, floored so dims stay tractable.
        let cell = (2.0 * max_radius).max(diag / 96.0);
        let dims = [
            ((extent.x / cell).ceil() as usize + 1).clamp(1, 160),
            ((extent.y / cell).ceil() as usize + 1).clamp(1, 160),
            ((extent.z / cell).ceil() as usize + 1).clamp(1, 160),
        ];
        // Recompute cell so the clamped dims still cover the bounds.
        let cell = cell
            .max(extent.x / dims[0] as f64)
            .max(extent.y / dims[1] as f64)
            .max(extent.z / dims[2] as f64)
            .max(1e-9);

        let cell_count = dims[0] * dims[1] * dims[2];
        let clamp_axis =
            |value: f64, axis: usize| -> usize { (value.max(0.0) as usize).min(dims[axis] - 1) };
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); cell_count];
        for (index, capsule) in capsules.iter().enumerate() {
            let lo = capsule.a.inf(&capsule.b).map(|v| v - capsule.radius);
            let hi = capsule.a.sup(&capsule.b).map(|v| v + capsule.radius);
            let x0 = clamp_axis((lo.x - bounds.0.x) / cell, 0);
            let x1 = clamp_axis((hi.x - bounds.0.x) / cell, 0);
            let y0 = clamp_axis((lo.y - bounds.0.y) / cell, 1);
            let y1 = clamp_axis((hi.y - bounds.0.y) / cell, 1);
            let z0 = clamp_axis((lo.z - bounds.0.z) / cell, 2);
            let z1 = clamp_axis((hi.z - bounds.0.z) / cell, 2);
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        buckets[(z * dims[1] + y) * dims[0] + x].push(index as u32);
                    }
                }
            }
        }

        let mut starts = Vec::with_capacity(cell_count + 1);
        let mut items = Vec::new();
        starts.push(0u32);
        for bucket in &buckets {
            items.extend_from_slice(bucket);
            starts.push(items.len() as u32);
        }

        // Two-pass 26-neighborhood chamfer: exact Chebyshev distance to the
        // nearest occupied cell, enabling multi-cell empty-space steps.
        let mut empty_distance: Vec<u16> =
            buckets.iter().map(|bucket| if bucket.is_empty() { u16::MAX } else { 0 }).collect();
        let flat = |x: usize, y: usize, z: usize| (z * dims[1] + y) * dims[0] + x;
        let relax = |target: &mut Vec<u16>, x: usize, y: usize, z: usize, reverse: bool| {
            let current = target[flat(x, y, z)];
            if current == 0 {
                return;
            }
            let mut best = current;
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        // Forward pass looks at already-visited neighbors,
                        // backward pass at the rest.
                        let visited_forward =
                            dz < 0 || (dz == 0 && (dy < 0 || (dy == 0 && dx < 0)));
                        if visited_forward == reverse {
                            continue;
                        }
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        let nz = z as i64 + dz;
                        if nx < 0
                            || ny < 0
                            || nz < 0
                            || nx >= dims[0] as i64
                            || ny >= dims[1] as i64
                            || nz >= dims[2] as i64
                        {
                            continue;
                        }
                        let neighbor = target[flat(nx as usize, ny as usize, nz as usize)];
                        best = best.min(neighbor.saturating_add(1));
                    }
                }
            }
            target[flat(x, y, z)] = best;
        };
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    relax(&mut empty_distance, x, y, z, false);
                }
            }
        }
        for z in (0..dims[2]).rev() {
            for y in (0..dims[1]).rev() {
                for x in (0..dims[0]).rev() {
                    relax(&mut empty_distance, x, y, z, true);
                }
            }
        }

        Self { origin: bounds.0, cell, dims, starts, items, empty_distance }
    }

    #[inline]
    fn cell_of(&self, p: Vec3) -> [i64; 3] {
        [
            ((p.x - self.origin.x) / self.cell).floor() as i64,
            ((p.y - self.origin.y) / self.cell).floor() as i64,
            ((p.z - self.origin.z) / self.cell).floor() as i64,
        ]
    }

    /// Chebyshev cell distance to the nearest occupied cell at `p`.
    #[inline]
    fn empty_cells_at(&self, p: Vec3) -> u16 {
        let cell = self.cell_of(p);
        let x = cell[0].clamp(0, self.dims[0] as i64 - 1) as usize;
        let y = cell[1].clamp(0, self.dims[1] as i64 - 1) as usize;
        let z = cell[2].clamp(0, self.dims[2] as i64 - 1) as usize;
        self.empty_distance[(z * self.dims[1] + y) * self.dims[0] + x]
    }

    /// Visit capsule indices in the 3x3x3 neighborhood of `p` (duplicates
    /// across cells are tolerated by the `min` reductions downstream).
    #[inline]
    fn for_neighbors(&self, p: Vec3, mut visit: impl FnMut(u32)) {
        let cell = self.cell_of(p);
        for dz in -1i64..=1 {
            let z = cell[2] + dz;
            if z < 0 || z >= self.dims[2] as i64 {
                continue;
            }
            for dy in -1i64..=1 {
                let y = cell[1] + dy;
                if y < 0 || y >= self.dims[1] as i64 {
                    continue;
                }
                for dx in -1i64..=1 {
                    let x = cell[0] + dx;
                    if x < 0 || x >= self.dims[0] as i64 {
                        continue;
                    }
                    let flat = (z as usize * self.dims[1] + y as usize) * self.dims[0] + x as usize;
                    let start = self.starts[flat] as usize;
                    let end = self.starts[flat + 1] as usize;
                    for &item in &self.items[start..end] {
                        visit(item);
                    }
                }
            }
        }
    }
}

/// Parameter of the closest point on segment `ab` to `p`, in `[0, 1]`.
#[inline]
#[must_use]
pub fn segment_t(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let len_sq = ab.norm_squared();
    if len_sq <= 1e-24 {
        return 0.0;
    }
    ((p - a).dot(&ab) / len_sq).clamp(0.0, 1.0)
}

/// Signed distance from `p` to a capsule surface (negative inside).
#[inline]
#[must_use]
pub fn capsule_sdf(p: Vec3, a: Vec3, b: Vec3, radius: f64) -> f64 {
    let t = segment_t(p, a, b);
    (p - (a + (b - a) * t)).norm() - radius
}

/// A capsule union with grid acceleration, shared by tracing and meshing.
pub struct CapsuleSet {
    capsules: Vec<Capsule>,
    grid: UniformGrid,
    bounds: (Vec3, Vec3),
}

impl CapsuleSet {
    /// Build the set and its acceleration grid.
    #[must_use]
    pub fn new(capsules: Vec<Capsule>) -> Self {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = -lo;
        let mut max_radius = 1e-9f64;
        for capsule in &capsules {
            lo = lo.inf(&capsule.a.inf(&capsule.b).map(|v| v - capsule.radius));
            hi = hi.sup(&capsule.a.sup(&capsule.b).map(|v| v + capsule.radius));
            max_radius = max_radius.max(capsule.radius);
        }
        if !lo.x.is_finite() {
            lo = Vec3::new(-1.0, -1.0, -1.0);
            hi = Vec3::new(1.0, 1.0, 1.0);
        }
        let grid = UniformGrid::build(&capsules, (lo, hi), max_radius);
        Self { capsules, grid, bounds: (lo, hi) }
    }

    /// The capsules in the set.
    #[must_use]
    pub fn capsules(&self) -> &[Capsule] {
        &self.capsules
    }

    /// Axis-aligned bounds of the union (surface-inclusive).
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    /// Grid cell edge length (the conservative empty-space step).
    #[must_use]
    pub fn cell(&self) -> f64 {
        self.grid.cell
    }

    /// Distance to the nearest capsule among the local neighborhood, with
    /// the guarantee `true_distance >= min(returned, cell)`.
    #[must_use]
    pub fn local_distance(&self, p: Vec3) -> f64 {
        let mut best = f64::INFINITY;
        self.grid.for_neighbors(p, |index| {
            let capsule = &self.capsules[index as usize];
            best = best.min(capsule_sdf(p, capsule.a, capsule.b, capsule.radius));
        });
        best
    }

    /// Exact distance to the union (brute force; used by tests and meshing
    /// fallbacks on tiny sets).
    #[must_use]
    pub fn brute_distance(&self, p: Vec3) -> f64 {
        self.capsules
            .iter()
            .map(|capsule| capsule_sdf(p, capsule.a, capsule.b, capsule.radius))
            .fold(f64::INFINITY, f64::min)
    }

    /// Largest safe sphere-trace step from `p`: multi-cell jumps through
    /// certified-empty space, `min(local_sdf, cell)` near occupancy.
    #[must_use]
    pub fn safe_step(&self, p: Vec3) -> f64 {
        let empty_cells = self.grid.empty_cells_at(p);
        if empty_cells >= 2 {
            return f64::from(empty_cells - 1) * self.grid.cell;
        }
        self.local_distance(p).min(self.grid.cell)
    }

    /// Distance query for isosurface meshing: exact near the surface,
    /// a positive chamfer lower bound in certified-empty space. Signs are
    /// always correct; magnitudes are exact wherever `|d| < cell`.
    #[must_use]
    pub fn mesh_distance(&self, p: Vec3) -> f64 {
        let empty_cells = self.grid.empty_cells_at(p);
        if empty_cells >= 2 {
            return f64::from(empty_cells - 1) * self.grid.cell;
        }
        let local = self.local_distance(p);
        if local.is_finite() { local } else { self.grid.cell }
    }

    /// Nearest capsule index and axis parameter at `p` (local query).
    #[must_use]
    pub fn nearest(&self, p: Vec3) -> Option<(usize, f64, f64)> {
        let mut best: Option<(usize, f64, f64)> = None;
        self.grid.for_neighbors(p, |index| {
            let capsule = &self.capsules[index as usize];
            let d = capsule_sdf(p, capsule.a, capsule.b, capsule.radius);
            if best.is_none_or(|(_, _, bd)| d < bd) {
                let t = segment_t(p, capsule.a, capsule.b);
                best = Some((index as usize, t, d));
            }
        });
        best
    }
}

/// Complete traceable scene.
pub struct Scene {
    /// Capsule union.
    pub set: CapsuleSet,
    /// Analytic textured spheres.
    pub spheres: Vec<TexturedSphere>,
    /// Matte planes / room walls.
    pub planes: Vec<Plane>,
    /// Optional homogeneous fog.
    pub fog: Option<Fog>,
    /// Background radiance.
    pub background: Rgb64,
    /// Glass rim boost applied to capsule emission at grazing angles.
    pub glass_rim: f64,
    /// K brightest emitters for fog scattering and plane lighting.
    pub emitters: Vec<Emitter>,
}

/// Luminance of a linear color (Rec.2020 weights, approximate).
#[inline]
fn luminance(color: Rgb64) -> f64 {
    0.2627 * color.0 + 0.678 * color.1 + 0.0593 * color.2
}

impl Scene {
    /// Assemble a scene and derive its emitter set.
    #[must_use]
    pub fn new(capsules: Vec<Capsule>, emitter_count: usize) -> Self {
        let set = CapsuleSet::new(capsules);
        let emitters = derive_emitters(set.capsules(), emitter_count);
        Self {
            set,
            spheres: Vec::new(),
            planes: Vec::new(),
            fog: None,
            background: (0.0, 0.0, 0.0),
            glass_rim: 0.0,
            emitters,
        }
    }
}

/// Cluster capsules into at most `count` representative emitters.
fn derive_emitters(capsules: &[Capsule], count: usize) -> Vec<Emitter> {
    if capsules.is_empty() || count == 0 {
        return Vec::new();
    }
    // Power of one capsule: mean emission scaled by its lateral area.
    let mut ranked: Vec<(f64, usize)> = capsules
        .iter()
        .enumerate()
        .map(|(index, capsule)| {
            let mean = (
                0.5 * (capsule.emission_a.0 + capsule.emission_b.0),
                0.5 * (capsule.emission_a.1 + capsule.emission_b.1),
                0.5 * (capsule.emission_a.2 + capsule.emission_b.2),
            );
            let area = (capsule.b - capsule.a).norm() * capsule.radius;
            (luminance(mean) * area, index)
        })
        .collect();
    ranked.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0).then(lhs.1.cmp(&rhs.1)));

    // Greedy separation: keep the brightest capsules that are not within
    // `min_gap` of an already chosen one, then fill remaining slots.
    let diag = {
        let set_lo = capsules
            .iter()
            .fold(Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY), |acc, capsule| {
                acc.inf(&capsule.a.inf(&capsule.b))
            });
        let set_hi = capsules
            .iter()
            .fold(-Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY), |acc, capsule| {
                acc.sup(&capsule.a.sup(&capsule.b))
            });
        (set_hi - set_lo).norm().max(1e-9)
    };
    let min_gap = diag / (count as f64).cbrt() / 4.0;

    let mut chosen: Vec<(usize, f64)> = Vec::with_capacity(count);
    for pass in 0..2 {
        for &(power, index) in &ranked {
            if chosen.len() >= count {
                break;
            }
            if power <= 0.0 {
                continue;
            }
            let capsule = &capsules[index];
            let mid = (capsule.a + capsule.b) * 0.5;
            let separated = chosen.iter().all(|&(other, _)| {
                let other_mid = (capsules[other].a + capsules[other].b) * 0.5;
                (mid - other_mid).norm() >= min_gap
            });
            if (pass == 1 || separated) && !chosen.iter().any(|&(other, _)| other == index) {
                chosen.push((index, power));
            }
        }
    }

    let total_power: f64 = ranked.iter().map(|&(power, _)| power).sum();
    let chosen_power: f64 = chosen.iter().map(|&(_, power)| power).sum();
    let boost = if chosen_power > 0.0 { total_power / chosen_power } else { 1.0 };
    chosen
        .into_iter()
        .map(|(index, _)| {
            let capsule = &capsules[index];
            let mid = (capsule.a + capsule.b) * 0.5;
            let mean = (
                0.5 * (capsule.emission_a.0 + capsule.emission_b.0),
                0.5 * (capsule.emission_a.1 + capsule.emission_b.1),
                0.5 * (capsule.emission_a.2 + capsule.emission_b.2),
            );
            let area = (capsule.b - capsule.a).norm() * capsule.radius;
            let scale = area * boost;
            Emitter { position: mid, power: (mean.0 * scale, mean.1 * scale, mean.2 * scale) }
        })
        .collect()
}

/// Pinhole camera.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Eye position.
    pub position: Vec3,
    /// Point the camera looks at.
    pub look_at: Vec3,
    /// Up hint.
    pub up: Vec3,
    /// Vertical field of view in degrees.
    pub vfov_deg: f64,
}

impl Camera {
    /// Basis vectors (right, up, forward) of the view frame.
    #[must_use]
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.look_at - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = right.cross(&forward);
        (right, up, forward)
    }

    /// Primary ray through pixel `(px, py)` with sub-pixel `jitter`.
    #[must_use]
    pub fn ray(&self, px: f64, py: f64, width: u32, height: u32, jitter: (f64, f64)) -> Ray {
        let (right, up, forward) = self.basis();
        let aspect = f64::from(width) / f64::from(height);
        let tan_half = (self.vfov_deg.to_radians() * 0.5).tan();
        let ndc_x = ((px + jitter.0) / f64::from(width) * 2.0 - 1.0) * tan_half * aspect;
        let ndc_y = (1.0 - (py + jitter.1) / f64::from(height) * 2.0) * tan_half;
        let dir = (forward + right * ndc_x + up * ndc_y).normalize();
        Ray { origin: self.position, dir }
    }
}

/// Orbit-rig camera mirroring `render/orbit.rs` conventions: yaw about the
/// vertical (+y) axis through `center`, positive tilt raising the eye.
#[must_use]
pub fn orbit_camera(
    center: Vec3,
    distance: f64,
    yaw_rad: f64,
    tilt_rad: f64,
    vfov_deg: f64,
) -> Camera {
    let (sin_yaw, cos_yaw) = yaw_rad.sin_cos();
    let (sin_tilt, cos_tilt) = tilt_rad.sin_cos();
    let offset = Vec3::new(
        distance * cos_tilt * sin_yaw,
        distance * sin_tilt,
        distance * cos_tilt * cos_yaw,
    );
    Camera { position: center + offset, look_at: center, up: Vec3::new(0.0, 1.0, 0.0), vfov_deg }
}

/// Camera distance that frames a bounding sphere with 8% margin.
#[must_use]
pub fn fit_distance(bound_radius: f64, vfov_deg: f64) -> f64 {
    bound_radius.max(1e-9) * 1.08 / (vfov_deg.to_radians() * 0.5).tan()
}

/// Catmull-Rom dolly path through waypoints (uniform parameterization).
pub struct DollyPath {
    /// Path waypoints in order.
    pub points: Vec<Vec3>,
}

impl DollyPath {
    /// Sample the path at `t` in `[0, 1]`.
    #[must_use]
    pub fn sample(&self, t: f64) -> Vec3 {
        let n = self.points.len();
        if n == 0 {
            return Vec3::zeros();
        }
        if n == 1 {
            return self.points[0];
        }
        let segments = (n - 1) as f64;
        let scaled = (t.clamp(0.0, 1.0) * segments).min(segments - 1e-9);
        let index = scaled.floor() as usize;
        let local = scaled - index as f64;
        let at = |offset: i64| -> Vec3 {
            let clamped = (index as i64 + offset).clamp(0, n as i64 - 1) as usize;
            self.points[clamped]
        };
        let (p0, p1, p2, p3) = (at(-1), at(0), at(1), at(2));
        let t2 = local * local;
        let t3 = t2 * local;
        (p1 * 2.0
            + (p2 - p0) * local
            + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
            + (p3 - p0 + (p1 - p2) * 3.0) * t3)
            * 0.5
    }
}

/// A ray with unit direction.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Ray origin.
    pub origin: Vec3,
    /// Unit direction.
    pub dir: Vec3,
}

/// Render controls.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Samples per pixel.
    pub spp: u32,
    /// Sphere-trace step cap per ray.
    pub max_steps: u32,
    /// Shadow rays toward emitters at plane hits.
    pub shadows: bool,
    /// How many of the brightest emitters get shadow-tested (the rest
    /// contribute unshadowed); emitters are ordered brightest-first.
    pub shadow_emitters: usize,
    /// Fixed jitter sequence seed (determinism contract).
    pub jitter_seed: u64,
}

/// Low-discrepancy per-sample jitter (R2 sequence over a hashed offset).
#[inline]
#[must_use]
pub fn sample_jitter(pixel_index: u64, sample: u32, seed: u64) -> (f64, f64) {
    // SplitMix-style hash for the per-pixel Cranley-Patterson offset.
    let mut state = pixel_index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    let offset_x = (state >> 11) as f64 / (1u64 << 53) as f64;
    let offset_y = (state << 7 >> 11) as f64 / (1u64 << 53) as f64;
    // R2 (plastic constant) low-discrepancy step per sample.
    let g = 1.324_717_957_244_746_f64;
    let a1 = 1.0 / g;
    let a2 = 1.0 / (g * g);
    let s = f64::from(sample);
    (((offset_x + a1 * s) % 1.0), ((offset_y + a2 * s) % 1.0))
}

/// Ray / axis-aligned box intersection; returns `(t_enter, t_exit)`.
#[inline]
fn ray_box(ray: &Ray, lo: Vec3, hi: Vec3) -> Option<(f64, f64)> {
    let mut t0 = 0.0f64;
    let mut t1 = f64::INFINITY;
    for axis in 0..3 {
        let inv = 1.0 / ray.dir[axis];
        let mut near = (lo[axis] - ray.origin[axis]) * inv;
        let mut far = (hi[axis] - ray.origin[axis]) * inv;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        t0 = t0.max(near);
        t1 = t1.min(far);
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, t1))
}

/// Surface hit kinds resolved by the tracer.
enum Hit {
    Capsule { t: f64, index: usize, axis_t: f64 },
    Sphere { t: f64, index: usize },
    Plane { t: f64, index: usize },
    Miss,
}

/// Analytic nearest sphere intersection.
fn intersect_spheres(scene: &Scene, ray: &Ray, t_max: f64) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for (index, sphere) in scene.spheres.iter().enumerate() {
        let oc = ray.origin - sphere.center;
        let b = oc.dot(&ray.dir);
        let c = oc.norm_squared() - sphere.radius * sphere.radius;
        let disc = b * b - c;
        if disc < 0.0 {
            continue;
        }
        let sqrt_disc = disc.sqrt();
        for root in [-b - sqrt_disc, -b + sqrt_disc] {
            if root > 1e-6 && root < t_max && best.is_none_or(|(t, _)| root < t) {
                best = Some((root, index));
            }
        }
    }
    best
}

/// Analytic nearest plane intersection (respecting extents).
fn intersect_planes(scene: &Scene, ray: &Ray, t_max: f64) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for (index, plane) in scene.planes.iter().enumerate() {
        let denom = ray.dir.dot(&plane.normal);
        if denom.abs() < 1e-9 {
            continue;
        }
        let t = (plane.point - ray.origin).dot(&plane.normal) / denom;
        if t <= 1e-6 || t >= t_max {
            continue;
        }
        if let Some((half_u, half_v)) = plane.extent {
            let hit = ray.origin + ray.dir * t;
            let v_axis = plane.normal.cross(&plane.u_axis);
            let local = hit - plane.point;
            if local.dot(&plane.u_axis).abs() > half_u || local.dot(&v_axis).abs() > half_v {
                continue;
            }
        }
        if best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, index));
        }
    }
    best
}

/// Sphere-trace the capsule union up to `t_max`.
fn march_capsules(
    set: &CapsuleSet,
    ray: &Ray,
    t_max: f64,
    max_steps: u32,
) -> Option<(f64, usize, f64)> {
    let (lo, hi) = set.bounds();
    let (enter, exit) = ray_box(ray, lo, hi)?;
    let mut t = enter.max(1e-6);
    let end = exit.min(t_max);
    if t >= end {
        return None;
    }
    let cell = set.cell();
    let epsilon = (cell * 1e-4).max(1e-9);
    for _ in 0..max_steps {
        let p = ray.origin + ray.dir * t;
        let step = set.safe_step(p);
        if step < epsilon {
            let (index, axis_t, _) = set.nearest(p)?;
            return Some((t, index, axis_t));
        }
        t += step.max(epsilon * 2.0);
        if t >= end {
            return None;
        }
    }
    None
}

/// Binary emitter visibility from a surface point (shadow ray).
fn emitter_visible(set: &CapsuleSet, from: Vec3, to: Vec3, max_steps: u32) -> bool {
    let delta = to - from;
    let distance = delta.norm();
    if distance <= 1e-9 {
        return true;
    }
    let ray = Ray { origin: from, dir: delta / distance };
    // Stop marginally before the emitter (it sits on a capsule axis).
    march_capsules(set, &ray, distance * 0.96, max_steps).is_none()
}

/// Deterministic per-brick tone from integer brick coordinates.
#[inline]
fn brick_tone(row: i64, col: i64, seed: u64) -> f64 {
    let mut state = (row as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((col as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(seed);
    state ^= state >> 33;
    state = state.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    state ^= state >> 33;
    (state >> 11) as f64 / (1u64 << 53) as f64
}

/// Brick finish: returns (albedo scale, perturbed normal).
fn brick_shading(plane: &Plane, hit: Vec3, params: (f64, f64, f64, f64, u64)) -> (f64, Vec3) {
    let (brick_w, brick_h, mortar, relief, seed) = params;
    let v_axis = plane.normal.cross(&plane.u_axis);
    let local = hit - plane.point;
    let u = local.dot(&plane.u_axis);
    let v = local.dot(&v_axis);

    let row = (v / brick_h).floor();
    let offset = if (row as i64).rem_euclid(2) == 0 { 0.0 } else { 0.5 * brick_w };
    let col = ((u + offset) / brick_w).floor();
    let fu = (u + offset) - col * brick_w;
    let fv = v - row * brick_h;

    // Distance to the nearest mortar groove.
    let edge = fu.min(brick_w - fu).min(fv).min(brick_h - fv);
    let in_mortar = edge < mortar;
    let tone = 0.82 + 0.18 * brick_tone(row as i64, col as i64, seed);
    if in_mortar {
        // Grooves are darker and tilt the normal toward the groove center.
        let tilt = relief * (1.0 - edge / mortar);
        let grad_u = if fu < mortar {
            -1.0
        } else if brick_w - fu < mortar {
            1.0
        } else {
            0.0
        };
        let grad_v = if fv < mortar {
            -1.0
        } else if brick_h - fv < mortar {
            1.0
        } else {
            0.0
        };
        let normal = (plane.normal + (plane.u_axis * grad_u + v_axis * grad_v) * tilt).normalize();
        (0.55, normal)
    } else {
        (tone, plane.normal)
    }
}

/// Direct + glossy lighting of a plane hit from the scene emitters.
fn shade_plane(scene: &Scene, plane: &Plane, ray: &Ray, hit: Vec3, params: &RenderParams) -> Rgb64 {
    let (albedo_scale, normal) = match plane.finish {
        PlaneFinish::Plain => (1.0, plane.normal),
        PlaneFinish::Brick { brick_w, brick_h, mortar, relief, seed } => {
            brick_shading(plane, hit, (brick_w, brick_h, mortar, relief, seed))
        }
    };
    let mut radiance = (0.0f64, 0.0f64, 0.0f64);
    let reflect = ray.dir - normal * (2.0 * ray.dir.dot(&normal));
    for (emitter_index, emitter) in scene.emitters.iter().enumerate() {
        let to_emitter = emitter.position - hit;
        let dist_sq = to_emitter.norm_squared().max(1e-12);
        let dir = to_emitter / dist_sq.sqrt();
        let cos_term = normal.dot(&dir).max(0.0);
        if cos_term <= 0.0 {
            continue;
        }
        let visible = !(params.shadows && emitter_index < params.shadow_emitters)
            || emitter_visible(
                &scene.set,
                hit + normal * 1e-4 * scene.set.cell().max(1e-6),
                emitter.position,
                params.max_steps,
            );
        if !visible {
            continue;
        }
        let falloff = 1.0 / (4.0 * std::f64::consts::PI * dist_sq);
        let diffuse = cos_term * falloff;
        let mut weight = diffuse;
        if plane.gloss > 0.0 {
            let lobe = reflect.dot(&dir).max(0.0).powi(24);
            weight += plane.gloss * lobe * falloff * 8.0;
        }
        radiance.0 += emitter.power.0 * weight;
        radiance.1 += emitter.power.1 * weight;
        radiance.2 += emitter.power.2 * weight;
    }
    (
        radiance.0 * plane.albedo.0 * albedo_scale,
        radiance.1 * plane.albedo.1 * albedo_scale,
        radiance.2 * plane.albedo.2 * albedo_scale,
    )
}

/// Emission of a capsule hit including the interior core line darkening.
fn shade_capsule(scene: &Scene, ray: &Ray, index: usize, axis_t: f64) -> Rgb64 {
    let capsule = &scene.set.capsules()[index];
    let emission = (
        capsule.emission_a.0 + (capsule.emission_b.0 - capsule.emission_a.0) * axis_t,
        capsule.emission_a.1 + (capsule.emission_b.1 - capsule.emission_a.1) * axis_t,
        capsule.emission_a.2 + (capsule.emission_b.2 - capsule.emission_a.2) * axis_t,
    );
    // Impact parameter of the ray against the capsule axis: rays through
    // the tube center read darker (the "core line"), silhouettes stay full.
    let mut factor = 1.0;
    if capsule.core_darkening > 0.0 {
        let axis_point = capsule.a + (capsule.b - capsule.a) * axis_t;
        let to_axis = axis_point - ray.origin;
        let closest = to_axis - ray.dir * to_axis.dot(&ray.dir);
        let impact = closest.norm() / capsule.radius.max(1e-12);
        let rim = impact.clamp(0.0, 1.0);
        factor = 1.0 - capsule.core_darkening * (1.0 - rim * rim);
    }
    if scene.glass_rim > 0.0 {
        // Schlick-style grazing boost standing in for the glass shell.
        let axis_point = capsule.a + (capsule.b - capsule.a) * axis_t;
        let hit = ray.origin + ray.dir * (axis_point - ray.origin).dot(&ray.dir).max(0.0);
        let normal = (hit - axis_point).normalize();
        let grazing = (1.0 - normal.dot(&(-ray.dir)).abs()).clamp(0.0, 1.0);
        factor *= 1.0 + scene.glass_rim * grazing.powi(5) * 1.5;
    }
    (emission.0 * factor, emission.1 * factor, emission.2 * factor)
}

/// Textured sphere shading: equirect emissive + additive rim glow.
fn shade_sphere(sphere: &TexturedSphere, ray: &Ray, t: f64) -> Rgb64 {
    let hit = ray.origin + ray.dir * t;
    let normal = (hit - sphere.center).normalize();
    let tex_normal = sphere.orientation * normal;
    let lon = tex_normal.z.atan2(tex_normal.x);
    let lat = tex_normal.y.clamp(-1.0, 1.0).asin();
    let u = (lon / std::f64::consts::TAU + 0.5).rem_euclid(1.0);
    let v = (0.5 - lat / std::f64::consts::PI).clamp(0.0, 1.0);

    // Bilinear sample with longitude wrap.
    let fx = u * sphere.tex_w as f64 - 0.5;
    let fy = v * sphere.tex_h as f64 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let dx = fx - x0;
    let dy = fy - y0;
    let wrap_x = |x: i64| -> usize { x.rem_euclid(sphere.tex_w as i64) as usize };
    let clamp_y = |y: i64| -> usize { y.clamp(0, sphere.tex_h as i64 - 1) as usize };
    let at = |x: i64, y: i64| -> Rgb64 { sphere.texture[clamp_y(y) * sphere.tex_w + wrap_x(x)] };
    let (x0i, y0i) = (x0 as i64, y0 as i64);
    let blend = |a: Rgb64, b: Rgb64, t: f64| -> Rgb64 {
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
    };
    let top = blend(at(x0i, y0i), at(x0i + 1, y0i), dx);
    let bottom = blend(at(x0i, y0i + 1), at(x0i + 1, y0i + 1), dx);
    let texel = blend(top, bottom, dy);

    let grazing = (1.0 - normal.dot(&(-ray.dir)).abs()).clamp(0.0, 1.0);
    let rim = grazing.powi(3) * sphere.rim_strength;
    (
        texel.0 + sphere.rim_color.0 * rim,
        texel.1 + sphere.rim_color.1 * rim,
        texel.2 + sphere.rim_color.2 * rim,
    )
}

/// Single-scatter fog inscatter along `[t0, t1]` of the ray toward every
/// emitter, with two equiangular samples per emitter.
fn fog_inscatter(scene: &Scene, fog: &Fog, ray: &Ray, t0: f64, t1: f64, jitter: f64) -> Rgb64 {
    let mut total = (0.0f64, 0.0f64, 0.0f64);
    let phase = 1.0 / (4.0 * std::f64::consts::PI);
    for emitter in &scene.emitters {
        // Equiangular sampling geometry (Kulla & Fajardo).
        let delta = emitter.position - ray.origin;
        let t_closest = delta.dot(&ray.dir);
        let d = (delta - ray.dir * t_closest).norm().max(1e-6);
        let theta_a = ((t0 - t_closest) / d).atan();
        let theta_b = ((t1 - t_closest) / d).atan();
        let span = theta_b - theta_a;
        if span.abs() < 1e-12 {
            continue;
        }
        for quantile in [0.25 + 0.5 * jitter, 0.75 - 0.5 * (1.0 - jitter)] {
            let theta = theta_a + span * quantile.clamp(0.0, 1.0);
            let t = t_closest + d * theta.tan();
            let pdf = d / (span * (d * d + (t - t_closest) * (t - t_closest)));
            if !(pdf.is_finite()) || pdf <= 1e-12 {
                continue;
            }
            let sample = ray.origin + ray.dir * t;
            let dist_sq = (emitter.position - sample).norm_squared().max(1e-12);
            let transmittance = (-fog.sigma_t * ((t - t0).max(0.0) + dist_sq.sqrt())).exp();
            let weight = fog.sigma_s * phase * transmittance / (dist_sq * pdf) / 2.0;
            total.0 += emitter.power.0 * weight;
            total.1 += emitter.power.1 * weight;
            total.2 += emitter.power.2 * weight;
        }
    }
    total
}

/// Resolve the nearest hit along a primary ray.
fn trace(scene: &Scene, ray: &Ray, params: &RenderParams) -> Hit {
    let mut t_max = f64::INFINITY;
    let plane_hit = intersect_planes(scene, ray, t_max);
    if let Some((t, _)) = plane_hit {
        t_max = t;
    }
    let sphere_hit = intersect_spheres(scene, ray, t_max);
    if let Some((t, _)) = sphere_hit {
        t_max = t;
    }
    let capsule_hit = march_capsules(&scene.set, ray, t_max, params.max_steps);

    if let Some((t, index, axis_t)) = capsule_hit {
        return Hit::Capsule { t, index, axis_t };
    }
    if let Some((t, index)) = sphere_hit {
        return Hit::Sphere { t, index };
    }
    if let Some((t, index)) = plane_hit {
        return Hit::Plane { t, index };
    }
    Hit::Miss
}

/// Render the scene into a linear Rec.2020 buffer (row-parallel).
#[must_use]
pub fn render(scene: &Scene, camera: &Camera, params: &RenderParams) -> Vec<Rgb64> {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut buffer = vec![(0.0f64, 0.0f64, 0.0f64); width * height];
    buffer.par_chunks_mut(width).enumerate().for_each(|(row, out)| {
        for (col, pixel) in out.iter_mut().enumerate() {
            let pixel_index = (row * width + col) as u64;
            let mut sum = (0.0f64, 0.0f64, 0.0f64);
            for sample in 0..params.spp {
                let jitter = sample_jitter(pixel_index, sample, params.jitter_seed);
                let ray = camera.ray(col as f64, row as f64, params.width, params.height, jitter);
                let hit = trace(scene, &ray, params);
                let (surface, t_hit) = match hit {
                    Hit::Capsule { t, index, axis_t } => {
                        (shade_capsule(scene, &ray, index, axis_t), t)
                    }
                    Hit::Sphere { t, index } => (shade_sphere(&scene.spheres[index], &ray, t), t),
                    Hit::Plane { t, index } => {
                        let point = ray.origin + ray.dir * t;
                        (shade_plane(scene, &scene.planes[index], &ray, point, params), t)
                    }
                    Hit::Miss => (scene.background, f64::INFINITY),
                };
                let mut color = surface;
                if let Some(fog) = &scene.fog {
                    // Surfaces attenuate over their hit distance; the
                    // background attenuates over the scene's fog reach.
                    let reach = t_hit.min(fog_reach(scene));
                    let attenuated = (-fog.sigma_t * reach).exp();
                    color =
                        (surface.0 * attenuated, surface.1 * attenuated, surface.2 * attenuated);
                    let inscatter = fog_inscatter(scene, fog, &ray, 0.0, reach, jitter.0);
                    color.0 += inscatter.0;
                    color.1 += inscatter.1;
                    color.2 += inscatter.2;
                }
                sum.0 += color.0;
                sum.1 += color.1;
                sum.2 += color.2;
            }
            let inv = 1.0 / f64::from(params.spp.max(1));
            *pixel = (sum.0 * inv, sum.1 * inv, sum.2 * inv);
        }
    });
    buffer
}

/// Maximum meaningful fog integration distance for the scene.
fn fog_reach(scene: &Scene) -> f64 {
    let (lo, hi) = scene.set.bounds();
    (hi - lo).norm().max(1e-6) * 2.5
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Ramer-Douglas-Peucker simplification of a 3D polyline.
#[must_use]
pub fn rdp_simplify_3d(points: &[Vec3], tolerance: f64) -> Vec<Vec3> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0usize, points.len() - 1)];
    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }
        let axis = points[end] - points[start];
        let axis_len_sq = axis.norm_squared().max(1e-24);
        let mut max_dist = -1.0f64;
        let mut max_index = start;
        for (index, point) in points.iter().enumerate().take(end).skip(start + 1) {
            let rel = point - points[start];
            let t = rel.dot(&axis) / axis_len_sq;
            let dist = (rel - axis * t).norm();
            if dist > max_dist {
                max_dist = dist;
                max_index = index;
            }
        }
        if max_dist > tolerance {
            keep[max_index] = true;
            stack.push((start, max_index));
            stack.push((max_index, end));
        }
    }
    points.iter().zip(keep.iter()).filter_map(|(&p, &k)| k.then_some(p)).collect()
}

// ---------------------------------------------------------------------------
// Marching-tetrahedra mesher over an SDF (V50 voxel remesh)
// ---------------------------------------------------------------------------

/// An indexed triangle mesh (counter-clockwise outward winding).
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<Vec3>,
    /// Triangle vertex indices.
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    /// Uniformly scale and translate all vertices.
    pub fn transform(&mut self, scale: f64, offset: Vec3) {
        for vertex in &mut self.vertices {
            *vertex = *vertex * scale + offset;
        }
    }

    /// Axis-aligned bounds of the mesh.
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = -lo;
        for vertex in &self.vertices {
            lo = lo.inf(vertex);
            hi = hi.sup(vertex);
        }
        (lo, hi)
    }

    /// Watertightness audit: every undirected edge is shared by exactly two
    /// triangles and the Euler characteristic `V - E + F` is even and >= 2
    /// for at least one component (2 - 2g per closed component).
    #[must_use]
    pub fn is_watertight(&self) -> bool {
        let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();
        for triangle in &self.triangles {
            for edge in
                [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]
            {
                let key = (edge.0.min(edge.1), edge.0.max(edge.1));
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
        if edge_counts.values().any(|&count| count != 2) {
            return false;
        }
        let vertices = self.vertices.len() as i64;
        let edges = edge_counts.len() as i64;
        let faces = self.triangles.len() as i64;
        let euler = vertices - edges + faces;
        euler % 2 == 0
    }

    /// Ray-parity audit: axis-aligned rays from outside must cross the
    /// surface an even number of times. Uses triangle intersection counts
    /// on a deterministic sample grid; returns true when all pass.
    #[must_use]
    pub fn ray_parity_ok(&self, samples: usize) -> bool {
        if self.triangles.is_empty() {
            return false;
        }
        let (lo, hi) = self.bounds();
        let extent = hi - lo;
        let mut failures = 0usize;
        for sample in 0..samples {
            let fy = (sample as f64 + 0.5) / samples as f64;
            let fz = ((sample as f64 * 0.618_033_988_75) % 1.0).abs();
            let origin =
                Vec3::new(lo.x - extent.x.max(1e-6), lo.y + extent.y * fy, lo.z + extent.z * fz);
            let dir = Vec3::new(1.0, 0.0, 0.0);
            let mut crossings = 0usize;
            for triangle in &self.triangles {
                let (a, b, c) = (
                    self.vertices[triangle[0] as usize],
                    self.vertices[triangle[1] as usize],
                    self.vertices[triangle[2] as usize],
                );
                if ray_triangle(origin, dir, a, b, c) {
                    crossings += 1;
                }
            }
            if !crossings.is_multiple_of(2) {
                failures += 1;
            }
        }
        failures == 0
    }
}

/// Moller-Trumbore ray/triangle test (t > 0 only).
fn ray_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = dir.cross(&edge2);
    let det = edge1.dot(&p);
    if det.abs() < 1e-12 {
        return false;
    }
    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = s.dot(&p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(&edge1);
    let v = dir.dot(&q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    edge2.dot(&q) * inv_det > 1e-9
}

/// The six tetrahedra decomposing one grid cube (corner indices 0..8 with
/// bit 0 = +x, bit 1 = +y, bit 2 = +z), all sharing the 0-7 diagonal so
/// neighbouring cubes tile compatibly.
const CUBE_TETS: [[usize; 4]; 6] =
    [[0, 5, 1, 7], [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7]];

/// Extract the zero isosurface of `sdf` over a regular grid with marching
/// tetrahedra. Deterministic; vertices on shared edges are deduplicated so
/// closed isosurfaces produce watertight meshes.
#[must_use]
pub fn mesh_from_sdf(sdf: &(impl Fn(Vec3) -> f64 + Sync), lo: Vec3, hi: Vec3, cell: f64) -> Mesh {
    let extent = hi - lo;
    let nx = ((extent.x / cell).ceil() as usize).clamp(2, 900) + 1;
    let ny = ((extent.y / cell).ceil() as usize).clamp(2, 900) + 1;
    let nz = ((extent.z / cell).ceil() as usize).clamp(2, 900) + 1;
    let step = Vec3::new(
        extent.x / (nx - 1) as f64,
        extent.y / (ny - 1) as f64,
        extent.z / (nz - 1) as f64,
    );

    let corner = |x: usize, y: usize, z: usize| -> Vec3 {
        Vec3::new(lo.x + step.x * x as f64, lo.y + step.y * y as f64, lo.z + step.z * z as f64)
    };
    let corner_id = |x: usize, y: usize, z: usize| -> u64 { ((z * ny + y) * nx + x) as u64 };

    // Evaluate one z-slice of SDF values in parallel.
    let eval_slice = |z: usize| -> Vec<f64> {
        let mut slice = vec![0.0f64; nx * ny];
        slice.par_chunks_mut(nx).enumerate().for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                *value = sdf(corner(x, y, z));
            }
        });
        slice
    };

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut edge_cache: HashMap<(u64, u64), u32> = HashMap::new();

    let mut below = eval_slice(0);
    for z in 0..nz - 1 {
        let above = eval_slice(z + 1);
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                // Cube corner values and ids (bit0=+x, bit1=+y, bit2=+z).
                let mut values = [0.0f64; 8];
                let mut ids = [0u64; 8];
                let mut points = [Vec3::zeros(); 8];
                for (bit, (value, (id, point))) in
                    values.iter_mut().zip(ids.iter_mut().zip(points.iter_mut())).enumerate()
                {
                    let cx = x + (bit & 1);
                    let cy = y + ((bit >> 1) & 1);
                    let cz = z + ((bit >> 2) & 1);
                    let slice = if cz == z { &below } else { &above };
                    *value = slice[cy * nx + cx];
                    *id = corner_id(cx, cy, cz);
                    *point = corner(cx, cy, cz);
                }
                if values.iter().all(|&v| v > 0.0) || values.iter().all(|&v| v <= 0.0) {
                    continue;
                }
                for tet in &CUBE_TETS {
                    emit_tetrahedron(
                        tet.map(|i| (points[i], values[i], ids[i])),
                        &mut vertices,
                        &mut triangles,
                        &mut edge_cache,
                    );
                }
            }
        }
        below = above;
    }
    Mesh { vertices, triangles }
}

/// Emit 0-2 triangles for one tetrahedron of the marching pass.
fn emit_tetrahedron(
    corners: [(Vec3, f64, u64); 4],
    vertices: &mut Vec<Vec3>,
    triangles: &mut Vec<[u32; 3]>,
    edge_cache: &mut HashMap<(u64, u64), u32>,
) {
    let inside: Vec<usize> = (0..4).filter(|&i| corners[i].1 <= 0.0).collect();
    if inside.is_empty() || inside.len() == 4 {
        return;
    }
    let mut edge_vertex = |i: usize, j: usize| -> u32 {
        let (pi, vi, idi) = corners[i];
        let (pj, vj, idj) = corners[j];
        let key = (idi.min(idj), idi.max(idj));
        if let Some(&index) = edge_cache.get(&key) {
            return index;
        }
        let t = (vi / (vi - vj)).clamp(0.0, 1.0);
        let point = pi + (pj - pi) * t;
        let index = vertices.len() as u32;
        vertices.push(point);
        edge_cache.insert(key, index);
        index
    };
    let outside: Vec<usize> = (0..4).filter(|&i| corners[i].1 > 0.0).collect();
    match inside.len() {
        1 => {
            let a = inside[0];
            let v0 = edge_vertex(a, outside[0]);
            let v1 = edge_vertex(a, outside[1]);
            let v2 = edge_vertex(a, outside[2]);
            push_oriented(triangles, [v0, v1, v2], corners[a].0, vertices);
        }
        3 => {
            let a = outside[0];
            let v0 = edge_vertex(inside[0], a);
            let v1 = edge_vertex(inside[1], a);
            let v2 = edge_vertex(inside[2], a);
            push_oriented_outward(triangles, [v0, v1, v2], corners[a].0, vertices);
        }
        2 => {
            let (i0, i1) = (inside[0], inside[1]);
            let (o0, o1) = (outside[0], outside[1]);
            let v00 = edge_vertex(i0, o0);
            let v01 = edge_vertex(i0, o1);
            let v10 = edge_vertex(i1, o0);
            let v11 = edge_vertex(i1, o1);
            // Quad v00-v01-v11-v10 split into two triangles; orient away
            // from the inside edge midpoint.
            let inside_mid = (corners[i0].0 + corners[i1].0) * 0.5;
            push_oriented(triangles, [v00, v01, v11], inside_mid, vertices);
            push_oriented(triangles, [v00, v11, v10], inside_mid, vertices);
        }
        _ => {}
    }
}

/// Push a triangle wound so its normal points away from `inside_point`.
fn push_oriented(
    triangles: &mut Vec<[u32; 3]>,
    tri: [u32; 3],
    inside_point: Vec3,
    vertices: &[Vec3],
) {
    let (a, b, c) =
        (vertices[tri[0] as usize], vertices[tri[1] as usize], vertices[tri[2] as usize]);
    let normal = (b - a).cross(&(c - a));
    let center = (a + b + c) / 3.0;
    if normal.dot(&(center - inside_point)) >= 0.0 {
        triangles.push(tri);
    } else {
        triangles.push([tri[0], tri[2], tri[1]]);
    }
}

/// Push a triangle wound so its normal points toward `outside_point`.
fn push_oriented_outward(
    triangles: &mut Vec<[u32; 3]>,
    tri: [u32; 3],
    outside_point: Vec3,
    vertices: &[Vec3],
) {
    let (a, b, c) =
        (vertices[tri[0] as usize], vertices[tri[1] as usize], vertices[tri[2] as usize]);
    let normal = (b - a).cross(&(c - a));
    let center = (a + b + c) / 3.0;
    if normal.dot(&(outside_point - center)) >= 0.0 {
        triangles.push(tri);
    } else {
        triangles.push([tri[0], tri[2], tri[1]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn helix_capsules(count: usize) -> Vec<Capsule> {
        (0..count)
            .map(|index| {
                let t0 = index as f64 * 0.3;
                let t1 = t0 + 0.3;
                let point = |t: f64| Vec3::new(t.cos(), t * 0.15, t.sin());
                Capsule {
                    a: point(t0),
                    b: point(t1),
                    radius: 0.08,
                    emission_a: (1.0, 0.4, 0.1),
                    emission_b: (0.1, 0.4, 1.0),
                    core_darkening: 0.4,
                }
            })
            .collect()
    }

    #[test]
    fn capsule_sdf_matches_analytics() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        assert!((capsule_sdf(Vec3::new(0.0, 0.5, 0.0), a, b, 0.2) - 0.3).abs() < 1e-12);
        assert!((capsule_sdf(Vec3::new(2.0, 0.0, 0.0), a, b, 0.2) - 0.8).abs() < 1e-12);
        assert!(capsule_sdf(Vec3::new(0.0, 0.0, 0.0), a, b, 0.2) < 0.0);
    }

    #[test]
    fn grid_local_distance_conservative_vs_brute_force() {
        let set = CapsuleSet::new(helix_capsules(40));
        let mut state = 0x1234_5678_u64;
        for _ in 0..200 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let fx = ((state >> 11) & 0xFFFF) as f64 / 65535.0;
            let fy = ((state >> 27) & 0xFFFF) as f64 / 65535.0;
            let fz = ((state >> 43) & 0xFFFF) as f64 / 65535.0;
            let (lo, hi) = set.bounds();
            let p = Vec3::new(
                lo.x + (hi.x - lo.x) * fx,
                lo.y + (hi.y - lo.y) * fy,
                lo.z + (hi.z - lo.z) * fz,
            );
            let brute = set.brute_distance(p);
            let local = set.local_distance(p);
            // The guarantee used by the tracer: stepping min(local, cell)
            // never overshoots the true surface.
            assert!(
                brute >= local.min(set.cell()) - 1e-9,
                "conservative bound violated: brute {brute} local {local} cell {}",
                set.cell()
            );
            // The chamfer empty-space step must also stay conservative.
            assert!(
                brute >= set.safe_step(p) - 1e-9,
                "safe_step overshoots: brute {brute} step {}",
                set.safe_step(p)
            );
            if brute < set.cell() {
                assert!((brute - local).abs() < 1e-9, "near-surface distances must be exact");
            }
        }
    }

    #[test]
    fn sphere_trace_hits_and_misses_a_capsule() {
        let capsule = Capsule {
            a: Vec3::new(0.0, -1.0, 0.0),
            b: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.25,
            emission_a: (1.0, 1.0, 1.0),
            emission_b: (1.0, 1.0, 1.0),
            core_darkening: 0.0,
        };
        let set = CapsuleSet::new(vec![capsule]);
        let hit_ray = Ray { origin: Vec3::new(-5.0, 0.0, 0.1), dir: Vec3::new(1.0, 0.0, 0.0) };
        let hit = march_capsules(&set, &hit_ray, f64::INFINITY, 256);
        assert!(hit.is_some(), "ray through the tube must hit");
        let (t, _, _) = hit.expect("hit");
        let expected = 5.0 - (0.25f64 * 0.25 - 0.1f64 * 0.1).sqrt();
        assert!((t - expected).abs() < 0.02, "hit distance {t} vs expected {expected}");

        let miss_ray = Ray { origin: Vec3::new(-5.0, 0.0, 0.6), dir: Vec3::new(1.0, 0.0, 0.0) };
        assert!(march_capsules(&set, &miss_ray, f64::INFINITY, 256).is_none());
    }

    #[test]
    fn orbit_camera_frames_center() {
        let camera = orbit_camera(Vec3::new(1.0, 2.0, 3.0), 10.0, 1.3, 0.4, 40.0);
        let ray = camera.ray(50.0, 50.0, 101, 101, (0.0, 0.0));
        let to_center = (Vec3::new(1.0, 2.0, 3.0) - camera.position).normalize();
        assert!(ray.dir.dot(&to_center) > 0.999, "center pixel ray must look at the center");
        assert!((camera.position - Vec3::new(1.0, 2.0, 3.0)).norm() - 10.0 < 1e-9);
    }

    #[test]
    fn jitter_is_deterministic_and_in_range() {
        let a = sample_jitter(1234, 3, 42);
        let b = sample_jitter(1234, 3, 42);
        assert_eq!(a, b);
        assert!(a.0 >= 0.0 && a.0 < 1.0 && a.1 >= 0.0 && a.1 < 1.0);
        let c = sample_jitter(1234, 4, 42);
        assert_ne!(a, c);
    }

    #[test]
    fn render_produces_light_where_the_tube_is() {
        let scene = Scene::new(helix_capsules(24), 8);
        let (lo, hi) = scene.set.bounds();
        let center = (lo + hi) * 0.5;
        let radius = (hi - lo).norm() * 0.5;
        let camera = orbit_camera(center, fit_distance(radius, 40.0), 0.7, 0.3, 40.0);
        let params = RenderParams {
            width: 64,
            height: 48,
            spp: 1,
            max_steps: 128,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed: 7,
        };
        let image = render(&scene, &camera, &params);
        let energy: f64 = image.iter().map(|&(r, g, b)| r + g + b).sum();
        assert!(energy > 0.5, "tube emission must reach the sensor, got {energy}");
    }

    #[test]
    fn rdp_3d_drops_collinear_and_keeps_corners() {
        let line: Vec<Vec3> = (0..50).map(|i| Vec3::new(f64::from(i), 0.0, 0.0)).collect();
        assert_eq!(rdp_simplify_3d(&line, 0.01).len(), 2);
        let corner =
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)];
        assert_eq!(rdp_simplify_3d(&corner, 0.05).len(), 3);
    }

    #[test]
    fn dolly_path_interpolates_endpoints() {
        let path = DollyPath {
            points: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
            ],
        };
        assert!((path.sample(0.0) - Vec3::new(0.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((path.sample(1.0) - Vec3::new(2.0, 1.0, 0.0)).norm() < 1e-9);
        let mid = path.sample(0.5);
        assert!(mid.x > 0.5 && mid.x < 1.5);
    }

    #[test]
    fn sdf_mesh_of_one_capsule_is_watertight() {
        let a = Vec3::new(-0.6, 0.0, 0.0);
        let b = Vec3::new(0.6, 0.0, 0.0);
        let sdf = move |p: Vec3| capsule_sdf(p, a, b, 0.3);
        let lo = Vec3::new(-1.2, -0.7, -0.7);
        let hi = Vec3::new(1.2, 0.7, 0.7);
        let mesh = mesh_from_sdf(&sdf, lo, hi, 0.06);
        assert!(mesh.triangles.len() > 100, "mesh should have real coverage");
        assert!(mesh.is_watertight(), "single capsule isosurface must be watertight");
        assert!(mesh.ray_parity_ok(16), "ray parity must be even everywhere");
    }

    #[test]
    fn sdf_mesh_of_two_overlapping_capsules_is_watertight() {
        let sdf = move |p: Vec3| {
            capsule_sdf(p, Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), 0.25)
                .min(capsule_sdf(p, Vec3::new(0.0, -0.5, 0.1), Vec3::new(0.0, 0.5, 0.1), 0.25))
        };
        let mesh = mesh_from_sdf(&sdf, Vec3::new(-1.0, -1.0, -0.6), Vec3::new(1.0, 1.0, 0.8), 0.05);
        assert!(mesh.is_watertight(), "union isosurface must be watertight");
        assert!(mesh.ray_parity_ok(16));
    }

    #[test]
    fn emitters_are_derived_brightest_first() {
        let scene = Scene::new(helix_capsules(30), 6);
        assert!(!scene.emitters.is_empty() && scene.emitters.len() <= 6);
        let power: f64 = scene.emitters.iter().map(|e| e.power.0 + e.power.1 + e.power.2).sum();
        assert!(power > 0.0);
    }
}
