//! Deterministic covered-tile rasterization of genuinely translucent art layers.
//!
//! Triangles own shared edges exactly once. Fine fibers use analytic footprint
//! filtering, including when narrower than a pixel. Every coverage sample has a
//! depth-sorted optical stack; farther surfaces remain visible through nearer
//! ones. Analytic studio lighting avoids stochastic light-sampling noise.
mod shading;

use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use shading::{Optical, Studio};
use smallvec::SmallVec;

use super::{Camera, Material, RenderConfig, Scene, SilkResult, V3};

const TILE: usize = 16;
const NONE: usize = usize::MAX;
const FIBER_SIGMA: f64 = 0.5;

struct CameraFrame {
    forward: V3,
    right: V3,
    up: V3,
    position: V3,
    target: V3,
    scale: f64,
    width: f64,
    height: f64,
}

impl CameraFrame {
    fn project(&self, point: V3) -> Point {
        let relative = point - self.target;
        Point {
            x: self.width * 0.5 + relative.dot(self.right) * self.scale,
            y: self.height * 0.5 - relative.dot(self.up) * self.scale,
            depth: (point - self.position).dot(self.forward),
        }
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    depth: f64,
}

#[derive(Clone, Copy)]
struct ProjectedVertex {
    point: Point,
    world: V3,
    normal: V3,
    tangent: V3,
    uv: [f64; 2],
}

#[derive(Clone, Copy)]
struct Bounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Bounds {
    fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64, config: &RenderConfig) -> Self {
        Self {
            x0: min_x.floor().clamp(0.0, f64::from(config.width)) as usize,
            y0: min_y.floor().clamp(0.0, f64::from(config.height)) as usize,
            x1: max_x.ceil().clamp(0.0, f64::from(config.width)) as usize,
            y1: max_y.ceil().clamp(0.0, f64::from(config.height)) as usize,
        }
    }
    fn empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }
    fn intersection(self, other: Self) -> Self {
        Self {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }
}

struct SurfaceTriangle {
    indices: [usize; 3],
    material: usize,
    bounds: Bounds,
    inverse_area: f64,
    inclusive: [bool; 3],
    fallback_normal: V3,
    transverse_variance: f64,
}

struct FiberSegment {
    first: Point,
    last: Point,
    world_first: V3,
    world_last: V3,
    radius_first: f64,
    radius_last: f64,
    tangent: V3,
    material: usize,
    group: usize,
    arc_start: f64,
    arc_length: f64,
    bounds: Bounds,
}

struct Projected {
    vertices: Vec<ProjectedVertex>,
    triangles: Vec<SurfaceTriangle>,
    fibers: Vec<FiberSegment>,
    bins: Vec<Vec<usize>>,
    tiles_x: usize,
}

#[derive(Clone, Copy)]
struct Fragment {
    depth: f64,
    optical: Optical,
    coverage: f64,
    group: usize,
    arc: f64,
    merge_range: f64,
    order: usize,
    next: usize,
}

struct TileBuffer {
    bounds: Bounds,
    aa_samples: usize,
    heads: Vec<usize>,
    fragments: Vec<Fragment>,
}

impl TileBuffer {
    fn add(&mut self, x: usize, y: usize, sample: usize, mut fragment: Fragment) {
        let slot = ((y - self.bounds.y0) * (self.bounds.x1 - self.bounds.x0) + x - self.bounds.x0)
            * self.aa_samples
            + sample;
        fragment.next = self.heads[slot];
        self.heads[slot] = self.fragments.len();
        self.fragments.push(fragment);
    }
}

struct TileImage {
    bounds: Bounds,
    colors: Vec<V3>,
}

fn nonnegative(value: V3) -> bool {
    value.is_finite() && value.x >= 0.0 && value.y >= 0.0 && value.z >= 0.0
}

fn validate(scene: &Scene, camera: &Camera, config: &RenderConfig) -> SilkResult<()> {
    if config.width == 0 || config.height == 0 || config.aa == 0 {
        return Err("atelier render dimensions and AA grid must be positive".into());
    }
    let pixels = (config.width as usize)
        .checked_mul(config.height as usize)
        .ok_or("image dimensions overflow")?;
    let samples =
        (config.aa as usize).checked_mul(config.aa as usize).ok_or("AA dimensions overflow")?;
    pixels
        .checked_mul(3)
        .and_then(|n| n.checked_mul(std::mem::size_of::<u16>()))
        .filter(|&n| isize::try_from(n).is_ok())
        .ok_or("image allocation exceeds addressable memory")?;
    samples
        .checked_mul(TILE * TILE)
        .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
        .filter(|&n| isize::try_from(n).is_ok())
        .ok_or("AA tile allocation exceeds addressable memory")?;
    if !camera.position.is_finite()
        || !camera.target.is_finite()
        || !camera.up.is_finite()
        || !camera.orthographic_height.is_finite()
        || camera.orthographic_height <= 0.0
    {
        return Err("invalid orthographic camera".into());
    }
    let direction = (camera.target - camera.position).normalized();
    if direction.length_squared() < 0.5 || direction.cross(camera.up).length_squared() < 1e-12 {
        return Err("camera direction and up axis must form a usable frame".into());
    }
    if !config.exposure.is_finite()
        || !config.exposure.exp2().is_finite()
        || config.exposure.exp2() == 0.0
        || !config.light_rotation_degrees.is_finite()
        || !nonnegative(config.background)
        || [config.key_strength, config.rim_strength, config.fill_strength, config.bloom_strength]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("invalid studio or exposure settings".into());
    }
    for material in &scene.materials {
        if !nonnegative(material.front_color)
            || !nonnegative(material.back_color)
            || !nonnegative(material.optical_depth)
            || !nonnegative(material.emission)
            || !material.roughness.is_finite()
            || !(0.0..=1.0).contains(&material.roughness)
            || !material.anisotropy.is_finite()
            || !(0.0..=1.0).contains(&material.anisotropy)
            || !material.sheen.is_finite()
            || material.sheen < 0.0
            || !material.fiber_frequency.is_finite()
            || material.fiber_frequency < 0.0
            || !material.fiber_strength.is_finite()
            || !(0.0..=1.0).contains(&material.fiber_strength)
        {
            return Err("invalid thin-surface material".into());
        }
    }
    if scene.vertices.iter().any(|v| {
        !v.position.is_finite()
            || !v.normal.is_finite()
            || !v.tangent.is_finite()
            || !v.uv.iter().all(|x| x.is_finite())
    }) {
        return Err("surface vertices must have finite attributes".into());
    }
    if scene.triangles.iter().any(|t| {
        t.material >= scene.materials.len()
            || t.indices.iter().any(|&i| i as usize >= scene.vertices.len())
    }) {
        return Err("triangle references an absent vertex or material".into());
    }
    for strand in &scene.strands {
        if strand.points.len() != strand.radii.len()
            || strand.material >= scene.materials.len()
            || strand.points.iter().any(|p| !p.is_finite())
            || strand.radii.iter().any(|r| !r.is_finite() || *r < 0.0)
        {
            return Err("invalid strand points, radii, or material".into());
        }
    }
    Ok(())
}

fn edge(a: Point, b: Point, x: f64, y: f64) -> f64 {
    (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)
}
fn top_left(a: Point, b: Point) -> bool {
    b.y < a.y || (b.y == a.y && b.x > a.x)
}

fn project(scene: &Scene, camera: &CameraFrame, config: &RenderConfig) -> SilkResult<Projected> {
    let vertices: Vec<_> = scene
        .vertices
        .iter()
        .map(|v| ProjectedVertex {
            point: camera.project(v.position),
            world: v.position,
            normal: v.normal,
            tangent: v.tangent,
            uv: v.uv,
        })
        .collect();
    if vertices.iter().any(|v| ![v.point.x, v.point.y, v.point.depth].iter().all(|v| v.is_finite()))
    {
        return Err("surface projection exceeds finite coordinates".into());
    }
    let mut triangles = Vec::with_capacity(scene.triangles.len());
    for triangle in &scene.triangles {
        let mut indices = triangle.indices.map(|i| i as usize);
        let mut p = indices.map(|i| vertices[i].point);
        let mut area = edge(p[0], p[1], p[2].x, p[2].y);
        if area.abs() < 1e-12 || p.iter().all(|p| p.depth <= 0.0) {
            continue;
        }
        let original = indices.map(|i| scene.vertices[i].position);
        let fallback_normal =
            (original[1] - original[0]).cross(original[2] - original[0]).normalized();
        if area < 0.0 {
            indices.swap(1, 2);
            p.swap(1, 2);
            area = -area;
        }
        let bounds = Bounds::new(
            p.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            p.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
            p.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
            p.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
            config,
        );
        if bounds.empty() {
            continue;
        }
        let v = indices.map(|i| vertices[i].uv[1]);
        let dx =
            (v[0] * (p[1].y - p[2].y) + v[1] * (p[2].y - p[0].y) + v[2] * (p[0].y - p[1].y)) / area;
        let dy =
            (v[0] * (p[2].x - p[1].x) + v[1] * (p[0].x - p[2].x) + v[2] * (p[1].x - p[0].x)) / area;
        triangles.push(SurfaceTriangle {
            indices,
            material: triangle.material,
            bounds,
            inverse_area: 1.0 / area,
            inclusive: [top_left(p[1], p[2]), top_left(p[2], p[0]), top_left(p[0], p[1])],
            fallback_normal,
            transverse_variance: (dx * dx + dy * dy) / (12.0 * f64::from(config.aa).powi(2)),
        });
    }
    let mut fibers = Vec::new();
    for (group, strand) in scene.strands.iter().enumerate() {
        let mut arc = 0.0;
        for (index, pair) in strand.points.windows(2).enumerate() {
            let length = (pair[1] - pair[0]).length();
            if length <= 1e-12 {
                continue;
            }
            let first = camera.project(pair[0]);
            let last = camera.project(pair[1]);
            if ![first.x, first.y, first.depth, last.x, last.y, last.depth]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err("fiber projection exceeds finite coordinates".into());
            }
            let radius_first = strand.radii[index] * camera.scale;
            let radius_last = strand.radii[index + 1] * camera.scale;
            let padding = radius_first.max(radius_last) + FIBER_SIGMA * 4.0;
            let bounds = Bounds::new(
                first.x.min(last.x) - padding,
                first.y.min(last.y) - padding,
                first.x.max(last.x) + padding,
                first.y.max(last.y) + padding,
                config,
            );
            if !bounds.empty()
                && (first.depth > 0.0 || last.depth > 0.0)
                && radius_first.max(radius_last) > 0.0
            {
                fibers.push(FiberSegment {
                    first,
                    last,
                    world_first: pair[0],
                    world_last: pair[1],
                    radius_first,
                    radius_last,
                    tangent: (pair[1] - pair[0]) / length,
                    material: strand.material,
                    group,
                    arc_start: arc,
                    arc_length: length,
                    bounds,
                });
            }
            arc += length;
        }
    }
    let tiles_x = (config.width as usize).div_ceil(TILE);
    let tiles_y = (config.height as usize).div_ceil(TILE);
    let mut bins = vec![Vec::new(); tiles_x * tiles_y];
    for (index, bounds) in
        triangles.iter().map(|t| t.bounds).chain(fibers.iter().map(|f| f.bounds)).enumerate()
    {
        for ty in bounds.y0 / TILE..=(bounds.y1 - 1) / TILE {
            for tx in bounds.x0 / TILE..=(bounds.x1 - 1) / TILE {
                bins[ty * tiles_x + tx].push(index);
            }
        }
    }
    Ok(Projected { vertices, triangles, fibers, bins, tiles_x })
}

fn raster_triangle(
    index: usize,
    triangle: &SurfaceTriangle,
    geometry: &Projected,
    scene: &Scene,
    studio: &Studio,
    offsets: &[[f64; 2]],
    buffer: &mut TileBuffer,
) {
    let bounds = triangle.bounds.intersection(buffer.bounds);
    let v = triangle.indices.map(|i| geometry.vertices[i]);
    for y in bounds.y0..bounds.y1 {
        for x in bounds.x0..bounds.x1 {
            for (sample, offset) in offsets.iter().enumerate() {
                let px = x as f64 + offset[0];
                let py = y as f64 + offset[1];
                let edges = [
                    edge(v[1].point, v[2].point, px, py),
                    edge(v[2].point, v[0].point, px, py),
                    edge(v[0].point, v[1].point, px, py),
                ];
                if edges
                    .iter()
                    .zip(triangle.inclusive)
                    .any(|(&e, inclusive)| e < 0.0 || (e == 0.0 && !inclusive))
                {
                    continue;
                }
                let w = edges.map(|e| e * triangle.inverse_area);
                let depth =
                    v[0].point.depth * w[0] + v[1].point.depth * w[1] + v[2].point.depth * w[2];
                if depth <= 0.0 {
                    continue;
                }
                let mut normal =
                    (v[0].normal * w[0] + v[1].normal * w[1] + v[2].normal * w[2]).normalized();
                if normal.length_squared() < 0.5 {
                    normal = triangle.fallback_normal;
                }
                let tangent = v[0].tangent * w[0] + v[1].tangent * w[1] + v[2].tangent * w[2];
                let uv = [
                    v[0].uv[0] * w[0] + v[1].uv[0] * w[1] + v[2].uv[0] * w[2],
                    v[0].uv[1] * w[0] + v[1].uv[1] * w[1] + v[2].uv[1] * w[2],
                ];
                let optical = studio.shade(
                    &scene.materials[triangle.material],
                    v[0].world * w[0] + v[1].world * w[1] + v[2].world * w[2],
                    normal,
                    tangent,
                    uv,
                    triangle.transverse_variance,
                );
                buffer.add(
                    x,
                    y,
                    sample,
                    Fragment {
                        depth,
                        optical,
                        coverage: 1.0,
                        group: NONE,
                        arc: 0.0,
                        merge_range: 0.0,
                        order: index,
                        next: NONE,
                    },
                );
            }
        }
    }
}

/// Smooth Gaussian footprint integral; its finite error is well below a code value.
fn erf(value: f64) -> f64 {
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let p = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592)
        * t;
    (1.0 - p * (-x * x).exp()).copysign(value)
}

fn raster_fiber(
    index: usize,
    fiber: &FiberSegment,
    scene: &Scene,
    studio: &Studio,
    camera: &CameraFrame,
    offsets: &[[f64; 2]],
    buffer: &mut TileBuffer,
) {
    let dx = fiber.last.x - fiber.first.x;
    let dy = fiber.last.y - fiber.first.y;
    let length = (dx * dx + dy * dy).sqrt();
    let (tx, ty) = if length > 1e-10 { (dx / length, dy / length) } else { (1.0, 0.0) };
    let view = -camera.forward;
    let mut face = (view - fiber.tangent * view.dot(fiber.tangent)).normalized();
    if face.length_squared() < 0.5 {
        face = view;
    }
    let side = fiber.tangent.cross(view).normalized();
    let bounds = fiber.bounds.intersection(buffer.bounds);
    let gaussian_scale = FIBER_SIGMA * std::f64::consts::SQRT_2;
    for y in bounds.y0..bounds.y1 {
        for x in bounds.x0..bounds.x1 {
            for (sample, offset) in offsets.iter().enumerate() {
                let px = x as f64 + offset[0] - fiber.first.x;
                let py = y as f64 + offset[1] - fiber.first.y;
                let along = px * tx + py * ty;
                let distance = -px * ty + py * tx;
                let t = if length > 1e-10 { (along / length).clamp(0.0, 1.0) } else { 0.5 };
                let radius = fiber.radius_first * (1.0 - t) + fiber.radius_last * t;
                if radius <= 0.0 {
                    continue;
                }
                let longitudinal = if length > 1e-10 {
                    0.5 * (erf(along / gaussian_scale) + erf((length - along) / gaussian_scale))
                } else {
                    1.0
                };
                let transverse = if radius < 0.25 {
                    2.0 * radius / ((2.0 * std::f64::consts::PI).sqrt() * FIBER_SIGMA)
                        * (-0.5 * (distance / FIBER_SIGMA).powi(2)).exp()
                } else {
                    0.5 * (erf((radius - distance) / gaussian_scale)
                        + erf((radius + distance) / gaussian_scale))
                };
                let coverage = if length > 1e-10 {
                    transverse * longitudinal
                } else {
                    radius * radius / (2.0 * FIBER_SIGMA * FIBER_SIGMA)
                        * (-0.5 * (px * px + py * py) / (FIBER_SIGMA * FIBER_SIGMA)).exp()
                }
                .clamp(0.0, 1.0);
                if coverage < 1e-8 {
                    continue;
                }
                let depth = fiber.first.depth * (1.0 - t) + fiber.last.depth * t;
                if depth <= 0.0 {
                    continue;
                }
                let q = if radius >= 0.85 { (distance / radius).clamp(-0.98, 0.98) } else { 0.0 };
                let normal = (face * (1.0 - q * q).sqrt() + side * q).normalized();
                let arc = fiber.arc_start + fiber.arc_length * t;
                let optical = studio.shade(
                    &scene.materials[fiber.material],
                    fiber.world_first * (1.0 - t) + fiber.world_last * t,
                    normal,
                    fiber.tangent,
                    [arc, 0.0],
                    0.0,
                );
                buffer.add(
                    x,
                    y,
                    sample,
                    Fragment {
                        depth,
                        optical,
                        coverage,
                        group: fiber.group,
                        arc,
                        merge_range: (4.0 * FIBER_SIGMA + 2.0 * radius) / camera.scale,
                        order: index,
                        next: NONE,
                    },
                );
            }
        }
    }
}

fn composite(buffer: &TileBuffer, head: usize, background: V3) -> V3 {
    let mut indices = SmallVec::<[usize; 8]>::new();
    let mut next = head;
    while next != NONE {
        indices.push(next);
        next = buffer.fragments[next].next;
    }
    indices.sort_unstable_by(|&a, &b| {
        buffer.fragments[a]
            .depth
            .total_cmp(&buffer.fragments[b].depth)
            .then_with(|| buffer.fragments[a].order.cmp(&buffer.fragments[b].order))
    });
    let white = V3::new(1.0, 1.0, 1.0);
    let mut throughput = white;
    let mut result = V3::ZERO;
    let mut cursor = 0;
    while cursor < indices.len() {
        let first = buffer.fragments[indices[cursor]];
        let mut coverage = first.coverage;
        let mut light = first.optical.light * coverage;
        let mut transmission = first.optical.transmission * coverage;
        cursor += 1;
        // Adjacent Gaussian segment footprints describe one fiber, not multiple
        // stacked fibers. Their partition weights must sum before compositing.
        if first.group != NONE {
            while cursor < indices.len() {
                let other = buffer.fragments[indices[cursor]];
                if other.group != first.group
                    || (other.arc - first.arc).abs() > first.merge_range.max(other.merge_range)
                    || (other.depth - first.depth).abs() > first.merge_range.max(other.merge_range)
                {
                    break;
                }
                coverage += other.coverage;
                light += other.optical.light * other.coverage;
                transmission += other.optical.transmission * other.coverage;
                cursor += 1;
            }
        }
        let covered = coverage.min(1.0);
        result += throughput.hadamard(light) * (covered / coverage);
        let layer_transmission = white * (1.0 - covered) + transmission * (covered / coverage);
        throughput = throughput.hadamard(layer_transmission);
    }
    result + throughput.hadamard(background)
}

/// Rasterize one geometric pose into unclipped linear RGB radiance.
///
/// This performs coverage and depth-sorted optical composition only. Average
/// subframe results in this representation, then call [`finish_linear`] once.
pub fn render_linear(scene: &Scene, camera: &Camera, config: &RenderConfig) -> SilkResult<Vec<V3>> {
    validate(scene, camera, config)?;
    let forward = (camera.target - camera.position).normalized();
    let right = forward.cross(camera.up).normalized();
    let frame = CameraFrame {
        forward,
        right,
        up: right.cross(forward).normalized(),
        position: camera.position,
        target: camera.target,
        scale: f64::from(config.height) / camera.orthographic_height,
        width: f64::from(config.width),
        height: f64::from(config.height),
    };
    let geometry = project(scene, &frame, config)?;
    let studio = Studio::new(&frame, config);
    let aa = config.aa as usize;
    let offsets: Vec<_> = (0..aa)
        .flat_map(|y| {
            (0..aa).map(move |x| [(x as f64 + 0.5) / aa as f64, (y as f64 + 0.5) / aa as f64])
        })
        .collect();
    let tiles: SilkResult<Vec<TileImage>> = geometry
        .bins
        .par_iter()
        .enumerate()
        .filter(|(_, bin)| !bin.is_empty())
        .map(|(tile, bin)| {
            let x0 = tile % geometry.tiles_x * TILE;
            let y0 = tile / geometry.tiles_x * TILE;
            let bounds = Bounds {
                x0,
                y0,
                x1: (x0 + TILE).min(config.width as usize),
                y1: (y0 + TILE).min(config.height as usize),
            };
            let pixels = (bounds.x1 - bounds.x0) * (bounds.y1 - bounds.y0);
            let mut buffer = TileBuffer {
                bounds,
                aa_samples: offsets.len(),
                heads: vec![NONE; pixels * offsets.len()],
                fragments: Vec::new(),
            };
            for &index in bin {
                if index < geometry.triangles.len() {
                    raster_triangle(
                        index,
                        &geometry.triangles[index],
                        &geometry,
                        scene,
                        &studio,
                        &offsets,
                        &mut buffer,
                    );
                } else {
                    raster_fiber(
                        index,
                        &geometry.fibers[index - geometry.triangles.len()],
                        scene,
                        &studio,
                        &frame,
                        &offsets,
                        &mut buffer,
                    );
                }
            }
            let mut colors = vec![V3::ZERO; pixels];
            for (pixel, color) in colors.iter_mut().enumerate() {
                for &head in &buffer.heads[pixel * offsets.len()..(pixel + 1) * offsets.len()] {
                    *color += composite(&buffer, head, config.background);
                }
                *color /= offsets.len() as f64;
                if !color.is_finite() {
                    return Err("material lighting produced a non-finite pixel".into());
                }
            }
            Ok(TileImage { bounds, colors })
        })
        .collect();
    let mut output = vec![config.background; config.width as usize * config.height as usize];
    for tile in tiles? {
        let width = tile.bounds.x1 - tile.bounds.x0;
        for y in tile.bounds.y0..tile.bounds.y1 {
            output[y * config.width as usize + tile.bounds.x0
                ..y * config.width as usize + tile.bounds.x1]
                .copy_from_slice(
                    &tile.colors[(y - tile.bounds.y0) * width..(y - tile.bounds.y0 + 1) * width],
                );
        }
    }
    Ok(output)
}

fn blur(input: &[V3], width: usize, height: usize, stride: usize) -> Vec<V3> {
    let kernel = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut horizontal = vec![V3::ZERO; input.len()];
    horizontal.par_iter_mut().enumerate().for_each(|(index, out)| {
        let x = index % width;
        let y = index / width;
        for (k, &weight) in kernel.iter().enumerate() {
            let sx = (x as isize + (k as isize - 2) * stride as isize).clamp(0, width as isize - 1)
                as usize;
            *out += input[y * width + sx] * (weight / 16.0);
        }
    });
    let mut result = vec![V3::ZERO; input.len()];
    result.par_iter_mut().enumerate().for_each(|(index, out)| {
        let x = index % width;
        let y = index / width;
        for (k, &weight) in kernel.iter().enumerate() {
            let sy = (y as isize + (k as isize - 2) * stride as isize).clamp(0, height as isize - 1)
                as usize;
            *out += horizontal[sy * width + x] * (weight / 16.0);
        }
    });
    result
}

fn bloom(colors: &mut [V3], config: &RenderConfig) {
    if config.bloom_strength == 0.0 {
        return;
    }
    let width = (config.width as usize).div_ceil(4);
    let height = (config.height as usize).div_ceil(4);
    let mut highlights = vec![V3::ZERO; width * height];
    highlights.par_iter_mut().enumerate().for_each(|(index, out)| {
        let x = index % width * 4;
        let y = index / width * 4;
        let mut count = 0;
        for sy in y..(y + 4).min(config.height as usize) {
            for sx in x..(x + 4).min(config.width as usize) {
                let color = colors[sy * config.width as usize + sx];
                let luma = color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722;
                if luma > 1.0 {
                    *out += color * ((luma - 1.0) / luma);
                }
                count += 1;
            }
        }
        *out /= f64::from(count);
    });
    let near = blur(&highlights, width, height, 1);
    let far = blur(&near, width, height, 3);
    colors.par_iter_mut().enumerate().for_each(|(index, color)| {
        let x = index % config.width as usize;
        let y = index / config.width as usize;
        let gx = (x as f64 - 1.5) / 4.0;
        let gy = (y as f64 - 1.5) / 4.0;
        let x0 = gx.floor();
        let y0 = gy.floor();
        let fx = gx - x0;
        let fy = gy - y0;
        let mut glow = V3::ZERO;
        for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
            for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
                let ix = (x0 as isize + dx).clamp(0, width as isize - 1) as usize;
                let iy = (y0 as isize + dy).clamp(0, height as isize - 1) as usize;
                glow += (near[iy * width + ix] * 0.72 + far[iy * width + ix] * 0.28) * (wx * wy);
            }
        }
        *color += glow * config.bloom_strength;
    });
}

fn display(color: V3, exposure: f64) -> [u16; 3] {
    let color = color * exposure;
    let peak = color.x.max(color.y).max(color.z).max(0.0);
    let mapped =
        if peak > 0.70 { 0.70 + 0.30 * (1.0 - (-(peak - 0.70) / 0.30).exp()) } else { peak };
    let mut color = if peak > 0.0 { color * (mapped / peak) } else { V3::ZERO };
    let neutral = ((peak - 1.4) * 0.055).clamp(0.0, 0.25);
    color = color.lerp(V3::new(mapped, mapped, mapped), neutral);
    [color.x, color.y, color.z].map(|linear| {
        let value = linear.clamp(0.0, 1.0);
        let encoded = if value <= 0.003_130_8 {
            12.92 * value
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 65535.0 + 0.5) as u16
    })
}

/// Render layered ribbons and fine fibers into a 16-bit display-encoded RGB image.
///
/// Worker count changes scheduling only. Coverage, depth order, optical sums and
/// filtering are deterministic; blank tiles never perform surface shading work.
pub fn render(
    scene: &Scene,
    camera: &Camera,
    config: &RenderConfig,
) -> SilkResult<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    finish_linear(render_linear(scene, camera, config)?, config)
}

/// Finish an already averaged linear-light image with bloom, exposure and encoding.
///
/// The input is a row-major `width * height` RGB radiance buffer. This boundary
/// rejects mismatched dimensions and non-finite values before applying the
/// highlight shoulder and sRGB transfer function.
pub fn finish_linear(
    mut colors: Vec<V3>,
    config: &RenderConfig,
) -> SilkResult<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let count = (config.width as usize)
        .checked_mul(config.height as usize)
        .ok_or("image dimensions overflow")?;
    if count == 0 || colors.len() != count || colors.iter().any(|color| !color.is_finite()) {
        return Err(
            "linear image must match nonzero dimensions and contain finite RGB values".into()
        );
    }
    if !config.exposure.is_finite()
        || !config.exposure.exp2().is_finite()
        || config.exposure.exp2() == 0.0
        || !config.bloom_strength.is_finite()
        || config.bloom_strength < 0.0
    {
        return Err("invalid finishing exposure or bloom strength".into());
    }
    count
        .checked_mul(3)
        .filter(|&n| isize::try_from(n).is_ok())
        .ok_or("finished image exceeds addressable memory")?;
    bloom(&mut colors, config);
    let exposure = config.exposure.exp2();
    let finite_limit = f64::MAX / exposure;
    if colors.iter().any(|color| {
        !color.is_finite()
            || color.x.abs() > finite_limit
            || color.y.abs() > finite_limit
            || color.z.abs() > finite_limit
    }) {
        return Err("bloom or exposure would produce non-finite radiance".into());
    }
    let mut pixels = vec![0_u16; colors.len() * 3];
    pixels
        .par_chunks_exact_mut(3)
        .zip(colors.par_iter())
        .for_each(|(pixel, &color)| pixel.copy_from_slice(&display(color, exposure)));
    ImageBuffer::from_raw(config.width, config.height, pixels)
        .ok_or_else(|| "invalid atelier image dimensions".into())
}

#[cfg(test)]
mod tests {
    use super::super::{Strand, Triangle, Vertex};
    use super::*;

    fn camera() -> Camera {
        Camera {
            position: V3::new(0.0, 0.0, 5.0),
            target: V3::ZERO,
            up: V3::new(0.0, 1.0, 0.0),
            orthographic_height: 2.0,
        }
    }
    fn config() -> RenderConfig {
        RenderConfig {
            width: 16,
            height: 16,
            aa: 2,
            exposure: 0.0,
            background: V3::ZERO,
            key_strength: 0.0,
            rim_strength: 0.0,
            fill_strength: 0.0,
            bloom_strength: 0.0,
            ..RenderConfig::default()
        }
    }
    fn material(emission: V3, depth: f64) -> Material {
        Material {
            emission,
            optical_depth: V3::new(depth, depth, depth),
            fiber_strength: 0.0,
            ..Material::default()
        }
    }
    fn quad(scene: &mut Scene, z: f64, material: usize, right: f64) {
        let start = scene.vertices.len() as u32;
        for (position, uv) in [
            (V3::new(-1.0, -1.0, z), [0.0, 0.0]),
            (V3::new(right, -1.0, z), [1.0, 0.0]),
            (V3::new(right, 1.0, z), [1.0, 1.0]),
            (V3::new(-1.0, 1.0, z), [0.0, 1.0]),
        ] {
            scene.vertices.push(Vertex {
                position,
                normal: V3::new(0.0, 0.0, 1.0),
                tangent: V3::new(1.0, 0.0, 0.0),
                uv,
            });
        }
        scene.triangles.extend([
            Triangle { indices: [start, start + 1, start + 2], material },
            Triangle { indices: [start, start + 2, start + 3], material },
        ]);
    }
    #[test]
    fn transparent_layers_follow_depth_not_input_order_and_shared_edges_do_not_double() {
        let mut scene = Scene {
            materials: vec![
                material(V3::new(0.4, 0.0, 0.0), std::f64::consts::LN_2),
                material(V3::new(0.0, 0.0, 0.4), std::f64::consts::LN_2),
            ],
            ..Scene::default()
        };
        quad(&mut scene, 1.0, 0, 1.0);
        quad(&mut scene, 0.0, 1, 1.0);
        let first = render_linear(&scene, &camera(), &config()).unwrap();
        for p in &first {
            assert!((p.x - 0.4).abs() < 1e-12);
            assert!((p.z - 0.2).abs() < 1e-12);
        }
        scene.triangles.reverse();
        assert_eq!(first, render_linear(&scene, &camera(), &config()).unwrap());
    }
    #[test]
    fn clear_front_surface_does_not_hide_far_layer() {
        let mut scene = Scene {
            materials: vec![material(V3::ZERO, 0.0), material(V3::new(0.1, 0.2, 0.3), 0.4)],
            ..Scene::default()
        };
        quad(&mut scene, 1.0, 0, 1.0);
        quad(&mut scene, 0.0, 1, 1.0);
        for p in render_linear(&scene, &camera(), &config()).unwrap() {
            assert!((p - V3::new(0.1, 0.2, 0.3)).length() < 1e-12);
        }
    }
    #[test]
    fn stratified_coverage_resolves_a_half_pixel_edge() {
        let mut scene =
            Scene { materials: vec![material(V3::new(1.0, 1.0, 1.0), 40.0)], ..Scene::default() };
        quad(&mut scene, 0.0, 0, 0.0);
        let c = RenderConfig { width: 1, height: 1, aa: 4, ..config() };
        let pixel = render_linear(&scene, &camera(), &c).unwrap()[0];
        assert!((pixel.x - 0.5).abs() < 1e-12);
    }
    #[test]
    fn subpixel_fibers_remain_visible_and_segmenting_does_not_make_beads() {
        let make = |points: Vec<V3>| Scene {
            strands: vec![Strand { radii: vec![0.008; points.len()], points, material: 0 }],
            materials: vec![material(V3::new(0.3, 0.3, 0.3), 0.5)],
            ..Scene::default()
        };
        let simple = make(vec![V3::new(-0.8, 0.027, 0.0), V3::new(0.8, 0.027, 0.0)]);
        let split =
            make((0..=16).map(|i| V3::new(-0.8 + f64::from(i) * 0.1, 0.027, 0.0)).collect());
        let a = render_linear(&simple, &camera(), &config()).unwrap();
        let b = render_linear(&split, &camera(), &config()).unwrap();
        assert!(a.iter().map(|p| p.x).sum::<f64>() > 0.1);
        let error = a.iter().zip(&b).map(|(a, b)| (a.x - b.x).abs()).sum::<f64>();
        assert!(error < 0.005, "segmentation error {error}");
    }
    #[test]
    fn pixel_values_are_identical_across_worker_counts() {
        let mut scene = Scene { materials: vec![Material::default()], ..Scene::default() };
        quad(&mut scene, 0.0, 0, 0.8);
        let c = RenderConfig {
            width: 48,
            height: 32,
            key_strength: 1.0,
            rim_strength: 1.0,
            fill_strength: 0.2,
            bloom_strength: 0.04,
            ..config()
        };
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| render(&scene, &camera(), &c).unwrap())
        };
        assert_eq!(run(1), run(4));
    }
    #[test]
    fn invalid_camera_or_material_is_rejected() {
        let mut scene = Scene::default();
        let mut bad = camera();
        bad.up = V3::new(0.0, 0.0, 1.0);
        assert!(render(&scene, &bad, &config()).is_err());
        scene.materials.push(material(V3::ZERO, -0.1));
        assert!(render(&scene, &camera(), &config()).is_err());
    }

    #[test]
    fn front_and_reverse_dyes_are_distinct() {
        let mut scene = Scene {
            materials: vec![Material {
                front_color: V3::new(0.8, 0.1, 0.1),
                back_color: V3::new(0.1, 0.1, 0.8),
                optical_depth: V3::new(0.5, 0.5, 0.5),
                ..Material::default()
            }],
            ..Scene::default()
        };
        quad(&mut scene, 0.0, 0, 1.0);
        let settings = RenderConfig { fill_strength: 1.0, ..config() };
        let front = render_linear(&scene, &camera(), &settings).unwrap();
        for vertex in &mut scene.vertices {
            vertex.normal = -vertex.normal;
        }
        let reverse = render_linear(&scene, &camera(), &settings).unwrap();
        assert!(front[0].x > front[0].z);
        assert!(reverse[0].z > reverse[0].x);
    }

    #[test]
    fn finishing_accepts_linear_temporal_means_and_rejects_invalid_buffers() {
        let settings = RenderConfig { width: 1, height: 1, ..config() };
        let dark = V3::ZERO;
        let bright = V3::new(1.0, 1.0, 1.0);
        let image = finish_linear(vec![(dark + bright) * 0.5], &settings).unwrap();
        let expected = ((1.055 * 0.5_f64.powf(1.0 / 2.4) - 0.055) * 65535.0 + 0.5) as u16;
        assert_eq!(image.as_raw(), &[expected; 3]);
        assert!(finish_linear(Vec::new(), &settings).is_err());
        assert!(finish_linear(vec![V3::new(f64::NAN, 0.0, 0.0)], &settings).is_err());
        assert!(
            finish_linear(
                vec![V3::new(f64::MAX, 0.0, 0.0)],
                &RenderConfig { exposure: 2.0, ..settings }
            )
            .is_err()
        );
    }
}
