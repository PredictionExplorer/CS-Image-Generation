//! Deterministic triangle acceleration and smoothly interpolated fabric frames.
use super::super::{Mesh, V3};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) struct Ray {
    pub origin: V3,
    pub direction: V3,
}

#[derive(Clone, Copy)]
struct Bounds {
    min: V3,
    max: V3,
}

impl Bounds {
    fn empty() -> Self {
        Self {
            min: V3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: V3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    fn include(&mut self, p: V3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    fn hits(self, ray: Ray, max_distance: f64) -> bool {
        let mut near = 0.0_f64;
        let mut far = max_distance;
        for axis in 0..3 {
            let d = ray.direction.axis(axis);
            let o = ray.origin.axis(axis);
            if d.abs() < 1e-14 {
                if o < self.min.axis(axis) - 1e-8 || o > self.max.axis(axis) + 1e-8 {
                    return false;
                }
            } else {
                let mut a = (self.min.axis(axis) - o) / d;
                let mut b = (self.max.axis(axis) - o) / d;
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                near = near.max(a);
                far = far.min(b);
                if far + 1e-9 < near {
                    return false;
                }
            }
        }
        true
    }
}

struct Node {
    bounds: Bounds,
    start: usize,
    count: usize,
    children: Option<(usize, usize)>,
}

#[derive(Clone, Copy)]
pub(super) struct Hit {
    pub distance: f64,
    pub triangle: usize,
    pub bary: [f64; 3],
}

pub(super) struct Surface {
    pub point: V3,
    pub normal: V3,
    pub geometric: V3,
    pub tangent: V3,
    pub uv: [f64; 2],
    pub hem_distance: f64,
}

pub(super) struct Geometry {
    positions: Vec<V3>,
    normals: Vec<V3>,
    triangles: Vec<[u32; 3]>,
    uv: Vec<[f64; 2]>,
    hem: Vec<f64>,
    order: Vec<usize>,
    nodes: Vec<Node>,
}

impl Geometry {
    pub fn new(mesh: &Mesh, positions: Vec<V3>) -> Self {
        let mut normals = vec![V3::ZERO; positions.len()];
        for &[a, b, c] in &mesh.triangles {
            let n = (positions[b as usize] - positions[a as usize])
                .cross(positions[c as usize] - positions[a as usize]);
            for i in [a, b, c] {
                normals[i as usize] += n;
            }
        }
        for n in &mut normals {
            *n = n.normalized();
        }
        let hem = boundary_distances(mesh);
        let mut geom = Self {
            positions,
            normals,
            triangles: mesh.triangles.clone(),
            uv: mesh.uv.clone(),
            hem,
            order: (0..mesh.triangles.len()).collect(),
            nodes: Vec::new(),
        };
        if !geom.order.is_empty() {
            geom.build(0, geom.order.len());
        }
        geom
    }

    fn build(&mut self, start: usize, count: usize) -> usize {
        let mut bounds = Bounds::empty();
        let mut centers = Bounds::empty();
        for &index in &self.order[start..start + count] {
            let tri = self.triangles[index];
            let p = tri.map(|v| self.positions[v as usize]);
            for vertex in p {
                bounds.include(vertex);
            }
            centers.include((p[0] + p[1] + p[2]) / 3.0);
        }
        let node = self.nodes.len();
        self.nodes.push(Node { bounds, start, count, children: None });
        if count > 6 {
            let span = centers.max - centers.min;
            let axis = if span.x >= span.y && span.x >= span.z {
                0
            } else if span.y >= span.z {
                1
            } else {
                2
            };
            let triangles = &self.triangles;
            let positions = &self.positions;
            self.order[start..start + count].sort_by(|&a, &b| {
                let centroid = |i: usize| {
                    triangles[i].iter().map(|&v| positions[v as usize].axis(axis)).sum::<f64>()
                };
                centroid(a).total_cmp(&centroid(b)).then_with(|| a.cmp(&b))
            });
            let half = count / 2;
            let left = self.build(start, half);
            let right = self.build(start + half, count - half);
            self.nodes[node].children = Some((left, right));
        }
        node
    }

    pub fn intersect(&self, ray: Ray, min_distance: f64, max_distance: f64) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut stack = [0usize; 96];
        let mut size = 1;
        let mut limit = max_distance;
        let mut best = None;
        while size != 0 {
            size -= 1;
            let node = &self.nodes[stack[size]];
            if !node.bounds.hits(ray, limit) {
                continue;
            }
            if let Some((left, right)) = node.children {
                stack[size] = left;
                stack[size + 1] = right;
                size += 2;
            } else {
                for &index in &self.order[node.start..node.start + node.count] {
                    if let Some(hit) = self.hit_triangle(index, ray, min_distance, limit) {
                        limit = hit.distance;
                        best = Some(hit);
                    }
                }
            }
        }
        best
    }

    fn hit_triangle(
        &self,
        index: usize,
        ray: Ray,
        min_distance: f64,
        max_distance: f64,
    ) -> Option<Hit> {
        let [a, b, c] = self.triangles[index].map(|v| self.positions[v as usize]);
        let edge1 = b - a;
        let edge2 = c - a;
        let cross = ray.direction.cross(edge2);
        let determinant = edge1.dot(cross);
        if determinant.abs() < 1e-14 {
            return None;
        }
        let inv = 1.0 / determinant;
        let relative = ray.origin - a;
        let u = relative.dot(cross) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = relative.cross(edge1);
        let v = ray.direction.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let distance = edge2.dot(q) * inv;
        if distance <= min_distance || distance >= max_distance {
            return None;
        }
        Some(Hit { distance, triangle: index, bary: [1.0 - u - v, u, v] })
    }

    pub fn surface(&self, hit: Hit, ray: Ray) -> Surface {
        let ids = self.triangles[hit.triangle].map(|v| v as usize);
        let points = ids.map(|i| self.positions[i]);
        let mut geometric = (points[1] - points[0]).cross(points[2] - points[0]).normalized();
        let mut normal = V3::ZERO;
        let mut uv = [0.0; 2];
        let mut hem_distance = 0.0;
        for (j, &i) in ids.iter().enumerate() {
            normal += self.normals[i] * hit.bary[j];
            uv[0] += self.uv[i][0] * hit.bary[j];
            uv[1] += self.uv[i][1] * hit.bary[j];
            hem_distance += self.hem[i] * hit.bary[j];
        }
        normal = normal.normalized();
        if normal.length_squared() < 0.5 {
            normal = geometric;
        }
        if geometric.dot(ray.direction) > 0.0 {
            geometric = -geometric;
        }
        if normal.dot(geometric) < 0.0 {
            normal = -normal;
        }
        // Prevent strongly smoothed silhouettes from pointing behind the eye.
        if normal.dot(-ray.direction) < 0.08 {
            normal = (normal + geometric * 0.5).normalized();
        }
        let du1 = self.uv[ids[1]][0] - self.uv[ids[0]][0];
        let dv1 = self.uv[ids[1]][1] - self.uv[ids[0]][1];
        let du2 = self.uv[ids[2]][0] - self.uv[ids[0]][0];
        let dv2 = self.uv[ids[2]][1] - self.uv[ids[0]][1];
        let determinant = du1 * dv2 - du2 * dv1;
        let raw = if determinant.abs() > 1e-14 {
            ((points[1] - points[0]) * dv2 - (points[2] - points[0]) * dv1) / determinant
        } else {
            points[1] - points[0]
        };
        let mut tangent = (raw - normal * raw.dot(normal)).normalized();
        if tangent.length_squared() < 0.5 {
            tangent = basis(normal);
        }
        Surface {
            point: ray.origin + ray.direction * hit.distance,
            normal,
            geometric,
            tangent,
            uv,
            hem_distance,
        }
    }
}

pub(super) fn basis(normal: V3) -> V3 {
    let other = if normal.x.abs() < 0.8 { V3::new(1.0, 0.0, 0.0) } else { V3::new(0.0, 1.0, 0.0) };
    normal.cross(other).normalized()
}

fn boundary_distances(mesh: &Mesh) -> Vec<f64> {
    let mut counts: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for &[a, b, c] in &mesh.triangles {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            *counts.entry((i.min(j), i.max(j))).or_default() += 1;
        }
    }
    let edges: Vec<_> = counts
        .into_iter()
        .filter_map(|((a, b), count)| {
            (count == 1).then_some((mesh.uv[a as usize], mesh.uv[b as usize]))
        })
        .collect();
    mesh.uv
        .iter()
        .map(|p| {
            edges
                .iter()
                .map(|&(a, b)| {
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let len = d[0] * d[0] + d[1] * d[1];
                    let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len.max(1e-24))
                        .clamp(0.0, 1.0);
                    (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1])
                })
                .fold(1.0, f64::min)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn triangle() -> Geometry {
        let mesh = Mesh {
            positions: vec![
                V3::new(-1.0, -1.0, 0.0),
                V3::new(1.0, -1.0, 0.0),
                V3::new(0.0, 1.0, 0.0),
            ],
            triangles: vec![[0, 1, 2]],
            uv: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        };
        Geometry::new(&mesh, mesh.positions.clone())
    }
    #[test]
    fn shadow_ray_respects_occluder_distance_and_open_space() {
        let geom = triangle();
        let ray = Ray { origin: V3::new(0.0, 0.0, 1.0), direction: V3::new(0.0, 0.0, -1.0) };
        assert!(geom.intersect(ray, 1e-6, 2.0).is_some());
        assert!(geom.intersect(ray, 1e-6, 0.5).is_none());
        assert!(geom.intersect(Ray { origin: V3::new(2.0, 0.0, 1.0), ..ray }, 1e-6, 2.0).is_none());
        assert!(
            geom.intersect(
                Ray { origin: V3::new(0.0, 0.0, -1.0), direction: -ray.direction },
                1e-6,
                2.0
            )
            .is_some()
        );
    }
}
