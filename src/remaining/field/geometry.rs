//! Conservative bounds and nearest-triangle queries for the fixed source blank.

use crate::silk::V3;

#[derive(Clone, Copy, Debug)]
pub(super) struct Bounds {
    pub min: V3,
    pub max: V3,
}

impl Bounds {
    pub fn empty() -> Self {
        Self {
            min: V3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: V3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    pub fn include(&mut self, other: Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    pub fn expand(self, radius: f64) -> Self {
        let padding = V3::new(radius, radius, radius);
        Self { min: self.min - padding, max: self.max + padding }
    }

    pub fn contains(self, point: V3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    fn distance_squared(self, point: V3) -> f64 {
        (0..3)
            .map(|axis| {
                let x = point.axis(axis);
                let delta = if x < self.min.axis(axis) {
                    self.min.axis(axis) - x
                } else if x > self.max.axis(axis) {
                    x - self.max.axis(axis)
                } else {
                    0.0
                };
                delta * delta
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
pub(super) struct Triangle {
    points: [V3; 3],
    bounds: Bounds,
}

impl Triangle {
    pub fn new(points: [V3; 3]) -> Self {
        let min = points[0].min(points[1]).min(points[2]);
        let max = points[0].max(points[1]).max(points[2]);
        Self { points, bounds: Bounds { min, max }.expand(1e-12) }
    }

    fn centroid(&self) -> V3 {
        (self.points[0] + self.points[1] + self.points[2]) / 3.0
    }

    fn distance_squared(&self, point: V3) -> f64 {
        let [a, b, c] = self.points;
        let ab = b - a;
        let ac = c - a;
        let normal = ab.cross(ac);
        let normal_squared = normal.length_squared();
        let scale_squared =
            ab.length_squared().max(ac.length_squared()).max((c - b).length_squared());
        // Collinear and coincident source triangles are meaningful source
        // events. Their geometric limit is their three finite edge segments.
        if normal_squared > 1e-24 * scale_squared * scale_squared {
            let ap = a - point;
            let bp = b - point;
            let cp = c - point;
            let wa = bp.cross(cp).dot(normal);
            let wb = cp.cross(ap).dot(normal);
            let wc = ap.cross(bp).dot(normal);
            if wa >= 0.0 && wb >= 0.0 && wc >= 0.0 {
                let height_numerator = ap.dot(normal);
                return height_numerator * height_numerator / normal_squared;
            }
        }
        segment_distance_squared(point, a, b)
            .min(segment_distance_squared(point, b, c))
            .min(segment_distance_squared(point, c, a))
    }
}

fn segment_distance_squared(point: V3, a: V3, b: V3) -> f64 {
    let direction = b - a;
    let length_squared = direction.length_squared();
    let fraction = if length_squared > 0.0 {
        ((point - a).dot(direction) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (point - (a + direction * fraction)).length_squared()
}

#[derive(Debug)]
struct Node {
    bounds: Bounds,
    start: usize,
    count: usize,
    children: Option<[usize; 2]>,
}

#[derive(Debug)]
pub(super) struct TriangleEnvelope {
    triangles: Vec<Triangle>,
    order: Vec<usize>,
    nodes: Vec<Node>,
    radius: f64,
}

impl TriangleEnvelope {
    pub fn new(triangles: Vec<Triangle>, radius: f64) -> Self {
        let order = (0..triangles.len()).collect();
        let mut result = Self { triangles, order, nodes: Vec::new(), radius };
        result.build(0, result.order.len());
        result
    }

    pub fn bounds(&self) -> Bounds {
        self.nodes[0].bounds.expand(self.radius)
    }

    pub fn value(&self, point: V3) -> f64 {
        let mut closest = f64::INFINITY;
        self.nearest(0, point, &mut closest);
        closest.sqrt() - self.radius
    }

    fn build(&mut self, start: usize, count: usize) -> usize {
        let mut bounds = Bounds::empty();
        for &index in &self.order[start..start + count] {
            bounds.include(self.triangles[index].bounds);
        }
        let node = self.nodes.len();
        self.nodes.push(Node { bounds, start, count, children: None });
        if count > 4 {
            let span = bounds.max - bounds.min;
            let axis = if span.x >= span.y && span.x >= span.z {
                0
            } else if span.y >= span.z {
                1
            } else {
                2
            };
            self.order[start..start + count].sort_by(|&a, &b| {
                self.triangles[a]
                    .centroid()
                    .axis(axis)
                    .total_cmp(&self.triangles[b].centroid().axis(axis))
                    .then_with(|| a.cmp(&b))
            });
            let middle = count / 2;
            let left = self.build(start, middle);
            let right = self.build(start + middle, count - middle);
            self.nodes[node].children = Some([left, right]);
        }
        node
    }

    fn nearest(&self, index: usize, point: V3, closest: &mut f64) {
        let node = &self.nodes[index];
        if node.bounds.distance_squared(point) > *closest {
            return;
        }
        if let Some([a, b]) = node.children {
            let da = self.nodes[a].bounds.distance_squared(point);
            let db = self.nodes[b].bounds.distance_squared(point);
            let (first, second) = if da <= db { (a, b) } else { (b, a) };
            self.nearest(first, point, closest);
            self.nearest(second, point, closest);
        } else {
            for &triangle in &self.order[node.start..node.start + node.count] {
                *closest = closest.min(self.triangles[triangle].distance_squared(point));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_distance_handles_face_edge_vertex_and_degenerate_limits() {
        let triangle = Triangle::new([V3::ZERO, V3::new(2.0, 0.0, 0.0), V3::new(0.0, 2.0, 0.0)]);
        assert!((triangle.distance_squared(V3::new(0.5, 0.5, 3.0)) - 9.0).abs() < 1e-14);
        assert!((triangle.distance_squared(V3::new(1.0, -2.0, 0.0)) - 4.0).abs() < 1e-14);
        assert!((triangle.distance_squared(V3::new(-1.0, -1.0, 0.0)) - 2.0).abs() < 1e-14);
        let line = Triangle::new([V3::ZERO, V3::new(2.0, 0.0, 0.0), V3::new(1.0, 0.0, 0.0)]);
        assert_eq!(line.distance_squared(V3::new(1.0, 2.0, 0.0)), 4.0);
        assert_eq!(Triangle::new([V3::ZERO; 3]).distance_squared(V3::new(0.0, 0.0, 3.0)), 9.0);
    }

    #[test]
    fn accelerated_triangle_queries_match_brute_force() {
        let triangles: Vec<_> = (0..23)
            .map(|index| {
                let t = f64::from(index) * 0.27;
                Triangle::new([
                    V3::new(t.sin(), t.cos(), t * 0.13),
                    V3::new(t.cos(), -t.sin(), -t * 0.07),
                    V3::new(0.0, t.sin() * 0.4, 0.7),
                ])
            })
            .collect();
        let envelope = TriangleEnvelope::new(triangles.clone(), 0.4);
        for index in 0..101 {
            let t = f64::from(index) * 0.37;
            let point = V3::new(2.0 * t.sin(), 1.7 * t.cos(), (t * 0.3).sin());
            let exact = triangles
                .iter()
                .map(|triangle| triangle.distance_squared(point))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
                - 0.4;
            assert!((envelope.value(point) - exact).abs() < 1e-13);
        }
    }
}
