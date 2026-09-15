//! Deterministically ordered surface contact with swept bounding volumes.
//!
//! Conservative advancement has a bounded iteration count. An unresolved query
//! is treated as unsafe, so the caller can retry a shorter interval. This is not
//! an exact-arithmetic collision guarantee for arbitrary degenerate meshes.

use super::V3;

const CANDIDATE_LIMIT: usize = 1_000_000;
const QUERY_ITERATIONS: usize = 256;

#[derive(Clone, Copy, Debug)]
enum Pair {
    VertexTriangle { vertex: usize, triangle: [usize; 3] },
    EdgeEdge { first: [usize; 2], second: [usize; 2] },
}

struct Geometry {
    vertices: [usize; 4],
    weights: [f64; 4],
    separation: V3,
    fallback_normal: V3,
}

impl Pair {
    fn geometry(self, positions: &[V3]) -> Geometry {
        match self {
            Self::VertexTriangle { vertex, triangle: [a, b, c] } => {
                let weights =
                    closest_triangle(positions[vertex], positions[a], positions[b], positions[c]);
                let closest = positions[a] * weights[0]
                    + positions[b] * weights[1]
                    + positions[c] * weights[2];
                Geometry {
                    vertices: [vertex, a, b, c],
                    weights: [1.0, -weights[0], -weights[1], -weights[2]],
                    separation: positions[vertex] - closest,
                    fallback_normal: (positions[b] - positions[a])
                        .cross(positions[c] - positions[a])
                        .normalized(),
                }
            }
            Self::EdgeEdge { first: [a, b], second: [c, d] } => {
                let (s, t) =
                    closest_segments(positions[a], positions[b], positions[c], positions[d]);
                let first = positions[a].lerp(positions[b], s);
                let second = positions[c].lerp(positions[d], t);
                let direction = positions[b] - positions[a];
                let mut normal = direction.cross(positions[d] - positions[c]).normalized();
                if normal.length_squared() < 0.5 {
                    let axis = if direction.x.abs() < direction.y.abs() {
                        V3::new(1.0, 0.0, 0.0)
                    } else {
                        V3::new(0.0, 1.0, 0.0)
                    };
                    normal = direction.cross(axis).normalized();
                }
                Geometry {
                    vertices: [a, b, c, d],
                    weights: [1.0 - s, s, t - 1.0, -t],
                    separation: first - second,
                    fallback_normal: normal,
                }
            }
        }
    }

    fn at_time(self, old: &[V3], next: &[V3], time: f64) -> Geometry {
        // Evaluate only the four participating points instead of allocating a mesh.
        let vertices = match self {
            Self::VertexTriangle { vertex, triangle: [a, b, c] } => [vertex, a, b, c],
            Self::EdgeEdge { first: [a, b], second: [c, d] } => [a, b, c, d],
        };
        let points = vertices.map(|index| old[index].lerp(next[index], time));
        match self {
            Self::VertexTriangle { .. } => {
                Self::VertexTriangle { vertex: 0, triangle: [1, 2, 3] }.geometry(&points)
            }
            Self::EdgeEdge { .. } => {
                Self::EdgeEdge { first: [0, 1], second: [2, 3] }.geometry(&points)
            }
        }
    }

    fn movement_bound(self, old: &[V3], next: &[V3]) -> f64 {
        match self {
            Self::VertexTriangle { vertex, triangle: [a, b, c] } => {
                let common = ((next[a] - old[a]) + (next[b] - old[b]) + (next[c] - old[c])) / 3.0;
                let motion = |index: usize| (next[index] - old[index] - common).length();
                motion(vertex) + motion(a).max(motion(b)).max(motion(c))
            }
            Self::EdgeEdge { first: [a, b], second: [c, d] } => {
                let common = ((next[a] - old[a])
                    + (next[b] - old[b])
                    + (next[c] - old[c])
                    + (next[d] - old[d]))
                    * 0.25;
                let motion = |index: usize| (next[index] - old[index] - common).length();
                motion(a).max(motion(b)) + motion(c).max(motion(d))
            }
        }
    }
}

pub(super) struct Contact {
    vertices: [usize; 4],
    weights: [f64; 4],
    normal: V3,
    lambda: f64,
}

impl Contact {
    pub(super) fn solve(
        &mut self,
        next: &mut [V3],
        old: &[V3],
        inverse_mass: &[f64],
        thickness: f64,
        friction: f64,
    ) {
        // Preserve the material contact points and their separating plane for
        // this linearized solve. Recomputing nearest points after tunneling can
        // otherwise switch features and incorrectly accept the opposite side.
        let normal = self.normal;
        let separation = self
            .vertices
            .iter()
            .zip(self.weights)
            .fold(V3::ZERO, |sum, (&index, weight)| sum + next[index] * weight);
        let error = separation.dot(normal) - thickness;
        let denominator = self
            .vertices
            .iter()
            .zip(self.weights)
            .fold(0.0, |sum, (&index, weight)| sum + inverse_mass[index] * weight * weight);
        if denominator < 1e-12 {
            return;
        }
        let new_lambda = (self.lambda - error / denominator).max(0.0);
        let change = new_lambda - self.lambda;
        self.lambda = new_lambda;
        self.normal = normal;
        for (&index, weight) in self.vertices.iter().zip(self.weights) {
            next[index] += normal * (inverse_mass[index] * weight * change);
        }
        if friction > 0.0 && change > 0.0 {
            let displacement =
                self.vertices.iter().zip(self.weights).fold(V3::ZERO, |sum, (&index, weight)| {
                    sum + (next[index] - old[index]) * weight
                });
            let tangent = displacement - normal * displacement.dot(normal);
            let length = tangent.length();
            if length > 1e-12 {
                let correction = tangent / length * (length / denominator).min(friction * change);
                for (&index, weight) in self.vertices.iter().zip(self.weights) {
                    next[index] -= correction * (inverse_mass[index] * weight);
                }
            }
        }
    }
}

pub(super) fn generate(
    old: &[V3],
    next: &[V3],
    triangles: &[[u32; 3]],
    edges: &[[usize; 2]],
    _inverse_mass: &[f64],
    thickness: f64,
) -> Result<Vec<Contact>, &'static str> {
    let pairs = candidates(old, next, triangles, edges, thickness * 1.5)?;
    let mut contacts = Vec::new();
    for pair in pairs {
        let current = pair.geometry(next);
        let event = first_approach(pair, old, next, thickness);
        if current.separation.length() <= thickness * 1.5 || event.is_some() {
            let vertices = current.vertices;
            let geometry = event.map_or(current, |time| pair.at_time(old, next, time));
            let mut normal = geometry.separation.normalized();
            if normal.length_squared() < 0.5 {
                normal = geometry.fallback_normal;
            }
            if normal.length_squared() < 0.5 {
                return Err("degenerate contact normal");
            }
            contacts.push(Contact { vertices, weights: geometry.weights, normal, lambda: 0.0 });
        }
    }
    Ok(contacts)
}

pub(super) fn safe_step(
    old: &[V3],
    next: &[V3],
    triangles: &[[u32; 3]],
    edges: &[[usize; 2]],
    _inverse_mass: &[f64],
    thickness: f64,
) -> Result<bool, &'static str> {
    // A smaller verification gap tolerates finite solver residual while still
    // rejecting an actual mid-surface crossing before it can persist as a tangle.
    let gap = thickness * 0.12;
    for pair in candidates(old, next, triangles, edges, gap)? {
        if approaches(pair, old, next, gap) {
            if std::env::var_os("SILK_CONTACT_DIAGNOSTIC").is_some() {
                eprintln!(
                    "silk contact diagnostic: {pair:?}; gap={gap:.9}; initial={:.9}; final={:.9}; relative_bound={:.9}",
                    pair.at_time(old, next, 0.0).separation.length(),
                    pair.at_time(old, next, 1.0).separation.length(),
                    pair.movement_bound(old, next)
                );
                if let Some(time) = first_approach(pair, old, next, gap) {
                    let mut minimum = f64::INFINITY;
                    for sample in 0..=1000 {
                        minimum = minimum.min(
                            pair.at_time(old, next, f64::from(sample) / 1000.0).separation.length(),
                        );
                    }
                    eprintln!(
                        "silk contact event: time={time:.9}; event_distance={:.9}; sampled_min={minimum:.9}",
                        pair.at_time(old, next, time).separation.length()
                    );
                }
            }
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn has_clearance(
    positions: &[V3],
    triangles: &[[u32; 3]],
    edges: &[[usize; 2]],
    gap: f64,
) -> Result<bool, &'static str> {
    for pair in candidates(positions, positions, triangles, edges, gap)? {
        if pair.geometry(positions).separation.length() < gap {
            return Ok(false);
        }
    }
    Ok(true)
}

fn approaches(pair: Pair, old: &[V3], next: &[V3], gap: f64) -> bool {
    first_approach(pair, old, next, gap).is_some()
}

fn first_approach(pair: Pair, old: &[V3], next: &[V3], gap: f64) -> Option<f64> {
    let bound = pair.movement_bound(old, next);
    let initial = pair.at_time(old, next, 0.0).separation.length();
    let final_distance = pair.at_time(old, next, 1.0).separation.length();
    if initial <= gap {
        return Some(0.0);
    }
    if bound <= 1e-14 || initial.min(final_distance) - bound > gap {
        return None;
    }
    let mut time = 0.0;
    for _ in 0..QUERY_ITERATIONS {
        let distance = pair.at_time(old, next, time).separation.length();
        if !distance.is_finite() || distance <= gap * 1.000_01 {
            return Some(time);
        }
        // Distance between two moving convex primitives is Lipschitz-bounded
        // by their maximum point speeds. Stay short of its conservative bound.
        let increment = 0.8 * (distance - gap) / bound;
        if time + increment >= 1.0 {
            return None;
        }
        if increment < 1e-10 {
            return Some(time);
        }
        time += increment;
    }
    // Failure to establish clearance is not silently accepted.
    Some(time)
}

#[derive(Clone, Copy)]
struct Bounds {
    low: V3,
    high: V3,
}

impl Bounds {
    fn empty() -> Self {
        Self {
            low: V3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            high: V3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    fn include(mut self, point: V3) -> Self {
        self.low = self.low.min(point);
        self.high = self.high.max(point);
        self
    }
    fn merge(self, other: Self) -> Self {
        Self { low: self.low.min(other.low), high: self.high.max(other.high) }
    }
    fn expanded(self, margin: f64) -> Self {
        let offset = V3::new(margin, margin, margin);
        Self { low: self.low - offset, high: self.high + offset }
    }
    fn overlaps(self, other: Self) -> bool {
        self.low.x <= other.high.x
            && self.high.x >= other.low.x
            && self.low.y <= other.high.y
            && self.high.y >= other.low.y
            && self.low.z <= other.high.z
            && self.high.z >= other.low.z
    }
    fn center(self) -> V3 {
        (self.low + self.high) * 0.5
    }
}

struct Node {
    bounds: Bounds,
    children: Option<[usize; 2]>,
    range: std::ops::Range<usize>,
}
struct Bvh {
    nodes: Vec<Node>,
    order: Vec<usize>,
}

impl Bvh {
    fn new(bounds: &[Bounds]) -> Self {
        let mut tree = Self { nodes: Vec::new(), order: (0..bounds.len()).collect() };
        if !bounds.is_empty() {
            tree.build(bounds, 0, bounds.len());
        }
        tree
    }

    fn build(&mut self, bounds: &[Bounds], start: usize, end: usize) -> usize {
        let combined = self.order[start..end]
            .iter()
            .fold(Bounds::empty(), |sum, &index| sum.merge(bounds[index]));
        let node_index = self.nodes.len();
        self.nodes.push(Node { bounds: combined, children: None, range: start..end });
        if end - start > 8 {
            let size = combined.high - combined.low;
            let axis = if size.x >= size.y && size.x >= size.z {
                0
            } else if size.y >= size.z {
                1
            } else {
                2
            };
            self.order[start..end].sort_by(|&a, &b| {
                bounds[a]
                    .center()
                    .axis(axis)
                    .total_cmp(&bounds[b].center().axis(axis))
                    .then(a.cmp(&b))
            });
            let middle = start + (end - start) / 2;
            let first = self.build(bounds, start, middle);
            let second = self.build(bounds, middle, end);
            self.nodes[node_index].children = Some([first, second]);
        }
        node_index
    }

    fn query(&self, bounds: Bounds, matches: &mut Vec<usize>) {
        matches.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0];
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            if !bounds.overlaps(node.bounds) {
                continue;
            }
            if let Some([first, second]) = node.children {
                stack.push(second);
                stack.push(first);
            } else {
                matches.extend_from_slice(&self.order[node.range.clone()]);
            }
        }
        matches.sort_unstable();
    }
}

fn swept_bounds(indices: &[usize], old: &[V3], next: &[V3]) -> Bounds {
    indices
        .iter()
        .fold(Bounds::empty(), |bounds, &index| bounds.include(old[index]).include(next[index]))
}

fn candidates(
    old: &[V3],
    next: &[V3],
    triangles: &[[u32; 3]],
    edges: &[[usize; 2]],
    margin: f64,
) -> Result<Vec<Pair>, &'static str> {
    let mut pairs = Vec::new();
    let mut used = vec![false; old.len()];
    let triangle_bounds: Vec<_> = triangles
        .iter()
        .map(|triangle| {
            let indices = triangle.map(|index| index as usize);
            for index in indices {
                used[index] = true;
            }
            swept_bounds(&indices, old, next)
        })
        .collect();
    let tree = Bvh::new(&triangle_bounds);
    let mut matches = Vec::new();
    for (vertex, &connected) in used.iter().enumerate() {
        if !connected {
            continue;
        }
        let bounds = swept_bounds(&[vertex], old, next).expanded(margin);
        tree.query(bounds, &mut matches);
        for &index in &matches {
            let triangle = triangles[index].map(|id| id as usize);
            if !triangle.contains(&vertex) && bounds.overlaps(triangle_bounds[index]) {
                pairs.push(Pair::VertexTriangle { vertex, triangle });
            }
        }
        if pairs.len() > CANDIDATE_LIMIT {
            return Err("self-contact candidate budget exceeded");
        }
    }
    let edge_bounds: Vec<_> = edges.iter().map(|edge| swept_bounds(edge, old, next)).collect();
    let tree = Bvh::new(&edge_bounds);
    for (index, &first) in edges.iter().enumerate() {
        let bounds = edge_bounds[index].expanded(margin);
        tree.query(bounds, &mut matches);
        for &other in &matches {
            let second = edges[other];
            if index < other
                && !first.iter().any(|vertex| second.contains(vertex))
                && bounds.overlaps(edge_bounds[other])
            {
                pairs.push(Pair::EdgeEdge { first, second });
            }
        }
        if pairs.len() > CANDIDATE_LIMIT {
            return Err("self-contact candidate budget exceeded");
        }
    }
    Ok(pairs)
}

fn closest_triangle(point: V3, a: V3, b: V3, c: V3) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3).max(1e-30);
        return [1.0 - v, v, 0.0];
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6).max(1e-30);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(1e-30);
        return [0.0, 1.0 - w, w];
    }
    let sum = va + vb + vc;
    if sum.abs() < 1e-24 {
        // A degenerate face has no stable plane: use its closest segment.
        let mut best = ([1.0, 0.0, 0.0], (point - a).length_squared());
        for (start, end, first, second) in [(a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)] {
            let edge = end - start;
            let t = if edge.length_squared() > 1e-24 {
                ((point - start).dot(edge) / edge.length_squared()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let distance = (point - start.lerp(end, t)).length_squared();
            if distance < best.1 {
                let mut weights = [0.0; 3];
                weights[first] = 1.0 - t;
                weights[second] = t;
                best = (weights, distance);
            }
        }
        return best.0;
    }
    let v = vb / sum;
    let w = vc / sum;
    [1.0 - v - w, v, w]
}

fn closest_segments(a: V3, b: V3, c: V3, d: V3) -> (f64, f64) {
    let first = b - a;
    let second = d - c;
    let offset = a - c;
    let aa = first.dot(first);
    let ee = second.dot(second);
    let ff = second.dot(offset);
    if aa <= 1e-24 && ee <= 1e-24 {
        return (0.0, 0.0);
    }
    if aa <= 1e-24 {
        return (0.0, (ff / ee).clamp(0.0, 1.0));
    }
    let cc = first.dot(offset);
    if ee <= 1e-24 {
        return ((-cc / aa).clamp(0.0, 1.0), 0.0);
    }
    let bb = first.dot(second);
    let denominator = aa * ee - bb * bb;
    let mut s =
        if denominator > 1e-24 { ((bb * ff - cc * ee) / denominator).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = (bb * s + ff) / ee;
    if t < 0.0 {
        t = 0.0;
        s = (-cc / aa).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((bb - cc) / aa).clamp(0.0, 1.0);
    }
    (s, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swept_vertex_triangle_detects_tunneling() {
        let old =
            [V3::new(0.2, 0.2, 1.0), V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)];
        let mut next = old;
        next[0].z = -1.0;
        let pair = Pair::VertexTriangle { vertex: 0, triangle: [1, 2, 3] };
        assert!(approaches(pair, &old, &next, 0.01));
        assert!(pair.geometry(&next).separation.length() > 0.9);
    }

    #[test]
    fn edge_edge_detects_crossing_between_vertices() {
        let old = [
            V3::new(-1.0, 0.0, 0.2),
            V3::new(1.0, 0.0, 0.2),
            V3::new(0.0, -1.0, 0.0),
            V3::new(0.0, 1.0, 0.0),
        ];
        let mut next = old;
        next[0].z = -0.2;
        next[1].z = -0.2;
        assert!(approaches(Pair::EdgeEdge { first: [0, 1], second: [2, 3] }, &old, &next, 0.001));
    }

    #[test]
    fn separating_surfaces_remain_safe() {
        let old =
            [V3::new(0.2, 0.2, 0.1), V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)];
        let mut next = old;
        next[0].z = 0.3;
        assert!(!approaches(
            Pair::VertexTriangle { vertex: 0, triangle: [1, 2, 3] },
            &old,
            &next,
            0.01
        ));
    }

    #[test]
    fn conservative_query_ignores_common_rigid_translation() {
        let old =
            [V3::new(0.2, 0.2, 0.002), V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)];
        let next = old.map(|point| point + V3::new(30.0, -17.0, 4.0));
        assert!(!approaches(
            Pair::VertexTriangle { vertex: 0, triangle: [1, 2, 3] },
            &old,
            &next,
            0.001
        ));
    }

    #[test]
    fn vertex_contact_moves_the_free_vertex_without_moving_a_pinned_face() {
        let old =
            [V3::new(0.2, 0.2, 0.05), V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)];
        let mut next = old;
        next[0].z = -0.01;
        let mut contact = Contact {
            vertices: [0, 1, 2, 3],
            weights: [1.0, -0.6, -0.2, -0.2],
            normal: V3::new(0.0, 0.0, 1.0),
            lambda: 0.0,
        };
        contact.solve(&mut next, &old, &[1.0, 0.0, 0.0, 0.0], 0.02, 0.0);
        assert!((next[0].z - 0.02).abs() < 1e-12);
        assert_eq!(&next[1..], &old[1..]);
    }
}
