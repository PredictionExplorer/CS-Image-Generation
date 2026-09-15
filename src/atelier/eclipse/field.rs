//! Corrected contour distances and jointly occluded luminous petal fields.
//!
//! Fixed contour brackets seed safeguarded stationary-point solves. A convex
//! outside normal foot certifies the global closest point; remaining cases use
//! conservative chord/curvature bounds instead of an implicit-distance proxy.

use super::{EclipseConfig, EclipsePetalConfig, Point};
use crate::atelier::V3;
use std::f64::consts::TAU;

const CONTOUR_INTERVALS: usize = 32;
const DISTANCE_TOLERANCE: f64 = 2e-6;
const ROOT_ITERATIONS: usize = 28;

/// World-space contour position and its oriented unit frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct ContourSample {
    /// Position on the unchanged identified petal.
    pub position: Point,
    /// Unit outward normal.
    pub normal: Point,
    /// Unit tangent in the increasing-theta direction.
    pub tangent: Point,
}

#[derive(Clone, Copy, Debug, Default)]
struct Jet {
    point: Point,
    first: Point,
    second: Point,
}

#[derive(Clone, Copy)]
struct Closest {
    jet: Jet,
    distance: f64,
}

/// One regular bent petal, prepared once per source exposure.
#[derive(Clone, Copy, Debug)]
pub(super) struct Petal {
    /// Fixed material and shape recipe.
    pub shape: EclipsePetalConfig,
    /// Actual current source-driven world center.
    pub center: Point,
    /// Current short/long semi-axes after the bounded source breathing.
    pub semi_axes: Point,
    cosine: f64,
    sine: f64,
    extent: Point,
    nodes: [Jet; CONTOUR_INTERVALS + 1],
    second_derivative_bound: f64,
    convex: bool,
    inverse_width_floor: f64,
    normalized_x_bound: f64,
    inside_lipschitz: f64,
}

impl Petal {
    /// Prepare the fixed contour roots and conservative geometric bounds.
    pub fn new(shape: EclipsePetalConfig, center: Point, long_axis_factor: f64) -> Self {
        let semi_axes = [shape.semi_axes[0], shape.semi_axes[1] * long_axis_factor];
        let (sine, cosine) = shape.angle_degrees.to_radians().sin_cos();
        let [a, b] = semi_axes;
        let shoulder = shape.shoulder.abs();
        let shear = shape.shear.abs();
        // |tanh(sin(theta))*cos(theta)| <= |sin(theta)*cos(theta)| <= 1/2.
        let lateral = a * (0.5 * shoulder + shear);
        let extent = [
            (a * cosine).hypot(b * sine) + cosine.abs() * lateral,
            (a * sine).hypot(b * cosine) + sine.abs() * lateral,
        ];
        let inverse_width_floor = 1.0 - shoulder;
        let normalized_x_bound = 1.0 + 0.5 * shoulder + shear;
        let z_bound = (normalized_x_bound + shear) / inverse_width_floor;
        let mixed = (2.0 * shear + shoulder * z_bound) / (b * inverse_width_floor);
        let inside_lipschitz = (1.0 / (a * inverse_width_floor)).hypot(mixed).hypot(1.0 / b);
        let mut result = Self {
            shape,
            center,
            semi_axes,
            cosine,
            sine,
            extent,
            nodes: [Jet::default(); CONTOUR_INTERVALS + 1],
            second_derivative_bound: (a * (1.0 + 6.0 * shoulder + 2.0 * shear)).hypot(b),
            // det(C',C'')/(a*b) >= 1 - (1+8/(3sqrt(3)))*|shoulder| - 2|shear|.
            // The slightly larger 2.54 coefficient is a conservative simplification.
            convex: 1.0 - 2.54 * shoulder - 2.0 * shear > 1e-12,
            inverse_width_floor,
            normalized_x_bound,
            inside_lipschitz,
        };
        result.nodes =
            std::array::from_fn(|i| result.local_jet(TAU * i as f64 / CONTOUR_INTERVALS as f64));
        result.nodes[CONTOUR_INTERVALS] = result.nodes[0];
        result
    }

    fn local_jet(&self, theta: f64) -> Jet {
        let (s, c) = theta.sin_cos();
        let h = s.tanh();
        let d = 1.0 - h * h;
        let eta = self.shape.shoulder;
        let kappa = self.shape.shear;
        let width = 1.0 + eta * h;
        let [a, b] = self.semi_axes;
        Jet {
            point: [a * (width * c + kappa * c * c), b * s],
            first: [a * (eta * d * c * c - width * s - 2.0 * kappa * s * c), b * c],
            second: [
                a * (-2.0 * eta * h * d * c.powi(3)
                    - 3.0 * eta * d * c * s
                    - width * c
                    - 2.0 * kappa * (c * c - s * s)),
                -b * s,
            ],
        }
    }

    fn local(&self, point: Point) -> Point {
        let x = point[0] - self.center[0];
        let y = point[1] - self.center[1];
        [self.cosine * x + self.sine * y, -self.sine * x + self.cosine * y]
    }

    fn rotate(&self, local: Point) -> Point {
        [
            self.cosine * local[0] - self.sine * local[1],
            self.sine * local[0] + self.cosine * local[1],
        ]
    }

    fn implicit_radius(&self, local: Point) -> f64 {
        let u = local[0] / self.semi_axes[0];
        let v = local[1] / self.semi_axes[1];
        let z = (u - self.shape.shear * (1.0 - v * v)) / (1.0 + self.shape.shoulder * v.tanh());
        z.hypot(v)
    }

    /// Evaluate the smooth contour and its world-space unit normal/tangent.
    pub fn contour(&self, theta: f64) -> ContourSample {
        let jet = self.local_jet(theta);
        let offset = self.rotate(jet.point);
        let tangent = self.rotate(normalize(jet.first));
        ContourSample {
            position: add(self.center, offset),
            normal: [tangent[1], -tangent[0]],
            tangent,
        }
    }

    /// Conservative world-space lower/upper bounds, expanded by nonnegative padding.
    pub fn bounds(&self, padding: f64) -> [Point; 2] {
        [
            std::array::from_fn(|a| self.center[a] - self.extent[a] - padding),
            std::array::from_fn(|a| self.center[a] + self.extent[a] + padding),
        ]
    }

    /// Conservative sign and distance magnitude used only for constant-mask decisions.
    /// It never substitutes for the distance shaping a visible crescent.
    fn distance_lower_bound(&self, point: Point) -> (bool, f64) {
        let local = self.local(point);
        let radius = self.implicit_radius(local);
        if radius < 1.0 {
            return (true, (1.0 - radius) / self.inside_lipschitz);
        }
        // Bound the inverse map's Jacobian on the rectangle containing the
        // query and the entire contour. Integrating along the straight segment
        // to any boundary point proves distance >= |inverse_radius-1|/L.
        let [a, b] = self.semi_axes;
        let u = (local[0] / a).abs().max(self.normalized_x_bound);
        let v = (local[1] / b).abs().max(1.0);
        let shear = self.shape.shear.abs();
        let shoulder = self.shape.shoulder.abs();
        let z = (u + shear * (1.0 + v * v)) / self.inverse_width_floor;
        let mixed = (2.0 * shear * v + shoulder * z) / (b * self.inverse_width_floor);
        let lipschitz = (1.0 / (a * self.inverse_width_floor)).hypot(mixed).hypot(1.0 / b);
        let box_distance = ((point[0] - self.center[0]).abs() - self.extent[0])
            .max(0.0)
            .hypot(((point[1] - self.center[1]).abs() - self.extent[1]).max(0.0));
        (false, ((radius - 1.0) / lipschitz).max(box_distance))
    }

    fn stationary_root(&self, point: Point, mut low: f64, mut high: f64) -> Jet {
        let mut theta = f64::midpoint(low, high);
        let mut value = self.local_jet(theta);
        for _ in 0..ROOT_ITERATIONS {
            let delta = subtract(value.point, point);
            let gradient = dot(delta, value.first);
            if gradient == 0.0 || high - low < 1e-12 {
                break;
            }
            if gradient < 0.0 {
                low = theta;
            } else {
                high = theta;
            }
            let curvature = dot(value.first, value.first) + dot(delta, value.second);
            let next = theta - gradient / curvature;
            theta = if curvature > 0.0 && next > low && next < high && next.is_finite() {
                next
            } else {
                f64::midpoint(low, high)
            };
            value = self.local_jet(theta);
        }
        value
    }

    fn update_closest(point: Point, jet: Jet, closest: &mut Closest) {
        let delta = subtract(jet.point, point);
        let distance = delta[0].hypot(delta[1]);
        if distance < closest.distance {
            *closest = Closest { jet, distance };
        }
    }

    fn outside_foot(&self, point: Point, jet: Jet, tolerance: f64) -> bool {
        let tangent = normalize(jet.first);
        let offset = subtract(point, jet.point);
        dot(offset, [tangent[1], -tangent[0]]) >= 0.0
            && dot(offset, tangent).abs()
                < (0.25 * tolerance).min(1e-10 * (1.0 + offset[0].hypot(offset[1])))
    }

    fn refine_arc(
        &self,
        point: Point,
        low: f64,
        high: f64,
        ends: [Jet; 2],
        closest: &mut Closest,
        tolerance: f64,
    ) {
        let (chord_distance, fraction) = segment_distance(point, ends[0].point, ends[1].point);
        let error = self.second_derivative_bound * (high - low).powi(2) / 8.0;
        if (chord_distance - error).max(0.0) >= closest.distance - tolerance {
            return;
        }
        if fraction > 0.0 && fraction < 1.0 {
            Self::update_closest(point, self.local_jet(low + (high - low) * fraction), closest);
        }
        let mid = f64::midpoint(low, high);
        if mid == low || mid == high {
            return;
        }
        let middle = self.local_jet(mid);
        Self::update_closest(point, middle, closest);
        self.refine_arc(point, low, mid, [ends[0], middle], closest, tolerance);
        self.refine_arc(point, mid, high, [middle, ends[1]], closest, tolerance);
    }

    /// Signed closest-contour distance and the corresponding outward world normal.
    /// Arc-search tolerance is at most 2e-6 world units and one millionth of
    /// the luminous width, subject to an outward floating-point scale floor.
    pub fn distance_normal(&self, point: Point) -> (f64, Point) {
        self.distance_normal_with_tolerance(point, self.shape.light_sigma * 1e-6)
    }

    fn distance_normal_with_tolerance(&self, point: Point, requested: f64) -> (f64, Point) {
        let local = self.local(point);
        let floor = 64.0
            * f64::EPSILON
            * (1.0 + local[0].abs().max(local[1].abs()) + self.semi_axes[0].max(self.semi_axes[1]));
        let tolerance = requested.min(DISTANCE_TOLERANCE).max(floor);
        let outside = self.implicit_radius(local) >= 1.0;
        // The original convex outside return never uses the nearest-node
        // prepass. Try its identical ordered roots first, retaining the whole
        // original algorithm if no normal foot receives that certificate.
        if self.convex
            && outside
            && let Some(projection) = self.certified_outside_projection(local, tolerance)
        {
            return projection;
        }
        self.distance_normal_original(local, tolerance, outside)
    }

    fn certified_outside_projection(&self, local: Point, tolerance: f64) -> Option<(f64, Point)> {
        let step = TAU / CONTOUR_INTERVALS as f64;
        for index in 0..CONTOUR_INTERVALS {
            let a = self.nodes[index];
            let b = self.nodes[index + 1];
            let ga = dot(subtract(a.point, local), a.first);
            let gb = dot(subtract(b.point, local), b.first);
            if ga <= 0.0 && gb >= 0.0 {
                let jet = if ga == 0.0 {
                    a
                } else if gb == 0.0 {
                    b
                } else {
                    self.stationary_root(local, index as f64 * step, (index + 1) as f64 * step)
                };
                if self.outside_foot(local, jet, tolerance) {
                    let tangent = normalize(jet.first);
                    let delta = subtract(local, jet.point);
                    return Some((
                        delta[0].hypot(delta[1]),
                        self.rotate([tangent[1], -tangent[0]]),
                    ));
                }
            }
        }
        None
    }

    fn distance_normal_original(
        &self,
        local: Point,
        tolerance: f64,
        outside: bool,
    ) -> (f64, Point) {
        let mut closest = Closest { jet: self.nodes[0], distance: f64::INFINITY };
        for &jet in self.nodes.iter().take(CONTOUR_INTERVALS) {
            Self::update_closest(local, jet, &mut closest);
        }
        let step = TAU / CONTOUR_INTERVALS as f64;
        for index in 0..CONTOUR_INTERVALS {
            let a = self.nodes[index];
            let b = self.nodes[index + 1];
            let ga = dot(subtract(a.point, local), a.first);
            let gb = dot(subtract(b.point, local), b.first);
            if ga <= 0.0 && gb >= 0.0 {
                let jet = if ga == 0.0 {
                    a
                } else if gb == 0.0 {
                    b
                } else {
                    self.stationary_root(local, index as f64 * step, (index + 1) as f64 * step)
                };
                Self::update_closest(local, jet, &mut closest);
                // On a strictly convex contour, an outward normal foot is the
                // unique metric projection. Its tiny tangential residual is
                // checked before accepting this inexpensive outside path.
                if self.convex && outside && self.outside_foot(local, jet, tolerance) {
                    let tangent = normalize(jet.first);
                    let delta = subtract(local, jet.point);
                    return (delta[0].hypot(delta[1]), self.rotate([tangent[1], -tangent[0]]));
                }
            }
        }
        // The analytic bound on |C''| encloses each arc within its chord plus
        // M*delta_theta^2/8. Branches farther than the best known foot can be
        // discarded; medial ties and unusual nonconvex shapes remain valid.
        for index in 0..CONTOUR_INTERVALS {
            self.refine_arc(
                local,
                index as f64 * step,
                (index + 1) as f64 * step,
                [self.nodes[index], self.nodes[index + 1]],
                &mut closest,
                tolerance,
            );
        }
        let tangent = normalize(closest.jet.first);
        let normal = self.rotate([tangent[1], -tangent[0]]);
        (if outside { closest.distance } else { -closest.distance }, normal)
    }
}

/// Prove that transmission is constant throughout a closed world-space ball.
///
/// Signed distance is 1-Lipschitz. Subtracting the ball radius from a proven
/// center clearance therefore bounds every point without evaluating or
/// approximating its light. `None` retains ordinary per-point integration.
pub(super) fn constant_transmission(
    center: Point,
    radius: f64,
    petals: &[Petal; 3],
    config: &EclipseConfig,
) -> Option<f64> {
    if !valid_ball(center, radius) {
        return None;
    }
    let count = petals.iter().filter(|petal| petal.shape.enabled).count();
    if count == 0 {
        return Some(1.0);
    }
    let dark_clearance = config.edge_width.next_up();
    let clear_clearance = (config.edge_width + config.smooth_join * (count as f64).ln()).next_up();
    let mut all_clear = true;
    for petal in petals.iter().filter(|petal| petal.shape.enabled) {
        let (inside, clearance) = ball_clearance(center, radius, petal);
        if inside && clearance.is_finite() && clearance >= dark_clearance {
            return Some(0.0);
        }
        if inside || clearance < clear_clearance || !clearance.is_finite() {
            all_clear = false;
        }
    }
    all_clear.then_some(1.0)
}

/// Prove that this petal emits no broad crescent light anywhere in a world ball.
/// The unsigned distance to the shifted contour is also 1-Lipschitz, on both
/// sides of the contour. This does not classify or discard any corona segment.
pub(super) fn crescent_is_zero(center: Point, radius: f64, petal: &Petal) -> bool {
    if !petal.shape.enabled || petal.shape.light_gain == 0.0 {
        return true;
    }
    if !valid_ball(center, radius) {
        return false;
    }
    let shifted = subtract(center, petal.shape.light_offset);
    if !shifted.iter().all(|v| v.is_finite()) {
        return false;
    }
    let (_, clearance) = ball_clearance(shifted, radius, petal);
    clearance.is_finite() && clearance >= (4.0 * petal.shape.light_sigma).next_up()
}

fn valid_ball(center: Point, radius: f64) -> bool {
    radius.is_finite() && radius >= 0.0 && center.iter().all(|v| v.is_finite())
}

fn ball_clearance(center: Point, radius: f64, petal: &Petal) -> (bool, f64) {
    let (inside, lower) = petal.distance_lower_bound(center);
    // Round only toward uncertainty: a tile touching a transition keeps the
    // original shader. This covers coordinate subtraction and the arithmetic
    // of the existing Lipschitz lower bounds for validated finite geometry.
    let magnitude = 1.0
        + center[0].abs().max(center[1].abs())
        + petal.center[0].abs().max(petal.center[1].abs())
        + petal.semi_axes[0].max(petal.semi_axes[1])
        + lower.abs()
        + radius;
    let padding = 128.0 * f64::EPSILON * magnitude;
    (inside, (lower - radius - padding).next_down())
}

/// Transmission of the symmetric common opaque union, in zero to one.
pub(super) fn transmission(point: Point, petals: &[Petal; 3], config: &EclipseConfig) -> f64 {
    let count = petals.iter().filter(|petal| petal.shape.enabled).count();
    if count == 0 {
        return 1.0;
    }
    let clear_distance = config.edge_width + config.smooth_join * (count as f64).ln();
    let mut all_clear = true;
    for petal in petals.iter().filter(|petal| petal.shape.enabled) {
        let (inside, distance) = petal.distance_lower_bound(point);
        if inside && distance >= config.edge_width {
            return 0.0;
        }
        if inside || distance < clear_distance {
            all_clear = false;
        }
    }
    if all_clear {
        return 1.0;
    }
    let mut distances = [f64::INFINITY; 3];
    let mut minimum = f64::INFINITY;
    for (distance, petal) in distances.iter_mut().zip(petals) {
        if petal.shape.enabled {
            *distance = petal
                .distance_normal_with_tolerance(
                    point,
                    config.edge_width * config.integration_tolerance * 0.25,
                )
                .0;
            if *distance <= -config.edge_width {
                return 0.0;
            }
            minimum = minimum.min(*distance);
        }
    }
    let distance = if config.smooth_join > 0.0 {
        let sum = distances
            .iter()
            .filter(|d| d.is_finite())
            .map(|d| ((minimum - d) / config.smooth_join).exp())
            .sum::<f64>();
        minimum - config.smooth_join * sum.ln()
    } else {
        minimum
    };
    smoothstep(-config.edge_width, config.edge_width, distance)
}

/// Unoccluded broad pearl/rose/copper light; the renderer applies common transmission.
pub(super) fn crescent(point: Point, petal: &Petal, config: &EclipseConfig) -> V3 {
    if !petal.shape.enabled || petal.shape.light_gain == 0.0 {
        return V3::ZERO;
    }
    let shifted = subtract(point, petal.shape.light_offset);
    let support = petal.bounds(4.0 * petal.shape.light_sigma);
    if (0..2).any(|a| shifted[a] < support[0][a] || shifted[a] > support[1][a]) {
        return V3::ZERO;
    }
    let (_, lower) = petal.distance_lower_bound(shifted);
    if lower >= 4.0 * petal.shape.light_sigma {
        return V3::ZERO;
    }
    let (distance, normal) = petal.distance_normal(shifted);
    let normalized = distance / petal.shape.light_sigma;
    if normalized.abs() >= 4.0 {
        return V3::ZERO;
    }
    let direction = normalize(config.light_direction);
    let facing = dot(normal, direction).max(0.0);
    let window = 0.06 + 0.94 * facing * facing;
    let cutoff = 1.0 - smoothstep(3.0, 4.0, normalized.abs());
    let strength =
        petal.shape.light_gain * (-0.5 * normalized * normalized).exp() * window * cutoff;
    let outward = normalized.max(0.0);
    let tint = config
        .pearl
        .lerp(config.rose, smoothstep(0.45, 1.8, outward))
        .lerp(config.copper, smoothstep(2.0, 4.0, outward));
    tint * strength
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1]]
}
fn subtract(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn normalize(a: Point) -> Point {
    let length = a[0].hypot(a[1]);
    if length > 0.0 { [a[0] / length, a[1] / length] } else { [1.0, 0.0] }
}
fn segment_distance(point: Point, a: Point, b: Point) -> (f64, f64) {
    let delta = subtract(b, a);
    let length = dot(delta, delta);
    let fraction =
        if length > 0.0 { (dot(subtract(point, a), delta) / length).clamp(0.0, 1.0) } else { 0.0 };
    let offset = [point[0] - a[0] - fraction * delta[0], point[1] - a[1] - fraction * delta[1]];
    (offset[0].hypot(offset[1]), fraction)
}
fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    (t.powi(3) * (t * (t * 6.0 - 15.0) + 10.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_ball(center: Point, radius: f64, mut check: impl FnMut(Point)) {
        check(center);
        for fraction in [0.5, 1.0] {
            for index in 0..48 {
                let (s, c) = (TAU * f64::from(index) / 48.0).sin_cos();
                check([center[0] + radius * fraction * c, center[1] + radius * fraction * s]);
            }
        }
    }

    fn assert_original_distance_bits(petal: &Petal, point: Point, requested: f64) {
        // Reproduce the old entry path, which always initialized its nearest
        // node before visiting stationary roots and the global arc search.
        let local = petal.local(point);
        let floor = 64.0
            * f64::EPSILON
            * (1.0
                + local[0].abs().max(local[1].abs())
                + petal.semi_axes[0].max(petal.semi_axes[1]));
        let tolerance = requested.min(DISTANCE_TOLERANCE).max(floor);
        let outside = petal.implicit_radius(local) >= 1.0;
        let original = petal.distance_normal_original(local, tolerance, outside);
        let optimized = petal.distance_normal_with_tolerance(point, requested);
        assert_eq!(optimized.0.to_bits(), original.0.to_bits(), "distance at {point:?}");
        assert_eq!(
            optimized.1.map(f64::to_bits),
            original.1.map(f64::to_bits),
            "normal at {point:?}"
        );
    }

    fn dense_contour(petal: &Petal) -> Vec<Point> {
        (0..65_536).map(|i| petal.contour(TAU * f64::from(i) / 65_536.0).position).collect()
    }

    fn dense_distance(point: Point, contour: &[Point]) -> f64 {
        contour
            .iter()
            .map(|&p| {
                let d = subtract(p, point);
                dot(d, d)
            })
            .fold(f64::INFINITY, f64::min)
            .sqrt()
    }

    #[test]
    fn contour_derivatives_frames_and_bounds_match_the_implicit_shape() {
        let config = EclipseConfig::default();
        for shape in config.petals {
            let petal = Petal::new(shape, [0.4, -0.2], 1.06);
            let bounds = petal.bounds(0.03);
            let h = 1e-5;
            for i in 0..=512 {
                let theta = TAU * f64::from(i) / 512.0;
                let jet = petal.local_jet(theta);
                let left = petal.local_jet(theta - h);
                let right = petal.local_jet(theta + h);
                for axis in 0..2 {
                    assert!(
                        ((right.point[axis] - left.point[axis]) / (2.0 * h) - jet.first[axis])
                            .abs()
                            < 2e-8
                    );
                    assert!(
                        ((right.first[axis] - left.first[axis]) / (2.0 * h) - jet.second[axis])
                            .abs()
                            < 3e-8
                    );
                }
                let sample = petal.contour(theta);
                assert!((dot(sample.tangent, sample.tangent) - 1.0).abs() < 1e-12);
                assert!((dot(sample.normal, sample.normal) - 1.0).abs() < 1e-12);
                assert!(dot(sample.tangent, sample.normal).abs() < 1e-12);
                assert!((petal.implicit_radius(petal.local(sample.position)) - 1.0).abs() < 1e-12);
                let inside = add(sample.position, sample.normal.map(|x| -1e-4 * x));
                let outside = add(sample.position, sample.normal.map(|x| 1e-4 * x));
                assert!(petal.implicit_radius(petal.local(inside)) < 1.0);
                assert!(petal.implicit_radius(petal.local(outside)) > 1.0);
                for (axis, &coordinate) in sample.position.iter().enumerate() {
                    assert!(coordinate - 0.03 >= bounds[0][axis] - 1e-12);
                    assert!(coordinate + 0.03 <= bounds[1][axis] + 1e-12);
                }
            }
            let a = petal.contour(-1e-10);
            let b = petal.contour(TAU - 1e-10);
            assert!(
                subtract(a.position, b.position)[0].hypot(subtract(a.position, b.position)[1])
                    < 1e-12
            );
        }
    }

    #[test]
    fn corrected_distances_match_dense_reference_through_the_entire_light_band() {
        let config = EclipseConfig::default();
        for shape in config.petals {
            let petal = Petal::new(shape, [0.2, -0.3], 0.94);
            assert!(petal.convex);
            let reference = dense_contour(&petal);
            for theta in [0.17, 0.83, 1.53, 2.61, 3.83, 5.77] {
                let boundary = petal.contour(theta);
                for offset in [-3.5, -1.0, -0.2, 0.2, 1.0, 3.5] {
                    let point = add(
                        boundary.position,
                        boundary.normal.map(|x| offset * shape.light_sigma * x),
                    );
                    let (distance, normal) = petal.distance_normal(point);
                    let dense = dense_distance(point, &reference);
                    assert!(
                        (distance.abs() - dense).abs() < 6e-6,
                        "angle {theta}, offset {offset}: corrected {distance}, dense {dense}"
                    );
                    assert_eq!(distance.is_sign_negative(), offset < 0.0);
                    assert!((dot(normal, normal) - 1.0).abs() < 1e-12);
                    let (inside, lower) = petal.distance_lower_bound(point);
                    assert_eq!(inside, offset < 0.0);
                    assert!(lower <= distance.abs() + 3e-6);
                }
            }
        }
    }

    #[test]
    fn medial_and_unproven_convex_cases_keep_a_global_closest_distance() {
        let shape =
            EclipsePetalConfig { shear: 0.32, shoulder: 0.22, ..EclipsePetalConfig::default() };
        let petal = Petal::new(shape, [0.2, -0.3], 1.0);
        assert!(!petal.convex);
        let reference = dense_contour(&petal);
        for point in [[0.2, -0.3], [0.27, -0.27], [-0.8, 0.3], [0.1, 1.5], [1.4, -0.8]] {
            let (distance, normal) = petal.distance_normal(point);
            assert!((distance.abs() - dense_distance(point, &reference)).abs() < 6e-6);
            assert!((dot(normal, normal) - 1.0).abs() < 1e-12);
            assert_eq!(
                distance.is_sign_negative(),
                petal.implicit_radius(petal.local(point)) < 1.0
            );
        }
    }

    #[test]
    fn smooth_union_crosses_a_merger_without_a_body_order_seam() {
        let config = EclipseConfig::default();
        let circle = EclipsePetalConfig {
            semi_axes: [1.0; 2],
            shear: 0.0,
            shoulder: 0.0,
            angle_degrees: 0.0,
            ..EclipsePetalConfig::default()
        };
        for step in -12..=12 {
            let offset = f64::from(step) / 10.0 * config.edge_width;
            let center = 1.0 + config.smooth_join * 2.0_f64.ln() + offset;
            let petals = [
                Petal::new(circle, [-center, 0.0], 1.0),
                Petal::new(circle, [center, 0.0], 1.0),
                Petal::new(EclipsePetalConfig { enabled: false, ..circle }, [0.0; 2], 1.0),
            ];
            let actual = transmission([0.0; 2], &petals, &config);
            let expected = smoothstep(-config.edge_width, config.edge_width, offset);
            assert!((actual - expected).abs() < 2e-10, "{actual} vs {expected}");
            assert!(
                (actual - transmission([0.0; 2], &[petals[2], petals[1], petals[0]], &config))
                    .abs()
                    < 1e-14
            );
            assert_eq!(transmission([-center, 0.0], &petals, &config), 0.0);
            assert_eq!(transmission([10.0; 2], &petals, &config), 1.0);
        }
        let disabled =
            [Petal::new(EclipsePetalConfig { enabled: false, ..circle }, [0.0; 2], 1.0); 3];
        assert_eq!(transmission([0.0; 2], &disabled, &config), 1.0);
    }

    #[test]
    fn crescent_width_follows_true_world_distance_and_has_a_compact_smooth_tail() {
        let mut config = EclipseConfig {
            pearl: V3::new(1.0, 1.0, 1.0),
            rose: V3::new(1.0, 1.0, 1.0),
            copper: V3::new(1.0, 1.0, 1.0),
            ..EclipseConfig::default()
        };
        let petal = Petal::new(config.petals[0], [0.0; 2], 1.0);
        let boundary = petal.contour(0.31);
        config.light_direction = boundary.normal;
        for offset in [-1.0, 0.0, 1.0, 2.0, 3.5, 4.1] {
            let point = add(
                add(boundary.position, petal.shape.light_offset),
                boundary.normal.map(|x| offset * petal.shape.light_sigma * x),
            );
            let color = crescent(point, &petal, &config);
            let expected = petal.shape.light_gain
                * (-0.5 * offset * offset).exp()
                * (1.0 - smoothstep(3.0, 4.0, offset.abs()));
            assert!((color.x - expected).abs() < 4e-5, "offset {offset}: {color:?} vs {expected}");
            assert_eq!(color.x, color.y);
            assert_eq!(color.y, color.z);
        }
        assert_eq!(crescent([100.0; 2], &petal, &config), V3::ZERO);
        let disabled =
            Petal::new(EclipsePetalConfig { enabled: false, ..petal.shape }, [0.0; 2], 1.0);
        assert_eq!(crescent(boundary.position, &disabled, &config), V3::ZERO);
    }

    #[test]
    fn deferred_nearest_prepass_preserves_every_distance_and_normal_bit() {
        let config = EclipseConfig::default();
        for shape in config.petals {
            for factor in [0.94, 1.06] {
                let petal = Petal::new(shape, [0.23, -0.37], factor);
                assert!(petal.convex);
                let bounds = petal.bounds(4.0 * shape.light_sigma);
                for y in 0..=20 {
                    for x in 0..=24 {
                        let point = [
                            bounds[0][0] + (bounds[1][0] - bounds[0][0]) * f64::from(x) / 24.0,
                            bounds[0][1] + (bounds[1][1] - bounds[0][1]) * f64::from(y) / 20.0,
                        ];
                        assert_original_distance_bits(&petal, point, shape.light_sigma * 1e-6);
                    }
                }
                for index in 0_u32..128 {
                    let contour = petal.contour(TAU * f64::from(index) / 128.0);
                    let requested = if index.is_multiple_of(2) {
                        shape.light_sigma * 1e-6
                    } else {
                        config.edge_width * config.integration_tolerance * 0.25
                    };
                    for offset in [
                        -4.0 * shape.light_sigma,
                        -shape.light_sigma,
                        -1e-9,
                        0.0,
                        1e-9,
                        shape.light_sigma,
                        4.0 * shape.light_sigma,
                    ] {
                        let point = add(contour.position, contour.normal.map(|v| v * offset));
                        assert_original_distance_bits(&petal, point, requested);
                    }
                }
                let contour = petal.contour(0.31);
                let outside = add(contour.position, contour.normal.map(|v| v * shape.light_sigma));
                assert!(
                    petal
                        .certified_outside_projection(
                            petal.local(outside),
                            shape.light_sigma * 1e-6
                        )
                        .is_some()
                );
            }
        }
        let unproven = Petal::new(
            EclipsePetalConfig { shear: 0.30, shoulder: 0.22, ..config.petals[0] },
            [0.23, -0.37],
            1.0,
        );
        assert!(!unproven.convex);
        for point in [[0.23, -0.37], [-0.9, 0.2], [0.1, 1.7], [1.6, -0.8]] {
            assert_original_distance_bits(&unproven, point, 1e-7);
        }
    }

    #[test]
    fn certified_transmission_balls_preserve_dense_original_shader_samples() {
        let config = EclipseConfig::default();
        let centers = [[-0.8, -0.3], [0.7, 0.2], [0.0, 1.0]];
        let petals = std::array::from_fn(|i| Petal::new(config.petals[i], centers[i], 1.06));
        let mut dark = 0;
        let mut clear = 0;
        for y in 0..=16 {
            for x in 0..=20 {
                let center = [-4.0 + 8.0 * f64::from(x) / 20.0, -3.5 + 7.0 * f64::from(y) / 16.0];
                for radius in [0.0, 0.025, 0.09] {
                    if let Some(value) = constant_transmission(center, radius, &petals, &config) {
                        if value == 0.0 {
                            dark += 1;
                        } else {
                            clear += 1;
                        }
                        probe_ball(center, radius, |point| {
                            assert_eq!(
                                transmission(point, &petals, &config),
                                value,
                                "center {center:?}, radius {radius}, point {point:?}"
                            );
                        });
                    }
                }
            }
        }
        assert!(dark > 20 && clear > 200, "dark {dark}, clear {clear}");
    }

    #[test]
    fn transition_touching_balls_stay_uncertain_and_enabled_count_sets_clear_reach() {
        let config = EclipseConfig::default();
        let circle = EclipsePetalConfig {
            semi_axes: [1.0; 2],
            shear: 0.0,
            shoulder: 0.0,
            angle_degrees: 0.0,
            ..config.petals[0]
        };
        let radius = 0.1;
        let single = [
            Petal::new(circle, [0.0; 2], 1.0),
            Petal::new(EclipsePetalConfig { enabled: false, ..circle }, [0.0; 2], 1.0),
            Petal::new(EclipsePetalConfig { enabled: false, ..circle }, [0.0; 2], 1.0),
        ];
        let touch = [1.0 + config.edge_width + radius, 0.0];
        assert_eq!(constant_transmission(touch, radius, &single, &config), None);
        let clear = [touch[0] + 1e-6, 0.0];
        assert_eq!(constant_transmission(clear, radius, &single, &config), Some(1.0));
        probe_ball(clear, radius, |p| assert_eq!(transmission(p, &single, &config), 1.0));
        let crossing = [touch[0] - 0.001, 0.0];
        assert_eq!(constant_transmission(crossing, radius, &single, &config), None);
        assert!(transmission([crossing[0] - radius, 0.0], &single, &config) < 1.0);

        let lower = single[0].distance_lower_bound([0.0; 2]).1;
        let proof_touch = lower - config.edge_width;
        assert_eq!(constant_transmission([0.0; 2], proof_touch, &single, &config), None);
        assert_eq!(
            constant_transmission([0.0; 2], proof_touch - 1e-6, &single, &config),
            Some(0.0)
        );
        let dark_crossing = [1.0 - config.edge_width - radius + 0.001, 0.0];
        assert_eq!(constant_transmission(dark_crossing, radius, &single, &config), None);
        assert!(transmission([dark_crossing[0] + radius, 0.0], &single, &config) > 0.0);

        for count in 2..=3 {
            let petals = std::array::from_fn(|i| {
                Petal::new(EclipsePetalConfig { enabled: i < count, ..circle }, [0.0; 2], 1.0)
            });
            let join = config.smooth_join * (count as f64).ln();
            let too_close = [1.0 + radius + config.edge_width + 0.5 * join, 0.0];
            assert_eq!(constant_transmission(too_close, radius, &petals, &config), None);
            assert!(transmission([too_close[0] - radius, 0.0], &petals, &config) < 1.0);
            let clear = [1.0 + radius + config.edge_width + join + 1e-6, 0.0];
            assert_eq!(constant_transmission(clear, radius, &petals, &config), Some(1.0));
        }
        let disabled = [single[1]; 3];
        assert_eq!(constant_transmission([0.0; 2], 100.0, &disabled, &config), Some(1.0));
        for invalid in [-1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(constant_transmission([0.0; 2], invalid, &single, &config), None);
        }
    }

    #[test]
    fn certified_zero_crescent_balls_preserve_both_sides_of_the_complete_light_band() {
        let config = EclipseConfig::default();
        let mut certified = 0;
        for shape in config.petals {
            let petal = Petal::new(shape, [0.23, -0.37], 0.94);
            let bounds = petal.bounds(4.0 * shape.light_sigma + 0.20);
            for y in 0..=12 {
                for x in 0..=16 {
                    let center = [
                        bounds[0][0]
                            + (bounds[1][0] - bounds[0][0]) * f64::from(x) / 16.0
                            + shape.light_offset[0],
                        bounds[0][1]
                            + (bounds[1][1] - bounds[0][1]) * f64::from(y) / 12.0
                            + shape.light_offset[1],
                    ];
                    for radius in [0.0, 0.025, 0.09] {
                        if crescent_is_zero(center, radius, &petal) {
                            certified += 1;
                            probe_ball(center, radius, |point| {
                                assert_eq!(crescent(point, &petal, &config), V3::ZERO);
                            });
                        }
                    }
                }
            }
        }
        assert!(certified > 300, "only {certified} zero balls");
    }

    #[test]
    fn shifted_crescent_cutoff_touching_balls_never_discard_visible_light() {
        let config = EclipseConfig::default();
        let shape = EclipsePetalConfig {
            semi_axes: [1.0; 2],
            shear: 0.0,
            shoulder: 0.0,
            angle_degrees: 0.0,
            light_sigma: 0.1,
            light_offset: [0.17, -0.11],
            ..config.petals[0]
        };
        let petal = Petal::new(shape, [0.0; 2], 1.0);
        let radius = 0.1;
        let touch = add(shape.light_offset, [1.0 + 4.0 * shape.light_sigma + radius, 0.0]);
        assert!(!crescent_is_zero(touch, radius, &petal));
        let zero = [touch[0] + 1e-6, touch[1]];
        assert!(crescent_is_zero(zero, radius, &petal));
        probe_ball(zero, radius, |p| assert_eq!(crescent(p, &petal, &config), V3::ZERO));
        let crossing = [touch[0] - 0.001, touch[1]];
        assert!(!crescent_is_zero(crossing, radius, &petal));
        assert!(
            crescent([crossing[0] - radius, crossing[1]], &petal, &config).length_squared() > 0.0
        );
        assert!(crescent_is_zero(shape.light_offset, radius, &petal));
        probe_ball(shape.light_offset, radius, |p| {
            assert_eq!(crescent(p, &petal, &config), V3::ZERO);
        });
        for changed in [
            EclipsePetalConfig { enabled: false, ..shape },
            EclipsePetalConfig { light_gain: 0.0, ..shape },
        ] {
            let absent = Petal::new(changed, [0.0; 2], 1.0);
            assert!(crescent_is_zero([0.0; 2], 10.0, &absent));
        }
        for invalid in [-1.0, f64::INFINITY, f64::NAN] {
            assert!(!crescent_is_zero([0.0; 2], invalid, &petal));
        }
    }
}
