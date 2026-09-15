//! Continuous, fixed-frame access to an unchanged recorded three-body orbit.
//!
//! Positions pass through every original source knot after one translation and
//! one uniform scale. Geometry uses the original dense samples; a smaller table
//! carries rotation-minimizing frames and gently smoothed artistic measurements.
use super::{SilkResult, V3};
use crate::silk::OrbitData;
use crate::sim::{self, Body};
use nalgebra::Vector3;

/// Maximum number of transport/measurement knots over the complete source.
const FRAME_KNOTS: usize = 16_385;

/// One body's continuous position, stable local frame, and artistic measurements.
#[derive(Clone, Copy, Debug)]
pub struct BodySample {
    /// Position after the one fixed whole-source translation and uniform scale.
    pub position: V3,
    /// Unit direction along the original interpolated trajectory.
    pub tangent: V3,
    /// Rotation-minimizing unit normal, perpendicular to the tangent.
    pub normal: V3,
    /// Unit binormal, equal to `tangent.cross(normal)`.
    pub binormal: V3,
    /// Smoothed speed normalized to `[0, 1]` against the whole orbit.
    pub speed: f64,
    /// Smoothed geometric curvature normalized to `[0, 1]` against the whole orbit.
    pub curvature: f64,
    /// Smoothed proximity to the nearer companion, from zero (far) to one (near).
    pub proximity: f64,
    /// Cumulative world-space length, with negative values in verified prehistory.
    pub arc_length: f64,
}

/// The three bodies evaluated at one fraction of the recorded source interval.
#[derive(Clone, Copy, Debug)]
pub struct SourceFrame {
    /// Source fraction; `[0, 1]` is the original recording and negative values are verified prehistory.
    pub fraction: f64,
    /// Samples in original body order, preserving each body's identity.
    pub bodies: [BodySample; 3],
}

#[derive(Clone, Copy, Debug, Default)]
struct Knot {
    tangent: V3,
    normal: V3,
    speed: f64,
    curvature: f64,
    proximity: f64,
}

#[derive(Clone, Copy)]
struct Evaluation {
    position: V3,
    velocity: V3,
    acceleration: V3,
    left: usize,
    t: f64,
}

#[derive(Clone, Copy, Debug)]
struct MetricScale {
    speed_low: f64,
    speed_high: f64,
    curve_high: f64,
    distance_high: f64,
}

#[derive(Debug)]
struct Prelude {
    start_fraction: f64,
    positions: Vec<[V3; 3]>,
    arc_lengths: Vec<[f64; 3]>,
    knots: Vec<[Knot; 3]>,
    join_slopes: [V3; 3],
}

/// Reusable dense source geometry and precomputed, continuously transported frames.
///
/// The maximum span of the complete source's bounding box becomes four world
/// units. Each body receives the same transform. No per-frame or per-body fit,
/// projection, drift, endpoint wrap, or history extrapolation is introduced.
#[derive(Debug)]
pub struct OrbitSeries {
    positions: Vec<[V3; 3]>,
    arc_lengths: Vec<[f64; 3]>,
    knots: Vec<[Knot; 3]>,
    metric_scale: MetricScale,
    prelude: Option<Prelude>,
    center: V3,
    scale: f64,
    bounds: (V3, V3),
}

impl OrbitSeries {
    /// Prepare one immutable original source for repeated artistic sampling.
    pub fn new(orbit: &OrbitData) -> SilkResult<Self> {
        if orbit.samples.len() < 2 || !orbit.dt.is_finite() || orbit.dt <= 0.0 {
            return Err("atelier source needs at least two samples and a positive timestep".into());
        }
        if !orbit.masses.iter().all(|mass| mass.is_finite() && *mass > 0.0)
            || !orbit.samples.iter().flatten().all(|point| point.is_finite())
        {
            return Err("atelier source contains invalid body masses or positions".into());
        }
        let mut minimum = V3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut maximum = -minimum;
        for &point in orbit.samples.iter().flatten() {
            minimum = minimum.min(point);
            maximum = maximum.max(point);
        }
        let center = minimum * 0.5 + maximum * 0.5;
        let span = maximum - minimum;
        let extent = span.x.max(span.y).max(span.z);
        if !extent.is_finite() || !center.is_finite() {
            return Err("atelier source bounds exceed finite geometry range".into());
        }
        let scale = if extent > 1e-12 { 4.0 / extent } else { 1.0 };
        let positions: Vec<[V3; 3]> = orbit
            .samples
            .iter()
            .map(|points| points.map(|point| (point - center) * scale))
            .collect();
        let mut arc_lengths = vec![[0.0; 3]; positions.len()];
        for step in 1..positions.len() {
            for body in 0..3 {
                arc_lengths[step][body] = arc_lengths[step - 1][body]
                    + (positions[step][body] - positions[step - 1][body]).length();
            }
        }
        let (knots, metric_scale) = prepare_knots(&positions);
        Ok(Self {
            positions,
            arc_lengths,
            knots,
            metric_scale,
            prelude: None,
            center,
            scale,
            bounds: ((minimum - center) * scale, (maximum - center) * scale),
        })
    }

    /// Add real physical prehistory recovered from the original warm-up integration.
    ///
    /// Recorded initial-condition bits, integration cadence and warm-up length are
    /// required. The regenerated warm-up endpoint must match every cached first
    /// position bit. Visible positions, frames, metrics, bounds and arc phase stay
    /// identical to `new`; this method only extends the accepted negative domain.
    /// A zero history fraction is equivalent to `new` and requires no replay.
    pub fn with_prelude(orbit: &OrbitData, history_fraction: f64) -> SilkResult<Self> {
        if !history_fraction.is_finite() || history_fraction < 0.0 {
            return Err("prelude history fraction must be finite and nonnegative".into());
        }
        let mut series = Self::new(orbit)?;
        if history_fraction == 0.0 {
            return Ok(series);
        }
        let requested = (history_fraction * (orbit.samples.len() - 1) as f64).ceil();
        if !requested.is_finite() || requested >= usize::MAX as f64 {
            return Err("requested prelude exceeds the supported sample count".into());
        }
        let past_count = (requested as usize).max(1);
        let raw = replay_prelude(orbit, past_count)?;
        let mut positions: Vec<[V3; 3]> = raw
            .iter()
            .map(|row| row.map(|position| (position - series.center) * series.scale))
            .collect();
        // Use the original cached endpoint directly after verifying the replay.
        positions.push(series.positions[0]);
        let mut arc_lengths = vec![[0.0; 3]; positions.len()];
        for index in (0..past_count).rev() {
            for body in 0..3 {
                arc_lengths[index][body] = arc_lengths[index + 1][body]
                    - (positions[index + 1][body] - positions[index][body]).length();
            }
        }
        let start_fraction = -(past_count as f64) / (orbit.samples.len() - 1) as f64;
        let join_slopes = std::array::from_fn(|body| slope(&series.positions, body, 0));
        let count = (((series.knots.len() - 1) as f64 * -start_fraction).ceil() as usize + 1)
            .clamp(2, positions.len().min(FRAME_KNOTS));
        let (mut knots, pair_distances) = measure_knots(&positions, count, Some(&join_slopes));
        normalize_knots(&mut knots, &pair_distances, series.metric_scale);
        smooth_knots(&mut knots, ((series.knots.len() - 1) / 2048).max(1));
        let blend_count = 8.min(count - 1);
        for body in 0..3 {
            let canonical = series.body_at(body, 0.0);
            let last = count - 1;
            knots[last][body].tangent = canonical.tangent;
            knots[last][body].normal = canonical.normal;
            let mut tangent = canonical.tangent;
            let mut normal = canonical.normal;
            for index in (0..last).rev() {
                if knots[index][body].tangent.length_squared() < 0.5 {
                    knots[index][body].tangent = tangent;
                }
                normal = transport(tangent, knots[index][body].tangent, normal);
                tangent = knots[index][body].tangent;
                knots[index][body].normal = normal;
            }
            // Match the canonical one-sided smoothed measurements at zero,
            // without changing any visible-domain normalization or values.
            for (index, row) in knots.iter_mut().enumerate().skip(last - blend_count) {
                let t = (index - (last - blend_count)) as f64 / blend_count as f64;
                let weight = t * t * (3.0 - 2.0 * t);
                row[body].speed = lerp(row[body].speed, canonical.speed, weight);
                row[body].curvature = lerp(row[body].curvature, canonical.curvature, weight);
                row[body].proximity = lerp(row[body].proximity, canonical.proximity, weight);
            }
        }
        series.prelude =
            Some(Prelude { start_fraction, positions, arc_lengths, knots, join_slopes });
        Ok(series)
    }

    /// Earliest available source fraction; zero unless real prehistory was verified.
    #[must_use]
    pub fn history_start_fraction(&self) -> f64 {
        self.prelude.as_ref().map_or(0.0, |prelude| prelude.start_fraction)
    }

    /// Original-space center subtracted from every body position.
    pub fn bounds_center(&self) -> V3 {
        self.center
    }

    /// One uniform multiplier applied to every centered source position.
    #[must_use]
    pub fn world_scale(&self) -> f64 {
        self.scale
    }

    /// World-space bounds of all original source knots, as `(minimum, maximum)`.
    pub fn bounds(&self) -> (V3, V3) {
        self.bounds
    }

    /// Evaluate all three bodies within the available, verified source interval.
    ///
    /// Returns `None` outside `[history_start_fraction(), 1]`. `new` accepts only
    /// `[0, 1]`; `with_prelude` additionally accepts its verified negative prefix.
    #[must_use]
    pub fn sample(&self, fraction: f64) -> Option<SourceFrame> {
        if !valid_fraction(fraction, self.history_start_fraction()) {
            return None;
        }
        Some(SourceFrame {
            fraction,
            bodies: std::array::from_fn(|body| self.body_at(body, fraction)),
        })
    }

    /// Evaluate a body by its original zero-based index without wrapping time.
    #[must_use]
    pub fn sample_body(&self, body: usize, fraction: f64) -> Option<BodySample> {
        (body < 3 && valid_fraction(fraction, self.history_start_fraction()))
            .then(|| self.body_at(body, fraction))
    }

    fn body_at(&self, body: usize, fraction: f64) -> BodySample {
        if fraction < 0.0 {
            return self.prelude_body_at(body, fraction);
        }
        let evaluated = evaluate(&self.positions, body, fraction);
        let index = fraction * (self.knots.len() - 1) as f64;
        let left = (index.floor() as usize).min(self.knots.len() - 1);
        let right = (left + 1).min(self.knots.len() - 1);
        let t = index - left as f64;
        let a = self.knots[left][body];
        let b = self.knots[right][body];
        let tangent = if evaluated.velocity.length_squared() > 1e-24 {
            evaluated.velocity.normalized()
        } else {
            a.tangent
        };
        // Transport from the left table frame onto the actual dense-source
        // tangent. At the right knot this is exactly the next table normal,
        // so adjacent intervals join without an artificial roll discontinuity.
        let normal = transport(a.tangent, tangent, a.normal);
        let binormal = tangent.cross(normal).normalized();
        let arc_a = self.arc_lengths[evaluated.left][body];
        let arc_b = self.arc_lengths[evaluated.left + 1][body];
        BodySample {
            position: evaluated.position,
            tangent,
            normal,
            binormal,
            speed: lerp(a.speed, b.speed, t),
            curvature: lerp(a.curvature, b.curvature, t),
            proximity: lerp(a.proximity, b.proximity, t),
            arc_length: lerp(arc_a, arc_b, evaluated.t),
        }
    }

    fn prelude_body_at(&self, body: usize, fraction: f64) -> BodySample {
        let prelude =
            self.prelude.as_ref().expect("negative fractions require verified prehistory");
        // Domain validation precedes this conversion. Clamping only absorbs
        // floating-point roundoff at the two already-validated prefix endpoints.
        let local = ((fraction - prelude.start_fraction) / -prelude.start_fraction).clamp(0.0, 1.0);
        let evaluated =
            evaluate_with_join(&prelude.positions, body, local, Some(&prelude.join_slopes));
        let index = local * (prelude.knots.len() - 1) as f64;
        let left = (index.floor() as usize).min(prelude.knots.len() - 1);
        let right = (left + 1).min(prelude.knots.len() - 1);
        let t = index - left as f64;
        let a = prelude.knots[left][body];
        let b = prelude.knots[right][body];
        let tangent = if evaluated.velocity.length_squared() > 1e-24 {
            evaluated.velocity.normalized()
        } else {
            a.tangent
        };
        let normal = transport(a.tangent, tangent, a.normal);
        BodySample {
            position: evaluated.position,
            tangent,
            normal,
            binormal: tangent.cross(normal).normalized(),
            speed: lerp(a.speed, b.speed, t),
            curvature: lerp(a.curvature, b.curvature, t),
            proximity: lerp(a.proximity, b.proximity, t),
            arc_length: lerp(
                prelude.arc_lengths[evaluated.left][body],
                prelude.arc_lengths[evaluated.left + 1][body],
                evaluated.t,
            ),
        }
    }
}

fn valid_fraction(fraction: f64, start: f64) -> bool {
    fraction.is_finite() && (start..=1.0).contains(&fraction)
}

fn replay_prelude(orbit: &OrbitData, past_count: usize) -> SilkResult<Vec<[V3; 3]>> {
    let provenance = &orbit.provenance;
    if provenance["integrator"].as_str() != Some("production-yoshida4-native-f64-v1") {
        return Err("prelude requires production Yoshida integrator provenance".into());
    }
    let integration_dt = provenance["integration_dt"]
        .as_f64()
        .ok_or("prelude requires integration_dt provenance")?;
    let stride = usize::try_from(
        provenance["sample_stride"].as_u64().ok_or("prelude requires sample_stride provenance")?,
    )?;
    let warmup = usize::try_from(
        provenance["warmup_steps"].as_u64().ok_or("prelude requires warmup_steps provenance")?,
    )?;
    let recording = usize::try_from(
        provenance["recording_steps"]
            .as_u64()
            .ok_or("prelude requires recording_steps provenance")?,
    )?;
    if !integration_dt.is_finite()
        || integration_dt <= 0.0
        || stride == 0
        || (integration_dt * stride as f64).to_bits() != orbit.dt.to_bits()
        || recording == 0
        || (recording - 1) / stride + 1 != orbit.samples.len()
    {
        return Err("prelude source cadence or recording sample count is incompatible".into());
    }
    if provenance["gravitational_constant"].as_f64().map(f64::to_bits) != Some(sim::G.to_bits()) {
        return Err("prelude gravitational constant differs from the production integrator".into());
    }
    let retained_steps =
        past_count.checked_mul(stride).ok_or("requested prelude overflows integration length")?;
    if retained_steps > warmup {
        return Err(format!(
            "insufficient warmup for prelude: need {retained_steps} steps, have {warmup}"
        )
        .into());
    }
    let bits: [[u64; 7]; 3] = serde_json::from_value(
        provenance["initial_condition_f64_bits"].clone(),
    )
    .map_err(|error| format!("prelude requires exact initial_condition_f64_bits: {error}"))?;
    let mut bodies = Vec::with_capacity(3);
    for (body, row) in bits.iter().enumerate() {
        let values = row.map(f64::from_bits);
        if !values.iter().all(|value| value.is_finite())
            || values[0] <= 0.0
            || row[0] != orbit.masses[body].to_bits()
        {
            return Err(format!("invalid prelude initial conditions for body {body}").into());
        }
        bodies.push(Body::new(
            values[0],
            Vector3::new(values[1], values[2], values[3]),
            Vector3::new(values[4], values[5], values[6]),
        ));
    }
    // This is the same single COM shift performed by the production replay on
    // the stored initial conditions. Never shift the regenerated history again.
    sim::shift_bodies_to_com(&mut bodies);
    let first_step = warmup - retained_steps;
    let mut retained = Vec::with_capacity(past_count);
    tracing::info!(
        warmup_steps = warmup,
        prefix_samples = past_count,
        "Replaying verified orbital prehistory"
    );
    for step in 0..warmup {
        if step >= first_step && (step - first_step).is_multiple_of(stride) {
            let points = std::array::from_fn(|body| {
                V3::new(bodies[body].position.x, bodies[body].position.y, bodies[body].position.z)
            });
            if !points.iter().all(|point| point.is_finite()) {
                return Err("prelude integration produced non-finite positions".into());
            }
            retained.push(points);
        }
        sim::symplectic_step(&mut bodies, integration_dt);
    }
    for (body, state) in bodies.iter().enumerate() {
        for axis in 0..3 {
            if state.position[axis].to_bits() != orbit.samples[0][body].axis(axis).to_bits() {
                return Err(format!("prelude replay does not match cached first sample bits: body {body}, axis {axis}").into());
            }
        }
    }
    debug_assert_eq!(retained.len(), past_count);
    Ok(retained)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

fn slope(positions: &[[V3; 3]], body: usize, index: usize) -> V3 {
    if index == 0 {
        positions[1][body] - positions[0][body]
    } else if index + 1 == positions.len() {
        positions[index][body] - positions[index - 1][body]
    } else {
        (positions[index + 1][body] - positions[index - 1][body]) * 0.5
    }
}

fn evaluate(positions: &[[V3; 3]], body: usize, fraction: f64) -> Evaluation {
    evaluate_with_join(positions, body, fraction, None)
}

fn evaluate_with_join(
    positions: &[[V3; 3]],
    body: usize,
    fraction: f64,
    join: Option<&[V3; 3]>,
) -> Evaluation {
    let index = fraction * (positions.len() - 1) as f64;
    let left = (index.floor() as usize).min(positions.len() - 2);
    let t = index - left as f64;
    let p = positions[left][body];
    let q = positions[left + 1][body];
    let a = slope(positions, body, left);
    let b = if left + 2 == positions.len() {
        join.map_or_else(|| slope(positions, body, left + 1), |slopes| slopes[body])
    } else {
        slope(positions, body, left + 1)
    };
    let t2 = t * t;
    let t3 = t2 * t;
    let position = if t == 0.0 {
        p
    } else if t == 1.0 {
        q
    } else {
        p * (2.0 * t3 - 3.0 * t2 + 1.0)
            + a * (t3 - 2.0 * t2 + t)
            + q * (-2.0 * t3 + 3.0 * t2)
            + b * (t3 - t2)
    };
    Evaluation {
        position,
        velocity: p * (6.0 * t2 - 6.0 * t)
            + a * (3.0 * t2 - 4.0 * t + 1.0)
            + q * (-6.0 * t2 + 6.0 * t)
            + b * (3.0 * t2 - 2.0 * t),
        acceleration: p * (12.0 * t - 6.0)
            + a * (6.0 * t - 4.0)
            + q * (-12.0 * t + 6.0)
            + b * (6.0 * t - 2.0),
        left,
        t,
    }
}

fn initial_normal(tangent: V3) -> V3 {
    let axis = if tangent.x.abs() <= tangent.y.abs() && tangent.x.abs() <= tangent.z.abs() {
        V3::new(1.0, 0.0, 0.0)
    } else if tangent.y.abs() <= tangent.z.abs() {
        V3::new(0.0, 1.0, 0.0)
    } else {
        V3::new(0.0, 0.0, 1.0)
    };
    (axis - tangent * axis.dot(tangent)).normalized()
}

fn transport(previous: V3, next: V3, normal: V3) -> V3 {
    let cross = previous.cross(next);
    let sine = cross.length();
    let cosine = previous.dot(next).clamp(-1.0, 1.0);
    let rotated = if sine > 1e-12 {
        let axis = cross / sine;
        normal * cosine + axis.cross(normal) * sine + axis * (axis.dot(normal) * (1.0 - cosine))
    } else {
        // A straight continuation preserves the normal. At an exact velocity
        // reversal, rotating around the existing normal also preserves it;
        // this is a deterministic cusp convention, not an arbitrary frame flip.
        normal
    };
    let projected = rotated - next * rotated.dot(next);
    if projected.length_squared() > 1e-24 { projected.normalized() } else { initial_normal(next) }
}

fn measure_knots(
    positions: &[[V3; 3]],
    count: usize,
    join: Option<&[V3; 3]>,
) -> (Vec<[Knot; 3]>, Vec<[f64; 3]>) {
    let mut knots = vec![[Knot::default(); 3]; count];
    let mut pair_distances = Vec::with_capacity(count);
    for (index, row) in knots.iter_mut().enumerate() {
        let fraction = index as f64 / (count - 1) as f64;
        let samples: [Evaluation; 3] =
            std::array::from_fn(|body| evaluate_with_join(positions, body, fraction, join));
        for (body, knot) in row.iter_mut().enumerate() {
            let sample = samples[body];
            let speed = sample.velocity.length();
            knot.tangent = sample.velocity.normalized();
            knot.speed = speed;
            knot.curvature = if speed > 1e-11 {
                sample.velocity.cross(sample.acceleration).length() / speed.powi(3)
            } else {
                0.0
            };
        }
        pair_distances.push([
            (samples[0].position - samples[1].position).length(),
            (samples[0].position - samples[2].position).length(),
            (samples[1].position - samples[2].position).length(),
        ]);
    }
    (knots, pair_distances)
}

fn prepare_knots(positions: &[[V3; 3]]) -> (Vec<[Knot; 3]>, MetricScale) {
    let count = positions.len().min(FRAME_KNOTS);
    let (mut knots, pair_distances) = measure_knots(positions, count, None);
    let mut speed_values: Vec<f64> = knots.iter().flatten().map(|knot| knot.speed).collect();
    let speed_low = percentile(&mut speed_values, 0.02);
    let speed_high = percentile(&mut speed_values, 0.98);
    let mut curve_values: Vec<f64> = knots.iter().flatten().map(|knot| knot.curvature).collect();
    let curve_high = percentile(&mut curve_values, 0.98);
    let mut distance_values: Vec<f64> = pair_distances.iter().flatten().copied().collect();
    let distance_high = percentile(&mut distance_values, 0.90).max(1e-12);
    let scale = MetricScale { speed_low, speed_high, curve_high, distance_high };
    normalize_knots(&mut knots, &pair_distances, scale);
    smooth_knots(&mut knots, ((count - 1) / 2048).max(1));
    for body in 0..3 {
        let first = knots
            .iter()
            .map(|row| row[body].tangent)
            .find(|tangent| tangent.length_squared() > 0.5)
            .unwrap_or(V3::new(1.0, 0.0, 0.0));
        let mut tangent = first;
        let mut normal = initial_normal(tangent);
        for row in &mut knots {
            if row[body].tangent.length_squared() < 0.5 {
                row[body].tangent = tangent;
            }
            normal = transport(tangent, row[body].tangent, normal);
            tangent = row[body].tangent;
            row[body].normal = normal;
        }
    }
    (knots, scale)
}

fn normalize_knots(knots: &mut [[Knot; 3]], pair_distances: &[[f64; 3]], scale: MetricScale) {
    let MetricScale { speed_low, speed_high, curve_high, distance_high } = scale;
    for (index, row) in knots.iter_mut().enumerate() {
        let [ab, ac, bc] = pair_distances[index];
        let nearest = [ab.min(ac), ab.min(bc), ac.min(bc)];
        for (body, knot) in row.iter_mut().enumerate() {
            knot.speed = if speed_high - speed_low > speed_high.max(1e-12) * 1e-9 {
                ((knot.speed - speed_low) / (speed_high - speed_low)).clamp(0.0, 1.0)
            } else if speed_high > 1e-12 {
                0.5
            } else {
                0.0
            };
            knot.curvature = if curve_high > 1e-10 {
                (knot.curvature / curve_high).clamp(0.0, 1.0)
            } else {
                0.0
            };
            knot.proximity = (1.0 - nearest[body] / distance_high).clamp(0.0, 1.0);
        }
    }
}

fn smooth_knots(knots: &mut [[Knot; 3]], radius: usize) {
    let count = knots.len();
    let unsmoothed = knots.to_vec();
    for (index, row) in knots.iter_mut().enumerate() {
        for (body, knot) in row.iter_mut().enumerate() {
            let mut sum = [0.0; 3];
            let mut total = 0.0;
            let begin = index.saturating_sub(radius);
            let end = (index + radius + 1).min(count);
            for (other, values) in unsmoothed.iter().enumerate().take(end).skip(begin) {
                let weight = ((radius + 1 - other.abs_diff(index)) as f64).powi(2);
                sum[0] += values[body].speed * weight;
                sum[1] += values[body].curvature * weight;
                sum[2] += values[body].proximity * weight;
                total += weight;
            }
            knot.speed = sum[0] / total;
            knot.curvature = sum[1] / total;
            knot.proximity = sum[2] / total;
        }
    }
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn orbit(samples: Vec<[V3; 3]>) -> OrbitData {
        OrbitData {
            seed: "0x01".into(),
            dt: 0.001,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({}),
        }
    }

    fn replay_fixture(stride: usize) -> (OrbitData, Vec<[V3; 3]>) {
        let initial = vec![
            Body::new(2.0, Vector3::new(-2.0, 0.1, 0.3), Vector3::new(0.12, 0.25, -0.04)),
            Body::new(3.0, Vector3::new(1.5, -0.3, -0.2), Vector3::new(-0.2, 0.04, 0.03)),
            Body::new(1.0, Vector3::new(0.2, 1.8, 0.1), Vector3::new(0.06, -0.15, 0.01)),
        ];
        let bits: Vec<[u64; 7]> = initial
            .iter()
            .map(|body| {
                [
                    body.mass.to_bits(),
                    body.position.x.to_bits(),
                    body.position.y.to_bits(),
                    body.position.z.to_bits(),
                    body.velocity.x.to_bits(),
                    body.velocity.y.to_bits(),
                    body.velocity.z.to_bits(),
                ]
            })
            .collect();
        let dt = 0.001;
        let warmup = 20;
        let recording = 1 + 5 * stride;
        let mut bodies = initial;
        sim::shift_bodies_to_com(&mut bodies);
        let mut states = Vec::new();
        for step in 0..warmup + recording {
            states.push(std::array::from_fn(|body| {
                V3::new(bodies[body].position.x, bodies[body].position.y, bodies[body].position.z)
            }));
            if step + 1 < warmup + recording {
                sim::symplectic_step(&mut bodies, dt);
            }
        }
        let samples = (0..6).map(|index| states[warmup + index * stride]).collect();
        (
            OrbitData {
                seed: "0x01".into(),
                dt: dt * stride as f64,
                masses: [2.0, 3.0, 1.0],
                samples,
                provenance: serde_json::json!({
                    "initial_condition_f64_bits":bits,"integrator":"production-yoshida4-native-f64-v1",
                    "integration_dt":dt,"sample_stride":stride,"warmup_steps":warmup,
                    "recording_steps":recording,"gravitational_constant":sim::G,
                }),
            },
            states,
        )
    }

    fn assert_same_bits(a: BodySample, b: BodySample) {
        for (first, second) in [
            (a.position, b.position),
            (a.tangent, b.tangent),
            (a.normal, b.normal),
            (a.binormal, b.binormal),
        ] {
            for axis in 0..3 {
                assert_eq!(first.axis(axis).to_bits(), second.axis(axis).to_bits());
            }
        }
        for (first, second) in [
            (a.speed, b.speed),
            (a.curvature, b.curvature),
            (a.proximity, b.proximity),
            (a.arc_length, b.arc_length),
        ] {
            assert_eq!(first.to_bits(), second.to_bits());
        }
    }

    #[test]
    fn verified_prelude_uses_real_warmup_samples_and_preserves_visible_bits() {
        for stride in [1, 2] {
            let (orbit, states) = replay_fixture(stride);
            let original = OrbitSeries::new(&orbit).unwrap();
            let extended = OrbitSeries::with_prelude(&orbit, 0.4).unwrap();
            assert_eq!(extended.history_start_fraction(), -0.4);
            assert_eq!(extended.bounds(), original.bounds());
            assert_eq!(extended.world_scale().to_bits(), original.world_scale().to_bits());
            assert_eq!(extended.bounds_center(), original.bounds_center());
            for (body, (&first_state, &second_state)) in
                states[20 - 2 * stride].iter().zip(states[20 - stride].iter()).enumerate()
            {
                for fraction in [0.0, 0.0001, 0.2, 0.5, 0.99, 1.0] {
                    assert_same_bits(
                        original.sample_body(body, fraction).unwrap(),
                        extended.sample_body(body, fraction).unwrap(),
                    );
                }
                let first = extended.sample_body(body, -0.4).unwrap();
                assert_eq!(first.position, (first_state - original.center) * original.scale);
                let second = extended.sample_body(body, -0.2).unwrap();
                assert_eq!(second.position, (second_state - original.center) * original.scale);
                assert!(first.arc_length < second.arc_length && second.arc_length < 0.0);
                let before = extended.sample_body(body, -1e-9).unwrap();
                let zero = extended.sample_body(body, 0.0).unwrap();
                assert!((before.position - zero.position).length() < 1e-7);
                assert!(before.tangent.dot(zero.tangent) > 1.0 - 1e-10);
                assert!(before.normal.dot(zero.normal) > 1.0 - 1e-10);
                assert!(before.arc_length <= 0.0 && before.arc_length > -1e-7);
            }
            assert!(extended.sample(-0.4 - 1e-12).is_none());
            assert!(extended.sample(1.0 + 1e-12).is_none());
            assert!(original.sample(-0.1).is_none());
        }
    }

    #[test]
    fn prelude_rejects_unverifiable_or_incompatible_sources() {
        let (orbit, _) = replay_fixture(1);
        let mut changed = orbit.clone();
        changed.samples[0][0].x = f64::from_bits(changed.samples[0][0].x.to_bits() + 1);
        assert!(
            OrbitSeries::with_prelude(&changed, 0.4)
                .unwrap_err()
                .to_string()
                .contains("first sample bits")
        );
        changed = orbit.clone();
        changed.dt *= 2.0;
        assert!(
            OrbitSeries::with_prelude(&changed, 0.4).unwrap_err().to_string().contains("cadence")
        );
        changed = orbit.clone();
        changed.provenance["initial_condition_f64_bits"] = serde_json::Value::Null;
        assert!(
            OrbitSeries::with_prelude(&changed, 0.4)
                .unwrap_err()
                .to_string()
                .contains("initial_condition_f64_bits")
        );
        assert!(
            OrbitSeries::with_prelude(&orbit, 5.0)
                .unwrap_err()
                .to_string()
                .contains("insufficient warmup")
        );
        assert!(OrbitSeries::with_prelude(&orbit, f64::NAN).is_err());
        assert_eq!(OrbitSeries::with_prelude(&changed, 0.0).unwrap().history_start_fraction(), 0.0);
    }

    #[test]
    fn endpoints_and_arc_length_use_one_shared_world_transform() {
        let original = orbit(vec![
            [V3::new(0.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.5, 0.5, 0.0)],
            [V3::new(1.0, 0.0, 0.0), V3::new(1.0, 1.0, 0.0), V3::new(1.5, 0.5, 0.0)],
        ]);
        let series = OrbitSeries::new(&original).unwrap();
        assert_eq!(series.world_scale(), 4.0 / 1.5);
        let (lo, hi) = series.bounds();
        assert!((hi.x - lo.x - 4.0).abs() < 1e-12);
        for body in 0..3 {
            let start = series.sample_body(body, 0.0).unwrap();
            let end = series.sample_body(body, 1.0).unwrap();
            assert_eq!(
                start.position,
                (original.samples[0][body] - series.bounds_center()) * series.world_scale()
            );
            assert_eq!(
                end.position,
                (original.samples[1][body] - series.bounds_center()) * series.world_scale()
            );
            assert_eq!(start.arc_length, 0.0);
            assert!((end.arc_length - series.world_scale()).abs() < 1e-12);
            assert!(
                (series.sample_body(body, 0.5).unwrap().arc_length - end.arc_length * 0.5).abs()
                    < 1e-12
            );
        }
        assert!(series.sample(-1e-12).is_none());
        assert!(series.sample(1.0 + 1e-12).is_none());
        assert!(series.sample(f64::NAN).is_none());
        assert!(series.sample_body(3, 0.5).is_none());
    }

    #[test]
    fn transported_frames_do_not_flip_at_planar_inflections() {
        let samples = (0..1001)
            .map(|step| {
                let t = f64::from(step) / 1000.0 * TAU;
                std::array::from_fn(|body| {
                    let phase = t + body as f64 * 0.7;
                    V3::new(phase.sin(), (2.0 * phase).sin() * 0.5, 0.0)
                })
            })
            .collect();
        let series = OrbitSeries::new(&orbit(samples)).unwrap();
        let mut previous = series.sample(0.0).unwrap();
        for step in 1..4001 {
            let current = series.sample(f64::from(step) / 4000.0).unwrap();
            for body in 0..3 {
                let a = previous.bodies[body];
                let b = current.bodies[body];
                assert!(a.normal.dot(b.normal) > 0.999999, "normal flipped at planar inflection");
                assert!((b.tangent.length() - 1.0).abs() < 1e-12);
                assert!((b.normal.length() - 1.0).abs() < 1e-12);
                assert!((b.binormal.length() - 1.0).abs() < 1e-12);
                assert!(b.tangent.dot(b.normal).abs() < 1e-12);
                assert!((b.tangent.cross(b.normal) - b.binormal).length() < 1e-12);
                assert!(b.arc_length >= a.arc_length);
                for metric in [b.speed, b.curvature, b.proximity] {
                    assert!((0.0..=1.0).contains(&metric));
                }
            }
            previous = current;
        }
    }

    #[test]
    fn dense_helix_has_continuous_frames_and_source_knot_positions() {
        let samples: Vec<[V3; 3]> = (0..201)
            .map(|step| {
                let t = f64::from(step) / 200.0 * TAU * 2.0;
                std::array::from_fn(|body| {
                    V3::new((t + body as f64).cos(), (t + body as f64).sin(), t * 0.15)
                })
            })
            .collect();
        let source = orbit(samples.clone());
        let series = OrbitSeries::new(&source).unwrap();
        for (index, points) in samples.iter().enumerate() {
            let sample = series.sample(index as f64 / 200.0).unwrap();
            for (body, point) in points.iter().enumerate() {
                assert!(
                    (sample.bodies[body].position - (*point - series.center) * series.scale)
                        .length()
                        < 1e-12
                );
            }
        }
        let mut previous = series.sample(0.0).unwrap();
        for index in 1..4001 {
            let current = series.sample(f64::from(index) / 4000.0).unwrap();
            for body in 0..3 {
                assert!(previous.bodies[body].normal.dot(current.bodies[body].normal) > 0.999);
            }
            previous = current;
        }
    }
}
