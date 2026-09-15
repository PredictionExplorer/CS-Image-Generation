//! Persistent corona hairs with finite Gaussian line footprints.
//!
//! Arc-length anchors and long-hair identities belong to the undeformed recipe,
//! never the current pose. Each exposure carries those identities through the
//! moving contour. The caller applies the common opaque transmission jointly
//! with this emitted field before averaging any receiving footprint.

use super::{EclipseConfig, Point, field::Petal};
use crate::atelier::{SilkResult, V3};
use std::f64::consts::{SQRT_2, TAU};

const SUPPORT_SIGMAS: f64 = 6.0;
const MAX_CHORD_ERROR_PIXELS: f64 = 0.10;

/// One finite piece of a persistent quadratic hair.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Segment {
    first: Point,
    tangent: Point,
    length: f64,
    radius: f64,
    sigma: f64,
    color: V3,
    bounds: [Point; 2],
}

impl Segment {
    fn new(
        first: Point,
        last: Point,
        radius: f64,
        reconstruction_sigma: f64,
        color: V3,
    ) -> SilkResult<Self> {
        let direction = sub(last, first);
        let length = length(direction);
        let sigma = radius.hypot(reconstruction_sigma);
        if !first.iter().chain(&last).all(|v| v.is_finite())
            || !length.is_finite()
            || length <= 0.0
            || !radius.is_finite()
            || radius <= 0.0
            || !sigma.is_finite()
            || sigma <= 0.0
            || !color.is_finite()
            || color.min(V3::ZERO) != V3::ZERO
        {
            return Err(
                "corona segment needs finite positive geometry and nonnegative light".into()
            );
        }
        let padding = SUPPORT_SIGMAS * sigma;
        Ok(Self {
            first,
            tangent: scale(direction, length.recip()),
            length,
            radius,
            sigma,
            color,
            bounds: [
                [
                    (first[0].min(last[0]) - padding).next_down(),
                    (first[1].min(last[1]) - padding).next_down(),
                ],
                [
                    (first[0].max(last[0]) + padding).next_up(),
                    (first[1].max(last[1]) + padding).next_up(),
                ],
            ],
        })
    }

    /// World-space bounds with six standard deviations of reconstruction halo.
    pub(super) fn bounds(&self) -> [Point; 2] {
        self.bounds
    }

    /// Unoccluded linear emitted radiance; the caller integrates it with the mask.
    pub(super) fn radiance(&self, point: Point) -> V3 {
        if point[0] < self.bounds[0][0]
            || point[0] > self.bounds[1][0]
            || point[1] < self.bounds[0][1]
            || point[1] > self.bounds[1][1]
        {
            return V3::ZERO;
        }
        let relative = sub(point, self.first);
        let along = dot(relative, self.tangent);
        let across = relative[0] * -self.tangent[1] + relative[1] * self.tangent[0];
        let longitudinal = (0.5
            * (erf(along / (self.sigma * SQRT_2))
                + erf((self.length - along) / (self.sigma * SQRT_2))))
        .clamp(0.0, 1.0);
        let transverse = (-0.5 * (across / self.sigma).powi(2)).exp();
        self.color * (self.radius / self.sigma * longitudinal * transverse)
    }

    fn estimated_luminance(&self) -> f64 {
        // The longitudinal CDF integrates to segment length. Convolution adds
        // width but radius/sigma preserves the Gaussian cross-section integral.
        luminance(self.color) * TAU.sqrt() * self.radius * self.length
    }
}

/// Prepared unoccluded corona geometry for one source exposure.
pub(super) struct Corona {
    /// Fixed body, hair and segment order, ready for deterministic tile bins.
    pub segments: Vec<Segment>,
    /// Number of persistent hairs, including the selected extended subset.
    pub curve_count: usize,
    /// Exact quadratic midpoint/chord discrepancy in final pixels.
    pub max_chord_error_pixels: f64,
    /// Estimated full unoccluded luminance integral before six-sigma truncation.
    pub estimated_luminance: f64,
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn scale(a: Point, value: f64) -> Point {
    [a[0] * value, a[1] * value]
}
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn length(a: Point) -> f64 {
    a[0].hypot(a[1])
}
fn luminance(value: V3) -> f64 {
    value.x * 0.2126 + value.y * 0.7152 + value.z * 0.0722
}

fn smooth(value: f64) -> f64 {
    let t = value.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn noise(mut value: u64) -> f64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / (1_u64 << 53) as f64
}

fn smooth_variation(ordinal: usize, hairs: usize, seed: u64) -> f64 {
    // Default spacing is exactly eight anchors. Equal cyclic cells also keep
    // nonmultiples of eight C2-continuous across the closed contour's seam.
    let controls = hairs.div_ceil(8);
    let coordinate = ordinal as f64 * controls as f64 / hairs as f64;
    let left = coordinate.floor() as usize % controls;
    let right = (left + 1) % controls;
    let a = noise(seed ^ (left as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    let b = noise(seed ^ (right as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    a + (b - a) * smooth(coordinate.fract())
}

fn gathering(theta: f64, seed: u64) -> f64 {
    let phase = TAU * noise(seed ^ 0x31b7);
    let second = TAU * noise(seed ^ 0x942d);
    let broad = 0.60 + 0.25 * (2.0 * theta + phase).cos() + 0.15 * (3.0 * theta + second).cos();
    0.12 + 0.88 * broad.clamp(0.0, 1.0).powi(2)
}

fn directional(normal: Point, light: Point) -> f64 {
    0.06 + 0.94 * dot(normal, light).max(0.0).powi(2)
}

fn anchor_angles(reference: &Petal, hairs: usize, intervals: usize) -> SilkResult<Vec<f64>> {
    let mut arc = Vec::with_capacity(intervals + 1);
    arc.push(0.0);
    let mut previous = reference.contour(0.0).position;
    for index in 1..=intervals {
        let current = reference.contour(TAU * index as f64 / intervals as f64).position;
        arc.push(arc[index - 1] + length(sub(current, previous)));
        previous = current;
    }
    let total = arc[intervals];
    if !total.is_finite() || total <= 0.0 {
        return Err("corona reference contour has no finite arc length".into());
    }
    let mut cell = 0;
    let mut angles = Vec::with_capacity(hairs);
    for ordinal in 0..hairs {
        let target = total * ordinal as f64 / hairs as f64;
        while cell + 1 < intervals && arc[cell + 1] < target {
            cell += 1;
        }
        let fraction = (target - arc[cell]) / (arc[cell + 1] - arc[cell]);
        angles.push(TAU * (cell as f64 + fraction) / intervals as f64);
    }
    Ok(angles)
}

fn extended_hairs(
    reference: &Petal,
    angles: &[f64],
    count: usize,
    light: Point,
    seed: u64,
) -> Vec<bool> {
    let mut selected = vec![false; angles.len()];
    if count == 0 {
        return selected;
    }
    let mut ranking: Vec<_> = angles
        .iter()
        .enumerate()
        .map(|(ordinal, &theta)| {
            let shade = directional(reference.contour(theta).normal, light);
            let weight = shade.powi(3) * gathering(theta, seed);
            let key = -noise(seed ^ (ordinal as u64).wrapping_mul(0x94d0_49bb_1331_11eb))
                .max(f64::MIN_POSITIVE)
                .ln()
                / weight;
            (key, ordinal)
        })
        .collect();
    ranking.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let spacing = (angles.len() / count.saturating_mul(3)).max(1);
    let mut indices = Vec::with_capacity(count);
    for (_, ordinal) in ranking {
        if indices.iter().any(|&previous: &usize| {
            let distance = ordinal.abs_diff(previous);
            distance.min(angles.len() - distance) < spacing
        }) {
            continue;
        }
        selected[ordinal] = true;
        indices.push(ordinal);
        if indices.len() == count {
            break;
        }
    }
    debug_assert_eq!(indices.len(), count);
    selected
}

/// Carry the recipe's fixed hair identities through one posed set of petals.
pub(super) fn prepare(
    petals: &[Petal; 3],
    qz: [f64; 3],
    config: &EclipseConfig,
    pixels_per_world: f64,
) -> SilkResult<Corona> {
    let mut result = Corona {
        segments: Vec::new(),
        curve_count: 0,
        max_chord_error_pixels: 0.0,
        estimated_luminance: 0.0,
    };
    if config.corona_fraction == 0.0 || config.corona_hairs == 0 {
        return Ok(result);
    }
    if !pixels_per_world.is_finite()
        || pixels_per_world <= 0.0
        || !config.minimum_sigma_pixels.is_finite()
        || config.minimum_sigma_pixels < 0.25
        || !config.corona_fraction.is_finite()
        || config.corona_fraction < 0.0
        || config.anchor_intervals < 4
        || config.corona_segments == 0
        || config.corona_long_hairs > config.corona_hairs
        || !qz.iter().all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
    {
        return Err("invalid corona geometry, source component or reconstruction settings".into());
    }
    for range in [config.corona_lengths, config.corona_long_lengths, config.corona_radii] {
        if !range.iter().all(|value| value.is_finite() && *value > 0.0) || range[0] > range[1] {
            return Err("corona lengths and radii need finite positive ordered ranges".into());
        }
    }
    let light_length = length(config.light_direction);
    if !light_length.is_finite() || light_length <= 0.0 {
        return Err("corona needs a finite nonzero light direction".into());
    }
    let light = scale(config.light_direction, light_length.recip());
    let reconstruction_sigma = config.minimum_sigma_pixels / pixels_per_world;
    let count = config
        .corona_hairs
        .checked_mul(config.corona_segments)
        .and_then(|value| value.checked_mul(3))
        .ok_or("corona segment count overflow")?;
    result.segments.try_reserve(count)?;
    for body in 0..3 {
        let shape = config.petals[body];
        if !shape.enabled || shape.light_gain == 0.0 {
            continue;
        }
        let reference = Petal::new(shape, [0.0; 2], 1.0);
        let seed = config.corona_seed ^ (body as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
        let angles = anchor_angles(&reference, config.corona_hairs, config.anchor_intervals)?;
        let extended = extended_hairs(&reference, &angles, config.corona_long_hairs, light, seed);
        let anchors: Vec<_> = angles.iter().map(|&theta| petals[body].contour(theta)).collect();
        let mut weighted_perimeter = 0.0;
        for ordinal in 0..anchors.len() {
            let next = (ordinal + 1) % anchors.len();
            weighted_perimeter += length(sub(anchors[next].position, anchors[ordinal].position))
                * 0.5
                * (directional(anchors[ordinal].normal, light)
                    + directional(anchors[next].normal, light));
        }
        // This is explicitly a pearl-colored, unoccluded broad-contour
        // estimate. It does not inspect visible pixels or normalize a frame's
        // exposure, peak, occlusion, or dark joining regions.
        let target = config.corona_fraction
            * luminance(config.pearl)
            * shape.light_gain
            * TAU.sqrt()
            * shape.light_sigma
            * weighted_perimeter;
        let start = result.segments.len();
        let bend = shape.corona_bend + 0.08 * qz[body];
        for (ordinal, (&theta, anchor)) in angles.iter().zip(&anchors).enumerate() {
            let group = gathering(theta, seed);
            let variation = smooth_variation(ordinal, angles.len(), seed ^ 0x5ac1);
            let range =
                if extended[ordinal] { config.corona_long_lengths } else { config.corona_lengths };
            let hair_length = range[0] + (range[1] - range[0]) * (0.75 * variation + 0.25 * group);
            let radius_variation = smooth_variation(ordinal, angles.len(), seed ^ 0x39a7);
            let radius = config.corona_radii[0]
                + (config.corona_radii[1] - config.corona_radii[0]) * radius_variation;
            let strength = group
                * (0.35 + 0.65 * smooth_variation(ordinal, angles.len(), seed ^ 0x918d))
                * directional(anchor.normal, light)
                * shape.light_gain;
            let root = add(anchor.position, shape.light_offset);
            let point = |u: f64| {
                add(
                    root,
                    add(
                        scale(anchor.normal, hair_length * u),
                        scale(anchor.tangent, bend * hair_length * u * u),
                    ),
                )
            };
            let chord_error = bend.abs() * hair_length * pixels_per_world
                / (4.0 * (config.corona_segments as f64).powi(2));
            result.max_chord_error_pixels = result.max_chord_error_pixels.max(chord_error);
            if chord_error > MAX_CHORD_ERROR_PIXELS {
                return Err(format!("corona midpoint error {chord_error} pixels exceeds0.10; increase the fixed segment count for the recipe").into());
            }
            for segment in 0..config.corona_segments {
                let a = segment as f64 / config.corona_segments as f64;
                let b = (segment + 1) as f64 / config.corona_segments as f64;
                let u = f64::midpoint(a, b);
                let taper = smooth(u / 0.08) * (1.0 - smooth((u - 0.50) / 0.50));
                let tint = config.pearl.lerp(config.rose, 0.18 * smooth((u - 0.2) / 0.8));
                result.segments.push(Segment::new(
                    point(a),
                    point(b),
                    radius,
                    reconstruction_sigma,
                    tint * (strength * taper),
                )?);
            }
            result.curve_count += 1;
        }
        let estimate: f64 = result.segments[start..].iter().map(Segment::estimated_luminance).sum();
        if !estimate.is_finite() || estimate <= 0.0 || !target.is_finite() || target <= 0.0 {
            return Err("corona normalization needs finite positive unoccluded estimates".into());
        }
        let normalization = target / estimate;
        for segment in &mut result.segments[start..] {
            segment.color *= normalization;
            if !segment.color.is_finite() {
                return Err("corona normalization exceeded finite radiance".into());
            }
            result.estimated_luminance += segment.estimated_luminance();
        }
    }
    Ok(result)
}

// Same approximation as the shared renderer; it changes no shared optical path.
fn erf(value: f64) -> f64 {
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let p = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592)
        * t;
    (1.0 - p * (-x * x).exp()).copysign(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> EclipseConfig {
        EclipseConfig {
            corona_hairs: 64,
            corona_long_hairs: 4,
            anchor_intervals: 512,
            corona_segments: 16,
            ..EclipseConfig::default()
        }
    }

    fn petals(config: &EclipseConfig, offset: Point) -> [Petal; 3] {
        config.petals.map(|shape| Petal::new(shape, offset, 1.0))
    }

    #[test]
    fn straight_finite_segments_preserve_radiance_and_energy_under_subdivision() {
        let color = V3::new(1.0, 0.8, 0.6);
        let whole = Segment::new([0.0, 0.0], [1.0, 0.0], 0.002, 0.003, color).unwrap();
        let pieces: Vec<_> = (0..32)
            .map(|i| {
                Segment::new(
                    [f64::from(i) / 32.0, 0.0],
                    [f64::from(i + 1) / 32.0, 0.0],
                    0.002,
                    0.003,
                    color,
                )
                .unwrap()
            })
            .collect();
        let energy: f64 = pieces.iter().map(Segment::estimated_luminance).sum();
        assert!((energy - whole.estimated_luminance()).abs() < 1e-15);
        for x in -10..=110 {
            for y in -7..=7 {
                let point = [f64::from(x) / 100.0, f64::from(y) * whole.sigma];
                let split =
                    pieces.iter().fold(V3::ZERO, |total, part| total + part.radiance(point));
                assert!((whole.radiance(point) - split).length() < 5e-8);
            }
        }
        let wider = Segment::new([0.0, 0.0], [1.0, 0.0], 0.002, 0.006, color).unwrap();
        assert_eq!(whole.estimated_luminance(), wider.estimated_luminance());
        assert!(wider.radiance([0.5, 0.0]).x < whole.radiance([0.5, 0.0]).x);
    }

    #[test]
    fn persistent_anchor_selection_is_closed_spaced_and_shoulder_weighted() {
        let config = EclipseConfig::default();
        let shape = Petal::new(config.petals[0], [0.0; 2], 1.0);
        let light = scale(config.light_direction, length(config.light_direction).recip());
        let angles = anchor_angles(&shape, config.corona_hairs, config.anchor_intervals).unwrap();
        assert_eq!(angles.len(), 1536);
        assert_eq!(angles[0], 0.0);
        assert!(angles.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(angles[1535] < TAU);
        let selected = extended_hairs(&shape, &angles, 36, light, config.corona_seed);
        assert_eq!(selected.iter().filter(|&&value| value).count(), 36);
        let average = angles
            .iter()
            .map(|&theta| directional(shape.contour(theta).normal, light))
            .sum::<f64>()
            / 1536.0;
        let extended = angles
            .iter()
            .zip(&selected)
            .filter(|&(_, flag)| *flag)
            .map(|(&theta, _)| directional(shape.contour(theta).normal, light))
            .sum::<f64>()
            / 36.0;
        assert!(extended > average + 0.2);
        for i in 0..1536 {
            let a = shape.contour(angles[i]).position;
            let b = shape.contour(angles[(i + 1) % 1536]).position;
            let next = shape.contour(angles[(i + 2) % 1536]).position;
            assert!((length(sub(a, b)) - length(sub(b, next))).abs() < 1e-5);
        }
    }

    #[test]
    fn stationary_preparation_and_workers_preserve_segments_radii_and_order() {
        let config = settings();
        let shapes = petals(&config, [0.0; 2]);
        let first = prepare(&shapes, [0.0; 3], &config, 100.0).unwrap();
        assert_eq!(first.curve_count, 3 * config.corona_hairs);
        assert_eq!(first.segments.len(), 3 * config.corona_hairs * config.corona_segments);
        assert!(first.max_chord_error_pixels < 0.10);
        assert!(first.estimated_luminance > 0.0);
        let second = std::thread::scope(|scope| {
            scope.spawn(|| prepare(&shapes, [0.0; 3], &config, 100.0).unwrap()).join().unwrap()
        });
        assert_eq!(first.segments, second.segments);
        assert_eq!(first.estimated_luminance, second.estimated_luminance);
        let shifted = prepare(&petals(&config, [0.3, -0.7]), [0.0; 3], &config, 100.0).unwrap();
        for (a, b) in first.segments.iter().zip(&shifted.segments) {
            assert_eq!(a.radius, b.radius);
            assert!((config.corona_radii[0]..=config.corona_radii[1]).contains(&a.radius));
            assert_eq!(a.sigma, b.sigma);
            assert!(length(sub(add(a.first, [0.3, -0.7]), b.first)) < 1e-14);
            assert!(a.bounds().iter().flatten().all(|value| value.is_finite()));
            let value = a.radiance(a.first);
            assert!(value.is_finite() && value.min(V3::ZERO) == V3::ZERO);
        }
    }

    #[test]
    fn estimated_unoccluded_budget_scales_with_fraction_and_respects_disabled_bodies() {
        let mut config = settings();
        config.petals[1].enabled = false;
        config.petals[2].enabled = false;
        let shapes = petals(&config, [0.0; 2]);
        let original = prepare(&shapes, [0.0; 3], &config, 100.0).unwrap();
        assert_eq!(original.curve_count, config.corona_hairs);
        let sum = original.segments.iter().map(Segment::estimated_luminance).sum::<f64>();
        assert!((sum - original.estimated_luminance).abs() < 1e-13);
        config.corona_fraction *= 2.0;
        let brighter = prepare(&shapes, [0.0; 3], &config, 100.0).unwrap();
        assert!((brighter.estimated_luminance - 2.0 * original.estimated_luminance).abs() < 1e-12);
        assert_eq!(original.segments.len(), brighter.segments.len());
        for (a, b) in original.segments.iter().zip(&brighter.segments) {
            assert_eq!(a.first, b.first);
            assert_eq!(a.radius, b.radius);
            assert!((b.color - a.color * 2.0).length() < 1e-12);
        }
        config.corona_fraction = 0.0;
        assert!(prepare(&shapes, [0.0; 3], &config, 100.0).unwrap().segments.is_empty());
    }

    #[test]
    fn quadratic_chord_error_is_reported_and_insufficient_fixed_sampling_is_rejected() {
        let mut config = settings();
        let shapes = petals(&config, [0.0; 2]);
        config.corona_segments = 1;
        assert!(prepare(&shapes, [1.0, -1.0, 1.0], &config, 100.0).is_err());
        config.corona_segments = 64;
        let result = prepare(&shapes, [1.0, -1.0, 1.0], &config, 2160.0 / 7.4).unwrap();
        assert!(result.max_chord_error_pixels < 0.003);
        assert!(prepare(&shapes, [f64::NAN, 0.0, 0.0], &config, 100.0).is_err());
    }
}
