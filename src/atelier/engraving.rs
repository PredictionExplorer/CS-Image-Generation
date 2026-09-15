//! Open source-driven rosettes with analytically filtered paired engravings.
//!
//! Fine contour carriers and broad registration fields are combined before
//! filtering. Adaptive spatial and exposure cells bound sampled affine-phase
//! residuals without discarding the resolved difference-frequency pattern.
mod filter;

use super::{Camera, OrbitSeries, RenderConfig, SilkResult, V3};
use filter::PhaseFootprint;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

type Point = [f64; 2];
const LOBE_THREE: f64 = 0.065;
const LOBE_FIVE: f64 = 0.025;
const PATTERN_MAX: f64 = 1.88;
const FILTER_ROUNDOFF: f64 = 1e-12;

/// One fixed open engraving, preserving the original body's identity.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngravingRosetteConfig {
    /// Whether this body's engraving is visible.
    pub enabled: bool,
    /// Positive local ellipse semi-axes in world units.
    pub semi_axes: Point,
    /// Permanent ellipse orientation, in degrees.
    pub angle_degrees: f64,
    /// Permanent local direction of the open mouth, in degrees.
    pub mouth_angle_degrees: f64,
    /// Fixed angular change per local radial unit, in radians; never a time phase.
    /// Omitted at zero to preserve archived untwisted recipe hashes.
    #[serde(skip_serializing_if = "twist_is_zero")]
    pub radial_twist: f64,
    /// Three-lobed cosine and five-lobed sine amplitudes of the radial field.
    /// The original pair is omitted to preserve archived recipe hashes.
    #[serde(skip_serializing_if = "lobes_are_default")]
    pub lobe_amplitudes: Point,
    /// Fine radial phase cycles per normalized radius.
    pub carrier_cycles: f64,
    /// Broad radial difference phase cycles per normalized radius.
    pub beat_cycles: f64,
    /// Broad asymmetric saddle phase coefficient, in cycles.
    pub saddle_cycles: f64,
    /// Nonnegative linear RGB reflection tint.
    pub tint: V3,
    /// Fixed reflection hierarchy for this source body.
    pub strength: f64,
}

impl Default for EngravingRosetteConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            semi_axes: [2.12, 1.17],
            angle_degrees: -24.0,
            mouth_angle_degrees: -38.0,
            radial_twist: 0.0,
            lobe_amplitudes: [LOBE_THREE, LOBE_FIVE],
            carrier_cycles: 64.0,
            beat_cycles: 2.7,
            saddle_cycles: 1.8,
            tint: V3::new(0.72, 0.82, 0.90),
            strength: 1.0,
        }
    }
}

fn twist_is_zero(value: &f64) -> bool {
    *value == 0.0
}

fn lobes_are_default(value: &Point) -> bool {
    *value == [LOBE_THREE, LOBE_FIVE]
}

fn radial_factors(shape: EngravingRosetteConfig) -> Point {
    let [a, b] = shape.lobe_amplitudes.map(f64::abs);
    [1.0 - a - b, 1.0 + a + b]
}

fn radial_derivative_lower_bound(shape: EngravingRosetteConfig, outer_radius: f64) -> f64 {
    let minimum = radial_factors(shape)[0];
    let [a, b] = shape.lobe_amplitudes.map(f64::abs);
    minimum - shape.radial_twist.abs() * (outer_radius / minimum) * (3.0 * a + 5.0 * b)
}

/// Fixed source mapping, engraving composition and integration controls.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngravingConfig {
    /// Fixed orthonormal source basis; any seed-specific fit belongs in the recipe.
    pub source_axes: [V3; 3],
    /// Positive soft scales inside the bounded source mapping.
    pub source_scales: [f64; 3],
    /// Maximum horizontal and vertical source-center displacement.
    pub source_motion: Point,
    /// The three original bodies' permanent engraving parameters.
    pub rosettes: [EngravingRosetteConfig; 3],
    /// Inner zero/full and outer full/zero radii, in increasing order.
    pub radial_edges: [f64; 4],
    /// Half-width of the completely open angular mouth, in degrees.
    pub mouth_half_angle_degrees: f64,
    /// Angular shoulder fade beyond each side of the open mouth.
    pub mouth_feather_degrees: f64,
    /// Fixed world-plane direction of broad reflected light.
    pub light_angle_degrees: f64,
    /// Minimum broad directional reflection, between zero and one.
    pub directional_floor: f64,
    /// Nonnegative common linear-light reflection multiplier.
    pub gain: f64,
    /// Maximum sampled residual of the highest retained harmonic, in radians.
    pub phase_tolerance_radians: f64,
    /// Maximum sampled envelope change within an accepted integration cell.
    pub envelope_tolerance: f64,
    /// Maximum additional quadtree depth below the initial spatial subcells.
    pub max_spatial_depth: usize,
    /// Maximum additional binary exposure subdivisions within the supplied cell.
    pub max_temporal_depth: usize,
}

impl Default for EngravingConfig {
    fn default() -> Self {
        Self {
            source_axes: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)],
            source_scales: [1.5; 3],
            source_motion: [2.7, 1.35],
            rosettes: [
                EngravingRosetteConfig::default(),
                EngravingRosetteConfig {
                    semi_axes: [1.83, 1.45],
                    angle_degrees: 48.0,
                    mouth_angle_degrees: 104.0,
                    carrier_cycles: 74.0,
                    beat_cycles: 3.1,
                    saddle_cycles: -1.3,
                    tint: V3::new(0.35, 0.57, 0.69),
                    strength: 0.8,
                    ..EngravingRosetteConfig::default()
                },
                EngravingRosetteConfig {
                    semi_axes: [1.60, 0.93],
                    angle_degrees: 101.0,
                    mouth_angle_degrees: 234.0,
                    carrier_cycles: 54.0,
                    beat_cycles: 2.3,
                    saddle_cycles: 1.5,
                    tint: V3::new(0.72, 0.50, 0.39),
                    strength: 0.48,
                    ..EngravingRosetteConfig::default()
                },
            ],
            radial_edges: [0.28, 0.33, 0.96, 1.02],
            mouth_half_angle_degrees: 32.0,
            mouth_feather_degrees: 12.0,
            light_angle_degrees: 135.0,
            directional_floor: 0.35,
            gain: 0.85,
            phase_tolerance_radians: 0.03,
            envelope_tolerance: 0.01,
            max_spatial_depth: 5,
            max_temporal_depth: 5,
        }
    }
}

/// Reproducible source and sampled integration-error diagnostics for one raw cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngravingDiagnostics {
    /// Exact requested clamped midpoint fraction, unchanged from the caller.
    pub source_fraction: f64,
    /// Exact unclamped exposure-cell boundaries, unchanged from the caller.
    pub raw_cell_interval: Point,
    /// Current world-plane positions in original source-body order.
    pub centers: [Point; 3],
    /// Current broad registration offset and steering coefficient per body.
    pub registration: [Point; 3],
    /// Parts after splitting the raw interval at the source endpoints.
    pub exposure_parts: usize,
    /// Fraction of raw exposure duration held at the source endpoints.
    pub held_exposure_weight: f64,
    /// Number of initial spatial strata per output-pixel axis.
    pub spatial_subcells: usize,
    /// Accepted analytic integration cells, summed over source bodies.
    pub accepted_cells: u64,
    /// Cells skipped using the continuous source bounds and geometric support.
    pub culled_cells: u64,
    /// Spatial quadtree refinements performed.
    pub spatial_refinements: u64,
    /// Binary exposure refinements performed.
    pub temporal_refinements: u64,
    /// Deepest accepted or inspected spatial refinement.
    pub max_spatial_depth: usize,
    /// Deepest accepted or inspected temporal refinement.
    pub max_temporal_depth: usize,
    /// Largest sampled phase residual before any necessary refinement, in radians.
    pub max_inspected_phase_residual: f64,
    /// Largest sampled residual among accepted affine cells, in radians.
    pub max_accepted_phase_residual: f64,
    /// Largest sampled envelope variation among accepted cells.
    pub max_accepted_envelope_variation: f64,
    /// Required accepted phase tolerance, recorded for independent validation.
    pub phase_tolerance_radians: f64,
    /// Required accepted envelope tolerance, recorded for independent validation.
    pub envelope_tolerance: f64,
    /// Largest sampled individual-groove spatial gradient, in cycles/final pixel.
    pub max_groove_cycles_per_pixel: f64,
    /// Largest inspected combined harmonic exposure excursion, in cycles.
    pub max_temporal_harmonic_excursion: f64,
    /// Smallest raw filtered material value, including roundoff before correction.
    pub minimum_filtered_pattern: f64,
    /// Tiny negative filter roundoff values corrected to zero.
    pub rounded_negative_values: u64,
}

impl EngravingDiagnostics {
    /// Reject non-finite, mistimed or unconverged integration records.
    pub fn validate(&self) -> SilkResult<()> {
        let finite = [
            self.source_fraction,
            self.raw_cell_interval[0],
            self.raw_cell_interval[1],
            self.held_exposure_weight,
            self.max_inspected_phase_residual,
            self.max_accepted_phase_residual,
            self.max_accepted_envelope_variation,
            self.phase_tolerance_radians,
            self.envelope_tolerance,
            self.max_groove_cycles_per_pixel,
            self.max_temporal_harmonic_excursion,
            self.minimum_filtered_pattern,
        ];
        if !finite.iter().all(|v| v.is_finite())
            || !self.centers.iter().chain(&self.registration).flatten().all(|v| v.is_finite())
            || !(0.0..=1.0).contains(&self.source_fraction)
            || self.raw_cell_interval[0] > self.raw_cell_interval[1]
            || !(0.0..=1.0).contains(&self.held_exposure_weight)
            || !(1..=3).contains(&self.exposure_parts)
            || !(1..=8).contains(&self.spatial_subcells)
            || self.phase_tolerance_radians <= 0.0
            || self.envelope_tolerance <= 0.0
            || self.minimum_filtered_pattern < -FILTER_ROUNDOFF
            || self.minimum_filtered_pattern > PATTERN_MAX + FILTER_ROUNDOFF
            || self.max_spatial_depth > 8
            || self.max_temporal_depth > 8
            || self.max_inspected_phase_residual < self.max_accepted_phase_residual
            || self.max_accepted_phase_residual > self.phase_tolerance_radians * (1.0 + 1e-10)
            || self.max_accepted_envelope_variation > self.envelope_tolerance * (1.0 + 1e-10)
            || [
                self.max_inspected_phase_residual,
                self.max_accepted_phase_residual,
                self.max_accepted_envelope_variation,
                self.max_groove_cycles_per_pixel,
                self.max_temporal_harmonic_excursion,
            ]
            .iter()
            .any(|v| *v < 0.0)
        {
            return Err(
                "engraving diagnostics contain an invalid or unconverged integration record".into(),
            );
        }
        let midpoint =
            (self.raw_cell_interval[0] * 0.5 + self.raw_cell_interval[1] * 0.5).clamp(0.0, 1.0);
        if (midpoint - self.source_fraction).abs() > 1e-12 {
            return Err("engraving midpoint does not match its raw exposure cell".into());
        }
        let [start, end] = self.raw_cell_interval;
        let width = end - start;
        let parts =
            1 + usize::from(start < 0.0 && end > 0.0) + usize::from(start < 1.0 && end > 1.0);
        let held = if width == 0.0 {
            if start <= 0.0 || start >= 1.0 { 1.0 } else { 0.0 }
        } else {
            ((end.min(0.0) - start).max(0.0) + (end - start.max(1.0)).max(0.0)) / width
        };
        if self.exposure_parts != parts || (self.held_exposure_weight - held).abs() > 1e-12 {
            return Err(
                "engraving endpoint-hold budget does not match its raw exposure interval".into()
            );
        }
        Ok(())
    }
}

/// One untone-mapped linear frame and its complete integration record.
pub struct EngravingFrame {
    /// Linear RGB pixels in row-major order.
    pub pixels: Vec<V3>,
    /// Exact source time and sampled spatial/temporal convergence evidence.
    pub diagnostics: EngravingDiagnostics,
}

#[derive(Clone, Copy, Debug)]
struct Plane {
    target: Point,
    step_x: Point,
    step_y: Point,
    size: Point,
}

impl Plane {
    fn new(camera: &Camera, render: &RenderConfig) -> SilkResult<Self> {
        if !camera.position.is_finite()
            || !camera.target.is_finite()
            || !camera.up.is_finite()
            || !camera.orthographic_height.is_finite()
            || camera.orthographic_height <= 0.0
        {
            return Err("engraving camera must be finite with positive orthographic height".into());
        }
        let forward = (camera.target - camera.position).normalized();
        if forward.z.abs() < 1.0 - 1e-10 || forward.x.abs() > 1e-8 || forward.y.abs() > 1e-8 {
            return Err(
                "engraving camera must view the XY plane frontally; crop and roll are supported"
                    .into(),
            );
        }
        let right = forward.cross(camera.up).normalized();
        if right.length_squared() < 0.5 {
            return Err("engraving camera up must not be parallel to its view".into());
        }
        let up = right.cross(forward).normalized();
        let pixel = camera.orthographic_height / f64::from(render.height);
        Ok(Self {
            target: [camera.target.x, camera.target.y],
            step_x: [right.x * pixel, right.y * pixel],
            step_y: [-up.x * pixel, -up.y * pixel],
            size: [f64::from(render.width), f64::from(render.height)],
        })
    }

    fn world(self, pixel: Point) -> Point {
        std::array::from_fn(|a| {
            self.target[a]
                + (pixel[0] - self.size[0] * 0.5) * self.step_x[a]
                + (pixel[1] - self.size[1] * 0.5) * self.step_y[a]
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BodyState {
    center: Point,
    delta: f64,
    steer: f64,
}

fn source_states(
    source: &OrbitSeries,
    time: f64,
    config: &EngravingConfig,
) -> SilkResult<[BodyState; 3]> {
    let frame = source.sample(time).ok_or("engraving source time is outside the recording")?;
    let q: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|a| {
            (frame.bodies[i].position.dot(config.source_axes[a]) / config.source_scales[a]).tanh()
        })
    });
    Ok(std::array::from_fn(|i| BodyState {
        center: [config.source_motion[0] * q[i][0], config.source_motion[1] * q[i][1]],
        delta: 0.70 * q[i][2] + 1.05 * (q[(i + 1) % 3][1] - q[i][1]),
        steer: 0.55 * (q[(i + 1) % 3][0] - q[i][0]),
    }))
}

#[derive(Clone, Debug)]
struct TimeNode {
    samples: [[BodyState; 3]; 5],
    bounds: [[Point; 2]; 3],
    children: Option<[usize; 2]>,
    depth: usize,
}

struct Exposure {
    nodes: Vec<TimeNode>,
    weight: f64,
    held: bool,
}

struct TimePreparation<'a> {
    source: &'a OrbitSeries,
    config: &'a EngravingConfig,
    states: &'a [[BodyState; 3]],
    times: &'a [f64],
    max_depth: usize,
}

impl TimePreparation<'_> {
    fn build(
        &self,
        start: usize,
        end: usize,
        depth: usize,
        nodes: &mut Vec<TimeNode>,
    ) -> SilkResult<usize> {
        let indices = std::array::from_fn::<_, 5, _>(|i| start + (end - start) * i / 4);
        let mut bounds = [[[0.0; 2]; 2]; 3];
        for (body, target) in bounds.iter_mut().enumerate() {
            let (low, high) = self
                .source
                .position_bounds(body, self.times[start], self.times[end])
                .ok_or("engraving cannot bound the continuous recorded source interval")?;
            *target = projected_source_bounds(low, high, self.config);
        }
        let index = nodes.len();
        nodes.push(TimeNode {
            samples: indices.map(|i| self.states[i]),
            bounds,
            children: None,
            depth,
        });
        if depth < self.max_depth {
            let mid = usize::midpoint(start, end);
            let left = self.build(start, mid, depth + 1, nodes)?;
            let right = self.build(mid, end, depth + 1, nodes)?;
            nodes[index].children = Some([left, right]);
        }
        Ok(index)
    }
}

fn projected_source_bounds(low: V3, high: V3, config: &EngravingConfig) -> [Point; 2] {
    let mut result = [[0.0; 2]; 2];
    for (axis, &motion) in config.source_motion.iter().enumerate() {
        if motion == 0.0 {
            continue;
        }
        let mut minimum = 0.0;
        let mut maximum = 0.0;
        for component in 0..3 {
            let coefficient = config.source_axes[axis].axis(component);
            let a = coefficient * low.axis(component);
            let b = coefficient * high.axis(component);
            minimum += a.min(b);
            maximum += a.max(b);
        }
        // Outward padding covers projection arithmetic before the monotone
        // positive-scale/tanh map; the source helper already bounds its cubic.
        let padding = 32.0 * f64::EPSILON * (1.0 + minimum.abs().max(maximum.abs()));
        result[0][axis] =
            (((minimum - padding) / config.source_scales[axis]).tanh() * motion).next_down();
        result[1][axis] =
            (((maximum + padding) / config.source_scales[axis]).tanh() * motion).next_up();
    }
    result
}

fn prepare_exposures(
    source: &OrbitSeries,
    raw: Point,
    config: &EngravingConfig,
) -> SilkResult<Vec<Exposure>> {
    let width = raw[1] - raw[0];
    let mut cuts = vec![raw[0]];
    for endpoint in [0.0, 1.0] {
        if raw[0] < endpoint && endpoint < raw[1] {
            cuts.push(endpoint);
        }
    }
    cuts.push(raw[1]);
    let mut exposures = Vec::new();
    for pair in cuts.windows(2) {
        let start = pair[0].clamp(0.0, 1.0);
        let end = pair[1].clamp(0.0, 1.0);
        let constant = start == end;
        let held = constant && (pair[1] <= 0.0 || pair[0] >= 1.0);
        let depth = if constant { 0 } else { config.max_temporal_depth };
        let count = 1_usize << (depth + 2);
        let times: Vec<_> =
            (0..=count).map(|i| start + (end - start) * i as f64 / count as f64).collect();
        let states: Vec<_> = times
            .iter()
            .map(|&time| source_states(source, time, config))
            .collect::<SilkResult<_>>()?;
        let mut nodes = Vec::new();
        TimePreparation { source, config, states: &states, times: &times, max_depth: depth }
            .build(0, count, 0, &mut nodes)?;
        exposures.push(Exposure {
            nodes,
            weight: if width == 0.0 { 1.0 } else { (pair[1] - pair[0]) / width },
            held,
        });
    }
    Ok(exposures)
}

#[derive(Clone, Copy)]
struct Rosette {
    config: EngravingRosetteConfig,
    cosine: f64,
    sine: f64,
    mouth: Point,
    mouth_full_cos: f64,
    mouth_zero_cos: f64,
    light: Point,
    radial_factors: Point,
}

impl Rosette {
    fn new(shape: EngravingRosetteConfig, config: &EngravingConfig) -> Self {
        let (sine, cosine) = shape.angle_degrees.to_radians().sin_cos();
        let (ms, mc) = shape.mouth_angle_degrees.to_radians().sin_cos();
        let (ls, lc) = config.light_angle_degrees.to_radians().sin_cos();
        Self {
            config: shape,
            cosine,
            sine,
            mouth: [mc, ms],
            mouth_full_cos: config.mouth_half_angle_degrees.to_radians().cos(),
            mouth_zero_cos: (config.mouth_half_angle_degrees + config.mouth_feather_degrees)
                .to_radians()
                .cos(),
            light: [lc, ls],
            radial_factors: radial_factors(shape),
        }
    }

    fn local(self, point: Point, center: Point) -> Point {
        let x = point[0] - center[0];
        let y = point[1] - center[1];
        [
            (self.cosine * x + self.sine * y) / self.config.semi_axes[0],
            (-self.sine * x + self.cosine * y) / self.config.semi_axes[1],
        ]
    }

    fn world_gradient(self, local: Point) -> Point {
        let x = local[0] / self.config.semi_axes[0];
        let y = local[1] / self.config.semi_axes[1];
        [self.cosine * x - self.sine * y, self.sine * x + self.cosine * y]
    }

    fn warped_direction(self, direction: Point, radius: f64) -> Point {
        // The explicit zero path retains the archived phase and mouth arithmetic.
        if self.config.radial_twist == 0.0 {
            return direction;
        }
        let (sine, cosine) = (self.config.radial_twist * radius).sin_cos();
        [cosine * direction[0] - sine * direction[1], sine * direction[0] + cosine * direction[1]]
    }

    fn field(self, point: Point, state: BodyState, config: &EngravingConfig) -> Field {
        let [u, v] = self.local(point, state.center);
        let r = u.hypot(v);
        if r < 1e-12 {
            return Field::default();
        }
        let [c, s] = [u / r, v / r];
        let [wc, ws] = self.warped_direction([c, s], r);
        let c3 = wc * (wc * wc - 3.0 * ws * ws);
        let s3 = ws * (3.0 * wc * wc - ws * ws);
        let c5 = wc * (wc.powi(4) - 10.0 * wc * wc * ws * ws + 5.0 * ws.powi(4));
        let s5 = ws * (5.0 * wc.powi(4) - 10.0 * wc * wc * ws * ws + ws.powi(4));
        let [a, b] = self.config.lobe_amplitudes;
        let radial = 1.0 + a * c3 + b * s5;
        let angular = -3.0 * a * s3 + 5.0 * b * c5;
        let rho = r * radial;
        let rho_gradient = if self.config.radial_twist == 0.0 {
            [radial * c - angular * s, radial * s + angular * c]
        } else {
            let derivative = radial + self.config.radial_twist * r * angular;
            // The polar basis remains the original direction. Only the lobe
            // phase rotates; the chain rule adds twist*r*angular radially.
            [derivative * c - angular * s, derivative * s + angular * c]
        };
        let denominator = 1.0 + u * u + v * v;
        let numerator = self.config.saddle_cycles * (u * u - v * v) + 1.5 * u * v;
        let num_gradient = [
            2.0 * self.config.saddle_cycles * u + 1.5 * v,
            -2.0 * self.config.saddle_cycles * v + 1.5 * u,
        ];
        let phi_gradient =
            self.world_gradient(rho_gradient.map(|x| self.config.carrier_cycles * x));
        let psi_gradient = self.world_gradient(std::array::from_fn(|a| {
            self.config.beat_cycles * rho_gradient[a] + num_gradient[a] / denominator
                - numerator * 2.0 * [u, v][a] / denominator.powi(2)
                + if a == 0 { state.steer } else { 0.0 }
        }));
        let world_rho = self.world_gradient(rho_gradient);
        let normal_length = world_rho[0].hypot(world_rho[1]);
        let facing = (0.5 + 0.5 * dot(world_rho, self.light) / normal_length).clamp(0.0, 1.0);
        let direction =
            config.directional_floor + (1.0 - config.directional_floor) * facing * facing;
        let edges = config.radial_edges;
        let radial_envelope =
            smoothstep(edges[0], edges[1], rho) * (1.0 - smoothstep(edges[2], edges[3], rho));
        let gap = 1.0
            - smoothstep(
                self.mouth_zero_cos,
                self.mouth_full_cos,
                wc * self.mouth[0] + ws * self.mouth[1],
            );
        Field {
            phi: self.config.carrier_cycles * rho,
            psi: self.config.beat_cycles * rho
                + numerator / denominator
                + state.steer * u
                + state.delta,
            phi_gradient,
            psi_gradient,
            envelope: radial_envelope * gap * direction,
        }
    }

    fn empty(
        self,
        point: Point,
        spatial_width: f64,
        plane: Plane,
        time: &TimeNode,
        body: usize,
        config: &EngravingConfig,
    ) -> bool {
        let center = time.samples[2][body].center;
        let local = self.local(point, center);
        let r = local[0].hypot(local[1]);
        let bounds = time.bounds[body];
        let movement = (bounds[0][0] - center[0])
            .abs()
            .max((bounds[1][0] - center[0]).abs())
            .hypot((bounds[0][1] - center[1]).abs().max((bounds[1][1] - center[1]).abs()));
        let pixel_radius =
            spatial_width * 0.5 * 2.0_f64.sqrt() * plane.step_x[0].hypot(plane.step_x[1]);
        let radius =
            (movement + pixel_radius) / self.config.semi_axes[0].min(self.config.semi_axes[1]);
        if (r + radius) * self.radial_factors[1] < config.radial_edges[0]
            || (r - radius).max(0.0) * self.radial_factors[0] > config.radial_edges[3]
        {
            return true;
        }
        if r > radius && r > 1e-12 {
            let (angle, uncertainty) = if self.config.radial_twist == 0.0 {
                ((dot(local, self.mouth) / r).clamp(-1.0, 1.0).acos(), (radius / r).asin())
            } else {
                let direction = self.warped_direction(local.map(|x| x / r), r);
                (
                    dot(direction, self.mouth).clamp(-1.0, 1.0).acos(),
                    (radius / r).asin() + self.config.radial_twist.abs() * radius,
                )
            };
            if angle + uncertainty < config.mouth_half_angle_degrees.to_radians() {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Copy, Default)]
struct Field {
    phi: f64,
    psi: f64,
    phi_gradient: Point,
    psi_gradient: Point,
    envelope: f64,
}

struct Statistics {
    accepted: u64,
    culled: u64,
    spatial: u64,
    temporal: u64,
    negative: u64,
    spatial_depth: usize,
    temporal_depth: usize,
    inspected_phase: f64,
    accepted_phase: f64,
    accepted_envelope: f64,
    groove_gradient: f64,
    temporal_excursion: f64,
    minimum: f64,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            accepted: 0,
            culled: 0,
            spatial: 0,
            temporal: 0,
            negative: 0,
            spatial_depth: 0,
            temporal_depth: 0,
            inspected_phase: 0.0,
            accepted_phase: 0.0,
            accepted_envelope: 0.0,
            groove_gradient: 0.0,
            temporal_excursion: 0.0,
            minimum: f64::INFINITY,
        }
    }
}

impl Statistics {
    fn merge(&mut self, other: &Self) {
        self.accepted += other.accepted;
        self.culled += other.culled;
        self.spatial += other.spatial;
        self.temporal += other.temporal;
        self.negative += other.negative;
        self.spatial_depth = self.spatial_depth.max(other.spatial_depth);
        self.temporal_depth = self.temporal_depth.max(other.temporal_depth);
        self.inspected_phase = self.inspected_phase.max(other.inspected_phase);
        self.accepted_phase = self.accepted_phase.max(other.accepted_phase);
        self.accepted_envelope = self.accepted_envelope.max(other.accepted_envelope);
        self.groove_gradient = self.groove_gradient.max(other.groove_gradient);
        self.temporal_excursion = self.temporal_excursion.max(other.temporal_excursion);
        self.minimum = self.minimum.min(other.minimum);
    }
}

#[derive(Clone, Copy)]
struct Cell {
    pixel: Point,
    width: f64,
    depth: usize,
    time: usize,
}

struct Integrator<'a> {
    config: &'a EngravingConfig,
    plane: Plane,
    rosette: Rosette,
    exposure: &'a Exposure,
    body: usize,
}

impl Integrator<'_> {
    fn cell(&self, cell: Cell, stats: &mut Statistics) -> SilkResult<f64> {
        let time = &self.exposure.nodes[cell.time];
        let point = self.plane.world(cell.pixel);
        stats.spatial_depth = stats.spatial_depth.max(cell.depth);
        stats.temporal_depth = stats.temporal_depth.max(time.depth);
        if self.rosette.empty(point, cell.width, self.plane, time, self.body, self.config) {
            stats.culled += 1;
            return Ok(0.0);
        }
        let center = self.rosette.field(point, time.samples[2][self.body], self.config);
        let first = self.rosette.field(point, time.samples[0][self.body], self.config);
        let last = self.rosette.field(point, time.samples[4][self.body], self.config);
        let phi = PhaseFootprint {
            value: center.phi,
            delta: [
                dot(center.phi_gradient, self.plane.step_x) * cell.width,
                dot(center.phi_gradient, self.plane.step_y) * cell.width,
                last.phi - first.phi,
            ],
        };
        let psi = PhaseFootprint {
            value: center.psi,
            delta: [
                dot(center.psi_gradient, self.plane.step_x) * cell.width,
                dot(center.psi_gradient, self.plane.step_y) * cell.width,
                last.psi - first.psi,
            ],
        };
        let mut spatial_error: f64 = 0.0;
        let mut temporal_error: f64 = 0.0;
        let mut envelope_error: f64 = 0.0;
        let mut spatial_envelope: f64 = 0.0;
        let mut temporal_envelope: f64 = 0.0;
        let mut inspect = |value: Field, offset: [f64; 3], temporal: bool| {
            let phi_residual = value.phi - center.phi - dot3(phi.delta, offset);
            let psi_residual = value.psi - center.psi - dot3(psi.delta, offset);
            let residual = TAU * 8.0 * (2.0 * phi_residual.abs() + psi_residual.abs());
            let envelope = (value.envelope - center.envelope).abs();
            if temporal {
                temporal_error = temporal_error.max(residual);
                temporal_envelope = temporal_envelope.max(envelope);
            } else {
                spatial_error = spatial_error.max(residual);
                spatial_envelope = spatial_envelope.max(envelope);
            }
            envelope_error = envelope_error.max(envelope);
        };
        for (index, frame) in time.samples.iter().enumerate() {
            if index != 2 {
                let value = self.rosette.field(point, frame[self.body], self.config);
                inspect(value, [0.0, 0.0, index as f64 * 0.25 - 0.5], true);
            }
        }
        for sy in [-0.5, 0.0, 0.5] {
            for sx in [-0.5, 0.0, 0.5] {
                if sx == 0.0 && sy == 0.0 {
                    continue;
                }
                let shifted = self
                    .plane
                    .world([cell.pixel[0] + sx * cell.width, cell.pixel[1] + sy * cell.width]);
                inspect(
                    self.rosette.field(shifted, time.samples[2][self.body], self.config),
                    [sx, sy, 0.0],
                    false,
                );
                if sx != 0.0 && sy != 0.0 {
                    for (index, offset) in [(0, -0.5), (4, 0.5)] {
                        inspect(
                            self.rosette.field(
                                shifted,
                                time.samples[index][self.body],
                                self.config,
                            ),
                            [sx, sy, offset],
                            true,
                        );
                    }
                }
            }
        }
        let residual = spatial_error.max(temporal_error);
        stats.inspected_phase = stats.inspected_phase.max(residual);
        stats.temporal_excursion =
            stats.temporal_excursion.max(8.0 * (2.0 * phi.delta[2].abs() + psi.delta[2].abs()));
        for sign in [-0.5, 0.5] {
            let gx = (phi.delta[0] + sign * psi.delta[0]) / cell.width;
            let gy = (phi.delta[1] + sign * psi.delta[1]) / cell.width;
            stats.groove_gradient = stats.groove_gradient.max(gx.hypot(gy));
        }
        if residual > self.config.phase_tolerance_radians
            || envelope_error > self.config.envelope_tolerance
        {
            let can_spatial = cell.depth < self.config.max_spatial_depth;
            let can_temporal = time.children.is_some();
            let spatial_score = (spatial_error / self.config.phase_tolerance_radians)
                .max(spatial_envelope / self.config.envelope_tolerance);
            let temporal_score = (temporal_error / self.config.phase_tolerance_radians)
                .max(temporal_envelope / self.config.envelope_tolerance);
            if can_temporal && (!can_spatial || temporal_score > spatial_score * 1.25) {
                stats.temporal += 1;
                let [left, right] = time.children.expect("temporal refinement checked above");
                return Ok(0.5
                    * (self.cell(Cell { time: left, ..cell }, stats)?
                        + self.cell(Cell { time: right, ..cell }, stats)?));
            }
            if can_spatial {
                stats.spatial += 1;
                let mut sum = 0.0;
                for sy in [-0.25, 0.25] {
                    for sx in [-0.25, 0.25] {
                        sum += self.cell(
                            Cell {
                                pixel: [
                                    cell.pixel[0] + sx * cell.width,
                                    cell.pixel[1] + sy * cell.width,
                                ],
                                width: cell.width * 0.5,
                                depth: cell.depth + 1,
                                ..cell
                            },
                            stats,
                        )?;
                    }
                }
                return Ok(sum * 0.25);
            }
            return Err(format!("engraving refinement exhausted at pixel {:?}, body {}, spatial depth {}, temporal depth {}: phase residual {residual}, envelope variation {envelope_error}",
                cell.pixel, self.body, cell.depth, time.depth).into());
        }
        let value = filter::evaluate(phi, psi).value;
        if !value.is_finite()
            || !(-FILTER_ROUNDOFF..=PATTERN_MAX + FILTER_ROUNDOFF).contains(&value)
        {
            return Err(format!("engraving filter returned out-of-range pattern {value}").into());
        }
        stats.accepted += 1;
        stats.minimum = stats.minimum.min(value);
        stats.negative += u64::from(value < 0.0);
        stats.accepted_phase = stats.accepted_phase.max(residual);
        stats.accepted_envelope = stats.accepted_envelope.max(envelope_error);
        Ok(value.max(0.0) * center.envelope)
    }
}

/// Integrate one raw exposure cell, including any held source-endpoint portions.
///
/// The caller supplies the unchanged clamped midpoint and unclamped interval.
/// Spatial strata and adaptive exposure subcells integrate the complete paired
/// phase field; output remains linear for the shared shutter/postprocess path.
pub fn render_linear(
    source: &OrbitSeries,
    center_time: f64,
    raw_cell_interval: Point,
    config: &EngravingConfig,
    camera: &Camera,
    render: &RenderConfig,
) -> SilkResult<EngravingFrame> {
    validate(center_time, raw_cell_interval, config, render)?;
    let plane = Plane::new(camera, render)?;
    let current = source_states(source, center_time, config)?;
    let exposures = prepare_exposures(source, raw_cell_interval, config)?;
    let rosettes = config.rosettes.map(|shape| Rosette::new(shape, config));
    let mut pixels = vec![render.background; render.width as usize * render.height as usize];
    let rows: Vec<SilkResult<Statistics>> = pixels
        .par_chunks_mut(render.width as usize)
        .enumerate()
        .map(|(y, row)| {
            let mut stats = Statistics::default();
            for (x, pixel) in row.iter_mut().enumerate() {
                let mut color = render.background;
                for (body, &rosette) in rosettes.iter().enumerate() {
                    if !rosette.config.enabled
                        || rosette.config.strength == 0.0
                        || config.gain == 0.0
                    {
                        continue;
                    }
                    let mut signal = 0.0;
                    for exposure in &exposures {
                        let integrator = Integrator { config, plane, rosette, exposure, body };
                        let count = render.aa as usize;
                        let mut subtotal = 0.0;
                        for sy in 0..count {
                            for sx in 0..count {
                                subtotal += integrator.cell(
                                    Cell {
                                        pixel: [
                                            x as f64 + (sx as f64 + 0.5) / count as f64,
                                            y as f64 + (sy as f64 + 0.5) / count as f64,
                                        ],
                                        width: 1.0 / count as f64,
                                        depth: 0,
                                        time: 0,
                                    },
                                    &mut stats,
                                )?;
                            }
                        }
                        signal += exposure.weight * subtotal / (count * count) as f64;
                    }
                    color += rosette.config.tint * (signal * config.gain * rosette.config.strength);
                }
                if !color.is_finite() {
                    return Err("engraving radiance exceeded finite RGB range".into());
                }
                *pixel = color;
            }
            Ok(stats)
        })
        .collect();
    let mut stats = Statistics::default();
    for row in rows {
        stats.merge(&row?);
    }
    let diagnostics = EngravingDiagnostics {
        source_fraction: center_time,
        raw_cell_interval,
        centers: current.map(|state| state.center),
        registration: current.map(|state| [state.delta, state.steer]),
        exposure_parts: exposures.len(),
        held_exposure_weight: exposures
            .iter()
            .filter(|e| e.held)
            .map(|e| e.weight)
            .sum::<f64>()
            .min(1.0),
        spatial_subcells: render.aa as usize,
        accepted_cells: stats.accepted,
        culled_cells: stats.culled,
        spatial_refinements: stats.spatial,
        temporal_refinements: stats.temporal,
        max_spatial_depth: stats.spatial_depth,
        max_temporal_depth: stats.temporal_depth,
        max_inspected_phase_residual: stats.inspected_phase,
        max_accepted_phase_residual: stats.accepted_phase,
        max_accepted_envelope_variation: stats.accepted_envelope,
        phase_tolerance_radians: config.phase_tolerance_radians,
        envelope_tolerance: config.envelope_tolerance,
        max_groove_cycles_per_pixel: stats.groove_gradient,
        max_temporal_harmonic_excursion: stats.temporal_excursion,
        minimum_filtered_pattern: if stats.accepted == 0 { 0.0 } else { stats.minimum },
        rounded_negative_values: stats.negative,
    };
    diagnostics.validate()?;
    Ok(EngravingFrame { pixels, diagnostics })
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t.powi(3) * (t * (t * 6.0 - 15.0) + 10.0)
}

fn validate(
    time: f64,
    interval: Point,
    config: &EngravingConfig,
    render: &RenderConfig,
) -> SilkResult<()> {
    if !time.is_finite()
        || !(0.0..=1.0).contains(&time)
        || !interval.iter().all(|x| x.is_finite())
        || interval[0] > interval[1]
        || interval[1] - interval[0] > 1.0
        || interval[0] < -1.0
        || interval[1] > 2.0
        || ((interval[0] * 0.5 + interval[1] * 0.5).clamp(0.0, 1.0) - time).abs() > 1e-12
    {
        return Err(
            "engraving needs a valid recorded midpoint and matching raw exposure cell".into()
        );
    }
    if render.width == 0
        || render.height == 0
        || (render.width as usize)
            .checked_mul(render.height as usize)
            .is_none_or(|n| n > 100_000_000)
        || !render.background.is_finite()
        || render.background.min(V3::ZERO) != V3::ZERO
        || !(1..=8).contains(&render.aa)
        || config.max_spatial_depth > 8
        || config.max_temporal_depth > 8
    {
        return Err("engraving image or refinement settings exceed supported bounds".into());
    }
    for axis in 0..3 {
        if !config.source_axes[axis].is_finite()
            || (config.source_axes[axis].length_squared() - 1.0).abs() > 1e-6
            || config
                .source_axes
                .iter()
                .skip(axis + 1)
                .any(|other| config.source_axes[axis].dot(*other).abs() > 1e-6)
        {
            return Err("engraving source_axes must be a fixed orthonormal basis".into());
        }
    }
    if config
        .source_scales
        .iter()
        .chain(config.radial_edges.iter())
        .any(|x| !x.is_finite() || *x <= 0.0)
        || config.radial_edges.windows(2).any(|p| p[0] >= p[1])
        || config.radial_edges[0] < 0.05
        || config.radial_edges[3] > 4.0
        || config
            .source_motion
            .iter()
            .chain([&config.gain])
            .any(|x| !x.is_finite() || *x < 0.0 || *x > 100.0)
        || !config.directional_floor.is_finite()
        || !(0.0..=1.0).contains(&config.directional_floor)
        || !config.light_angle_degrees.is_finite()
        || [config.phase_tolerance_radians, config.envelope_tolerance]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0 || *x > 1.0)
        || [config.mouth_half_angle_degrees, config.mouth_feather_degrees]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        || config.mouth_half_angle_degrees + config.mouth_feather_degrees >= 170.0
    {
        return Err("engraving scalar parameters need finite, ordered and bounded values".into());
    }
    for shape in config.rosettes {
        if shape.semi_axes.iter().any(|x| !x.is_finite() || *x < 0.05 || *x > 100.0)
            || !shape.carrier_cycles.is_finite()
            || !(0.0..=100_000.0).contains(&shape.carrier_cycles)
            || !shape.beat_cycles.is_finite()
            || shape.beat_cycles.abs() > 1000.0
            || !shape.saddle_cycles.is_finite()
            || shape.saddle_cycles.abs() > 1000.0
            || !shape.angle_degrees.is_finite()
            || !shape.mouth_angle_degrees.is_finite()
            || !shape.radial_twist.is_finite()
            || shape.radial_twist.abs() > 4.0
            || shape.lobe_amplitudes.iter().any(|x| !x.is_finite())
            || radial_factors(shape)[0] <= 0.0
            || radial_derivative_lower_bound(shape, config.radial_edges[3]) <= 0.10
            || !shape.strength.is_finite()
            || !(0.0..=100.0).contains(&shape.strength)
            || !shape.tint.is_finite()
            || shape.tint.min(V3::ZERO) != V3::ZERO
            || shape.tint.x.max(shape.tint.y).max(shape.tint.z) > 100.0
        {
            return Err("engraving rosettes need finite geometry, phase and reflection values, and a radial derivative lower bound above 0.10 throughout their support".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;

    fn source(moving: bool) -> OrbitSeries {
        let samples = (0..129)
            .map(|step| {
                let t = if moving { f64::from(step) / 128.0 } else { 0.0 };
                std::array::from_fn(|body| {
                    if !moving {
                        return V3::ZERO;
                    }
                    let a = TAU * (t + body as f64 / 3.0);
                    V3::new(a.cos(), a.sin(), 0.3 * (1.3 * a).sin())
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "engraving-test".into(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({"synthetic_unit_fixture":true}),
        })
        .unwrap()
    }

    fn proof_config() -> EngravingConfig {
        let mut config = EngravingConfig {
            source_motion: [0.0; 2],
            phase_tolerance_radians: 0.06,
            envelope_tolerance: 0.025,
            max_spatial_depth: 5,
            max_temporal_depth: 5,
            ..EngravingConfig::default()
        };
        for shape in &mut config.rosettes {
            shape.enabled = false;
        }
        config.rosettes[0] = EngravingRosetteConfig {
            semi_axes: [1.0, 1.0],
            angle_degrees: 0.0,
            mouth_angle_degrees: 0.0,
            carrier_cycles: 2.0,
            beat_cycles: 0.8,
            saddle_cycles: 0.3,
            ..EngravingRosetteConfig::default()
        };
        config
    }

    fn camera() -> Camera {
        Camera {
            position: V3::new(-0.68, 0.10, 10.0),
            target: V3::new(-0.68, 0.10, 0.0),
            up: V3::new(0.0, 1.0, 0.0),
            orthographic_height: 0.45,
        }
    }

    fn render_config() -> RenderConfig {
        RenderConfig {
            width: 12,
            height: 8,
            aa: 2,
            background: V3::new(0.003, 0.0018, 0.0048),
            ..RenderConfig::default()
        }
    }

    #[test]
    fn source_centers_and_registration_use_the_unchanged_current_samples() {
        let source = source(true);
        let config = EngravingConfig::default();
        for time in [0.0, 0.37, 1.0] {
            let frame = source.sample(time).unwrap();
            let states = source_states(&source, time, &config).unwrap();
            for (body, state) in states.iter().enumerate() {
                let p = frame.bodies[body].position;
                assert_eq!(state.center, [2.7 * (p.x / 1.5).tanh(), 1.35 * (p.y / 1.5).tanh()]);
                let next = frame.bodies[(body + 1) % 3].position;
                assert_eq!(
                    state.delta,
                    0.70 * (p.z / 1.5).tanh() + 1.05 * ((next.y / 1.5).tanh() - (p.y / 1.5).tanh())
                );
            }
        }
        let a = source_states(&source, 0.0, &config).unwrap();
        let b = source_states(&source, 0.4, &config).unwrap();
        assert!((a[0].center[0] - b[0].center[0]).abs() > 1.0);
        assert!((a[0].delta - b[0].delta).abs() > 0.1);
    }

    #[test]
    fn contour_and_registration_gradients_match_finite_differences_without_angular_seams() {
        let config = EngravingConfig::default();
        let rosette = Rosette::new(config.rosettes[1], &config);
        let state = BodyState { center: [0.1, -0.2], delta: 0.8, steer: -0.3 };
        let h = 1e-6;
        for point in [[-0.9, 0.3], [0.5, -1.1], [1.5, 0.6], [-0.5, -0.9]] {
            let value = rosette.field(point, state, &config);
            for axis in 0..2 {
                let mut left = point;
                left[axis] -= h;
                let mut right = point;
                right[axis] += h;
                let a = rosette.field(left, state, &config);
                let b = rosette.field(right, state, &config);
                assert!(((b.phi - a.phi) / (2.0 * h) - value.phi_gradient[axis]).abs() < 2e-7);
                assert!(((b.psi - a.psi) / (2.0 * h) - value.psi_gradient[axis]).abs() < 2e-8);
            }
        }
        let mut straight = config.rosettes[0];
        straight.angle_degrees = 0.0;
        let rosette = Rosette::new(straight, &config);
        let a = rosette.field([-1.0, -1e-10], BodyState::default(), &config);
        let b = rosette.field([-1.0, 1e-10], BodyState::default(), &config);
        assert!((a.phi - b.phi).abs() < 1e-7);
        assert!((a.envelope - b.envelope).abs() < 1e-7);
    }

    #[test]
    fn annular_openings_and_angular_mouth_have_no_filled_underlay() {
        let config = proof_config();
        let shape = Rosette::new(config.rosettes[0], &config);
        for point in [[0.0, 0.0], [0.1, 0.1], [2.0, 0.0], [0.6, 0.0]] {
            assert_eq!(shape.field(point, BodyState::default(), &config).envelope, 0.0);
        }
        assert!(shape.field([-0.6, 0.0], BodyState::default(), &config).envelope > 0.0);
        for step in 0..1000 {
            let angle = TAU * f64::from(step) / 1000.0;
            let (s, c) = angle.sin_cos();
            let point = [0.7 * c, 0.7 * s];
            let field = shape.field(point, BodyState::default(), &config);
            assert!((0.0..=1.0).contains(&field.envelope));
            assert!(field.phi_gradient[0].hypot(field.phi_gradient[1]) > 0.1);
        }
    }

    #[test]
    fn raw_exposure_splits_preserve_held_endpoint_duration() {
        let source = source(true);
        let config = proof_config();
        let first = prepare_exposures(&source, [-0.01, 0.03], &config).unwrap();
        assert_eq!(first.len(), 2);
        assert!(first[0].held);
        assert!(!first[1].held);
        assert!((first[0].weight - 0.25).abs() < 1e-14);
        assert!((first[1].weight - 0.75).abs() < 1e-14);
        let initial = source_states(&source, 0.0, &config).unwrap();
        for frame in first[0].nodes[0].samples {
            assert_eq!(frame[0].delta, initial[0].delta);
        }
        let end = prepare_exposures(&source, [0.99, 1.01], &config).unwrap();
        assert_eq!(end.len(), 2);
        assert!(end[1].held);
        assert!((end[1].weight - 0.5).abs() < 1e-14);
        let point = prepare_exposures(&source, [0.4, 0.4], &config).unwrap();
        assert_eq!(point.len(), 1);
        assert_eq!(point[0].weight, 1.0);
    }

    #[test]
    fn temporal_support_bounds_include_source_motion_between_quarter_probes() {
        let source = source(true);
        let root_half = 0.5_f64.sqrt();
        let config = EngravingConfig {
            source_axes: [
                V3::new(root_half, root_half, 0.0),
                V3::new(-root_half, root_half, 0.0),
                V3::new(0.0, 0.0, 1.0),
            ],
            max_temporal_depth: 0,
            ..EngravingConfig::default()
        };
        let exposures = prepare_exposures(&source, [0.07, 0.61], &config).unwrap();
        let node = &exposures[0].nodes[0];
        for index in 0..=2000 {
            let time = 0.07 + 0.54 * f64::from(index) / 2000.0;
            let states = source_states(&source, time, &config).unwrap();
            for (state, bounds) in states.iter().zip(&node.bounds) {
                for (axis, &center) in state.center.iter().enumerate() {
                    assert!(center >= bounds[0][axis]);
                    assert!(center <= bounds[1][axis]);
                }
            }
        }
    }

    #[test]
    fn complete_endpoint_cell_equals_duration_weighted_held_and_live_parts() {
        let source = source(true);
        let config = proof_config();
        let render = render_config();
        let whole =
            render_linear(&source, 0.0, [-0.002, 0.002], &config, &camera(), &render).unwrap();
        let held = render_linear(&source, 0.0, [0.0, 0.0], &config, &camera(), &render).unwrap();
        let live =
            render_linear(&source, 0.001, [0.0, 0.002], &config, &camera(), &render).unwrap();
        for ((actual, a), b) in whole.pixels.iter().zip(&held.pixels).zip(&live.pixels) {
            assert!((*actual - (*a + *b) * 0.5).length() < 1e-12);
        }
        assert_eq!(whole.diagnostics.held_exposure_weight, 0.5);
        assert!(whole.diagnostics.accepted_cells > 0);
        let mut invalid = whole.diagnostics.clone();
        invalid.max_accepted_phase_residual = 2.0 * invalid.phase_tolerance_radians;
        assert!(invalid.validate().is_err());
        let mut invalid = whole.diagnostics;
        invalid.raw_cell_interval[1] = 0.5;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn frozen_source_and_worker_count_preserve_exact_pixels() {
        let source = source(false);
        let config = proof_config();
        let render = render_config();
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let two = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let a = one
            .install(|| render_linear(&source, 0.2, [0.199, 0.201], &config, &camera(), &render))
            .unwrap();
        let b = two
            .install(|| render_linear(&source, 0.2, [0.199, 0.201], &config, &camera(), &render))
            .unwrap();
        assert_eq!(a.pixels, b.pixels);
        let c = render_linear(&source, 0.8, [0.799, 0.801], &config, &camera(), &render).unwrap();
        assert_eq!(a.pixels, c.pixels);
        assert!(a.pixels.iter().any(|p| (*p - render.background).length() > 0.01));
    }

    #[test]
    fn frontal_plane_honors_crop_roll_and_pixel_scale() {
        let render = RenderConfig { width: 80, height: 60, ..RenderConfig::default() };
        let mut camera = Camera {
            position: V3::new(1.0, 2.0, 10.0),
            target: V3::new(1.0, 2.0, 0.0),
            up: V3::new(0.0, 1.0, 0.0),
            orthographic_height: 6.0,
        };
        let plane = Plane::new(&camera, &render).unwrap();
        assert_eq!(plane.world([40.0, 30.0]), [1.0, 2.0]);
        assert_eq!(plane.world([50.0, 20.0]), [2.0, 3.0]);
        camera.up = V3::new(1.0, 0.0, 0.0);
        assert_eq!(Plane::new(&camera, &render).unwrap().world([50.0, 20.0]), [2.0, 1.0]);
        camera.position.x += 0.01;
        assert!(Plane::new(&camera, &render).is_err());
    }

    #[test]
    fn unavailable_refinement_and_invalid_scalar_controls_are_errors() {
        let source = source(true);
        let mut config = proof_config();
        config.phase_tolerance_radians = 1e-12;
        config.max_spatial_depth = 0;
        config.max_temporal_depth = 0;
        assert!(
            render_linear(&source, 0.4, [0.4, 0.4], &config, &camera(), &render_config()).is_err()
        );
        config = proof_config();
        config.source_axes[1] = config.source_axes[0];
        assert!(
            render_linear(&source, 0.4, [0.4, 0.4], &config, &camera(), &render_config()).is_err()
        );
        config = proof_config();
        config.radial_edges.swap(1, 2);
        assert!(
            render_linear(&source, 0.4, [0.4, 0.4], &config, &camera(), &render_config()).is_err()
        );
    }

    #[test]
    fn default_warp_preserves_legacy_rosette_serialization_and_zero_rotation() {
        const LEGACY: &str = r#"{"enabled":true,"semi_axes":[2.12,1.17],"angle_degrees":-24.0,"mouth_angle_degrees":-38.0,"carrier_cycles":64.0,"beat_cycles":2.7,"saddle_cycles":1.8,"tint":{"x":0.72,"y":0.82,"z":0.9},"strength":1.0}"#;
        let shape: EngravingRosetteConfig = serde_json::from_str(LEGACY).unwrap();
        assert_eq!(shape.radial_twist, 0.0);
        assert_eq!(shape.lobe_amplitudes, [LOBE_THREE, LOBE_FIVE]);
        assert_eq!(serde_json::to_string(&shape).unwrap(), LEGACY);
        let rosette = Rosette::new(shape, &EngravingConfig::default());
        for direction in [[0.6, 0.8], [-0.6, -0.8], [-0.0, 1.0]] {
            for radius in [0.0, 0.7, 2.0] {
                assert_eq!(
                    rosette.warped_direction(direction, radius).map(f64::to_bits),
                    direction.map(f64::to_bits)
                );
            }
        }
        let changed =
            EngravingRosetteConfig { radial_twist: 0.8, lobe_amplitudes: [0.10, 0.035], ..shape };
        let value = serde_json::to_value(changed).unwrap();
        assert_eq!(value["radial_twist"], 0.8);
        assert_eq!(value["lobe_amplitudes"], serde_json::json!([0.10, 0.035]));
    }

    #[test]
    fn twisted_lobe_gradients_follow_the_full_chain_rule_without_an_angular_seam() {
        let config = EngravingConfig::default();
        let state = BodyState { center: [0.1, -0.2], delta: 0.8, steer: -0.3 };
        let h = 1e-6;
        for twist in [0.0, 0.8, -0.65, 1.0] {
            let shape = EngravingRosetteConfig {
                radial_twist: twist,
                lobe_amplitudes: [0.10, 0.035],
                ..config.rosettes[1]
            };
            let rosette = Rosette::new(shape, &config);
            for point in [[-0.9, 0.3], [0.5, -1.1], [1.5, 0.6], [-0.5, -0.9]] {
                let value = rosette.field(point, state, &config);
                for axis in 0..2 {
                    let mut left = point;
                    left[axis] -= h;
                    let mut right = point;
                    right[axis] += h;
                    let a = rosette.field(left, state, &config);
                    let b = rosette.field(right, state, &config);
                    assert!(((b.phi - a.phi) / (2.0 * h) - value.phi_gradient[axis]).abs() < 2e-6);
                    assert!(((b.psi - a.psi) / (2.0 * h) - value.psi_gradient[axis]).abs() < 2e-7);
                }
            }
            let straight =
                Rosette::new(EngravingRosetteConfig { angle_degrees: 0.0, ..shape }, &config);
            let a = straight.field([-0.8, -1e-10], BodyState::default(), &config);
            let b = straight.field([-0.8, 1e-10], BodyState::default(), &config);
            assert!((a.phi - b.phi).abs() < 1e-7);
            assert!((a.psi - b.psi).abs() < 1e-7);
            assert!((a.envelope - b.envelope).abs() < 1e-7);
            for (ga, gb) in a.phi_gradient.into_iter().zip(b.phi_gradient) {
                assert!((ga - gb).abs() < 2e-7);
            }
        }
    }

    #[test]
    fn supported_twists_keep_every_radial_contour_ordered_and_reject_excessive_twist() {
        let mut config = proof_config();
        for twist in [0.8, -0.65, 1.0] {
            config.rosettes[0].radial_twist = twist;
            config.rosettes[0].lobe_amplitudes = [0.10, 0.035];
            assert!(validate(0.4, [0.4, 0.4], &config, &render_config()).is_ok());
            let shape = config.rosettes[0];
            let lower = radial_derivative_lower_bound(shape, config.radial_edges[3]);
            assert!(lower > 0.30);
            let maximum = config.radial_edges[3] / radial_factors(shape)[0];
            let rosette = Rosette::new(shape, &config);
            for theta in 0..256 {
                let (s, c) = (TAU * f64::from(theta) / 256.0).sin_cos();
                let mut previous = 0.0;
                for radial in 1..=64 {
                    let r = maximum * f64::from(radial) / 64.0;
                    let field = rosette.field([r * c, r * s], BodyState::default(), &config);
                    let derivative = dot(field.phi_gradient, [c, s]) / shape.carrier_cycles;
                    assert!(derivative >= lower - 1e-12);
                    assert!(field.phi > previous);
                    previous = field.phi;
                }
            }
        }
        for twist in [-1.4, 1.4, f64::NAN] {
            config.rosettes[0].radial_twist = twist;
            assert!(validate(0.4, [0.4, 0.4], &config, &render_config()).is_err());
        }
        config.rosettes[0].radial_twist = 0.0;
        config.rosettes[0].lobe_amplitudes = [0.8, 0.2];
        assert!(validate(0.4, [0.4, 0.4], &config, &render_config()).is_err());
    }

    #[test]
    fn circular_carriers_remain_radial_while_their_mouth_can_curve() {
        let mut config = proof_config();
        config.rosettes[0].radial_twist = 1.0;
        config.rosettes[0].lobe_amplitudes = [0.0; 2];
        let shape = config.rosettes[0];
        let rosette = Rosette::new(shape, &config);
        for point in [[0.5_f64, 0.2], [-0.7, 0.1], [-0.3, -0.8]] {
            let r = point[0].hypot(point[1]);
            let value = rosette.field(point, BodyState::default(), &config);
            assert_eq!(value.phi, shape.carrier_cycles * r);
            for (gradient, coordinate) in value.phi_gradient.into_iter().zip(point) {
                assert!((gradient - shape.carrier_cycles * coordinate / r).abs() < 1e-14);
            }
        }
        for r in [0.4_f64, 0.8] {
            let (s, c) = (-r).sin_cos();
            assert_eq!(rosette.field([r * c, r * s], BodyState::default(), &config).envelope, 0.0);
        }
    }

    #[test]
    fn twisted_mouth_culling_retains_a_visible_shoulder_inside_the_pixel_footprint() {
        let mut config = proof_config();
        config.rosettes[0].radial_twist = 1.0;
        config.rosettes[0].lobe_amplitudes = [0.0; 2];
        let rosette = Rosette::new(config.rosettes[0], &config);
        let radius = 0.7;
        let footprint = 0.1;
        let half_angle = config.mouth_half_angle_degrees.to_radians();
        let theta = half_angle - 0.16 - radius;
        let (s, c) = theta.sin_cos();
        let center = [radius * c, radius * s];
        let corner_step = footprint / 2.0_f64.sqrt();
        let corner = [center[0] + corner_step, center[1] + corner_step];
        // Omitting the twist*radial-excursion term would incorrectly cull.
        assert!(half_angle - 0.16 + (footprint / radius).asin() < half_angle);
        assert!(rosette.field(corner, BodyState::default(), &config).envelope > 0.0);
        let plane = Plane {
            target: [0.0; 2],
            step_x: [footprint * 2.0_f64.sqrt(), 0.0],
            step_y: [0.0, footprint * 2.0_f64.sqrt()],
            size: [100.0; 2],
        };
        let time = TimeNode {
            samples: [[BodyState::default(); 3]; 5],
            bounds: [[[0.0; 2]; 2]; 3],
            children: None,
            depth: 0,
        };
        assert!(!rosette.empty(center, 1.0, plane, &time, 0, &config));
    }
}
