//! Dark asymmetric petals, displaced pearl crescents and persistent fine coronas.
//!
//! Every source state is sampled from the unchanged recording. Light and the
//! common opaque union are integrated together in linear light on the CPU.
mod corona;
mod field;

use super::{Camera, OrbitSeries, RenderConfig, SilkResult, V3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

type Point = [f64; 2];
fn minimum_channel(v: V3) -> f64 {
    v.x.min(v.y).min(v.z)
}
fn maximum_channel(v: V3) -> f64 {
    v.x.max(v.y).max(v.z)
}
fn maximum_absolute_channel(v: V3) -> f64 {
    v.x.abs().max(v.y.abs()).max(v.z.abs())
}

/// One permanently identified dark petal and its displaced luminous contour.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EclipsePetalConfig {
    /// Whether this body's silhouette, crescent and corona participate.
    pub enabled: bool,
    /// Short and long semi-axes, in world units.
    pub semi_axes: Point,
    /// Fixed counterclockwise orientation, in degrees.
    pub angle_degrees: f64,
    /// Permanent lateral bend of the local implicit contour.
    pub shear: f64,
    /// Permanent asymmetric shoulder; bounded away from zero width.
    pub shoulder: f64,
    /// Displacement of the luminous contour behind the opaque petal.
    pub light_offset: Point,
    /// World-space standard deviation across the luminous contour.
    pub light_sigma: f64,
    /// Relative linear radiance of this body's crescent.
    pub light_gain: f64,
    /// Fixed tangential curvature of its fine corona hairs.
    pub corona_bend: f64,
}

impl Default for EclipsePetalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            semi_axes: [0.82, 1.70],
            angle_degrees: -32.0,
            shear: 0.20,
            shoulder: 0.18,
            light_offset: [-0.18, 0.23],
            light_sigma: 0.085,
            light_gain: 1.25,
            corona_bend: 0.18,
        }
    }
}

/// Fixed composition and integration controls for Eclipse Garden.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EclipseConfig {
    /// Fixed orthonormal source axes, archived in the chosen recipe.
    pub source_axes: [V3; 3],
    /// Positive soft scales inside the source mapping.
    pub source_scales: [f64; 3],
    /// Bounded horizontal and vertical source-center motion.
    pub source_motion: Point,
    /// Small long-axis change driven by the third source component.
    pub breathing: f64,
    /// Three original bodies' permanent petal designs.
    pub petals: [EclipsePetalConfig; 3],
    /// World-space width of the symmetric smooth union.
    pub smooth_join: f64,
    /// Half-width of the smooth opaque silhouette transition.
    pub edge_width: f64,
    /// Fixed world-plane direction of the studio illumination.
    pub light_direction: Point,
    /// Dominant warm white in linear RGB.
    pub pearl: V3,
    /// Restrained shell pink in linear RGB.
    pub rose: V3,
    /// Dim copper in the outermost light.
    pub copper: V3,
    /// Nearly black color of fully opaque petals.
    pub dark_color: V3,
    /// Number of persistent short corona hairs per body.
    pub corona_hairs: usize,
    /// Fixed sparse extended hairs per body.
    pub corona_long_hairs: usize,
    /// Number of contour intervals used for fixed arc-length anchors.
    pub anchor_intervals: usize,
    /// Fixed line segments for every quadratic corona hair.
    pub corona_segments: usize,
    /// Short corona lengths in world units.
    pub corona_lengths: Point,
    /// Extended corona lengths in world units.
    pub corona_long_lengths: Point,
    /// Corona Gaussian radii in world units.
    pub corona_radii: Point,
    /// Unoccluded corona luminance relative to the broad crescents.
    pub corona_fraction: f64,
    /// Fixed seed for material-index variation; never changes with frame time.
    pub corona_seed: u64,
    /// Additional Gaussian reconstruction sigma, in output pixels.
    pub minimum_sigma_pixels: f64,
    /// Local linear-RGB quadrature error indicator required for acceptance.
    pub integration_tolerance: f64,
    /// Maximum spatial refinement below each initial AA cell.
    pub max_spatial_depth: usize,
}

impl Default for EclipseConfig {
    fn default() -> Self {
        Self {
            source_axes: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)],
            source_scales: [1.5; 3],
            source_motion: [2.55, 1.35],
            breathing: 0.06,
            petals: [
                EclipsePetalConfig::default(),
                EclipsePetalConfig {
                    semi_axes: [1.14, 1.30],
                    angle_degrees: 52.0,
                    shear: -0.16,
                    shoulder: -0.20,
                    light_offset: [-0.16, 0.20],
                    light_sigma: 0.075,
                    light_gain: 0.90,
                    corona_bend: -0.15,
                    ..EclipsePetalConfig::default()
                },
                EclipsePetalConfig {
                    semi_axes: [0.66, 1.62],
                    angle_degrees: 117.0,
                    shear: 0.24,
                    shoulder: 0.12,
                    light_offset: [-0.14, 0.19],
                    light_sigma: 0.065,
                    light_gain: 0.65,
                    corona_bend: 0.12,
                    ..EclipsePetalConfig::default()
                },
            ],
            smooth_join: 0.025,
            edge_width: 0.003,
            light_direction: [-0.62, 0.78],
            pearl: V3::new(1.0, 0.88, 0.71),
            rose: V3::new(0.76, 0.36, 0.32),
            copper: V3::new(0.55, 0.22, 0.10),
            dark_color: V3::new(0.00010, 0.00007, 0.00012),
            corona_hairs: 1536,
            corona_long_hairs: 36,
            anchor_intervals: 8192,
            corona_segments: 64,
            corona_lengths: [0.07, 0.24],
            corona_long_lengths: [0.38, 0.48],
            corona_radii: [0.00055, 0.0011],
            corona_fraction: 0.12,
            corona_seed: 0xb7f327f9f722,
            minimum_sigma_pixels: 0.25,
            integration_tolerance: 1e-4,
            max_spatial_depth: 6,
        }
    }
}

/// One exposure's numerical and material provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EclipseDiagnostics {
    /// Exact clamped source fraction supplied by the film clock.
    pub source_fraction: f64,
    /// Centers in the original body order.
    pub centers: [Point; 3],
    /// Current long-axis scale for each original body.
    pub axis_factors: [f64; 3],
    /// Persistent curves in this exposure.
    pub curves: usize,
    /// Quadratic-curve chord segments in this exposure.
    pub segments: usize,
    /// Initial spatial cells per final-pixel axis.
    pub spatial_subcells: u32,
    /// Accepted joint mask/light integration cells.
    pub accepted_cells: u64,
    /// Spatial subdivisions required by the joint product.
    pub refinements: u64,
    /// Deepest spatial refinement used.
    pub max_spatial_depth: usize,
    /// Largest accepted coarse/fine quadrature difference in linear RGB.
    pub max_accepted_error_indicator: f64,
    /// Required local quadrature threshold.
    pub integration_tolerance: f64,
    /// Conservative quadratic-chord midpoint deviation in final pixels.
    pub max_chord_error_pixels: f64,
    /// Estimated unoccluded corona luminance, before the common mask.
    pub estimated_corona_luminance: f64,
    /// Minimum returned linear channel value.
    pub minimum_linear_channel: f64,
    /// Maximum returned linear channel value.
    pub maximum_linear_channel: f64,
}

impl EclipseDiagnostics {
    /// Reject missing numerical evidence, non-finite radiance or exhausted quality.
    pub fn validate(&self) -> SilkResult<()> {
        if !(0.0..=1.0).contains(&self.source_fraction)
            || !self.centers.iter().flatten().all(|x| x.is_finite())
            || !self.axis_factors.iter().all(|x| x.is_finite() && *x > 0.0)
            || !(1..=8).contains(&self.spatial_subcells)
            || self.max_spatial_depth > 8
            || ![
                self.max_accepted_error_indicator,
                self.integration_tolerance,
                self.max_chord_error_pixels,
                self.estimated_corona_luminance,
                self.minimum_linear_channel,
                self.maximum_linear_channel,
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
            || self.integration_tolerance <= 0.0
            || self.max_accepted_error_indicator > self.integration_tolerance * (1.0 + 1e-10)
            || self.max_chord_error_pixels > 0.10
            || self.minimum_linear_channel > self.maximum_linear_channel
        {
            return Err("Invalid or unconverged Eclipse diagnostics".into());
        }
        Ok(())
    }
}

/// Linear pixels and deterministic evidence for one unchanged exposure time.
pub struct EclipseFrame {
    /// Linear RGB, before common film exposure, bloom and encoding.
    pub pixels: Vec<V3>,
    /// Source, material and integration evidence.
    pub diagnostics: EclipseDiagnostics,
}

#[derive(Clone, Copy)]
struct Plane {
    target: Point,
    step: f64,
    width: usize,
    height: usize,
}
impl Plane {
    fn new(camera: &Camera, render: &RenderConfig) -> SilkResult<Self> {
        if !camera.position.is_finite()
            || !camera.target.is_finite()
            || !camera.up.is_finite()
            || (camera.position.x - camera.target.x).abs() > 1e-10
            || (camera.position.y - camera.target.y).abs() > 1e-10
            || camera.position.z <= camera.target.z
            || camera.up.x.abs() > 1e-10
            || camera.up.y <= 0.0
            || !camera.orthographic_height.is_finite()
            || camera.orthographic_height <= 0.0
        {
            return Err("Eclipse requires a finite frontal, unrolled camera".into());
        }
        Ok(Self {
            target: [camera.target.x, camera.target.y],
            step: camera.orthographic_height / f64::from(render.height),
            width: render.width as usize,
            height: render.height as usize,
        })
    }
    fn world(self, x: f64, y: f64) -> Point {
        [
            self.target[0] + (x - self.width as f64 * 0.5) * self.step,
            self.target[1] - (y - self.height as f64 * 0.5) * self.step,
        ]
    }
    fn pixel_bounds(self, bounds: [Point; 2]) -> [Point; 2] {
        [
            [
                (bounds[0][0] - self.target[0]) / self.step + self.width as f64 * 0.5,
                (self.target[1] - bounds[1][1]) / self.step + self.height as f64 * 0.5,
            ],
            [
                (bounds[1][0] - self.target[0]) / self.step + self.width as f64 * 0.5,
                (self.target[1] - bounds[0][1]) / self.step + self.height as f64 * 0.5,
            ],
        ]
    }
}

fn validate(config: &EclipseConfig, render: &RenderConfig, time: f64) -> SilkResult<()> {
    if !(0.0..=1.0).contains(&time)
        || render.width == 0
        || render.height == 0
        || render.width > 16384
        || render.height > 16384
        || !(1..=8).contains(&render.aa)
        || !render.background.is_finite()
        || minimum_channel(render.background) < 0.0
        || !config.source_scales.iter().all(|v| v.is_finite() && (1e-6..=1e6).contains(v))
        || !config.source_motion.iter().all(|v| v.is_finite() && (0.0..=100.0).contains(v))
        || !config.breathing.is_finite()
        || !(0.0..=0.20).contains(&config.breathing)
        || !config.smooth_join.is_finite()
        || config.smooth_join <= 0.0
        || !config.edge_width.is_finite()
        || !(1e-6..=1.0).contains(&config.edge_width)
        || !config.light_direction.iter().all(|v| v.is_finite())
        || config.light_direction[0].hypot(config.light_direction[1]) < 1e-12
        || ![config.pearl, config.rose, config.copper, config.dark_color]
            .iter()
            .all(|v| v.is_finite() && minimum_channel(*v) >= 0.0)
        || !(16..=32768).contains(&config.corona_hairs)
        || config.corona_long_hairs > config.corona_hairs
        || !(64..=65536).contains(&config.anchor_intervals)
        || !(8..=1024).contains(&config.corona_segments)
        || ![config.corona_lengths, config.corona_long_lengths, config.corona_radii].iter().all(
            |r| {
                r[0].is_finite()
                    && r[1].is_finite()
                    && r[0] >= 1e-8
                    && r[1] >= r[0]
                    && r[1] <= 100.0
            },
        )
        || !config.corona_fraction.is_finite()
        || config.corona_fraction < 0.0
        || !config.minimum_sigma_pixels.is_finite()
        || config.minimum_sigma_pixels < 0.25
        || !config.integration_tolerance.is_finite()
        || !(1e-8..=0.01).contains(&config.integration_tolerance)
        || !(1..=8).contains(&config.max_spatial_depth)
    {
        return Err("Invalid Eclipse source, material or integration settings".into());
    }
    for (i, axis) in config.source_axes.iter().enumerate() {
        if !axis.is_finite()
            || (axis.length() - 1.0).abs() > 1e-6
            || config.source_axes.iter().take(i).any(|other| axis.dot(*other).abs() > 1e-6)
        {
            return Err("Eclipse source axes must be fixed and orthonormal".into());
        }
    }
    for petal in config.petals {
        if !petal.semi_axes.iter().all(|v| v.is_finite() && (1e-3..=100.0).contains(v))
            || !petal.angle_degrees.is_finite()
            || !petal.shear.is_finite()
            || petal.shear.abs() > 0.30
            || !petal.shoulder.is_finite()
            || petal.shoulder.abs() > 0.22
            || !petal.light_offset.iter().all(|v| v.is_finite())
            || !petal.light_sigma.is_finite()
            || !(1e-6..=10.0).contains(&petal.light_sigma)
            || !petal.light_gain.is_finite()
            || petal.light_gain < 0.0
            || !petal.corona_bend.is_finite()
            || petal.corona_bend.abs() > 2.0
        {
            return Err("Invalid Eclipse petal geometry or light".into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Stats {
    accepted: u64,
    refined: u64,
    depth: usize,
    error: f64,
}
impl Stats {
    fn add(&mut self, other: Self) {
        self.accepted += other.accepted;
        self.refined += other.refined;
        self.depth = self.depth.max(other.depth);
        self.error = self.error.max(other.error);
    }
}

#[derive(Clone, Copy)]
struct QuadratureCell {
    center: Point,
    width: f64,
    center_value: V3,
    depth: usize,
}

fn joint_cell(
    sample: &impl Fn(f64, f64) -> V3,
    cell: QuadratureCell,
    config: &EclipseConfig,
    stats: &mut Stats,
) -> SilkResult<V3> {
    // All nodes are in the centered unit box; the weights integrate its mean.
    const GAUSS_TWO: [(f64, f64); 2] = [(-0.28867513459481287, 0.5), (0.28867513459481287, 0.5)];
    const GAUSS_THREE: [(f64, f64); 3] =
        [(-0.3872983346207417, 5.0 / 18.0), (0.0, 4.0 / 9.0), (0.3872983346207417, 5.0 / 18.0)];
    const SIMPSON: [(f64, f64); 3] = [(-0.5, 1.0 / 6.0), (0.0, 2.0 / 3.0), (0.5, 1.0 / 6.0)];
    let mut constant = true;
    let mut integrate = |nodes: &[(f64, f64)]| {
        let mut result = V3::ZERO;
        for &(dy, wy) in nodes {
            for &(dx, wx) in nodes {
                let value = if dx == 0.0 && dy == 0.0 {
                    cell.center_value
                } else {
                    sample(cell.center[0] + cell.width * dx, cell.center[1] + cell.width * dy)
                };
                constant &= value == cell.center_value;
                result += value * (wx * wy);
            }
        }
        result
    };
    let low = integrate(&GAUSS_TWO);
    let high = integrate(&GAUSS_THREE);
    // Two interior Gauss rules can agree falsely near a masked Gaussian's
    // inflection. A boundary-touching rule supplies independent evidence and
    // catches a thin edge lying beyond both sets of interior nodes.
    let boundary = integrate(&SIMPSON);
    if !low.is_finite() || !high.is_finite() || !boundary.is_finite() {
        return Err("Eclipse joint quadrature produced non-finite linear radiance".into());
    }
    let error = maximum_absolute_channel(high - low).max(maximum_absolute_channel(high - boundary));
    stats.depth = stats.depth.max(cell.depth);
    if error <= config.integration_tolerance {
        stats.accepted += 1;
        stats.error = stats.error.max(error);
        // Preserve exact constant dark/background cells despite non-binary
        // Gauss weights, including when the whole composite is opaque.
        return Ok(if constant { cell.center_value } else { high });
    }
    if cell.depth >= config.max_spatial_depth {
        return Err(format!(
            "Eclipse joint integration exhausted at ({},{}), error {error}",
            cell.center[0], cell.center[1]
        )
        .into());
    }
    stats.refined += 1;
    let mut value = V3::ZERO;
    for [dx, dy] in [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]] {
        let center = [cell.center[0] + cell.width * dx, cell.center[1] + cell.width * dy];
        value += joint_cell(
            sample,
            QuadratureCell {
                center,
                width: cell.width * 0.5,
                center_value: sample(center[0], center[1]),
                depth: cell.depth + 1,
            },
            config,
            stats,
        )? * 0.25;
    }
    Ok(value)
}

struct Integrator<'a> {
    petals: &'a [field::Petal; 3],
    corona: &'a corona::Corona,
    candidates: &'a [usize],
    config: &'a EclipseConfig,
    background: V3,
    plane: Plane,
}
impl Integrator<'_> {
    fn sample(&self, x: f64, y: f64) -> V3 {
        let point = self.plane.world(x, y);
        let transmission = field::transmission(point, self.petals, self.config);
        if transmission == 0.0 {
            return self.config.dark_color;
        }
        let mut light = self.background;
        for petal in self.petals {
            light += field::crescent(point, petal, self.config);
        }
        for &index in self.candidates {
            light += self.corona.segments[index].radiance(point);
        }
        light * transmission + self.config.dark_color * (1.0 - transmission)
    }
    fn cell(
        &self,
        x: f64,
        y: f64,
        width: f64,
        coarse: V3,
        depth: usize,
        stats: &mut Stats,
    ) -> SilkResult<V3> {
        joint_cell(
            &|px, py| self.sample(px, py),
            QuadratureCell { center: [x, y], width, center_value: coarse, depth },
            self.config,
            stats,
        )
    }
}

fn bin_range(bounds: [Point; 2], plane: Plane, tile: usize) -> Option<[usize; 4]> {
    let raw = plane.pixel_bounds(bounds);
    // Boundary Simpson probes touch the neighboring tile at an exact edge.
    let bounds = [raw[0].map(f64::next_down), raw[1].map(f64::next_up)];
    if bounds[1][0] < 0.0
        || bounds[1][1] < 0.0
        || bounds[0][0] > plane.width as f64
        || bounds[0][1] > plane.height as f64
    {
        return None;
    }
    let x0 = bounds[0][0].floor().max(0.0) as usize / tile;
    let y0 = bounds[0][1].floor().max(0.0) as usize / tile;
    let x1 = (bounds[1][0].ceil().max(0.0) as usize).min(plane.width - 1) / tile;
    let y1 = (bounds[1][1].ceil().max(0.0) as usize).min(plane.height - 1) / tile;
    Some([x0, y0, x1, y1])
}

fn covering_ball(plane: Plane, x: usize, y: usize, width: usize, height: usize) -> (Point, f64) {
    let center = plane.world(x as f64 + width as f64 * 0.5, y as f64 + height as f64 * 0.5);
    let mut radius: f64 = 0.0;
    for cy in [y, y + height] {
        for cx in [x, x + width] {
            let corner = plane.world(cx as f64, cy as f64);
            radius = radius.max((corner[0] - center[0]).hypot(corner[1] - center[1]));
        }
    }
    // Include absolute coordinate error as well as the cell's radius. This
    // remains conservative for off-center crops and every boundary probe.
    let frame_extent = plane.step.abs() * (plane.width as f64 + plane.height as f64);
    let padding = 64.0 * f64::EPSILON * (1.0 + center[0].abs() + center[1].abs() + frame_extent);
    (center, (radius + padding).next_up())
}

#[derive(Clone, Copy)]
enum ConstantRegion {
    Dark,
    Background,
}

fn constant_region(
    center: Point,
    radius: f64,
    petals: &[field::Petal; 3],
    has_corona: bool,
    config: &EclipseConfig,
) -> Option<ConstantRegion> {
    if !radius.is_finite() {
        return None;
    }
    match field::constant_transmission(center, radius, petals, config) {
        Some(0.0) => Some(ConstantRegion::Dark),
        Some(1.0)
            if !has_corona
                && petals.iter().all(|petal| field::crescent_is_zero(center, radius, petal)) =>
        {
            Some(ConstantRegion::Background)
        }
        _ => None,
    }
}

fn constant_pixel(color: V3, aa: u32) -> V3 {
    // Keep the original AA addition order: nine additions of color/9 need
    // not have the same last bit as directly assigning color.
    let weight = 1.0 / f64::from(aa * aa);
    let mut value = V3::ZERO;
    for _ in 0..aa {
        for _ in 0..aa {
            value += color * weight;
        }
    }
    value
}

/// Render one exposure with joint opaque-mask, crescent and corona integration.
///
/// Spatial refinement measures the complete composite, so independently averaged
/// light cannot leak through a separately averaged moving opaque edge.
pub fn render_linear(
    source: &OrbitSeries,
    time: f64,
    config: &EclipseConfig,
    camera: &Camera,
    render: &RenderConfig,
) -> SilkResult<EclipseFrame> {
    render_linear_impl::<true>(source, time, config, camera, render)
}

fn render_linear_impl<const CACHE_CONSTANT_REGIONS: bool>(
    source: &OrbitSeries,
    time: f64,
    config: &EclipseConfig,
    camera: &Camera,
    render: &RenderConfig,
) -> SilkResult<EclipseFrame> {
    validate(config, render, time)?;
    let plane = Plane::new(camera, render)?;
    let frame = source.sample(time).ok_or("Eclipse time is outside the recording")?;
    let q: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|a| {
            (frame.bodies[i].position.dot(config.source_axes[a]) / config.source_scales[a]).tanh()
        })
    });
    let centers = std::array::from_fn(|i| {
        [config.source_motion[0] * q[i][0], config.source_motion[1] * q[i][1]]
    });
    let factors = std::array::from_fn(|i| 1.0 + config.breathing * q[i][2]);
    let petals =
        std::array::from_fn(|i| field::Petal::new(config.petals[i], centers[i], factors[i]));
    let corona = corona::prepare(&petals, q.map(|v| v[2]), config, 1.0 / plane.step)?;
    if corona.max_chord_error_pixels > 0.10 {
        return Err("Eclipse corona requires more fixed curve segments".into());
    }
    const TILE: usize = 4;
    let nx = plane.width.div_ceil(TILE);
    let ny = plane.height.div_ceil(TILE);
    let mut bins = vec![Vec::new(); nx * ny];
    let mut active = vec![false; nx * ny];
    for (index, segment) in corona.segments.iter().enumerate() {
        if let Some([x0, y0, x1, y1]) = bin_range(segment.bounds(), plane, TILE) {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    bins[y * nx + x].push(index);
                    active[y * nx + x] = true;
                }
            }
        }
    }
    for (petal, shape) in petals.iter().zip(config.petals) {
        if !shape.enabled {
            continue;
        }
        let padding = 4.0 * shape.light_sigma
            + shape.light_offset[0].hypot(shape.light_offset[1])
            + config.smooth_join * 3.0_f64.ln()
            + config.edge_width
            + plane.step;
        if let Some([x0, y0, x1, y1]) = bin_range(petal.bounds(padding), plane, TILE) {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    active[y * nx + x] = true;
                }
            }
        }
    }
    let dark_pixel = constant_pixel(config.dark_color, render.aa);
    // Reproduce the uncached transparent sample arithmetic, including signed
    // zero behavior. Inactive tiles retain their existing literal background.
    let mut clear_sample = render.background;
    for _ in &petals {
        clear_sample += V3::ZERO;
    }
    clear_sample = clear_sample * 1.0 + config.dark_color * 0.0;
    let background_pixel = constant_pixel(clear_sample, render.aa);
    let tile_results: Vec<SilkResult<(Vec<V3>, Stats)>> = (0..nx * ny)
        .into_par_iter()
        .map(|index| {
            let x0 = (index % nx) * TILE;
            let y0 = (index / nx) * TILE;
            let width = TILE.min(plane.width - x0);
            let height = TILE.min(plane.height - y0);
            let mut pixels = vec![render.background; width * height];
            let mut stats = Stats::default();
            if active[index] {
                if CACHE_CONSTANT_REGIONS {
                    let (center, radius) = covering_ball(plane, x0, y0, width, height);
                    if let Some(region) =
                        constant_region(center, radius, &petals, !bins[index].is_empty(), config)
                    {
                        pixels.fill(match region {
                            ConstantRegion::Dark => dark_pixel,
                            ConstantRegion::Background => background_pixel,
                        });
                        return Ok((pixels, stats));
                    }
                }
                let integrator = Integrator {
                    petals: &petals,
                    corona: &corona,
                    candidates: &bins[index],
                    config,
                    background: render.background,
                    plane,
                };
                let count = render.aa;
                let weight = 1.0 / f64::from(count * count);
                for y in 0..height {
                    for x in 0..width {
                        if CACHE_CONSTANT_REGIONS {
                            let (center, radius) = covering_ball(plane, x0 + x, y0 + y, 1, 1);
                            if let Some(region) = constant_region(
                                center,
                                radius,
                                &petals,
                                !bins[index].is_empty(),
                                config,
                            ) {
                                pixels[y * width + x] = match region {
                                    ConstantRegion::Dark => dark_pixel,
                                    ConstantRegion::Background => background_pixel,
                                };
                                continue;
                            }
                        }
                        let mut value = V3::ZERO;
                        for sy in 0..count {
                            for sx in 0..count {
                                let px = (x0 + x) as f64 + (f64::from(sx) + 0.5) / f64::from(count);
                                let py = (y0 + y) as f64 + (f64::from(sy) + 0.5) / f64::from(count);
                                let coarse = integrator.sample(px, py);
                                value += integrator.cell(
                                    px,
                                    py,
                                    1.0 / f64::from(count),
                                    coarse,
                                    0,
                                    &mut stats,
                                )? * weight;
                            }
                        }
                        if !value.is_finite() || minimum_channel(value) < 0.0 {
                            return Err("Eclipse radiance exceeded finite nonnegative RGB".into());
                        }
                        pixels[y * width + x] = value;
                    }
                }
            }
            Ok((pixels, stats))
        })
        .collect();
    let mut pixels = vec![render.background; plane.width * plane.height];
    let mut stats = Stats::default();
    let mut low = f64::INFINITY;
    let mut high: f64 = 0.0;
    for (index, result) in tile_results.into_iter().enumerate() {
        let (tile, s) = result?;
        stats.add(s);
        let x0 = (index % nx) * TILE;
        let y0 = (index / nx) * TILE;
        let width = TILE.min(plane.width - x0);
        for (row, values) in tile.chunks(width).enumerate() {
            for &v in values {
                low = low.min(minimum_channel(v));
                high = high.max(maximum_channel(v));
            }
            pixels[(y0 + row) * plane.width + x0..(y0 + row) * plane.width + x0 + width]
                .copy_from_slice(values);
        }
    }
    let diagnostics = EclipseDiagnostics {
        source_fraction: time,
        centers,
        axis_factors: factors,
        curves: corona.curve_count,
        segments: corona.segments.len(),
        spatial_subcells: render.aa,
        accepted_cells: stats.accepted,
        refinements: stats.refined,
        max_spatial_depth: stats.depth,
        max_accepted_error_indicator: stats.error,
        integration_tolerance: config.integration_tolerance,
        max_chord_error_pixels: corona.max_chord_error_pixels,
        estimated_corona_luminance: corona.estimated_luminance,
        minimum_linear_channel: low,
        maximum_linear_channel: high,
    };
    diagnostics.validate()?;
    Ok(EclipseFrame { pixels, diagnostics })
}

/// Conservative full-recording motion and crop evidence for a fixed film recipe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EclipseMotionAudit {
    /// Number of encoded source frames considered.
    pub frames: usize,
    /// Proposed fixed midpoint exposures per film frame.
    pub temporal_samples: usize,
    /// Open fraction of each frame interval.
    pub shutter_fraction: f64,
    /// Largest continuous point-motion bound per encoded frame, in pixels.
    pub maximum_pixels_per_frame: f64,
    /// Resulting point-motion bound between exposure centers, in pixels.
    pub maximum_pixels_per_exposure_step: f64,
    /// Minimum fixed count needed for at most 0.15 pixels per exposure step.
    pub required_temporal_samples: usize,
    /// Source-frame interval producing the largest bound.
    pub worst_frame_interval: usize,
    /// Original body index producing the largest bound.
    pub worst_body: usize,
    /// Conservative whole-film artwork bounds in world XY.
    pub world_bounds: [Point; 2],
    /// Conservative minimum margin to the camera crop, in final pixels.
    pub minimum_crop_margin_pixels: f64,
}

fn sech_squared_upper(value: f64) -> f64 {
    let exponential = (-2.0 * value.abs()).exp();
    // The exponential form keeps a positive derivative after tanh rounds to1.
    // next_up also bounds a derivative below the least positive subnormal.
    (4.0 * exponential / (1.0 + exponential).powi(2) * (1.0 + 64.0 * f64::EPSILON)).next_up()
}

fn projected_interval(low: V3, high: V3, axis: V3) -> Point {
    let mut result = [0.0; 2];
    for a in 0..3 {
        let x = low.axis(a) * axis.axis(a);
        let y = high.axis(a) * axis.axis(a);
        result[0] += x.min(y);
        result[1] += x.max(y);
    }
    let pad = 64.0 * f64::EPSILON * (1.0 + result[0].abs().max(result[1].abs()));
    [(result[0] - pad).next_down(), (result[1] + pad).next_up()]
}

/// Bound every petal and quadratic-corona point across the continuous recording.
///
/// Hermite velocity hulls bound source motion between samples. Long-axis
/// breathing changes a unit contour normal at no more than half its relative
/// axis rate; this also bounds the tangentially bent corona tip. Held shutter
/// endpoints cannot increase these bounds.
pub fn audit_motion(
    source: &OrbitSeries,
    config: &EclipseConfig,
    camera: &Camera,
    render: &RenderConfig,
    frames: usize,
    shutter_fraction: f64,
    temporal_samples: usize,
) -> SilkResult<EclipseMotionAudit> {
    validate(config, render, 0.0)?;
    let plane = Plane::new(camera, render)?;
    if frames < 2
        || temporal_samples == 0
        || !shutter_fraction.is_finite()
        || !(0.0..=1.0).contains(&shutter_fraction)
    {
        return Err("Invalid Eclipse film clock for motion audit".into());
    }
    let mut maximum: f64 = 0.0;
    let mut worst = 0;
    let mut body_index = 0;
    let mut bounds = [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]];
    for frame in 0..frames - 1 {
        let start = frame as f64 / (frames - 1) as f64;
        let end = (frame + 1) as f64 / (frames - 1) as f64;
        for body in 0..3 {
            let shape = config.petals[body];
            if !shape.enabled {
                continue;
            }
            let (low, high) = source
                .position_bounds(body, start, end)
                .ok_or("Cannot bound Eclipse source positions")?;
            let (vlo, vhi) = source
                .velocity_bounds(body, start, end)
                .ok_or("Cannot bound Eclipse source velocities")?;
            let mut q = [[0.0; 2]; 3];
            let mut rate = [0.0; 3];
            for a in 0..3 {
                let p = projected_interval(low, high, config.source_axes[a])
                    .map(|v| v / config.source_scales[a]);
                q[a] = p.map(f64::tanh);
                let velocity = projected_interval(vlo, vhi, config.source_axes[a]);
                let closest = if p[0] > 0.0 {
                    p[0]
                } else if p[1] < 0.0 {
                    p[1]
                } else {
                    0.0
                };
                let derivative = sech_squared_upper(closest);
                rate[a] = (velocity[0].abs().max(velocity[1].abs()) / config.source_scales[a]
                    * derivative)
                    * (1.0 + 128.0 * f64::EPSILON);
                if !rate[a].is_finite() {
                    return Err("Eclipse source-rate bound exceeded finite arithmetic".into());
                }
            }
            let center_rate =
                (config.source_motion[0] * rate[0]).hypot(config.source_motion[1] * rate[1]);
            let normal_rate = config.breathing * rate[2] / (2.0 * (1.0 - config.breathing));
            let length = config.corona_lengths[1].max(config.corona_long_lengths[1]);
            let bend = shape.corona_bend.abs() + 0.08;
            let contour_rate = shape.semi_axes[1] * config.breathing * rate[2];
            let corona_rate = length * ((1.0 + bend) * normal_rate + 0.08 * rate[2]);
            let emitter_rate = 4.0 * shape.light_sigma * normal_rate;
            let pixels = (center_rate + contour_rate + corona_rate.max(emitter_rate))
                / ((frames - 1) as f64 * plane.step);
            if !pixels.is_finite() {
                return Err("Eclipse pixel-motion bound exceeded finite arithmetic".into());
            }
            if pixels > maximum {
                maximum = pixels;
                worst = frame;
                body_index = body;
            }
            for factor in q[2].map(|z| 1.0 + config.breathing * z) {
                let petal = field::Petal::new(shape, [0.0; 2], factor);
                let expansion = (length * (1.0 + bend * bend).sqrt()
                    + 6.0
                        * (config.corona_radii[1].powi(2)
                            + (config.minimum_sigma_pixels * plane.step).powi(2))
                        .sqrt())
                .max(4.0 * shape.light_sigma)
                .max(config.smooth_join * 3.0_f64.ln() + config.edge_width);
                let local = petal.bounds(expansion);
                for a in 0..2 {
                    bounds[0][a] = bounds[0][a].min(
                        local[0][a]
                            + config.source_motion[a] * q[a][0]
                            + shape.light_offset[a].min(0.0),
                    );
                    bounds[1][a] = bounds[1][a].max(
                        local[1][a]
                            + config.source_motion[a] * q[a][1]
                            + shape.light_offset[a].max(0.0),
                    );
                }
            }
        }
    }
    if !config.petals.iter().any(|p| p.enabled) {
        bounds = [plane.target; 2];
    }
    if !bounds.iter().flatten().all(|v| v.is_finite()) {
        return Err("Eclipse artwork bounds exceeded finite arithmetic".into());
    }
    let pixel_bounds = plane.pixel_bounds(bounds);
    let margin = pixel_bounds[0][0]
        .min(pixel_bounds[0][1])
        .min(plane.width as f64 - pixel_bounds[1][0])
        .min(plane.height as f64 - pixel_bounds[1][1]);
    Ok(EclipseMotionAudit {
        frames,
        temporal_samples,
        shutter_fraction,
        maximum_pixels_per_frame: maximum,
        maximum_pixels_per_exposure_step: maximum * shutter_fraction / temporal_samples as f64,
        required_temporal_samples: (maximum * shutter_fraction / 0.15).ceil().max(1.0) as usize,
        worst_frame_interval: worst,
        worst_body: body_index,
        world_bounds: bounds,
        minimum_crop_margin_pixels: margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;
    fn source(moving: bool) -> OrbitSeries {
        let samples = (0..129)
            .map(|i| {
                std::array::from_fn(|body| {
                    if !moving {
                        return V3::ZERO;
                    }
                    let angle = std::f64::consts::TAU * (f64::from(i) / 128.0 + body as f64 / 3.0);
                    V3::new(angle.cos(), angle.sin(), 0.25 * (1.3 * angle).sin())
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "eclipse-test".into(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({"synthetic":true}),
        })
        .unwrap()
    }
    fn config() -> EclipseConfig {
        EclipseConfig {
            corona_hairs: 32,
            corona_long_hairs: 3,
            anchor_intervals: 128,
            corona_segments: 16,
            ..EclipseConfig::default()
        }
    }
    fn camera() -> Camera {
        Camera {
            position: V3::new(0.0, 0.0, 12.0),
            target: V3::ZERO,
            orthographic_height: 7.4,
            ..Camera::default()
        }
    }
    fn render() -> RenderConfig {
        RenderConfig {
            width: 16,
            height: 12,
            aa: 1,
            background: V3::new(0.0005, 0.00035, 0.00065),
            ..RenderConfig::default()
        }
    }
    #[test]
    fn saturated_source_rates_and_boundary_touching_bins_remain_conservative() {
        assert_eq!(20.0_f64.tanh(), 1.0);
        assert!(sech_squared_upper(20.0) >= 4.0 * (-40.0_f64).exp());
        assert!(sech_squared_upper(800.0) > 0.0);
        let plane = Plane { target: [0.0; 2], step: 1.0, width: 32, height: 32 };
        // World x=0 is exactly the boundary between pixels15 and16.
        let bins = bin_range([[0.0, -0.25], [0.25, 0.25]], plane, 4).unwrap();
        assert_eq!([bins[0], bins[2]], [3, 4]);
    }

    #[test]
    fn covering_balls_include_boundary_probes_with_large_absolute_camera_offsets() {
        for target in [[0.37, -0.81], [1e12, -2e12]] {
            let plane = Plane { target, step: 7.4 / 2160.0, width: 3840, height: 2160 };
            for (x, y, width, height) in [(1371, 653, 4, 4), (3839, 2159, 1, 1), (0, 0, 3, 2)] {
                let (center, radius) = covering_ball(plane, x, y, width, height);
                assert!(radius.is_finite());
                for sy in 0..=16 {
                    for sx in 0..=16 {
                        let point = plane.world(
                            x as f64 + width as f64 * f64::from(sx) / 16.0,
                            y as f64 + height as f64 * f64::from(sy) / 16.0,
                        );
                        assert!((point[0] - center[0]).hypot(point[1] - center[1]) <= radius);
                    }
                }
            }
        }
    }

    #[test]
    fn constant_regions_ignore_buried_corona_but_require_empty_clear_bins() {
        let c = config();
        let petals = c.petals.map(|shape| field::Petal::new(shape, [0.0; 2], 1.0));
        assert!(matches!(
            constant_region([0.0; 2], 0.01, &petals, true, &c),
            Some(ConstantRegion::Dark)
        ));
        assert!(matches!(
            constant_region([10.0, 10.0], 0.01, &petals, false, &c),
            Some(ConstantRegion::Background)
        ));
        assert!(constant_region([10.0, 10.0], 0.01, &petals, true, &c).is_none());
    }

    #[test]
    fn constant_region_cache_preserves_visible_pixel_bits_for_shifted_cameras() {
        let source = source(true);
        let mut c = config();
        // Resolve the thumbnail's silhouette without changing the strict local
        // error tolerance or the production recipe's edge/refinement settings.
        c.edge_width = 0.03;
        c.max_spatial_depth = 8;
        let mut r = RenderConfig { width: 32, height: 24, aa: 3, ..render() };
        let workers = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        for (target, time) in [([0.0, 0.0], 0.17), ([-0.43, 0.29], 0.63)] {
            let camera = Camera {
                position: V3::new(target[0], target[1], 12.0),
                target: V3::new(target[0], target[1], 0.0),
                ..camera()
            };
            let uncached = render_linear_impl::<false>(&source, time, &c, &camera, &r).unwrap();
            let cached = workers
                .install(|| render_linear_impl::<true>(&source, time, &c, &camera, &r))
                .unwrap();
            assert!(uncached.pixels.iter().any(|value| maximum_channel(*value) > 0.005));
            assert!(cached.diagnostics.accepted_cells < uncached.diagnostics.accepted_cells);
            assert_eq!(cached.diagnostics.refinements, uncached.diagnostics.refinements);
            for (index, (&a, &b)) in cached.pixels.iter().zip(&uncached.pixels).enumerate() {
                for axis in 0..3 {
                    assert_eq!(
                        a.axis(axis).to_bits(),
                        b.axis(axis).to_bits(),
                        "pixel {index}, channel {axis}"
                    );
                }
            }
            assert_eq!(
                cached.diagnostics.minimum_linear_channel.to_bits(),
                uncached.diagnostics.minimum_linear_channel.to_bits()
            );
            assert_eq!(
                cached.diagnostics.maximum_linear_channel.to_bits(),
                uncached.diagnostics.maximum_linear_channel.to_bits()
            );
            // Also exercise signed-zero arithmetic and untouched inactive tiles.
            r.background.x = -0.0;
            c.dark_color.z = -0.0;
        }
    }

    #[test]
    fn stationary_source_has_no_material_animation_or_worker_dependence() {
        let source = source(false);
        let mut c = config();
        // This thumbnail checks scheduling and material identity. Give its
        // silhouette a meaningful pixel width; full-detail edge accuracy has
        // its own native-scale joint-product regression below.
        c.edge_width = 0.03;
        c.max_spatial_depth = 8;
        let camera = camera();
        let render = render();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| render_linear(&source, 0.0, &c, &camera, &render).unwrap());
        let three = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap()
            .install(|| render_linear(&source, 0.8, &c, &camera, &render).unwrap());
        assert_eq!(one.pixels, three.pixels);
        assert!(one.pixels.iter().any(|v| maximum_channel(*v) > 0.005));
        assert_eq!(one.diagnostics.curves, 96);
        one.diagnostics.validate().unwrap();
        three.diagnostics.validate().unwrap();
    }
    #[test]
    fn disabled_petals_return_the_exact_background_and_invalid_inputs_fail() {
        let source = source(false);
        let mut c = config();
        for shape in &mut c.petals {
            shape.enabled = false;
        }
        let camera = camera();
        let render = render();
        let image = render_linear(&source, 0.4, &c, &camera, &render).unwrap();
        assert!(image.pixels.iter().all(|v| *v == render.background));
        assert_eq!(image.diagnostics.curves, 0);
        c.minimum_sigma_pixels = 0.1;
        assert!(render_linear(&source, 0.4, &c, &camera, &render).is_err());
        c.minimum_sigma_pixels = 0.25;
        let mut tilted = camera;
        tilted.position.x = 1.0;
        assert!(render_linear(&source, 0.4, &c, &tilted, &render).is_err());
    }
    #[test]
    fn joint_occlusion_quadrature_matches_dense_product_integration() {
        let mut c = config();
        c.petals[0] = EclipsePetalConfig {
            semi_axes: [1.0, 1.0],
            angle_degrees: 0.0,
            shear: 0.0,
            shoulder: 0.0,
            light_offset: [-0.01, 0.03],
            light_sigma: 0.015,
            ..EclipsePetalConfig::default()
        };
        c.petals[1].enabled = false;
        c.petals[2].enabled = false;
        c.light_direction = [0.0, 1.0];
        let petals = std::array::from_fn(|i| field::Petal::new(c.petals[i], [0.0; 2], 1.0));
        let plane = Plane { target: [0.0, 1.0], step: 0.008, width: 1, height: 1 };
        let corona = corona::prepare(&petals, [0.0; 3], &c, 1.0 / plane.step).unwrap();
        let candidates: Vec<_> = (0..corona.segments.len()).collect();
        let integrator = Integrator {
            petals: &petals,
            corona: &corona,
            candidates: &candidates,
            config: &c,
            background: render().background,
            plane,
        };
        for offset in [-0.37, 0.0, 0.29] {
            let x = 0.5;
            let y = 0.5 + offset;
            let mut stats = Stats::default();
            let actual =
                integrator.cell(x, y, 1.0, integrator.sample(x, y), 0, &mut stats).unwrap();
            let mut reference = V3::ZERO;
            let count = 128;
            for sy in 0..count {
                for sx in 0..count {
                    reference += integrator.sample(
                        x + (f64::from(sx) + 0.5) / f64::from(count) - 0.5,
                        y + (f64::from(sy) + 0.5) / f64::from(count) - 0.5,
                    );
                }
            }
            reference /= f64::from(count * count);
            assert!(
                maximum_absolute_channel(actual - reference) < c.integration_tolerance,
                "{actual:?} {reference:?}"
            );
        }
    }

    fn integrate_fixture(sample: &impl Fn(f64, f64) -> V3, config: &EclipseConfig) -> (V3, Stats) {
        let mut stats = Stats::default();
        let value = joint_cell(
            sample,
            QuadratureCell {
                center: [0.0; 2],
                width: 1.0,
                center_value: sample(0.0, 0.0),
                depth: 0,
            },
            config,
            &mut stats,
        )
        .unwrap();
        (value, stats)
    }

    fn smooth_fixture(value: f64) -> f64 {
        let t = value.clamp(0.0, 1.0);
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn dense_line_fixture(sample: &impl Fn(f64, f64) -> V3) -> V3 {
        const COUNT: u32 = 65_536;
        let mut result = V3::ZERO;
        for i in 0..COUNT {
            result += sample((f64::from(i) + 0.5) / f64::from(COUNT) - 0.5, 0.0);
        }
        result / f64::from(COUNT)
    }

    #[test]
    fn independent_quadrature_catches_the_midpoint_gaussian_cancellation() {
        let sigma = 0.75_f64;
        let peak = sigma * sigma / 0.25 * (0.25_f64.powi(2) / (2.0 * sigma * sigma)).exp().acosh();
        let sample = |x: f64, _y: f64| {
            let value = (-0.5 * ((x - peak) / sigma).powi(2)).exp();
            V3::new(value, value * 0.7, value * 0.3)
        };
        let midpoint = sample(0.0, 0.0);
        let old_fine = (sample(-0.25, 0.0) + sample(0.25, 0.0)) * 0.5;
        assert!(maximum_absolute_channel(midpoint - old_fine) < 1e-14);
        let reference = dense_line_fixture(&sample);
        assert!(maximum_absolute_channel(old_fine - reference) > 1e-3);
        let (actual, stats) = integrate_fixture(&sample, &config());
        assert!(stats.refined > 0);
        assert!(maximum_absolute_channel(actual - reference) < 1e-6);
    }

    #[test]
    fn boundary_estimate_catches_native_gauss_pair_cancellation() {
        // Native AA3: .25-pixel minimum Gaussian sigma and .003-world-unit
        // silhouette half-width at 2160 pixels over 7.4 world units.
        let half_width = 0.003 / (7.4 / 2160.0) * 3.0;
        let peak = -0.5147290205167565;
        let sample = |x: f64, _y: f64| {
            let light = 3.0 * (-0.5 * ((x - peak) / 0.75).powi(2)).exp();
            let transmission = smooth_fixture((x + 2.0 * half_width) / (2.0 * half_width));
            V3::new(light * transmission, light * transmission * 0.8, light * transmission * 0.4)
        };
        let g2 = (sample(-1.0 / 12.0_f64.sqrt(), 0.0) + sample(1.0 / 12.0_f64.sqrt(), 0.0)) * 0.5;
        let g3 = (sample(-0.3872983346207417, 0.0) + sample(0.3872983346207417, 0.0))
            * (5.0 / 18.0)
            + sample(0.0, 0.0) * (4.0 / 9.0);
        let reference = dense_line_fixture(&sample);
        assert!(maximum_absolute_channel(g2 - g3) < 1e-12);
        assert!(maximum_absolute_channel(g3 - reference) > 1e-4);
        let (actual, stats) = integrate_fixture(&sample, &config());
        assert!(stats.refined > 0);
        assert!(maximum_absolute_channel(actual - reference) < 1e-6);
    }

    #[test]
    fn boundary_nodes_catch_thin_edges_and_exhaustion_remains_an_error() {
        let sample = |x: f64, _y: f64| {
            let value = 3.0 * smooth_fixture((x - 0.445) / 0.01);
            V3::new(value, value, value)
        };
        assert_eq!(sample(0.3872983346207417, 0.0), V3::ZERO);
        let mut config = config();
        config.max_spatial_depth = 0;
        let mut stats = Stats::default();
        assert!(
            joint_cell(
                &sample,
                QuadratureCell { center: [0.0; 2], width: 1.0, center_value: V3::ZERO, depth: 0 },
                &config,
                &mut stats
            )
            .is_err()
        );
        config.max_spatial_depth = 10;
        let (actual, stats) = integrate_fixture(&sample, &config);
        assert!(stats.refined > 0);
        assert!(maximum_absolute_channel(actual - dense_line_fixture(&sample)) < 1e-6);
        // Production validation remains capped at eight. The extra fixture
        // depth only establishes convergence for this intentionally tiny edge.
    }
    #[test]
    fn continuous_motion_audit_covers_breathing_and_curved_corona_tips() {
        let source = source(true);
        let c = config();
        let camera = camera();
        let r = render();
        let audit = audit_motion(&source, &c, &camera, &r, 33, 0.5, 128).unwrap();
        let point = |time: f64, body: usize, theta: f64| {
            let frame = source.sample(time).unwrap();
            let p = frame.bodies[body].position;
            let q = std::array::from_fn::<_, 3, _>(|a| {
                (p.dot(c.source_axes[a]) / c.source_scales[a]).tanh()
            });
            let petal = field::Petal::new(
                c.petals[body],
                [c.source_motion[0] * q[0], c.source_motion[1] * q[1]],
                1.0 + c.breathing * q[2],
            );
            let anchor = petal.contour(theta);
            let length = c.corona_long_lengths[1];
            let bend = c.petals[body].corona_bend + 0.08 * q[2];
            std::array::from_fn::<_, 2, _>(|a| {
                anchor.position[a] + length * (anchor.normal[a] + bend * anchor.tangent[a])
            })
        };
        for step in 0..1024 {
            let a = f64::from(step) / 1024.0;
            let b = f64::from(step + 1) / 1024.0;
            for body in 0..3 {
                for theta in [0.0, 0.71, 2.4, 4.2] {
                    let x = point(a, body, theta);
                    let y = point(b, body, theta);
                    let pixels = (y[0] - x[0]).hypot(y[1] - x[1])
                        / (camera.orthographic_height / f64::from(r.height));
                    assert!(
                        pixels / (b - a) / 32.0 <= audit.maximum_pixels_per_frame * (1.0 + 1e-10)
                    );
                }
            }
        }
        let held = audit_motion(&self::source(false), &c, &camera, &r, 33, 0.5, 128).unwrap();
        assert!(held.maximum_pixels_per_frame < 1e-10);
    }
}
