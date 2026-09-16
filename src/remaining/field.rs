//! Fixed three-dimensional stock and cumulative, irreversible excavation.
//!
//! The unchanged source is sampled at fixed canonical times. Compact ellipsoid
//! kernels follow its transported tangent, normal and binormal. We linearly
//! interpolate these nonnegative kernel fields in source time and integrate
//! that interpolation exactly, including partial intervals. Every canonical
//! contribution therefore has a nonnegative, nondecreasing weight. No film fps,
//! previous frame, averaging by elapsed time, stochastic sampling or healing
//! enters the field. Source time is the fixed recorded interval [0, 1]. A return
//! during additional source time adds dose; compressing two identical passes
//! into the same duration preserves their occupation integral.
//!
//! The stock is either the rounded union of a declared coarse sampling of actual
//! source triangles, or an explicitly configured control ellipsoid. It is frozen
//! using the complete recording before excavation begins. The optional reveal
//! plane or rounded aperture is an art-direction cut, not an inferred physical
//! encounter. Both only remove from the fixed stock and remain frozen in time.
//! Optional rim rounding applies a monotone smooth maximum at the blank/dose
//! junction. It removes additional material there; it never heals or edits a
//! generated mesh. Its width is in implicit-field units, not a physical bevel radius.
//!
//! The output is an implicit sign field: negative means material remains. It is
//! NOT a signed distance field and must not supply sphere-tracing step lengths.
//! Brush AABBs are indexed into fixed spatial bricks in canonical order; each
//! parallel z slab owns its output and sums contributions in that same order.

mod geometry;

use super::Grid;
use crate::atelier::{BodySample, OrbitSeries, SilkResult, V3};
use geometry::{Bounds, Triangle, TriangleEnvelope};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const BRICK_WIDTH: usize = 8;

/// Initial stock construction; both choices stay fixed during excavation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StockMode {
    /// Rounded union of sampled instantaneous three-body triangles.
    #[default]
    TriangleEnvelope,
    /// Explicit axis-aligned control ellipsoid, independent of the moving tools.
    Ellipsoid,
}

/// Declared planar reveal retaining the half-space `normal.dot(x) <= offset`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RevealPlane {
    /// Outward unit normal of the retained half-space.
    pub normal: V3,
    /// Signed distance from the origin along the unit normal.
    pub offset: f64,
}

impl Default for RevealPlane {
    fn default() -> Self {
        Self { normal: V3::new(0.15, -0.2, 1.0).normalized(), offset: 0.15 }
    }
}

/// An explicitly authored ellipsoidal opening removed before the moving cutters.
/// This creates a rounded viewing aperture; it is not attributed to the source.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealAperture {
    /// Fixed world-space center of the volume to remove.
    pub center: V3,
    /// Positive world-axis semi-axes of the volume to remove.
    pub axes: [f64; 3],
}

/// Material envelope, broad source-carried tools and fixed dose calibration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldConfig {
    /// Tool semi-axes along transported tangent, normal and binormal, in world units.
    pub cutter_axes: [f64; 3],
    /// Common nonnegative multiplier of all three bodies' integrated kernels.
    pub dose_strength: f64,
    /// Positive cumulative dose at which material is removed.
    pub dose_threshold: f64,
    /// Number of fixed source-time intervals; there are this many plus one kernels per body.
    pub time_samples: usize,
    /// Source-derived envelope or explicit control ellipsoid.
    pub stock_mode: StockMode,
    /// Number of instantaneous source triangles, including both recording endpoints.
    pub stock_samples: usize,
    /// Radius rounding each coarse source triangle into a three-dimensional solid.
    pub stock_radius: f64,
    /// Fixed center of the optional ellipsoid control.
    pub ellipsoid_center: V3,
    /// Fixed world-axis semi-axes of the optional ellipsoid control.
    pub ellipsoid_axes: [f64; 3],
    /// Optional fixed reveal cut, explicitly independent of orbital causation.
    pub reveal: Option<RevealPlane>,
    /// Optional rounded viewing aperture, fixed before any source-dose excavation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aperture: Option<RevealAperture>,
    /// Width of a smooth maximum at the blank/dose intersection; zero is exact max.
    /// This only adds removal. Units are scalar-field units, not exact bevel distance.
    #[serde(skip_serializing_if = "zero_rounding")]
    pub rim_rounding: f64,
    /// Physical outer margin in addition to two complete protective grid cells.
    pub grid_margin: f64,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            cutter_axes: [0.7, 0.45, 0.22],
            dose_strength: 1.0,
            dose_threshold: 0.012,
            time_samples: 1024,
            stock_mode: StockMode::TriangleEnvelope,
            stock_samples: 96,
            stock_radius: 0.45,
            ellipsoid_center: V3::ZERO,
            ellipsoid_axes: [2.4, 1.8, 1.45],
            reveal: Some(RevealPlane::default()),
            aperture: None,
            rim_rounding: 0.0,
            grid_margin: 0.1,
        }
    }
}

impl FieldConfig {
    /// Validate bounded allocation controls and finite positive geometric scales.
    pub fn validate(&self) -> SilkResult<()> {
        if !self.cutter_axes.iter().all(|axis| axis.is_finite() && (0.01..=8.0).contains(axis))
            || !self.dose_strength.is_finite()
            || !(0.0..=100.0).contains(&self.dose_strength)
            || !self.dose_threshold.is_finite()
            || !(1e-9..=300.0).contains(&self.dose_threshold)
            || !(4..=16_384).contains(&self.time_samples)
            || !(2..=4096).contains(&self.stock_samples)
            || !self.stock_radius.is_finite()
            || !(0.01..=8.0).contains(&self.stock_radius)
            || !self.ellipsoid_center.is_finite()
            || !self
                .ellipsoid_axes
                .iter()
                .all(|axis| axis.is_finite() && (0.01..=16.0).contains(axis))
            || !self.grid_margin.is_finite()
            || !(0.0..=8.0).contains(&self.grid_margin)
            || !self.rim_rounding.is_finite()
            || !(0.0..=0.5).contains(&self.rim_rounding)
        {
            return Err("invalid Remaining Form field configuration".into());
        }
        if let Some(plane) = self.reveal
            && (!plane.normal.is_finite()
                || !plane.offset.is_finite()
                || (plane.normal.length_squared() - 1.0).abs() > 1e-8)
        {
            return Err("reveal plane requires a finite offset and a unit normal".into());
        }
        if let Some(aperture) = self.aperture
            && (!aperture.center.is_finite()
                || !aperture
                    .axes
                    .iter()
                    .all(|axis| axis.is_finite() && (0.01..=16.0).contains(axis)))
        {
            return Err(
                "rounded reveal aperture requires a finite center and positive finite axes".into(),
            );
        }
        Ok(())
    }
}

/// Grid and dose measurements; node volumes are approximate, not mesh volumes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FieldDiagnostics {
    /// Requested fraction of the original recording, without wrapping or prehistory.
    pub source_fraction: f64,
    /// Final dimensions in x/y/z order, including exterior protective cells.
    pub grid_dimensions: [usize; 3],
    /// Common world-space distance between adjacent grid nodes.
    pub grid_spacing: f64,
    /// Largest dose sampled in the fixed stock's immediate neighborhood.
    pub max_dose: f64,
    /// Fixed threshold used for this field.
    pub dose_threshold: f64,
    /// Nodes inside the original stock after its fixed plane and aperture cuts.
    pub stock_nodes: usize,
    /// Nodes still containing material at this source time.
    pub remaining_nodes: usize,
    /// Previously stocked nodes removed by the cumulative source dose.
    pub removed_nodes: usize,
    /// Stock node count multiplied by cubed spacing, for coarse calibration only.
    pub approximate_stock_volume: f64,
    /// Remaining node count multiplied by cubed spacing, for coarse calibration only.
    pub approximate_remaining_volume: f64,
    /// Largest source displacement between adjacent canonical samples, over all bodies.
    pub max_source_step: f64,
    /// Smallest cutter semi-axis divided by grid spacing; monitor spatial resolution.
    pub minimum_cutter_radius_cells: f64,
    /// Canonical body centers outside the chosen coarse stock, before the reveal.
    pub centers_outside_stock: usize,
    /// Smallest implicit value on the entire exterior grid boundary; must be positive.
    pub minimum_boundary_value: f64,
    /// Smallest implicit value in the volume; negative means some sampled material remains.
    pub minimum_value: f64,
    /// Largest implicit value in the volume.
    pub maximum_value: f64,
}

impl FieldDiagnostics {
    /// Validate finite numerical evidence and consistent material counts.
    pub fn validate(&self) -> SilkResult<()> {
        let scalars = [
            self.grid_spacing,
            self.max_dose,
            self.dose_threshold,
            self.approximate_stock_volume,
            self.approximate_remaining_volume,
            self.max_source_step,
            self.minimum_cutter_radius_cells,
            self.minimum_boundary_value,
        ];
        let count = self.grid_dimensions.iter().try_fold(1_usize, |value, &n| value.checked_mul(n));
        if !(0.0..=1.0).contains(&self.source_fraction)
            || !scalars.iter().all(|value| value.is_finite() && *value >= 0.0)
            || self.grid_spacing <= 0.0
            || self.dose_threshold <= 0.0
            || self.minimum_boundary_value <= 0.0
            || !self.minimum_value.is_finite()
            || !self.maximum_value.is_finite()
            || self.minimum_value > self.maximum_value
            || self.grid_dimensions.iter().any(|n| *n < 5 || *n > 512)
            || count.is_none_or(|n| self.stock_nodes > n)
            || self.remaining_nodes > self.stock_nodes
            || self.removed_nodes != self.stock_nodes - self.remaining_nodes
        {
            return Err("invalid Remaining Form field diagnostics".into());
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Brush {
    center: V3,
    scaled_axes: [V3; 3],
    bounds: Bounds,
    time_index: usize,
}

impl Brush {
    fn new(body: BodySample, axes: [f64; 3], time_index: usize) -> SilkResult<Self> {
        let basis = [body.tangent, body.normal, body.binormal];
        if !body.position.is_finite()
            || !basis
                .iter()
                .all(|axis| axis.is_finite() && (axis.length_squared() - 1.0).abs() < 1e-8)
            || basis[0].dot(basis[1]).abs() > 1e-8
            || basis[0].dot(basis[2]).abs() > 1e-8
            || basis[1].dot(basis[2]).abs() > 1e-8
        {
            return Err("source supplied an invalid three-dimensional cutter frame".into());
        }
        let extent: [f64; 3] = std::array::from_fn(|coordinate| {
            (0..3)
                .map(|axis| (basis[axis].axis(coordinate) * axes[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        });
        let extent = V3::new(extent[0], extent[1], extent[2]);
        Ok(Self {
            center: body.position,
            scaled_axes: std::array::from_fn(|axis| basis[axis] / axes[axis]),
            bounds: Bounds { min: body.position - extent, max: body.position + extent }
                .expand(1e-10),
            time_index,
        })
    }

    fn kernel(&self, point: V3) -> f64 {
        if !self.bounds.contains(point) {
            return 0.0;
        }
        let delta = point - self.center;
        let q_squared: f64 = self.scaled_axes.iter().map(|axis| axis.dot(delta).powi(2)).sum();
        if q_squared >= 1.0 {
            return 0.0;
        }
        let q = q_squared.sqrt();
        let edge = 1.0 - q;
        let square = edge * edge;
        square * square * (1.0 + 4.0 * q)
    }
}

#[derive(Debug)]
enum Stock {
    Envelope(TriangleEnvelope),
    Ellipsoid { center: V3, axes: [f64; 3] },
}

impl Stock {
    fn bounds(&self) -> Bounds {
        match self {
            Self::Envelope(envelope) => envelope.bounds(),
            Self::Ellipsoid { center, axes } => {
                let extent = V3::new(axes[0], axes[1], axes[2]);
                Bounds { min: *center - extent, max: *center + extent }
            }
        }
    }

    fn value(&self, point: V3) -> f64 {
        match self {
            Self::Envelope(envelope) => envelope.value(point),
            Self::Ellipsoid { center, axes } => ellipsoid_value(point, *center, *axes),
        }
    }
}

/// Immutable source geometry and spatial tools reusable for arbitrary time queries.
#[derive(Debug)]
pub struct PreparedField {
    config: FieldConfig,
    brushes: Vec<Brush>,
    stock: Stock,
    max_source_step: f64,
    centers_outside_stock: usize,
}

/// Outward normals of the continuous implicit field, independent of mesh facets.
/// Geometry and sampled grid values are unchanged by this optional shading pass.
#[derive(Clone, Debug)]
pub struct SurfaceNormals {
    /// Raw normalized vectors pointing toward increasing implicit-field value.
    pub normals: Vec<V3>,
    /// Smallest nonzero pre-normalization gradient length among the supplied points.
    pub minimum_gradient_length: f64,
    /// Largest pre-normalization gradient length among the supplied points.
    pub maximum_gradient_length: f64,
    /// Central-difference step for triangle-envelope stock; zero for an analytic ellipsoid.
    pub stock_difference_step: f64,
    /// Points requiring a declared full-field finite difference at an undefined junction.
    pub finite_difference_fallbacks: usize,
}

impl PreparedField {
    /// Prepare fixed source samples and the frozen stock, without changing the source.
    pub fn new(source: &OrbitSeries, config: &FieldConfig) -> SilkResult<Self> {
        config.validate()?;
        let mut brushes = Vec::with_capacity(3 * (config.time_samples + 1));
        let mut previous: Option<[V3; 3]> = None;
        let mut max_source_step = 0.0_f64;
        for index in 0..=config.time_samples {
            let time = index as f64 / config.time_samples as f64;
            let frame =
                source.sample(time).ok_or("source cannot supply canonical carving sample")?;
            let centers = frame.bodies.map(|body| body.position);
            if let Some(last) = previous {
                for (a, b) in centers.iter().zip(last) {
                    max_source_step = max_source_step.max((*a - b).length());
                }
            }
            previous = Some(centers);
            for body in frame.bodies {
                brushes.push(Brush::new(body, config.cutter_axes, index)?);
            }
        }
        let stock = match config.stock_mode {
            StockMode::TriangleEnvelope => {
                let mut triangles = Vec::with_capacity(config.stock_samples);
                for index in 0..config.stock_samples {
                    let time = index as f64 / (config.stock_samples - 1) as f64;
                    let frame = source.sample(time).ok_or("source cannot supply stock triangle")?;
                    triangles.push(Triangle::new(frame.bodies.map(|body| body.position)));
                }
                Stock::Envelope(TriangleEnvelope::new(triangles, config.stock_radius))
            }
            StockMode::Ellipsoid => {
                Stock::Ellipsoid { center: config.ellipsoid_center, axes: config.ellipsoid_axes }
            }
        };
        let centers_outside_stock =
            brushes.iter().filter(|brush| stock.value(brush.center) > 0.0).count();
        Ok(Self { config: config.clone(), brushes, stock, max_source_step, centers_outside_stock })
    }

    /// Sample the cumulative sign field with isotropic spacing and x-fastest storage.
    /// `resolution` is the maximum node count on any axis, including guard cells.
    pub fn grid(&self, time: f64, resolution: usize) -> SilkResult<(Grid, FieldDiagnostics)> {
        if !(0.0..=1.0).contains(&time) || !(8..=512).contains(&resolution) {
            return Err("field time must be in [0,1] and maximum grid dimension in [8,512]".into());
        }
        let weights = temporal_weights(self.config.time_samples, time);
        let stock_bounds = self.stock.bounds();
        let span = stock_bounds.max - stock_bounds.min;
        let inner_span =
            [span.x, span.y, span.z].map(|value| value + 2.0 * self.config.grid_margin);
        let spacing = inner_span.iter().copied().fold(0.0, f64::max) / (resolution - 5) as f64;
        let dims =
            inner_span.map(|value| ((value / spacing).ceil() as usize).min(resolution - 5) + 5);
        let margin = self.config.grid_margin + 2.0 * spacing;
        let origin = stock_bounds.min - V3::new(margin, margin, margin);
        let count = dims
            .iter()
            .try_fold(1_usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or("field grid dimensions overflow")?;
        let bricks = BrushBricks::new(&self.brushes, &weights, origin, spacing, dims);
        let mut values = vec![0.0; count];
        let minimum_axis = self.config.cutter_axes.iter().copied().fold(f64::INFINITY, f64::min);
        let slabs: Vec<SlabStats> = values
            .par_chunks_mut(dims[0] * dims[1])
            .enumerate()
            .map(|(z, slab)| {
                let mut stats = SlabStats::default();
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let point = origin + V3::new(x as f64, y as f64, z as f64) * spacing;
                        let stock_value = self.stock.value(point);
                        let mut blank_value = if let Some(plane) = self.config.reveal {
                            stock_value.max(plane.normal.dot(point) - plane.offset)
                        } else {
                            stock_value
                        };
                        if let Some(aperture) = self.config.aperture {
                            blank_value = blank_value.max(-ellipsoid_value(
                                point,
                                aperture.center,
                                aperture.axes,
                            ));
                        }
                        // Both stock representations are 1-Lipschitz. Beyond
                        // two cells every corner of an incident cell is outside,
                        // so skipping its dose cannot change an extracted surface.
                        let dose = if stock_value > 2.0 * spacing
                            || time == 0.0
                            || self.config.dose_strength == 0.0
                        {
                            0.0
                        } else {
                            self.dose(point, &weights, bricks.candidates(x, y, z))
                        };
                        let carving = (dose / self.config.dose_threshold - 1.0) * minimum_axis;
                        let value = smooth_max(blank_value, carving, self.config.rim_rounding);
                        slab[y * dims[0] + x] = value;
                        stats.max_dose = stats.max_dose.max(dose);
                        stats.minimum = stats.minimum.min(value);
                        stats.maximum = stats.maximum.max(value);
                        stats.stock += usize::from(blank_value < 0.0);
                        stats.remaining += usize::from(value < 0.0);
                        if x == 0
                            || y == 0
                            || z == 0
                            || x + 1 == dims[0]
                            || y + 1 == dims[1]
                            || z + 1 == dims[2]
                        {
                            stats.boundary_minimum = stats.boundary_minimum.min(value);
                        }
                    }
                }
                stats
            })
            .collect();
        let mut total = SlabStats::default();
        for slab in slabs {
            total.max_dose = total.max_dose.max(slab.max_dose);
            total.minimum = total.minimum.min(slab.minimum);
            total.maximum = total.maximum.max(slab.maximum);
            total.boundary_minimum = total.boundary_minimum.min(slab.boundary_minimum);
            total.stock += slab.stock;
            total.remaining += slab.remaining;
        }
        let diagnostics = FieldDiagnostics {
            source_fraction: time,
            grid_dimensions: dims,
            grid_spacing: spacing,
            max_dose: total.max_dose,
            dose_threshold: self.config.dose_threshold,
            stock_nodes: total.stock,
            remaining_nodes: total.remaining,
            removed_nodes: total.stock - total.remaining,
            approximate_stock_volume: total.stock as f64 * spacing.powi(3),
            approximate_remaining_volume: total.remaining as f64 * spacing.powi(3),
            max_source_step: self.max_source_step,
            minimum_cutter_radius_cells: minimum_axis / spacing,
            centers_outside_stock: self.centers_outside_stock,
            minimum_boundary_value: total.boundary_minimum,
            minimum_value: total.minimum,
            maximum_value: total.maximum,
        };
        diagnostics.validate()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("excavation produced a non-finite scalar field".into());
        }
        Ok((Grid { dims, origin, spacing, values }, diagnostics))
    }

    /// Evaluate continuous field-gradient normals without modifying geometry or grid values.
    ///
    /// The grid supplies only its dimensions, origin and spacing; its values may
    /// be omitted when shading a previously archived mesh. Kernel, ellipsoid,
    /// plane, aperture and smooth-maximum derivatives are analytic. Triangle-
    /// envelope stock uses a central difference of its accelerated scalar query
    /// at `0.02 * spacing`. Exact hard-maximum ties or zero analytic gradients
    /// use the same-step full-field central difference and are counted explicitly.
    /// A still-zero or nonfinite result is an error; normals are never flipped
    /// to agree with a triangle or replaced by an arbitrary direction.
    pub fn surface_normals(
        &self,
        points: &[V3],
        time: f64,
        grid: &Grid,
    ) -> SilkResult<SurfaceNormals> {
        if points.is_empty()
            || !(0.0..=1.0).contains(&time)
            || !grid.origin.is_finite()
            || !grid.spacing.is_finite()
            || grid.spacing <= 0.0
            || grid.dims.iter().any(|dimension| !(2..=512).contains(dimension))
        {
            return Err("surface normals require finite grid geometry, valid source time and nonempty points".into());
        }
        let step = 0.02 * grid.spacing;
        let extent = V3::new(
            (grid.dims[0] - 1) as f64,
            (grid.dims[1] - 1) as f64,
            (grid.dims[2] - 1) as f64,
        ) * grid.spacing;
        let maximum = grid.origin + extent;
        if !step.is_finite() || step <= 0.0 || !maximum.is_finite() {
            return Err("surface-normal grid exceeds finite coordinate range".into());
        }
        let weights = temporal_weights(self.config.time_samples, time);
        let bricks =
            BrushBricks::new(&self.brushes, &weights, grid.origin, grid.spacing, grid.dims);
        // Collect first, then select any error in input order. A worker race
        // must not change either returned normals or the reported failing point.
        let samples: Vec<SilkResult<(V3, f64, bool)>> = points.par_iter().enumerate()
            .map(|(index, &point)| {
                if !point.is_finite() || (0..3).any(|axis| {
                    point.axis(axis) - step < grid.origin.axis(axis)
                        || point.axis(axis) + step > maximum.axis(axis)
                }) {
                    return Err(format!("surface normal point {index} lacks a finite grid guard margin").into());
                }
                let jet = self.field_jet(point, &weights, &bricks, grid, step);
                let analytic = jet.gradient.filter(|gradient| gradient.is_finite()
                    && vector_length(*gradient) > 0.0);
                let used_fallback = analytic.is_none();
                let gradient = analytic.unwrap_or_else(|| {
                    central_difference(point, step, |sample| {
                        self.field_jet(sample, &weights, &bricks, grid, step).value
                    })
                });
                let length = vector_length(gradient);
                if !gradient.is_finite() || !length.is_finite() || length == 0.0 {
                    return Err(format!("surface normal point {index} has an undefined or zero field gradient after the declared finite-difference fallback").into());
                }
                Ok((gradient / length, length, used_fallback))
            }).collect();
        let samples = samples.into_iter().collect::<SilkResult<Vec<_>>>()?;
        let mut result = SurfaceNormals {
            normals: Vec::with_capacity(points.len()),
            minimum_gradient_length: f64::INFINITY,
            maximum_gradient_length: 0.0,
            stock_difference_step: if matches!(self.stock, Stock::Envelope(_)) {
                step
            } else {
                0.0
            },
            finite_difference_fallbacks: 0,
        };
        for (normal, length, used_fallback) in samples {
            result.normals.push(normal);
            result.minimum_gradient_length = result.minimum_gradient_length.min(length);
            result.maximum_gradient_length = result.maximum_gradient_length.max(length);
            result.finite_difference_fallbacks += usize::from(used_fallback);
        }
        Ok(result)
    }

    fn field_jet(
        &self,
        point: V3,
        weights: &[f64],
        bricks: &BrushBricks,
        grid: &Grid,
        step: f64,
    ) -> FieldJet {
        let mut blank = match &self.stock {
            Stock::Ellipsoid { center, axes } => ellipsoid_jet(point, *center, *axes),
            Stock::Envelope(envelope) => FieldJet {
                value: envelope.value(point),
                gradient: Some(central_difference(point, step, |sample| envelope.value(sample))),
            },
        };
        if let Some(plane) = self.config.reveal {
            blank = blank.hard_max(FieldJet {
                value: plane.normal.dot(point) - plane.offset,
                gradient: Some(plane.normal),
            });
        }
        if let Some(aperture) = self.config.aperture {
            let aperture = ellipsoid_jet(point, aperture.center, aperture.axes);
            blank = blank.hard_max(FieldJet {
                value: -aperture.value,
                gradient: aperture.gradient.map(|gradient| -gradient),
            });
        }
        let node: [usize; 3] = std::array::from_fn(|axis| {
            (((point.axis(axis) - grid.origin.axis(axis)) / grid.spacing).floor() as usize)
                .min(grid.dims[axis] - 1)
        });
        let mut dose = 0.0;
        let mut gradient = V3::ZERO;
        if self.config.dose_strength != 0.0 {
            for &index in bricks.candidates(node[0], node[1], node[2]) {
                let brush = &self.brushes[index as usize];
                let weight = weights[brush.time_index];
                let sample = brush.kernel_jet(point);
                dose += weight * sample.value;
                // Compact kernels are differentiable including their centers.
                gradient += sample.gradient.expect("kernel gradient is defined") * weight;
            }
        }
        let minimum_axis = self.config.cutter_axes.iter().copied().fold(f64::INFINITY, f64::min);
        let scale = self.config.dose_strength * minimum_axis / self.config.dose_threshold;
        let carving = FieldJet {
            value: (dose * self.config.dose_strength / self.config.dose_threshold - 1.0)
                * minimum_axis,
            gradient: Some(gradient * scale),
        };
        blank.smooth_max(carving, self.config.rim_rounding)
    }

    fn dose(&self, point: V3, weights: &[f64], candidates: &[u32]) -> f64 {
        let mut dose = 0.0;
        for &index in candidates {
            let brush = &self.brushes[index as usize];
            dose += weights[brush.time_index] * brush.kernel(point);
        }
        dose * self.config.dose_strength
    }
}

#[derive(Clone, Copy, Debug)]
struct FieldJet {
    value: f64,
    gradient: Option<V3>,
}

impl FieldJet {
    fn hard_max(self, other: Self) -> Self {
        if self.value > other.value {
            self
        } else if other.value > self.value {
            other
        } else {
            // A differing gradient at an exact tie has no unique derivative.
            // The caller records a finite-difference fallback for that point.
            Self {
                value: self.value.max(other.value),
                gradient: if self.gradient == other.gradient { self.gradient } else { None },
            }
        }
    }

    fn smooth_max(self, other: Self, width: f64) -> Self {
        if width == 0.0 || (self.value - other.value).abs() >= width {
            return self.hard_max(other);
        }
        let self_weight = (0.5 + (self.value - other.value) / (2.0 * width)).clamp(0.0, 1.0);
        let gradient = match (self.gradient, other.gradient) {
            (Some(a), Some(b)) => Some(a * self_weight + b * (1.0 - self_weight)),
            _ => None,
        };
        Self { value: smooth_max(self.value, other.value, width), gradient }
    }
}

impl Brush {
    fn kernel_jet(&self, point: V3) -> FieldJet {
        if !self.bounds.contains(point) {
            return FieldJet { value: 0.0, gradient: Some(V3::ZERO) };
        }
        let delta = point - self.center;
        let components = self.scaled_axes.map(|axis| axis.dot(delta));
        let q_squared: f64 = components.iter().map(|component| component.powi(2)).sum();
        if q_squared >= 1.0 {
            return FieldJet { value: 0.0, gradient: Some(V3::ZERO) };
        }
        let q = q_squared.sqrt();
        let edge = 1.0 - q;
        let square = edge * edge;
        let value = square * square * (1.0 + 4.0 * q);
        // dK/dq=-20q(1-q)^3 cancels the 1/q from differentiating
        // ellipsoidal radius. This remains regular at the exact tool center.
        let radial = self.scaled_axes[0] * components[0]
            + self.scaled_axes[1] * components[1]
            + self.scaled_axes[2] * components[2];
        FieldJet { value, gradient: Some(radial * (-20.0 * square * edge)) }
    }
}

fn ellipsoid_jet(point: V3, center: V3, axes: [f64; 3]) -> FieldJet {
    let delta = point - center;
    let q = (0..3).map(|axis| (delta.axis(axis) / axes[axis]).powi(2)).sum::<f64>().sqrt();
    let scale = axes.iter().copied().fold(f64::INFINITY, f64::min);
    let gradient = if q > 0.0 {
        Some(
            V3::new(
                delta.x / (axes[0] * axes[0]),
                delta.y / (axes[1] * axes[1]),
                delta.z / (axes[2] * axes[2]),
            ) * (scale / q),
        )
    } else {
        None
    };
    FieldJet { value: (q - 1.0) * scale, gradient }
}

fn central_difference(point: V3, step: f64, value: impl Fn(V3) -> f64) -> V3 {
    let axes = [V3::new(step, 0.0, 0.0), V3::new(0.0, step, 0.0), V3::new(0.0, 0.0, step)];
    let components = axes.map(|axis| (value(point + axis) - value(point - axis)) / (2.0 * step));
    V3::new(components[0], components[1], components[2])
}

fn vector_length(vector: V3) -> f64 {
    vector.x.hypot(vector.y).hypot(vector.z)
}

/// Signed, scaled radial ellipsoid value; correct inside/outside, not exact distance.
fn ellipsoid_value(point: V3, center: V3, axes: [f64; 3]) -> f64 {
    let delta = point - center;
    let q = (0..3).map(|axis| (delta.axis(axis) / axes[axis]).powi(2)).sum::<f64>().sqrt();
    (q - 1.0) * axes.iter().copied().fold(f64::INFINITY, f64::min)
}

fn zero_rounding(value: &f64) -> bool {
    *value == 0.0
}

/// Monotone in both inputs, at least their maximum, and equal to it outside the band.
fn smooth_max(a: f64, b: f64, width: f64) -> f64 {
    let maximum = a.max(b);
    let difference = (a - b).abs();
    if width == 0.0 || difference >= width {
        return maximum;
    }
    let blend = width - difference;
    maximum + blend * blend / (4.0 * width)
}

#[derive(Debug)]
struct BrushBricks {
    dims: [usize; 3],
    entries: Vec<Vec<u32>>,
}

impl BrushBricks {
    fn new(brushes: &[Brush], weights: &[f64], origin: V3, spacing: f64, grid: [usize; 3]) -> Self {
        let dims = grid.map(|value| value.div_ceil(BRICK_WIDTH));
        let mut entries = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        for (index, brush) in brushes.iter().enumerate() {
            if weights[brush.time_index] == 0.0 {
                continue;
            }
            let lower: [isize; 3] = std::array::from_fn(|axis| {
                ((brush.bounds.min.axis(axis) - origin.axis(axis)) / spacing).floor() as isize
            });
            let upper: [isize; 3] = std::array::from_fn(|axis| {
                ((brush.bounds.max.axis(axis) - origin.axis(axis)) / spacing).ceil() as isize
            });
            if (0..3).any(|axis| upper[axis] < 0 || lower[axis] >= grid[axis] as isize) {
                continue;
            }
            let low: [usize; 3] =
                std::array::from_fn(|axis| lower[axis].max(0) as usize / BRICK_WIDTH);
            let high: [usize; 3] = std::array::from_fn(|axis| {
                (upper[axis].min(grid[axis] as isize - 1) as usize) / BRICK_WIDTH
            });
            for z in low[2]..=high[2] {
                for y in low[1]..=high[1] {
                    for x in low[0]..=high[0] {
                        entries[(z * dims[1] + y) * dims[0] + x].push(index as u32);
                    }
                }
            }
        }
        Self { dims, entries }
    }

    fn candidates(&self, x: usize, y: usize, z: usize) -> &[u32] {
        &self.entries
            [((z / BRICK_WIDTH) * self.dims[1] + y / BRICK_WIDTH) * self.dims[0] + x / BRICK_WIDTH]
    }
}

#[derive(Debug)]
struct SlabStats {
    max_dose: f64,
    minimum: f64,
    maximum: f64,
    boundary_minimum: f64,
    stock: usize,
    remaining: usize,
}

impl Default for SlabStats {
    fn default() -> Self {
        Self {
            max_dose: 0.0,
            minimum: f64::INFINITY,
            maximum: f64::NEG_INFINITY,
            boundary_minimum: f64::INFINITY,
            stock: 0,
            remaining: 0,
        }
    }
}

fn temporal_weights(intervals: usize, time: f64) -> Vec<f64> {
    let h = 1.0 / intervals as f64;
    let coordinate = time * intervals as f64;
    let complete = (coordinate.floor() as usize).min(intervals);
    let mut result = vec![0.0; intervals + 1];
    if complete > 0 {
        result[0] = 0.5 * h;
        result[1..complete].fill(h);
        result[complete] = 0.5 * h;
    }
    if complete < intervals {
        let partial = (coordinate - complete as f64).clamp(0.0, 1.0);
        // These forms are monotone even under ordinary rounded arithmetic:
        // each subtraction subtracts a monotonically decreasing nonnegative term.
        result[complete] += 0.5 * h * (1.0 - (1.0 - partial).powi(2));
        result[complete + 1] += 0.5 * h * partial.powi(2);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;
    use std::f64::consts::TAU;

    fn source(moving: bool) -> OrbitSeries {
        let samples = (0..=256_u32)
            .map(|index| {
                let phase = TAU * f64::from(index) / 128.0;
                let step = if moving {
                    V3::new(0.42 * phase.cos(), 0.35 * phase.sin(), 0.3 * (2.0 * phase).sin())
                } else {
                    V3::ZERO
                };
                [
                    V3::new(-1.0, -0.3, -0.2) + step,
                    V3::new(1.0, -0.3, 0.2) + step,
                    V3::new(0.0, 0.9, 0.0) + step,
                ]
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "0xab51".into(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::Value::Null,
        })
        .unwrap()
    }

    fn config() -> FieldConfig {
        FieldConfig { time_samples: 32, stock_samples: 12, reveal: None, ..FieldConfig::default() }
    }

    fn dose_at(field: &PreparedField, point: V3, time: f64) -> f64 {
        let all: Vec<u32> = (0..field.brushes.len()).map(|index| index as u32).collect();
        field.dose(point, &temporal_weights(field.config.time_samples, time), &all)
    }

    #[test]
    fn temporal_integral_is_normalized_continuous_and_monotone_at_knots() {
        let count = 37;
        let mut times = vec![0.0, 1.0];
        for index in 1..count {
            let knot = index as f64 / count as f64;
            times.extend([knot.next_down(), knot, knot.next_up()]);
        }
        times.sort_by(f64::total_cmp);
        let mut previous = vec![0.0; count + 1];
        for time in times {
            let weights = temporal_weights(count, time);
            assert!((weights.iter().sum::<f64>() - time).abs() < 1e-14);
            for (a, b) in previous.iter().zip(&weights) {
                assert!(b >= a);
            }
            previous = weights;
        }
    }

    #[test]
    fn cumulative_excavation_never_restores_material() {
        let field = PreparedField::new(&source(true), &config()).unwrap();
        let (a, da) = field.grid(0.13, 33).unwrap();
        let (b, db) = field.grid(0.61, 33).unwrap();
        let (c, dc) = field.grid(1.0, 33).unwrap();
        assert_eq!(a.dims, b.dims);
        assert_eq!(a.origin, c.origin);
        for ((a, b), c) in a.values.iter().zip(&b.values).zip(&c.values) {
            assert!(b >= a && c >= b);
        }
        assert!(
            da.remaining_nodes >= db.remaining_nodes && db.remaining_nodes >= dc.remaining_nodes
        );
        assert!(dc.removed_nodes > da.removed_nodes);
    }

    #[test]
    fn repeated_visits_add_exposure_and_stationary_dwell_is_exact() {
        let source = source(false);
        let field = PreparedField::new(&source, &config()).unwrap();
        let point = source.sample(0.0).unwrap().bodies[0].position;
        let first = dose_at(&field, point, 0.25);
        let repeated = dose_at(&field, point, 0.5);
        assert!((first - 0.25).abs() < 1e-14);
        assert!((repeated - 2.0 * first).abs() < 1e-14);
        let finer =
            PreparedField::new(&source, &FieldConfig { time_samples: 127, ..config() }).unwrap();
        assert!((dose_at(&finer, point, 0.37) - 0.37).abs() < 1e-14);
    }

    #[test]
    fn returning_along_the_same_three_dimensional_loop_doubles_local_dose() {
        let source = source(true);
        let cfg = FieldConfig { cutter_axes: [0.8; 3], time_samples: 64, ..config() };
        let field = PreparedField::new(&source, &cfg).unwrap();
        let point = source.sample(0.217).unwrap().bodies[0].position;
        let first_pass = dose_at(&field, point, 0.5);
        let two_passes = dose_at(&field, point, 1.0);
        assert!(first_pass > 0.0);
        assert!((two_passes - 2.0 * first_pass).abs() < 1e-13);
    }

    #[test]
    fn temporal_refinement_converges_for_a_moving_three_dimensional_tool() {
        let source = source(true);
        let mut cfg = config();
        cfg.cutter_axes = [0.8; 3];
        let point = source.sample(0.217).unwrap().bodies[0].position + V3::new(0.12, 0.08, 0.11);
        let evaluate = |count| {
            let field =
                PreparedField::new(&source, &FieldConfig { time_samples: count, ..cfg.clone() })
                    .unwrap();
            dose_at(&field, point, 0.713)
        };
        let reference = evaluate(1024);
        let coarse = (evaluate(16) - reference).abs();
        let fine = (evaluate(128) - reference).abs();
        assert!(fine < coarse * 0.15, "coarse={coarse:e}, fine={fine:e}");
    }

    #[test]
    fn grid_is_finite_guarded_and_identical_across_time_order_and_workers() {
        let source = source(true);
        let field = PreparedField::new(&source, &config()).unwrap();
        let render = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| field.grid(0.43, 25).unwrap())
        };
        let (a, da) = render(1);
        field.grid(0.81, 25).unwrap();
        let (b, db) = render(3);
        assert_eq!(a.values, b.values);
        assert_eq!(serde_json::to_vec(&da).unwrap(), serde_json::to_vec(&db).unwrap());
        assert!(da.minimum_boundary_value > 0.0);
        assert!(a.values.iter().all(|value| value.is_finite()));
        assert!(field.grid(-0.01, 25).is_err());
    }

    #[test]
    fn brush_keeps_actual_depth_and_full_transported_axes() {
        let body = BodySample {
            position: V3::new(0.2, -0.1, 1.7),
            tangent: V3::new(0.0, 0.0, 1.0),
            normal: V3::new(1.0, 0.0, 0.0),
            binormal: V3::new(0.0, 1.0, 0.0),
            speed: 0.0,
            curvature: 0.0,
            proximity: 0.0,
            arc_length: 0.0,
        };
        let brush = Brush::new(body, [0.7, 0.45, 0.22], 0).unwrap();
        assert_eq!(brush.kernel(body.position), 1.0);
        assert_eq!(brush.kernel(V3::new(0.2, -0.1, 0.0)), 0.0);
        assert!(brush.kernel(body.position + V3::new(0.0, 0.0, 0.35)) > 0.0);
        assert_eq!(brush.kernel(body.position + V3::new(0.0, 0.35, 0.0)), 0.0);
    }

    #[test]
    fn spatial_bricks_match_ordered_brute_force_dose() {
        let field = PreparedField::new(&source(true), &config()).unwrap();
        let origin = V3::new(-3.0, -3.0, -3.0);
        let spacing = 0.2;
        let dims = [31; 3];
        let weights = temporal_weights(field.config.time_samples, 0.617);
        let bricks = BrushBricks::new(&field.brushes, &weights, origin, spacing, dims);
        let all: Vec<_> = (0..field.brushes.len()).map(|index| index as u32).collect();
        for z in (0..31).step_by(3) {
            for y in (0..31).step_by(3) {
                for x in (0..31).step_by(3) {
                    let point = origin + V3::new(x as f64, y as f64, z as f64) * spacing;
                    assert_eq!(
                        field.dose(point, &weights, bricks.candidates(x, y, z)),
                        field.dose(point, &weights, &all)
                    );
                }
            }
        }
    }

    #[test]
    fn declared_plane_only_removes_material_from_the_fixed_control_stock() {
        let source = source(false);
        let cfg = FieldConfig { stock_mode: StockMode::Ellipsoid, dose_strength: 0.0, ..config() };
        let whole = PreparedField::new(&source, &cfg).unwrap().grid(1.0, 25).unwrap().0;
        let cut = PreparedField::new(
            &source,
            &FieldConfig {
                reveal: Some(RevealPlane { normal: V3::new(0.0, 0.0, 1.0), offset: 0.15 }),
                ..cfg
            },
        )
        .unwrap()
        .grid(1.0, 25)
        .unwrap()
        .0;
        assert_eq!(whole.dims, cut.dims);
        assert!(whole.values.iter().zip(cut.values).all(|(before, after)| after >= *before));
    }

    #[test]
    fn rounded_aperture_is_fixed_subtraction_with_positive_exterior_guards() {
        let source = source(false);
        let cfg = FieldConfig { stock_mode: StockMode::Ellipsoid, ..config() };
        let no_aperture = PreparedField::new(&source, &cfg).unwrap().grid(0.0, 33).unwrap().0;
        let aperture = RevealAperture { center: V3::new(0.0, 0.0, 1.1), axes: [1.1, 0.9, 0.8] };
        let prepared =
            PreparedField::new(&source, &FieldConfig { aperture: Some(aperture), ..cfg }).unwrap();
        let (initial, diagnostics) = prepared.grid(0.0, 33).unwrap();
        let (later, _) = prepared.grid(0.7, 33).unwrap();
        assert_eq!(initial.dims, no_aperture.dims);
        assert_eq!(initial.origin, no_aperture.origin);
        assert_eq!(initial.spacing, no_aperture.spacing);
        assert!(diagnostics.minimum_boundary_value > 0.0);
        assert!(
            initial.values.iter().zip(&no_aperture.values).all(|(after, before)| after >= before)
        );
        assert!(initial.values.iter().zip(&later.values).all(|(early, late)| late >= early));
        assert!(
            initial
                .values
                .iter()
                .zip(&no_aperture.values)
                .any(|(after, before)| *after > 0.0 && *before < 0.0)
        );
        assert_eq!(ellipsoid_value(aperture.center, aperture.center, aperture.axes), -0.8);
        assert!(
            ellipsoid_value(
                aperture.center + V3::new(1.2, 0.0, 0.0),
                aperture.center,
                aperture.axes
            ) > 0.0
        );
    }

    #[test]
    fn rounded_aperture_rejects_invalid_geometry_and_is_absent_by_default() {
        let mut cfg = config();
        assert!(serde_json::to_value(&cfg).unwrap().get("aperture").is_none());
        cfg.aperture = Some(RevealAperture { center: V3::ZERO, axes: [1.0, 0.0, 1.0] });
        assert!(cfg.validate().is_err());
        cfg.aperture = Some(RevealAperture { center: V3::new(f64::NAN, 0.0, 0.0), axes: [1.0; 3] });
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn smooth_max_rounding_only_removes_and_is_monotone_in_both_inputs() {
        let width = 0.16;
        for i in -50..=50 {
            for j in -50..=50 {
                let a = f64::from(i) * 0.02;
                let b = f64::from(j) * 0.02;
                let rounded = smooth_max(a, b, width);
                assert!(rounded >= a.max(b));
                assert!(smooth_max(a + 0.001, b, width) >= rounded);
                assert!(smooth_max(a, b + 0.001, width) >= rounded);
                assert_eq!(smooth_max(a, b, 0.0).to_bits(), a.max(b).to_bits());
                if (a - b).abs() >= width {
                    assert_eq!(rounded.to_bits(), a.max(b).to_bits());
                }
            }
        }
        assert_eq!(smooth_max(0.0, 0.0, width), width / 4.0);
    }

    #[test]
    fn rounded_field_preserves_zero_dose_blank_and_irreversible_prefixes() {
        let source = source(true);
        let cfg = config();
        let sharp = PreparedField::new(&source, &cfg).unwrap().grid(0.0, 25).unwrap().0;
        let field =
            PreparedField::new(&source, &FieldConfig { rim_rounding: 0.12, ..cfg }).unwrap();
        let (initial, _) = field.grid(0.0, 25).unwrap();
        let (early, _) = field.grid(0.2, 25).unwrap();
        let (late, evidence) = field.grid(0.8, 25).unwrap();
        // Width below the minimum cutter semi-axis leaves zero-dose membership
        // unchanged, although its interior scalar values may be blended.
        assert!(initial.values.iter().zip(&sharp.values).all(|(a, b)| (*a < 0.0) == (*b < 0.0)));
        assert!(early.values.iter().zip(&late.values).all(|(a, b)| b >= a));
        assert!(evidence.minimum_boundary_value > 0.0);
        let mut invalid = config();
        invalid.rim_rounding = 0.51;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn analytic_compact_kernel_gradient_matches_independent_differences() {
        let body = BodySample {
            position: V3::new(0.2, -0.1, 1.7),
            tangent: V3::new(0.0, 0.0, 1.0),
            normal: V3::new(1.0, 0.0, 0.0),
            binormal: V3::new(0.0, 1.0, 0.0),
            speed: 0.0,
            curvature: 0.0,
            proximity: 0.0,
            arc_length: 0.0,
        };
        let brush = Brush::new(body, [0.7, 0.45, 0.22], 0).unwrap();
        for offset in [V3::new(0.1, 0.07, 0.22), V3::new(-0.12, 0.01, 0.05), V3::new(0.0, 0.3, 0.0)]
        {
            let point = body.position + offset;
            let jet = brush.kernel_jet(point);
            assert_eq!(jet.value.to_bits(), brush.kernel(point).to_bits());
            let reference = central_difference(point, 1e-6, |sample| brush.kernel(sample));
            assert!(vector_length(jet.gradient.unwrap() - reference) < 2e-9);
        }
        assert_eq!(brush.kernel_jet(body.position).gradient, Some(V3::ZERO));
    }

    #[test]
    fn analytic_ellipsoid_and_smooth_max_derivatives_match_differences() {
        let point = V3::new(0.4, -0.25, 0.7);
        let center = V3::new(0.1, 0.05, -0.2);
        let axes = [1.9, 1.65, 1.45];
        let jet = ellipsoid_jet(point, center, axes);
        let reference =
            central_difference(point, 1e-6, |sample| ellipsoid_value(sample, center, axes));
        assert!(vector_length(jet.gradient.unwrap() - reference) < 2e-10);
        let a_gradient = V3::new(0.3, -0.5, 0.8);
        let b_gradient = V3::new(-0.2, 0.9, 0.1);
        let evaluate = |p: V3| {
            FieldJet { value: -0.03 + a_gradient.dot(p), gradient: Some(a_gradient) }.smooth_max(
                FieldJet { value: -0.02 + b_gradient.dot(p), gradient: Some(b_gradient) },
                0.16,
            )
        };
        let actual = evaluate(V3::ZERO).gradient.unwrap();
        let reference = central_difference(V3::ZERO, 1e-6, |p| evaluate(p).value);
        assert!(vector_length(actual - reference) < 1e-10);
    }

    #[test]
    fn sphere_and_ellipsoid_surface_normals_are_the_exact_outward_directions() {
        let source = source(false);
        let center = V3::new(0.1, -0.2, 0.15);
        for axes in [[1.2; 3], [1.9, 1.65, 1.45]] {
            let cfg = FieldConfig {
                stock_mode: StockMode::Ellipsoid,
                ellipsoid_center: center,
                ellipsoid_axes: axes,
                dose_strength: 0.0,
                ..config()
            };
            let field = PreparedField::new(&source, &cfg).unwrap();
            let (mut grid, _) = field.grid(0.0, 33).unwrap();
            // Cached-mesh shading requires lattice metadata only.
            grid.values.clear();
            let directions = [
                V3::new(1.0, 0.0, 0.0),
                V3::new(-0.3, 0.8, 0.4).normalized(),
                V3::new(0.4, -0.2, -0.7).normalized(),
            ];
            let points: Vec<V3> = directions
                .iter()
                .map(|u| center + V3::new(u.x * axes[0], u.y * axes[1], u.z * axes[2]))
                .collect();
            let result = field.surface_normals(&points, 0.0, &grid).unwrap();
            assert_eq!(result.finite_difference_fallbacks, 0);
            assert_eq!(result.stock_difference_step, 0.0);
            for ((point, normal), direction) in points.iter().zip(result.normals).zip(directions) {
                let delta = *point - center;
                let expected = V3::new(
                    delta.x / axes[0].powi(2),
                    delta.y / axes[1].powi(2),
                    delta.z / axes[2].powi(2),
                )
                .normalized();
                assert!(vector_length(normal - expected) < 1e-13);
                assert!(normal.dot(direction) > 0.0);
            }
        }
    }

    #[test]
    fn continuous_composed_field_derivative_matches_scalar_queries() {
        let source = source(true);
        let cfg = FieldConfig { stock_mode: StockMode::Ellipsoid, rim_rounding: 0.12, ..config() };
        let field = PreparedField::new(&source, &cfg).unwrap();
        let time = 0.67;
        let (grid, _) = field.grid(time, 33).unwrap();
        let weights = temporal_weights(cfg.time_samples, time);
        let bricks =
            BrushBricks::new(&field.brushes, &weights, grid.origin, grid.spacing, grid.dims);
        for fraction in [0.13, 0.27, 0.41] {
            let point =
                source.sample(fraction).unwrap().bodies[0].position + V3::new(0.03, 0.05, 0.08);
            let actual = field.field_jet(point, &weights, &bricks, &grid, grid.spacing * 0.02);
            let reference = central_difference(point, 1e-6, |p| {
                let stock = field.stock.value(p);
                let dose = dose_at(&field, p, time);
                let carving = (dose / cfg.dose_threshold - 1.0)
                    * cfg.cutter_axes.iter().copied().fold(f64::INFINITY, f64::min);
                smooth_max(stock, carving, cfg.rim_rounding)
            });
            assert!(vector_length(actual.gradient.unwrap() - reference) < 1e-7);
        }
    }

    #[test]
    fn continuous_surface_normals_are_identical_across_worker_counts() {
        let source = source(false);
        let cfg = FieldConfig { stock_mode: StockMode::Ellipsoid, dose_strength: 0.0, ..config() };
        let field = PreparedField::new(&source, &cfg).unwrap();
        let (grid, _) = field.grid(0.0, 25).unwrap();
        let points: Vec<V3> = (0..32)
            .map(|i| {
                let angle = f64::from(i) * TAU / 32.0;
                V3::new(
                    cfg.ellipsoid_axes[0] * angle.cos(),
                    cfg.ellipsoid_axes[1] * angle.sin(),
                    0.0,
                )
            })
            .collect();
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| field.surface_normals(&points, 0.0, &grid).unwrap())
        };
        let a = run(1);
        let b = run(3);
        assert_eq!(a.normals, b.normals);
        assert_eq!(a.minimum_gradient_length, b.minimum_gradient_length);
        assert_eq!(a.maximum_gradient_length, b.maximum_gradient_length);
        assert_eq!(a.finite_difference_fallbacks, b.finite_difference_fallbacks);
    }

    #[test]
    fn hard_junction_fallback_is_reported_and_unresolved_zero_is_an_error() {
        let source = source(false);
        let cfg = FieldConfig {
            stock_mode: StockMode::Ellipsoid,
            ellipsoid_axes: [1.0; 3],
            dose_strength: 0.0,
            reveal: Some(RevealPlane { normal: V3::new(0.0, 0.0, 1.0), offset: 0.0 }),
            ..config()
        };
        let field = PreparedField::new(&source, &cfg).unwrap();
        let (grid, _) = field.grid(0.0, 33).unwrap();
        let result = field.surface_normals(&[V3::new(1.0, 0.0, 0.0)], 0.0, &grid).unwrap();
        assert_eq!(result.finite_difference_fallbacks, 1);
        assert!(result.normals[0].x > 0.0 && result.normals[0].z > 0.0);
        let plain = PreparedField::new(&source, &FieldConfig { reveal: None, ..cfg }).unwrap();
        let error = plain.surface_normals(&[V3::ZERO], 0.0, &grid).unwrap_err().to_string();
        assert!(error.contains("undefined or zero field gradient"));
    }
}
