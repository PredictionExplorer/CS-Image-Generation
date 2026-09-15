//! Light Cast by Gravity: source-driven phase lenses and spectral caustic light.
//!
//! A thin-element ray map redistributes uniform incident light. Twelve groups
//! preserve the existing 64-bin XYZ weights, and conservative triangle flux
//! accumulation resolves folds without dividing by a pointwise Jacobian.
mod accumulate;

use super::{Camera, OrbitSeries, RenderConfig, SilkResult, V3};
use crate::spectrum::{BIN_XYZ_LUT, NUM_BINS, wavelength_nm_for_bin, xyz_to_linear_srgb};
use serde::{Deserialize, Serialize};

const BAND_COUNT: usize = 12;
const NEGATIVE_TOLERANCE: f64 = 1e-7;
type Point = [f64; 2];

/// Display the physical illuminated receiver or an artistic focusing-gain view.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightDisplay {
    /// Full irradiance on a warm diffuse ground, including depleted regions.
    #[default]
    Ivory,
    /// Positive spectral gain above the uniform reference on a dark background.
    DarkGain,
}

/// One smooth compact phase lens, associated with its original source body.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightLensConfig {
    /// Whether this body's lens contributes optical path.
    pub enabled: bool,
    /// Positive ellipse semi-axes in world units.
    pub semi_axes: [f64; 2],
    /// Fixed ellipse orientation in the source/receiver plane.
    pub angle_degrees: f64,
    /// Dimensionless focusing power at the reference wavelength; zero is identity.
    pub power: f64,
    /// Small fixed asymmetric thickness term, bounded in absolute value by 0.15.
    pub coma: f64,
    /// Fixed quadratic bend of the normalized transverse coordinate, with |bend| <= 0.5.
    /// Omitted at zero so existing straight-lens recipes retain their hash.
    #[serde(skip_serializing_if = "bend_is_zero")]
    pub bend: f64,
}

fn bend_is_zero(value: &f64) -> bool {
    *value == 0.0
}

impl Default for LightLensConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            semi_axes: [1.0, 0.72],
            angle_degrees: -20.0,
            power: 1.55,
            coma: 0.08,
            bend: 0.0,
        }
    }
}

/// Optical, source and presentation controls for a full spectral caustic frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightConfig {
    /// Fixed orthonormal source basis, explicitly stored for any seed-specific fit.
    pub source_axes: [V3; 3],
    /// Positive soft scales of the three bounded source components.
    pub source_scales: [f64; 3],
    /// Maximum horizontal and vertical lens-center movements.
    pub source_motion: [f64; 2],
    /// Maximum fractional power change from source depth and proximity.
    pub power_modulation: f64,
    /// Lens parameters in original body order A, B, C.
    pub lenses: [LightLensConfig; 3],
    /// Fixed source-plane grid intervals; all active cells are covered once.
    pub source_grid: [usize; 2],
    /// World-space width and height of the fixed sampled source domain.
    pub source_domain: [f64; 2],
    /// World-space center of the sampled source domain.
    pub source_center: [f64; 2],
    /// Optical distance from the virtual phase plate to the receiver at z=0.
    pub receiver_distance: f64,
    /// Refractive index at the reference wavelength.
    pub refractive_index: f64,
    /// Reference wavelength in nanometres for focusing power and dispersion.
    pub reference_wavelength_nm: f64,
    /// Cauchy-like index coefficient multiplying ((reference/wavelength)^2 - 1).
    pub dispersion: f64,
    /// Conservative maximum permitted transverse ray slope for this thin model.
    pub max_ray_slope: f64,
    /// Uniform per-band reference irradiance before focusing.
    pub incident_irradiance: f64,
    /// Finite optical footprint width measured in final output pixels.
    pub footprint_pixels: f64,
    /// Full illuminated ground or positive focusing gain.
    pub display: LightDisplay,
    /// Linear-RGB diffuse reflectance of the ivory receiver.
    pub receiver_tint: V3,
    /// Independent ambient irradiance on the ivory receiver.
    pub ambient_irradiance: f64,
    /// Fixed multiplier for the dark focusing-gain view.
    pub gain_scale: f64,
    /// Artistic dark-view contrast exponent in [1, 3], applied per spectral band.
    /// Omitted at one so existing recipes retain their hash and display behavior.
    #[serde(skip_serializing_if = "gain_exponent_is_one")]
    pub gain_exponent: f64,
}

fn gain_exponent_is_one(value: &f64) -> bool {
    *value == 1.0
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            source_axes: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)],
            source_scales: [1.5; 3],
            source_motion: [3.1, 1.7],
            power_modulation: 0.10,
            lenses: [
                LightLensConfig::default(),
                LightLensConfig {
                    semi_axes: [0.90, 0.65],
                    angle_degrees: 35.0,
                    power: 1.70,
                    coma: -0.10,
                    ..LightLensConfig::default()
                },
                LightLensConfig {
                    semi_axes: [1.10, 0.80],
                    angle_degrees: 78.0,
                    power: 1.45,
                    coma: 0.06,
                    ..LightLensConfig::default()
                },
            ],
            source_grid: [2304, 1536],
            source_domain: [9.6, 6.4],
            source_center: [0.0; 2],
            receiver_distance: 8.0,
            refractive_index: 1.50,
            reference_wavelength_nm: 550.0,
            dispersion: 0.008,
            max_ray_slope: 0.25,
            incident_irradiance: 0.55,
            footprint_pixels: 0.9,
            display: LightDisplay::Ivory,
            receiver_tint: V3::new(0.82, 0.78, 0.68),
            ambient_irradiance: 0.035,
            gain_scale: 1.0,
            gain_exponent: 1.0,
        }
    }
}

/// Optical budget for one representative wavelength and its summed XYZ weights.
///
/// Flux uses a common unit-irradiance spectral reference times world area. These
/// per-band values must not be added together and described as calibrated watts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightBandDiagnostics {
    /// First included bin of the existing 64-bin table.
    pub first_bin: usize,
    /// Exclusive final included bin.
    pub end_bin: usize,
    /// XYZ-weighted representative wavelength in nanometres.
    pub wavelength_nm: f64,
    /// Refractive index evaluated at the representative wavelength.
    pub refractive_index: f64,
    /// Sum of the original bin XYZ weights, without another normalization.
    pub xyz_weight: V3,
    /// Incident flux over the active union of source cells.
    pub incident_flux: f64,
    /// Mapped flux received inside the output crop after its finite footprint.
    pub received_flux: f64,
    /// Mapped flux escaping the crop; never redistributed back into the image.
    pub escaped_flux: f64,
    /// Unrefracted active-cell flux removed from the uniform reference in the crop.
    pub baseline_removed_flux: f64,
    /// Unrefracted active-cell flux whose reference footprint falls outside the crop.
    pub reference_escaped_flux: f64,
    /// Signed integrated crop redistribution; depletion can make this negative.
    pub net_crop_flux: f64,
    /// Integrated crop redistribution minus (received minus baseline-removed flux).
    pub conservation_residual: f64,
    /// Mapped triangles using a near-degenerate finite-footprint treatment.
    pub degenerate_triangles: usize,
    /// Mapped triangles whose orientation reversed through a caustic fold.
    pub reversed_triangles: usize,
    /// Smallest sampled determinant of the analytical ray-map Jacobian.
    pub minimum_jacobian: f64,
    /// Smallest irradiance before rounding tiny numerical negative residuals.
    pub minimum_irradiance: f64,
}

/// Reproducible source, sampling, and optical diagnostics for one shutter sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightDiagnostics {
    /// Exact requested fraction of the unchanged recorded source.
    pub source_fraction: f64,
    /// Uniform per-band irradiance of the unfocused spectral reference.
    pub reference_irradiance: f64,
    /// World-space area carried by one source triangle.
    pub source_triangle_area: f64,
    /// World-space area of one final receiver pixel.
    pub receiver_pixel_area: f64,
    /// Current source-driven lens centers in receiver-plane world coordinates.
    pub lens_centers: [[f64; 2]; 3],
    /// Actual dimensionless powers after bounded source modulation.
    pub lens_powers: [f64; 3],
    /// Actual central thickness multipliers in world units.
    pub lens_heights: [f64; 3],
    /// Configured fixed source-grid intervals.
    pub source_grid: [usize; 2],
    /// Source cells intersecting at least one active lens support.
    pub active_cells: usize,
    /// Cells outside the complete support union, omitted from optical work.
    pub skipped_cells: usize,
    /// Shared grid nodes whose optical derivatives were evaluated.
    pub evaluated_nodes: usize,
    /// Source triangles submitted for each wavelength band.
    pub triangles_per_band: usize,
    /// Total mapped triangles submitted across all twelve bands.
    pub spectral_triangle_count: usize,
    /// Conservative analytical slope bound summed over the active lenses.
    pub conservative_ray_slope_bound: f64,
    /// Largest slope sampled at source-grid nodes over the wavelength range.
    pub sampled_maximum_ray_slope: f64,
    /// Largest sampled edge/center ray-map interpolation error in output pixels.
    pub sampled_midpoint_error_pixels: f64,
    /// World-space mapped-node envelope: minimum x/y, maximum x/y.
    pub sampled_receiver_bounds: [f64; 4],
    /// Y-weighted incident flux; a luminance budget, not calibrated radiant watts.
    pub incident_luminance_flux: f64,
    /// Y-weighted mapped flux received inside the crop.
    pub received_luminance_flux: f64,
    /// Y-weighted mapped flux escaping the crop.
    pub escaped_luminance_flux: f64,
    /// Number of band/pixel values with tiny negative irradiance rounded to zero.
    pub rounded_negative_values: usize,
    /// Number of output colors gently desaturated to nonnegative linear RGB.
    pub gamut_compressed_pixels: usize,
    /// Full per-band spectral and flux diagnostics.
    pub bands: Vec<LightBandDiagnostics>,
}

/// One untone-mapped linear-light frame, ready for common shutter accumulation.
#[derive(Clone, Debug)]
pub struct LightFrame {
    /// Row-major linear-RGB pixels, matching the requested output dimensions.
    pub pixels: Vec<V3>,
    /// Optical diagnostics for this exact source time.
    pub diagnostics: LightDiagnostics,
}

impl LightDiagnostics {
    /// Validate a complete optical record before assembly or video encoding.
    ///
    /// Flux checks allow `2e-7 * max(incident_flux, 1)` absolute numerical error
    /// per band. Signed crop redistribution and Jacobian determinants may be
    /// negative; incident, received, escaped and reference fluxes may not.
    pub fn validate(&self) -> SilkResult<()> {
        if !self.source_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.source_fraction)
            || [self.reference_irradiance, self.source_triangle_area, self.receiver_pixel_area]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || !self.lens_centers.iter().flatten().all(|v| v.is_finite())
            || self
                .lens_powers
                .iter()
                .chain(self.lens_heights.iter())
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.source_grid.iter().any(|&n| n < 2)
            || self.source_grid[0].checked_mul(self.source_grid[1])
                != self.active_cells.checked_add(self.skipped_cells)
            || self.active_cells.checked_mul(2) != Some(self.triangles_per_band)
            || self.triangles_per_band.checked_mul(BAND_COUNT) != Some(self.spectral_triangle_count)
            || self.bands.len() != BAND_COUNT
        {
            return Err(
                "light diagnostics have invalid source, dimensions, or spectral counts".into()
            );
        }
        let nodes = self.source_grid[0]
            .checked_add(1)
            .and_then(|x| self.source_grid[1].checked_add(1).and_then(|y| x.checked_mul(y)))
            .ok_or("light diagnostic grid overflows")?;
        if self.evaluated_nodes > nodes
            || (self.active_cells > 0 && self.evaluated_nodes < 4)
            || [
                self.conservative_ray_slope_bound,
                self.sampled_maximum_ray_slope,
                self.sampled_midpoint_error_pixels,
                self.incident_luminance_flux,
                self.received_luminance_flux,
                self.escaped_luminance_flux,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || !self.sampled_receiver_bounds.iter().all(|v| v.is_finite())
            || self.sampled_receiver_bounds[0] > self.sampled_receiver_bounds[2]
            || self.sampled_receiver_bounds[1] > self.sampled_receiver_bounds[3]
            || self.sampled_maximum_ray_slope > self.conservative_ray_slope_bound + 1e-10
        {
            return Err("light diagnostics contain invalid sampled optical bounds".into());
        }
        let expected_flux =
            self.reference_irradiance * self.source_triangle_area * self.triangles_per_band as f64;
        let mut expected_luminance = [0.0; 3];
        for (band, expected) in self.bands.iter().zip(spectral_bands()) {
            let tolerance = 2e-7 * band.incident_flux.max(1.0);
            let nonnegative = [
                band.incident_flux,
                band.received_flux,
                band.escaped_flux,
                band.baseline_removed_flux,
                band.reference_escaped_flux,
            ];
            if nonnegative.iter().any(|v| !v.is_finite() || *v < 0.0)
                || !band.net_crop_flux.is_finite()
                || !band.conservation_residual.is_finite()
                || !band.minimum_jacobian.is_finite()
                || !band.minimum_irradiance.is_finite()
                || band.minimum_irradiance < -NEGATIVE_TOLERANCE * self.reference_irradiance
                || band.first_bin != expected.first
                || band.end_bin != expected.end
                || !band.wavelength_nm.is_finite()
                || (band.wavelength_nm - expected.wavelength).abs() > 1e-9
                || !band.xyz_weight.is_finite()
                || (band.xyz_weight - expected.weight).length() > 1e-12
                || !band.refractive_index.is_finite()
                || band.refractive_index <= 1.0
                || band.degenerate_triangles > self.triangles_per_band
                || band.reversed_triangles > self.triangles_per_band
                || (band.incident_flux - expected_flux).abs() > tolerance
                || (band.incident_flux - band.received_flux - band.escaped_flux).abs() > tolerance
                || (band.incident_flux - band.baseline_removed_flux - band.reference_escaped_flux)
                    .abs()
                    > tolerance
                || band.conservation_residual.abs() > tolerance
                || (band.net_crop_flux
                    - (band.received_flux - band.baseline_removed_flux)
                    - band.conservation_residual)
                    .abs()
                    > tolerance
            {
                return Err("light band diagnostics have non-finite, incomplete, or inconsistent flux accounting".into());
            }
            for (sum, flux) in expected_luminance.iter_mut().zip([
                band.incident_flux,
                band.received_flux,
                band.escaped_flux,
            ]) {
                *sum += band.xyz_weight.y * flux;
            }
        }
        for (actual, expected) in [
            self.incident_luminance_flux,
            self.received_luminance_flux,
            self.escaped_luminance_flux,
        ]
        .into_iter()
        .zip(expected_luminance)
        {
            if (actual - expected).abs() > 2e-7 * expected.max(1.0) {
                return Err(
                    "light aggregate luminance budget differs from its spectral records".into()
                );
            }
        }
        Ok(())
    }
}

/// One positive-flux source triangle and its mapped footprint in final pixels.
#[derive(Clone, Copy)]
pub(super) struct MappedTriangle {
    /// Unrefracted pixel coordinates of the three source vertices.
    pub source: [[f64; 2]; 3],
    /// Refracted pixel coordinates at the receiver.
    pub receiver: [[f64; 2]; 3],
    /// Positive incident irradiance times source-triangle world area.
    pub flux: f64,
}

#[derive(Clone, Copy, Debug)]
struct SpectralBand {
    first: usize,
    end: usize,
    wavelength: f64,
    weight: V3,
}

fn spectral_bands() -> [SpectralBand; BAND_COUNT] {
    std::array::from_fn(|band| {
        let first = band * NUM_BINS / BAND_COUNT;
        let end = (band + 1) * NUM_BINS / BAND_COUNT;
        let mut weight = V3::ZERO;
        let mut weighted_wavelength = 0.0;
        let mut scalar_weight = 0.0;
        for bin in first..end {
            let (x, y, z, _) = BIN_XYZ_LUT[bin];
            weight += V3::new(x, y, z);
            let observer_weight = x + y + z;
            weighted_wavelength += wavelength_nm_for_bin(bin) * observer_weight;
            scalar_weight += observer_weight;
        }
        SpectralBand { first, end, wavelength: weighted_wavelength / scalar_weight, weight }
    })
}

fn refractive_index(wavelength: f64, config: &LightConfig) -> f64 {
    config.refractive_index
        + config.dispersion * ((config.reference_wavelength_nm / wavelength).powi(2) - 1.0)
}

#[derive(Clone, Copy, Debug)]
struct Lens {
    center: Point,
    axes: Point,
    cosine: f64,
    sine: f64,
    height: f64,
    power: f64,
    coma: f64,
    bend: f64,
    extent: Point,
    quadratic: [f64; 3],
}

impl Lens {
    fn phase(self, point: Point) -> Phase {
        if self.height == 0.0 {
            return Phase::default();
        }
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let x = self.cosine * dx + self.sine * dy;
        let y = -self.sine * dx + self.cosine * dy;
        if self.bend != 0.0 {
            return self.curved_phase(x, y);
        }
        // Preserve the original operation order at zero bend: old recipes
        // retain their exact phase, ray map, and accumulated pixel values.
        let ax = 1.0 / self.axes[0].powi(2);
        let ay = 1.0 / self.axes[1].powi(2);
        let radius = x * x * ax + y * y * ay;
        if radius >= 1.0 {
            return Phase::default();
        }
        let p = 1.0 - radius;
        let qx = self.coma / self.axes[0];
        let q = 1.0 + qx * x;
        let sx = 2.0 * x * ax;
        let sy = 2.0 * y * ay;
        let gx = self.height * (-4.0 * p.powi(3) * sx * q + p.powi(4) * qx);
        let gy = self.height * (-4.0 * p.powi(3) * sy * q);
        let xx = self.height
            * (12.0 * p.powi(2) * sx * sx * q
                - 8.0 * p.powi(3) * ax * q
                - 8.0 * p.powi(3) * sx * qx);
        let xy = self.height * (12.0 * p.powi(2) * sx * sy * q - 4.0 * p.powi(3) * sy * qx);
        let yy = self.height * (12.0 * p.powi(2) * sy * sy * q - 8.0 * p.powi(3) * ay * q);
        let c = self.cosine;
        let s = self.sine;
        Phase {
            height: self.height * p.powi(4) * q,
            gradient: [c * gx - s * gy, s * gx + c * gy],
            hessian: [
                c * c * xx - 2.0 * c * s * xy + s * s * yy,
                c * s * (xx - yy) + (c * c - s * s) * xy,
                s * s * xx + 2.0 * c * s * xy + c * c * yy,
            ],
        }
    }

    fn curved_phase(self, x: f64, y: f64) -> Phase {
        let [a, b] = self.axes;
        let u = x / a;
        let v = y / b - self.bend * u * u;
        let radius = u * u + v * v;
        if radius >= 1.0 {
            return Phase::default();
        }
        let p = 1.0 - radius;
        let q = 1.0 + self.coma * u;
        let qx = self.coma / a;
        // Derivatives of s = u^2 + (eta/b - bend*u^2)^2, in the
        // rotated physical coordinates xi, eta. The second derivatives of
        // this coordinate warp are essential to the caustic Jacobian.
        let sx = 2.0 * u * (1.0 - 2.0 * self.bend * v) / a;
        let sy = 2.0 * v / b;
        let sxx = 2.0 * (1.0 - 2.0 * self.bend * v + 4.0 * self.bend.powi(2) * u * u) / a.powi(2);
        let sxy = -4.0 * self.bend * u / (a * b);
        let syy = 2.0 / b.powi(2);
        let gx = self.height * (-4.0 * p.powi(3) * sx * q + p.powi(4) * qx);
        let gy = self.height * (-4.0 * p.powi(3) * sy * q);
        let xx = self.height
            * (12.0 * p.powi(2) * sx * sx * q
                - 4.0 * p.powi(3) * sxx * q
                - 8.0 * p.powi(3) * sx * qx);
        let xy = self.height
            * (12.0 * p.powi(2) * sx * sy * q
                - 4.0 * p.powi(3) * sxy * q
                - 4.0 * p.powi(3) * sy * qx);
        let yy = self.height * (12.0 * p.powi(2) * sy * sy * q - 4.0 * p.powi(3) * syy * q);
        let c = self.cosine;
        let s = self.sine;
        Phase {
            height: self.height * p.powi(4) * q,
            gradient: [c * gx - s * gy, s * gx + c * gy],
            hessian: [
                c * c * xx - 2.0 * c * s * xy + s * s * yy,
                c * s * (xx - yy) + (c * c - s * s) * xy,
                s * s * xx + 2.0 * c * s * xy + c * c * yy,
            ],
        }
    }

    fn intersects(self, low: Point, high: Point) -> bool {
        if self.height == 0.0
            || high[0] < self.center[0] - self.extent[0]
            || low[0] > self.center[0] + self.extent[0]
            || high[1] < self.center[1] - self.extent[1]
            || low[1] > self.center[1] + self.extent[1]
        {
            return false;
        }
        // A bent support is not a quadratic ellipse. Every cell in its
        // conservative world AABB is a candidate; unaffected triangles use
        // exact identity cancellation in the common flux accumulator.
        if self.bend != 0.0 {
            return true;
        }
        if (low[0]..=high[0]).contains(&self.center[0])
            && (low[1]..=high[1]).contains(&self.center[1])
        {
            return true;
        }
        let [xx, xy, yy] = self.quadratic;
        let evaluate = |point: Point| {
            let x = point[0] - self.center[0];
            let y = point[1] - self.center[1];
            xx * x * x + 2.0 * xy * x * y + yy * y * y
        };
        for x in [low[0], high[0]] {
            let y = (self.center[1] - xy / yy * (x - self.center[0])).clamp(low[1], high[1]);
            if evaluate([x, y]) <= 1.0 {
                return true;
            }
        }
        for y in [low[1], high[1]] {
            let x = (self.center[0] - xy / xx * (y - self.center[1])).clamp(low[0], high[0]);
            if evaluate([x, y]) <= 1.0 {
                return true;
            }
        }
        false
    }

    fn gradient_bound(self) -> f64 {
        let radial_max = 8.0 * (6.0_f64 / 7.0).powi(3) / 7.0_f64.sqrt();
        let straight = self.height
            * (radial_max * (1.0 + self.coma.abs()) / self.axes[0].min(self.axes[1])
                + self.coma.abs() / self.axes[0]);
        if self.bend == 0.0 {
            return straight;
        }
        // The coordinate warp adds -2*bend*u/a times the normalized
        // transverse derivative. With |u*v| <= r^2/2, its maximum is bounded
        // by 8*max(r^2*(1-r^2)^3) = 27/32, for 0 <= r <= 1.
        straight
            + self.height * (27.0 / 32.0) * self.bend.abs() * (1.0 + self.coma.abs()) / self.axes[0]
    }
}

fn support_extent(axes: Point, cosine: f64, sine: f64, bend: f64) -> Point {
    let [a, b] = axes;
    let straight = [(a * cosine).hypot(b * sine), (a * sine).hypot(b * cosine)];
    if bend == 0.0 {
        return straight;
    }
    // Support points are the ordinary ellipse plus a local transverse shift
    // b*bend*u^2. Bound that shift in each world coordinate before culling.
    let displacement = b * bend.abs();
    [straight[0] + displacement * sine.abs(), straight[1] + displacement * cosine.abs()]
}

#[derive(Clone, Copy, Debug, Default)]
struct Phase {
    height: f64,
    gradient: Point,
    /// xx, xy, yy of the symmetric world-space Hessian.
    hessian: [f64; 3],
}

fn phase(lenses: &[Lens; 3], point: Point) -> Phase {
    let mut total = Phase::default();
    for &lens in lenses {
        let value = lens.phase(point);
        total.height += value.height;
        for axis in 0..2 {
            total.gradient[axis] += value.gradient[axis];
        }
        for axis in 0..3 {
            total.hessian[axis] += value.hessian[axis];
        }
    }
    total
}

fn lenses(source: &OrbitSeries, time: f64, config: &LightConfig) -> SilkResult<[Lens; 3]> {
    let frame =
        source.sample(time).ok_or("light source time lies outside the recorded interval")?;
    let lenses: [Lens; 3] = std::array::from_fn(|body| {
        let sample = frame.bodies[body];
        let q: [f64; 3] = std::array::from_fn(|axis| {
            (sample.position.dot(config.source_axes[axis]) / config.source_scales[axis]).tanh()
        });
        let shape = config.lenses[body];
        let power = if shape.enabled {
            shape.power
                * (1.0
                    + config.power_modulation * (0.6 * q[2] + 0.4 * (2.0 * sample.proximity - 1.0)))
        } else {
            0.0
        };
        let [a, b] = shape.semi_axes;
        let (sine, cosine) = shape.angle_degrees.to_radians().sin_cos();
        let ax = 1.0 / (a * a);
        let ay = 1.0 / (b * b);
        Lens {
            center: [config.source_motion[0] * q[0], config.source_motion[1] * q[1]],
            axes: shape.semi_axes,
            cosine,
            sine,
            height: power * a * b
                / (8.0 * config.receiver_distance * (config.refractive_index - 1.0)),
            power,
            coma: shape.coma,
            bend: shape.bend,
            extent: support_extent(shape.semi_axes, cosine, sine, shape.bend),
            quadratic: [
                cosine * cosine * ax + sine * sine * ay,
                cosine * sine * (ax - ay),
                sine * sine * ax + cosine * cosine * ay,
            ],
        }
    });
    if lenses.iter().any(|lens| {
        !lens
            .center
            .iter()
            .chain(lens.extent.iter())
            .chain(lens.quadratic.iter())
            .all(|v| v.is_finite())
            || !lens.height.is_finite()
            || !lens.power.is_finite()
            || (lens.power > 0.0 && lens.height <= 0.0)
    }) {
        return Err("light lens parameters exceed representable optical geometry".into());
    }
    Ok(lenses)
}

struct Receiver {
    target: Point,
    right: Point,
    up: Point,
    scale: f64,
    width: usize,
    height: usize,
}

impl Receiver {
    fn new(camera: &Camera, render: &RenderConfig) -> SilkResult<Self> {
        if !camera.position.is_finite()
            || !camera.target.is_finite()
            || !camera.up.is_finite()
            || !camera.orthographic_height.is_finite()
            || camera.orthographic_height <= 0.0
            || (camera.position.x - camera.target.x).abs() > 1e-8
            || (camera.position.y - camera.target.y).abs() > 1e-8
            || camera.target.z.abs() > 1e-8
            || camera.position.z <= 0.0
            || camera.up.z.abs() > 1e-8
            || camera.up.x.hypot(camera.up.y) < 1e-10
        {
            return Err("light camera must face receiver z=0 frontally; target XY, height, and in-plane up/roll are supported".into());
        }
        let norm = camera.up.x.hypot(camera.up.y);
        let scale = f64::from(render.height) / camera.orthographic_height;
        let area = 1.0 / (scale * scale);
        if !norm.is_finite() || !scale.is_finite() || !area.is_finite() || area <= 0.0 {
            return Err("light camera scale or up direction exceeds finite projection range".into());
        }
        let up = [camera.up.x / norm, camera.up.y / norm];
        Ok(Self {
            target: [camera.target.x, camera.target.y],
            right: [up[1], -up[0]],
            up,
            scale,
            width: render.width as usize,
            height: render.height as usize,
        })
    }

    fn project(&self, point: Point) -> Point {
        let relative = [point[0] - self.target[0], point[1] - self.target[1]];
        [
            self.width as f64 * 0.5 + dot(relative, self.right) * self.scale,
            self.height as f64 * 0.5 - dot(relative, self.up) * self.scale,
        ]
    }

    fn gradient_pixels(&self, gradient: Point) -> Point {
        [dot(gradient, self.right) * self.scale, -dot(gradient, self.up) * self.scale]
    }
}

#[derive(Clone, Copy)]
struct Node {
    point: Point,
    pixel: Point,
    phase: Phase,
}

struct Grid {
    nodes: Vec<Node>,
    cells: Vec<[u32; 4]>,
    triangle_area: f64,
    midpoint_error: f64,
}

fn prepare_grid(
    lenses: &[Lens; 3],
    config: &LightConfig,
    receiver: &Receiver,
    optical_scale: f64,
) -> SilkResult<Grid> {
    let [nx, ny] = config.source_grid;
    let step = [config.source_domain[0] / nx as f64, config.source_domain[1] / ny as f64];
    let origin = [
        config.source_center[0] - config.source_domain[0] * 0.5,
        config.source_center[1] - config.source_domain[1] * 0.5,
    ];
    let stride = nx + 1;
    let mut lookup = vec![u32::MAX; stride * (ny + 1)];
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    let mut midpoint_error: f64 = 0.0;
    for y in 0..ny {
        for x in 0..nx {
            let low = [origin[0] + x as f64 * step[0], origin[1] + y as f64 * step[1]];
            let high = [origin[0] + (x + 1) as f64 * step[0], origin[1] + (y + 1) as f64 * step[1]];
            if !lenses.iter().any(|lens| lens.intersects(low, high)) {
                continue;
            }
            let mut cell = [0_u32; 4];
            for (corner, (xx, yy)) in
                [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)].into_iter().enumerate()
            {
                let index = yy * stride + xx;
                if lookup[index] == u32::MAX {
                    let point = [origin[0] + xx as f64 * step[0], origin[1] + yy as f64 * step[1]];
                    lookup[index] = u32::try_from(nodes.len())?;
                    nodes.push(Node {
                        point,
                        pixel: receiver.project(point),
                        phase: phase(lenses, point),
                    });
                }
                cell[corner] = lookup[index];
            }
            for (first, last) in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)] {
                let a = nodes[cell[first] as usize];
                let b = nodes[cell[last] as usize];
                let point = [(a.point[0] + b.point[0]) * 0.5, (a.point[1] + b.point[1]) * 0.5];
                let actual = phase(lenses, point).gradient;
                let predicted = [
                    (a.phase.gradient[0] + b.phase.gradient[0]) * 0.5,
                    (a.phase.gradient[1] + b.phase.gradient[1]) * 0.5,
                ];
                midpoint_error = midpoint_error.max(
                    (actual[0] - predicted[0]).hypot(actual[1] - predicted[1])
                        * optical_scale
                        * receiver.scale,
                );
            }
            cells.push(cell);
        }
    }
    Ok(Grid { nodes, cells, triangle_area: step[0] * step[1] * 0.5, midpoint_error })
}

/// Render current-body optical caustics into untone-mapped linear RGB.
///
/// The receiver is the world z=0 plane. A frontal orthographic camera controls
/// its actual crop, scale and in-plane roll. Source sampling and the optical
/// model use the exact requested time; shutter averaging belongs to the caller.
pub fn render_linear(
    source: &OrbitSeries,
    time: f64,
    config: &LightConfig,
    camera: &Camera,
    render: &RenderConfig,
) -> SilkResult<LightFrame> {
    validate(time, config, render)?;
    let receiver = Receiver::new(camera, render)?;
    let lenses = lenses(source, time, config)?;
    let bands = spectral_bands();
    let maximum_index =
        bands.iter().map(|band| refractive_index(band.wavelength, config)).fold(1.0_f64, f64::max);
    let slope_bound =
        lenses.iter().map(|lens| lens.gradient_bound()).sum::<f64>() * (maximum_index - 1.0);
    if slope_bound > config.max_ray_slope {
        return Err(format!("light conservative ray-slope bound {slope_bound:.6} exceeds max_ray_slope; increase optical distance or reduce lens power").into());
    }
    let grid =
        prepare_grid(&lenses, config, &receiver, config.receiver_distance * (maximum_index - 1.0))?;
    let mut xyz = vec![V3::ZERO; receiver.width * receiver.height];
    let mut diagnostics = LightDiagnostics {
        source_fraction: time,
        reference_irradiance: config.incident_irradiance,
        source_triangle_area: grid.triangle_area,
        receiver_pixel_area: 1.0 / (receiver.scale * receiver.scale),
        lens_centers: lenses.map(|lens| lens.center),
        lens_powers: lenses.map(|lens| lens.power),
        lens_heights: lenses.map(|lens| lens.height),
        source_grid: config.source_grid,
        active_cells: grid.cells.len(),
        skipped_cells: config.source_grid[0] * config.source_grid[1] - grid.cells.len(),
        evaluated_nodes: grid.nodes.len(),
        triangles_per_band: grid.cells.len() * 2,
        spectral_triangle_count: grid.cells.len() * 2 * BAND_COUNT,
        conservative_ray_slope_bound: slope_bound,
        sampled_maximum_ray_slope: 0.0,
        sampled_midpoint_error_pixels: grid.midpoint_error,
        sampled_receiver_bounds: [0.0; 4],
        incident_luminance_flux: 0.0,
        received_luminance_flux: 0.0,
        escaped_luminance_flux: 0.0,
        rounded_negative_values: 0,
        gamut_compressed_pixels: 0,
        bands: Vec::with_capacity(BAND_COUNT),
    };
    let mut bounds = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for band in bands {
        let index = refractive_index(band.wavelength, config);
        let optical_scale = config.receiver_distance * (index - 1.0);
        let mut minimum_jacobian = 1.0_f64;
        let mut mapped = Vec::with_capacity(grid.nodes.len());
        for node in &grid.nodes {
            let [xx, xy, yy] = node.phase.hessian;
            let determinant = (1.0 + optical_scale * xx) * (1.0 + optical_scale * yy)
                - (optical_scale * xy).powi(2);
            minimum_jacobian = minimum_jacobian.min(determinant);
            diagnostics.sampled_maximum_ray_slope = diagnostics
                .sampled_maximum_ray_slope
                .max(node.phase.gradient[0].hypot(node.phase.gradient[1]) * (index - 1.0));
            let point = [
                node.point[0] + optical_scale * node.phase.gradient[0],
                node.point[1] + optical_scale * node.phase.gradient[1],
            ];
            for axis in 0..2 {
                bounds[axis] = bounds[axis].min(point[axis]);
                bounds[axis + 2] = bounds[axis + 2].max(point[axis]);
            }
            let gradient_pixel = receiver.gradient_pixels(node.phase.gradient);
            mapped.push([
                node.pixel[0] + optical_scale * gradient_pixel[0],
                node.pixel[1] + optical_scale * gradient_pixel[1],
            ]);
        }
        let mut triangles = Vec::with_capacity(grid.cells.len() * 2);
        let mut reversed_triangles = 0;
        for cell in &grid.cells {
            for local in [[0, 1, 2], [1, 3, 2]] {
                let indices = local.map(|corner| cell[corner] as usize);
                let original = indices.map(|i| grid.nodes[i].pixel);
                let projected = indices.map(|i| mapped[i]);
                if signed_area(original) * signed_area(projected) < 0.0 {
                    reversed_triangles += 1;
                }
                triangles.push(MappedTriangle {
                    source: original,
                    receiver: projected,
                    flux: config.incident_irradiance * grid.triangle_area,
                });
            }
        }
        let accumulated = accumulate::accumulate(
            &triangles,
            receiver.width,
            receiver.height,
            render.aa as usize,
            1.0 / (receiver.scale * receiver.scale),
            config.footprint_pixels,
        )?;
        if accumulated.delta_irradiance.len() != xyz.len() {
            return Err("light accumulator returned a mismatched receiving image".into());
        }
        let mut minimum_irradiance = config.incident_irradiance;
        for (color, &delta) in xyz.iter_mut().zip(&accumulated.delta_irradiance) {
            let irradiance = config.incident_irradiance + delta;
            minimum_irradiance = minimum_irradiance.min(irradiance);
            if !irradiance.is_finite()
                || irradiance < -NEGATIVE_TOLERANCE * config.incident_irradiance
            {
                return Err(format!("light band {} has nonphysical irradiance {irradiance}; flux redistribution needs correction",band.first).into());
            }
            if irradiance < 0.0 {
                diagnostics.rounded_negative_values += 1;
            }
            let signal = match config.display {
                LightDisplay::Ivory => irradiance.max(0.0),
                LightDisplay::DarkGain => dark_gain_signal(delta, irradiance, config)?,
            };
            *color += band.weight * signal;
        }
        diagnostics.incident_luminance_flux += band.weight.y * accumulated.incident_flux;
        diagnostics.received_luminance_flux += band.weight.y * accumulated.received_flux;
        diagnostics.escaped_luminance_flux += band.weight.y * accumulated.escaped_flux;
        diagnostics.bands.push(LightBandDiagnostics {
            first_bin: band.first,
            end_bin: band.end,
            wavelength_nm: band.wavelength,
            refractive_index: index,
            xyz_weight: band.weight,
            incident_flux: accumulated.incident_flux,
            received_flux: accumulated.received_flux,
            escaped_flux: accumulated.escaped_flux,
            baseline_removed_flux: accumulated.baseline_removed_flux,
            reference_escaped_flux: accumulated.reference_escaped_flux,
            net_crop_flux: accumulated.received_flux - accumulated.baseline_removed_flux
                + accumulated.conservation_residual,
            conservation_residual: accumulated.conservation_residual,
            degenerate_triangles: accumulated.degenerate_triangles,
            reversed_triangles,
            minimum_jacobian,
            minimum_irradiance,
        });
    }
    if !grid.nodes.is_empty() {
        diagnostics.sampled_receiver_bounds = bounds;
    }
    let mut pixels = Vec::with_capacity(xyz.len());
    for color in xyz {
        let (mut rgb, compressed) = nonnegative_rgb(color)?;
        diagnostics.gamut_compressed_pixels += usize::from(compressed);
        rgb = match config.display {
            LightDisplay::Ivory => {
                rgb.hadamard(config.receiver_tint)
                    + config.receiver_tint * config.ambient_irradiance
            }
            LightDisplay::DarkGain => rgb + render.background,
        };
        if !rgb.is_finite() {
            return Err("light spectral conversion exceeded finite RGB range".into());
        }
        pixels.push(rgb);
    }
    diagnostics.validate()?;
    Ok(LightFrame { pixels, diagnostics })
}

fn dark_gain_signal(delta: f64, irradiance: f64, config: &LightConfig) -> SilkResult<f64> {
    let signal = if config.gain_exponent == 1.0 {
        // Keep the original operation order, including the subtraction after
        // division. Direct delta/incident differs in its lowest floating bits.
        ((irradiance / config.incident_irradiance) - 1.0).max(0.0) * config.gain_scale
    } else {
        (delta / config.incident_irradiance).max(0.0).powf(config.gain_exponent) * config.gain_scale
    };
    if !signal.is_finite() {
        return Err("light display contrast exceeded finite irradiance range".into());
    }
    Ok(signal)
}

fn nonnegative_rgb(xyz: V3) -> SilkResult<(V3, bool)> {
    if !xyz.is_finite() {
        return Err("light spectral accumulation exceeded finite XYZ range".into());
    }
    let (r, g, b) = xyz_to_linear_srgb(xyz.x, xyz.y, xyz.z);
    let rgb = V3::new(r, g, b);
    if !rgb.is_finite() {
        return Err("light spectral conversion exceeded finite RGB range".into());
    }
    let minimum = r.min(g).min(b);
    if minimum >= 0.0 {
        return Ok((rgb, false));
    }
    if xyz.y <= 0.0 {
        return Ok((V3::ZERO, true));
    }
    let amount = xyz.y / (xyz.y - minimum);
    let neutral = V3::new(xyz.y, xyz.y, xyz.y);
    let desaturated = neutral + (rgb - neutral) * amount;
    if !desaturated.is_finite() {
        return Err("light gamut compression exceeded finite RGB range".into());
    }
    Ok((desaturated.max(V3::ZERO), true))
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn signed_area(points: [Point; 3]) -> f64 {
    let a = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
    let b = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
    (a[0] * b[1] - a[1] * b[0]) * 0.5
}

fn validate(time: f64, config: &LightConfig, render: &RenderConfig) -> SilkResult<()> {
    if !time.is_finite() || !(0.0..=1.0).contains(&time) {
        return Err("light time must be a finite recorded-source fraction in [0,1]".into());
    }
    if config.source_grid.iter().any(|&n| n < 2)
        || config.source_grid[0]
            .checked_add(1)
            .and_then(|x| config.source_grid[1].checked_add(1).and_then(|y| x.checked_mul(y)))
            .is_none_or(|count| count > 40_000_000)
        || render.width == 0
        || render.height == 0
        || !(1..=8).contains(&render.aa)
        || (render.width as usize)
            .checked_mul(render.height as usize)
            .is_none_or(|n| n > 100_000_000)
    {
        return Err(
            "light source grid or receiving-image dimensions exceed supported limits".into()
        );
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
            return Err("light source_axes must be a fixed orthonormal basis".into());
        }
    }
    if [
        config.receiver_distance,
        config.incident_irradiance,
        config.footprint_pixels,
        config.max_ray_slope,
    ]
    .iter()
    .chain(config.source_scales.iter())
    .chain(config.source_domain.iter())
    .any(|v| !v.is_finite() || *v <= 0.0)
        || config
            .source_motion
            .iter()
            .chain([&config.dispersion, &config.ambient_irradiance, &config.gain_scale])
            .any(|v| !v.is_finite() || *v < 0.0)
        || config.source_center.iter().any(|v| !v.is_finite())
        || !config.refractive_index.is_finite()
        || config.refractive_index <= 1.0
        || !config.reference_wavelength_nm.is_finite()
        || !(380.0..=700.0).contains(&config.reference_wavelength_nm)
        || !(0.0..=0.35).contains(&config.power_modulation)
        || !config.gain_exponent.is_finite()
        || !(1.0..=3.0).contains(&config.gain_exponent)
        || !config.receiver_tint.is_finite()
        || config.receiver_tint.min(V3::ZERO) != V3::ZERO
        || !render.background.is_finite()
        || render.background.min(V3::ZERO) != V3::ZERO
    {
        return Err(
            "light optical/source controls must be finite with valid signs and ranges".into()
        );
    }
    for shape in config.lenses {
        if shape.semi_axes.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !shape.angle_degrees.is_finite()
            || !shape.power.is_finite()
            || shape.power < 0.0
            || !shape.coma.is_finite()
            || shape.coma.abs() > 0.15
            || !shape.bend.is_finite()
            || shape.bend.abs() > 0.5
        {
            return Err(
                "light lenses need positive finite axes, nonnegative power, |coma| <= 0.15 and |bend| <= 0.5"
                    .into(),
            );
        }
        if shape.enabled && shape.power > 0.0 {
            let (s, c) = shape.angle_degrees.to_radians().sin_cos();
            for (axis, extent) in
                support_extent(shape.semi_axes, c, s, shape.bend).into_iter().enumerate()
            {
                if config.source_domain[axis] * 0.5
                    < config.source_center[axis].abs() + config.source_motion[axis] + extent
                {
                    return Err("light source domain must cover the complete bounded motion of every lens support".into());
                }
            }
        }
    }
    for band in spectral_bands() {
        let index = refractive_index(band.wavelength, config);
        if !index.is_finite() || index <= 1.0 {
            return Err(
                "light refractive index must stay finite and above one across all spectral bands"
                    .into(),
            );
        }
    }
    let triangle_area = (config.source_domain[0] / config.source_grid[0] as f64)
        * (config.source_domain[1] / config.source_grid[1] as f64)
        * 0.5;
    if !triangle_area.is_finite() || triangle_area <= 0.0 {
        return Err("light source triangle area exceeds finite numeric range".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;
    use sha2::{Digest, Sha256};

    fn source() -> OrbitSeries {
        let samples = (0..129)
            .map(|i| {
                let t = f64::from(i) / 128.0;
                std::array::from_fn(|body| {
                    let angle = std::f64::consts::TAU * (t + body as f64 / 3.0);
                    V3::new(angle.cos(), angle.sin(), 0.3 * (1.3 * angle).sin())
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "light-test".into(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({"synthetic_unit_fixture":true}),
        })
        .unwrap()
    }

    fn camera() -> Camera {
        Camera {
            position: V3::new(0.0, 0.0, 10.0),
            target: V3::ZERO,
            up: V3::new(0.0, 1.0, 0.0),
            orthographic_height: 6.0,
        }
    }

    fn small_config() -> LightConfig {
        LightConfig { source_grid: [32, 24], ..LightConfig::default() }
    }

    fn normalized_point(lens: Lens, u: f64, v: f64) -> Point {
        let x = lens.axes[0] * u;
        let y = lens.axes[1] * (v + lens.bend * u * u);
        [
            lens.center[0] + lens.cosine * x - lens.sine * y,
            lens.center[1] + lens.sine * x + lens.cosine * y,
        ]
    }

    #[test]
    fn default_gain_exponent_preserves_the_archived_light_config_hash() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/atelier-v12-light-manifest.json"
        ))
        .unwrap();
        let legacy = manifest["config"]["light"].clone();
        let mut config: LightConfig = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(config.gain_exponent, 1.0);
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value, legacy);
        assert_eq!(
            hex::encode(Sha256::digest(serde_json::to_vec(&value).unwrap())),
            "1078da5bfa50a29945b152d9f80237348418ea62b1b2c24d2a7d0e7d7df85b0e"
        );
        config.gain_exponent = 2.0;
        let value = serde_json::to_value(config).unwrap();
        assert_eq!(value["gain_exponent"], 2.0);
        assert_eq!(serde_json::from_value::<LightConfig>(value).unwrap().gain_exponent, 2.0);
    }

    #[test]
    fn exponent_one_keeps_the_frozen_dark_signal_bits() {
        let config = LightConfig { gain_scale: 0.0012, ..LightConfig::default() };
        // Captured before adding the contrast exponent. Tiny positive deltas
        // also detect replacing the legacy subtraction with direct delta/E0.
        for (delta, bits) in [
            (-0.55, 0),
            (-0.3, 0),
            (0.0, 0),
            (f64::EPSILON / 16.0, 0),
            (f64::EPSILON, 4333493265125159521),
            (0.1, 4552181512339903715),
            (0.55, 4563176846121054817),
            (1.1, 4567680445748425313),
            (2.2, 4572184045375795809),
            (10.0, 4581945884160824918),
        ] {
            assert_eq!(
                dark_gain_signal(delta, config.incident_irradiance + delta, &config)
                    .unwrap()
                    .to_bits(),
                bits
            );
        }
    }

    #[test]
    fn artistic_contrast_is_nonnegative_and_selectively_emphasizes_focusing() {
        let mut config = LightConfig { gain_scale: 0.0012, ..LightConfig::default() };
        for exponent in [1.0, 1.5, 2.0, 3.0] {
            config.gain_exponent = exponent;
            let signal = |ratio| {
                let delta = ratio * config.incident_irradiance;
                dark_gain_signal(delta, config.incident_irradiance + delta, &config).unwrap()
            };
            let levels = [-1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 100.0].map(signal);
            assert!(levels.iter().all(|value| value.is_finite() && *value >= 0.0));
            assert_eq!(&levels[..3], &[0.0; 3]);
            assert!(levels.windows(2).all(|pair| pair[0] <= pair[1]));
            assert_eq!(levels[4], config.gain_scale);
            if exponent > 1.0 {
                assert!(levels[3] < 0.25 * config.gain_scale);
                assert!(levels[5] > 4.0 * config.gain_scale);
            }
        }
    }

    #[test]
    fn contrast_changes_only_the_dark_view_and_preserves_physical_diagnostics() {
        let source = source();
        let mut config = small_config();
        config.display = LightDisplay::DarkGain;
        config.gain_scale = 0.0012;
        let render = RenderConfig {
            width: 32,
            height: 18,
            aa: 1,
            background: V3::ZERO,
            ..RenderConfig::default()
        };
        let linear = render_linear(&source, 0.4, &config, &camera(), &render).unwrap();
        config.gain_exponent = 2.0;
        let contrast = render_linear(&source, 0.4, &config, &camera(), &render).unwrap();
        assert!(
            contrast
                .pixels
                .iter()
                .all(|value| value.is_finite() && value.min(V3::ZERO) == V3::ZERO)
        );
        assert_ne!(linear.pixels, contrast.pixels);
        let physical = |diagnostics: LightDiagnostics| {
            let mut value = serde_json::to_value(diagnostics).unwrap();
            value.as_object_mut().unwrap().remove("gamut_compressed_pixels");
            value
        };
        assert_eq!(physical(linear.diagnostics), physical(contrast.diagnostics));

        config.display = LightDisplay::Ivory;
        let ivory = render_linear(&source, 0.4, &config, &camera(), &render).unwrap();
        config.gain_exponent = 1.0;
        let legacy = render_linear(&source, 0.4, &config, &camera(), &render).unwrap();
        assert_eq!(ivory.pixels, legacy.pixels);
        assert_eq!(
            serde_json::to_value(ivory.diagnostics).unwrap(),
            serde_json::to_value(legacy.diagnostics).unwrap()
        );
    }

    #[test]
    fn invalid_contrast_and_color_overflow_are_rejected() {
        let render = RenderConfig { width: 16, height: 12, ..RenderConfig::default() };
        let mut config = small_config();
        for invalid in [0.0, 0.999_999, 3.000_001, f64::INFINITY, f64::NAN] {
            config.gain_exponent = invalid;
            assert!(validate(0.4, &config, &render).is_err());
        }
        for valid in [1.0, 3.0] {
            config.gain_exponent = valid;
            validate(0.4, &config, &render).unwrap();
        }
        assert!(dark_gain_signal(1e200, 1e200, &config).is_err());
        assert!(nonnegative_rgb(V3::new(f64::INFINITY, 1.0, 0.0)).is_err());
        assert!(nonnegative_rgb(V3::new(f64::MAX, 0.0, 0.0)).is_err());
        assert!(
            nonnegative_rgb(V3::new(0.1 * f64::MAX, 0.46 * f64::MAX, 0.84 * f64::MAX)).is_err()
        );
    }

    #[test]
    fn zero_bend_preserves_the_legacy_lens_serialization_and_hash() {
        let legacy = r#"{"enabled":true,"semi_axes":[1.0,0.72],"angle_degrees":-20.0,"power":1.55,"coma":0.08}"#;
        let mut config: LightLensConfig = serde_json::from_str(legacy).unwrap();
        assert_eq!(config.bend, 0.0);
        for bend in [0.0, -0.0] {
            config.bend = bend;
            let bytes = serde_json::to_vec(&config).unwrap();
            assert_eq!(bytes, legacy.as_bytes());
            assert_eq!(
                hex::encode(Sha256::digest(&bytes)),
                "afb5ee51abd6f17a9ae58786d2ce86acd4baa58dccb71f1921bf406e7e1ca361"
            );
        }
        config.bend = 0.35;
        let bent = serde_json::to_value(config).unwrap();
        assert_eq!(bent["bend"], 0.35);
        assert_eq!(serde_json::from_value::<LightLensConfig>(bent).unwrap().bend, 0.35);
    }

    #[test]
    fn zero_bend_preserves_frozen_phase_bits() {
        // Captured from the straight-lens implementation before bend existed.
        let mut lens = Lens {
            center: [0.25, -0.4],
            axes: [1.7, 0.6],
            cosine: 0.8,
            sine: 0.6,
            height: 0.073,
            power: 2.3,
            coma: -0.13,
            bend: 0.0,
            extent: [2.0; 2],
            quadratic: [1.0; 3],
        };
        let expected = [
            (
                [0.25, -0.4],
                [
                    4589924625027933667,
                    13795170870810208859,
                    13793241677083713417,
                    13827972350312728226,
                    4604315158204037619,
                    13831054218425899407,
                ],
            ),
            (
                [0.45, -0.21],
                [
                    4589245850001308385,
                    13803573055296397885,
                    13813081659926512676,
                    13827281288265543868,
                    4603808262838112269,
                    13829987298318283113,
                ],
            ),
            (
                [-0.35, -0.35],
                [
                    4571486443743430904,
                    4588007352191398160,
                    13811484089393568107,
                    4601668832069958852,
                    13825261534667532651,
                    4601128637061001350,
                ],
            ),
            ([2.0, 2.0], [0; 6]),
        ];
        for bend in [0.0, -0.0] {
            lens.bend = bend;
            for (point, bits) in expected {
                let value = lens.phase(point);
                assert_eq!(
                    [
                        value.height,
                        value.gradient[0],
                        value.gradient[1],
                        value.hessian[0],
                        value.hessian[1],
                        value.hessian[2]
                    ]
                    .map(f64::to_bits),
                    bits
                );
            }
            assert_eq!(lens.gradient_bound().to_bits(), 4598488145953252500);
        }
    }

    #[test]
    fn curved_phase_gradient_and_hessian_match_finite_differences() {
        let source = source();
        let mut config = small_config();
        for bend in [-0.5, -0.25, 0.35, 0.5] {
            config.lenses[1].bend = bend;
            for angle in [-50.0, 35.0, 103.0] {
                config.lenses[1].angle_degrees = angle;
                let lens = lenses(&source, 0.4, &config).unwrap()[1];
                for [u, v] in [[0.31, -0.27], [-0.47, 0.38], [0.67, 0.16]] {
                    let point = normalized_point(lens, u, v);
                    let actual = lens.phase(point);
                    let h = 1e-5;
                    for axis in 0..2 {
                        let mut left = point;
                        let mut right = point;
                        left[axis] -= h;
                        right[axis] += h;
                        let a = lens.phase(left);
                        let b = lens.phase(right);
                        assert!(
                            ((b.height - a.height) / (2.0 * h) - actual.gradient[axis]).abs()
                                < 1e-8
                        );
                        for component in 0..2 {
                            assert!(
                                ((b.gradient[component] - a.gradient[component]) / (2.0 * h)
                                    - actual.hessian[axis + component])
                                    .abs()
                                    < 1e-7
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn curved_support_candidates_and_gradient_bound_cover_rotated_edges() {
        let source = source();
        let mut config = small_config();
        config.lenses[0].semi_axes = [1.7, 0.6];
        config.lenses[0].coma = 0.15;
        let mut beyond_straight_bounds = 0;
        for bend in [-0.5, -0.35, 0.35, 0.5] {
            config.lenses[0].bend = bend;
            for angle in [0.0, 37.0, 90.0, 137.0] {
                config.lenses[0].angle_degrees = angle;
                let lens = lenses(&source, 0.4, &config).unwrap()[0];
                let straight = support_extent(lens.axes, lens.cosine, lens.sine, 0.0);
                for radius in [0.25, 1.0 / 7.0_f64.sqrt(), 0.7, 0.999_999] {
                    for index in 0..128 {
                        let theta = std::f64::consts::TAU * f64::from(index) / 128.0;
                        let point =
                            normalized_point(lens, radius * theta.cos(), radius * theta.sin());
                        let value = lens.phase(point);
                        assert!(value.height > 0.0);
                        assert!(
                            value.gradient[0].hypot(value.gradient[1]) <= lens.gradient_bound()
                        );
                        for (axis, extent) in straight.into_iter().enumerate() {
                            let distance = (point[axis] - lens.center[axis]).abs();
                            assert!(distance <= lens.extent[axis]);
                            beyond_straight_bounds += usize::from(distance > extent);
                        }
                        let low = point.map(|v| v - 1e-7);
                        let high = point.map(|v| v + 1e-7);
                        assert!(lens.intersects(low, high));
                    }
                }
                assert!(!lens.intersects([100.0, 100.0], [101.0, 101.0]));
            }
        }
        assert!(beyond_straight_bounds > 0, "fixture must expose the old ellipse-culling error");
    }

    #[test]
    fn curved_support_boundary_keeps_the_smooth_cutoff() {
        let source = source();
        let mut config = small_config();
        for bend in [-0.5, 0.35, 0.5] {
            config.lenses[1].bend = bend;
            let lens = lenses(&source, 0.4, &config).unwrap()[1];
            for angle in [0.7_f64, 1.9, 3.8] {
                let at = |radius| {
                    lens.phase(normalized_point(lens, radius * angle.cos(), radius * angle.sin()))
                };
                let coarse = at(1.0 - 1e-4);
                let fine = at(1.0 - 1e-5);
                let outside = at(1.0 + 1e-5);
                assert_eq!(outside.height, 0.0);
                assert_eq!(outside.gradient, [0.0; 2]);
                assert_eq!(outside.hessian, [0.0; 3]);
                assert!(fine.height < coarse.height * 0.000_2);
                assert!(
                    fine.gradient[0].hypot(fine.gradient[1])
                        < coarse.gradient[0].hypot(coarse.gradient[1]) * 0.002
                );
                let norm = |values: [f64; 3]| values.into_iter().map(f64::abs).fold(0.0, f64::max);
                assert!(norm(fine.hessian) < norm(coarse.hessian) * 0.02);
            }
        }
    }

    #[test]
    fn bend_validation_and_source_domain_include_the_full_curved_support() {
        let source = source();
        let mut config = small_config();
        let render = RenderConfig { width: 16, height: 12, ..RenderConfig::default() };
        config.source_motion = [0.0; 2];
        config.lenses[1].enabled = false;
        config.lenses[2].enabled = false;
        config.lenses[0].semi_axes = [1.7, 0.6];
        config.lenses[0].angle_degrees = 45.0;
        config.lenses[0].bend = 0.5;
        let lens = lenses(&source, 0.4, &config).unwrap()[0];
        let straight = support_extent(lens.axes, lens.cosine, lens.sine, 0.0);
        config.source_domain = straight.map(|extent| 2.0 * extent);
        assert!(validate(0.4, &config, &render).is_err());
        config.source_domain = lens.extent.map(|extent| 2.0 * extent + 1e-6);
        validate(0.4, &config, &render).unwrap();
        for invalid in [-0.500_001, 0.500_001, f64::INFINITY, f64::NAN] {
            config.lenses[0].bend = invalid;
            assert!(validate(0.4, &config, &render).is_err());
        }
    }

    #[test]
    fn twelve_spectral_groups_cover_every_bin_and_preserve_existing_white() {
        let bands = spectral_bands();
        assert_eq!(bands[0].first, 0);
        assert_eq!(bands[BAND_COUNT - 1].end, NUM_BINS);
        for pair in bands.windows(2) {
            assert_eq!(pair[0].end, pair[1].first);
        }
        let grouped = bands.iter().fold(V3::ZERO, |sum, band| sum + band.weight);
        let original =
            BIN_XYZ_LUT.iter().fold(V3::ZERO, |sum, &(x, y, z, _)| sum + V3::new(x, y, z));
        assert!((grouped - original).length() < 1e-14);
        assert!((grouped - V3::new(0.95047, 1.0, 1.08883)).length() < 1e-12);
        for band in bands {
            assert!(band.wavelength >= wavelength_nm_for_bin(band.first));
            assert!(band.wavelength <= wavelength_nm_for_bin(band.end - 1));
        }
    }

    #[test]
    fn phase_gradient_and_hessian_match_finite_differences() {
        let source = source();
        let config = small_config();
        let lens = lenses(&source, 0.4, &config).unwrap()[1];
        let point = [lens.center[0] + 0.21, lens.center[1] - 0.13];
        let actual = lens.phase(point);
        let h = 1e-5;
        for axis in 0..2 {
            let mut left = point;
            let mut right = point;
            left[axis] -= h;
            right[axis] += h;
            let a = lens.phase(left);
            let b = lens.phase(right);
            assert!(((b.height - a.height) / (2.0 * h) - actual.gradient[axis]).abs() < 1e-8);
            for component in 0..2 {
                let expected = actual.hessian[axis + component];
                assert!(
                    ((b.gradient[component] - a.gradient[component]) / (2.0 * h) - expected).abs()
                        < 1e-7
                );
            }
        }
        let outside = lens.phase([lens.center[0] + 3.0, lens.center[1]]);
        assert_eq!(outside.height, 0.0);
        assert_eq!(outside.gradient, [0.0; 2]);
        assert_eq!(outside.hessian, [0.0; 3]);
    }

    #[test]
    fn positive_boss_focuses_and_compact_support_is_smooth() {
        let source = source();
        let mut config = small_config();
        config.lenses[0].coma = 0.0;
        config.lenses[0].angle_degrees = 0.0;
        let lens = lenses(&source, 0.4, &config).unwrap()[0];
        let center = lens.phase(lens.center);
        assert_eq!(center.gradient, [0.0; 2]);
        assert!(center.hessian[0] < 0.0 && center.hessian[2] < 0.0);
        let near = lens.phase([lens.center[0] + lens.axes[0] * (1.0 - 1e-6), lens.center[1]]);
        assert!(near.gradient[0].abs() < 1e-14);
        assert!(near.hessian[0].abs() < 1e-9);
    }

    #[test]
    fn fixed_receiver_uses_crop_scale_and_roll_and_rejects_tilt() {
        let render = RenderConfig { width: 80, height: 60, ..RenderConfig::default() };
        let frame = Receiver::new(&camera(), &render).unwrap();
        assert_eq!(frame.project([0.0, 0.0]), [40.0, 30.0]);
        let mut moved = camera();
        moved.position.x = 1.0;
        moved.target.x = 1.0;
        assert_eq!(Receiver::new(&moved, &render).unwrap().project([1.0, 0.0]), [40.0, 30.0]);
        moved.up = V3::new(1.0, 0.0, 0.0);
        assert_eq!(Receiver::new(&moved, &render).unwrap().project([1.0, 1.0]), [30.0, 30.0]);
        moved.position.x += 0.01;
        assert!(Receiver::new(&moved, &render).is_err());
    }

    #[test]
    fn support_union_is_covered_once_and_lens_centers_follow_current_source() {
        let source = source();
        let config = small_config();
        let a = lenses(&source, 0.0, &config).unwrap();
        let b = lenses(&source, 1.0, &config).unwrap();
        for (time, current) in [(0.0, a), (1.0, b)] {
            let frame = source.sample(time).unwrap();
            for (body, lens) in current.iter().enumerate() {
                assert_eq!(
                    lens.center[0],
                    config.source_motion[0]
                        * (frame.bodies[body].position.dot(config.source_axes[0])
                            / config.source_scales[0])
                            .tanh()
                );
            }
        }
        let render = RenderConfig { width: 80, height: 60, ..RenderConfig::default() };
        let receiver = Receiver::new(&camera(), &render).unwrap();
        let once = prepare_grid(&[a[0], a[0], a[0]], &config, &receiver, 4.0).unwrap();
        let mut absent = a[0];
        absent.height = 0.0;
        let single = prepare_grid(&[a[0], absent, absent], &config, &receiver, 4.0).unwrap();
        assert_eq!(once.cells, single.cells);
        assert!(once.cells.len() < config.source_grid[0] * config.source_grid[1]);
        assert!(!a[0].intersects([100.0, 100.0], [101.0, 101.0]));
    }

    #[test]
    fn identity_light_is_the_existing_white_and_dark_gain_is_zero() {
        let source = source();
        let mut config = small_config();
        for lens in &mut config.lenses {
            lens.enabled = false;
        }
        let render = RenderConfig {
            width: 12,
            height: 8,
            aa: 1,
            background: V3::ZERO,
            ..RenderConfig::default()
        };
        let first = render_linear(&source, 0.0, &config, &camera(), &render).unwrap();
        assert_eq!(first.diagnostics.active_cells, 0);
        assert!(first.pixels.windows(2).all(|p| p[0] == p[1]));
        let expected =
            config.receiver_tint * (config.incident_irradiance + config.ambient_irradiance);
        assert!((first.pixels[0] - expected).length() < 1e-6);
        config.display = LightDisplay::DarkGain;
        let dark = render_linear(&source, 1.0, &config, &camera(), &render).unwrap();
        assert!(dark.pixels.iter().all(|&p| p == V3::ZERO));
    }

    #[test]
    fn complete_optical_records_allow_signed_crop_flux_and_reject_corruption() {
        let source = source();
        let config = small_config();
        let render = RenderConfig {
            width: 32,
            height: 18,
            aa: 1,
            background: V3::ZERO,
            ..RenderConfig::default()
        };
        let frame = render_linear(&source, 0.4, &config, &camera(), &render).unwrap();
        assert!(frame.diagnostics.active_cells > 0);
        frame.diagnostics.validate().unwrap();
        let mut record = frame.diagnostics.clone();
        for band in &mut record.bands {
            band.received_flux = 0.3 * band.incident_flux;
            band.escaped_flux = 0.7 * band.incident_flux;
            band.baseline_removed_flux = 0.8 * band.incident_flux;
            band.reference_escaped_flux = 0.2 * band.incident_flux;
            band.net_crop_flux = band.received_flux - band.baseline_removed_flux;
            band.conservation_residual = 0.0;
        }
        record.received_luminance_flux = 0.3 * record.incident_luminance_flux;
        record.escaped_luminance_flux = 0.7 * record.incident_luminance_flux;
        record.validate().unwrap();
        let mut broken = record.clone();
        broken.bands.pop();
        assert!(broken.validate().is_err());
        let mut broken = record.clone();
        broken.bands[0].escaped_flux += 1.0;
        assert!(broken.validate().is_err());
        let mut broken = record.clone();
        broken.source_fraction = f64::NAN;
        assert!(broken.validate().is_err());
        let mut broken = record;
        broken.bands[0].net_crop_flux = 0.0;
        assert!(broken.validate().is_err());
    }
}
