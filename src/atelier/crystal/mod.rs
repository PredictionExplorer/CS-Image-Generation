//! Polarized Crystal: shared elastic stress revealed by spectral polarization.
//!
//! A clamped, uniform plane-stress sheet supplies the mechanical field. The
//! polished silhouette is an optical viewing window inside that larger sheet,
//! not a traction-free mechanical boundary. Its shallow thickness and studio
//! reflections are designed presentation. The three-dimensional recorded orbit
//! remains unchanged; a declared fixed projection maps its balanced pair loads
//! into this two-dimensional material. Short memory filters the forcing rather
//! than extrapolating paths or accumulating frame-dependent state.
pub mod field;
pub mod optics;
pub mod presentation;
pub mod surface;

use super::{Camera, OrbitSeries, RenderConfig, SilkResult, V3};
use field::{ElasticSheet, FieldConfig, FieldDiagnostics, StressField};
use optics::{OpticsConfig, PolarizedOptics};
use presentation::SurfaceRaster;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use surface::{Surface, SurfaceConfig};

/// Complete material and presentation recipe for a polarized crystal study.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CrystalConfig {
    /// Balanced loads, linear elastic material, and finite temporal memory.
    pub field: FieldConfig,
    /// Spectral retardation and the fixed crossed-polarizer orientation.
    pub optics: OpticsConfig,
    /// Uniform baseline stress `[xx, yy, xy]` held by the remote supports.
    /// The elastic solver supplies increments about this prestretched reference.
    pub uniform_pre_stress: [f64; 3],
    /// Horizontal and vertical semi-axes of the visible optical window.
    pub semi_axes: [f64; 2],
    /// Fixed in-plane rotation of the polished window; loads remain in world coordinates.
    pub rotation_degrees: f64,
    /// Small, smooth asymmetry of the window radius, in zero to 0.15.
    pub outline_asymmetry: f64,
    /// Height of the convex polished front surface in world units.
    pub surface_depth: f64,
    /// Index for dielectric reflection and the local optical path correction.
    pub refractive_index: f64,
    /// Optical thickness at the center, relative to the material calibration.
    pub center_thickness: f64,
    /// Fraction of center thickness retained near the rim, in zero to one.
    pub edge_thickness: f64,
    /// Fractional radius over which internal illumination fades near the rim.
    pub interior_feather: f64,
    /// Per-channel absorption through one unit of optical thickness.
    pub absorption: V3,
    /// Restrained neutral surface reflection, independent of the internal stress.
    pub surface_strength: f64,
    /// Reflection along the polished edge, independent of the internal stress.
    pub edge_strength: f64,
    /// Optional held mechanical source time for a separate optical examination.
    pub freeze_source_fraction: Option<f64>,
    /// Rotation of both crossed polarizers during the examination; zero during motion.
    pub polarizer_sweep_degrees: f64,
    /// Reuse fixed full-precision surface samples when they fit the cache budget.
    /// Larger stills automatically use the bounded-memory streaming path.
    pub cache_presentation: bool,
}

impl Default for CrystalConfig {
    fn default() -> Self {
        Self {
            field: FieldConfig::default(),
            optics: OpticsConfig::default(),
            uniform_pre_stress: [0.0; 3],
            semi_axes: [3.55, 2.15],
            rotation_degrees: -12.0,
            outline_asymmetry: 0.055,
            surface_depth: 0.65,
            refractive_index: 1.5,
            center_thickness: 1.0,
            edge_thickness: 0.65,
            interior_feather: 0.08,
            absorption: V3::new(0.10, 0.07, 0.05),
            surface_strength: 1.0,
            edge_strength: 0.14,
            freeze_source_fraction: None,
            polarizer_sweep_degrees: 0.0,
            cache_presentation: true,
        }
    }
}

/// Mechanical and optical evidence saved with every rendered shutter sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrystalDiagnostics {
    /// Requested time on the film or examination timeline.
    pub timeline_fraction: f64,
    /// Actual source time driving the mechanical state.
    pub source_fraction: f64,
    /// Rotation of the crossed polarizer pair beyond its recipe orientation.
    pub polarizer_offset_degrees: f64,
    /// Solver convergence, applied forces, and stress ranges.
    pub field: FieldDiagnostics,
    /// Number of pixels whose footprint intersects the optical window.
    pub material_pixels: usize,
    /// Largest finite linear-light channel value before finishing.
    pub maximum_radiance: f64,
}

impl CrystalDiagnostics {
    /// Reject invalid temporal state, numerical mechanics, or optical output.
    pub fn validate(&self) -> SilkResult<()> {
        if !self.timeline_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.timeline_fraction)
            || !self.source_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.source_fraction)
            || !self.polarizer_offset_degrees.is_finite()
            || !self.maximum_radiance.is_finite()
            || self.maximum_radiance < 0.0
            || self.material_pixels == 0
        {
            return Err("invalid crystal diagnostics".into());
        }
        self.field.validate()
    }
}

/// Untone-mapped frame and evidence for the exposure that generated it.
pub struct CrystalFrame {
    /// Row-major scene-linear RGB radiance.
    pub pixels: Vec<V3>,
    /// Source, mechanical, and optical diagnostics for this exposure.
    pub diagnostics: CrystalDiagnostics,
}

/// Immutable mechanics and spectral tables reused for all independent frames.
pub struct PreparedCrystal<'a> {
    source: &'a OrbitSeries,
    sheet: ElasticSheet,
    optics: PolarizedOptics,
    recipe: Vec<u8>,
    frozen_field: Option<StressField>,
    surface: Surface,
    presentation: Mutex<Option<CachedPresentation>>,
}

struct CachedPresentation {
    key: Vec<u8>,
    raster: Arc<SurfaceRaster>,
}

fn surface_config(config: &CrystalConfig) -> SurfaceConfig {
    SurfaceConfig {
        semi_axes: config.semi_axes,
        rotation_degrees: config.rotation_degrees,
        outline_asymmetry: config.outline_asymmetry,
        depth: config.surface_depth,
        refractive_index: config.refractive_index,
        reflection_strength: 1.0,
    }
}

fn valid_material(config: &CrystalConfig) -> SilkResult<()> {
    config.field.validate()?;
    config.optics.validate()?;
    surface_config(config).validate()?;
    if config.semi_axes.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || config.uniform_pre_stress.iter().any(|v| !v.is_finite() || v.abs() > 100.0)
        || !config.rotation_degrees.is_finite()
        || !config.outline_asymmetry.is_finite()
        || !(0.0..=0.15).contains(&config.outline_asymmetry)
        || !config.center_thickness.is_finite()
        || !(0.001..=100.0).contains(&config.center_thickness)
        || !config.edge_thickness.is_finite()
        || !(0.0..=1.0).contains(&config.edge_thickness)
        || !config.interior_feather.is_finite()
        || !(0.001..=0.5).contains(&config.interior_feather)
        || !config.absorption.is_finite()
        || config.absorption.x < 0.0
        || config.absorption.y < 0.0
        || config.absorption.z < 0.0
        || [config.surface_strength, config.edge_strength]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=10.0).contains(v))
        || !config.polarizer_sweep_degrees.is_finite()
        || config.polarizer_sweep_degrees.abs() > 360.0
        || config
            .freeze_source_fraction
            .is_some_and(|t| !t.is_finite() || !(0.0..=1.0).contains(&t))
        || (config.polarizer_sweep_degrees != 0.0 && config.freeze_source_fraction.is_none())
    {
        return Err("invalid crystal material or optical examination settings".into());
    }
    let angle = config.rotation_degrees.to_radians();
    let extent = [
        (config.semi_axes[0] * angle.cos()).hypot(config.semi_axes[1] * angle.sin()),
        (config.semi_axes[0] * angle.sin()).hypot(config.semi_axes[1] * angle.cos()),
    ];
    if extent
        .iter()
        .zip(config.field.domain_half_size)
        .any(|(e, bound)| e * (1.0 + 1.5 * config.outline_asymmetry) >= bound * 0.96)
    {
        return Err("crystal optical window must fit inside the remote mechanical rim".into());
    }
    Ok(())
}

/// Validate the recipe and fixed frontal camera before allocating mechanical work.
pub fn validate(config: &CrystalConfig, camera: &Camera, render: &RenderConfig) -> SilkResult<()> {
    valid_material(config)?;
    let view = camera.position - camera.target;
    if !camera.position.is_finite()
        || !camera.target.is_finite()
        || !camera.up.is_finite()
        || view.z <= 0.0
        || view.x.abs() > 1e-10
        || view.y.abs() > 1e-10
        || camera.up.x.abs() > 1e-10
        || camera.up.y <= 0.0
        || camera.up.z.abs() > 1e-10
        || !camera.orthographic_height.is_finite()
        || camera.orthographic_height <= 0.0
        || render.width == 0
        || render.height == 0
        || render.width > 16_384
        || render.height > 16_384
        || !(1..=8).contains(&render.aa)
        || !render.background.is_finite()
        || render.background.x < 0.0
        || render.background.y < 0.0
        || render.background.z < 0.0
        || !render.exposure.is_finite()
        || !render.exposure.exp2().is_finite()
        || render.exposure.exp2() == 0.0
        || !render.bloom_strength.is_finite()
        || render.bloom_strength < 0.0
    {
        return Err(
            "crystal requires finite dimensions and a frontal upright orthographic camera".into()
        );
    }
    Ok(())
}

impl<'a> PreparedCrystal<'a> {
    /// Prepare deterministic mechanics and polarization tables once per recipe.
    pub fn new(source: &'a OrbitSeries, config: &CrystalConfig) -> SilkResult<Self> {
        valid_material(config)?;
        let sheet = ElasticSheet::new(&config.field)?;
        let frozen_field =
            config.freeze_source_fraction.map(|time| sheet.solve(source, time)).transpose()?;
        Ok(Self {
            source,
            sheet,
            optics: PolarizedOptics::new(&config.optics)?,
            recipe: serde_json::to_vec(config)?,
            frozen_field,
            surface: Surface::new(&surface_config(config))?,
            presentation: Mutex::new(None),
        })
    }

    fn presentation(
        &self,
        config: &CrystalConfig,
        camera: &Camera,
        render: &RenderConfig,
    ) -> SilkResult<Option<Arc<SurfaceRaster>>> {
        let budget = usize::try_from(4_u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX);
        if !config.cache_presentation
            || presentation::required_bytes_upper_bound(config, camera, render)? > budget
        {
            return Ok(None);
        }
        let key = serde_json::to_vec(&(camera, render))?;
        let mut cache =
            self.presentation.lock().map_err(|_| "crystal presentation cache lock poisoned")?;
        if let Some(cached) = cache.as_ref()
            && cached.key == key
        {
            return Ok(Some(Arc::clone(&cached.raster)));
        }
        let raster = Arc::new(SurfaceRaster::new(config, camera, render, budget)?);
        *cache = Some(CachedPresentation { key, raster: Arc::clone(&raster) });
        Ok(Some(raster))
    }

    /// Evaluate one source time with order-independent memory and fixed exposure.
    pub fn render_linear(
        &self,
        source: &OrbitSeries,
        time: f64,
        config: &CrystalConfig,
        camera: &Camera,
        render: &RenderConfig,
    ) -> SilkResult<CrystalFrame> {
        validate(config, camera, render)?;
        if !time.is_finite() || !(0.0..=1.0).contains(&time) {
            return Err("crystal source fraction must be in zero to one".into());
        }
        if self.recipe != serde_json::to_vec(config)? {
            return Err("prepared crystal recipe differs from requested rendering recipe".into());
        }
        if !std::ptr::eq(self.source, source) {
            return Err("prepared crystal belongs to a different recorded source".into());
        }
        let source_time = config.freeze_source_fraction.unwrap_or(time);
        let transient;
        let field = if let Some(frozen) = &self.frozen_field {
            frozen
        } else {
            transient = self.sheet.solve(source, source_time)?;
            &transient
        };
        let offset = config.polarizer_sweep_degrees * smoothstep(time);
        let rotated_optics =
            if offset == 0.0 { None } else { Some(self.optics.with_angle_offset_degrees(offset)?) };
        let optics = rotated_optics.as_ref().unwrap_or(&self.optics);
        if let Some(raster) = self.presentation(config, camera, render)? {
            let frame = raster.render(field, optics, config.uniform_pre_stress)?;
            let diagnostics = CrystalDiagnostics {
                timeline_fraction: time,
                source_fraction: source_time,
                polarizer_offset_degrees: offset,
                field: field.diagnostics.clone(),
                material_pixels: frame.material_pixels,
                maximum_radiance: frame.maximum_radiance,
            };
            diagnostics.validate()?;
            return Ok(CrystalFrame { pixels: frame.pixels, diagnostics });
        }
        let pixel_size = camera.orthographic_height / f64::from(render.height);
        let aa = render.aa;
        let painter = Painter {
            field,
            optics,
            config,
            background: render.background,
            surface: &self.surface,
        };
        let mut pixels = vec![V3::ZERO; render.width as usize * render.height as usize];
        let material_pixels: usize = pixels
            .par_chunks_mut(render.width as usize)
            .enumerate()
            .map(|(row, pixels)| {
                let mut material = 0;
                for (column, pixel) in pixels.iter_mut().enumerate() {
                    let mut covered = false;
                    for sy in 0..aa {
                        let y = camera.target.y
                            + (f64::from(render.height) * 0.5
                                - row as f64
                                - (f64::from(sy) + 0.5) / f64::from(aa))
                                * pixel_size;
                        for sx in 0..aa {
                            let x = camera.target.x
                                + (column as f64 + (f64::from(sx) + 0.5) / f64::from(aa)
                                    - f64::from(render.width) * 0.5)
                                    * pixel_size;
                            let (sample, inside) = painter.sample(x, y);
                            *pixel += sample;
                            covered |= inside;
                        }
                    }
                    *pixel /= f64::from(aa * aa);
                    material += usize::from(covered);
                }
                material
            })
            .sum();
        if pixels.iter().any(|p| !p.is_finite() || p.x < 0.0 || p.y < 0.0 || p.z < 0.0) {
            return Err("crystal optical calculation produced invalid radiance".into());
        }
        if material_pixels == 0 {
            return Err("crystal optical window is outside the image".into());
        }
        let maximum_radiance = pixels.iter().map(|p| p.x.max(p.y).max(p.z)).fold(0.0, f64::max);
        let diagnostics = CrystalDiagnostics {
            timeline_fraction: time,
            source_fraction: source_time,
            polarizer_offset_degrees: offset,
            field: field.diagnostics.clone(),
            material_pixels,
            maximum_radiance,
        };
        diagnostics.validate()?;
        Ok(CrystalFrame { pixels, diagnostics })
    }
}

struct Painter<'a> {
    field: &'a StressField,
    optics: &'a PolarizedOptics,
    config: &'a CrystalConfig,
    background: V3,
    surface: &'a Surface,
}

impl Painter<'_> {
    fn sample(&self, x: f64, y: f64) -> (V3, bool) {
        let config = self.config;
        let Some(surface) = self.surface.sample(x, y) else {
            return (self.background, false);
        };
        let thickness = config.center_thickness
            * (config.edge_thickness + (1.0 - config.edge_thickness) * surface.dome)
            * surface.optical_path_scale;
        let stress = self.field.sample(surface.refracted_point[0], surface.refracted_point[1]);
        let stress = std::array::from_fn(|axis| stress[axis] + config.uniform_pre_stress[axis]);
        let internal = self.optics.sample(stress, thickness, 0.0);
        let absorption = V3::new(
            (-config.absorption.x * thickness).exp(),
            (-config.absorption.y * thickness).exp(),
            (-config.absorption.z * thickness).exp(),
        );
        let reflection = surface.reflection
            * (config.surface_strength + config.edge_strength * surface.rho.powi(8));
        let optical_edge =
            smoothstep(((1.0 - surface.rho) / config.interior_feather).clamp(0.0, 1.0));
        let transmission = (1.0 - surface.fresnel).powi(2);
        // The external studio background is outside the optical aperture.
        // Adding it inside would make an extinguished specimen emit that color.
        (internal.hadamard(absorption) * (optical_edge * transmission) + reflection, true)
    }
}

fn smoothstep(value: f64) -> f64 {
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> OrbitSeries {
        OrbitSeries::new(&crate::silk::OrbitData {
            seed: "0xc1".into(),
            dt: 0.1,
            masses: [1.0; 3],
            samples: (0..25)
                .map(|step| {
                    let t = f64::from(step) / 24.0;
                    [
                        V3::new(-1.0 + t, -0.45, 0.2),
                        V3::new(1.0 - 0.5 * t, 0.2, -0.2),
                        V3::new(0.25, 0.9 - t * 0.3, 0.0),
                    ]
                })
                .collect(),
            provenance: serde_json::Value::Null,
        })
        .unwrap()
    }

    fn small_material() -> CrystalConfig {
        let mut config = CrystalConfig::default();
        config.field.grid = [17, 13];
        config.field.load_softness = 1.0;
        config
    }

    fn camera() -> Camera {
        Camera { position: V3::new(0.0, 0.0, 12.0), orthographic_height: 7.2, ..Camera::default() }
    }

    #[test]
    fn presentation_requires_explicit_frozen_state_for_optical_examination() {
        let mut c = CrystalConfig::default();
        let r = RenderConfig { width: 32, height: 24, ..RenderConfig::default() };
        assert!(validate(&c, &camera(), &r).is_ok());
        c.polarizer_sweep_degrees = 90.0;
        assert!(validate(&c, &camera(), &r).is_err());
        c.freeze_source_fraction = Some(0.75);
        assert!(validate(&c, &camera(), &r).is_ok());
        c.freeze_source_fraction = Some(f64::NAN);
        assert!(validate(&c, &camera(), &r).is_err());
    }

    #[test]
    fn viewing_window_cannot_reach_mechanical_boundary() {
        let c = CrystalConfig { semi_axes: [30.0, 20.0], ..CrystalConfig::default() };
        assert!(validate(&c, &camera(), &RenderConfig::default()).is_err());
    }

    #[test]
    fn perspective_or_tilt_is_rejected_instead_of_silently_ignored() {
        let mut tilted = camera();
        tilted.position.x = 1.0;
        assert!(validate(&CrystalConfig::default(), &tilted, &RenderConfig::default()).is_err());
    }

    #[test]
    fn full_optical_frames_are_identical_across_worker_counts() {
        let source = source();
        let config = small_material();
        let prepared = PreparedCrystal::new(&source, &config).unwrap();
        let render = RenderConfig { width: 48, height: 32, aa: 2, ..RenderConfig::default() };
        let run = |workers| {
            rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap().install(|| {
                prepared.render_linear(&source, 0.7, &config, &camera(), &render).unwrap()
            })
        };
        let a = run(1);
        let b = run(3);
        assert_eq!(a.pixels, b.pixels);
        assert_eq!(
            serde_json::to_vec(&a.diagnostics).unwrap(),
            serde_json::to_vec(&b.diagnostics).unwrap()
        );
        assert!(a.diagnostics.material_pixels > 0);
        assert!(a.diagnostics.maximum_radiance > 0.01);
    }

    #[test]
    fn frozen_examination_changes_optics_and_preserves_mechanics() {
        let source = source();
        let mut config = small_material();
        config.freeze_source_fraction = Some(0.7);
        config.polarizer_sweep_degrees = 45.0;
        let prepared = PreparedCrystal::new(&source, &config).unwrap();
        let render = RenderConfig { width: 32, height: 24, aa: 1, ..RenderConfig::default() };
        let a = prepared.render_linear(&source, 0.0, &config, &camera(), &render).unwrap();
        let b = prepared.render_linear(&source, 1.0, &config, &camera(), &render).unwrap();
        assert_ne!(a.pixels, b.pixels);
        assert_eq!(a.diagnostics.source_fraction, b.diagnostics.source_fraction);
        assert_eq!(
            serde_json::to_vec(&a.diagnostics.field).unwrap(),
            serde_json::to_vec(&b.diagnostics.field).unwrap()
        );
        assert_eq!(a.diagnostics.polarizer_offset_degrees, 0.0);
        assert_eq!(b.diagnostics.polarizer_offset_degrees, 45.0);
        let another_source = self::source();
        assert!(prepared.render_linear(&another_source, 0.0, &config, &camera(), &render).is_err());
        config.surface_strength *= 2.0;
        assert!(prepared.render_linear(&source, 0.0, &config, &camera(), &render).is_err());
    }
}
