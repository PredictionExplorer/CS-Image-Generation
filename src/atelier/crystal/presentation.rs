//! Reusable double-precision presentation samples for a fixed crystal and camera.
//!
//! Only covered subpixels are cached. Offsets preserve pixel order, and each
//! pixel's samples preserve the direct renderer's SY/SX order. A bounded batch
//! of rows is prepared in parallel and appended in row order. The cache changes
//! no geometry, sampling count, material law or spectral calculation. Combining
//! static transmission factors and collecting uncovered background samples can
//! change floating-point rounding; agreement with the direct path is numerical,
//! not an assertion of identical PNG hashes across the two implementations.

use super::{CrystalConfig, field::StressField, optics::PolarizedOptics, surface::Surface};
use crate::atelier::{Camera, RenderConfig, SilkResult, V3};
use rayon::prelude::*;
use std::mem::size_of;

const ROW_BATCH: usize = 16;

/// One covered subpixel; all values retain the direct renderer's f64 precision.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CachedSample {
    point: [f64; 2],
    thickness: f64,
    throughput: V3,
    reflection: V3,
}

struct Row {
    counts: Vec<u8>,
    samples: Vec<CachedSample>,
}

struct Layout {
    width: usize,
    height: usize,
    pixels: usize,
    samples_per_pixel: u32,
    columns: [usize; 2],
    rows: [usize; 2],
    maximum_samples: usize,
    required_bytes: usize,
}

fn product(a: usize, b: usize) -> SilkResult<usize> {
    a.checked_mul(b).ok_or_else(|| "crystal presentation allocation size overflow".into())
}

fn total(a: usize, b: usize) -> SilkResult<usize> {
    a.checked_add(b).ok_or_else(|| "crystal presentation allocation size overflow".into())
}

fn pixel_interval(low: f64, high: f64, length: usize) -> [usize; 2] {
    [low.floor().clamp(0.0, length as f64) as usize, high.ceil().clamp(0.0, length as f64) as usize]
}

fn layout(config: &CrystalConfig, camera: &Camera, render: &RenderConfig) -> SilkResult<Layout> {
    super::validate(config, camera, render)?;
    let width = render.width as usize;
    let height = render.height as usize;
    let pixels = product(width, height)?;
    let samples_per_pixel = render.aa.checked_mul(render.aa).ok_or("crystal AA count overflow")?;
    let angle = config.rotation_degrees.to_radians();
    let support = 1.0 + 1.5 * config.outline_asymmetry;
    let extent = [
        (config.semi_axes[0] * angle.cos()).hypot(config.semi_axes[1] * angle.sin()) * support,
        (config.semi_axes[0] * angle.sin()).hypot(config.semi_axes[1] * angle.cos()) * support,
    ];
    let pixel_size = camera.orthographic_height / f64::from(render.height);
    // Expand the analytic bound by roundoff before mapping it to whole pixels.
    let padding = 64.0 * f64::EPSILON * extent[0].max(extent[1]).max(1.0);
    let columns = pixel_interval(
        (-extent[0] - padding - camera.target.x) / pixel_size + width as f64 * 0.5,
        (extent[0] + padding - camera.target.x) / pixel_size + width as f64 * 0.5,
        width,
    );
    let rows = pixel_interval(
        (camera.target.y - extent[1] - padding) / pixel_size + height as f64 * 0.5,
        (camera.target.y + extent[1] + padding) / pixel_size + height as f64 * 0.5,
        height,
    );
    let covered_columns = columns[1] - columns[0];
    let covered_rows = rows[1] - rows[0];
    let maximum_samples =
        product(product(covered_columns, covered_rows)?, samples_per_pixel as usize)?;
    u32::try_from(maximum_samples)
        .map_err(|_| "crystal presentation offsets exceed u32 capacity")?;
    let offset_bytes = product(total(pixels, 1)?, size_of::<u32>())?;
    let sample_bytes = product(maximum_samples, size_of::<CachedSample>())?;
    let temporary_rows = ROW_BATCH.min(height);
    let row_samples = product(covered_columns, samples_per_pixel as usize)?;
    let temporary_bytes = product(
        temporary_rows,
        total(total(product(row_samples, size_of::<CachedSample>())?, width)?, size_of::<Row>())?,
    )?;
    // Include one output frame as well as peak row-construction storage.
    let required_bytes = total(
        size_of::<SurfaceRaster>(),
        total(
            offset_bytes,
            total(sample_bytes, total(temporary_bytes, product(pixels, size_of::<V3>())?)?)?,
        )?,
    )?;
    Ok(Layout {
        width,
        height,
        pixels,
        samples_per_pixel,
        columns,
        rows,
        maximum_samples,
        required_bytes,
    })
}

/// Conservative cache, bounded construction workspace and one-frame memory bound.
///
/// A projected enclosing rectangle bounds covered samples. This rejects invalid
/// dimensions, arithmetic overflow and more than `u32::MAX` cached samples without
/// allocating an image. Allocator bookkeeping and unrelated process state are
/// outside this bound. Callers may choose their streaming path when it is too large.
pub fn required_bytes_upper_bound(
    config: &CrystalConfig,
    camera: &Camera,
    render: &RenderConfig,
) -> SilkResult<usize> {
    Ok(layout(config, camera, render)?.required_bytes)
}

/// Cached fixed surface samples and compact offsets into each pixel's coverage.
pub struct SurfaceRaster {
    offsets: Vec<u32>,
    samples: Vec<CachedSample>,
    samples_per_pixel: u32,
    background: V3,
    material_pixels: usize,
}

/// Scene-linear output with the same coverage and radiance evidence as streaming.
pub struct RasterFrame {
    /// Complete row-major image including the outside studio background.
    pub pixels: Vec<V3>,
    /// Number of pixels with at least one covered subpixel.
    pub material_pixels: usize,
    /// Maximum finite RGB channel before finishing.
    pub maximum_radiance: f64,
}

impl SurfaceRaster {
    /// Cache fixed presentation, rejecting an excessive bound before allocation.
    pub fn new(
        config: &CrystalConfig,
        camera: &Camera,
        render: &RenderConfig,
        max_bytes: usize,
    ) -> SilkResult<Self> {
        let layout = layout(config, camera, render)?;
        if layout.required_bytes > max_bytes {
            return Err(format!(
                "crystal presentation cache requires at most {} bytes, exceeding the {} byte budget; use streaming presentation",
                layout.required_bytes, max_bytes
            ).into());
        }
        if layout.maximum_samples == 0 {
            return Err("crystal optical window is outside the image".into());
        }
        let surface = Surface::new(&super::surface_config(config))?;
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(total(layout.pixels, 1)?)?;
        offsets.push(0);
        let mut samples = Vec::new();
        samples.try_reserve_exact(layout.maximum_samples)?;
        let mut material_pixels = 0;
        for first in (0..layout.height).step_by(ROW_BATCH) {
            let end = (first + ROW_BATCH).min(layout.height);
            let batch: Result<Vec<Row>, String> = (first..end)
                .into_par_iter()
                .map(|row| prepare_row(row, &layout, &surface, config, camera, render))
                .collect();
            for mut row in
                batch.map_err(|message| format!("crystal presentation row: {message}"))?
            {
                for count in row.counts {
                    material_pixels += usize::from(count > 0);
                    let previous: u32 = *offsets.last().ok_or("missing crystal offset origin")?;
                    offsets.push(
                        previous
                            .checked_add(u32::from(count))
                            .ok_or("crystal presentation offset overflow")?,
                    );
                }
                if row.samples.len() > layout.maximum_samples - samples.len() {
                    return Err(
                        "crystal presentation exceeded its conservative coverage bound".into()
                    );
                }
                samples.append(&mut row.samples);
            }
        }
        if material_pixels == 0 {
            return Err("crystal optical window is outside the image".into());
        }
        if offsets.len() != layout.pixels + 1
            || offsets.last().copied() != Some(samples.len() as u32)
        {
            return Err("inconsistent crystal presentation sample offsets".into());
        }
        Ok(Self {
            offsets,
            samples,
            samples_per_pixel: layout.samples_per_pixel,
            background: render.background,
            material_pixels,
        })
    }

    /// Evaluate only the changing stress and polarization against cached geometry.
    pub fn render(
        &self,
        field: &StressField,
        optics: &PolarizedOptics,
        pre_stress: [f64; 3],
    ) -> SilkResult<RasterFrame> {
        if pre_stress.iter().any(|value| !value.is_finite() || value.abs() > 100.0) {
            return Err("invalid crystal cached-presentation prestress".into());
        }
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(self.offsets.len() - 1)?;
        pixels.resize(self.offsets.len() - 1, V3::ZERO);
        pixels.par_iter_mut().enumerate().for_each(|(index, pixel)| {
            let start = self.offsets[index] as usize;
            let end = self.offsets[index + 1] as usize;
            let uncovered = self.samples_per_pixel - (end - start) as u32;
            let mut radiance = self.background * f64::from(uncovered);
            for sample in &self.samples[start..end] {
                let stress = field.sample(sample.point[0], sample.point[1]);
                let stress = std::array::from_fn(|axis| stress[axis] + pre_stress[axis]);
                radiance +=
                    optics.sample(stress, sample.thickness, 0.0).hadamard(sample.throughput)
                        + sample.reflection;
            }
            *pixel = radiance / f64::from(self.samples_per_pixel);
        });
        let maximum_radiance = pixels
            .par_iter()
            .map(|pixel| -> Result<f64, &'static str> {
                if !pixel.is_finite() || pixel.x < 0.0 || pixel.y < 0.0 || pixel.z < 0.0 {
                    Err("crystal cached optical calculation produced invalid radiance")
                } else {
                    Ok(pixel.x.max(pixel.y).max(pixel.z))
                }
            })
            .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))?;
        Ok(RasterFrame { pixels, material_pixels: self.material_pixels, maximum_radiance })
    }
}

fn prepare_row(
    row: usize,
    layout: &Layout,
    surface: &Surface,
    config: &CrystalConfig,
    camera: &Camera,
    render: &RenderConfig,
) -> Result<Row, String> {
    let mut counts = Vec::new();
    counts.try_reserve_exact(layout.width).map_err(|e| e.to_string())?;
    counts.resize(layout.width, 0u8);
    let mut samples = Vec::new();
    if row < layout.rows[0] || row >= layout.rows[1] {
        return Ok(Row { counts, samples });
    }
    samples
        .try_reserve_exact(
            (layout.columns[1] - layout.columns[0]) * layout.samples_per_pixel as usize,
        )
        .map_err(|e| e.to_string())?;
    let pixel_size = camera.orthographic_height / f64::from(render.height);
    for (column, count) in
        counts.iter_mut().enumerate().take(layout.columns[1]).skip(layout.columns[0])
    {
        for sy in 0..render.aa {
            let y = camera.target.y
                + (f64::from(render.height) * 0.5
                    - row as f64
                    - (f64::from(sy) + 0.5) / f64::from(render.aa))
                    * pixel_size;
            for sx in 0..render.aa {
                let x = camera.target.x
                    + (column as f64 + (f64::from(sx) + 0.5) / f64::from(render.aa)
                        - f64::from(render.width) * 0.5)
                        * pixel_size;
                let Some(surface) = surface.sample(x, y) else { continue };
                let thickness = config.center_thickness
                    * (config.edge_thickness + (1.0 - config.edge_thickness) * surface.dome)
                    * surface.optical_path_scale;
                let absorption = V3::new(
                    (-config.absorption.x * thickness).exp(),
                    (-config.absorption.y * thickness).exp(),
                    (-config.absorption.z * thickness).exp(),
                );
                let edge = super::smoothstep(
                    ((1.0 - surface.rho) / config.interior_feather).clamp(0.0, 1.0),
                );
                let transmission = (1.0 - surface.fresnel).powi(2);
                samples.push(CachedSample {
                    point: surface.refracted_point,
                    thickness,
                    throughput: absorption * (edge * transmission),
                    reflection: surface.reflection
                        * (config.surface_strength + config.edge_strength * surface.rho.powi(8)),
                });
                *count += 1;
            }
        }
    }
    Ok(Row { counts, samples })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atelier::{OrbitSeries, crystal::field::ElasticSheet};

    fn setup() -> (CrystalConfig, Camera, RenderConfig, StressField, PolarizedOptics) {
        let mut config = CrystalConfig::default();
        config.field.grid = [17, 13];
        config.field.load_softness = 1.0;
        let camera = Camera {
            position: V3::new(0.13, -0.2, 12.0),
            target: V3::new(0.13, -0.2, 0.0),
            orthographic_height: 7.2,
            ..Camera::default()
        };
        let render = RenderConfig {
            width: 41,
            height: 29,
            aa: 3,
            background: V3::new(0.04, 0.03, 0.02),
            ..RenderConfig::default()
        };
        let source = OrbitSeries::new(&crate::silk::OrbitData {
            seed: "0xc1".into(),
            dt: 0.1,
            masses: [1.0; 3],
            samples: (0..5)
                .map(|i| {
                    let t = f64::from(i) / 4.0;
                    [
                        V3::new(-1.0 + t, -0.45, 0.2),
                        V3::new(1.0 - 0.5 * t, 0.2, -0.2),
                        V3::new(0.25, 0.9 - 0.3 * t, 0.0),
                    ]
                })
                .collect(),
            provenance: serde_json::Value::Null,
        })
        .unwrap();
        let field = ElasticSheet::new(&config.field).unwrap().solve(&source, 0.4).unwrap();
        let optics = PolarizedOptics::new(&config.optics).unwrap();
        (config, camera, render, field, optics)
    }

    #[test]
    fn cached_samples_remain_seventy_two_bytes_and_bounds_cover_4k_aa3() {
        assert_eq!(size_of::<CachedSample>(), 72);
        let config = CrystalConfig::default();
        let camera = Camera {
            position: V3::new(0.0, 0.0, 12.0),
            orthographic_height: 7.2,
            ..Camera::default()
        };
        let render = RenderConfig { width: 3840, height: 2160, aa: 3, ..RenderConfig::default() };
        assert!(
            required_bytes_upper_bound(&config, &camera, &render).unwrap()
                < 4 * 1024 * 1024 * 1024usize
        );
        assert!(SurfaceRaster::new(&config, &camera, &render, 1).is_err());
        let huge = RenderConfig { width: 16384, height: 16384, aa: 8, ..render };
        assert!(required_bytes_upper_bound(&config, &camera, &huge).is_err());
    }

    #[test]
    fn cache_matches_independent_direct_surface_loop_at_double_precision() {
        let (config, camera, render, field, optics) = setup();
        let raster = SurfaceRaster::new(&config, &camera, &render, 16 * 1024 * 1024).unwrap();
        let pre_stress = [0.002, 0.001, 0.004];
        let actual = raster.render(&field, &optics, pre_stress).unwrap();
        let surface = Surface::new(&super::super::surface_config(&config)).unwrap();
        let pixel_size = camera.orthographic_height / f64::from(render.height);
        let mut material_pixels = 0;
        for row in 0..render.height {
            for column in 0..render.width {
                let mut expected = V3::ZERO;
                let mut covered = false;
                for sy in 0..render.aa {
                    for sx in 0..render.aa {
                        let x = camera.target.x
                            + (f64::from(column) + (f64::from(sx) + 0.5) / f64::from(render.aa)
                                - f64::from(render.width) * 0.5)
                                * pixel_size;
                        let y = camera.target.y
                            + (f64::from(render.height) * 0.5
                                - f64::from(row)
                                - (f64::from(sy) + 0.5) / f64::from(render.aa))
                                * pixel_size;
                        let Some(hit) = surface.sample(x, y) else {
                            expected += render.background;
                            continue;
                        };
                        covered = true;
                        let thickness = config.center_thickness
                            * (config.edge_thickness + (1.0 - config.edge_thickness) * hit.dome)
                            * hit.optical_path_scale;
                        let stress = field.sample(hit.refracted_point[0], hit.refracted_point[1]);
                        let internal = optics.sample(
                            std::array::from_fn(|a| stress[a] + pre_stress[a]),
                            thickness,
                            0.0,
                        );
                        let absorption = V3::new(
                            (-config.absorption.x * thickness).exp(),
                            (-config.absorption.y * thickness).exp(),
                            (-config.absorption.z * thickness).exp(),
                        );
                        let edge = super::super::smoothstep(
                            ((1.0 - hit.rho) / config.interior_feather).clamp(0.0, 1.0),
                        );
                        expected += internal.hadamard(absorption)
                            * (edge * (1.0 - hit.fresnel).powi(2))
                            + hit.reflection
                                * (config.surface_strength
                                    + config.edge_strength * hit.rho.powi(8));
                    }
                }
                expected /= f64::from(render.aa * render.aa);
                let pixel = actual.pixels[(row * render.width + column) as usize];
                assert!(
                    (pixel - expected).length() < 5e-15,
                    "pixel {column},{row}: {pixel:?} / {expected:?}"
                );
                material_pixels += usize::from(covered);
            }
        }
        assert_eq!(actual.material_pixels, material_pixels);
        assert!(actual.maximum_radiance.is_finite() && actual.maximum_radiance > 0.0);
    }

    #[test]
    fn parallel_row_preparation_preserves_exact_coverage_and_sample_order() {
        let (config, camera, render, _, _) = setup();
        let build = |workers| {
            rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap().install(|| {
                SurfaceRaster::new(&config, &camera, &render, 16 * 1024 * 1024).unwrap()
            })
        };
        let one = build(1);
        let three = build(3);
        assert_eq!(one.offsets, three.offsets);
        assert_eq!(one.samples, three.samples);
        assert_eq!(one.material_pixels, three.material_pixels);
    }
}
