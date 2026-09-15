//! Cropped linear-light convergence checks for Light, Engraving and Eclipse recipes.
//!
//! No rendering starts without `--run`. Example:
//! `atelier_convergence --config recipe.json --orbit orbit.bin --frame 1670
//! --crop 1400,700,256,256 --output comparison.json --png-dir comparison --run`.
//! Set either reference factor to one to isolate the other comparison axis.

use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};
use three_body_problem::{
    atelier::{
        Camera, OrbitSeries, RenderConfig, SilkResult, V3, eclipse, engraving, light, render,
    },
    silk::cache,
};

#[derive(Parser)]
#[command(
    about = "Plan or run a cropped linear-light convergence comparison",
    arg_required_else_help = true
)]
struct Args {
    /// Complete study config JSON, or a render manifest containing its config.
    #[arg(long)]
    config: PathBuf,
    /// Original CSORBIT1 cache used by the study.
    #[arg(long)]
    orbit: PathBuf,
    /// Zero-based frame index in the original film.
    #[arg(long)]
    frame: usize,
    /// Rectangle in original output pixels: x,y,width,height.
    #[arg(long)]
    crop: Crop,
    /// Reference pixels per native pixel axis; one isolates temporal/grid changes.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=2))]
    spatial_factor: u32,
    /// Reference shutter sample multiplier; one isolates spatial/grid changes.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=2))]
    temporal_factor: u32,
    /// Double both Light source-grid dimensions for the reference only.
    #[arg(long)]
    double_light_source_grid: bool,
    /// Optional CPU worker count; results do not depend on scheduling.
    #[arg(long)]
    threads: Option<usize>,
    /// Also save the JSON report at this path; stdout always receives JSON.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Save native.png and reference.png after linear reference downsampling.
    #[arg(long)]
    png_dir: Option<PathBuf>,
    /// Explicitly start the two renders. Without this flag, only print the plan.
    #[arg(long)]
    run: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Crop {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl FromStr for Crop {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let coordinates: Vec<u32> = value
            .split(',')
            .map(|part| {
                part.trim().parse().map_err(|_| "crop needs four unsigned integers".to_owned())
            })
            .collect::<Result<_, _>>()?;
        let [x, y, width, height]: [u32; 4] =
            coordinates.try_into().map_err(|_| "crop must be x,y,width,height".to_owned())?;
        Ok(Self { x, y, width, height })
    }
}

// Unused styles and manifest fields are deliberately accepted. The actual
// selected renderer still parses and validates its complete public config.
#[derive(Clone, Deserialize)]
struct StudyConfig {
    kind: String,
    frames: usize,
    fps: u32,
    temporal_samples: usize,
    shutter_fraction: f64,
    camera: Camera,
    render: RenderConfig,
    light: Option<light::LightConfig>,
    engraving: Option<engraving::EngravingConfig>,
    eclipse: Option<eclipse::EclipseConfig>,
}

struct Pass {
    camera: Camera,
    render: RenderConfig,
    temporal_samples: usize,
    light: Option<light::LightConfig>,
    engraving: Option<engraving::EngravingConfig>,
    eclipse: Option<eclipse::EclipseConfig>,
}

#[derive(Serialize)]
struct PassPlan {
    dimensions: [u32; 2],
    aa_per_axis: u32,
    temporal_samples: usize,
    camera: Camera,
    world_units_per_pixel: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    light_source_grid: Option<[usize; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    light_footprint_pixels: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    light_footprint_world_units: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eclipse_minimum_sigma_pixels: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eclipse_minimum_sigma_world_units: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eclipse_corona_radii_world_units: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eclipse_hairs_per_body: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eclipse_segments_per_hair: Option<usize>,
}

impl Pass {
    fn plan(&self) -> PassPlan {
        let pitch = self.camera.orthographic_height / f64::from(self.render.height);
        PassPlan {
            dimensions: [self.render.width, self.render.height],
            aa_per_axis: self.render.aa,
            temporal_samples: self.temporal_samples,
            camera: self.camera.clone(),
            world_units_per_pixel: pitch,
            light_source_grid: self.light.as_ref().map(|value| value.source_grid),
            light_footprint_pixels: self.light.as_ref().map(|value| value.footprint_pixels),
            light_footprint_world_units: self
                .light
                .as_ref()
                .map(|value| value.footprint_pixels * pitch),
            eclipse_minimum_sigma_pixels: self
                .eclipse
                .as_ref()
                .map(|value| value.minimum_sigma_pixels),
            eclipse_minimum_sigma_world_units: self
                .eclipse
                .as_ref()
                .map(|value| value.minimum_sigma_pixels * pitch),
            eclipse_corona_radii_world_units: self.eclipse.as_ref().map(|value| value.corona_radii),
            eclipse_hairs_per_body: self.eclipse.as_ref().map(|value| value.corona_hairs),
            eclipse_segments_per_hair: self.eclipse.as_ref().map(|value| value.corona_segments),
        }
    }
}

#[derive(Serialize)]
struct Plan {
    config_path: PathBuf,
    orbit_path: PathBuf,
    kind: String,
    original_dimensions: [u32; 2],
    frame: usize,
    film_frames: usize,
    fps: u32,
    frame_source_fraction: f64,
    raw_shutter_interval: [f64; 2],
    crop: Crop,
    spatial_factor: u32,
    temporal_factor: u32,
    double_light_source_grid: bool,
    native: PassPlan,
    reference: PassPlan,
    comparison_domain: &'static str,
    preview_scope: &'static str,
}

fn load_config(path: &Path) -> SilkResult<StudyConfig> {
    let mut document: serde_json::Value = serde_json::from_reader(fs::File::open(path)?)?;
    if let Some(config) = document.get_mut("config") {
        document = config.take();
    }
    Ok(serde_json::from_value(document)?)
}

fn validate(config: &StudyConfig, args: &Args) -> SilkResult<()> {
    if !matches!(config.kind.as_str(), "light" | "engraving" | "eclipse")
        || (config.kind == "light" && config.light.is_none())
        || (config.kind == "engraving" && config.engraving.is_none())
        || (config.kind == "eclipse" && config.eclipse.is_none())
    {
        return Err("probe requires a complete Light, Engraving or Eclipse config".into());
    }
    if config.frames < 2
        || config.fps == 0
        || args.frame >= config.frames
        || config.temporal_samples == 0
        || !config.shutter_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.shutter_fraction)
        || args.threads == Some(0)
        || !(1..=2).contains(&args.spatial_factor)
        || !(1..=2).contains(&args.temporal_factor)
    {
        return Err("invalid film timing, frame, reference factor or worker count".into());
    }
    let crop = args.crop;
    if config.render.width == 0
        || config.render.height == 0
        || config.render.aa == 0
        || crop.width == 0
        || crop.height == 0
        || crop.x.checked_add(crop.width).is_none_or(|end| end > config.render.width)
        || crop.y.checked_add(crop.height).is_none_or(|end| end > config.render.height)
    {
        return Err("crop must be a nonempty rectangle inside the original output".into());
    }
    let camera = &config.camera;
    if !camera.position.is_finite()
        || !camera.target.is_finite()
        || !camera.up.is_finite()
        || !camera.orthographic_height.is_finite()
        || camera.orthographic_height <= 0.0
        || camera.position.x != camera.target.x
        || camera.position.y != camera.target.y
        || camera.target.z != 0.0
        || camera.position.z <= 0.0
        || camera.up.x != 0.0
        || camera.up.z != 0.0
        || camera.up.y <= 0.0
    {
        return Err("probe currently requires a frontal, unrolled +Y-up camera facing z=0".into());
    }
    if args.double_light_source_grid && config.kind != "light" {
        return Err("--double-light-source-grid applies only to Light".into());
    }
    Ok(())
}

fn make_pass(
    config: &StudyConfig,
    crop: Crop,
    spatial_factor: u32,
    temporal_factor: u32,
    double_grid: bool,
) -> SilkResult<Pass> {
    let pitch = config.camera.orthographic_height / f64::from(config.render.height);
    let shift = V3::new(
        (f64::from(crop.x) + f64::from(crop.width) * 0.5 - f64::from(config.render.width) * 0.5)
            * pitch,
        (f64::from(config.render.height) * 0.5 - f64::from(crop.y) - f64::from(crop.height) * 0.5)
            * pitch,
        0.0,
    );
    let mut camera = config.camera.clone();
    camera.position += shift;
    camera.target += shift;
    camera.orthographic_height = pitch * f64::from(crop.height);
    let mut render = config.render.clone();
    render.width = crop.width.checked_mul(spatial_factor).ok_or("reference width overflow")?;
    render.height = crop.height.checked_mul(spatial_factor).ok_or("reference height overflow")?;
    if u64::from(render.width) * u64::from(render.height) > 100_000_000 {
        return Err("probe pass exceeds the 100-million-pixel receiving bound".into());
    }
    let temporal_samples = config
        .temporal_samples
        .checked_mul(temporal_factor as usize)
        .ok_or("reference shutter sample count overflow")?;
    let light = if config.kind == "light" {
        let mut value = config.light.clone().ok_or("missing Light parameters")?;
        // Keep the optical footprint fixed in world units. Otherwise doubling
        // receiving resolution would silently halve its physical blur radius.
        value.footprint_pixels *= f64::from(spatial_factor);
        if double_grid {
            for dimension in &mut value.source_grid {
                *dimension = dimension.checked_mul(2).ok_or("Light source-grid overflow")?;
            }
        }
        Some(value)
    } else {
        None
    };
    let eclipse = if config.kind == "eclipse" {
        let mut value = config.eclipse.clone().ok_or("missing Eclipse parameters")?;
        // Physical hair radii and all material geometry stay fixed. Only the
        // pixel-unit reconstruction sigma scales with receiving resolution.
        value.minimum_sigma_pixels *= f64::from(spatial_factor);
        Some(value)
    } else {
        None
    };
    Ok(Pass {
        camera,
        render,
        temporal_samples,
        light,
        engraving: (config.kind == "engraving").then(|| config.engraving.clone()).flatten(),
        eclipse,
    })
}

fn sample_time(config: &StudyConfig, frame: usize, samples: usize, sample: usize) -> f64 {
    let fraction = frame as f64 / (config.frames - 1) as f64;
    let offset = ((sample as f64 + 0.5) / samples as f64 - 0.5) * config.shutter_fraction
        / (config.frames - 1) as f64;
    (fraction + offset).clamp(0.0, 1.0)
}

fn sample_interval(config: &StudyConfig, frame: usize, samples: usize, sample: usize) -> [f64; 2] {
    let fraction = frame as f64 / (config.frames - 1) as f64;
    [sample, sample + 1].map(|edge| {
        fraction
            + (edge as f64 / samples as f64 - 0.5) * config.shutter_fraction
                / (config.frames - 1) as f64
    })
}

#[derive(Default, Serialize)]
struct LightSummary {
    gamut_compressed_pixel_samples: u64,
    max_absolute_conservation_residual: f64,
    max_source_midpoint_error_pixels: f64,
}

#[derive(Default, Serialize)]
struct EngravingSummary {
    accepted_cells: u64,
    spatial_refinements: u64,
    temporal_refinements: u64,
    rounded_negative_values: u64,
    max_accepted_phase_residual: f64,
    max_accepted_envelope_variation: f64,
}

#[derive(Default, Serialize)]
struct EclipseSummary {
    accepted_cells: u64,
    refinements: u64,
    max_spatial_depth: usize,
    max_accepted_error_indicator: f64,
    max_chord_error_pixels: f64,
    curves_per_exposure: usize,
    segments_per_exposure: usize,
    mean_estimated_unoccluded_corona_luminance: f64,
}

#[derive(Serialize)]
struct PassSummary {
    seconds: f64,
    clamped_midpoints: Vec<f64>,
    raw_exposure_intervals: Vec<[f64; 2]>,
    pixel_samples: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    light: Option<LightSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engraving: Option<EngravingSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eclipse: Option<EclipseSummary>,
}

fn render_pass(
    label: &str,
    source: &OrbitSeries,
    config: &StudyConfig,
    frame: usize,
    pass: &Pass,
) -> SilkResult<(Vec<V3>, PassSummary)> {
    let instant = Instant::now();
    let mut summary = PassSummary {
        seconds: 0.0,
        clamped_midpoints: Vec::with_capacity(pass.temporal_samples),
        raw_exposure_intervals: Vec::with_capacity(pass.temporal_samples),
        pixel_samples: (u64::from(pass.render.width) * u64::from(pass.render.height))
            .checked_mul(pass.temporal_samples as u64)
            .ok_or("pixel/shutter sample count overflow")?,
        light: None,
        engraving: None,
        eclipse: None,
    };
    let mut accumulated: Vec<V3> = Vec::new();
    let weight = 1.0 / pass.temporal_samples as f64;
    for sample in 0..pass.temporal_samples {
        let time = sample_time(config, frame, pass.temporal_samples, sample);
        let interval = sample_interval(config, frame, pass.temporal_samples, sample);
        summary.clamped_midpoints.push(time);
        summary.raw_exposure_intervals.push(interval);
        let pixels = if let Some(light) = &pass.light {
            let product = light::render_linear(source, time, light, &pass.camera, &pass.render)?;
            product.diagnostics.validate()?;
            let diagnostics = summary.light.get_or_insert_with(LightSummary::default);
            diagnostics.gamut_compressed_pixel_samples +=
                product.diagnostics.gamut_compressed_pixels as u64;
            diagnostics.max_source_midpoint_error_pixels = diagnostics
                .max_source_midpoint_error_pixels
                .max(product.diagnostics.sampled_midpoint_error_pixels);
            for band in &product.diagnostics.bands {
                diagnostics.max_absolute_conservation_residual = diagnostics
                    .max_absolute_conservation_residual
                    .max(band.conservation_residual.abs());
            }
            product.pixels
        } else if let Some(eclipse) = &pass.eclipse {
            let product =
                eclipse::render_linear(source, time, eclipse, &pass.camera, &pass.render)?;
            product.diagnostics.validate()?;
            let diagnostics = summary.eclipse.get_or_insert_with(EclipseSummary::default);
            diagnostics.accepted_cells += product.diagnostics.accepted_cells;
            diagnostics.refinements += product.diagnostics.refinements;
            diagnostics.max_spatial_depth =
                diagnostics.max_spatial_depth.max(product.diagnostics.max_spatial_depth);
            diagnostics.max_accepted_error_indicator = diagnostics
                .max_accepted_error_indicator
                .max(product.diagnostics.max_accepted_error_indicator);
            diagnostics.max_chord_error_pixels =
                diagnostics.max_chord_error_pixels.max(product.diagnostics.max_chord_error_pixels);
            diagnostics.curves_per_exposure = product.diagnostics.curves;
            diagnostics.segments_per_exposure = product.diagnostics.segments;
            diagnostics.mean_estimated_unoccluded_corona_luminance +=
                product.diagnostics.estimated_corona_luminance * weight;
            product.pixels
        } else {
            let product = engraving::render_linear(
                source,
                time,
                interval,
                pass.engraving.as_ref().ok_or("missing Engraving parameters")?,
                &pass.camera,
                &pass.render,
            )?;
            product.diagnostics.validate()?;
            let diagnostics = summary.engraving.get_or_insert_with(EngravingSummary::default);
            diagnostics.accepted_cells += product.diagnostics.accepted_cells;
            diagnostics.spatial_refinements += product.diagnostics.spatial_refinements;
            diagnostics.temporal_refinements += product.diagnostics.temporal_refinements;
            diagnostics.rounded_negative_values += product.diagnostics.rounded_negative_values;
            diagnostics.max_accepted_phase_residual = diagnostics
                .max_accepted_phase_residual
                .max(product.diagnostics.max_accepted_phase_residual);
            diagnostics.max_accepted_envelope_variation = diagnostics
                .max_accepted_envelope_variation
                .max(product.diagnostics.max_accepted_envelope_variation);
            product.pixels
        };
        if accumulated.is_empty() {
            accumulated = pixels;
            accumulated.par_iter_mut().for_each(|value| *value *= weight);
        } else {
            if accumulated.len() != pixels.len() {
                return Err("receiving dimensions changed between shutter samples".into());
            }
            accumulated.par_iter_mut().zip(pixels.par_iter()).for_each(|(a, b)| *a += *b * weight);
        }
        eprintln!(
            "{label}: shutter sample {}/{}, {:.2}s",
            sample + 1,
            pass.temporal_samples,
            instant.elapsed().as_secs_f64()
        );
    }
    summary.seconds = instant.elapsed().as_secs_f64();
    Ok((accumulated, summary))
}

fn downsample(values: &[V3], crop: Crop, factor: u32) -> SilkResult<Vec<V3>> {
    let width = crop.width as usize;
    let height = crop.height as usize;
    let factor = factor as usize;
    if factor == 0 || values.len() != width * height * factor * factor {
        return Err("reference buffer does not match the declared spatial factor".into());
    }
    let mut result = vec![V3::ZERO; width * height];
    result.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, out) in row.iter_mut().enumerate() {
            for dy in 0..factor {
                for dx in 0..factor {
                    *out += values[(y * factor + dy) * width * factor + x * factor + dx];
                }
            }
            *out /= (factor * factor) as f64;
        }
    });
    Ok(result)
}

#[derive(Clone, Copy, Default)]
struct Sum {
    value: f64,
    error: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.error;
        let next = self.value + corrected;
        self.error = (next - self.value) - corrected;
        self.value = next;
    }
}

fn luminance(value: V3) -> f64 {
    value.x * 0.2126 + value.y * 0.7152 + value.z * 0.0722
}

#[derive(Serialize)]
struct ErrorStats {
    max_absolute: f64,
    mean_absolute: f64,
    rms: f64,
    mean_signed_native_minus_reference: f64,
}

#[derive(Serialize)]
struct Metrics {
    pixels: usize,
    rgb_all_channels: ErrorStats,
    rgb_by_channel: [ErrorStats; 3],
    luminance: ErrorStats,
    native_mean_luminance: f64,
    reference_mean_luminance: f64,
    relative_integrated_luminance_difference: Option<f64>,
    weighting: &'static str,
    subtracted_background_luminance: f64,
    foreground_weight_sum: f64,
    foreground_weighted_luminance_mae: Option<f64>,
    foreground_weighted_luminance_rms: Option<f64>,
    foreground_weighted_relative_luminance_rms: Option<f64>,
    foreground_weighted_rgb_rms: Option<f64>,
    active_pixel_threshold: f64,
    active_pixels: usize,
    active_rgb_rms: Option<f64>,
    worst_global_pixel: [u32; 2],
    worst_native_rgb: V3,
    worst_reference_rgb: V3,
}

fn compare(native: &[V3], reference: &[V3], crop: Crop, background: f64) -> SilkResult<Metrics> {
    if native.is_empty()
        || native.len() != reference.len()
        || native.len() != crop.width as usize * crop.height as usize
        || native.iter().chain(reference).any(|value| !value.is_finite())
    {
        return Err("comparison needs matching finite, nonempty linear RGB crops".into());
    }
    let count = native.len() as f64;
    let mut maximum = [0.0_f64; 4];
    let mut absolute = [Sum::default(); 4];
    let mut square = [Sum::default(); 4];
    let mut signed = [Sum::default(); 4];
    let mut means = [Sum::default(); 2];
    let mut weight_sum = Sum::default();
    let mut weighted_abs = Sum::default();
    let mut weighted_square = Sum::default();
    let mut weighted_rgb_square = Sum::default();
    let mut weighted_signal_square = Sum::default();
    let mut peak_signal: f64 = 0.0;
    let mut worst = (0.0_f64, 0_usize);
    let mut pixel_scores = Vec::with_capacity(native.len());
    for (index, (&a, &b)) in native.iter().zip(reference).enumerate() {
        let ya = luminance(a);
        let yb = luminance(b);
        let delta = a - b;
        let errors = [delta.x, delta.y, delta.z, ya - yb];
        for axis in 0..4 {
            maximum[axis] = maximum[axis].max(errors[axis].abs());
            absolute[axis].add(errors[axis].abs());
            square[axis].add(errors[axis] * errors[axis]);
            signed[axis].add(errors[axis]);
        }
        let pixel_error = delta.x.abs().max(delta.y.abs()).max(delta.z.abs());
        if pixel_error > worst.0 {
            worst = (pixel_error, index);
        }
        means[0].add(ya);
        means[1].add(yb);
        let weight = (ya.max(yb) - background).max(0.0);
        let rgb_square = delta.length_squared() / 3.0;
        peak_signal = peak_signal.max(weight);
        weight_sum.add(weight);
        weighted_abs.add(weight * (ya - yb).abs());
        weighted_square.add(weight * (ya - yb).powi(2));
        weighted_rgb_square.add(weight * rgb_square);
        weighted_signal_square.add(weight * weight * weight);
        pixel_scores.push((weight, rgb_square));
    }
    let channel = |axis: usize| ErrorStats {
        max_absolute: maximum[axis],
        mean_absolute: absolute[axis].value / count,
        rms: (square[axis].value / count).sqrt(),
        mean_signed_native_minus_reference: signed[axis].value / count,
    };
    let ratio =
        |numerator: f64, denominator: f64| (denominator > 0.0).then(|| numerator / denominator);
    let threshold = peak_signal * 0.01;
    let mut active = 0;
    let mut active_square = Sum::default();
    for (weight, error) in pixel_scores {
        if weight > 0.0 && weight >= threshold {
            active += 1;
            active_square.add(error);
        }
    }
    if square.iter().chain(&absolute).chain(&signed).chain(&means).any(|sum| !sum.value.is_finite())
        || [
            weight_sum,
            weighted_abs,
            weighted_square,
            weighted_rgb_square,
            weighted_signal_square,
            active_square,
        ]
        .iter()
        .any(|sum| !sum.value.is_finite())
    {
        return Err("linear comparison totals exceeded finite numeric range".into());
    }
    Ok(Metrics {
        pixels: native.len(),
        rgb_all_channels: ErrorStats {
            max_absolute: maximum[..3].iter().copied().fold(0.0, f64::max),
            mean_absolute: absolute[..3].iter().map(|sum| sum.value).sum::<f64>() / (3.0 * count),
            rms: (square[..3].iter().map(|sum| sum.value).sum::<f64>() / (3.0 * count)).sqrt(),
            mean_signed_native_minus_reference: signed[..3]
                .iter()
                .map(|sum| sum.value)
                .sum::<f64>()
                / (3.0 * count),
        },
        rgb_by_channel: [channel(0), channel(1), channel(2)],
        luminance: channel(3),
        native_mean_luminance: means[0].value / count,
        reference_mean_luminance: means[1].value / count,
        relative_integrated_luminance_difference: ratio(
            means[0].value - means[1].value,
            means[1].value,
        ),
        weighting: "w=max(max(native_Y,reference_Y)-background_Y,0); relative RMS=sqrt(sum(w*dY^2)/sum(w^3)); active pixels have w>=1% of crop peak",
        subtracted_background_luminance: background,
        foreground_weight_sum: weight_sum.value,
        foreground_weighted_luminance_mae: ratio(weighted_abs.value, weight_sum.value),
        foreground_weighted_luminance_rms: ratio(weighted_square.value, weight_sum.value)
            .map(f64::sqrt),
        foreground_weighted_relative_luminance_rms: ratio(
            weighted_square.value,
            weighted_signal_square.value,
        )
        .map(f64::sqrt),
        foreground_weighted_rgb_rms: ratio(weighted_rgb_square.value, weight_sum.value)
            .map(f64::sqrt),
        active_pixel_threshold: threshold,
        active_pixels: active,
        active_rgb_rms: ratio(active_square.value, active as f64).map(f64::sqrt),
        worst_global_pixel: [
            crop.x + (worst.1 % crop.width as usize) as u32,
            crop.y + (worst.1 / crop.width as usize) as u32,
        ],
        worst_native_rgb: native[worst.1],
        worst_reference_rgb: reference[worst.1],
    })
}

fn emit(value: &impl Serialize, output: Option<&Path>) -> SilkResult<()> {
    let text = serde_json::to_string_pretty(value)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, format!("{text}\n"))?;
    }
    println!("{text}");
    Ok(())
}

fn main() -> SilkResult<()> {
    let args = Args::parse();
    let config = load_config(&args.config)?;
    validate(&config, &args)?;
    let native_pass = make_pass(&config, args.crop, 1, 1, false)?;
    let reference_pass = make_pass(
        &config,
        args.crop,
        args.spatial_factor,
        args.temporal_factor,
        args.double_light_source_grid,
    )?;
    let plan = Plan {
        config_path: args.config.clone(),
        orbit_path: args.orbit.clone(),
        kind: config.kind.clone(),
        original_dimensions: [config.render.width, config.render.height],
        frame: args.frame,
        film_frames: config.frames,
        fps: config.fps,
        frame_source_fraction: args.frame as f64 / (config.frames - 1) as f64,
        raw_shutter_interval: sample_interval(&config, args.frame, 1, 0),
        crop: args.crop,
        spatial_factor: args.spatial_factor,
        temporal_factor: args.temporal_factor,
        double_light_source_grid: args.double_light_source_grid,
        native: native_pass.plan(),
        reference: reference_pass.plan(),
        comparison_domain: "Public renderer linear artistic RGB, before bloom/exposure/encoding. Light gain and gamut handling are already applied; this measures the finished rendering path, not isolated optical transport.",
        preview_scope: "Both previews finish the native-sized linear crop independently; any bloom uses crop-local surroundings. Metrics are computed before finishing.",
    };
    if !args.run {
        return emit(
            &serde_json::json!({"status":"planned", "render_started":false, "plan":plan}),
            args.output.as_deref(),
        );
    }
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    let orbit = cache::read_orbit(&args.orbit)?;
    let source = OrbitSeries::new(&orbit)?;
    let (native, native_summary) =
        render_pass("native", &source, &config, args.frame, &native_pass)?;
    let (reference, reference_summary) =
        render_pass("reference", &source, &config, args.frame, &reference_pass)?;
    let reference = downsample(&reference, args.crop, args.spatial_factor)?;
    // Eclipse also contains meaningful values below the configured background;
    // retain total luminance weights for its opaque dark forms, as for Ivory.
    let background = if config.kind == "engraving"
        || (config.kind == "light"
            && config
                .light
                .as_ref()
                .is_some_and(|value| matches!(value.display, light::LightDisplay::DarkGain)))
    {
        luminance(config.render.background)
    } else {
        0.0
    };
    let metrics = compare(&native, &reference, args.crop, background)?;
    let previews = if let Some(directory) = &args.png_dir {
        fs::create_dir_all(directory)?;
        let native_path = directory.join("native.png");
        let reference_path = directory.join("reference.png");
        render::finish_linear(native, &native_pass.render)?
            .save_with_format(&native_path, image::ImageFormat::Png)?;
        render::finish_linear(reference, &native_pass.render)?
            .save_with_format(&reference_path, image::ImageFormat::Png)?;
        Some([native_path, reference_path])
    } else {
        None
    };
    emit(
        &serde_json::json!({
            "status":"complete", "plan":plan,
            "executable_sha256":cache::file_hash(&std::env::current_exe()?)?,
            "config_sha256":cache::file_hash(&args.config)?, "orbit_sha256":cache::file_hash(&args.orbit)?,
            "orbit_seed":orbit.seed, "native":native_summary, "reference":reference_summary,
            "metrics":metrics, "previews":previews,
        }),
        args.output.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn study() -> StudyConfig {
        StudyConfig {
            kind: "light".into(),
            frames: 1802,
            fps: 60,
            temporal_samples: 16,
            shutter_fraction: 0.5,
            camera: Camera {
                position: V3::new(0.3, -0.2, 10.0),
                target: V3::new(0.3, -0.2, 0.0),
                up: V3::new(0.0, 1.0, 0.0),
                orthographic_height: 7.4,
            },
            render: RenderConfig { width: 3840, height: 2160, aa: 3, ..RenderConfig::default() },
            light: Some(light::LightConfig::default()),
            engraving: None,
            eclipse: None,
        }
    }

    fn world(camera: &Camera, render: &RenderConfig, x: f64, y: f64) -> V3 {
        let pitch = camera.orthographic_height / f64::from(render.height);
        camera.target
            + V3::new(
                (x - f64::from(render.width) * 0.5) * pitch,
                (f64::from(render.height) * 0.5 - y) * pitch,
                0.0,
            )
    }

    #[test]
    fn crop_and_reference_pixels_preserve_the_original_camera_lattice_and_light_footprint() {
        let config = study();
        let crop = Crop { x: 1371, y: 653, width: 19, height: 13 };
        let native = make_pass(&config, crop, 1, 1, false).unwrap();
        let reference = make_pass(&config, crop, 2, 2, true).unwrap();
        for y in 0..crop.height {
            for x in 0..crop.width {
                let expected = world(
                    &config.camera,
                    &config.render,
                    f64::from(crop.x + x) + 0.5,
                    f64::from(crop.y + y) + 0.5,
                );
                let actual =
                    world(&native.camera, &native.render, f64::from(x) + 0.5, f64::from(y) + 0.5);
                let children_center = world(
                    &reference.camera,
                    &reference.render,
                    2.0 * f64::from(x) + 1.0,
                    2.0 * f64::from(y) + 1.0,
                );
                assert!((actual - expected).length() < 2e-15);
                assert!((children_center - expected).length() < 2e-15);
                for dy in [0.25, 0.75] {
                    for dx in [0.25, 0.75] {
                        let expected_child = world(
                            &config.camera,
                            &config.render,
                            f64::from(crop.x + x) + dx,
                            f64::from(crop.y + y) + dy,
                        );
                        let actual_child = world(
                            &reference.camera,
                            &reference.render,
                            2.0 * (f64::from(x) + dx),
                            2.0 * (f64::from(y) + dy),
                        );
                        assert!((actual_child - expected_child).length() < 2e-15);
                    }
                }
            }
        }
        assert_eq!(native.render.aa, reference.render.aa);
        let a = native.light.unwrap();
        let b = reference.light.unwrap();
        assert_eq!(a.source_domain, b.source_domain);
        assert_eq!(a.source_center, b.source_center);
        assert_eq!(b.source_grid, a.source_grid.map(|value| value * 2));
        assert_eq!(b.footprint_pixels, a.footprint_pixels * 2.0);
        assert_eq!(native.temporal_samples * 2, reference.temporal_samples);
    }

    #[test]
    fn eclipse_reference_preserves_crop_world_footprint_and_physical_hair_geometry() {
        let mut config = study();
        config.kind = "eclipse".into();
        config.light = None;
        config.eclipse = Some(eclipse::EclipseConfig::default());
        config.temporal_samples = 128;
        let crop = Crop { x: 271, y: 137, width: 17, height: 23 };
        let native = make_pass(&config, crop, 1, 1, false).unwrap();
        let reference = make_pass(&config, crop, 2, 2, false).unwrap();
        assert!(native.light.is_none() && native.engraving.is_none());
        let a = native.eclipse.as_ref().unwrap();
        let b = reference.eclipse.as_ref().unwrap();
        assert_eq!(b.minimum_sigma_pixels, 2.0 * a.minimum_sigma_pixels);
        assert_eq!(a.corona_radii, b.corona_radii);
        assert_eq!(a.corona_hairs, b.corona_hairs);
        assert_eq!(a.corona_segments, b.corona_segments);
        assert_eq!(a.corona_seed, b.corona_seed);
        assert_eq!(a.edge_width, b.edge_width);
        assert_eq!(a.smooth_join, b.smooth_join);
        assert_eq!(native.render.aa, reference.render.aa);
        assert_eq!(reference.temporal_samples, 256);
        let a_plan = native.plan();
        let b_plan = reference.plan();
        assert_eq!(
            a_plan.eclipse_minimum_sigma_world_units,
            b_plan.eclipse_minimum_sigma_world_units
        );
        for radius in a.corona_radii {
            let native_sigma = radius.hypot(a.minimum_sigma_pixels * a_plan.world_units_per_pixel);
            let reference_sigma =
                radius.hypot(b.minimum_sigma_pixels * b_plan.world_units_per_pixel);
            assert_eq!(native_sigma, reference_sigma);
        }
        for (x, y) in [(0, 0), (8, 11), (16, 22)] {
            for (dx, dy) in [(0.25, 0.25), (0.75, 0.75)] {
                let actual = world(
                    &reference.camera,
                    &reference.render,
                    2.0 * (f64::from(x) + dx),
                    2.0 * (f64::from(y) + dy),
                );
                let expected = world(
                    &config.camera,
                    &config.render,
                    f64::from(crop.x) + f64::from(x) + dx,
                    f64::from(crop.y) + f64::from(y) + dy,
                );
                assert!((actual - expected).length() < 3e-15);
            }
        }
        let mut args = Args::try_parse_from([
            "atelier_convergence",
            "--config",
            "unused.json",
            "--orbit",
            "unused.bin",
            "--frame",
            "1670",
            "--crop",
            "271,137,17,23",
        ])
        .unwrap();
        assert!(!args.run);
        assert!(validate(&config, &args).is_ok());
        args.double_light_source_grid = true;
        assert!(validate(&config, &args).is_err());
    }

    #[test]
    fn doubled_shutter_cells_partition_the_same_raw_interval_and_keep_endpoint_holds() {
        let mut config = study();
        for frame in [0, 1670, 1801] {
            for sample in 0..16 {
                let native = sample_interval(&config, frame, 16, sample);
                let left = sample_interval(&config, frame, 32, sample * 2);
                let right = sample_interval(&config, frame, 32, sample * 2 + 1);
                assert_eq!(native[0], left[0]);
                assert_eq!(left[1], right[0]);
                assert_eq!(native[1], right[1]);
                let time = sample_time(&config, frame, 16, sample);
                assert!((time - (native[0] * 0.5 + native[1] * 0.5).clamp(0.0, 1.0)).abs() < 1e-15);
            }
        }
        assert!(sample_interval(&config, 0, 16, 0)[0] < 0.0);
        assert_eq!(sample_time(&config, 0, 16, 0), 0.0);
        assert!(sample_interval(&config, 1801, 16, 15)[1] > 1.0);
        assert_eq!(sample_time(&config, 1801, 16, 15), 1.0);
        config.shutter_fraction = 0.0;
        assert_eq!(sample_interval(&config, 1670, 32, 0), [1670.0 / 1801.0; 2]);
    }

    #[test]
    fn linear_box_downsampling_precedes_any_nonlinear_finishing() {
        let crop = Crop { x: 0, y: 0, width: 2, height: 1 };
        let values =
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0].map(|x| V3::new(x, x * 0.5, x * 0.25));
        let result = downsample(&values, crop, 2).unwrap();
        assert_eq!(result, [V3::new(5.0, 2.5, 1.25), V3::new(9.0, 4.5, 2.25)]);
    }

    #[test]
    fn metrics_detect_sparse_candidate_only_errors_without_background_dilution() {
        let background = V3::new(0.003, 0.002, 0.004);
        let mut relative = None;
        for count in [1, 100, 10000] {
            let crop = Crop { x: 10, y: 20, width: count, height: 1 };
            let mut native = vec![background; count as usize];
            let reference = native.clone();
            native[0] += V3::new(1.0, 1.0, 1.0);
            let result = compare(&native, &reference, crop, luminance(background)).unwrap();
            assert!((result.rgb_all_channels.max_absolute - 1.0).abs() < 1e-15);
            assert_eq!(result.active_pixels, 1);
            assert_eq!(result.worst_global_pixel, [10, 20]);
            if let Some(previous) = relative {
                assert_eq!(result.foreground_weighted_relative_luminance_rms, Some(previous));
            }
            relative = result.foreground_weighted_relative_luminance_rms;
            assert!((relative.unwrap() - 1.0).abs() < 1e-15);
            let identical = compare(&reference, &reference, crop, luminance(background)).unwrap();
            assert_eq!(identical.rgb_all_channels.rms, 0.0);
            assert_eq!(identical.foreground_weighted_relative_luminance_rms, None);
        }
    }
}
