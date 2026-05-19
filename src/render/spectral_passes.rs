//! Spectral accumulation passes for histogram, video-frame, and final-frame output.

use super::batch_drawing::{
    BatchDrawParams, draw_triangle_batch_spectral_rows, prepare_triangle_vertices,
};
use super::context::{FramingConfig, RenderContext};
use super::effect_config::build_effect_config_from_resolved;
use super::effects::{FinishEffectPipeline, FrameParams, try_convert_spd_buffer_to_rgba};
use super::error::{RenderError, Result};
use super::histogram::HistogramData;
use super::tonemapping::{quantize_display_buffer_to_16bit, tonemap_to_display_buffer};
use super::types::ChannelLevels;
use super::{FinishOutputMode, SpectralRenderSettings, SpectralScene, constants, velocity_hdr};
use crate::spectrum::NUM_BINS;
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use tracing::{debug, info};

fn framing_from_resolved(
    resolved: &super::randomizable_config::ResolvedEffectConfig,
) -> FramingConfig {
    FramingConfig {
        zoom: if resolved.composition_zoom > 0.0 { resolved.composition_zoom } else { 1.0 },
        offset_x: resolved.composition_offset_x,
        offset_y: resolved.composition_offset_y,
    }
}

/// Apply energy density wavelength shift to spectral buffer.
/// Hot regions (high energy) shift toward red, cool regions stay blue.
fn apply_energy_density_shift(accum_spd: &mut [[f64; NUM_BINS]]) {
    use constants::{ENERGY_DENSITY_SHIFT_STRENGTH, ENERGY_DENSITY_SHIFT_THRESHOLD};

    accum_spd.par_iter_mut().for_each(|spd| {
        let total_energy: f64 = spd.iter().sum();
        if total_energy < ENERGY_DENSITY_SHIFT_THRESHOLD {
            return;
        }

        let excess_energy = total_energy - ENERGY_DENSITY_SHIFT_THRESHOLD;
        let shift_amount = (excess_energy * ENERGY_DENSITY_SHIFT_STRENGTH).min(1.0);

        let mut shifted_spd = *spd;
        for i in (1..NUM_BINS).rev() {
            shifted_spd[i] = spd[i] * (1.0 - shift_amount) + spd[i - 1] * shift_amount;
        }
        shifted_spd[0] = spd[0] * (1.0 - shift_amount);

        *spd = shifted_spd;
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AccumulationBackend {
    ParallelScanlines,
    #[cfg(test)]
    SerialReference,
}

#[inline]
fn default_accumulation_backend() -> AccumulationBackend {
    AccumulationBackend::ParallelScanlines
}

fn checkpoint_steps(total_steps: usize, frame_interval: usize) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }

    let mut checkpoints = Vec::new();
    let mut checkpoint = frame_interval;
    while checkpoint < total_steps {
        if checkpoint > 0 {
            checkpoints.push(checkpoint);
        }
        checkpoint += frame_interval;
    }

    let final_step = total_steps - 1;
    if checkpoints.last().copied() != Some(final_step) {
        checkpoints.push(final_step);
    }

    checkpoints
}

pub(crate) struct AccumulationParams<'a> {
    pub(crate) scene: SpectralScene<'a>,
    pub(crate) ctx: &'a RenderContext,
    pub(crate) velocity_calc: &'a velocity_hdr::VelocityHdrCalculator<'a>,
    pub(crate) step_start: usize,
    pub(crate) step_end: usize,
    pub(crate) hdr_scale: f64,
}

fn accumulate_spectral_steps_into_rows(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
) {
    if params.step_start >= params.step_end || row_start >= row_end {
        return;
    }

    let triangle_alphas = params.scene.triangle_alphas();
    for step in params.step_start..params.step_end {
        let vertices = prepare_triangle_vertices(
            params.scene.positions,
            params.scene.colors,
            &triangle_alphas,
            step,
            params.ctx,
        );

        let hdr_mult_01 = params.velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = params.velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = params.velocity_calc.compute_segment_multiplier(step, 2, 0);

        draw_triangle_batch_spectral_rows(
            accum_spd,
            &BatchDrawParams {
                width: params.ctx.width,
                height: params.ctx.height,
                row_start,
                row_end,
                vertices,
                hdr_multipliers: [hdr_mult_01, hdr_mult_12, hdr_mult_20],
                hdr_scale: params.hdr_scale,
            },
        );
    }
}

pub(crate) fn accumulate_spectral_steps(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    backend: AccumulationBackend,
) {
    if params.step_start >= params.step_end || accum_spd.is_empty() {
        return;
    }

    match backend {
        AccumulationBackend::ParallelScanlines => {
            let ctx = params.ctx;
            let band_count = ctx.height_usize.min(rayon::current_num_threads().max(1));
            if band_count <= 1 {
                accumulate_spectral_steps_into_rows(accum_spd, params, 0, ctx.height_usize);
                return;
            }

            let rows_per_band = ctx.height_usize.div_ceil(band_count);
            let pixels_per_band = ctx.width_usize * rows_per_band;
            accum_spd.par_chunks_mut(pixels_per_band).enumerate().for_each(|(band_idx, band)| {
                let row_start = band_idx * rows_per_band;
                let row_end = row_start + band.len() / ctx.width_usize;
                accumulate_spectral_steps_into_rows(band, params, row_start, row_end);
            });
        }
        #[cfg(test)]
        AccumulationBackend::SerialReference => {
            accumulate_spectral_steps_into_rows(accum_spd, params, 0, params.ctx.height_usize);
        }
    }
}

fn pass_1_build_histogram_spectral_with_backend(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<HistogramData> {
    scene.validate()?;
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction, .. } = settings;
    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::try_new_with_framing(
        width,
        height,
        scene.positions,
        aspect_correction,
        framing_from_resolved(resolved_config),
    )?;
    let pixel_count = ctx.try_pixel_count()?;
    let histogram_capacity =
        pixel_count.checked_mul(10).ok_or_else(|| RenderError::InvalidScene {
            reason: format!("histogram capacity overflows usize for {width}x{height}"),
        })?;
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; pixel_count];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); pixel_count];
    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(&effect_config);
    let mut histogram = HistogramData::with_capacity(histogram_capacity);

    let total_steps = scene.step_count();
    let checkpoints = checkpoint_steps(total_steps, frame_interval);
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);
    let mut step_start = 0;

    for &checkpoint_step in &checkpoints {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 1, mode = "spectral", "Histogram pass progress");
        }

        accumulate_spectral_steps(
            &mut accum_spd,
            &AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start,
                step_end: checkpoint_step + 1,
                hdr_scale: render_config.hdr_scale,
            },
            backend,
        );

        apply_energy_density_shift(&mut accum_spd);
        try_convert_spd_buffer_to_rgba(
            &accum_spd,
            &mut accum_rgba,
            width as usize,
            height as usize,
            render_config.sat_boost,
            render_config.dispersion_boost,
        )?;

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let trajectory_proxy = finish_pipeline.process_trajectory(
            rgba_buffer,
            width as usize,
            height as usize,
            &frame_params,
        )?;
        accum_rgba.clear();
        accum_rgba.resize(pixel_count, (0.0, 0.0, 0.0, 0.0));

        histogram.reserve(pixel_count);
        for &(r, g, b, _a) in &trajectory_proxy {
            histogram.push(r, g, b);
        }

        step_start = checkpoint_step + 1;
    }

    info!("   pass 1 (spectral histogram): 100% done");
    Ok(histogram)
}

/// Pass 1: gather global histogram for final color leveling (spectral).
///
/// # Errors
///
/// Returns an error if trajectory-stage effects fail while preparing histogram
/// samples.
pub fn pass_1_build_histogram_spectral(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
) -> Result<HistogramData> {
    pass_1_build_histogram_spectral_with_backend(
        scene,
        frame_interval,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn pass_1_build_histogram_spectral_serial_reference(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
) -> Result<HistogramData> {
    pass_1_build_histogram_spectral_with_backend(
        scene,
        frame_interval,
        settings,
        AccumulationBackend::SerialReference,
    )
}

/// Bundled parameters for `pass_2_write_frames_spectral` and its backend variants.
///
/// Groups the non-closure state that every pass-2 call site must supply,
/// keeping the function signatures under clippy's argument-count threshold.
pub struct Pass2Params<'a> {
    /// Trajectory positions, colours, and per-body alphas to draw.
    pub scene: SpectralScene<'a>,
    /// Simulation steps between emitted video frames.
    pub frame_interval: usize,
    /// Per-channel black/white levels from pass 1 histogram analysis.
    pub levels: &'a ChannelLevels,
    /// Resolved effects, HDR scale, bloom, noise seed, and aspect handling.
    pub settings: SpectralRenderSettings<'a>,
    /// Receives the final frame as 16-bit RGB when the pass completes.
    pub last_frame_out: &'a mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    /// When true, blends consecutive display frames to reduce temporal noise.
    pub enable_temporal_smoothing: bool,
    /// Scratch buffer for per-pixel spectral power distributions (reused across checkpoints).
    pub accum_spd: &'a mut Vec<[f64; NUM_BINS]>,
}

#[cfg(test)]
pub(crate) fn pass_2_write_frames_spectral_serial_reference(
    params: Pass2Params<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(
        params,
        frame_sink,
        AccumulationBackend::SerialReference,
    )
}

fn pass_2_write_frames_spectral_with_backend(
    params: Pass2Params<'_>,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
    backend: AccumulationBackend,
) -> Result<()> {
    let Pass2Params {
        scene,
        frame_interval,
        levels,
        settings,
        last_frame_out,
        enable_temporal_smoothing,
        accum_spd,
    } = params;
    scene.validate()?;
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction } = settings;
    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::try_new_with_framing(
        width,
        height,
        scene.positions,
        aspect_correction,
        framing_from_resolved(resolved_config),
    )?;
    let pixel_count = ctx.try_pixel_count()?;
    accum_spd.resize(pixel_count, [0.0f64; NUM_BINS]);
    for s in accum_spd.iter_mut() {
        *s = [0.0; NUM_BINS];
    }
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); pixel_count];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(&effect_config);

    let total_steps = scene.step_count();
    let checkpoints = checkpoint_steps(total_steps, frame_interval);
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    use crate::post_effects::{TemporalSmoothing, TemporalSmoothingConfig};
    let temporal_smoother = if enable_temporal_smoothing {
        Some(TemporalSmoothing::new(TemporalSmoothingConfig {
            blend_factor: 0.10,
            alpha_threshold: 0.01,
        }))
    } else {
        None
    };
    let mut step_start = 0;

    for &checkpoint_step in &checkpoints {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 2, mode = "spectral", "Render pass progress");
        }

        accumulate_spectral_steps(
            accum_spd,
            &AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start,
                step_end: checkpoint_step + 1,
                hdr_scale: render_config.hdr_scale,
            },
            backend,
        );

        apply_energy_density_shift(accum_spd);
        try_convert_spd_buffer_to_rgba(
            accum_spd,
            &mut accum_rgba,
            width as usize,
            height as usize,
            render_config.sat_boost,
            render_config.dispersion_boost,
        )?;

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let mut trajectory_pixels = finish_pipeline.process_trajectory(
            rgba_buffer,
            width as usize,
            height as usize,
            &frame_params,
        )?;

        let display_buffer =
            tonemap_to_display_buffer(&trajectory_pixels, levels, render_config.aces_tweak);

        // Reclaim the trajectory buffer's allocation back into accum_rgba.
        // It will be fully overwritten by SPD conversion next iteration,
        // so we just need the capacity -- no need to clear or resize.
        trajectory_pixels.resize(pixel_count, (0.0, 0.0, 0.0, 0.0));
        accum_rgba = trajectory_pixels;
        let smoothed_display = match &temporal_smoother {
            Some(smoother) => smoother.process_frame(display_buffer),
            None => display_buffer,
        };

        let final_display = finish_pipeline.process_image(
            smoothed_display,
            width as usize,
            height as usize,
            &frame_params,
        )?;
        let buf_16bit = quantize_display_buffer_to_16bit(&final_display);
        let buf_bytes: &[u8] = bytemuck::cast_slice(&buf_16bit);

        frame_sink(buf_bytes)?;
        if checkpoint_step + 1 == total_steps {
            *last_frame_out = ImageBuffer::from_raw(width, height, buf_16bit);
        }

        step_start = checkpoint_step + 1;
    }

    info!("   pass 2 (spectral render): 100% done");
    Ok(())
}

/// Pass 2: render frames with spectral accumulation and feed 16-bit bytes to `frame_sink`.
///
/// # Errors
///
/// Returns an error if frame rendering, post-processing, or `frame_sink` fails.
pub fn pass_2_write_frames_spectral(
    params: Pass2Params<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(params, frame_sink, default_accumulation_backend())
}

/// Render the fully accumulated final frame without writing intermediate video frames.
///
/// This is the correct preview path for still-image QA because it matches the final
/// accumulated composition instead of an early timeline slice.
///
/// # Errors
///
/// Returns an error if accumulation, post-processing, or image construction fails.
pub fn render_final_frame_spectral(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_final_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn render_final_frame_spectral_serial_reference(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_final_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        AccumulationBackend::SerialReference,
    )
}

fn render_final_frame_spectral_with_backend(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    scene.validate()?;
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction } = settings;
    info!("   Rendering final accumulated frame (preview mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::try_new_with_framing(
        width,
        height,
        scene.positions,
        aspect_correction,
        framing_from_resolved(resolved_config),
    )?;
    let pixel_count = ctx.try_pixel_count()?;
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; pixel_count];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); pixel_count];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(&effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        &AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: total_steps,
            hdr_scale: render_config.hdr_scale,
        },
        backend,
    );

    apply_energy_density_shift(&mut accum_spd);
    try_convert_spd_buffer_to_rgba(
        &accum_spd,
        &mut accum_rgba,
        width as usize,
        height as usize,
        render_config.sat_boost,
        render_config.dispersion_boost,
    )?;

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let preview_frame_number = total_steps.saturating_sub(1) / frame_interval;
    let frame_params = FrameParams { frame_number: preview_frame_number, density: None };
    let trajectory_pixels = finish_pipeline.process_trajectory(
        accum_rgba,
        width as usize,
        height as usize,
        &frame_params,
    )?;

    let display_buffer =
        tonemap_to_display_buffer(&trajectory_pixels, levels, render_config.aces_tweak);
    let final_display = finish_pipeline.process_image(
        display_buffer,
        width as usize,
        height as usize,
        &frame_params,
    )?;
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| RenderError::ImageEncoding {
        reason: "Failed to create 16-bit image buffer".into(),
    })
}

/// Render the first timeline slice only for tests.
#[cfg(test)]
pub(crate) fn render_single_frame_spectral(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_single_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn render_single_frame_spectral_serial_reference(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_single_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        AccumulationBackend::SerialReference,
    )
}

#[cfg(test)]
fn render_single_frame_spectral_with_backend(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    scene.validate()?;
    let SpectralRenderSettings { resolved_config, render_config, aspect_correction } = settings;
    info!("   Rendering first timeline slice only (legacy test mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::try_new_with_framing(
        width,
        height,
        scene.positions,
        aspect_correction,
        framing_from_resolved(resolved_config),
    )?;
    let pixel_count = ctx.try_pixel_count()?;
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; pixel_count];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); pixel_count];

    let effect_config =
        build_effect_config_from_resolved(resolved_config, render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(&effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let first_frame_step = frame_interval;

    accumulate_spectral_steps(
        &mut accum_spd,
        &AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: first_frame_step + 1,
            hdr_scale: render_config.hdr_scale,
        },
        backend,
    );

    apply_energy_density_shift(&mut accum_spd);
    try_convert_spd_buffer_to_rgba(
        &accum_spd,
        &mut accum_rgba,
        width as usize,
        height as usize,
        render_config.sat_boost,
        render_config.dispersion_boost,
    )?;

    let frame_params = FrameParams { frame_number: 0, density: None };
    let trajectory_pixels = finish_pipeline.process_trajectory(
        accum_rgba,
        width as usize,
        height as usize,
        &frame_params,
    )?;

    let display_buffer =
        tonemap_to_display_buffer(&trajectory_pixels, levels, render_config.aces_tweak);
    let final_display = finish_pipeline.process_image(
        display_buffer,
        width as usize,
        height as usize,
        &frame_params,
    )?;
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| RenderError::ImageEncoding {
        reason: "Failed to create 16-bit image buffer".into(),
    })
}
