//! Rendering timelapse — a short video showing the image building up from
//! nothing, stroke by stroke, as the spectral accumulation progresses.
//! No fade envelope; the drama is the image emerging from black.

use crate::render::context::RenderContext;
use crate::render::effects::{FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use crate::render::velocity_hdr;
use crate::render::{
    ChannelLevels, FinishOutputMode, SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    accumulate_spectral_steps, apply_energy_density_shift, build_effect_config_from_resolved,
    constants, create_video_from_frames_singlepass,
    default_accumulation_backend, quantize_display_buffer_to_16bit,
    tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const TIMELAPSE_SECONDS: f64 = 6.0;

pub fn render_timelapse(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!("Rendering timelapse video ({:.0}s)...", TIMELAPSE_SECONDS);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (TIMELAPSE_SECONDS * fps as f64) as usize;
    let total_steps = scene.step_count();

    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let effect_config =
        build_effect_config_from_resolved(resolved, settings.render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut accum_spd = vec![[0.0f32; NUM_BINS]; ctx.pixel_count()];
            let mut step_cursor = 0usize;

            for frame_idx in 0..total_frames {
                let progress = (frame_idx + 1) as f64 / total_frames as f64;
                let target_step = (progress * total_steps as f64) as usize;
                let target_step = target_step.min(total_steps);

                if step_cursor < target_step {
                    accumulate_spectral_steps(
                        &mut accum_spd,
                        None,
                        scene,
                        &ctx,
                        &velocity_calc,
                        step_cursor,
                        target_step,
                        settings.render_config.hdr_scale,
                        default_accumulation_backend(),
                    );
                    step_cursor = target_step;
                }

                let mut spd_snap = accum_spd.clone();
                apply_energy_density_shift(&mut spd_snap);

                let mut rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
                convert_spd_buffer_to_rgba(&spd_snap, &mut rgba, width as usize, height as usize);

                let fp = FrameParams { frame_number: frame_idx, density: None };
                let default_ctx = crate::post_effects::EffectContext::default();
                let trajectory = finish_pipeline
                    .process_trajectory(rgba, width as usize, height as usize, &fp, &default_ctx)
                    .expect("timelapse frame trajectory processing failed");

                let display = tonemap_to_display_buffer(&trajectory, levels);
                let final_display = finish_pipeline
                    .process_image(display, width as usize, height as usize, &fp, &default_ctx)
                    .expect("timelapse frame image processing failed");

                let buf_16bit = quantize_display_buffer_to_16bit(&final_display);
                let bytes = crate::utils::u16_slice_as_bytes(&buf_16bit);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved timelapse video => {}", output_path);
    Ok(())
}
