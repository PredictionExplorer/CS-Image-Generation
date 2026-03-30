//! Cinemagraph / living poster — a high-resolution subtle looping animation
//! from the tail end of the simulation where visual change is minimal,
//! producing a serene, almost-still moving image for digital frames.

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

const LIVING_SECONDS: f64 = 4.0;
const CROSSFADE_RATIO: f64 = 0.15;

pub fn render_cinemagraph(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    _fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!("Rendering cinemagraph / living poster ({:.0}s)...", LIVING_SECONDS);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (LIVING_SECONDS * fps as f64) as usize;
    let total_steps = scene.step_count();

    let tail_start = total_steps.saturating_sub(total_steps / 50);
    let tail_steps = total_steps - tail_start;
    let frame_interval = (tail_steps / total_frames).max(1);

    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let effect_config =
        build_effect_config_from_resolved(resolved, settings.render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let mut accum_spd = vec![[0.0f32; NUM_BINS]; ctx.pixel_count()];
    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        tail_start,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let mut raw_frames: Vec<Vec<u16>> = Vec::with_capacity(total_frames);
    let mut step = tail_start;

    for frame_idx in 0..total_frames {
        let next_step = (step + frame_interval).min(total_steps);
        if step < next_step {
            accumulate_spectral_steps(
                &mut accum_spd,
                None,
                scene,
                &ctx,
                &velocity_calc,
                step,
                next_step,
                settings.render_config.hdr_scale,
                default_accumulation_backend(),
            );
            step = next_step;
        }

        let mut spd_snap = accum_spd.clone();
        apply_energy_density_shift(&mut spd_snap);

        let mut rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
        convert_spd_buffer_to_rgba(&spd_snap, &mut rgba, width as usize, height as usize);

        let fp = FrameParams { frame_number: frame_idx, density: None };
        let default_ctx = crate::post_effects::EffectContext::default();
        let trajectory = finish_pipeline
            .process_trajectory(rgba, width as usize, height as usize, &fp, &default_ctx)
            .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;

        let display = tonemap_to_display_buffer(&trajectory, levels);
        let final_display = finish_pipeline
            .process_image(display, width as usize, height as usize, &fp, &default_ctx)
            .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;

        raw_frames.push(quantize_display_buffer_to_16bit(&final_display));
    }

    let fade_len = ((total_frames as f64 * CROSSFADE_RATIO) as usize).max(2);
    let output_count = total_frames.saturating_sub(fade_len);
    let looped = apply_crossfade(&raw_frames, fade_len, output_count);

    let options = VideoEncodingOptions {
        codec: "libvpx-vp9".to_string(),
        preset: String::new(),
        crf: 24,
        bitrate: String::new(),
        pixel_format: "yuv420p".to_string(),
        input_pixel_format: "rgb48le".to_string(),
        extra_args: vec![
            "-row-mt".to_string(), "1".to_string(),
            "-b:v".to_string(), "0".to_string(),
        ],
    };

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            for frame in &looped {
                let bytes = crate::utils::u16_slice_as_bytes(frame);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved cinemagraph => {}", output_path);
    Ok(())
}

fn apply_crossfade(frames: &[Vec<u16>], fade_len: usize, output_count: usize) -> Vec<Vec<u16>> {
    let n = frames.len();
    let mut output = Vec::with_capacity(output_count);

    for i in 0..output_count {
        if i < fade_len {
            let t = i as f64 / fade_len as f64;
            let blend_tail = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
            let tail_idx = n - fade_len + i;

            let blended: Vec<u16> = frames[i]
                .iter()
                .zip(frames[tail_idx].iter())
                .map(|(&start, &end)| {
                    let v = end as f64 * blend_tail + start as f64 * (1.0 - blend_tail);
                    v.round().clamp(0.0, 65535.0) as u16
                })
                .collect();
            output.push(blended);
        } else {
            output.push(frames[i].clone());
        }
    }
    output
}
