//! Cinematic "reveal" video — a short (8-10 second) polished video designed
//! for social media: fade from black, dramatic orbital reveal, hold, fade out.

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
use rayon::prelude::*;
use tracing::info;

const REVEAL_SECONDS: f64 = 8.0;
const FADE_IN_SECONDS: f64 = 1.5;
const FADE_OUT_SECONDS: f64 = 2.0;

pub fn render_reveal_video(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!("Rendering cinematic reveal video ({:.0}s)...", REVEAL_SECONDS);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (REVEAL_SECONDS * fps as f64) as usize;
    let total_steps = scene.step_count();
    let frame_interval = (total_steps / total_frames).max(1);

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
            let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
            let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
            let mut step_cursor = 0usize;

            for frame_idx in 0..total_frames {
                let target_step = ((frame_idx + 1) * frame_interval).min(total_steps);
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
                accum_rgba.fill((0.0, 0.0, 0.0, 0.0));
                convert_spd_buffer_to_rgba(
                    &spd_snap,
                    &mut accum_rgba,
                    width as usize,
                    height as usize,
                );

                let fp = FrameParams { frame_number: frame_idx, density: None };
                let default_ctx = crate::post_effects::EffectContext::default();
                let rgba_buf = std::mem::take(&mut accum_rgba);
                let trajectory = finish_pipeline
                    .process_trajectory(rgba_buf, width as usize, height as usize, &fp, &default_ctx)
                    .expect("reveal frame trajectory processing failed");
                accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

                let display = tonemap_to_display_buffer(&trajectory, levels);
                let final_display = finish_pipeline
                    .process_image(display, width as usize, height as usize, &fp, &default_ctx)
                    .expect("reveal frame image processing failed");

                let envelope = reveal_envelope(frame_idx, total_frames, fps);
                let faded: Vec<(f64, f64, f64, f64)> = final_display
                    .par_iter()
                    .map(|&(r, g, b, a)| (r * envelope, g * envelope, b * envelope, a))
                    .collect();

                let buf_16bit = quantize_display_buffer_to_16bit(&faded);
                let bytes = crate::utils::u16_slice_as_bytes(&buf_16bit);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved reveal video => {}", output_path);
    Ok(())
}

fn reveal_envelope(frame: usize, total: usize, fps: u32) -> f64 {
    let t = frame as f64 / fps as f64;
    let duration = total as f64 / fps as f64;

    let fade_in = (t / FADE_IN_SECONDS).clamp(0.0, 1.0);
    let fade_in_smooth = fade_in * fade_in * (3.0 - 2.0 * fade_in);

    let time_left = duration - t;
    let fade_out = (time_left / FADE_OUT_SECONDS).clamp(0.0, 1.0);
    let fade_out_smooth = fade_out * fade_out * (3.0 - 2.0 * fade_out);

    fade_in_smooth * fade_out_smooth
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_FPS: u32 = 60;
    const TEST_TOTAL: usize = 480; // 8 seconds at 60fps

    #[test]
    fn test_reveal_envelope_near_zero_at_start() {
        assert!(reveal_envelope(0, TEST_TOTAL, TEST_FPS) < 0.01);
    }

    #[test]
    fn test_reveal_envelope_near_one_at_midpoint() {
        let mid = TEST_TOTAL / 2;
        let val = reveal_envelope(mid, TEST_TOTAL, TEST_FPS);
        assert!(val > 0.9, "mid-frame should be near 1.0, got {val}");
    }

    #[test]
    fn test_reveal_envelope_near_zero_at_end() {
        let val = reveal_envelope(TEST_TOTAL - 1, TEST_TOTAL, TEST_FPS);
        assert!(val < 0.05, "last frame should be near 0, got {val}");
    }

    #[test]
    fn test_reveal_envelope_all_in_unit_range() {
        for frame in 0..TEST_TOTAL {
            let val = reveal_envelope(frame, TEST_TOTAL, TEST_FPS);
            assert!((0.0..=1.0).contains(&val), "frame {frame} out of range: {val}");
        }
    }

    #[test]
    fn test_reveal_envelope_monotonic_during_fade_in() {
        let fade_in_frames = (FADE_IN_SECONDS * TEST_FPS as f64) as usize;
        let mut prev = 0.0f64;
        for frame in 0..fade_in_frames {
            let val = reveal_envelope(frame, TEST_TOTAL, TEST_FPS);
            assert!(val >= prev - 1e-10, "not monotonic at frame {frame}: {val} < {prev}");
            prev = val;
        }
    }
}
