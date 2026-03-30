//! Seamless loop clip — a short (3-5 second) perfectly looping video
//! optimized for NFT marketplace thumbnails, social media, and wallets.

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

const LOOP_DURATION_SECONDS: f64 = 4.0;
const CROSSFADE_FRAMES: usize = 36;

pub fn render_loop_clip(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_mp4: &str,
    output_webm: Option<&str>,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!("Rendering seamless loop clip ({:.0}s)...", LOOP_DURATION_SECONDS);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let loop_frames = (LOOP_DURATION_SECONDS * fps as f64) as usize;
    let total_steps = scene.step_count();

    let clip_start = total_steps.saturating_sub(total_steps / 3);
    let clip_steps = total_steps - clip_start;
    let frame_interval = (clip_steps / loop_frames).max(1);
    let actual_frames = clip_steps / frame_interval;

    let display_frames = render_display_frames(
        scene,
        levels,
        settings,
        clip_start,
        frame_interval,
        actual_frames,
    )?;

    let fade = CROSSFADE_FRAMES.min(actual_frames / 3);
    let output_count = actual_frames.saturating_sub(fade);
    if output_count == 0 {
        return Err(crate::render::error::RenderError::InvalidConfig(
            "Not enough frames for loop clip".to_string(),
        ));
    }

    let looped = apply_crossfade(&display_frames, fade, output_count);

    write_frames_to_video(&looped, width, height, fps, output_mp4, fast_encode)?;
    info!("   Saved loop clip => {}", output_mp4);

    if let Some(webm_path) = output_webm {
        write_frames_to_webm(&looped, width, height, fps, webm_path)?;
        info!("   Saved loop clip => {}", webm_path);
    }

    Ok(())
}

fn render_display_frames(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    start_step: usize,
    frame_interval: usize,
    frame_count: usize,
) -> crate::render::error::Result<Vec<Vec<u16>>> {
    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let effect_config =
        build_effect_config_from_resolved(resolved, settings.render_config, FinishOutputMode::Video);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        start_step,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let mut frames = Vec::with_capacity(frame_count);
    let mut step = start_step;

    for frame_idx in 0..frame_count {
        let next_step = (step + frame_interval).min(scene.step_count());
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

        let mut spd_snapshot = accum_spd.clone();
        apply_energy_density_shift(&mut spd_snapshot);

        let mut rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
        convert_spd_buffer_to_rgba(&spd_snapshot, &mut rgba, width as usize, height as usize);

        let frame_params = FrameParams { frame_number: frame_idx, density: None };
        let default_ctx = crate::post_effects::EffectContext::default();
        let trajectory_pixels = finish_pipeline
            .process_trajectory(rgba, width as usize, height as usize, &frame_params, &default_ctx)
            .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;

        let display = tonemap_to_display_buffer(&trajectory_pixels, levels);
        let final_display = finish_pipeline
            .process_image(display, width as usize, height as usize, &frame_params, &default_ctx)
            .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;

        frames.push(quantize_display_buffer_to_16bit(&final_display));
    }

    Ok(frames)
}

fn apply_crossfade(frames: &[Vec<u16>], fade_len: usize, output_count: usize) -> Vec<Vec<u16>> {
    let n = frames.len();
    let mut output = Vec::with_capacity(output_count);

    for i in 0..output_count {
        if i < fade_len {
            let t = i as f64 / fade_len as f64;
            let blend_start = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
            let tail_idx = n - fade_len + i;

            let blended: Vec<u16> = frames[i]
                .iter()
                .zip(frames[tail_idx].iter())
                .map(|(&start, &end)| {
                    let v = end as f64 * blend_start + start as f64 * (1.0 - blend_start);
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

fn write_frames_to_video(
    frames: &[Vec<u16>],
    width: u32,
    height: u32,
    fps: u32,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
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
            for frame in frames {
                out.write_all(crate::utils::u16_slice_as_bytes(frame))
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )
}

fn write_frames_to_webm(
    frames: &[Vec<u16>],
    width: u32,
    height: u32,
    fps: u32,
    output_path: &str,
) -> crate::render::error::Result<()> {
    let options = VideoEncodingOptions {
        codec: "libvpx-vp9".to_string(),
        preset: String::new(),
        crf: 30,
        bitrate: String::new(),
        pixel_format: "yuv420p".to_string(),
        input_pixel_format: "rgb48le".to_string(),
        extra_args: vec![
            "-row-mt".to_string(),
            "1".to_string(),
            "-b:v".to_string(),
            "0".to_string(),
        ],
    };

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            for frame in frames {
                out.write_all(crate::utils::u16_slice_as_bytes(frame))
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frames(count: usize, pixel_count: usize, base_val: u16) -> Vec<Vec<u16>> {
        (0..count)
            .map(|i| vec![base_val.wrapping_add(i as u16); pixel_count])
            .collect()
    }

    #[test]
    fn test_crossfade_zero_fade_is_identity() {
        let frames = make_frames(10, 6, 1000);
        let result = apply_crossfade(&frames, 0, 10);
        assert_eq!(result.len(), 10);
        for (i, frame) in result.iter().enumerate() {
            assert_eq!(frame, &frames[i]);
        }
    }

    #[test]
    fn test_crossfade_output_length() {
        let frames = make_frames(20, 6, 100);
        let result = apply_crossfade(&frames, 5, 15);
        assert_eq!(result.len(), 15);
    }

    #[test]
    fn test_crossfade_non_fade_frames_unchanged() {
        let frames = make_frames(20, 6, 500);
        let fade = 4;
        let output_count = 16;
        let result = apply_crossfade(&frames, fade, output_count);

        for i in fade..output_count {
            assert_eq!(result[i], frames[i], "frame {i} outside fade zone should be unchanged");
        }
    }

    #[test]
    fn test_crossfade_blends_boundary_frames() {
        let n = 12;
        let px = 3;
        let fade = 3;
        let output_count = n - fade; // 9
        let mut frames: Vec<Vec<u16>> = vec![vec![1000u16; px]; n];
        // Set tail frames (indices n-fade..n = 9,10,11) to a very different value
        for tail_i in 0..fade {
            frames[n - fade + tail_i] = vec![50000u16; px];
        }
        // Set start frames (indices 0..fade = 0,1,2) to a low value
        for frame in frames.iter_mut().take(fade) {
            *frame = vec![100u16; px];
        }

        let result = apply_crossfade(&frames, fade, output_count);
        // Frame i=1: blends frames[1] (100) with frames[n-fade+1=10] (50000)
        // blend_start = 0.5*(1+cos(pi*1/3)) = 0.5*(1+0.5) = 0.75
        // result = 50000*0.75 + 100*0.25 = 37525
        assert!(result[1][0] > 100, "blended frame should exceed start value");
        assert!(result[1][0] < 50000, "blended frame should be below tail value");
    }

    #[test]
    fn test_crossfade_no_panic_with_high_values() {
        let frames = make_frames(20, 6, 60000);
        let result = apply_crossfade(&frames, 5, 15);
        assert_eq!(result.len(), 15);
        assert!(!result[0].is_empty());
    }
}
