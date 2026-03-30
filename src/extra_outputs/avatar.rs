//! Animated avatar — a small square looping animation for profile pictures.
//! Detects the visual center of mass and crops a square region, then renders
//! a short animation encoded as animated WebP via ffmpeg.

use crate::render::context::RenderContext;
use crate::render::effects::{FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use crate::render::velocity_hdr;
use crate::render::{
    ChannelLevels, FinishOutputMode, SpectralRenderSettings, SpectralScene,
    accumulate_spectral_steps, apply_energy_density_shift, build_effect_config_from_resolved,
    constants, default_accumulation_backend, quantize_display_buffer_to_16bit,
    tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use std::io::Write;
use std::process::{Command, Stdio};
use tracing::info;

const AVATAR_SIZE: u32 = 512;
const AVATAR_FRAMES: usize = 60;
const AVATAR_FPS: u32 = 15;

pub fn render_animated_avatar(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering animated avatar ({}x{}, {} frames)...", AVATAR_SIZE, AVATAR_SIZE, AVATAR_FRAMES);

    let resolved = settings.resolved_config;
    let orig_width = resolved.width as usize;
    let orig_height = resolved.height as usize;

    let energy_center = find_energy_center(scene, settings);
    let crop = compute_square_crop(orig_width, orig_height, energy_center);

    let total_steps = scene.step_count();
    let clip_start = total_steps * 2 / 3;
    let clip_steps = total_steps - clip_start;
    let frame_interval = (clip_steps / AVATAR_FRAMES).max(1);

    let ctx = RenderContext::new(
        resolved.width,
        resolved.height,
        scene.positions,
        settings.aspect_correction,
    );
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
        clip_start,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let bytes_per_frame = AVATAR_SIZE as usize * AVATAR_SIZE as usize * 6;
    let mut all_frame_bytes = Vec::with_capacity(AVATAR_FRAMES * bytes_per_frame);

    let mut step = clip_start;
    for frame_idx in 0..AVATAR_FRAMES {
        let next_step = (step + frame_interval).min(total_steps);
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

        let mut spd_snap = accum_spd.clone();
        apply_energy_density_shift(&mut spd_snap);

        let mut rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
        convert_spd_buffer_to_rgba(&spd_snap, &mut rgba, orig_width, orig_height);

        let frame_params = FrameParams { frame_number: frame_idx, density: None };
        let default_ctx = crate::post_effects::EffectContext::default();
        let processed = finish_pipeline
            .process_trajectory(rgba, orig_width, orig_height, &frame_params, &default_ctx)
            .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;
        let display = tonemap_to_display_buffer(&processed, levels);
        let final_display = finish_pipeline
            .process_image(display, orig_width, orig_height, &frame_params, &default_ctx)
            .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;
        let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

        let cropped = crop_and_resize_16bit(&buf_16bit, orig_width, orig_height, &crop, AVATAR_SIZE);
        all_frame_bytes.extend_from_slice(crate::utils::u16_slice_as_bytes(&cropped));
    }

    encode_webp(&all_frame_bytes, AVATAR_SIZE, AVATAR_SIZE, AVATAR_FPS, output_path)?;
    info!("   Saved animated avatar => {}", output_path);
    Ok(())
}

struct CropRegion {
    x: usize,
    y: usize,
    size: usize,
}

fn find_energy_center(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
) -> (f64, f64) {
    let width = settings.resolved_config.width as usize;
    let height = settings.resolved_config.height as usize;
    let ctx = RenderContext::new(
        settings.resolved_config.width,
        settings.resolved_config.height,
        scene.positions,
        settings.aspect_correction,
    );

    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let sample_steps = scene.step_count() / 10;
    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        sample_steps,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let mut total_energy = 0.0f64;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;

    for (idx, spd) in accum_spd.iter().enumerate() {
        let energy: f64 = spd.iter().sum();
        let x = (idx % width) as f64;
        let y = (idx / width) as f64;
        cx += x * energy;
        cy += y * energy;
        total_energy += energy;
    }

    if total_energy > 0.0 {
        (cx / total_energy / width as f64, cy / total_energy / height as f64)
    } else {
        (0.5, 0.5)
    }
}

fn compute_square_crop(width: usize, height: usize, center: (f64, f64)) -> CropRegion {
    let size = width.min(height);
    let cx = (center.0 * width as f64) as isize;
    let cy = (center.1 * height as f64) as isize;
    let half = size as isize / 2;
    let x = (cx - half).clamp(0, (width - size) as isize) as usize;
    let y = (cy - half).clamp(0, (height - size) as isize) as usize;
    CropRegion { x, y, size }
}

fn crop_and_resize_16bit(
    buf: &[u16],
    src_width: usize,
    src_height: usize,
    crop: &CropRegion,
    target_size: u32,
) -> Vec<u16> {
    let ts = target_size as usize;
    let mut out = vec![0u16; ts * ts * 3];

    for ty in 0..ts {
        for tx in 0..ts {
            let sx_f = tx as f64 * crop.size as f64 / ts as f64;
            let sy_f = ty as f64 * crop.size as f64 / ts as f64;
            let sx = (crop.x + sx_f as usize).min(src_width - 1);
            let sy = (crop.y + sy_f as usize).min(src_height - 1);
            let src_idx = (sy * src_width + sx) * 3;
            let dst_idx = (ty * ts + tx) * 3;
            if src_idx + 2 < buf.len() {
                out[dst_idx] = buf[src_idx];
                out[dst_idx + 1] = buf[src_idx + 1];
                out[dst_idx + 2] = buf[src_idx + 2];
            }
        }
    }
    out
}

fn encode_webp(
    frame_bytes: &[u8],
    width: u32,
    height: u32,
    fps: u32,
    output_path: &str,
) -> crate::render::error::Result<()> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "-s", &format!("{}x{}", width, height),
            "-r", &fps.to_string(),
            "-i", "-",
            "-vf", &format!("format=rgba,scale={}:{}", width, height),
            "-c:v", "libwebp_anim",
            "-quality", "80",
            "-loop", "0",
            "-lossless", "0",
            output_path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(crate::render::error::RenderError::VideoEncoding)?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(frame_bytes);
        let _ = stdin.flush();
        drop(stdin);
    }

    let status = child.wait().map_err(crate::render::error::RenderError::VideoEncoding)?;
    if !status.success() {
        return Err(crate::render::error::RenderError::InvalidConfig(
            "WebP encoding failed; ffmpeg may lack libwebp_anim support".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_square_crop_centered() {
        let crop = compute_square_crop(1920, 1080, (0.5, 0.5));
        assert_eq!(crop.size, 1080);
        assert!(crop.x + crop.size <= 1920);
        assert!(crop.y + crop.size <= 1080);
    }

    #[test]
    fn test_compute_square_crop_clamps_to_bounds() {
        let crop = compute_square_crop(1920, 1080, (0.0, 0.0));
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
        assert!(crop.x + crop.size <= 1920);
        assert!(crop.y + crop.size <= 1080);

        let crop2 = compute_square_crop(1920, 1080, (1.0, 1.0));
        assert!(crop2.x + crop2.size <= 1920);
        assert!(crop2.y + crop2.size <= 1080);
    }

    #[test]
    fn test_compute_square_crop_square_source() {
        let crop = compute_square_crop(500, 500, (0.5, 0.5));
        assert_eq!(crop.size, 500);
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
    }

    #[test]
    fn test_crop_and_resize_output_length() {
        let crop = CropRegion { x: 0, y: 0, size: 4 };
        let buf = vec![100u16; 4 * 4 * 3];
        let result = crop_and_resize_16bit(&buf, 4, 4, &crop, 2);
        assert_eq!(result.len(), 2 * 2 * 3);
    }

    #[test]
    fn test_crop_and_resize_identity() {
        let size = 4;
        let crop = CropRegion { x: 0, y: 0, size };
        let mut buf = vec![0u16; size * size * 3];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = (i * 100) as u16;
        }

        let result = crop_and_resize_16bit(&buf, size, size, &crop, size as u32);
        assert_eq!(result.len(), buf.len());
        for (i, (&got, &expected)) in result.iter().zip(buf.iter()).enumerate() {
            assert_eq!(got, expected, "pixel {i} differs in identity crop");
        }
    }
}
