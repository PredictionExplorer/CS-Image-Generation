//! Spectral cycle videos — continuous sweeps through all 64 wavelength bins.
//!
//! The original forward cycle plus five variants: reverse, ping-pong, eased,
//! radial wave, and complementary dual-cursor.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const CYCLE_DURATION_SECONDS: f64 = 12.0;

fn encoding_options(fast_encode: bool) -> VideoEncodingOptions {
    if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    }
}

// ── Original forward cycle ──────────────────────────────────────────────────

pub fn render_spectral_cycle_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (CYCLE_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral cycle video ({:.0}s, {} bins)...",
        CYCLE_DURATION_SECONDS, NUM_BINS
    );

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = Vec::<u16>::new();
            for frame in 0..total_frames {
                let bin_f = frame as f64 * NUM_BINS as f64 / total_frames as f64;
                bins.lerp_bins(bin_f, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &encoding_options(fast_encode),
    )?;

    info!("   Saved spectral cycle video => {}", output_path);
    Ok(())
}

// ── Reverse cycle (red → violet) ────────────────────────────────────────────

pub fn render_spectral_cycle_reverse_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (CYCLE_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral cycle reverse video ({:.0}s, {} bins)...",
        CYCLE_DURATION_SECONDS, NUM_BINS
    );

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = Vec::<u16>::new();
            for frame in 0..total_frames {
                let bin_f = NUM_BINS as f64
                    - (frame as f64 * NUM_BINS as f64 / total_frames as f64);
                bins.lerp_bins(bin_f, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &encoding_options(fast_encode),
    )?;

    info!("   Saved spectral cycle reverse video => {}", output_path);
    Ok(())
}

// ── Ping-pong cycle (violet → red → violet) ────────────────────────────────

pub fn render_spectral_cycle_pingpong_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (CYCLE_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral cycle ping-pong video ({:.0}s, {} bins)...",
        CYCLE_DURATION_SECONDS, NUM_BINS
    );

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = Vec::<u16>::new();
            let max_bin = (NUM_BINS - 1) as f64;
            for frame in 0..total_frames {
                let t = frame as f64 / total_frames as f64;
                let bin_f = if t < 0.5 {
                    t * 2.0 * max_bin
                } else {
                    (1.0 - (t - 0.5) * 2.0) * max_bin
                };
                bins.lerp_bins(bin_f, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &encoding_options(fast_encode),
    )?;

    info!("   Saved spectral cycle ping-pong video => {}", output_path);
    Ok(())
}

// ── Ease cycle (cosine-eased, lingers at violet & red extremes) ─────────────

pub fn render_spectral_cycle_ease_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (CYCLE_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral cycle ease video ({:.0}s, {} bins)...",
        CYCLE_DURATION_SECONDS, NUM_BINS
    );

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = Vec::<u16>::new();
            for frame in 0..total_frames {
                let t = frame as f64 / total_frames as f64;
                let bin_f = (1.0 - (t * std::f64::consts::TAU).cos()) * 0.5
                    * NUM_BINS as f64;
                bins.lerp_bins(bin_f, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &encoding_options(fast_encode),
    )?;

    info!("   Saved spectral cycle ease video => {}", output_path);
    Ok(())
}

// ── Radial wave cycle (concentric color rings from center) ──────────────────

const RADIAL_SPREAD: f64 = 24.0;

pub fn render_spectral_cycle_radial_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (CYCLE_DURATION_SECONDS * fps as f64) as usize;
    let pixel_count = (width * height) as usize;

    info!(
        "Rendering spectral cycle radial video ({:.0}s, {} bins)...",
        CYCLE_DURATION_SECONDS, NUM_BINS
    );

    let cx = (width as f64 - 1.0) * 0.5;
    let cy = (height as f64 - 1.0) * 0.5;
    let max_dist = (cx * cx + cy * cy).sqrt();

    let mut dist_map = vec![0.0f64; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            dist_map[(y * width + x) as usize] = (dx * dx + dy * dy).sqrt() / max_dist;
        }
    }

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = Vec::<u16>::new();
            let mut bin_map = vec![0.0f64; pixel_count];
            for frame in 0..total_frames {
                let base_bin = frame as f64 * NUM_BINS as f64 / total_frames as f64;
                for (i, dist) in dist_map.iter().enumerate() {
                    bin_map[i] = base_bin - dist * RADIAL_SPREAD;
                }
                bins.per_pixel_bin_select(&bin_map, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &encoding_options(fast_encode),
    )?;

    info!("   Saved spectral cycle radial video => {}", output_path);
    Ok(())
}

// ── Complementary split cycle (two cursors half-spectrum apart) ──────────────

pub fn render_spectral_cycle_complementary_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (CYCLE_DURATION_SECONDS * fps as f64) as usize;
    let half = NUM_BINS as f64 * 0.5;

    info!(
        "Rendering spectral cycle complementary video ({:.0}s, {} bins)...",
        CYCLE_DURATION_SECONDS, NUM_BINS
    );

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut buf_a = Vec::<u16>::new();
            let mut buf_b = Vec::<u16>::new();
            let mut frame_buf = Vec::<u16>::new();
            for frame in 0..total_frames {
                let bin_f = frame as f64 * NUM_BINS as f64 / total_frames as f64;
                bins.lerp_bins(bin_f, &mut buf_a);
                bins.lerp_bins(bin_f + half, &mut buf_b);

                frame_buf.resize(buf_a.len(), 0u16);
                for (i, out_px) in frame_buf.iter_mut().enumerate() {
                    let mixed = (buf_a[i] as u32 + buf_b[i] as u32) / 2;
                    *out_px = mixed.min(65535) as u16;
                }

                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &encoding_options(fast_encode),
    )?;

    info!(
        "   Saved spectral cycle complementary video => {}",
        output_path
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra_outputs::spectral_video_utils::test_helpers::*;
    use crate::render::{
        SpectralRenderSettings, SpectralScene, accumulate_spectral_steps,
        apply_energy_density_shift, constants, default_accumulation_backend,
    };
    use crate::render::context::RenderContext;
    use crate::render::velocity_hdr;
    use crate::spectrum::NUM_BINS;

    #[test]
    fn test_cycle_video_smoke() {
        let (positions, colors, body_alphas) = make_test_scene(200);
        let resolved = make_test_resolved_config(64, 36);
        let render_config = make_test_render_config(resolved.hdr_scale);
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let ctx = RenderContext::new(64, 36, scene.positions, settings.aspect_correction);
        let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, constants::DEFAULT_DT);
        let mut accum_spd = vec![[0.0f32; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(
            &mut accum_spd, None, scene, &ctx, &velocity_calc,
            0, scene.step_count(), settings.render_config.hdr_scale,
            default_accumulation_backend(),
        );
        apply_energy_density_shift(&mut accum_spd);
        let bins = BinBuffers::from_shifted_spd(&accum_spd, 64, 36);
        let dir = std::env::temp_dir().join("spectral_test_cycle");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cycle.mp4");
        let result = render_spectral_cycle_video(&bins, path.to_str().unwrap(), true);
        assert!(result.is_ok(), "cycle video render failed: {:?}", result.err());
    }
}
