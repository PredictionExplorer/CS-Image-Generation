//! Spectral slit-scan video — each row shows a different spectral bin,
//! creating a flowing rainbow curtain as the mapping scrolls vertically.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions, accumulate_spectral_steps,
    constants, create_video_from_frames_singlepass, default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const SLIT_SCAN_DURATION_SECONDS: f64 = 8.0;

pub fn render_spectral_slit_scan_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering spectral slit-scan video ({:.0}s, vertical scroll)...",
        SLIT_SCAN_DURATION_SECONDS
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (SLIT_SCAN_DURATION_SECONDS * fps as f64).max(1.0) as usize;

    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let total_steps = scene.step_count();
    accumulate_spectral_steps(
        &mut accum_spd,
        scene,
        &ctx,
        &velocity_calc,
        0,
        total_steps,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let bins = BinBuffers::from_spd(accum_spd, width, height);
    let pixel_count = bins.pixel_count;
    let w = width as usize;
    let h = height as usize;

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
            let mut bin_per_pixel = vec![0.0f64; pixel_count];
            let mut frame_buf = vec![0u16; pixel_count * 3];
            for frame in 0..total_frames {
                let offset = frame as f64 * NUM_BINS as f64 / total_frames as f64;
                for y in 0..h {
                    let bin_f = (y as f64 * NUM_BINS as f64 / h as f64 + offset)
                        .rem_euclid(NUM_BINS as f64);
                    let row_base = y * w;
                    for x in 0..w {
                        bin_per_pixel[row_base + x] = bin_f;
                    }
                }
                bins.per_pixel_bin_select(&bin_per_pixel, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                std::io::Write::write_all(out, bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved spectral slit-scan video => {}", output_path);
    Ok(())
}
