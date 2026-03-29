//! Spectral breathing video — sinusoidal modulation wave travels across all
//! 64 bins, creating a fluid, organic color pulse. Seamless loop.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    accumulate_spectral_steps, constants, create_video_from_frames_singlepass,
    default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use std::io::Write;
use tracing::info;

const BREATHING_DURATION_SECONDS: f64 = 6.0;
const BASE_WEIGHT: f64 = 0.4;
const AMPLITUDE: f64 = 0.6;

pub fn render_spectral_breathing_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering spectral breathing video ({:.0}s, {} fps)...",
        BREATHING_DURATION_SECONDS,
        constants::DEFAULT_VIDEO_FPS
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (BREATHING_DURATION_SECONDS * fps as f64) as usize;
    let total_frames = total_frames.max(1);

    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];

    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        scene,
        &ctx,
        &velocity_calc,
        0,
        scene.step_count(),
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let bins = BinBuffers::from_spd(accum_spd, width, height);

    let options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    let angle_per_bin = std::f64::consts::TAU / NUM_BINS as f64;

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = Vec::new();
            for frame in 0..total_frames {
                let t = frame as f64 / total_frames as f64;
                let mut weights = [0.0f64; NUM_BINS];
                for (bin, w) in weights.iter_mut().enumerate() {
                    let phase = bin as f64 * angle_per_bin;
                    *w = BASE_WEIGHT + AMPLITUDE * (std::f64::consts::TAU * t + phase).sin();
                }
                bins.weighted_blend(&weights, &mut frame_buf);
                Write::write_all(out, crate::utils::u16_slice_as_bytes(&frame_buf))
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved spectral breathing video => {}", output_path);
    Ok(())
}
