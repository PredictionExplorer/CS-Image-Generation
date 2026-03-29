//! Narrowband scan video — a tight Gaussian window sweeps across the 64-bin
//! spectrum, isolating narrow wavelength slices like a tunable astronomical filter.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions, accumulate_spectral_steps,
    constants, create_video_from_frames_singlepass, default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const SCAN_DURATION_SECONDS: f64 = 10.0;
/// Gaussian standard deviation in bins (FWHM ≈ 2.355σ ≈ 5.9 bins, ~30 nm).
const GAUSSIAN_SIGMA: f64 = 2.5;

pub fn render_spectral_narrowband_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering narrowband scan video ({:.0}s, σ={} bins)...",
        SCAN_DURATION_SECONDS, GAUSSIAN_SIGMA
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (SCAN_DURATION_SECONDS * fps as f64) as usize;

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
            let mut frame_buf = Vec::<u16>::new();
            let n = NUM_BINS as f64;
            for frame in 0..total_frames {
                let center = frame as f64 * n / total_frames as f64;
                let mut weights = [0.0f64; NUM_BINS];
                for bin in 0..NUM_BINS {
                    let diff = (bin as f64 - center).rem_euclid(n);
                    let d = diff.min(n - diff);
                    weights[bin] = (-0.5 * (d / GAUSSIAN_SIGMA).powi(2)).exp();
                }
                bins.weighted_blend(&weights, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved narrowband scan video => {}", output_path);
    Ok(())
}
