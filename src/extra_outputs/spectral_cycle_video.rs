//! Spectral cycle video — continuous sweep through all 64 wavelength bins,
//! smoothly interpolating between adjacent bins for a seamless prism-like loop.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const CYCLE_DURATION_SECONDS: f64 = 12.0;

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
        &options,
    )?;

    info!("   Saved spectral cycle video => {}", output_path);
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
        let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
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
