//! Spectral breathing video — sinusoidal modulation wave travels across all
//! 64 bins, creating a fluid, organic color pulse. Seamless loop.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::NUM_BINS;
use std::io::Write;
use tracing::info;

const BREATHING_DURATION_SECONDS: f64 = 6.0;
const BASE_WEIGHT: f64 = 0.4;
const AMPLITUDE: f64 = 0.6;

pub fn render_spectral_breathing_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (BREATHING_DURATION_SECONDS * fps as f64).max(1.0) as usize;

    info!(
        "Rendering spectral breathing video ({:.0}s, {} fps)...",
        BREATHING_DURATION_SECONDS, fps
    );

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
    fn test_breathing_video_smoke() {
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
        let dir = std::env::temp_dir().join("spectral_test_breathing");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("breathing.mp4");
        let result = render_spectral_breathing_video(&bins, path.to_str().unwrap(), true);
        assert!(result.is_ok(), "breathing video render failed: {:?}", result.err());
    }
}
