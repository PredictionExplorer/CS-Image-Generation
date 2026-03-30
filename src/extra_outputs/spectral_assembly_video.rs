//! Spectral assembly video — bins arrive in 8 wavelength-grouped waves,
//! progressively reconstructing the full artwork from its spectral components.

use crate::extra_outputs::spectral_video_utils::{BinBuffers, NUM_GROUPS, BINS_PER_GROUP};
use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const ASSEMBLY_DURATION_SECONDS: f64 = 15.0;
const HOLD_SECONDS: f64 = 3.0;
const WAVE_FADE_SECONDS: f64 = 0.5;

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn render_spectral_assembly_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (ASSEMBLY_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral assembly video ({:.0}s, {} groups × {} bins)...",
        ASSEMBLY_DURATION_SECONDS,
        NUM_GROUPS,
        BINS_PER_GROUP
    );

    let active_duration = ASSEMBLY_DURATION_SECONDS - HOLD_SECONDS;
    let wave_interval = active_duration / NUM_GROUPS as f64;

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
            let mut weights = [0.0f64; NUM_BINS];

            for frame in 0..total_frames {
                let t = frame as f64 / fps as f64;

                if t >= active_duration {
                    weights.fill(1.0);
                } else {
                    let current_group = (t / wave_interval).floor() as usize;
                    let current_group = current_group.min(NUM_GROUPS - 1);
                    let group_start = current_group as f64 * wave_interval;

                    for (bin, w) in weights.iter_mut().enumerate() {
                        let g = bin / BINS_PER_GROUP;
                        *w = if g < current_group {
                            1.0
                        } else if g > current_group {
                            0.0
                        } else {
                            let linear = ((t - group_start) / WAVE_FADE_SECONDS).clamp(0.0, 1.0);
                            smoothstep(linear)
                        };
                    }
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

    info!("   Saved spectral assembly video => {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothstep_boundaries() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-10);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_clamps_out_of_range() {
        assert!((smoothstep(-1.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_monotonic() {
        let mut prev = 0.0f64;
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let val = smoothstep(t);
            assert!(val >= prev - 1e-10, "not monotonic at t={t}: {val} < {prev}");
            prev = val;
        }
    }

    #[test]
    fn test_smoothstep_output_in_unit_range() {
        for i in -10..=110 {
            let t = i as f64 / 100.0;
            let val = smoothstep(t);
            assert!((0.0..=1.0).contains(&val), "smoothstep({t}) = {val} out of [0,1]");
        }
    }

    #[test]
    fn test_assembly_video_smoke() {
        use crate::extra_outputs::spectral_video_utils::test_helpers::*;
        use crate::render::{
            SpectralRenderSettings, SpectralScene, accumulate_spectral_steps,
            apply_energy_density_shift, constants, default_accumulation_backend,
        };
        use crate::render::context::RenderContext;
        use crate::render::velocity_hdr;
        use crate::spectrum::NUM_BINS;
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
        let dir = std::env::temp_dir().join("spectral_test_assembly");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("assembly.mp4");
        let result = render_spectral_assembly_video(&bins, path.to_str().unwrap(), true);
        assert!(result.is_ok(), "assembly video render failed: {:?}", result.err());
    }
}
