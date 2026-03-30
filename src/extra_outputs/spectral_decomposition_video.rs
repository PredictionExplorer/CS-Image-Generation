//! Spectral decomposition video — starts with the full artwork, then peels
//! away wavelength groups from red to violet, revealing what each band hides.

use crate::extra_outputs::spectral_video_utils::{BinBuffers, NUM_GROUPS, BINS_PER_GROUP};
use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const DECOMP_DURATION_SECONDS: f64 = 15.0;
const HOLD_SECONDS: f64 = 2.0;
const FADE_OUT_SECONDS: f64 = 1.0;
const WAVE_FADE_SECONDS: f64 = 0.5;

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn weights_for_decomp_t(
    weights: &mut [f64; NUM_BINS],
    decomp_t: f64,
    active_duration: f64,
    wave_interval: f64,
) {
    let decomp_t = decomp_t.clamp(0.0, active_duration);
    let mut current_removing_group = (decomp_t / wave_interval).floor() as usize;
    if current_removing_group >= NUM_GROUPS {
        current_removing_group = NUM_GROUPS - 1;
    }
    let reverse_group = NUM_GROUPS - 1 - current_removing_group;
    let group_start = current_removing_group as f64 * wave_interval;

    for (bin, w) in weights.iter_mut().enumerate() {
        let g = bin / BINS_PER_GROUP;
        *w = if g > reverse_group {
            0.0
        } else if g < reverse_group {
            1.0
        } else {
            let linear = ((decomp_t - group_start) / WAVE_FADE_SECONDS).clamp(0.0, 1.0);
            1.0 - smoothstep(linear)
        };
    }
}

pub fn render_spectral_decomposition_video(
    bins: &BinBuffers,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (DECOMP_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral decomposition video ({:.0}s, {} groups × {} bins)...",
        DECOMP_DURATION_SECONDS,
        NUM_GROUPS,
        BINS_PER_GROUP
    );

    let decomp_end = DECOMP_DURATION_SECONDS - FADE_OUT_SECONDS;
    let active_duration = DECOMP_DURATION_SECONDS - HOLD_SECONDS - FADE_OUT_SECONDS;
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

                if t < HOLD_SECONDS {
                    weights.fill(1.0);
                } else if t >= decomp_end {
                    weights_for_decomp_t(
                        &mut weights,
                        active_duration,
                        active_duration,
                        wave_interval,
                    );
                    let fade_elapsed = t - decomp_end;
                    let fade_linear = (fade_elapsed / FADE_OUT_SECONDS).clamp(0.0, 1.0);
                    let fade_factor = 1.0 - smoothstep(fade_linear);
                    for w in &mut weights {
                        *w *= fade_factor;
                    }
                } else {
                    let decomp_t = t - HOLD_SECONDS;
                    weights_for_decomp_t(&mut weights, decomp_t, active_duration, wave_interval);
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

    info!("   Saved spectral decomposition video => {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ACTIVE: f64 = 12.0;
    const TEST_INTERVAL: f64 = TEST_ACTIVE / NUM_GROUPS as f64;

    #[test]
    fn test_decomp_t_zero_all_weights_one() {
        let mut w = [0.0; NUM_BINS];
        weights_for_decomp_t(&mut w, 0.0, TEST_ACTIVE, TEST_INTERVAL);
        for (bin, &val) in w.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-6,
                "at t=0 all bins should be 1.0, bin {bin} = {val}"
            );
        }
    }

    #[test]
    fn test_decomp_t_end_only_violet_partially_remains() {
        let mut w = [0.0; NUM_BINS];
        weights_for_decomp_t(&mut w, TEST_ACTIVE, TEST_ACTIVE, TEST_INTERVAL);
        for bin in BINS_PER_GROUP..NUM_BINS {
            assert!(
                w[bin] < 1e-6,
                "at t=active_duration, bin {bin} (group {}) should be ~0, got {}",
                bin / BINS_PER_GROUP,
                w[bin]
            );
        }
    }

    #[test]
    fn test_decomp_weights_all_in_unit_range() {
        let mut w = [0.0; NUM_BINS];
        for i in 0..=100 {
            let t = TEST_ACTIVE * i as f64 / 100.0;
            weights_for_decomp_t(&mut w, t, TEST_ACTIVE, TEST_INTERVAL);
            for (bin, &val) in w.iter().enumerate() {
                assert!(
                    (0.0..=1.0 + 1e-10).contains(&val),
                    "weight out of [0,1] at t={t}, bin {bin}: {val}"
                );
            }
        }
    }

    #[test]
    fn test_decomp_mid_removes_red_first() {
        let mut w = [0.0; NUM_BINS];
        let mid_t = TEST_INTERVAL * 0.9;
        weights_for_decomp_t(&mut w, mid_t, TEST_ACTIVE, TEST_INTERVAL);
        let last_group_start = (NUM_GROUPS - 1) * BINS_PER_GROUP;
        for bin in last_group_start..NUM_BINS {
            assert!(
                w[bin] < 0.5,
                "red group (bin {bin}) should be mostly removed at t={mid_t}, got {}",
                w[bin]
            );
        }
        assert!(
            w[0] > 0.9,
            "violet bin 0 should still be near 1.0 at t={mid_t}, got {}",
            w[0]
        );
    }

    #[test]
    fn test_decomp_negative_t_clamped() {
        let mut w = [0.0; NUM_BINS];
        weights_for_decomp_t(&mut w, -5.0, TEST_ACTIVE, TEST_INTERVAL);
        for &val in &w {
            assert!((val - 1.0).abs() < 1e-6, "negative t should clamp to 0 => all 1.0");
        }
    }

    #[test]
    fn test_smoothstep_boundaries() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decomposition_video_smoke() {
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
        let dir = std::env::temp_dir().join("spectral_test_decomposition");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("decomposition.mp4");
        let result = render_spectral_decomposition_video(&bins, path.to_str().unwrap(), true);
        assert!(result.is_ok(), "decomposition video render failed: {:?}", result.err());
    }
}
