//! Spectral ripple video — radial waves expand from the center, momentarily
//! revealing each pixel's dominant wavelength as a monochromatic flash.

use crate::extra_outputs::spectral_video_utils::{BinBuffers, dominant_colors_from_shifted_spd};
use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::NUM_BINS;
use rayon::prelude::*;
use tracing::info;

const RIPPLE_DURATION_SECONDS: f64 = 6.0;
const NUM_WAVES: usize = 3;
const WAVE_WIDTH: f64 = 0.08;

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn render_spectral_ripple_video(
    bins: &BinBuffers,
    shifted_spd: &[[f64; NUM_BINS]],
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let width = bins.width;
    let height = bins.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (RIPPLE_DURATION_SECONDS * fps as f64) as usize;
    let tf = total_frames as f64;

    info!(
        "Rendering spectral ripple video ({:.0}s, {} waves)...",
        RIPPLE_DURATION_SECONDS, NUM_WAVES
    );

    let dominant_colors = dominant_colors_from_shifted_spd(shifted_spd);

    let mut composite_u16 = Vec::<u16>::new();
    bins.full_composite(&mut composite_u16);
    let composite_f: Vec<[f64; 3]> = composite_u16
        .par_chunks(3)
        .map(|c| {
            [
                c[0] as f64 / 65535.0,
                c[1] as f64 / 65535.0,
                c[2] as f64 / 65535.0,
            ]
        })
        .collect();

    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let max_radius = (center_x * center_x + center_y * center_y).sqrt();

    let options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    let pixel_count = bins.pixel_count;
    let width_usize = width as usize;

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = vec![0u16; pixel_count * 3];
            for frame in 0..total_frames {
                let t = frame as f64 / tf;
                frame_buf
                    .par_chunks_mut(3)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        let x = (i % width_usize) as f64 + 0.5;
                        let y = (i / width_usize) as f64 + 0.5;
                        let dx = x - center_x;
                        let dy = y - center_y;
                        let r = (dx * dx + dy * dy).sqrt();
                        let r_norm = (r / max_radius).clamp(0.0, 1.0);

                        let mut total_wave_alpha = 0.0f64;
                        for wave_idx in 0..NUM_WAVES {
                            let wave_center =
                                (t + wave_idx as f64 / NUM_WAVES as f64).fract();
                            let dist_to_wave =
                                ((r_norm - wave_center + 0.5).fract() - 0.5).abs();
                            let wave_alpha = if dist_to_wave < WAVE_WIDTH {
                                smoothstep(1.0 - dist_to_wave / WAVE_WIDTH)
                            } else {
                                0.0
                            };
                            total_wave_alpha = total_wave_alpha.max(wave_alpha);
                        }

                        let comp = composite_f[i];
                        let dom = dominant_colors[i];
                        let a = total_wave_alpha.clamp(0.0, 1.0);
                        let r_out = comp[0] * (1.0 - a) + dom[0] * a;
                        let g_out = comp[1] * (1.0 - a) + dom[1] * a;
                        let b_out = comp[2] * (1.0 - a) + dom[2] * a;
                        chunk[0] = (r_out.clamp(0.0, 1.0) * 65535.0).round() as u16;
                        chunk[1] = (g_out.clamp(0.0, 1.0) * 65535.0).round() as u16;
                        chunk[2] = (b_out.clamp(0.0, 1.0) * 65535.0).round() as u16;
                    });

                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                std::io::Write::write_all(out, bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved spectral ripple video => {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra_outputs::spectral_video_utils::dominant_colors_from_shifted_spd;

    #[test]
    fn test_smoothstep_boundaries() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-10);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_clamps() {
        assert!((smoothstep(-1.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dominant_colors_zero_energy_is_black() {
        let spd = vec![[0.0; NUM_BINS]; 4];
        let colors = dominant_colors_from_shifted_spd(&spd);
        assert_eq!(colors.len(), 4);
        for c in &colors {
            assert_eq!(*c, [0.0, 0.0, 0.0], "zero energy should be black");
        }
    }

    #[test]
    fn test_dominant_colors_single_bin_hot() {
        let mut spd = vec![[0.0; NUM_BINS]; 1];
        spd[0][32] = 1.0;
        let colors = dominant_colors_from_shifted_spd(&spd);
        assert_eq!(colors.len(), 1);
        let has_color = colors[0][0] > 0.0 || colors[0][1] > 0.0 || colors[0][2] > 0.0;
        assert!(has_color, "single-bin-hot pixel should have non-zero color");
    }

    #[test]
    fn test_dominant_colors_values_in_unit_range() {
        let mut spd = vec![[0.0; NUM_BINS]; 4];
        spd[0][10] = 5.0;
        spd[1][30] = 0.5;
        spd[2][50] = 100.0;
        let colors = dominant_colors_from_shifted_spd(&spd);
        for (i, c) in colors.iter().enumerate() {
            for (ch, &val) in c.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&val),
                    "pixel {i} channel {ch} = {val} out of [0,1]"
                );
            }
        }
    }

    #[test]
    fn test_ripple_video_smoke() {
        use crate::extra_outputs::spectral_video_utils::test_helpers::*;
        use crate::render::{
            SpectralRenderSettings, SpectralScene, accumulate_spectral_steps,
            apply_energy_density_shift, constants, default_accumulation_backend,
        };
        use crate::render::context::RenderContext;
        use crate::render::velocity_hdr;
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
        let dir = std::env::temp_dir().join("spectral_test_ripple");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ripple.mp4");
        let result = render_spectral_ripple_video(&bins, &accum_spd, path.to_str().unwrap(), true);
        assert!(result.is_ok(), "ripple video render failed: {:?}", result.err());
    }
}
