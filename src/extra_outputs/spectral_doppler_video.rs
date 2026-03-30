//! Spectral Doppler video — circular SPD shift oscillating between blueshift
//! and redshift, creating a smooth cosmic color transformation. Seamless loop.

use crate::render::{
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use rayon::prelude::*;
use tracing::info;

const DOPPLER_DURATION_SECONDS: f64 = 8.0;
const MAX_SHIFT_BINS: f64 = 10.0;
const DISPLAY_GAMMA: f64 = 2.2;

/// Apply a circular spectral shift to a single pixel's SPD and accumulate
/// tinted RGB.  Extracted for testability.
fn doppler_shift_pixel(
    spd: &[f64; NUM_BINS],
    tint_lut: &[(f64, f64, f64)],
    shift: f64,
) -> (f64, f64, f64) {
    let mut r = 0.0f64;
    let mut g = 0.0f64;
    let mut b = 0.0f64;
    for (bin, &(tr, tg, tb)) in tint_lut.iter().enumerate() {
        let src_f = (bin as f64 - shift).rem_euclid(NUM_BINS as f64);
        let lo = (src_f.floor() as usize).min(NUM_BINS - 1);
        let hi = (lo + 1) % NUM_BINS;
        let frac = src_f - src_f.floor();
        let val = spd[lo] * (1.0 - frac) + spd[hi] * frac;
        r += val * tr;
        g += val * tg;
        b += val * tb;
    }
    (r, g, b)
}

pub fn render_spectral_doppler_video(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (DOPPLER_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral Doppler video ({:.0}s, {} frames @ {} fps, max shift ±{:.1} bins)...",
        DOPPLER_DURATION_SECONDS, total_frames, fps, MAX_SHIFT_BINS
    );

    let tint_lut: Vec<(f64, f64, f64)> = (0..NUM_BINS)
        .map(|bin| wavelength_to_rgb(wavelength_nm_for_bin(bin)))
        .collect();

    let global_max = accum_spd
        .par_iter()
        .map(|spd| {
            let mut sum = 0.0f64;
            for bin in 0..NUM_BINS {
                let (tr, tg, tb) = tint_lut[bin];
                sum += spd[bin] * (tr + tg + tb);
            }
            sum
        })
        .reduce(|| 0.0f64, f64::max)
        .max(1e-10);

    let pixel_count = (width * height) as usize;
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
            let mut frame_buf = vec![0u16; pixel_count * 3];
            let tf = total_frames as f64;

            for frame in 0..total_frames {
                let t = frame as f64 / tf;
                let shift = MAX_SHIFT_BINS * (std::f64::consts::TAU * t).sin();

                frame_buf
                    .par_chunks_mut(3)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        let (r, g, b) = doppler_shift_pixel(&accum_spd[i], &tint_lut, shift);

                        chunk[0] = ((r / global_max).clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA)
                            * 65535.0)
                            .round() as u16;
                        chunk[1] = ((g / global_max).clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA)
                            * 65535.0)
                            .round() as u16;
                        chunk[2] = ((b / global_max).clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA)
                            * 65535.0)
                            .round() as u16;
                    });

                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }

            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved spectral Doppler video => {}", output_path);
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

    fn make_tint_lut() -> Vec<(f64, f64, f64)> {
        (0..NUM_BINS)
            .map(|bin| wavelength_to_rgb(wavelength_nm_for_bin(bin)))
            .collect()
    }

    fn make_flat_spd(value: f64) -> [f64; NUM_BINS] {
        [value; NUM_BINS]
    }

    #[test]
    fn test_doppler_zero_shift_identity() {
        let spd = make_flat_spd(1.0);
        let tint = make_tint_lut();
        let (r0, g0, b0) = doppler_shift_pixel(&spd, &tint, 0.0);
        assert!(r0 > 0.0 && g0 > 0.0 && b0 > 0.0, "zero shift should produce non-zero RGB");
    }

    #[test]
    fn test_doppler_full_cycle_shift_wraps_to_identity() {
        let mut spd = [0.0; NUM_BINS];
        for i in 0..NUM_BINS {
            spd[i] = (i as f64 + 1.0) * 0.1;
        }
        let tint = make_tint_lut();
        let (r0, g0, b0) = doppler_shift_pixel(&spd, &tint, 0.0);
        let (r_full, g_full, b_full) = doppler_shift_pixel(&spd, &tint, NUM_BINS as f64);
        assert!((r0 - r_full).abs() < 1e-10, "full cycle shift should wrap to identity");
        assert!((g0 - g_full).abs() < 1e-10);
        assert!((b0 - b_full).abs() < 1e-10);
    }

    #[test]
    fn test_doppler_bin_boundary_no_panic() {
        let spd = make_flat_spd(1.0);
        let tint = make_tint_lut();
        for shift in [0.0, 0.5, -0.5, 63.999, 64.0, -64.0, 128.0, -128.0, 1e10, -1e10] {
            let (r, g, b) = doppler_shift_pixel(&spd, &tint, shift);
            assert!(r.is_finite() && g.is_finite() && b.is_finite(),
                "shift {shift} produced non-finite RGB: ({r}, {g}, {b})");
        }
    }

    #[test]
    fn test_doppler_energy_approximately_conserved() {
        let mut spd = [0.0; NUM_BINS];
        for i in 0..NUM_BINS {
            spd[i] = (i as f64 + 1.0) * 0.05;
        }
        let tint = make_tint_lut();
        let (r0, g0, b0) = doppler_shift_pixel(&spd, &tint, 0.0);
        let energy_0 = r0 + g0 + b0;

        for shift in [3.7, -5.2, 15.0, 32.0] {
            let (r, g, b) = doppler_shift_pixel(&spd, &tint, shift);
            let energy = r + g + b;
            let ratio = energy / energy_0;
            assert!(
                (0.5..2.0).contains(&ratio),
                "shift {shift}: energy ratio {ratio:.3} is too far from 1.0"
            );
        }
    }

    #[test]
    fn test_doppler_zero_spd_produces_zero_rgb() {
        let spd = make_flat_spd(0.0);
        let tint = make_tint_lut();
        let (r, g, b) = doppler_shift_pixel(&spd, &tint, 5.0);
        assert_eq!(r, 0.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_doppler_video_smoke() {
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
        let dir = std::env::temp_dir().join("spectral_test_doppler");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("doppler.mp4");
        let result = render_spectral_doppler_video(&accum_spd, 64, 36, path.to_str().unwrap(), true);
        assert!(result.is_ok(), "doppler video render failed: {:?}", result.err());
    }
}
