//! Spectral Doppler video — circular SPD shift oscillating between blueshift
//! and redshift, creating a smooth cosmic color transformation. Seamless loop.

use crate::render::apply_energy_density_shift;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions, accumulate_spectral_steps,
    constants, create_video_from_frames_singlepass, default_accumulation_backend,
};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use rayon::prelude::*;
use tracing::info;

const DOPPLER_DURATION_SECONDS: f64 = 8.0;
const MAX_SHIFT_BINS: f64 = 10.0;
const DISPLAY_GAMMA: f64 = 2.2;

pub fn render_spectral_doppler_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (DOPPLER_DURATION_SECONDS * fps as f64) as usize;

    info!(
        "Rendering spectral Doppler video ({:.0}s, {} frames @ {} fps, max shift ±{:.1} bins)...",
        DOPPLER_DURATION_SECONDS, total_frames, fps, MAX_SHIFT_BINS
    );

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

    apply_energy_density_shift(&mut accum_spd);

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

    let pixel_count = ctx.pixel_count();
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
                        let spd = &accum_spd[i];
                        let mut r = 0.0f64;
                        let mut g = 0.0f64;
                        let mut b = 0.0f64;

                        for (bin, &(tr, tg, tb)) in tint_lut.iter().enumerate() {
                            let src_f = (bin as f64 - shift).rem_euclid(NUM_BINS as f64);
                            let lo = src_f.floor() as usize;
                            let hi = (lo + 1) % NUM_BINS;
                            let frac = src_f - src_f.floor();
                            let val = spd[lo] * (1.0 - frac) + spd[hi] * frac;

                            r += val * tr;
                            g += val * tg;
                            b += val * tb;
                        }

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
