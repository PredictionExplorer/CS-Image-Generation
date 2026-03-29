//! Spectral ripple video — radial waves expand from the center, momentarily
//! revealing each pixel's dominant wavelength as a monochromatic flash.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::apply_energy_density_shift;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    accumulate_spectral_steps, constants, create_video_from_frames_singlepass,
    default_accumulation_backend,
};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use rayon::prelude::*;
use tracing::info;

const RIPPLE_DURATION_SECONDS: f64 = 6.0;
const NUM_WAVES: usize = 3;
const WAVE_WIDTH: f64 = 0.08;
const DISPLAY_GAMMA: f64 = 2.2;

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Same as [`BinBuffers::from_spd`], but `accum_spd` already has
/// [`apply_energy_density_shift`] applied (avoids double-shifting).
fn bin_buffers_from_shifted_spd(
    accum_spd: Vec<[f64; NUM_BINS]>,
    width: u32,
    height: u32,
) -> BinBuffers {
    let pixel_count = (width * height) as usize;
    let buffers: Vec<Vec<[f64; 3]>> = (0..NUM_BINS)
        .into_par_iter()
        .map(|bin| {
            let wavelength = wavelength_nm_for_bin(bin);
            let (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength);

            let max_val: f64 = accum_spd
                .iter()
                .map(|spd| spd[bin])
                .fold(0.0f64, f64::max)
                .max(1e-10);

            let mut buf = vec![[0.0f64; 3]; pixel_count];
            for (px, spd) in buf.iter_mut().zip(accum_spd.iter()) {
                let normalized = (spd[bin] / max_val).clamp(0.0, 1.0);
                px[0] = (normalized * tint_r).powf(1.0 / DISPLAY_GAMMA);
                px[1] = (normalized * tint_g).powf(1.0 / DISPLAY_GAMMA);
                px[2] = (normalized * tint_b).powf(1.0 / DISPLAY_GAMMA);
            }
            buf
        })
        .collect();

    BinBuffers { buffers, pixel_count, width, height }
}

fn dominant_colors_from_shifted_spd(shifted_spd: &[[f64; NUM_BINS]]) -> Vec<[f64; 3]> {
    let max_total_energy = shifted_spd
        .par_iter()
        .map(|spd| spd.iter().sum::<f64>())
        .reduce(|| 0.0f64, f64::max)
        .max(1e-10);
    let log_denom = (1.0 + max_total_energy).ln();

    shifted_spd
        .par_iter()
        .map(|spd| {
            let total_energy: f64 = spd.iter().sum();
            let dominant = spd
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let wl = wavelength_nm_for_bin(dominant);
            let (tr, tg, tb) = wavelength_to_rgb(wl);
            let brightness = (1.0 + total_energy).ln() / log_denom;
            let r = (tr * brightness).powf(1.0 / DISPLAY_GAMMA);
            let g = (tg * brightness).powf(1.0 / DISPLAY_GAMMA);
            let b = (tb * brightness).powf(1.0 / DISPLAY_GAMMA);
            [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
        })
        .collect()
}

pub fn render_spectral_ripple_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering spectral ripple video ({:.0}s, {} waves)...",
        RIPPLE_DURATION_SECONDS, NUM_WAVES
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (RIPPLE_DURATION_SECONDS * fps as f64) as usize;
    let tf = total_frames as f64;

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

    let mut shifted_spd = accum_spd.clone();
    apply_energy_density_shift(&mut shifted_spd);

    let dominant_colors = dominant_colors_from_shifted_spd(&shifted_spd);
    let bins = bin_buffers_from_shifted_spd(shifted_spd, width, height);

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

    let width_usize = width as usize;

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = vec![0u16; ctx.pixel_count() * 3];
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
