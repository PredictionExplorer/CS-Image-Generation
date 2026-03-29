//! Spectral aurora video — narrow wavelength curtains drift across the image
//! at different speeds, blending additively like northern lights.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    accumulate_spectral_steps, constants, create_video_from_frames_singlepass,
    default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use rayon::prelude::*;
use tracing::info;

const AURORA_DURATION_SECONDS: f64 = 10.0;
const NUM_CURTAINS: usize = 6;

struct Curtain {
    center_bin: f64,
    width: f64,
    speed: f64,
    wave_amp: f64,
    wave_freq: f64,
}

const CURTAINS: [Curtain; NUM_CURTAINS] = [
    Curtain { center_bin: 5.0, width: 3.0, speed: 0.3, wave_amp: 0.06, wave_freq: 1.5 },
    Curtain { center_bin: 15.0, width: 2.5, speed: -0.2, wave_amp: 0.04, wave_freq: 2.0 },
    Curtain { center_bin: 25.0, width: 4.0, speed: 0.15, wave_amp: 0.08, wave_freq: 1.0 },
    Curtain { center_bin: 38.0, width: 3.5, speed: -0.25, wave_amp: 0.05, wave_freq: 2.5 },
    Curtain { center_bin: 50.0, width: 2.0, speed: 0.35, wave_amp: 0.07, wave_freq: 1.8 },
    Curtain { center_bin: 58.0, width: 3.0, speed: -0.18, wave_amp: 0.03, wave_freq: 3.0 },
];

pub fn render_spectral_aurora_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering spectral aurora video ({:.0}s, {} curtains)...",
        AURORA_DURATION_SECONDS, NUM_CURTAINS
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (AURORA_DURATION_SECONDS * fps as f64).max(1.0) as usize;

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
    let pixel_count = bins.pixel_count;

    let curtain_profiles: Vec<[f64; NUM_BINS]> = CURTAINS
        .iter()
        .map(|c| {
            let mut profile = [0.0f64; NUM_BINS];
            for (bin, val) in profile.iter_mut().enumerate() {
                let d = bin as f64 - c.center_bin;
                *val = (-0.5 * (d / c.width).powi(2)).exp();
            }
            profile
        })
        .collect();

    let options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    let width_f = width as f64;
    let height_f = height as f64;
    let inv_tf = 1.0 / total_frames as f64;

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf = vec![0u16; pixel_count * 3];
            for frame in 0..total_frames {
                let t = frame as f64 * inv_tf;

                frame_buf
                    .par_chunks_mut(3)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        let x = (i % width as usize) as f64 / width_f;
                        let y = (i / width as usize) as f64 / height_f;

                        let mut r = 0.0f64;
                        let mut g = 0.0f64;
                        let mut b = 0.0f64;

                        for (ci, curtain) in CURTAINS.iter().enumerate() {
                            let drift = curtain.speed * (std::f64::consts::TAU * t).sin();
                            let wave = curtain.wave_amp
                                * (curtain.wave_freq * std::f64::consts::TAU * y
                                    + std::f64::consts::TAU * t)
                                    .sin();
                            let cx = (x + drift + wave).fract().abs();
                            let spatial = (-0.5 * ((cx - 0.5) / 0.18).powi(2)).exp();

                            #[allow(clippy::needless_range_loop)]
                            for bin in 0..NUM_BINS {
                                let w = spatial * curtain_profiles[ci][bin];
                                if w > 0.001 {
                                    r += bins.buffers[bin][i][0] * w;
                                    g += bins.buffers[bin][i][1] * w;
                                    b += bins.buffers[bin][i][2] * w;
                                }
                            }
                        }

                        chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
                        chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
                        chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
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

    info!("   Saved spectral aurora video => {}", output_path);
    Ok(())
}
