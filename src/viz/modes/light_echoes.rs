//! V32 `light-echoes` -- Three Boats on a Dark Pond.
//!
//! Each body emits continuous wavefronts into its own 2D wave field; motion
//! Doppler-compresses ripples ahead and stretches them behind, and the three
//! palette-locked frequencies weave colored interference moires. The video
//! shows the live field; the still integrates the energy envelope over the
//! whole run.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch};
use crate::render::context::RenderContext;
use crate::render::{
    ImageBuffer, VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass,
};
use crate::spectrum::linear_rec2020_to_display_p3;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::raster::upsample_bicubic;
use crate::viz::common::wave::{Oscillator, WaveField};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Oscillator frequency range (cycles per video second), hue-locked.
const FREQUENCY_RANGE: (f32, f32) = (2.2, 4.8);
/// Target maximum body Mach number.
const MAX_MACH: f32 = 0.7;
/// Amplitude exponent on normalized speed.
const AMPLITUDE_EXPONENT: f64 = 0.7;
/// Field damping for ~9 s visual persistence (amplitude e-fold at 2/gamma).
const DAMPING: f32 = 2.0 / 9.0;
/// Deep-water base color (`OKLab`).
const BASE: (f64, f64, f64) = (0.16, 0.0, -0.015);
/// Field-to-chroma swing.
const CHROMA_SWING: f64 = 0.13;
/// Field-to-lightness swing.
const LIGHT_SWING: f64 = 0.09;
/// Display gamma.
const DISPLAY_GAMMA: f64 = 2.2;

/// The light echoes mode.
pub struct LightEchoes;

impl VizMode for LightEchoes {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("light-echoes").expect("light-echoes is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("light-echoes skipped: empty trajectory");
            return Ok(());
        }
        let grid_w = (((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16)) as usize;
        let grid_h = (((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16)) as usize;
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let relative = grid_w.min(grid_h) as f32 / 1117.0;

        let render_ctx = RenderContext::new(
            grid_w as u32,
            grid_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );

        // Per-frame body pixel positions and speeds (px per video second).
        let body_px: Vec<[(f32, f32); 3]> = (0..frame_count)
            .map(|frame| {
                let step = ((frame + 1) * steps / frame_count).min(steps - 1);
                std::array::from_fn(|body| {
                    let position = ctx.positions[body][step];
                    render_ctx.to_pixel(position.x, position.y)
                })
            })
            .collect();
        let mut speeds_px: Vec<[f32; 3]> = vec![[0.0; 3]; frame_count];
        for frame in 1..frame_count {
            for body in 0..3 {
                let (x0, y0) = body_px[frame - 1][body];
                let (x1, y1) = body_px[frame][body];
                speeds_px[frame][body] = (x1 - x0).hypot(y1 - y0) * 60.0;
            }
        }
        if frame_count > 1 {
            speeds_px[0] = speeds_px[1];
        }

        // Wave speed: fastest body stays subsonic at ~MAX_MACH (p98 of
        // sampled speeds so single-frame spikes cannot explode the substep
        // count; exceedances are logged).
        let mut pooled: Vec<f32> =
            speeds_px.iter().flat_map(|frame| frame.iter().copied()).collect();
        pooled.sort_by(f32::total_cmp);
        let p98 = pooled[((pooled.len() - 1) as f64 * 0.98) as usize];
        let wave_speed = (p98 / MAX_MACH).clamp(60.0 * relative, 1500.0 * relative);
        let peak = pooled.last().copied().unwrap_or(0.0);
        if peak > wave_speed * MAX_MACH * 1.5 {
            info!(
                "   light-echoes: brief supersonic spikes (peak {peak:.0} px/s, c {wave_speed:.0})"
            );
        }

        // Hue-locked frequencies and palette directions.
        let mut frequencies = [0.0f32; 3];
        let mut hue_directions = [(0.0f64, 0.0f64); 3];
        for body in 0..3 {
            let (l, a, b) = ctx.mean_color(body);
            let (_, _, hue) = oklab_to_oklch(l, a, b);
            frequencies[body] =
                FREQUENCY_RANGE.0 + (FREQUENCY_RANGE.1 - FREQUENCY_RANGE.0) * (hue / 360.0) as f32;
            let radians = hue.to_radians();
            hue_directions[body] = (radians.cos(), radians.sin());
        }
        info!(
            "   light-echoes: {grid_w}x{grid_h} fields, c {wave_speed:.0} px/s, f {:?}",
            frequencies
        );

        let speed_window = ctx.kinematics().speed_window();
        let mut fields: [WaveField; 3] =
            std::array::from_fn(|_| WaveField::new(grid_w, grid_h, wave_speed, DAMPING));
        let mut envelopes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; grid_w * grid_h]);

        let dt_frame = 1.0f32 / 60.0;
        let mut substeps_used = 0usize;
        let mut frame_bytes: Vec<u16> = Vec::new();
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("echoes.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("echoes_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        create_videos_from_frames_singlepass(
            grid_w as u32,
            grid_h as u32,
            60,
            |out| {
                for (frame, frame_px) in body_px.iter().enumerate() {
                    let step = ((frame + 1) * steps / frame_count).min(steps - 1);
                    let time = frame as f32 * dt_frame;
                    for body in 0..3 {
                        let (px, py) = frame_px[body];
                        let speed = ctx.kinematics().speeds[body][step];
                        let normalized = ctx.kinematics().normalized_speed(speed_window, speed);
                        let amplitude = (normalized.powf(AMPLITUDE_EXPONENT) as f32).max(0.06);
                        let oscillator =
                            Oscillator { x: px, y: py, amplitude, frequency: frequencies[body] };
                        substeps_used = fields[body].advance_frame(dt_frame, &[oscillator], time);
                        fields[body].accumulate_envelope(&mut envelopes[body]);
                    }

                    // Compose: signed heights displace OKLab a/b along each
                    // body's palette direction; total motion lifts L.
                    frame_bytes.resize(grid_w * grid_h * 3, 0);
                    let field_refs: [&[f32]; 3] = std::array::from_fn(|body| fields[body].field());
                    frame_bytes.par_chunks_mut(3).enumerate().for_each(|(index, chunk)| {
                        let mut lightness = BASE.0;
                        let mut a = BASE.1;
                        let mut b = BASE.2;
                        for body in 0..3 {
                            let height = f64::from(field_refs[body][index]).clamp(-2.0, 2.0);
                            a += CHROMA_SWING * height * hue_directions[body].0;
                            b += CHROMA_SWING * height * hue_directions[body].1;
                            lightness += LIGHT_SWING * height;
                        }
                        let (r, g, bl) = oklab_to_linear_rec2020(lightness.clamp(0.02, 0.95), a, b);
                        let (p3_r, p3_g, p3_b) =
                            linear_rec2020_to_display_p3(r.max(0.0), g.max(0.0), bl.max(0.0));
                        let encode = |v: f64| {
                            (v.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16
                        };
                        chunk[0] = encode(p3_r);
                        chunk[1] = encode(p3_g);
                        chunk[2] = encode(p3_b);
                    });
                    out.write_all(bytemuck::cast_slice(&frame_bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("echoes.mp4", "video");
        sink.record("echoes_hq.mp4", "video");

        // --- Still: time-integrated envelope, upsampled to full res.
        let still_w = ctx.quality.scale_dim(ctx.width) as usize;
        let still_h = ctx.quality.scale_dim(ctx.height) as usize;
        let upsampled: [Vec<f32>; 3] = std::array::from_fn(|body| {
            upsample_bicubic(&envelopes[body], grid_w, grid_h, still_w, still_h)
        });
        // Normalize by the p99 of the combined envelope.
        let mut pooled: Vec<f32> = upsampled[0]
            .iter()
            .zip(upsampled[1].iter())
            .zip(upsampled[2].iter())
            .map(|((&a, &b), &c)| a + b + c)
            .collect();
        let mut sorted: Vec<f32> = pooled.iter().copied().step_by(17).collect();
        sorted.sort_by(f32::total_cmp);
        let reference = sorted[((sorted.len() - 1) as f64 * 0.99) as usize].max(1e-9);

        let mut still_bytes = vec![0u16; still_w * still_h * 3];
        still_bytes.par_chunks_mut(3).enumerate().for_each(|(index, chunk)| {
            let total = f64::from(pooled[index] / reference).clamp(0.0, 2.0);
            let mut a = BASE.1;
            let mut b = BASE.2;
            for body in 0..3 {
                let share = f64::from(upsampled[body][index] / reference).clamp(0.0, 1.5);
                a += 0.10 * share * hue_directions[body].0;
                b += 0.10 * share * hue_directions[body].1;
            }
            let lightness = 0.10 + 0.72 * total.powf(0.55);
            let (r, g, bl) = oklab_to_linear_rec2020(lightness.clamp(0.0, 0.97), a, b);
            let (p3_r, p3_g, p3_b) =
                linear_rec2020_to_display_p3(r.max(0.0), g.max(0.0), bl.max(0.0));
            let encode =
                |v: f64| (v.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
            chunk[0] = encode(p3_r);
            chunk[1] = encode(p3_g);
            chunk[2] = encode(p3_b);
        });
        pooled.clear();
        let image = ImageBuffer::from_raw(still_w as u32, still_h as u32, still_bytes)
            .expect("echoes exposure buffer has width*height*3 samples");
        sink.save_png16(&image, "echoes_exposure.png")?;

        let meta = serde_json::json!({
            "grid": [grid_w, grid_h],
            "wave_speed_px_s": wave_speed,
            "substeps_per_frame": substeps_used,
            "frequencies_hz": frequencies,
            "damping": DAMPING,
            "max_mach": MAX_MACH,
            "note": "envelope accumulated at field resolution, bicubic-upsampled for the still",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("echoes_params.json", &json, "data")?;
        Ok(())
    }
}
