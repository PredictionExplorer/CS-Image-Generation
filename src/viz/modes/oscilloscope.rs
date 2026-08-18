//! V54 `oscilloscope` -- Sound That Draws.
//!
//! Emits a stereo WAV whose channels are the x(t) / y(t) coordinates of the
//! steadiest body (with the other two time-multiplexed in at lower gain):
//! played into any oscilloscope in XY mode, the audio draws the orbit. A
//! phosphor-scope emulator video renders exactly that signal.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio;
use crate::viz::common::display::encode_linear_rec2020_to_u16;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;

/// Audio duration in seconds.
const AUDIO_SECONDS: usize = 60;
/// Samples per multiplex burst.
const BURST_SAMPLES: usize = 400;
/// Crossfade length between bursts (samples).
const BURST_FADE: usize = 16;
/// Gain of the two secondary bodies.
const SECONDARY_GAIN: f64 = 0.25;
/// Maximum slew per sample (fraction of full scale).
const SLEW_LIMIT: f64 = 0.45;
/// DC-block cutoff (Hz).
const DC_CUTOFF_HZ: f64 = 8.0;
/// Emulator video side length at final quality.
const SCOPE_SIDE: u32 = 1080;
/// Emulator video duration in seconds (consumes audio at 2x).
const SCOPE_SECONDS: usize = 30;
/// Emulator frame rate.
const SCOPE_FPS: u32 = 60;
/// Phosphor persistence time constant (seconds).
const PHOSPHOR_TAU: f64 = 0.09;

/// The oscilloscope-music mode.
pub struct Oscilloscope;

/// Normalized xy series for one body over the whole run, mapped to [-0.9, 0.9].
fn normalized_xy(ctx: &VizContext<'_>, body: usize) -> (Vec<f64>, Vec<f64>) {
    let path = &ctx.positions[body];
    let steps = path.len().max(1);
    let mut mean = (0.0, 0.0);
    for point in path {
        mean.0 += point.x;
        mean.1 += point.y;
    }
    mean.0 /= steps as f64;
    mean.1 /= steps as f64;
    let mut extent = 1e-12_f64;
    for point in path {
        extent = extent.max((point.x - mean.0).abs()).max((point.y - mean.1).abs());
    }
    let scale = 0.9 / extent;
    let xs = path.iter().map(|point| (point.x - mean.0) * scale).collect();
    let ys = path.iter().map(|point| (point.y - mean.1) * scale).collect();
    (xs, ys)
}

/// Apply a hard slew-rate limit in place (protects speakers and scope beams).
fn slew_limit(samples: &mut [f64]) {
    for index in 1..samples.len() {
        let delta = samples[index] - samples[index - 1];
        if delta.abs() > SLEW_LIMIT {
            samples[index] = samples[index - 1] + delta.signum() * SLEW_LIMIT;
        }
    }
}

impl VizMode for Oscilloscope {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("oscilloscope").expect("oscilloscope is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        // Steadiest body: smallest max excursion from its own mean.
        let excursion = |body: usize| -> f64 {
            let path = &ctx.positions[body];
            let steps = path.len().max(1) as f64;
            let mean_x = path.iter().map(|point| point.x).sum::<f64>() / steps;
            let mean_y = path.iter().map(|point| point.y).sum::<f64>() / steps;
            path.iter()
                .map(|point| (point.x - mean_x).abs().max((point.y - mean_y).abs()))
                .fold(0.0, f64::max)
        };
        let primary = (0..3).min_by(|&a, &b| excursion(a).total_cmp(&excursion(b))).unwrap_or(0);
        let body_order = [primary, (primary + 1) % 3, (primary + 2) % 3];
        let series: Vec<(Vec<f64>, Vec<f64>)> =
            body_order.iter().map(|&body| normalized_xy(ctx, body)).collect();

        // Time-multiplexed stereo signal: bursts cycle the three bodies.
        let total_samples = AUDIO_SECONDS * audio::SAMPLE_RATE as usize;
        let steps = ctx.step_count().max(2);
        let mut left = vec![0.0_f64; total_samples];
        let mut right = vec![0.0_f64; total_samples];
        for sample in 0..total_samples {
            let burst = sample / BURST_SAMPLES;
            let slot = burst % 3;
            let gain = if slot == 0 { 1.0 } else { SECONDARY_GAIN };
            let within = sample % BURST_SAMPLES;
            let fade = if within < BURST_FADE {
                within as f64 / BURST_FADE as f64
            } else if within >= BURST_SAMPLES - BURST_FADE {
                (BURST_SAMPLES - within) as f64 / BURST_FADE as f64
            } else {
                1.0
            };
            let t = sample as f64 / (total_samples - 1) as f64 * (steps - 1) as f64;
            let base = t.floor() as usize;
            let next = (base + 1).min(steps - 1);
            let frac = t - base as f64;
            let (xs, ys) = &series[slot];
            let x = xs[base] * (1.0 - frac) + xs[next] * frac;
            let y = ys[base] * (1.0 - frac) + ys[next] * frac;
            left[sample] = x * gain * fade;
            right[sample] = y * gain * fade;
        }
        audio::high_pass(&mut left, DC_CUTOFF_HZ);
        audio::high_pass(&mut right, DC_CUTOFF_HZ);
        slew_limit(&mut left);
        slew_limit(&mut right);
        audio::normalize_stereo_peak(&mut left, &mut right, 0.9);
        audio::fade_ends(&mut left, 2400);
        audio::fade_ends(&mut right, 2400);
        audio::write_wav_stereo_24bit(&sink.path("xy.wav"), &left, &right)?;
        sink.record("xy.wav", "audio");

        sink.write_text(
            "xy_notes.txt",
            "XY oscilloscope playback:\n\
             - Connect LEFT to the X input and RIGHT to the Y input.\n\
             - Set the scope to XY mode, both channels DC-coupled, equal gain.\n\
             - The audio literally draws this seed's orbit on the phosphor.\n\
             - Primary body at full scale; the other two are time-multiplexed\n\
             \x20 at -12 dB and appear as dimmer traces.\n",
            "data",
        )?;

        // --- Phosphor emulator video.
        let side = ctx.quality.scale_dim(SCOPE_SIDE);
        let side_px = side as usize;
        let frame_count = ctx.quality.scale_count(SCOPE_SECONDS * SCOPE_FPS as usize);
        let samples_per_frame = total_samples / frame_count.max(1);
        let decay = (-1.0 / (f64::from(SCOPE_FPS) * PHOSPHOR_TAU)).exp();

        let (mean_l, mean_a, mean_b) = ctx.mean_color(primary);
        let (_, chroma, hue) = oklab_to_oklch(mean_l, mean_a, mean_b);
        // Blend the body hue toward P31 phosphor green.
        let phosphor_hue = hue + (145.0 - hue) * 0.6;
        let phosphor_chroma = (chroma * 0.8).max(0.06);

        let mut intensity = vec![0.0_f32; side_px * side_px];
        let mut sample_cursor = 0usize;

        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("scope.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("scope_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];

        let mut frames_written = 0usize;
        create_videos_from_frames_singlepass(
            side,
            side,
            SCOPE_FPS,
            |out| {
                let mut frame_pixels = vec![(0.0, 0.0, 0.0); side_px * side_px];
                let mut frame_u16: Vec<u16> = Vec::new();
                for _ in 0..frame_count {
                    // Decay phosphor, then integrate this frame's samples.
                    for value in &mut intensity {
                        *value *= decay as f32;
                    }
                    let frame_end = (sample_cursor + samples_per_frame).min(total_samples);
                    for sample in sample_cursor.max(1)..frame_end {
                        let (x0, y0) = (left[sample - 1], right[sample - 1]);
                        let (x1, y1) = (left[sample], right[sample]);
                        let dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                        let brightness = (0.004 / (dist + 0.0015)).min(1.4) as f32;
                        // Subsample the beam path so fast sweeps stay connected.
                        let sub_steps = ((dist * f64::from(side) * 0.5) as usize).clamp(1, 24);
                        for sub in 0..sub_steps {
                            let t = sub as f64 / sub_steps as f64;
                            let x = x0 + (x1 - x0) * t;
                            let y = y0 + (y1 - y0) * t;
                            let px = ((x * 0.5 + 0.5) * f64::from(side - 1)).round() as usize;
                            let py = ((-y * 0.5 + 0.5) * f64::from(side - 1)).round() as usize;
                            if px < side_px && py < side_px {
                                intensity[py * side_px + px] += brightness / sub_steps as f32;
                            }
                        }
                    }
                    sample_cursor = frame_end;

                    // Intensity -> phosphor color -> Display P3 u16 frame.
                    frame_pixels.par_iter_mut().zip(intensity.par_iter()).for_each(
                        |(pixel, &raw)| {
                            let glow = 1.0 - (-f64::from(raw)).exp();
                            let lightness = 0.035 + 0.92 * glow;
                            let (l, a, b) = oklch_to_oklab(
                                lightness,
                                phosphor_chroma * glow.powf(0.4),
                                phosphor_hue,
                            );
                            *pixel = oklab_to_linear_rec2020(l, a, b);
                        },
                    );
                    encode_linear_rec2020_to_u16(&frame_pixels, &mut frame_u16);
                    out.write_all(bytemuck::cast_slice(&frame_u16))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    frames_written += 1;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("scope.mp4", "video");
        sink.record("scope_hq.mp4", "video");
        tracing::info!("   oscilloscope: {frames_written} emulator frames written");
        Ok(())
    }
}
