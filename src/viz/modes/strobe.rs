//! V20 `strobe` -- Phantom Triangles.
//!
//! Accumulates only every k-th simulation step, each flash drawn as the full
//! triangle: temporal aliasing conjures slow phantom triangles rotating
//! against the true motion (the wagon-wheel effect). The strobe interval is
//! chosen from the tightest pair's mean angular rate so a phantom actually
//! emerges (~10 degrees of true rotation per flash).

use crate::error::Result;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::SpdCanvas;
use crate::viz::common::kinematics::PAIRS;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;

/// Target true rotation per flash (degrees).
const TARGET_DEG_PER_FLASH: f64 = 10.0;
/// Strobe interval clamp (steps).
const INTERVAL_CLAMP: (usize, usize) = (40, 4000);
/// Energy compensation exponent.
const BOOST_EXPONENT: f64 = 0.7;
/// Base flash energy.
const FLASH_ENERGY: f64 = 0.006;
/// Video frames at 60 fps (plus a 2 s black lead-in).
const VIDEO_FRAMES: usize = 1200;
/// Black lead-in frames.
const LEAD_IN_FRAMES: usize = 120;

/// The stroboscopic sampling mode.
pub struct Strobe;

/// Mean angular rate (radians/step) of the tightest pair's separation vector.
fn tightest_pair_rate(ctx: &VizContext<'_>) -> (usize, usize, f64) {
    let kinematics = ctx.kinematics();
    let tightest = PAIRS
        .iter()
        .enumerate()
        .min_by(|(index_a, _), (index_b, _)| {
            let mean = |index: usize| {
                let series = &kinematics.pairwise[index];
                series.iter().step_by(97).sum::<f64>() / series.iter().step_by(97).count() as f64
            };
            mean(*index_a).total_cmp(&mean(*index_b))
        })
        .map_or((0, 1), |(_, &pair)| pair);

    let steps = ctx.step_count();
    let mut total_rotation = 0.0f64;
    let mut previous_angle: Option<f64> = None;
    let stride = 8usize;
    for step in (0..steps).step_by(stride) {
        let delta = ctx.positions[tightest.0][step] - ctx.positions[tightest.1][step];
        let angle = delta.y.atan2(delta.x);
        if let Some(prev) = previous_angle {
            let mut diff = angle - prev;
            while diff > std::f64::consts::PI {
                diff -= std::f64::consts::TAU;
            }
            while diff < -std::f64::consts::PI {
                diff += std::f64::consts::TAU;
            }
            total_rotation += diff.abs();
        }
        previous_angle = Some(angle);
    }
    let samples = (steps / stride).max(1);
    (tightest.0, tightest.1, total_rotation / (samples as f64 * stride as f64))
}

/// Draw flashes within a step range (flash-aligned for incremental use).
fn draw_flashes(
    canvas: &mut SpdCanvas,
    ctx: &VizContext<'_>,
    render_ctx: &RenderContext,
    range: std::ops::Range<usize>,
    interval: usize,
    energy: f64,
    scale: f32,
) {
    let steps = ctx.step_count();
    let mut start = range.start;
    if !start.is_multiple_of(interval) {
        start += interval - start % interval;
    }
    for step in (start..range.end.min(steps)).step_by(interval) {
        for edge in 0..3 {
            let a = ctx.positions[edge][step];
            let b = ctx.positions[(edge + 1) % 3][step];
            let (ax, ay) = render_ctx.to_pixel(a.x, a.y);
            let (bx, by) = render_ctx.to_pixel(b.x, b.y);
            canvas.draw_stroke(
                (ax * scale, ay * scale),
                (bx * scale, by * scale),
                ctx.colors[edge][step],
                ctx.colors[(edge + 1) % 3][step],
                0.9,
                energy,
                1.0,
            );
        }
    }
}

impl VizMode for Strobe {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("strobe").expect("strobe is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let render_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let steps = ctx.step_count();

        let (pair_a, pair_b, rate) = tightest_pair_rate(ctx);
        let target = TARGET_DEG_PER_FLASH.to_radians();
        let interval = if rate > 1e-9 {
            ((target / rate).round() as usize).clamp(INTERVAL_CLAMP.0, INTERVAL_CLAMP.1)
        } else {
            // Degenerate: fall back to a golden-ratio non-resonant interval.
            ((steps as f64 / 1000.0 * 1.618) as usize).clamp(INTERVAL_CLAMP.0, INTERVAL_CLAMP.1)
        };
        let energy =
            FLASH_ENERGY * (interval as f64 / INTERVAL_CLAMP.0 as f64).powf(BOOST_EXPONENT);

        // Full-resolution still.
        let mut canvas = SpdCanvas::new(ctx.width, ctx.height);
        draw_flashes(&mut canvas, ctx, &render_ctx, 0..steps, interval, energy, 1.0);
        sink.save_png16(&canvas.into_png16(clip_black, clip_white, 1.0), "strobe.png")?;

        // Half-resolution video: 2 s of black, then flashes reveal in order.
        let half_w = ctx.quality.scale_dim(ctx.width / 2 * 2) / 2 * 2;
        let half_h = (ctx.height * half_w / ctx.width) & !1;
        let scale = half_w as f32 / ctx.width as f32;
        let mut video_canvas = SpdCanvas::new(half_w, half_h);
        draw_flashes(&mut video_canvas, ctx, &render_ctx, 0..steps, interval, energy, scale);
        let levels = video_canvas.levels(clip_black, clip_white, 1.0);
        video_canvas.clear();

        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let lead_in = ctx.quality.scale_count(LEAD_IN_FRAMES);
        let active_frames = frame_count.saturating_sub(lead_in).max(1);
        let mut drawn = 0usize;
        stream_video(
            half_w,
            half_h,
            60,
            &sink.path("strobe.mp4"),
            &sink.path("strobe_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                if frame >= lead_in {
                    let progress = frame - lead_in + 1;
                    let target_step = (progress * steps / active_frames).min(steps);
                    draw_flashes(
                        &mut video_canvas,
                        ctx,
                        &render_ctx,
                        drawn..target_step,
                        interval,
                        energy,
                        scale,
                    );
                    drawn = target_step;
                }
                video_canvas.convert_into(rgba);
            },
        )?;
        sink.record("strobe.mp4", "video");
        sink.record("strobe_hq.mp4", "video");

        let meta = serde_json::json!({
            "tightest_pair": [pair_a, pair_b],
            "mean_angular_rate_rad_per_step": rate,
            "strobe_interval_steps": interval,
            "true_rotation_deg_per_flash": (rate * interval as f64).to_degrees(),
            "flash_count": steps / interval.max(1),
            "note": "interval from mean angular rate (deviation from FFT-peak spec)",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("strobe_params.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn interval_clamp_bounds_hold() {
        let clamped = 1_000_000usize.clamp(super::INTERVAL_CLAMP.0, super::INTERVAL_CLAMP.1);
        assert_eq!(clamped, 4000);
        let clamped_low = 1usize.clamp(super::INTERVAL_CLAMP.0, super::INTERVAL_CLAMP.1);
        assert_eq!(clamped_low, 40);
    }
}
