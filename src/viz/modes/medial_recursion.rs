//! V15 `medial-recursion` -- Vortex of Triangles.
//!
//! Each sampled timestep nests the medial (midpoint) triangle recursively
//! eight levels deep, each level dimmer, thinner, and hue-rotated -- the
//! accumulated exposure builds spiralling vortex tunnels wherever the orbit
//! dwells. A seeded gate occasionally swaps in the anticomplementary triangle
//! for outward bloom spikes.

use crate::error::Result;
use crate::render::OklabColor;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::SpdCanvas;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;

/// Recursion depth.
const LEVELS: usize = 8;
/// Per-level energy decay.
const LEVEL_DECAY: f64 = 0.55;
/// Per-level hue rotation (degrees).
const HUE_STEP_DEG: f64 = 9.0;
/// Per-level thickness decay.
const THICKNESS_DECAY: f64 = 0.9;
/// Step stride between samples.
const SAMPLE_STRIDE: usize = 4;
/// Base edge energy.
const BASE_ENERGY: f64 = 0.012;
/// Probability of the anticomplementary variant (seeded gate).
const ANTI_GATE: f64 = 0.5;
/// Video frames at 60 fps.
const VIDEO_FRAMES: usize = 1200;

/// The medial-recursion mode.
pub struct MedialRecursion;

/// Rotate an `OkLab` color's hue by `degrees`.
fn rotate_hue(color: OklabColor, degrees: f64) -> OklabColor {
    let (l, a, b) = color;
    let radians = degrees.to_radians();
    let (sin, cos) = radians.sin_cos();
    (l, a * cos - b * sin, a * sin + b * cos)
}

/// Draw the nested-triangle stack for one sampled step.
#[allow(clippy::too_many_arguments)]
fn draw_stack(
    canvas: &mut SpdCanvas,
    render_ctx: &RenderContext,
    scale: f32,
    step_colors: [OklabColor; 3],
    mut points: [(f64, f64); 3],
    anticomplementary: bool,
) {
    for level in 0..LEVELS {
        let energy = BASE_ENERGY * LEVEL_DECAY.powi(level as i32);
        let thickness = THICKNESS_DECAY.powi(level as i32);
        for edge in 0..3 {
            let from = points[edge];
            let to = points[(edge + 1) % 3];
            let (fx, fy) = render_ctx.to_pixel(from.0, from.1);
            let (tx, ty) = render_ctx.to_pixel(to.0, to.1);
            let color = rotate_hue(step_colors[edge], HUE_STEP_DEG * level as f64);
            canvas.draw_stroke(
                (fx * scale, fy * scale),
                (tx * scale, ty * scale),
                color,
                rotate_hue(step_colors[(edge + 1) % 3], HUE_STEP_DEG * level as f64),
                0.8,
                energy,
                thickness,
            );
        }
        points = if anticomplementary {
            // Anticomplementary step: A' = B + C - A (outward growth).
            [
                (points[1].0 + points[2].0 - points[0].0, points[1].1 + points[2].1 - points[0].1),
                (points[2].0 + points[0].0 - points[1].0, points[2].1 + points[0].1 - points[1].1),
                (points[0].0 + points[1].0 - points[2].0, points[0].1 + points[1].1 - points[2].1),
            ]
        } else {
            // Medial step: midpoints (inward collapse).
            [
                ((points[1].0 + points[2].0) * 0.5, (points[1].1 + points[2].1) * 0.5),
                ((points[2].0 + points[0].0) * 0.5, (points[2].1 + points[0].1) * 0.5),
                ((points[0].0 + points[1].0) * 0.5, (points[0].1 + points[1].1) * 0.5),
            ]
        };
    }
}

/// Draw all stacks within a step range (stride-aligned for incremental use).
#[allow(clippy::too_many_arguments)]
fn draw_range(
    canvas: &mut SpdCanvas,
    ctx: &VizContext<'_>,
    render_ctx: &RenderContext,
    range: std::ops::Range<usize>,
    scale: f32,
    anti_enabled: bool,
) {
    let steps = ctx.step_count();
    let mut start = range.start;
    if !start.is_multiple_of(SAMPLE_STRIDE) {
        start += SAMPLE_STRIDE - start % SAMPLE_STRIDE;
    }
    for step in (start..range.end.min(steps)).step_by(SAMPLE_STRIDE) {
        let points = [
            (ctx.positions[0][step].x, ctx.positions[0][step].y),
            (ctx.positions[1][step].x, ctx.positions[1][step].y),
            (ctx.positions[2][step].x, ctx.positions[2][step].y),
        ];
        let step_colors = [ctx.colors[0][step], ctx.colors[1][step], ctx.colors[2][step]];
        // Every 4th sample blooms outward when the seed gate is on.
        let anticomplementary = anti_enabled && (step / SAMPLE_STRIDE) % 4 == 3;
        draw_stack(canvas, render_ctx, scale, step_colors, points, anticomplementary);
    }
}

impl VizMode for MedialRecursion {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("medial-recursion").expect("medial-recursion is in the catalog")
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
        let mut rng = ctx.fork_rng(self.entry().flag);
        let anti_enabled = rng.next_f64() < ANTI_GATE;
        let steps = ctx.step_count();

        // Full-resolution still.
        let mut canvas = SpdCanvas::new(ctx.width, ctx.height);
        draw_range(&mut canvas, ctx, &render_ctx, 0..steps, 1.0, anti_enabled);
        sink.save_png16(&canvas.into_png16(clip_black, clip_white, 1.0), "vortex.png")?;

        // Half-resolution reveal video with fixed final levels.
        let half_w = ctx.quality.scale_dim(ctx.width / 2 * 2) / 2 * 2;
        let half_h = (ctx.height * half_w / ctx.width) & !1;
        let scale = half_w as f32 / ctx.width as f32;
        let mut video_canvas = SpdCanvas::new(half_w, half_h);
        draw_range(&mut video_canvas, ctx, &render_ctx, 0..steps, scale, anti_enabled);
        let levels = video_canvas.levels(clip_black, clip_white, 1.0);
        video_canvas.clear();

        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let steps_per_frame = steps / frame_count.max(1);
        let mut drawn = 0usize;
        stream_video(
            half_w,
            half_h,
            60,
            &sink.path("vortex.mp4"),
            &sink.path("vortex_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * steps_per_frame).min(steps);
                draw_range(&mut video_canvas, ctx, &render_ctx, drawn..target, scale, anti_enabled);
                drawn = target;
                video_canvas.convert_into(rgba);
            },
        )?;
        sink.record("vortex.mp4", "video");
        sink.record("vortex_hq.mp4", "video");

        let meta = serde_json::json!({
            "levels": LEVELS,
            "level_decay": LEVEL_DECAY,
            "hue_step_deg": HUE_STEP_DEG,
            "sample_stride": SAMPLE_STRIDE,
            "anticomplementary_enabled": anti_enabled,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("vortex.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hue_rotation_preserves_chroma() {
        let color = (0.7, 0.12, -0.05);
        let rotated = rotate_hue(color, 90.0);
        let chroma_before = (color.1 * color.1 + color.2 * color.2).sqrt();
        let chroma_after = (rotated.1 * rotated.1 + rotated.2 * rotated.2).sqrt();
        assert!((chroma_before - chroma_after).abs() < 1e-12);
        assert!((rotated.0 - color.0).abs() < 1e-12);
    }
}
