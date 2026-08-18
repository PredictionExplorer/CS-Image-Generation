//! V14 `triangle-centers` -- The Constellation of Centers.
//!
//! The triangle's five classical centers (centroid, incenter, circumcenter,
//! orthocenter, nine-point center) each trace their own secret curve through
//! the run; all five are drawn braided together with the Euler line sweeping
//! between them as a translucent fan. A hidden second artwork inside the
//! first.

use crate::error::Result;
use crate::oklab::{max_display_p3_chroma_for_lh, oklab_to_oklch, oklch_to_oklab};
use crate::render::OklabColor;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::SpdCanvas;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;

/// Stroke thickness multipliers per center (centroid boldest).
const THICKNESSES: [f64; 5] = [1.2, 0.9, 0.75, 0.65, 0.55];
/// Hue offsets from the palette anchor per center (degrees).
const HUE_OFFSETS: [f64; 5] = [-40.0, -20.0, 0.0, 20.0, 40.0];
/// Lightness ladder per center.
const LIGHTNESS: [f64; 5] = [0.62, 0.66, 0.70, 0.74, 0.78];
/// Base stroke energy.
const STROKE_ENERGY: f64 = 0.02;
/// Euler-line fan stride (steps) and energy fraction.
const FAN_STRIDE: usize = 400;
/// Fan energy relative to strand energy.
const FAN_ENERGY: f64 = 0.03 * STROKE_ENERGY / 0.02;
/// Degenerate-configuration quality floor (skip below).
const QUALITY_FLOOR: f64 = 0.002;
/// Video length in frames at 60 fps.
const VIDEO_FRAMES: usize = 1200;

/// The triangle-centers mode.
pub struct TriangleCenters;

/// The five centers of one triangle, in world xy (z ignored).
/// Returns `None` when the configuration is too degenerate.
fn centers(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> Option<[(f64, f64); 5]> {
    let (ax, ay) = (a.x, a.y);
    let (bx, by) = (b.x, b.y);
    let (cx, cy) = (c.x, c.y);

    let centroid = ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0);

    let side_a = ((bx - cx).powi(2) + (by - cy).powi(2)).sqrt(); // opposite A
    let side_b = ((ax - cx).powi(2) + (ay - cy).powi(2)).sqrt();
    let side_c = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
    let perimeter = side_a + side_b + side_c;
    if perimeter <= 1e-12 {
        return None;
    }
    let incenter = (
        (side_a * ax + side_b * bx + side_c * cx) / perimeter,
        (side_a * ay + side_b * by + side_c * cy) / perimeter,
    );

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-9 * perimeter * perimeter {
        return None; // near-collinear: circumcenter explodes
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let circumcenter = (
        (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d,
        (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d,
    );

    // H = A + B + C - 2 * circumcenter.
    let orthocenter = (ax + bx + cx - 2.0 * circumcenter.0, ay + by + cy - 2.0 * circumcenter.1);
    let nine_point =
        ((circumcenter.0 + orthocenter.0) * 0.5, (circumcenter.1 + orthocenter.1) * 0.5);

    Some([centroid, incenter, circumcenter, orthocenter, nine_point])
}

/// Triangle quality (0 collinear .. 1 equilateral) for alpha fading.
fn quality(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let area = 0.5 * (ab.x * ac.y - ab.y * ac.x).abs();
    let sum_sq = (b - a).norm_squared() + (c - b).norm_squared() + (a - c).norm_squared();
    if sum_sq <= 1e-24 { 0.0 } else { (4.0 * 3.0_f64.sqrt() * area / sum_sq).clamp(0.0, 1.0) }
}

/// Draw strands for the given step range into the canvas.
#[allow(clippy::too_many_arguments)]
fn draw_range(
    canvas: &mut SpdCanvas,
    ctx: &VizContext<'_>,
    render_ctx: &RenderContext,
    colors: &[OklabColor; 5],
    fan_color: OklabColor,
    range: std::ops::Range<usize>,
    stride: usize,
    scale: f32,
) {
    let steps = ctx.step_count();
    let mut previous: Option<[(f64, f64); 5]> = None;
    let mut start = range.start;
    // Align to stride so incremental calls never re-draw segments.
    if !start.is_multiple_of(stride) {
        start += stride - start % stride;
    }
    for step in (start..range.end.min(steps)).step_by(stride) {
        let (a, b, c) = (ctx.positions[0][step], ctx.positions[1][step], ctx.positions[2][step]);
        let q = quality(a, b, c);
        let Some(current) = centers(a, b, c) else {
            previous = None;
            continue;
        };
        if let Some(last) = previous {
            let fade = (q / 0.05).clamp(0.15, 1.0);
            for center in 0..5 {
                if center >= 2 && q < QUALITY_FLOOR {
                    continue; // circumcenter family explodes near collinear
                }
                let (px0, py0) = render_ctx.to_pixel(last[center].0, last[center].1);
                let (px1, py1) = render_ctx.to_pixel(current[center].0, current[center].1);
                canvas.draw_stroke(
                    (px0 * scale, py0 * scale),
                    (px1 * scale, py1 * scale),
                    colors[center],
                    colors[center],
                    0.85 * fade,
                    STROKE_ENERGY,
                    THICKNESSES[center],
                );
            }
            // Euler line fan: circumcenter -> orthocenter.
            if step % FAN_STRIDE < stride && q >= QUALITY_FLOOR {
                let (fx0, fy0) = render_ctx.to_pixel(current[2].0, current[2].1);
                let (fx1, fy1) = render_ctx.to_pixel(current[3].0, current[3].1);
                canvas.draw_stroke(
                    (fx0 * scale, fy0 * scale),
                    (fx1 * scale, fy1 * scale),
                    fan_color,
                    fan_color,
                    0.5,
                    FAN_ENERGY,
                    0.6,
                );
            }
        }
        previous = Some(current);
    }
}

impl VizMode for TriangleCenters {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("triangle-centers").expect("triangle-centers is in the catalog")
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

        // Five family hues around the palette anchor, low chroma.
        let (mean_l, mean_a, mean_b) = ctx.mean_color(0);
        let (_, _, anchor) = oklab_to_oklch(mean_l, mean_a, mean_b);
        let colors: [OklabColor; 5] = std::array::from_fn(|center| {
            let hue = (anchor + HUE_OFFSETS[center]).rem_euclid(360.0);
            let chroma = (max_display_p3_chroma_for_lh(LIGHTNESS[center], hue) * 0.5).min(0.09);
            oklch_to_oklab(LIGHTNESS[center], chroma, hue)
        });
        let fan_color = oklch_to_oklab(0.5, 0.02, (anchor + 180.0).rem_euclid(360.0));

        // Full-resolution still.
        let steps = ctx.step_count();
        let mut canvas = SpdCanvas::new(ctx.width, ctx.height);
        draw_range(&mut canvas, ctx, &render_ctx, &colors, fan_color, 0..steps, 2, 1.0);
        sink.save_png16(&canvas.into_png16(clip_black, clip_white, 1.0), "centers.png")?;

        // Half-resolution reveal video with fixed final-state levels.
        let half_w = ctx.quality.scale_dim(ctx.width / 2 * 2) / 2 * 2;
        let half_h = (ctx.height * half_w / ctx.width) & !1;
        let scale = half_w as f32 / ctx.width as f32;
        let mut video_canvas = SpdCanvas::new(half_w, half_h);
        draw_range(&mut video_canvas, ctx, &render_ctx, &colors, fan_color, 0..steps, 4, scale);
        let levels = video_canvas.levels(clip_black, clip_white, 1.0);
        video_canvas.clear();

        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let steps_per_frame = steps / frame_count.max(1);
        let mut drawn = 0usize;
        stream_video(
            half_w,
            half_h,
            60,
            &sink.path("centers.mp4"),
            &sink.path("centers_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * steps_per_frame).min(steps);
                draw_range(
                    &mut video_canvas,
                    ctx,
                    &render_ctx,
                    &colors,
                    fan_color,
                    drawn..target,
                    4,
                    scale,
                );
                drawn = target;
                video_canvas.convert_into(rgba);
            },
        )?;
        sink.record("centers.mp4", "video");
        sink.record("centers_hq.mp4", "video");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centers_of_equilateral_coincide() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(1.0, 0.0, 0.0);
        let c = Vector3::new(0.5, 3.0_f64.sqrt() / 2.0, 0.0);
        let all = centers(a, b, c).expect("equilateral is non-degenerate");
        for pair in all.windows(2) {
            assert!(
                (pair[0].0 - pair[1].0).abs() < 1e-12 && (pair[0].1 - pair[1].1).abs() < 1e-12,
                "all centers coincide for the equilateral triangle: {all:?}"
            );
        }
    }

    #[test]
    fn collinear_configuration_is_rejected() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(1.0, 0.0, 0.0);
        let c = Vector3::new(2.0, 0.0, 0.0);
        assert!(centers(a, b, c).is_none());
    }

    #[test]
    fn right_triangle_orthocenter_is_the_right_angle_vertex() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(4.0, 0.0, 0.0);
        let c = Vector3::new(0.0, 3.0, 0.0);
        let all = centers(a, b, c).expect("right triangle is non-degenerate");
        let orthocenter = all[3];
        assert!(orthocenter.0.abs() < 1e-9 && orthocenter.1.abs() < 1e-9);
    }
}
