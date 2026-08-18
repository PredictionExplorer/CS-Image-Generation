//! V29 `retarded-time` -- Where Their Light Says They Are.
//!
//! Introduces a finite virtual light speed: each body is also drawn where
//! the others *see* it (its retarded position), so the true triangle and
//! three mutually-seen ghost triangles shear apart as speeds rise. Ghost
//! strokes carry a hue-shifted "memory" tint; light-delay struts connect
//! observer and image at intervals.

use crate::error::Result;
use crate::render::OklabColor;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::SpdCanvas;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;

/// Fast segments reach this fraction of the virtual light speed.
const SPEED_FRACTION_AT_P99: f64 = 0.22;

/// The V29 virtual light speed: p99 pooled speed = 22% of c. Returns
/// `(c in world units per unit time, c in world units per step)`.
/// Exported for V65 `witness` ("same c as V29").
pub(crate) fn virtual_light_speed(
    kinematics: &crate::viz::common::kinematics::Kinematics,
) -> (f64, f64) {
    let mut sample: Vec<f64> =
        kinematics.speeds.iter().flat_map(|body| body.iter().step_by(97).copied()).collect();
    sample.sort_by(f64::total_cmp);
    let p99 = sample[((sample.len() - 1) as f64 * 0.99) as usize];
    let c_world_per_time = p99 / SPEED_FRACTION_AT_P99;
    (c_world_per_time, c_world_per_time * crate::render::constants::DEFAULT_DT)
}
/// Ghost stroke energy relative to true strokes.
const GHOST_ENERGY: f64 = 0.22;
/// Ghost hue shift (degrees, toward violet memory).
const GHOST_HUE_SHIFT_DEG: f64 = -12.0;
/// Strut stride in steps.
const STRUT_STRIDE: usize = 200;
/// Strut energy relative to true strokes.
const STRUT_ENERGY: f64 = 0.03;
/// Base stroke energy.
const BASE_ENERGY: f64 = 0.012;
/// Fixed-point iterations for the retarded-time solve.
const SOLVER_ITERATIONS: usize = 4;
/// Drawing stride over simulation steps.
const DRAW_STRIDE: usize = 2;
/// Video frames at 60 fps.
const VIDEO_FRAMES: usize = 1200;

/// The retarded-time mode.
pub struct RetardedTime;

/// Interpolated position of a body at a fractional step.
/// Exported for V65 `witness` (shared retarded-optics core).
pub(crate) fn position_at(path: &[Vector3<f64>], step: f64) -> Vector3<f64> {
    let clamped = step.clamp(0.0, (path.len() - 1) as f64);
    let base = clamped.floor() as usize;
    let next = (base + 1).min(path.len() - 1);
    let fraction = clamped - base as f64;
    path[base] * (1.0 - fraction) + path[next] * fraction
}

/// Solve the retarded step at which `observer` (at `step`) sees `emitter`.
///
/// `c_steps` is the light speed expressed in world units per simulation step.
/// Exported for V65 `witness` (its optics core).
pub(crate) fn retarded_step(
    observer: Vector3<f64>,
    emitter: &[Vector3<f64>],
    step: usize,
    c_steps: f64,
) -> f64 {
    let mut retarded = step as f64;
    for _ in 0..SOLVER_ITERATIONS {
        let seen = position_at(emitter, retarded);
        let distance = (observer - seen).norm();
        retarded = (step as f64 - distance / c_steps).max(0.0);
    }
    retarded
}

/// Rotate a color's hue (`OkLab` a/b plane) by degrees.
fn shift_hue(color: OklabColor, degrees: f64) -> OklabColor {
    let (l, a, b) = color;
    let (sin, cos) = degrees.to_radians().sin_cos();
    (l, a * cos - b * sin, a * sin + b * cos)
}

/// Draw the layered exposure for a step range (stride-aligned).
#[allow(clippy::too_many_arguments)]
fn draw_range(
    canvas: &mut SpdCanvas,
    ctx: &VizContext<'_>,
    render_ctx: &RenderContext,
    range: std::ops::Range<usize>,
    c_steps: f64,
    scale: f32,
) {
    let steps = ctx.step_count();
    let mut start = range.start.max(1);
    if !start.is_multiple_of(DRAW_STRIDE) {
        start += DRAW_STRIDE - start % DRAW_STRIDE;
    }
    for step in (start..range.end.min(steps - 1)).step_by(DRAW_STRIDE) {
        // True triangle edges at full energy.
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
                0.85,
                BASE_ENERGY,
                1.0,
            );
        }
        // Seen triangles: for each observer, the two others at retarded
        // positions, joined as ghost edges with the memory tint.
        for observer in 0..3 {
            let observer_position = ctx.positions[observer][step];
            let mut seen_points = Vec::with_capacity(2);
            for emitter in 0..3 {
                if emitter == observer {
                    continue;
                }
                let retarded =
                    retarded_step(observer_position, &ctx.positions[emitter], step, c_steps);
                let seen = position_at(&ctx.positions[emitter], retarded);
                let color_step = retarded.floor() as usize;
                seen_points.push((
                    seen,
                    shift_hue(ctx.colors[emitter][color_step.min(steps - 1)], GHOST_HUE_SHIFT_DEG),
                ));
            }
            // Ghost edges observer->seen_a, observer->seen_b, seen_a->seen_b.
            let (ox, oy) = render_ctx.to_pixel(observer_position.x, observer_position.y);
            let observer_color = shift_hue(ctx.colors[observer][step], GHOST_HUE_SHIFT_DEG);
            let mut pixel_points = Vec::with_capacity(2);
            for &(seen, color) in &seen_points {
                let (sx, sy) = render_ctx.to_pixel(seen.x, seen.y);
                pixel_points.push(((sx, sy), color));
                canvas.draw_stroke(
                    (ox * scale, oy * scale),
                    (sx * scale, sy * scale),
                    observer_color,
                    color,
                    0.7,
                    BASE_ENERGY * GHOST_ENERGY,
                    0.8,
                );
            }
            if pixel_points.len() == 2 {
                canvas.draw_stroke(
                    (pixel_points[0].0.0 * scale, pixel_points[0].0.1 * scale),
                    (pixel_points[1].0.0 * scale, pixel_points[1].0.1 * scale),
                    pixel_points[0].1,
                    pixel_points[1].1,
                    0.7,
                    BASE_ENERGY * GHOST_ENERGY,
                    0.8,
                );
            }
            // Light-delay struts, sparse.
            if step % STRUT_STRIDE < DRAW_STRIDE {
                for &((sx, sy), color) in &pixel_points {
                    canvas.draw_stroke(
                        (ox * scale, oy * scale),
                        (sx * scale, sy * scale),
                        color,
                        color,
                        0.5,
                        BASE_ENERGY * STRUT_ENERGY,
                        0.5,
                    );
                }
            }
        }
    }
}

impl VizMode for RetardedTime {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("retarded-time").expect("retarded-time is in the catalog")
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
        let kinematics = ctx.kinematics();

        // Virtual light speed: p99 pooled speed hits 22% of c.
        // Speeds are world-units/time; convert to world-units/step via dt.
        let (c_world_per_time, c_steps) = virtual_light_speed(kinematics);

        // Full-resolution long-exposure still.
        let mut canvas = SpdCanvas::new(ctx.width, ctx.height);
        draw_range(&mut canvas, ctx, &render_ctx, 0..steps, c_steps, 1.0);
        sink.save_png16(&canvas.into_png16(clip_black, clip_white, 1.0), "retarded.png")?;

        // Half-resolution reveal video.
        let half_w = ctx.quality.scale_dim(ctx.width / 2 * 2) / 2 * 2;
        let half_h = (ctx.height * half_w / ctx.width) & !1;
        let scale = half_w as f32 / ctx.width as f32;
        let mut video_canvas = SpdCanvas::new(half_w, half_h);
        draw_range(&mut video_canvas, ctx, &render_ctx, 0..steps, c_steps, scale);
        let levels = video_canvas.levels(clip_black, clip_white, 1.0);
        video_canvas.clear();

        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let steps_per_frame = steps / frame_count.max(1);
        let mut drawn = 0usize;
        stream_video(
            half_w,
            half_h,
            60,
            &sink.path("retarded.mp4"),
            &sink.path("retarded_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * steps_per_frame).min(steps);
                draw_range(&mut video_canvas, ctx, &render_ctx, drawn..target, c_steps, scale);
                drawn = target;
                video_canvas.convert_into(rgba);
            },
        )?;
        sink.record("retarded.mp4", "video");
        sink.record("retarded_hq.mp4", "video");

        let meta = serde_json::json!({
            "p99_speed": c_world_per_time * SPEED_FRACTION_AT_P99,
            "virtual_c_world_per_time": c_world_per_time,
            "speed_fraction_at_p99": SPEED_FRACTION_AT_P99,
            "ghost_energy": GHOST_ENERGY,
            "ghost_hue_shift_deg": GHOST_HUE_SHIFT_DEG,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("c_choice.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_converges_for_static_emitter() {
        // Static emitter at distance d: retarded step = step - d / c_steps.
        let emitter: Vec<Vector3<f64>> = vec![Vector3::new(3.0, 4.0, 0.0); 1000];
        let observer = Vector3::new(0.0, 0.0, 0.0);
        let c_steps = 0.05; // world units per step
        let solved = retarded_step(observer, &emitter, 900, c_steps);
        let expected = 900.0 - 5.0 / 0.05;
        assert!((solved - expected).abs() < 1e-9, "solved {solved}, expected {expected}");
    }

    #[test]
    fn solver_clamps_at_run_start() {
        let emitter: Vec<Vector3<f64>> = vec![Vector3::new(100.0, 0.0, 0.0); 100];
        let solved = retarded_step(Vector3::zeros(), &emitter, 10, 0.001);
        assert_eq!(solved, 0.0);
    }
}
