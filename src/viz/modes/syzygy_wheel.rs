//! V13 `syzygy-wheel` -- The Rhythm Clock.
//!
//! Every three-body alignment event plotted as a radial tick on a circular
//! clock face (angle = run time, color = middle body, length = alignment
//! sharpness), with an inner reference ring, five-percent hairline ticks,
//! and the initial triangle as a center glyph.

use crate::error::Result;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::SpdCanvas;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;

/// Poster side length at final quality.
const POSTER_SIDE: u32 = 2234;
/// Icon side length (always rendered, quality-independent).
const ICON_SIDE: u32 = 1024;
/// Inner reference ring radius as a fraction of the face radius.
const INNER_RING_FRACTION: f64 = 0.62;
/// Maximum tick length as a fraction of the face radius.
const MAX_TICK_FRACTION: f64 = 0.30;

/// The syzygy rhythm clock mode.
pub struct SyzygyWheel;

/// Render the wheel onto a fresh canvas of the given side length.
fn render_wheel(ctx: &VizContext<'_>, side: u32) -> SpdCanvas {
    let mut canvas = SpdCanvas::new(side, side);
    let center = f64::from(side) * 0.5;
    let face_radius = center * 0.92;
    let inner_radius = face_radius * INNER_RING_FRACTION;
    let steps = ctx.step_count().max(1);

    let angle_of = |step: usize| -> f64 {
        step as f64 / steps as f64 * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2
    };
    let point_at = |angle: f64, radius: f64| -> (f32, f32) {
        ((center + radius * angle.cos()) as f32, (center + radius * angle.sin()) as f32)
    };

    // Inner reference ring: quiet blue-gray circle.
    let ring_color = (0.45, -0.012, -0.035);
    let ring_segments = 720;
    for segment in 0..ring_segments {
        let a0 = f64::from(segment) / f64::from(ring_segments) * std::f64::consts::TAU;
        let a1 = f64::from(segment + 1) / f64::from(ring_segments) * std::f64::consts::TAU;
        canvas.draw_stroke(
            point_at(a0, inner_radius),
            point_at(a1, inner_radius),
            ring_color,
            ring_color,
            0.6,
            0.010,
            0.8,
        );
    }

    // Outer hairline ticks every 5% of the run.
    let tick_color = (0.55, 0.0, 0.0);
    for tick in 0..20 {
        let angle = angle_of(tick * steps / 20);
        canvas.draw_stroke(
            point_at(angle, face_radius * 0.985),
            point_at(angle, face_radius),
            tick_color,
            tick_color,
            0.7,
            0.012,
            0.7,
        );
    }

    // Syzygy ticks: from the inner ring outward, colored by the middle body's
    // palette color at that instant; additive splats let clusters bloom.
    for syzygy in &ctx.events().syzygies {
        let angle = angle_of(syzygy.step);
        let length = syzygy.sharpness.max(0.0).sqrt() * MAX_TICK_FRACTION * face_radius;
        let color = ctx.colors[syzygy.middle_body][syzygy.step.min(steps - 1)];
        canvas.draw_stroke(
            point_at(angle, inner_radius),
            point_at(angle, inner_radius + length),
            color,
            color,
            0.9,
            0.030,
            1.0,
        );
    }

    // Center glyph: the initial triangle, scaled into 12% of the face.
    let initial = [ctx.positions[0][0], ctx.positions[1][0], ctx.positions[2][0]];
    let centroid = (initial[0] + initial[1] + initial[2]) / 3.0;
    let extent = initial.iter().map(|point| (point - centroid).norm()).fold(1e-12_f64, f64::max);
    let glyph_scale = face_radius * 0.12 / extent;
    let glyph_point = |index: usize| -> (f32, f32) {
        let offset = initial[index] - centroid;
        ((center + offset.x * glyph_scale) as f32, (center + offset.y * glyph_scale) as f32)
    };
    for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
        canvas.draw_stroke(
            glyph_point(a),
            glyph_point(b),
            ctx.colors[a][0],
            ctx.colors[b][0],
            0.85,
            0.025,
            1.0,
        );
    }

    canvas
}

impl VizMode for SyzygyWheel {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("syzygy-wheel").expect("syzygy-wheel is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        let poster = render_wheel(ctx, ctx.quality.scale_dim(POSTER_SIDE));
        sink.save_png16(&poster.into_png16(clip_black, clip_white, 1.0), "wheel.png")?;

        let icon = render_wheel(ctx, ICON_SIDE);
        sink.save_png16(&icon.into_png16(clip_black, clip_white, 1.0), "wheel_icon.png")?;

        #[derive(serde::Serialize)]
        struct SyzygyRecord {
            step: usize,
            middle_body: usize,
            sharpness: f64,
        }
        let records: Vec<SyzygyRecord> = ctx
            .events()
            .syzygies
            .iter()
            .map(|syzygy| SyzygyRecord {
                step: syzygy.step,
                middle_body: syzygy.middle_body,
                sharpness: syzygy.sharpness,
            })
            .collect();
        let json = serde_json::to_string_pretty(&records).map_err(std::io::Error::other)?;
        sink.write_text("syzygies.json", &json, "data")?;
        Ok(())
    }
}
