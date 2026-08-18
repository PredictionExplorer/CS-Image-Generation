//! V07 `braid` -- The Orbit Is a Braid.
//!
//! Projects the three bodies onto the trajectory's principal separation axis
//! and draws position-vs-time as three woven strands with true over/under
//! crossings decided by depth. The crossing sequence is also extracted as a
//! braid word (the orbit's topological signature).

use crate::error::Result;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::SpdCanvas;
use crate::viz::common::kinematics::principal_axis_xy;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use serde::Serialize;

/// Canvas width at final quality (height is 3x for a 1:3 hanging scroll).
const BASE_WIDTH: u32 = 1152;
/// Horizontal margin fraction of canvas width on each side.
const MARGIN_FRACTION: f32 = 0.08;
/// Over/under gap half-height in pixels at final quality.
const CROSSING_GAP_PX: f64 = 14.0;
/// Minimum step separation between recorded crossings of the same pair.
const CROSSING_MERGE_STEPS: usize = 300;
/// Stroke energy per segment (auto-levels normalizes overall exposure).
const STROKE_ENERGY: f64 = 0.05;

/// One strand crossing event.
#[derive(Clone, Copy, Debug, Serialize)]
struct Crossing {
    /// Simulation step of the crossing.
    step: usize,
    /// The two bodies that swap order.
    bodies: (usize, usize),
    /// Body passing in front (greater depth toward the viewer).
    over_body: usize,
    /// Artin generator index (1 or 2) at crossing time.
    generator: usize,
    /// True for a positive (sigma) crossing, false for inverse.
    positive: bool,
}

/// The braid visualization mode.
pub struct Braid;

impl VizMode for Braid {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("braid").expect("braid is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        let width = ctx.quality.scale_dim(BASE_WIDTH);
        let height = width * 3;
        let samples = (height as usize * 2).min(steps.max(2));

        // Projected scalar series per body, sampled uniformly in time.
        let axis = principal_axis_xy(ctx.positions);
        let step_of = |sample: usize| sample * (steps - 1) / (samples - 1).max(1);
        let projected: Vec<Vec<f64>> = (0..3)
            .map(|body| {
                (0..samples)
                    .map(|sample| {
                        let point = ctx.positions[body][step_of(sample)];
                        point.x * axis.0 + point.y * axis.1
                    })
                    .collect()
            })
            .collect();

        let (mut u_min, mut u_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for series in &projected {
            for &value in series {
                u_min = u_min.min(value);
                u_max = u_max.max(value);
            }
        }
        let u_range = (u_max - u_min).max(1e-12);

        // Crossing detection with per-pair merge windows.
        let mut crossings: Vec<(usize, usize, usize)> = Vec::new(); // (sample, a, b)
        for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let mut last: Option<usize> = None;
            for sample in 1..samples {
                let before = projected[a][sample - 1] - projected[b][sample - 1];
                let after = projected[a][sample] - projected[b][sample];
                if before.signum() != after.signum() && before != 0.0 {
                    let step = step_of(sample);
                    if last.is_some_and(|prev| step - prev < CROSSING_MERGE_STEPS) {
                        continue;
                    }
                    crossings.push((sample, a, b));
                    last = Some(step);
                }
            }
        }
        crossings.sort_unstable_by_key(|&(sample, ..)| sample);

        // Braid word: track rank order (left-to-right along the axis),
        // over-strand = greater z (closer to the viewer).
        let mut word = Vec::new();
        for &(sample, a, b) in &crossings {
            let step = step_of(sample);
            let mut order: Vec<usize> = (0..3).collect();
            order.sort_by(|&lhs, &rhs| {
                projected[lhs][sample - 1].total_cmp(&projected[rhs][sample - 1])
            });
            let rank_a = order.iter().position(|&body| body == a).unwrap_or(0);
            let rank_b = order.iter().position(|&body| body == b).unwrap_or(0);
            if rank_a.abs_diff(rank_b) != 1 {
                continue; // simultaneous multi-crossing artifact; skip
            }
            let left_body = if rank_a < rank_b { a } else { b };
            let over_body =
                if ctx.positions[a][step].z >= ctx.positions[b][step].z { a } else { b };
            word.push(Crossing {
                step,
                bodies: (a, b),
                over_body,
                generator: rank_a.min(rank_b) + 1,
                positive: over_body == left_body,
            });
        }

        // Masked sample intervals for under-strands at each crossing.
        let gap_samples =
            ((CROSSING_GAP_PX / f64::from(height)) * samples as f64 * 2.0).ceil() as usize;
        let mut masked: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 3];
        for crossing in &word {
            let under = if crossing.over_body == crossing.bodies.0 {
                crossing.bodies.1
            } else {
                crossing.bodies.0
            };
            let sample = crossing.step * (samples - 1) / (steps - 1).max(1);
            let start = sample.saturating_sub(gap_samples);
            let end = (sample + gap_samples).min(samples - 1);
            masked[under].push((start, end));
        }

        // Draw strands.
        let mut canvas = SpdCanvas::new(width, height);
        let kinematics = ctx.kinematics();
        let window = kinematics.speed_window();
        let margin = f64::from(width) * f64::from(MARGIN_FRACTION);
        let x_of = |value: f64| -> f32 {
            (margin + (value - u_min) / u_range * (f64::from(width) - 2.0 * margin)) as f32
        };
        let y_of = |sample: usize| -> f32 {
            (sample as f64 / (samples - 1).max(1) as f64 * f64::from(height - 1)) as f32
        };
        let is_masked = |body: usize, sample: usize| -> bool {
            masked[body].iter().any(|&(start, end)| sample >= start && sample <= end)
        };

        for (body, series) in projected.iter().enumerate() {
            for sample in 0..samples - 1 {
                if is_masked(body, sample) || is_masked(body, sample + 1) {
                    continue;
                }
                let step = step_of(sample);
                let speed = kinematics.speeds[body][step];
                let normalized = kinematics.normalized_speed(window, speed);
                let thickness = 1.3 + (0.62 - 1.3) * normalized;
                canvas.draw_stroke(
                    (x_of(series[sample]), y_of(sample)),
                    (x_of(series[sample + 1]), y_of(sample + 1)),
                    ctx.colors[body][step],
                    ctx.colors[body][step_of(sample + 1)],
                    0.85,
                    STROKE_ENERGY,
                    thickness,
                );
            }
        }

        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let image = canvas.into_png16(clip_black, clip_white, 1.0);
        sink.save_png16(&image, "braid.png")?;

        // Braid word artifacts.
        use std::fmt::Write as _;
        let mut text = String::from("# Braid word (Artin generators; ' marks the inverse)\n");
        for crossing in &word {
            let mark = if crossing.positive { "" } else { "'" };
            let _ = write!(text, "s{}{} ", crossing.generator, mark);
        }
        text.push('\n');
        sink.write_text("braid_word.txt", &text, "data")?;
        let json = serde_json::to_string_pretty(&word).map_err(std::io::Error::other)?;
        sink.write_text("crossings.json", &json, "data")?;
        Ok(())
    }
}
