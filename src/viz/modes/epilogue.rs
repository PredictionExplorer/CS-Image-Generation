//! V23 `epilogue` -- How This Artwork Dies.
//!
//! The simulation continued past the render window to its statistically
//! inevitable ejection: a 15 s recap replays the familiar artwork at 4x,
//! then 30 s of honest physics follow -- one body flung out, the survivors
//! tightening into a binary, the framing widening in segments to hold the
//! fleeing ember. The extended trajectory is re-simulated deterministically
//! and dressed with the recovered master view rotation (drift excluded);
//! if no ejection occurs within budget, the dance simply continues and
//! `fate.json` says so.

use crate::error::Result;
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{BoundingBox, PixelBuffer, RenderContext};
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::velocity_hdr::VelocityHdrCalculator;
use crate::render::{
    AccumulationParams, OklabColor, SpectralScene, accumulate_spectral_steps,
    default_accumulation_backend,
};
use crate::sim::G;
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::resim::{DEFAULT_ESCAPE_THRESHOLD, extended, recover_orientation};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Extension factor over the original recorded window.
const EXTEND_FACTOR: usize = 8;
/// Total video frames at final quality (45 s at 30 fps).
const TOTAL_FRAMES: usize = 1350;
/// Recap share of the video (15 of 45 seconds).
const RECAP_FRACTION: f64 = 15.0 / 45.0;
/// Frames per framing segment (bounds recomputed at this cadence).
const SEGMENT_FRAMES: usize = 60;
/// Zoom margin around the escaper.
const ZOOM_MARGIN: f64 = 1.15;
/// Segment crossfade length in frames.
const CROSSFADE_FRAMES: usize = 12;
/// Splat budget per segment re-projection.
const SPLAT_BUDGET: usize = 200_000;

/// Orbital elements of the surviving pair.
fn binary_elements(bodies: &[crate::sim::Body], escaper: usize) -> Option<(f64, f64)> {
    let survivors: Vec<usize> = (0..bodies.len()).filter(|&index| index != escaper).collect();
    if survivors.len() != 2 {
        return None;
    }
    let (a, b) = (&bodies[survivors[0]], &bodies[survivors[1]]);
    let relative_position = a.position - b.position;
    let relative_velocity = a.velocity - b.velocity;
    let distance = relative_position.norm().max(1e-12);
    let reduced_mass = a.mass * b.mass / (a.mass + b.mass);
    let gravitational = G * a.mass * b.mass;
    let energy = 0.5 * reduced_mass * relative_velocity.norm_squared() - gravitational / distance;
    if energy >= 0.0 {
        return None;
    }
    let semi_major = -gravitational / (2.0 * energy);
    let angular_momentum = reduced_mass * relative_position.cross(&relative_velocity).norm();
    let eccentricity_sq = 1.0
        + 2.0 * energy * angular_momentum * angular_momentum
            / (reduced_mass * gravitational * gravitational);
    Some((semi_major, eccentricity_sq.max(0.0).sqrt()))
}

/// The epilogue mode.
pub struct Epilogue;

impl VizMode for Epilogue {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("epilogue").expect("epilogue is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 || ctx.bodies.len() != 3 {
            warn!("epilogue skipped: trajectory too short or bodies unavailable");
            return Ok(());
        }

        // --- Extended deterministic re-simulation, dressed with the
        // recovered master orientation.
        let sim_started = std::time::Instant::now();
        let mut ext = extended(ctx.bodies, steps, EXTEND_FACTOR, DEFAULT_ESCAPE_THRESHOLD);
        let rotation = recover_orientation(&ext.positions, ctx.positions);
        for body in &mut ext.positions {
            for point in body.iter_mut() {
                *point = rotation * *point;
            }
        }
        let total_steps = ext.positions[0].len();
        info!(
            "   epilogue: extended x{EXTEND_FACTOR} ({total_steps} steps) in {:.1}s, ejection {:?}",
            sim_started.elapsed().as_secs_f64(),
            ext.ejection.map(|e| (e.step, e.escaper))
        );

        // --- Strided scene with held tail colors.
        let stride = (3 * total_steps / SPLAT_BUDGET).max(1);
        let positions_strided: Vec<Vec<Vector3<f64>>> = ext
            .positions
            .iter()
            .map(|body| body.iter().copied().step_by(stride).collect())
            .collect();
        let colors_strided: Vec<Vec<OklabColor>> = (0..3)
            .map(|body| {
                let last = *ctx.colors[body].last().expect("colors nonempty");
                (0..total_steps)
                    .step_by(stride)
                    .map(|step| if step < steps { ctx.colors[body][step] } else { last })
                    .collect()
            })
            .collect();
        let strided_steps = positions_strided[0].len();
        let hdr_scale = ctx.settings.render_config.hdr_scale * stride as f64;
        let traits = ctx.settings.traits;

        // --- Frame -> strided-step schedule: recap at 4x, then epilogue.
        let frame_count = ctx.quality.scale_count(TOTAL_FRAMES);
        let recap_frames = ((frame_count as f64 * RECAP_FRACTION) as usize).max(1);
        let window_strided = (steps / stride).min(strided_steps);
        let step_at_frame = |frame: usize| -> usize {
            if frame < recap_frames {
                ((frame + 1) as f64 / recap_frames as f64 * window_strided as f64) as usize
            } else {
                let t =
                    (frame + 1 - recap_frames) as f64 / (frame_count - recap_frames).max(1) as f64;
                window_strided + (t * (strided_steps - window_strided) as f64) as usize
            }
            .min(strided_steps)
        };

        // --- Segment framing: monotone widening bounds holding both the
        // full trail so far and the escaper with margin.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let bounds_for = |end_step: usize, previous: Option<&BoundingBox>| -> BoundingBox {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for body in &positions_strided {
                for point in body.iter().take(end_step.max(2)) {
                    if point.x.is_finite() && point.y.is_finite() {
                        min_x = min_x.min(point.x);
                        max_x = max_x.max(point.x);
                        min_y = min_y.min(point.y);
                        max_y = max_y.max(point.y);
                    }
                }
            }
            // Escaper containment with margin.
            if let Some(ejection) = ext.ejection {
                let escaper_step = (end_step.saturating_sub(1)).min(strided_steps - 1);
                let escaper = positions_strided[ejection.escaper][escaper_step];
                min_x = min_x.min(escaper.x * ZOOM_MARGIN);
                max_x = max_x.max(escaper.x * ZOOM_MARGIN);
                min_y = min_y.min(escaper.y * ZOOM_MARGIN);
                max_y = max_y.max(escaper.y * ZOOM_MARGIN);
            }
            if let Some(previous) = previous {
                min_x = min_x.min(previous.min_x);
                max_x = max_x.max(previous.max_x);
                min_y = min_y.min(previous.min_y);
                max_y = max_y.max(previous.max_y);
            }
            let pad = 0.04 * (max_x - min_x).max(max_y - min_y).max(1e-9);
            let mut bounds = BoundingBox {
                min_x: min_x - pad,
                max_x: max_x + pad,
                min_y: min_y - pad,
                max_y: max_y + pad,
                width: (max_x - min_x + 2.0 * pad).max(1e-12),
                height: (max_y - min_y + 2.0 * pad).max(1e-12),
            };
            bounds.apply_aspect_correction(video_w, video_h);
            bounds
        };

        let mut spd: Vec<[f64; NUM_BINS]> =
            vec![[0.0; NUM_BINS]; video_w as usize * video_h as usize];
        let mut current_bounds = bounds_for(step_at_frame(0), None);
        let mut render_ctx = RenderContext::with_bounds(video_w, video_h, current_bounds);
        let mut previous_frame: PixelBuffer = Vec::new();
        let mut crossfade_left = 0usize;

        let accumulate = |spd: &mut Vec<[f64; NUM_BINS]>,
                          render_ctx: &RenderContext,
                          range: std::ops::Range<usize>| {
            if range.is_empty() {
                return;
            }
            let velocity_calc = VelocityHdrCalculator::new(&positions_strided, DEFAULT_DT);
            accumulate_spectral_steps(
                spd,
                &AccumulationParams {
                    scene: SpectralScene::new(&positions_strided, &colors_strided, ctx.body_alphas),
                    ctx: render_ctx,
                    velocity_calc: &velocity_calc,
                    step_start: range.start,
                    step_end: range.end.min(strided_steps),
                    hdr_scale,
                    traits,
                },
                default_accumulation_backend(),
            );
        };
        let mut cursor = step_at_frame(0);
        accumulate(&mut spd, &render_ctx, 0..cursor);

        let started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("epilogue.mp4"),
            &sink.path("epilogue_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            ctx.levels,
            |frame, rgba| {
                // Segment boundary: widen the framing and re-project.
                if frame > 0 && frame.is_multiple_of(SEGMENT_FRAMES) {
                    let target = step_at_frame(frame);
                    let widened = bounds_for(target, Some(&current_bounds));
                    let grew = (widened.width - current_bounds.width).abs()
                        > current_bounds.width * 0.01
                        || (widened.height - current_bounds.height).abs()
                            > current_bounds.height * 0.01;
                    if grew {
                        // Snapshot the outgoing framing for the crossfade.
                        previous_frame.clear();
                        previous_frame
                            .resize(video_w as usize * video_h as usize, (0.0, 0.0, 0.0, 0.0));
                        convert_spd_buffer_to_rgba(
                            &spd,
                            &mut previous_frame,
                            video_w as usize,
                            video_h as usize,
                        );
                        crossfade_left = CROSSFADE_FRAMES;
                        current_bounds = widened;
                        render_ctx = RenderContext::with_bounds(video_w, video_h, current_bounds);
                        spd.fill([0.0; NUM_BINS]);
                        accumulate(&mut spd, &render_ctx, 0..cursor);
                    }
                }
                let target = step_at_frame(frame);
                if target > cursor {
                    accumulate(&mut spd, &render_ctx, cursor..target);
                    cursor = target;
                }
                convert_spd_buffer_to_rgba(&spd, rgba, video_w as usize, video_h as usize);
                if crossfade_left > 0 {
                    let blend = crossfade_left as f64 / CROSSFADE_FRAMES as f64;
                    for (pixel, previous) in rgba.iter_mut().zip(previous_frame.iter()) {
                        pixel.0 = pixel.0 * (1.0 - blend) + previous.0 * blend;
                        pixel.1 = pixel.1 * (1.0 - blend) + previous.1 * blend;
                        pixel.2 = pixel.2 * (1.0 - blend) + previous.2 * blend;
                        pixel.3 = pixel.3.max(previous.3);
                    }
                    crossfade_left -= 1;
                }
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   epilogue video: {per_frame:.2}s first frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("epilogue.mp4", "video");
        sink.record("epilogue_hq.mp4", "video");

        // --- Fate sidecar.
        let fate = match ext.ejection {
            Some(ejection) => {
                let elements = binary_elements(&ext.final_bodies, ejection.escaper);
                let strided = ejection.step / stride;
                let ejection_second = if strided <= window_strided {
                    RECAP_FRACTION * 45.0 * strided as f64 / window_strided.max(1) as f64
                } else {
                    15.0 + 30.0 * (strided - window_strided) as f64
                        / (strided_steps - window_strided).max(1) as f64
                };
                serde_json::json!({
                    "outcome": "ejection",
                    "escaper": ejection.escaper,
                    "ejection_step": ejection.step,
                    "ejection_video_second": ejection_second,
                    "binary_semi_major": elements.map(|(a, _)| a),
                    "binary_eccentricity": elements.map(|(_, e)| e),
                })
            }
            None => serde_json::json!({
                "outcome": "no ejection within budget",
                "extend_factor": EXTEND_FACTOR,
            }),
        };
        let meta = serde_json::json!({
            "fate": fate,
            "extend_factor": EXTEND_FACTOR,
            "stride": stride,
            "frames": frame_count,
            "fps": 30,
            "segment_frames": SEGMENT_FRAMES,
            "crossfade_frames": CROSSFADE_FRAMES,
            "note": "V09 score muxing deferred to the sound wave; see Wave 7 addendum",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("fate.json", &json, "data")?;
        Ok(())
    }
}
