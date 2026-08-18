//! V28 `bullet-time` -- The Held Breath.
//!
//! Freeze at the closest triple approach: 4 s of normal incremental time,
//! a 14 s camera sweep 360 degrees around the frozen light-sculpture (the
//! accumulation up to that instant, world-rotated per frame like the orbit
//! renderer), then time resumes at 4x to the end. All segments share the
//! master levels and one view-invariant tilted framing, so the freeze cuts
//! are frame-exact. The sweep re-splat uses an adaptive step stride with
//! energy compensation to hold a fixed per-frame splat budget (the fixed
//! stride 3 of the spec is replaced after the V46 cost measurements).

use crate::error::Result;
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{BoundingBox, RenderContext};
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::velocity_hdr::VelocityHdrCalculator;
use crate::render::visual_profile::SpikeTraits;
use crate::render::{
    AccumulationParams, LineVertex, OklabColor, SpectralLineSegment, SpectralScene,
    accumulate_spectral_steps, apply_diffraction_spikes, default_accumulation_backend,
    draw_line_segment_aa_spectral,
};
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Total video frames at final quality (24 s at 60 fps).
const TOTAL_FRAMES: usize = 1440;
/// Approach / sweep / release durations in seconds (of 24).
const SEGMENT_SECONDS: (f64, f64, f64) = (4.0, 14.0, 6.0);
/// Camera tilt during the sweep, in degrees.
const TILT_DEG: f64 = 12.0;
/// Release-segment time acceleration.
const RELEASE_RATE: f64 = 4.0;
/// Per-sweep-frame splat budget (segments across all bodies).
const SPLAT_BUDGET: usize = 100_000;
/// Normal-time pacing: the whole run mapped onto this many video seconds.
const NORMAL_TIME_SECONDS: f64 = 30.0;

/// Yaw rotation about the image-vertical (+y) axis.
fn yaw(point: Vector3<f64>, angle: f64) -> Vector3<f64> {
    let (s, c) = angle.sin_cos();
    Vector3::new(c * point.x + s * point.z, point.y, -s * point.x + c * point.z)
}

/// Camera tilt about the +x axis.
fn tilt(point: Vector3<f64>, angle: f64) -> Vector3<f64> {
    let (s, c) = angle.sin_cos();
    Vector3::new(point.x, c * point.y - s * point.z, s * point.y + c * point.z)
}

/// Smoothstep ease.
fn ease(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The simulation steps shown by the film's first and last frames, from
/// the mode's own schedule constants. Exported for V49 `broadcast` (its
/// Act III/V match cuts need the poses at bullet-time's fixed ends).
pub(crate) fn boundary_steps(ctx: &VizContext<'_>, quality_frames: usize) -> (usize, usize) {
    let steps = ctx.step_count();
    if steps < 100 {
        return (0, steps.saturating_sub(1));
    }
    let freeze_step = ctx.events().closest_triple.clamp(steps / 20, steps - 2);
    let stride = (3 * freeze_step / SPLAT_BUDGET).max(1);
    let strided_steps = steps.div_ceil(stride);
    let freeze_strided = (freeze_step / stride).min(strided_steps - 1);
    let total_seconds = SEGMENT_SECONDS.0 + SEGMENT_SECONDS.1 + SEGMENT_SECONDS.2;
    let a_frames =
        ((quality_frames as f64 * SEGMENT_SECONDS.0 / total_seconds).round() as usize).max(1);
    let b_frames =
        ((quality_frames as f64 * SEGMENT_SECONDS.1 / total_seconds).round() as usize).max(1);
    let c_frames = quality_frames.saturating_sub(a_frames + b_frames).max(1);
    let rate = (strided_steps as f64 / (NORMAL_TIME_SECONDS * 60.0)).max(1.0 / 240.0);
    let approach_span = ((a_frames as f64 * rate) as usize).min(freeze_strided);
    let opening = (freeze_strided - approach_span) * stride;
    let closing = ((freeze_strided + (c_frames as f64 * rate * RELEASE_RATE) as usize)
        .min(strided_steps.saturating_sub(1)))
        * stride;
    (opening.min(steps - 1), closing.min(steps - 1))
}

/// The bullet-time mode.
pub struct BulletTime;

impl VizMode for BulletTime {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("bullet-time").expect("bullet-time is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("bullet-time skipped: trajectory too short");
            return Ok(());
        }
        let freeze_step = ctx.events().closest_triple.clamp(steps / 20, steps - 2);

        // --- Strided scene with energy compensation.
        let stride = (3 * freeze_step / SPLAT_BUDGET).max(1);
        let positions_strided: Vec<Vec<Vector3<f64>>> = ctx
            .positions
            .iter()
            .map(|body| body.iter().copied().step_by(stride).collect())
            .collect();
        let colors_strided: Vec<Vec<OklabColor>> =
            ctx.colors.iter().map(|body| body.iter().copied().step_by(stride).collect()).collect();
        let strided_steps = positions_strided[0].len();
        let freeze_strided = (freeze_step / stride).min(strided_steps - 1);
        let hdr_scale = ctx.settings.render_config.hdr_scale * stride as f64;
        let tilt_rad = TILT_DEG.to_radians();

        // --- One view-invariant framing for every segment: the union of
        // projected extents of the frozen set over the full yaw sweep at
        // the fixed tilt (the orbit renderer's exact-bounds construction).
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let (cos_t, sin_t) = (tilt_rad.cos(), tilt_rad.sin().abs());
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for body in &positions_strided {
            for point in body {
                if !(point.x.is_finite() && point.y.is_finite() && point.z.is_finite()) {
                    continue;
                }
                let radius = point.x.hypot(point.z);
                min_x = min_x.min(-radius);
                max_x = max_x.max(radius);
                min_y = min_y.min(cos_t * point.y - sin_t * radius);
                max_y = max_y.max(cos_t * point.y + sin_t * radius);
            }
        }
        if !(min_x.is_finite() && min_y.is_finite()) {
            warn!("bullet-time skipped: degenerate bounds");
            return Ok(());
        }
        let pad = 0.03 * (max_x - min_x).max(max_y - min_y).max(1e-9);
        let mut bounds = BoundingBox {
            min_x: min_x - pad,
            max_x: max_x + pad,
            min_y: min_y - pad,
            max_y: max_y + pad,
            width: (max_x - min_x + 2.0 * pad).max(1e-12),
            height: (max_y - min_y + 2.0 * pad).max(1e-12),
        };
        bounds.apply_aspect_correction(video_w, video_h);
        let render_ctx = RenderContext::with_bounds(video_w, video_h, bounds);

        // --- Segment frame counts.
        let frame_count = ctx.quality.scale_count(TOTAL_FRAMES);
        let total_seconds = SEGMENT_SECONDS.0 + SEGMENT_SECONDS.1 + SEGMENT_SECONDS.2;
        let a_frames =
            ((frame_count as f64 * SEGMENT_SECONDS.0 / total_seconds).round() as usize).max(1);
        let b_frames =
            ((frame_count as f64 * SEGMENT_SECONDS.1 / total_seconds).round() as usize).max(1);
        let c_frames = frame_count.saturating_sub(a_frames + b_frames).max(1);

        // Normal-time pacing: rate in strided steps per frame.
        let rate = (strided_steps as f64 / (NORMAL_TIME_SECONDS * 60.0)).max(1.0 / 240.0);
        let approach_span = ((a_frames as f64 * rate) as usize).min(freeze_strided);
        let approach_start = freeze_strided - approach_span;

        info!(
            "   bullet-time: freeze step {freeze_step} (strided {freeze_strided}/{strided_steps}, \
             stride {stride}), segments {a_frames}/{b_frames}/{c_frames} at {video_w}x{video_h}"
        );

        // --- Accumulation state and helpers.
        let mut spd: Vec<[f64; NUM_BINS]> =
            vec![[0.0; NUM_BINS]; video_w as usize * video_h as usize];
        let traits = ctx.settings.traits;
        let body_alphas = ctx.body_alphas.to_vec();
        let mut transformed: Vec<Vec<Vector3<f64>>> =
            std::iter::repeat_with(|| Vec::with_capacity(strided_steps)).take(3).collect();

        // Freeze-frame body cores (drawn during the sweep with the spike
        // finish forced on).
        let core_colors: [OklabColor; 3] = std::array::from_fn(|body| {
            ctx.colors[body][freeze_step.min(ctx.colors[body].len() - 1)]
        });
        let spikes = SpikeTraits {
            strength: ctx.settings.traits.spikes.strength.max(0.85),
            arms: if ctx.settings.traits.spikes.arms == 0 {
                4
            } else {
                ctx.settings.traits.spikes.arms
            },
            ..ctx.settings.traits.spikes
        };

        let accumulate = |spd: &mut Vec<[f64; NUM_BINS]>,
                          positions: &[Vec<Vector3<f64>>],
                          colors: &[Vec<OklabColor>],
                          range: std::ops::Range<usize>| {
            if range.is_empty() {
                return;
            }
            let velocity_calc = VelocityHdrCalculator::new(positions, DEFAULT_DT);
            accumulate_spectral_steps(
                spd,
                &AccumulationParams {
                    scene: SpectralScene::new(positions, colors, &body_alphas),
                    ctx: &render_ctx,
                    velocity_calc: &velocity_calc,
                    step_start: range.start,
                    step_end: range.end.min(positions[0].len()),
                    hdr_scale,
                    traits,
                },
                default_accumulation_backend(),
            );
        };

        // Tilted (yaw 0) scene used by the approach and release segments.
        let positions_tilted: Vec<Vec<Vector3<f64>>> = positions_strided
            .iter()
            .map(|body| body.iter().map(|&p| tilt(p, tilt_rad)).collect())
            .collect();

        // Pre-fill the sculpture up to the approach window.
        accumulate(&mut spd, &positions_tilted, &colors_strided, 0..approach_start);

        let started = std::time::Instant::now();
        let mut logged = false;
        let mut cursor = approach_start;
        let mut release_cursor = freeze_strided;
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("bullet_time.mp4"),
            &sink.path("bullet_time_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            ctx.levels,
            |frame, rgba| {
                if frame < a_frames {
                    // Segment A: incremental approach at normal pace.
                    let target = approach_start
                        + (((frame + 1) as f64 / a_frames as f64) * approach_span as f64) as usize;
                    let end = target.min(freeze_strided);
                    accumulate(&mut spd, &positions_tilted, &colors_strided, cursor..end);
                    cursor = end;
                    convert_spd_buffer_to_rgba(&spd, rgba, video_w as usize, video_h as usize);
                } else if frame < a_frames + b_frames {
                    // Segment B: the held breath. Re-splat the frozen set
                    // rotated by the eased yaw; the final sweep frame lands
                    // back at yaw 0 == the approach framing.
                    let sweep_t = (frame - a_frames) as f64 / b_frames as f64;
                    let angle = std::f64::consts::TAU * ease(sweep_t);
                    for (body, output) in transformed.iter_mut().enumerate() {
                        output.clear();
                        output.extend(
                            positions_strided[body].iter().map(|&p| tilt(yaw(p, angle), tilt_rad)),
                        );
                    }
                    spd.fill([0.0; NUM_BINS]);
                    accumulate(&mut spd, &transformed, &colors_strided, 0..freeze_strided);
                    // Burning cores at the frozen instant.
                    for body in 0..3 {
                        let world =
                            tilt(yaw(positions_strided[body][freeze_strided], angle), tilt_rad);
                        let (px, py) = render_ctx.to_pixel(world.x, world.y);
                        let vertex = LineVertex {
                            x: px,
                            y: py,
                            z: world.z as f32,
                            color: core_colors[body],
                            alpha: (body_alphas[body] * 220.0).min(0.9),
                        };
                        draw_line_segment_aa_spectral(
                            &mut spd,
                            video_w,
                            video_h,
                            SpectralLineSegment {
                                start: vertex,
                                end: vertex,
                                hdr_scale: hdr_scale * 2.0,
                                thickness_factor: 5.0,
                            },
                        );
                    }
                    convert_spd_buffer_to_rgba(&spd, rgba, video_w as usize, video_h as usize);
                    apply_diffraction_spikes(rgba, video_w as usize, video_h as usize, &spikes);
                } else {
                    // Segment C: time resumes at 4x from the frozen state.
                    if release_cursor == freeze_strided && cursor != freeze_strided {
                        // The sweep's last frame left the buffer at the
                        // frozen yaw-0 state already; nothing to rebuild.
                        cursor = freeze_strided;
                    }
                    let release_frame = frame - a_frames - b_frames;
                    let target = freeze_strided
                        + (((release_frame + 1) as f64) * rate * RELEASE_RATE) as usize;
                    let end = target.min(strided_steps);
                    accumulate(&mut spd, &positions_tilted, &colors_strided, release_cursor..end);
                    release_cursor = end;
                    convert_spd_buffer_to_rgba(&spd, rgba, video_w as usize, video_h as usize);
                }
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   bullet-time: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("bullet_time.mp4", "video");
        sink.record("bullet_time_hq.mp4", "video");

        let meta = serde_json::json!({
            "freeze_step": freeze_step,
            "stride": stride,
            "tilt_deg": TILT_DEG,
            "segments_frames": [a_frames, b_frames, c_frames],
            "release_rate": RELEASE_RATE,
            "splat_budget": SPLAT_BUDGET,
            "note": "adaptive stride replaces the spec's fixed stride 3 (V46 cost lesson)",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("bullet_time.json", &json, "data")?;
        Ok(())
    }
}
