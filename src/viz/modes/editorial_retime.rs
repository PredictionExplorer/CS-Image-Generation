//! V22 `editorial-retime` -- Drama-Adaptive Time.
//!
//! The main video re-scheduled: per-frame step advance is inversely
//! proportional to the drama curve, so the film lingers in slow motion at
//! near-collisions and glides through calm arcs. Same trajectory, same
//! master framing and levels, new pacing.

use crate::error::Result;
use crate::render::constants::DEFAULT_DT;
use crate::render::context::RenderContext;
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::velocity_hdr::VelocityHdrCalculator;
use crate::render::{
    AccumulationParams, SpectralScene, accumulate_spectral_steps, default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Output frames at final quality (30 s at 30 fps).
const TOTAL_FRAMES: usize = 900;
/// Drama-to-pace exponent.
const GAMMA: f64 = 0.8;
/// Drama floor keeping calm stretches finite.
const EPSILON: f64 = 0.05;
/// Per-frame step clamp relative to the mean advance (slow, fast).
const CLAMP: (f64, f64) = (1.0 / 6.0, 8.0);

/// Build the frame -> step schedule: cumulative frame density proportional
/// to `(drama + eps)^gamma`, clamped per frame and renormalized to land
/// exactly on the final step. Exported for V49 `broadcast` and V67
/// `vanitas` (their retimed acts share this pacing).
pub(crate) fn build_schedule(drama: &[f64], frames: usize) -> Vec<usize> {
    let steps = drama.len();
    if steps == 0 || frames == 0 {
        return Vec::new();
    }
    let weights: Vec<f64> = drama.iter().map(|value| (value + EPSILON).powf(GAMMA)).collect();
    let total: f64 = weights.iter().sum();
    // Frame boundaries where the cumulative weight crosses each 1/frames
    // quantile.
    let mut boundaries = Vec::with_capacity(frames);
    let mut cumulative = 0.0f64;
    let mut next_quantile = total / frames as f64;
    for (step, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        while cumulative >= next_quantile && boundaries.len() < frames {
            boundaries.push(step + 1);
            next_quantile = total * (boundaries.len() + 1) as f64 / frames as f64;
        }
    }
    while boundaries.len() < frames {
        boundaries.push(steps);
    }
    *boundaries.last_mut().expect("nonempty") = steps;

    // Clamp per-frame advances around the mean and re-integrate.
    let mean = steps as f64 / frames as f64;
    let (lo, hi) = (mean * CLAMP.0, mean * CLAMP.1);
    let mut deltas: Vec<f64> = Vec::with_capacity(frames);
    let mut previous = 0usize;
    for &boundary in &boundaries {
        deltas.push(((boundary - previous) as f64).clamp(lo, hi));
        previous = boundary;
    }
    let delta_total: f64 = deltas.iter().sum();
    let scale = steps as f64 / delta_total;
    let mut schedule = Vec::with_capacity(frames);
    let mut position = 0.0f64;
    for delta in &deltas {
        position += delta * scale;
        schedule.push((position.round() as usize).min(steps));
    }
    *schedule.last_mut().expect("nonempty") = steps;
    schedule
}

/// The editorial-retime mode.
pub struct EditorialRetime;

impl VizMode for EditorialRetime {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("editorial-retime").expect("editorial-retime is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("editorial-retime skipped: trajectory too short");
            return Ok(());
        }
        let frame_count = ctx.quality.scale_count(TOTAL_FRAMES);
        let drama = ctx.events().drama(ctx.kinematics());
        let schedule = build_schedule(&drama, frame_count);

        // Master framing at half resolution, master levels.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let render_ctx =
            RenderContext::new(video_w, video_h, ctx.positions, ctx.settings.aspect_correction);
        let mut spd: Vec<[f64; NUM_BINS]> =
            vec![[0.0; NUM_BINS]; video_w as usize * video_h as usize];
        let velocity_calc = VelocityHdrCalculator::new(ctx.positions, DEFAULT_DT);
        let traits = ctx.settings.traits;
        let hdr_scale = ctx.settings.render_config.hdr_scale;

        info!(
            "   editorial-retime: {frame_count} frames over {steps} steps \
             (drama-adaptive, gamma {GAMMA})"
        );
        let started = std::time::Instant::now();
        let mut logged = false;
        let mut cursor = 0usize;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("retimed.mp4"),
            &sink.path("retimed_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            ctx.levels,
            |frame, rgba| {
                let target = schedule[frame.min(schedule.len() - 1)];
                if target > cursor {
                    accumulate_spectral_steps(
                        &mut spd,
                        &AccumulationParams {
                            scene: SpectralScene::new(ctx.positions, ctx.colors, ctx.body_alphas),
                            ctx: &render_ctx,
                            velocity_calc: &velocity_calc,
                            step_start: cursor,
                            step_end: target,
                            hdr_scale,
                            traits,
                        },
                        default_accumulation_backend(),
                    );
                    cursor = target;
                }
                convert_spd_buffer_to_rgba(&spd, rgba, video_w as usize, video_h as usize);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   retimed: {per_frame:.2}s first frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("retimed.mp4", "video");
        sink.record("retimed_hq.mp4", "video");

        // Frame -> step map (full resolution of the schedule).
        let map: Vec<serde_json::Value> = schedule
            .iter()
            .enumerate()
            .step_by((schedule.len() / 300).max(1))
            .map(|(frame, step)| serde_json::json!([frame, step]))
            .collect();
        let meta = serde_json::json!({
            "frames": frame_count,
            "fps": 30,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "clamp_relative": [CLAMP.0, CLAMP.1],
            "schedule_samples": map,
            "final_step": schedule.last(),
            "note": "exposure swell and substep interpolation deferred; see Wave 7 addendum",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("schedule.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_lands_exactly_on_the_final_step_and_is_monotonic() {
        let mut drama = vec![0.1f64; 10_000];
        for (index, value) in drama.iter_mut().enumerate().take(6_000).skip(4_000) {
            *value = 1.0 - ((index as f64 - 5_000.0) / 1_000.0).powi(2).min(1.0);
        }
        let schedule = build_schedule(&drama, 120);
        assert_eq!(schedule.len(), 120);
        assert_eq!(*schedule.last().expect("nonempty"), 10_000);
        for pair in schedule.windows(2) {
            assert!(pair[1] >= pair[0], "schedule must be monotonic");
        }
        // High-drama middle gets denser frames (smaller step advances).
        let mid_frame = schedule.iter().position(|&step| step >= 5_000).expect("mid");
        let mid_advance = schedule[mid_frame + 1] - schedule[mid_frame];
        let early_advance = schedule[5] - schedule[4];
        assert!(
            mid_advance < early_advance,
            "drama must slow the pace: mid {mid_advance} early {early_advance}"
        );
    }
}
