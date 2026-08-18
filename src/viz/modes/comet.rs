//! V21 `comet` -- Forever Redrawing.
//!
//! A sliding-window video: the SPD decays exponentially between frames, so
//! the trio become comets with luminous fading tails, endlessly erasing and
//! redrawing. The loop closes by fading the final second toward a snapshot
//! of the first frame's state, and the most dramatic window is exported as a
//! full-resolution still.

use crate::error::Result;
use crate::render::context::PixelBuffer;
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, resized_config, scene_levels, stream_video};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;

/// Trail half-life in seconds of video time.
const HALF_LIFE_SECONDS: f64 = 1.2;
/// Video duration in seconds at 60 fps.
const VIDEO_SECONDS: usize = 30;
/// Loop crossfade length in frames.
const LOOP_FADE_FRAMES: usize = 60;
/// Pre-roll length in half-lives before the video window starts.
const PREROLL_HALF_LIVES: f64 = 4.0;

/// The comet decay-trail mode.
pub struct Comet;

impl VizMode for Comet {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("comet").expect("comet is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 60);
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        // Timeline: pre-roll settles the decay before frame 0, then the
        // video window covers the rest of the run.
        let video_fraction =
            1.0 / (1.0 + PREROLL_HALF_LIVES * HALF_LIFE_SECONDS / VIDEO_SECONDS as f64);
        let preroll_steps = (steps as f64 * (1.0 - video_fraction)) as usize;
        let window_steps = steps - preroll_steps;
        let steps_per_frame = (window_steps / frame_count.max(1)).max(1);
        let half_life_frames = HALF_LIFE_SECONDS * 60.0;
        let decay_per_frame = 0.5f64.powf(1.0 / half_life_frames);

        // Half-resolution accumulator for the video.
        let half_w = (ctx.width / 2) & !1;
        let half_h = (ctx.height / 2) & !1;
        let mut accumulator = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            half_w,
            half_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );

        // Fixed levels from the transformed scene at video size, biased a
        // touch brighter because the decayed window carries less energy than
        // a full accumulation.
        let resized = resized_config(ctx.settings.resolved_config, half_w, half_h);
        let render_config = *ctx.settings.render_config;
        let settings = crate::render::SpectralRenderSettings::new(
            &resized,
            &render_config,
            ctx.settings.aspect_correction,
        )
        .with_traits(ctx.settings.traits);
        let mut levels = scene_levels(accumulator.scene(), settings, 1.0);
        levels.exposure_scale *= 1.35;

        // Pre-roll with per-chunk decay to reach steady state at frame 0.
        let chunk = steps_per_frame.max(1);
        let mut cursor = 0usize;
        while cursor < preroll_steps {
            accumulator.decay(decay_per_frame);
            let end = (cursor + chunk).min(preroll_steps);
            accumulator.accumulate(cursor..end);
            cursor = end;
        }
        let frame0_snapshot: Vec<[f64; NUM_BINS]> = {
            // Advance one frame so the snapshot equals the displayed frame 0.
            accumulator.decay(decay_per_frame);
            let end = (cursor + steps_per_frame).min(steps);
            accumulator.accumulate(cursor..end);
            cursor = end;
            accumulator.snapshot()
        };

        // Stream the video: decay + accumulate per frame; final second fades
        // toward the frame-0 snapshot for a seamless loop.
        let mut scratch: Vec<[f64; NUM_BINS]> = Vec::new();
        let mut cursor_cell = cursor;
        stream_video(
            half_w,
            half_h,
            60,
            &sink.path("comet.mp4"),
            &sink.path("comet_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba: &mut PixelBuffer| {
                if frame > 0 {
                    accumulator.decay(decay_per_frame);
                    let end = (cursor_cell + steps_per_frame).min(steps);
                    accumulator.accumulate(cursor_cell..end);
                    cursor_cell = end;
                }
                let fade_start = frame_count.saturating_sub(LOOP_FADE_FRAMES);
                if frame >= fade_start {
                    let t = (frame - fade_start + 1) as f64 / LOOP_FADE_FRAMES as f64;
                    scratch.clear();
                    scratch.extend_from_slice(accumulator.spd());
                    scratch.par_iter_mut().zip(frame0_snapshot.par_iter()).for_each(
                        |(current, target)| {
                            for (value, &goal) in current.iter_mut().zip(target.iter()) {
                                *value = *value * (1.0 - t) + goal * t;
                            }
                        },
                    );
                    convert_spd_buffer_to_rgba(&scratch, rgba, half_w as usize, half_h as usize);
                } else {
                    accumulator.convert_into(rgba);
                }
            },
        )?;
        sink.record("comet.mp4", "video");
        sink.record("comet_hq.mp4", "video");

        // Full-resolution still: decayed window ending at the closest triple
        // approach (the most dramatic configuration).
        let drama_step = ctx.events().closest_triple.min(steps.saturating_sub(1));
        let window =
            (steps as f64 * HALF_LIFE_SECONDS / VIDEO_SECONDS as f64 * PREROLL_HALF_LIVES) as usize;
        let start = drama_step.saturating_sub(window);
        let mut still = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            ctx.width,
            ctx.height,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );
        let still_chunk = (window / 32).max(1);
        let chunk_decay = 0.5f64.powf(still_chunk as f64 / (steps as f64 * 0.04));
        let mut position = start;
        while position < drama_step {
            still.decay(chunk_decay);
            let end = (position + still_chunk).min(drama_step);
            still.accumulate(position..end);
            position = end;
        }
        let rgba = still.convert();
        let mut still_levels =
            crate::viz::common::display::auto_levels(&rgba, clip_black, clip_white, 1.0);
        still_levels.exposure_scale *= 1.1;
        let image = crate::viz::common::display::grade_with_levels(
            &rgba,
            ctx.width,
            ctx.height,
            &still_levels,
        );
        sink.save_png16(&image, "comet_still.png")?;

        let meta = serde_json::json!({
            "half_life_seconds": HALF_LIFE_SECONDS,
            "preroll_steps": preroll_steps,
            "loop_fade_frames": LOOP_FADE_FRAMES,
            "still_center_step": drama_step,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("comet.json", &json, "data")?;
        Ok(())
    }
}
