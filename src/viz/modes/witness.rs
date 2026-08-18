//! V65 `witness` -- First Person, Honest Optics.
//!
//! Ride the steadiest body with physical optics: companions are drawn at
//! their *retarded* positions (V29's solver, the same virtual c), their
//! strokes deposit into the frame's SPD with kernels Doppler-transported by
//! `lambda' = lambda (1 + v_r/c)` -- true energy-conserving bin transport,
//! blue-shifted approaches migrating violet-ward -- with relativistic
//! beaming `D^4` clamped at x2.5. Gravitational lensing warps the
//! accumulated trail field every frame, with the lens masses at the
//! companions' *apparent* positions and Einstein radii halved (interior
//! restraint). Accumulation decays comet-style (half-life 2 s): a witness
//! remembers recently. The 45 s film covers the top-drama window; the V16
//! triad, Doppler-bent and binaurally panned by apparent bearing, is what
//! you hear.

use crate::error::Result;
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{BoundingBox, PixelBuffer, RenderContext};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::audio::{
    SAMPLE_RATE, equal_power_pan, fade_ends, mux, normalize_to_lufs, soft_limit,
    write_wav_stereo_24bit,
};
use crate::viz::common::display::SpdCanvas;
use crate::viz::context::VizContext;
use crate::viz::modes::chord_progression::{SCORE_HOPS, note_frequency, quantized_score};
use crate::viz::modes::lensing::{Deflector, deflection, einstein_radii};
use crate::viz::modes::retarded_time::{position_at, retarded_step, virtual_light_speed};
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use rayon::prelude::*;
use tracing::{info, warn};

/// Film length in seconds at final quality (60 fps).
const WITNESS_SECONDS: f64 = 45.0;
/// Frame rate.
const FPS: u32 = 60;
/// The drama window covers this fraction of the run (45 s of a 60 s
/// witness clock over the full trajectory).
const WINDOW_FRACTION: f64 = 0.75;
/// Trail memory half-life in seconds.
const DECAY_HALF_LIFE: f64 = 2.0;
/// Relativistic beaming clamp.
const BEAMING_CLAMP: f64 = 2.5;
/// Einstein radii scale (interior-view restraint).
const THETA_E_FACTOR: f64 = 0.5;
/// Drawing stride in simulation steps.
const DRAW_STRIDE: usize = 2;
/// Base stroke energy.
const BASE_ENERGY: f64 = 0.014;

/// The witness mode.
pub struct Witness;

/// Rotate a relative position into the hero's velocity-heading frame.
fn to_hero_frame(relative: Vector3<f64>, heading: f64) -> Vector3<f64> {
    let (sin, cos) = (-heading).sin_cos();
    Vector3::new(
        relative.x * cos - relative.y * sin,
        relative.x * sin + relative.y * cos,
        relative.z,
    )
}

/// The top-drama contiguous window of `length` steps (max integrated drama).
pub(crate) fn top_drama_window(drama: &[f64], length: usize) -> (usize, usize) {
    let steps = drama.len();
    let length = length.min(steps).max(1);
    let mut sum: f64 = drama.iter().take(length).sum();
    let mut best = (0usize, sum);
    for start in 1..=steps - length {
        sum += drama[start + length - 1] - drama[start - 1];
        if sum > best.1 {
            best = (start, sum);
        }
    }
    (best.0, best.0 + length)
}

/// Doppler wavelength factor and beaming gain from a radial velocity
/// (`v_r > 0` receding): `lambda' = lambda (1 + v_r/c)`, `beam = D^4`
/// with `D = 1/(1 + v_r/c)`, clamped.
pub(crate) fn doppler_terms(v_r: f64, c: f64) -> (f64, f64) {
    let ratio = (v_r / c).clamp(-0.6, 0.6);
    let factor = 1.0 + ratio;
    let d = 1.0 / factor;
    (factor, (d * d * d * d).clamp(1.0 / BEAMING_CLAMP, BEAMING_CLAMP))
}

/// Inverse-map lens warp of an RGBA buffer (single sample per pixel; the
/// motion hides the aliasing the still-mode supersampler removes).
fn lens_rgba(
    source: &PixelBuffer,
    out: &mut PixelBuffer,
    width: usize,
    height: usize,
    deflectors: &[Deflector],
) {
    out.resize(width * height, (0.0, 0.0, 0.0, 0.0));
    out.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
        for (col, pixel) in line.iter_mut().enumerate() {
            let x = col as f64 + 0.5;
            let y = row as f64 + 0.5;
            let (ax, ay) = deflection(deflectors, x, y);
            let sx = x - ax - 0.5;
            let sy = y - ay - 0.5;
            *pixel = if sx < 0.0 || sy < 0.0 || sx > (width - 1) as f64 || sy > (height - 1) as f64
            {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                let tx = sx - x0 as f64;
                let ty = sy - y0 as f64;
                let at = |px: usize, py: usize| source[py * width + px];
                let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
                let lerp = |u: f64, v: f64, t: f64| u * (1.0 - t) + v * t;
                (
                    lerp(lerp(a.0, b.0, tx), lerp(c.0, d.0, tx), ty),
                    lerp(lerp(a.1, b.1, tx), lerp(c.1, d.1, tx), ty),
                    lerp(lerp(a.2, b.2, tx), lerp(c.2, d.2, tx), ty),
                    lerp(lerp(a.3, b.3, tx), lerp(c.3, d.3, tx), ty),
                )
            };
        }
    });
}

impl VizMode for Witness {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("witness").expect("witness is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 1_000 {
            warn!("witness skipped: trajectory too short");
            return Ok(());
        }
        let kinematics = ctx.kinematics();
        let (c_world, c_steps) = virtual_light_speed(kinematics);

        // --- Hero: the steadiest body (the V27 selection).
        let hero = (0..3)
            .min_by(|&a, &b| {
                let variance = |body: usize| {
                    let speeds = &kinematics.speeds[body];
                    let mean = speeds.iter().step_by(97).sum::<f64>()
                        / speeds.iter().step_by(97).count().max(1) as f64;
                    speeds.iter().step_by(97).map(|&speed| (speed - mean).powi(2)).sum::<f64>()
                };
                variance(a).total_cmp(&variance(b))
            })
            .unwrap_or(2);
        let companions: Vec<usize> = (0..3).filter(|&body| body != hero).collect();

        // --- The top-drama window.
        let drama = ctx.events().drama(kinematics);
        let window_len = ((steps as f64 * WINDOW_FRACTION) as usize).max(1_000);
        let (window_start, window_end) = top_drama_window(&drama, window_len);
        info!(
            "   witness: hero body {hero}, window steps {window_start}..{window_end} \
             (c = {c_world:.3} world/time)"
        );

        // --- Heading series and the hero-frame transform of the window.
        let heading_at = |step: usize| -> f64 {
            let velocity = kinematics.velocities[hero][step.min(steps - 1)];
            velocity.y.atan2(velocity.x)
        };
        let transformed: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (window_start..window_end)
                    .map(|step| {
                        let relative = ctx.positions[body][step] - ctx.positions[hero][step];
                        to_hero_frame(relative, heading_at(step))
                    })
                    .collect()
            })
            .collect();

        // --- Framing: companion extents in the hero frame, padded.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &companion in &companions {
            for point in transformed[companion].iter().step_by(7) {
                if point.x.is_finite() && point.y.is_finite() {
                    min_x = min_x.min(point.x);
                    max_x = max_x.max(point.x);
                    min_y = min_y.min(point.y);
                    max_y = max_y.max(point.y);
                }
            }
        }
        if !(min_x.is_finite() && min_y.is_finite()) {
            warn!("witness skipped: degenerate hero-frame bounds");
            return Ok(());
        }
        let pad = 0.06 * (max_x - min_x).max(max_y - min_y).max(1e-9);
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

        // --- The film.
        let frame_count = ctx.quality.scale_count((WITNESS_SECONDS * f64::from(FPS)) as usize);
        let steps_per_frame = ((window_end - window_start) / frame_count.max(1)).max(1);
        let decay_per_frame = 0.5f64.powf(1.0 / (DECAY_HALF_LIFE * f64::from(FPS)));
        let theta = einstein_radii(kinematics.masses, f64::from(video_w.min(video_h)));
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        let mut canvas = SpdCanvas::new(video_w, video_h);
        let mut scratch: PixelBuffer = Vec::new();
        let mut shift_extremes = (1.0f64, 1.0f64);
        let mut deepest: (f64, usize) = (f64::INFINITY, window_start);
        let started = std::time::Instant::now();
        let mut logged = false;

        // Draw the seen strokes for a step range into a canvas; returns the
        // frame's deflectors (apparent companion positions, last step).
        let draw_window = |canvas: &mut SpdCanvas,
                           range: std::ops::Range<usize>,
                           scale: f32,
                           render_ctx: &RenderContext,
                           shift_extremes: &mut (f64, f64),
                           deepest: &mut (f64, usize)|
         -> Vec<Deflector> {
            let mut deflectors = Vec::with_capacity(2);
            let mut start = range.start.max(window_start + DRAW_STRIDE);
            if !start.is_multiple_of(DRAW_STRIDE) {
                start += DRAW_STRIDE - start % DRAW_STRIDE;
            }
            for step in (start..range.end.min(steps - 1)).step_by(DRAW_STRIDE) {
                let observer = ctx.positions[hero][step];
                let heading = heading_at(step);
                let last_of_frame = step + DRAW_STRIDE >= range.end.min(steps - 1);
                for &companion in &companions {
                    let path = &ctx.positions[companion];
                    let retarded_now = retarded_step(observer, path, step, c_steps);
                    let retarded_prev = retarded_step(
                        ctx.positions[hero][step - DRAW_STRIDE],
                        path,
                        step - DRAW_STRIDE,
                        c_steps,
                    );
                    let seen_now = position_at(path, retarded_now);
                    let seen_prev = position_at(path, retarded_prev);
                    // Both endpoints render in the *current* observer frame.
                    let p_now = to_hero_frame(seen_now - observer, heading);
                    let p_prev = to_hero_frame(seen_prev - observer, heading);
                    let (x0, y0) = render_ctx.to_pixel(p_prev.x, p_prev.y);
                    let (x1, y1) = render_ctx.to_pixel(p_now.x, p_now.y);

                    // Radial velocity at emission: recession positive.
                    let emission_step = (retarded_now.floor() as usize).min(steps - 1);
                    let relative = path[emission_step] - ctx.positions[hero][emission_step];
                    let relative_velocity = kinematics.velocities[companion][emission_step]
                        - kinematics.velocities[hero][emission_step];
                    let v_r = relative.dot(&relative_velocity) / relative.norm().max(1e-12);
                    let (factor, beam) = doppler_terms(v_r, c_world);
                    shift_extremes.0 = shift_extremes.0.min(factor);
                    shift_extremes.1 = shift_extremes.1.max(factor);

                    let color =
                        ctx.colors[companion][emission_step.min(ctx.colors[companion].len() - 1)];
                    canvas.draw_stroke_shifted(
                        (x0 * scale, y0 * scale),
                        (x1 * scale, y1 * scale),
                        color,
                        color,
                        0.85,
                        BASE_ENERGY * beam,
                        1.0,
                        factor,
                    );
                    let distance = relative.norm();
                    if distance < deepest.0 {
                        *deepest = (distance, step);
                    }
                    if last_of_frame {
                        let mass_index = companion.min(2);
                        deflectors.push(Deflector {
                            x: f64::from(x1 * scale),
                            y: f64::from(y1 * scale),
                            theta_e_sq: (theta[mass_index] * THETA_E_FACTOR * f64::from(scale))
                                .powi(2),
                        });
                    }
                }
                // The witness herself: a quiet ember at the origin.
                let (hx, hy) = render_ctx.to_pixel(0.0, 0.0);
                if step.is_multiple_of(DRAW_STRIDE * 8) {
                    canvas.draw_stroke(
                        (hx * scale, hy * scale),
                        (hx * scale + 0.6, hy * scale),
                        ctx.colors[hero][step],
                        ctx.colors[hero][step],
                        0.7,
                        BASE_ENERGY * 0.5,
                        0.8,
                    );
                }
            }
            deflectors
        };

        // --- Levels from a steady-state prepass: the decayed memory holds
        // roughly the last two half-lives of strokes, so draw exactly that
        // window once, meter it, and clear (the canvas defines its own
        // exposure -- the production scene meter does not know these
        // stroke energies).
        let levels = {
            let steady_steps = ((2.0 * DECAY_HALF_LIFE / WITNESS_SECONDS)
                * (window_end - window_start) as f64) as usize;
            let mut extremes_scratch = (1.0f64, 1.0f64);
            let mut deepest_scratch = (f64::INFINITY, window_start);
            let _ = draw_window(
                &mut canvas,
                window_end.saturating_sub(steady_steps.max(steps_per_frame))..window_end,
                1.0,
                &render_ctx,
                &mut extremes_scratch,
                &mut deepest_scratch,
            );
            let metered = canvas.levels(clip_black, clip_white, 1.15);
            canvas.clear();
            metered
        };

        let mut cursor = window_start;
        stream_video(
            video_w,
            video_h,
            FPS,
            &sink.path("witness.mp4"),
            &sink.path("witness_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                canvas.decay_spd(decay_per_frame);
                let target = (window_start + (frame + 1) * steps_per_frame).min(window_end);
                let deflectors = draw_window(
                    &mut canvas,
                    cursor..target,
                    1.0,
                    &render_ctx,
                    &mut shift_extremes,
                    &mut deepest,
                );
                cursor = target;
                canvas.convert_into(&mut scratch);
                lens_rgba(&scratch, rgba, video_w as usize, video_h as usize, &deflectors);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   witness: {per_frame:.2}s first frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("witness.mp4", "video");
        sink.record("witness_hq.mp4", "video");

        // --- Hero still at full resolution: the deepest approach, with the
        // decayed memory replayed up to that instant.
        {
            let still_w = ctx.quality.scale_dim(ctx.width);
            let still_h = ctx.quality.scale_dim(ctx.height);
            let mut still_bounds = bounds;
            still_bounds.apply_aspect_correction(still_w, still_h);
            let still_ctx = RenderContext::with_bounds(still_w, still_h, still_bounds);
            let scale = 1.0f32;
            let mut still_canvas = SpdCanvas::new(still_w, still_h);
            let deep_step = deepest.1.max(window_start + steps_per_frame);
            let mut extremes = shift_extremes;
            let mut deepest_still = deepest;
            let mut chunk_start = window_start;
            let mut still_deflectors = Vec::new();
            while chunk_start < deep_step {
                let chunk_end = (chunk_start + steps_per_frame).min(deep_step);
                still_canvas.decay_spd(decay_per_frame);
                still_deflectors = draw_window(
                    &mut still_canvas,
                    chunk_start..chunk_end,
                    scale,
                    &still_ctx,
                    &mut extremes,
                    &mut deepest_still,
                );
                chunk_start = chunk_end;
            }
            let mut linear: PixelBuffer = Vec::new();
            still_canvas.convert_into(&mut linear);
            // Rescale deflectors for the still's pixel grid.
            let ratio = f64::from(still_w) / f64::from(video_w);
            for lens in &mut still_deflectors {
                lens.x *= ratio;
                lens.y *= ratio;
                lens.theta_e_sq *= ratio * ratio;
            }
            let mut lensed: PixelBuffer = Vec::new();
            lens_rgba(&linear, &mut lensed, still_w as usize, still_h as usize, &still_deflectors);
            let levels_still =
                crate::viz::common::display::auto_levels(&lensed, clip_black, clip_white, 1.1);
            let image = crate::viz::common::display::grade_with_levels(
                &lensed,
                still_w,
                still_h,
                &levels_still,
            );
            sink.save_png16(&image, "witness_still.png")?;
        }

        // --- Audio: the V16 triad with per-pair Doppler and binaural pan.
        let score = quantized_score(ctx);
        let sample_rate = f64::from(SAMPLE_RATE);
        let duration = frame_count as f64 / f64::from(FPS);
        let total_samples = (duration * sample_rate) as usize;
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];
        let pairs = crate::viz::common::kinematics::PAIRS;
        let mut phases = [0.0f64; 9];
        let mut doppler = [1.0f64; 3];
        let mut pans = [0.0f64; 3];
        for sample in 0..total_samples {
            let t = sample as f64 / total_samples as f64;
            let step = window_start
                + ((t * (window_end - window_start) as f64) as usize)
                    .min(window_end - window_start - 1);
            if sample.is_multiple_of(256) {
                for (voice, &(a, b)) in pairs.iter().enumerate() {
                    // Pair radial rate: d(separation)/dt over a short span.
                    let series = &kinematics.pairwise[voice];
                    let ahead = (step + 32).min(steps - 1);
                    let v_r = (series[ahead] - series[step])
                        / ((ahead - step).max(1) as f64 * DEFAULT_DT);
                    doppler[voice] = 1.0 / (1.0 + (v_r / c_world).clamp(-0.6, 0.6));
                    // Apparent bearing in the hero frame: the pair midpoint
                    // (or the companion itself for hero-involving pairs).
                    let heading = heading_at(step);
                    let observer = ctx.positions[hero][step];
                    let apparent = |body: usize| -> Vector3<f64> {
                        to_hero_frame(ctx.positions[body][step] - observer, heading)
                    };
                    let target = if a == hero {
                        apparent(b)
                    } else if b == hero {
                        apparent(a)
                    } else {
                        (apparent(a) + apparent(b)) * 0.5
                    };
                    pans[voice] = (target.y.atan2(target.x).sin() * 0.8).clamp(-1.0, 1.0);
                }
            }
            let hop = ((window_start as f64 + t * (window_end - window_start) as f64)
                / steps as f64
                * SCORE_HOPS as f64) as usize;
            let position = hop as f64 / SCORE_HOPS as f64;
            let mut frame = (0.0, 0.0);
            for (voice, notes) in score.voices.iter().enumerate() {
                let note = notes
                    .iter()
                    .find(|n| position >= n.start && position < n.end)
                    .or_else(|| notes.last())
                    .expect("voice has notes");
                let frequency =
                    (note_frequency(score.root_hz, note) * doppler[voice]).clamp(40.0, 2_400.0);
                let mut tone = 0.0;
                for (partial, gain) in [(1.0, 1.0), (2.0, 0.25), (3.0, 0.12)] {
                    let slot = voice * 3 + (partial as usize - 1);
                    phases[slot] = (phases[slot] + frequency * partial / sample_rate) % 1.0;
                    tone += (std::f64::consts::TAU * phases[slot]).sin() * gain;
                }
                let (l, r) = equal_power_pan(tone * 0.12, pans[voice]);
                frame.0 += l;
                frame.1 += r;
            }
            left[sample] = frame.0;
            right[sample] = frame.1;
        }
        soft_limit(&mut left, &mut right, 1.2);
        normalize_to_lufs(&mut left, &mut right, -16.0, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.3) as usize);
        fade_ends(&mut right, (sample_rate * 0.3) as usize);
        let mix_path = sink.path("witness_mix.wav");
        write_wav_stereo_24bit(&mix_path, &left, &right)?;
        for name in ["witness.mp4", "witness_hq.mp4"] {
            let silent = sink.path(name);
            let scored = sink.path(&format!("{}.scored.mp4", name.trim_end_matches(".mp4")));
            match mux(&silent, &mix_path, &scored) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&silent);
                    let _ = std::fs::rename(&scored, &silent);
                }
                Err(error) => warn!("witness mux failed for {name}: {error}; keeping silent cut"),
            }
        }
        let _ = std::fs::remove_file(&mix_path);

        let meta = serde_json::json!({
            "hero_body": hero,
            "virtual_c_world_per_time": c_world,
            "window_steps": [window_start, window_end],
            "decay_half_life_s": DECAY_HALF_LIFE,
            "theta_e_factor": THETA_E_FACTOR,
            "beaming_clamp": BEAMING_CLAMP,
            "wavelength_factor_extremes": [shift_extremes.0, shift_extremes.1],
            "deepest_approach_step": deepest.1,
            "doppler_convention": "recession positive; lambda' = lambda (1 + v_r/c)",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("optics.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doppler_terms_shift_and_beam_in_the_physical_directions() {
        let c = 10.0;
        // Approaching (v_r < 0): violet-ward factor < 1, brighter beam > 1.
        let (factor_in, beam_in) = doppler_terms(-2.0, c);
        assert!(factor_in < 1.0 && beam_in > 1.0);
        // Receding: red-ward, dimmer.
        let (factor_out, beam_out) = doppler_terms(2.0, c);
        assert!(factor_out > 1.0 && beam_out < 1.0);
        // Rest: identity.
        let (factor_rest, beam_rest) = doppler_terms(0.0, c);
        assert!((factor_rest - 1.0).abs() < 1e-15 && (beam_rest - 1.0).abs() < 1e-15);
        // Clamp: an impossible closing speed cannot exceed x2.5.
        let (_, beam_max) = doppler_terms(-9.0, c);
        assert!(beam_max <= BEAMING_CLAMP + 1e-12);
    }

    #[test]
    fn top_drama_window_finds_the_hot_stretch() {
        let mut drama = vec![0.1f64; 10_000];
        for value in drama.iter_mut().skip(6_000).take(2_000) {
            *value = 1.0;
        }
        let (start, end) = top_drama_window(&drama, 2_500);
        assert_eq!(end - start, 2_500);
        assert!((5_500..=6_000).contains(&start), "window should cover the hot stretch: {start}");
    }

    #[test]
    fn hero_frame_rotation_points_velocity_along_x() {
        let heading = 0.7f64;
        let velocity = Vector3::new(heading.cos(), heading.sin(), 0.0) * 3.0;
        let rotated = to_hero_frame(velocity, heading);
        assert!((rotated.y).abs() < 1e-12, "heading must map onto +x, got {rotated:?}");
        assert!(rotated.x > 2.9);
    }

    #[test]
    fn lens_warp_preserves_flat_fields_far_from_the_lens() {
        // A constant field warps to itself where deflection is smooth.
        let (width, height) = (32usize, 32usize);
        let source: PixelBuffer = vec![(0.5, 0.4, 0.3, 1.0); width * height];
        let mut out = PixelBuffer::new();
        let lenses = [Deflector { x: 16.0, y: 16.0, theta_e_sq: 4.0 }];
        lens_rgba(&source, &mut out, width, height, &lenses);
        // Away from the lens and the frame edge the field is unchanged.
        let pixel = out[4 * width + 4];
        assert!((pixel.0 - 0.5).abs() < 1e-6 && (pixel.3 - 1.0).abs() < 1e-6);
    }
}
