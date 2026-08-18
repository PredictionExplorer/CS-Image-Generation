//! V48 `mission-control` -- The 1969 Broadcast.
//!
//! A retro CRT multi-panel telemetry film: a phosphor vector scope drawing
//! the live orbit, three strip-chart recorders scribbling the pairwise
//! distances with overshooting spring pens, a teletype event ticker typing
//! syzygies and periapses character by character, and a mission clock --
//! all real data, dressed as period hardware with a barrel-distorted,
//! scanlined CRT finish.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::RenderContext;
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_to_u16;
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::text::{Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Film length in seconds at 60 fps.
const FILM_SECONDS: usize = 30;
/// Teletype speed in characters per second.
const TELETYPE_CPS: f64 = 24.0;
/// Barrel distortion strength.
const BARREL: f64 = 0.015;
/// Pen spring damping ratio.
const PEN_ZETA: f64 = 0.55;
/// Pen spring natural frequency (rad per video second).
const PEN_OMEGA: f64 = 26.0;

/// Second-order spring follower (the strip-chart pen).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SpringPen {
    /// Current pen position.
    pub position: f64,
    velocity: f64,
}

impl SpringPen {
    /// Advance toward `target` by `dt` seconds.
    pub(crate) fn advance(&mut self, target: f64, dt: f64) {
        let acceleration = PEN_OMEGA * PEN_OMEGA * (target - self.position)
            - 2.0 * PEN_ZETA * PEN_OMEGA * self.velocity;
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;
    }
}

/// The mission-control mode.
pub struct MissionControl;

impl VizMode for MissionControl {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("mission-control").expect("mission-control is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("mission-control skipped: trajectory too short");
            return Ok(());
        }
        let width = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(64) as usize;
        let height = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(64) as usize;
        let frame_count = ctx.quality.scale_count(FILM_SECONDS * 60);
        let mut rng = ctx.fork_rng("mission-control");
        let jitter_seed = rng.next_u64();

        // --- Panel geometry.
        let scope_w = width * 55 / 100;
        let bottom_h = height * 14 / 100;
        let chart_h = (height - bottom_h) / 3;
        let chart_x = scope_w + 2;
        let chart_w = width - chart_x;

        // --- Phosphor palette: P31 green tinted toward the palette accent.
        let phosphor = {
            let mean = ctx.mean_color(0);
            let (_, _, hue) = oklab_to_oklch(mean.0, mean.1, mean.2);
            let blended_hue = 140.0 + (hue - 140.0) * 0.25;
            let (l, a, b) = oklch_to_oklab(0.78, 0.15, blended_hue);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), blue.max(0.0))
        };
        let panel_bg: Rgb64 = (0.004, 0.006, 0.005);
        let grid_ink: Rgb64 = (phosphor.0 * 0.10, phosphor.1 * 0.10, phosphor.2 * 0.10);

        // --- Scope framing over the whole trajectory.
        let scope_ctx = RenderContext::new(
            scope_w as u32,
            (height - bottom_h) as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );

        // --- Teletype event feed.
        let kinematics = ctx.kinematics();
        let mut feed: Vec<(usize, String)> = Vec::new();
        for syzygy in &ctx.events().syzygies {
            feed.push((
                syzygy.step,
                format!(
                    "T+{met:07.1} SYZ B{body} SHARP {sharp:.2}",
                    met = syzygy.step as f64 / 10.0,
                    body = syzygy.middle_body + 1,
                    sharp = syzygy.sharpness
                ),
            ));
        }
        for approach in &ctx.events().periapses {
            feed.push((
                approach.step,
                format!(
                    "T+{met:07.1} PERIAPSIS P{a}{b} R={r:.4}",
                    met = approach.step as f64 / 10.0,
                    a = approach.pair.0 + 1,
                    b = approach.pair.1 + 1,
                    r = approach.distance
                ),
            ));
        }
        feed.sort_by_key(|(step, _)| *step);
        let full_log: String =
            feed.iter().map(|(_, line)| line.as_str()).collect::<Vec<_>>().join("\n");
        sink.write_text("mission_log.txt", &full_log, "data")?;

        // Deterministic sync-jitter schedule (about one glitch per 4 s).
        let jitter_frames: Vec<usize> = (0..frame_count)
            .filter(|frame| {
                let mut state = (*frame as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ jitter_seed;
                state ^= state >> 33;
                state.wrapping_mul(0xFF51_AFD7_ED55_8CCD).is_multiple_of(240)
            })
            .collect();

        // --- Chart normalization windows.
        let chart_windows: Vec<(f64, f64)> = (0..3)
            .map(|pair| {
                let mut sample: Vec<f64> =
                    kinematics.pairwise[pair].iter().step_by(37).copied().collect();
                sample.sort_by(f64::total_cmp);
                let lo = sample[(sample.len() - 1) / 20];
                let hi = sample[(sample.len() - 1) * 19 / 20];
                (lo, hi.max(lo + 1e-9))
            })
            .collect();

        let mono_px = crate::viz::common::style::type_px(height, 0).max(9.0);
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("mission_control.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("mission_control_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        let started = std::time::Instant::now();
        let mut logged = false;

        // Persistent state across frames.
        let mut phosphor_field = vec![0.0f32; scope_w * (height - bottom_h)];
        let mut charts: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let mut pens = [SpringPen::default(); 3];
        let mut composed = vec![panel_bg; width * height];
        let mut warped = vec![panel_bg; width * height];
        let mut bytes: Vec<u16> = Vec::new();

        create_videos_from_frames_singlepass(
            width as u32,
            height as u32,
            60,
            |out| {
                for frame in 0..frame_count {
                    let progress = frame as f64 / (frame_count - 1).max(1) as f64;
                    let step = ((progress * steps as f64) as usize).min(steps - 1);

                    // Phosphor decay + fresh vectors.
                    for value in &mut phosphor_field {
                        *value *= 0.90;
                    }
                    let trail_span = (steps / frame_count.max(1)).max(1) * 2;
                    for body in 0..3 {
                        for sub in 0..trail_span {
                            let s = step.saturating_sub(sub);
                            let point = ctx.positions[body][s];
                            let (px, py) = scope_ctx.to_pixel(point.x, point.y);
                            let (ix, iy) = (px as i64, py as i64);
                            if ix >= 0
                                && iy >= 0
                                && (ix as usize) < scope_w
                                && (iy as usize) < height - bottom_h
                            {
                                let slot = &mut phosphor_field[iy as usize * scope_w + ix as usize];
                                *slot = (*slot + 0.5).min(3.0);
                            }
                        }
                    }

                    // Compose the panel.
                    composed.fill(panel_bg);
                    // Scope grid + phosphor.
                    for gx in 0..=8 {
                        let x = gx * scope_w / 8;
                        draw_line(
                            &mut composed,
                            width,
                            height,
                            (x as f32, 0.0),
                            (x as f32, (height - bottom_h) as f32),
                            grid_ink,
                            1.0,
                            1.0,
                        );
                    }
                    for gy in 0..=6 {
                        let y = gy * (height - bottom_h) / 6;
                        draw_line(
                            &mut composed,
                            width,
                            height,
                            (0.0, y as f32),
                            (scope_w as f32, y as f32),
                            grid_ink,
                            1.0,
                            1.0,
                        );
                    }
                    for y in 0..height - bottom_h {
                        for x in 0..scope_w {
                            let glow = f64::from(phosphor_field[y * scope_w + x]);
                            if glow > 1e-3 {
                                let pixel = &mut composed[y * width + x];
                                pixel.0 += phosphor.0 * glow * 0.6;
                                pixel.1 += phosphor.1 * glow * 0.6;
                                pixel.2 += phosphor.2 * glow * 0.6;
                            }
                        }
                    }

                    // Strip charts with spring pens.
                    let dt = 1.0 / 60.0;
                    for pair in 0..3 {
                        let (lo, hi) = chart_windows[pair];
                        let target = 1.0
                            - ((kinematics.pairwise[pair][step] - lo) / (hi - lo)).clamp(0.0, 1.0);
                        pens[pair].advance(target, dt);
                        charts[pair].push(pens[pair].position.clamp(-0.15, 1.15));
                        let chart_top = pair * chart_h;
                        // Frame + baseline grid.
                        for gy in [0, chart_h - 1] {
                            draw_line(
                                &mut composed,
                                width,
                                height,
                                (chart_x as f32, (chart_top + gy) as f32),
                                (width as f32, (chart_top + gy) as f32),
                                grid_ink,
                                1.0,
                                1.0,
                            );
                        }
                        let history = &charts[pair];
                        let visible = chart_w.min(history.len());
                        for offset in 1..visible {
                            let index = history.len() - visible + offset;
                            let x0 = (chart_x + chart_w - visible + offset - 1) as f32;
                            let x1 = (chart_x + chart_w - visible + offset) as f32;
                            let y0 =
                                chart_top as f64 + history[index - 1] * (chart_h - 6) as f64 + 3.0;
                            let y1 = chart_top as f64 + history[index] * (chart_h - 6) as f64 + 3.0;
                            draw_line(
                                &mut composed,
                                width,
                                height,
                                (x0, y0 as f32),
                                (x1, y1 as f32),
                                phosphor,
                                1.3,
                                0.85,
                            );
                        }
                    }

                    // Ticker: lines whose events have happened, typed at 24 cps.
                    let elapsed = frame as f64 / 60.0;
                    let typed_budget = (elapsed * TELETYPE_CPS) as usize;
                    let mut consumed = 0usize;
                    let mut visible_lines: Vec<String> = Vec::new();
                    for (event_step, line) in &feed {
                        if *event_step > step {
                            break;
                        }
                        if consumed + line.len() <= typed_budget {
                            consumed += line.len();
                            visible_lines.push(line.clone());
                        } else {
                            let remaining = typed_budget.saturating_sub(consumed);
                            let partial: String = line.chars().take(remaining).collect();
                            visible_lines.push(format!("{partial}\u{2588}"));
                            break;
                        }
                    }
                    let ticker_style = TextStyle::new(Face::Mono, mono_px, phosphor);
                    let ticker_top = height - bottom_h;
                    for (row, line) in visible_lines.iter().rev().take(2).enumerate() {
                        draw_text(
                            &mut composed,
                            width,
                            height,
                            8.0,
                            (ticker_top + bottom_h / 2 + row * (bottom_h / 3)) as f64
                                - mono_px * 0.2,
                            &ticker_style,
                            line,
                        );
                    }
                    draw_text(
                        &mut composed,
                        width,
                        height,
                        width as f64 - mono_px * 12.0,
                        (ticker_top + bottom_h / 2) as f64,
                        &ticker_style,
                        &format!("MET T+{:07.1}", step as f64 / 10.0),
                    );

                    // CRT finish: barrel + scanlines + drift + sync jitter.
                    let drift = 1.0 + 0.03 * (progress * std::f64::consts::TAU * 1.7).sin();
                    let jitter = if jitter_frames.contains(&frame) { 2 } else { 0 };
                    let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
                    let norm = cx * cx + cy * cy;
                    for y in 0..height {
                        let scan = if y % 2 == 0 { 1.0 } else { 0.82 };
                        for x in 0..width {
                            let dx = x as f64 - cx;
                            let dy = y as f64 - cy;
                            let factor = 1.0 + BARREL * (dx * dx + dy * dy) / norm;
                            let sx = (cx + dx * factor) as i64 + jitter;
                            let sy = (cy + dy * factor) as i64;
                            let sample = if sx >= 0
                                && sy >= 0
                                && (sx as usize) < width
                                && (sy as usize) < height
                            {
                                composed[sy as usize * width + sx as usize]
                            } else {
                                panel_bg
                            };
                            let gain = scan * drift;
                            warped[y * width + x] =
                                (sample.0 * gain, sample.1 * gain, sample.2 * gain);
                        }
                    }
                    encode_linear_rec2020_to_u16(&warped, &mut bytes);
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    if frame == 0 && !logged {
                        logged = true;
                        let per_frame = started.elapsed().as_secs_f64();
                        info!(
                            "   mission-control: {per_frame:.2}s/frame, projected {:.1} min",
                            per_frame * frame_count as f64 / 60.0
                        );
                    }
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("mission_control.mp4", "video");
        sink.record("mission_control_hq.mp4", "video");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spring_pen_converges_with_overshoot() {
        let mut pen = SpringPen::default();
        let mut overshot = false;
        for _ in 0..600 {
            pen.advance(1.0, 1.0 / 60.0);
            if pen.position > 1.0 {
                overshot = true;
            }
        }
        assert!(overshot, "an underdamped pen must overshoot");
        assert!((pen.position - 1.0).abs() < 0.01, "pen settles at the target");
    }
}
