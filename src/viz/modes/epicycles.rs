//! V17 `epicycles` -- The Impossible Machine.
//!
//! Each body's complex path Fourier-decomposed into rotating circles; the
//! three epicycle machines draw the orbit live -- fine armature segments,
//! hairline circles for the dominant terms, and a persistent crisp trail
//! graded with the master levels. If the top-K reconstruction misses the
//! path by more than 1.5 px RMS, K escalates (logged, recorded).

use crate::error::Result;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::SpdCanvas;
use crate::viz::common::raster::draw_line_rgba;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rustfft::{FftPlanner, num_complex::Complex};
use tracing::{info, warn};

/// Path resample length (power of two for the FFT).
const RESAMPLE: usize = 8_192;
/// Initial coefficient count per body.
const BASE_K: usize = 96;
/// Maximum escalated coefficient count.
const MAX_K: usize = 768;
/// Reconstruction tolerance in pixels (at video scale).
const RMS_TOLERANCE_PX: f64 = 1.5;
/// Video frames at final quality (45 s at 30 fps).
const TOTAL_FRAMES: usize = 1_350;
/// Hairline circles drawn for the largest terms.
const CIRCLES_SHOWN: usize = 12;

/// One rotating term of a machine.
#[derive(Clone, Copy, Debug)]
struct Term {
    /// Signed rotations per traversal.
    frequency: f64,
    /// Complex amplitude.
    coefficient: Complex<f64>,
}

/// Decompose one body's path into its top-K terms.
fn decompose(samples: &[Complex<f64>], k: usize) -> Vec<Term> {
    let n = samples.len();
    let mut buffer = samples.to_vec();
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);
    let mut terms: Vec<Term> = buffer
        .iter()
        .enumerate()
        .map(|(index, &c)| {
            let signed = if index <= n / 2 { index as f64 } else { index as f64 - n as f64 };
            Term { frequency: signed, coefficient: c / n as f64 }
        })
        .collect();
    terms.sort_by(|a, b| b.coefficient.norm().total_cmp(&a.coefficient.norm()));
    terms.truncate(k);
    terms
}

/// Partial-sum chain positions of a machine at parameter `t` in [0, 1].
fn chain_at(terms: &[Term], t: f64) -> Vec<Complex<f64>> {
    let mut chain = Vec::with_capacity(terms.len() + 1);
    let mut sum = Complex::new(0.0, 0.0);
    chain.push(sum);
    for term in terms {
        let angle = std::f64::consts::TAU * term.frequency * t;
        sum += term.coefficient * Complex::new(angle.cos(), angle.sin());
        chain.push(sum);
    }
    chain
}

/// The epicycles mode.
pub struct Epicycles;

impl VizMode for Epicycles {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("epicycles").expect("epicycles is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < RESAMPLE / 4 {
            warn!("epicycles skipped: trajectory too short");
            return Ok(());
        }
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let render_ctx =
            RenderContext::new(video_w, video_h, ctx.positions, ctx.settings.aspect_correction);

        // --- Decompose each body in pixel space (uniform resample).
        let mut machines: Vec<Vec<Term>> = Vec::with_capacity(3);
        let mut k_used = [BASE_K; 3];
        let mut rms_report = [0.0f64; 3];
        for body in 0..3 {
            let samples: Vec<Complex<f64>> = (0..RESAMPLE)
                .map(|index| {
                    let step = (index * steps / RESAMPLE).min(steps - 1);
                    let point = ctx.positions[body][step];
                    let (px, py) = render_ctx.to_pixel(point.x, point.y);
                    Complex::new(f64::from(px), f64::from(py))
                })
                .collect();
            let mut k = BASE_K;
            loop {
                let terms = decompose(&samples, k);
                // RMS reconstruction error over a probe set.
                let rms = {
                    let mut sum_sq = 0.0;
                    let probes = 512usize;
                    for probe in 0..probes {
                        let t = probe as f64 / probes as f64;
                        let reconstructed = *chain_at(&terms, t).last().expect("chain nonempty");
                        let actual = samples[(t * RESAMPLE as f64) as usize % RESAMPLE];
                        sum_sq += (reconstructed - actual).norm_sqr();
                    }
                    (sum_sq / probes as f64).sqrt()
                };
                if rms <= RMS_TOLERANCE_PX || k >= MAX_K {
                    if rms > RMS_TOLERANCE_PX {
                        warn!("epicycles: body {body} RMS {rms:.2}px at K={k} (cap reached)");
                    }
                    k_used[body] = k;
                    rms_report[body] = rms;
                    machines.push(terms);
                    break;
                }
                k *= 2;
            }
        }
        info!(
            "   epicycles: K = {:?}, reconstruction RMS = [{:.2}, {:.2}, {:.2}] px",
            k_used, rms_report[0], rms_report[1], rms_report[2]
        );

        // --- Video: machines drawing the trail.
        let frame_count = ctx.quality.scale_count(TOTAL_FRAMES);
        let mut trail = SpdCanvas::new(video_w, video_h);
        let mut previous_pens: Vec<Complex<f64>> =
            machines.iter().map(|terms| *chain_at(terms, 0.0).last().expect("chain")).collect();
        let armature_ink = (0.055, 0.055, 0.062);
        let started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("epicycles.mp4"),
            &sink.path("epicycles_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            ctx.levels,
            |frame, rgba| {
                let t1 = (frame + 1) as f64 / frame_count as f64;
                let t0 = frame as f64 / frame_count as f64;
                // Advance trails with sub-steps for smooth strokes.
                for (body, terms) in machines.iter().enumerate() {
                    let color = ctx.mean_color(body);
                    let mut pen = previous_pens[body];
                    for substep in 1..=8 {
                        let t = t0 + (t1 - t0) * f64::from(substep) / 8.0;
                        let next = *chain_at(terms, t).last().expect("chain");
                        trail.draw_stroke(
                            (pen.re as f32, pen.im as f32),
                            (next.re as f32, next.im as f32),
                            color,
                            color,
                            0.85,
                            0.045,
                            1.0,
                        );
                        pen = next;
                    }
                    previous_pens[body] = pen;
                }
                trail.convert_into(rgba);
                // Armature overlay: chain segments + hairline circles.
                for terms in &machines {
                    let chain = chain_at(terms, t1);
                    for pair in chain.windows(2) {
                        draw_line_rgba(
                            rgba,
                            video_w as usize,
                            video_h as usize,
                            (pair[0].re as f32, pair[0].im as f32),
                            (pair[1].re as f32, pair[1].im as f32),
                            armature_ink,
                            1.0,
                            0.5,
                        );
                    }
                    for (index, term) in terms.iter().take(CIRCLES_SHOWN).enumerate() {
                        let center = chain[index];
                        let radius = term.coefficient.norm();
                        if radius < 1.0 {
                            continue;
                        }
                        let mut previous: Option<(f32, f32)> = None;
                        for sample in 0..=24 {
                            let angle = f64::from(sample) / 24.0 * std::f64::consts::TAU;
                            let point = (
                                (center.re + radius * angle.cos()) as f32,
                                (center.im + radius * angle.sin()) as f32,
                            );
                            if let Some(prev) = previous {
                                draw_line_rgba(
                                    rgba,
                                    video_w as usize,
                                    video_h as usize,
                                    prev,
                                    point,
                                    armature_ink,
                                    0.8,
                                    0.35,
                                );
                            }
                            previous = Some(point);
                        }
                    }
                }
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   epicycles: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("epicycles.mp4", "video");
        sink.record("epicycles_hq.mp4", "video");

        // --- Still: completed trail with the machine frozen at the
        // closest triple approach (video resolution, per the Wave 4
        // precedent for video-derived stills).
        let freeze_t = ctx.events().closest_triple as f64 / steps as f64;
        let mut rgba = Vec::new();
        trail.convert_into(&mut rgba);
        for terms in &machines {
            let chain = chain_at(terms, freeze_t);
            for pair in chain.windows(2) {
                draw_line_rgba(
                    &mut rgba,
                    video_w as usize,
                    video_h as usize,
                    (pair[0].re as f32, pair[0].im as f32),
                    (pair[1].re as f32, pair[1].im as f32),
                    (0.09, 0.09, 0.10),
                    1.1,
                    0.7,
                );
            }
        }
        let image =
            crate::viz::common::display::grade_with_levels(&rgba, video_w, video_h, ctx.levels);
        sink.save_png16(&image, "epicycles_still.png")?;

        // --- Coefficients sidecar.
        let coefficients: Vec<serde_json::Value> = machines
            .iter()
            .enumerate()
            .map(|(body, terms)| {
                serde_json::json!({
                    "body": body,
                    "k": k_used[body],
                    "rms_px": rms_report[body],
                    "terms": terms
                        .iter()
                        .take(BASE_K)
                        .map(|t| serde_json::json!([t.frequency, t.coefficient.re, t.coefficient.im]))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let meta = serde_json::json!({
            "resample": RESAMPLE,
            "base_k": BASE_K,
            "circles_shown": CIRCLES_SHOWN,
            "frames": frame_count,
            "machines": coefficients,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("coefficients.json", &json, "data")?;
        Ok(())
    }
}
