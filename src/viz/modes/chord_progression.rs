//! V16 `chord-progression` -- The Harmony of Distances.
//!
//! The three pairwise distances log-mapped to pitch, quantized (with
//! hysteresis) to a just-intonation lattice over a palette-rooted drone:
//! rendered as an illuminated 1:3 score poster, a scrolling-playhead video,
//! and the honest triad audio. Consonance (Tenney height of the reduced
//! ratio triple) warms or sharpens the ink.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio::{
    SAMPLE_RATE, fade_ends, mux, normalize_to_lufs, soft_limit, write_wav_stereo_24bit,
};
use crate::viz::common::display::encode_linear_rec2020_to_u16;
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::style::{Paper, draw_footer, ink_color, paper_color};
use crate::viz::common::text::{Align, Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Score duration in seconds.
const SCORE_SECONDS: f64 = 60.0;
/// Pitch clamp in semitones around the root.
const SEMITONE_CLAMP: f64 = 24.0;
/// Quantization hysteresis in cents.
const HYSTERESIS_CENTS: f64 = 60.0;
/// Just-intonation lattice (odd-limit 9), one octave.
const JI_LATTICE: [(u32, u32); 13] = [
    (1, 1),
    (10, 9),
    (9, 8),
    (6, 5),
    (5, 4),
    (4, 3),
    (7, 5),
    (3, 2),
    (8, 5),
    (5, 3),
    (16, 9),
    (9, 5),
    (15, 8),
];
/// Quantization hop count over the run.
const SEGMENT_HOPS: usize = 1_200;

/// One quantized note segment for a voice.
#[derive(Clone, Copy, Debug)]
struct Note {
    /// Start position in [0, 1] run time.
    start: f64,
    /// End position in [0, 1].
    end: f64,
    /// Lattice index.
    ratio: usize,
    /// Octave offset.
    octave: i32,
}

/// Tenney height of a lattice ratio.
fn tenney(ratio: (u32, u32)) -> f64 {
    f64::from(ratio.0 * ratio.1).log2()
}

/// Quantize a semitone offset to (lattice index, octave) in log space.
fn quantize(semitones: f64) -> (usize, i32) {
    let octave = (semitones / 12.0).floor() as i32;
    let folded = semitones - f64::from(octave) * 12.0;
    let mut best = (0usize, f64::INFINITY);
    for (index, &(n, d)) in JI_LATTICE.iter().enumerate() {
        let cents = (f64::from(n) / f64::from(d)).log2() * 12.0;
        let distance = (cents - folded).abs();
        if distance < best.1 {
            best = (index, distance);
        }
    }
    (best.0, octave)
}

/// The chord-progression mode.
pub struct ChordProgression;

impl VizMode for ChordProgression {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("chord-progression").expect("chord-progression is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("chord-progression skipped: trajectory too short");
            return Ok(());
        }
        let kinematics = ctx.kinematics();

        // Root frequency from the palette anchor hue.
        let root_hz = {
            let mean = ctx.mean_color(0);
            let (_, _, hue) = oklab_to_oklch(mean.0, mean.1, mean.2);
            110.0 * 2.0_f64.powf((hue / 360.0 * 12.0).floor() / 12.0)
        };

        // --- Quantized note segments per voice, with hysteresis.
        let mut voices: Vec<Vec<Note>> = Vec::with_capacity(3);
        let mut consonance = vec![0.0f64; SEGMENT_HOPS];
        let mut held: [(usize, i32); 3] = [(0, 0); 3];
        #[allow(clippy::needless_range_loop)]
        for pair in 0..3 {
            let series = &kinematics.pairwise[pair];
            let mut sorted: Vec<f64> = series.iter().step_by(37).copied().collect();
            sorted.sort_by(f64::total_cmp);
            let median = sorted[sorted.len() / 2].max(1e-12);
            let mut notes: Vec<Note> = Vec::new();
            for hop in 0..SEGMENT_HOPS {
                let step = (hop * steps / SEGMENT_HOPS).min(steps - 1);
                let semis =
                    (-24.0 * (series[step] / median).log2()).clamp(-SEMITONE_CLAMP, SEMITONE_CLAMP);
                let candidate = quantize(semis);
                let current = if hop == 0 { candidate } else { held[pair] };
                let current_cents = (f64::from(JI_LATTICE[current.0].0)
                    / f64::from(JI_LATTICE[current.0].1))
                .log2()
                    * 1200.0
                    + f64::from(current.1) * 1200.0;
                let chosen = if (semis * 100.0 - current_cents).abs() > HYSTERESIS_CENTS {
                    candidate
                } else {
                    current
                };
                held[pair] = chosen;
                let position = hop as f64 / SEGMENT_HOPS as f64;
                match notes.last_mut() {
                    Some(last) if last.ratio == chosen.0 && last.octave == chosen.1 => {
                        last.end = position + 1.0 / SEGMENT_HOPS as f64;
                    }
                    _ => notes.push(Note {
                        start: position,
                        end: position + 1.0 / SEGMENT_HOPS as f64,
                        ratio: chosen.0,
                        octave: chosen.1,
                    }),
                }
                consonance[hop] += tenney(JI_LATTICE[chosen.0]);
            }
            voices.push(notes);
        }
        let consonance_max = consonance.iter().copied().fold(1e-9, f64::max);
        for value in &mut consonance {
            *value = 1.0 - *value / consonance_max;
        }
        info!(
            "   chord-progression: {} + {} + {} note segments, root {root_hz:.1} Hz",
            voices[0].len(),
            voices[1].len(),
            voices[2].len()
        );

        // --- Poster: 1:3 landscape scroll on deep black.
        let poster_h = ctx.quality.scale_dim(1152).max(96) as usize;
        let poster_w = poster_h * 3;
        let paper = Paper::DeepBlack;
        let mut poster = vec![paper_color(paper); poster_w * poster_h];
        let margin = poster_h / 12;
        let lane_h = (poster_h - 2 * margin) / 3;
        // Consonance wash columns.
        let warm = oklch_to_oklab(0.30, 0.055, 75.0);
        let sharp = oklch_to_oklab(0.24, 0.05, 255.0);
        for x in 0..poster_w {
            let hop = (x * SEGMENT_HOPS / poster_w).min(SEGMENT_HOPS - 1);
            let t = consonance[hop];
            let (l, a, b) = (
                sharp.0 + (warm.0 - sharp.0) * t,
                sharp.1 + (warm.1 - sharp.1) * t,
                sharp.2 + (warm.2 - sharp.2) * t,
            );
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            for y in margin..poster_h - margin {
                let pixel = &mut poster[y * poster_w + x];
                pixel.0 = pixel.0 * 0.75 + r.max(0.0) * 0.25;
                pixel.1 = pixel.1 * 0.75 + g.max(0.0) * 0.25;
                pixel.2 = pixel.2 * 0.75 + blue.max(0.0) * 0.25;
            }
        }
        // Barlines at syzygies.
        let ink = ink_color(paper);
        for syzygy in &ctx.events().syzygies {
            let x = (syzygy.step as f64 / steps as f64 * poster_w as f64) as f32;
            draw_line(
                &mut poster,
                poster_w,
                poster_h,
                (x, margin as f32),
                (x, (poster_h - margin) as f32),
                (ink.0 * 0.4, ink.1 * 0.4, ink.2 * 0.4),
                1.2,
                0.7,
            );
        }
        // Note capsules per voice lane.
        let pair_color = |pair: usize| -> Rgb64 {
            let (a, b) = crate::viz::common::kinematics::PAIRS[pair];
            let ca = ctx.mean_color(a);
            let cb = ctx.mean_color(b);
            let (l, aa, bb) = (
                f64::midpoint(ca.0, cb.0) + 0.12,
                f64::midpoint(ca.1, cb.1),
                f64::midpoint(ca.2, cb.2),
            );
            let (r, g, blue) = oklab_to_linear_rec2020(l.min(0.9), aa, bb);
            (r.max(0.0) * 1.6, g.max(0.0) * 1.6, blue.max(0.0) * 1.6)
        };
        for (pair, notes) in voices.iter().enumerate() {
            let lane_top = margin + pair * lane_h;
            let color = pair_color(pair);
            for note in notes {
                let ratio = &JI_LATTICE[note.ratio];
                let semis = (f64::from(ratio.0) / f64::from(ratio.1)).log2() * 12.0
                    + f64::from(note.octave) * 12.0;
                let y_norm = ((semis + SEMITONE_CLAMP) / (2.0 * SEMITONE_CLAMP)).clamp(0.0, 1.0);
                let y = lane_top as f64 + (1.0 - y_norm) * lane_h as f64;
                let x0 = note.start * poster_w as f64;
                let x1 = note.end * poster_w as f64;
                draw_line(
                    &mut poster,
                    poster_w,
                    poster_h,
                    (x0 as f32, y as f32),
                    (x1 as f32, y as f32),
                    color,
                    (lane_h as f64 * 0.10).max(2.0),
                    0.95,
                );
            }
        }
        // Lane labels and footer.
        let label_px = crate::viz::common::style::type_px(poster_h, 0);
        for (pair, label) in ["R12", "R13", "R23"].iter().enumerate() {
            let style = TextStyle {
                align: Align::Left,
                opacity: 0.8,
                ..TextStyle::caption(Face::Mono, label_px, ink)
            };
            draw_text(
                &mut poster,
                poster_w,
                poster_h,
                margin as f64 * 0.35,
                (margin + pair * lane_h) as f64 + label_px * 1.2,
                &style,
                label,
            );
        }
        draw_footer(&mut poster, poster_w, poster_h, ctx, "V16 CHORD-PROGRESSION", paper);
        let mut quantized = Vec::new();
        encode_linear_rec2020_to_u16(&poster, &mut quantized);
        let image = crate::render::ImageBuffer::<crate::render::Rgb<u16>, Vec<u16>>::from_raw(
            poster_w as u32,
            poster_h as u32,
            quantized,
        )
        .expect("poster buffer sized");
        sink.save_png16(&image, "score_poster.png")?;

        // --- Audio: the honest triad.
        let sample_rate = f64::from(SAMPLE_RATE);
        let total_samples = (SCORE_SECONDS * sample_rate) as usize;
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];
        let mut phases = [0.0f64; 9];
        for sample in 0..total_samples {
            let position = sample as f64 / total_samples as f64;
            let hop = ((position * SEGMENT_HOPS as f64) as usize).min(SEGMENT_HOPS - 1);
            let mut frame = (0.0, 0.0);
            for (pair, notes) in voices.iter().enumerate() {
                let note = notes
                    .iter()
                    .find(|n| position >= n.start && position < n.end)
                    .or_else(|| notes.last())
                    .expect("voice has notes");
                let ratio = &JI_LATTICE[note.ratio];
                let frequency = (root_hz * f64::from(ratio.0) / f64::from(ratio.1)
                    * 2.0_f64.powi(note.octave.clamp(-1, 1)))
                .clamp(40.0, 2_000.0);
                let mut voice = 0.0;
                for (partial, gain) in [(1.0, 1.0), (2.0, 0.25), (3.0, 0.12)] {
                    let slot = pair * 3 + (partial as usize - 1);
                    phases[slot] = (phases[slot] + frequency * partial / sample_rate) % 1.0;
                    voice += (std::f64::consts::TAU * phases[slot]).sin() * gain;
                }
                // Consonance opens the voice slightly.
                let amplitude = 0.10 + 0.06 * consonance[hop];
                let (l, r) = crate::viz::common::audio::equal_power_pan(
                    voice * amplitude,
                    (pair as f64 - 1.0) * 0.5,
                );
                frame.0 += l;
                frame.1 += r;
            }
            left[sample] = frame.0;
            right[sample] = frame.1;
        }
        soft_limit(&mut left, &mut right, 1.2);
        normalize_to_lufs(&mut left, &mut right, -16.0, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.08) as usize);
        fade_ends(&mut right, (sample_rate * 0.08) as usize);
        write_wav_stereo_24bit(&sink.path("triad.wav"), &left, &right)?;
        sink.record("triad.wav", "audio");

        // --- Scrolling playhead video, scored.
        let video_h = ((poster_h / 2) & !1).max(16) as u32;
        let view_w = ((video_h as usize * 16 / 9) & !1).max(16) as u32;
        let frame_count = ctx.quality.scale_count((SCORE_SECONDS * 30.0) as usize);
        let silent_web = sink.path("score_silent.mp4");
        let silent_hq = sink.path("score_silent_hq.mp4");
        let outputs = [
            VideoOutputSpec {
                output_file: silent_web.clone(),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: silent_hq.clone(),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        let playhead_x = f64::from(view_w) * 0.30;
        create_videos_from_frames_singlepass(
            view_w,
            video_h,
            30,
            |out| {
                let mut crop = vec![(0.0, 0.0, 0.0); view_w as usize * video_h as usize];
                let mut bytes: Vec<u16> = Vec::new();
                for frame in 0..frame_count {
                    let position = frame as f64 / (frame_count - 1).max(1) as f64;
                    let center = position * poster_w as f64 - playhead_x;
                    for y in 0..video_h as usize {
                        let source_y = y * 2;
                        for x in 0..view_w as usize {
                            let source_x = center as i64 + x as i64 * 2;
                            let pixel = if source_x < 0 || source_x >= poster_w as i64 - 1 {
                                paper_color(paper)
                            } else {
                                poster[source_y.min(poster_h - 1) * poster_w + source_x as usize]
                            };
                            crop[y * view_w as usize + x] = pixel;
                        }
                    }
                    // Fixed playhead hairline.
                    for y in 0..video_h as usize {
                        let x = playhead_x as usize / 2;
                        crop[y * view_w as usize + x] = (1.2, 1.15, 1.0);
                    }
                    encode_linear_rec2020_to_u16(&crop, &mut bytes);
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        for (silent, scored) in [(&silent_web, "score.mp4"), (&silent_hq, "score_hq.mp4")] {
            match mux(silent, &sink.path("triad.wav"), &sink.path(scored)) {
                Ok(()) => {
                    sink.record(scored, "video");
                    let _ = std::fs::remove_file(silent);
                }
                Err(error) => warn!("chord-progression mux failed for {scored}: {error}"),
            }
        }

        let intervals: Vec<serde_json::Value> = voices
            .iter()
            .enumerate()
            .map(|(pair, notes)| {
                serde_json::json!({
                    "pair": pair,
                    "segments": notes
                        .iter()
                        .map(|n| {
                            let (num, den) = JI_LATTICE[n.ratio];
                            serde_json::json!([n.start, n.end, format!("{num}/{den}"), n.octave])
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let meta = serde_json::json!({
            "root_hz": root_hz,
            "lattice_odd_limit": 9,
            "hysteresis_cents": HYSTERESIS_CENTS,
            "voices": intervals,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("intervals.json", &json, "data")?;
        Ok(())
    }
}
