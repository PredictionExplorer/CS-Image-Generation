//! V49 `broadcast` -- The Five-Act Short Film.
//!
//! The three-minute auto-edited film: Act I discovery (mission control),
//! Act II the dance (the main render, crossfading into its drama retime),
//! Act III revelation (co-rotating morph), Act IV the held breath (bullet
//! time), Act V the end (epilogue, then the master still and the seed
//! title). Acts join on *match cuts*: at every boundary the trim points are
//! chosen to minimize triangle-pose distance between the outgoing and
//! incoming frames (each act's frame -> step map is reproduced from its
//! mode's own schedule; the EDL records the chosen frames and distances).
//! One continuous score runs across the acts -- a drone bed that arcs per
//! act, the V09 strain sidechained in from Act IV as sub-bass, a silence
//! beat before Act V, and the V16 triad as the final chord -- baked into a
//! single WAV, so the compositor's single-audio path suffices (the
//! `amix`/`adelay` graphs deferred in Wave 8 remain unnecessary).
//!
//! V49 consumes upstream artifacts and never re-renders; missing sources
//! skip the film with a warning (core-package safety).

use crate::error::Result;
use crate::render::{VideoEncodingOptions, VideoOutputSpec};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio::{
    Adsr, Biquad, SAMPLE_RATE, fade_ends, fm_bell, normalize_to_lufs, resample_linear,
    saw_bandlimited, soft_limit, write_wav_stereo_24bit,
};
use crate::viz::common::compositor::{Segment, Transition, assemble, timeline_seconds};
use crate::viz::common::display::encode_linear_rec2020_to_u16;
use crate::viz::common::style::{Paper, ink_color, paper_color, type_px};
use crate::viz::common::text::{Align, Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::modes::chord_progression::{note_frequency, quantized_score};
use crate::viz::modes::gw_chirp::strain_series;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Target film length at final quality (masterpiece bar: 180 s +/- 2).
const TARGET_SECONDS: f64 = 180.0;
/// Output frame rate.
const FPS: u32 = 30;
/// Act numeral card hold.
const ACT_CARD_SECONDS: f64 = 0.8;
/// Match-cut search granularity in seconds.
const SEARCH_STEP: f64 = 0.1;
/// The main production video length in seconds (the trailer's contract).
const MAIN_SECONDS: f64 = 30.0;

/// The broadcast mode.
pub struct Broadcast;

/// Rotation/translation/scale-invariant triangle pose: the three side
/// lengths sorted and normalized by the perimeter, plus a log-size term so
/// tight and wide configurations do not alias.
pub(crate) fn shape_pose(positions: &[Vec<Vector3<f64>>], step: usize) -> [f64; 4] {
    let step = step.min(positions[0].len().saturating_sub(1));
    let a = (positions[0][step] - positions[1][step]).norm();
    let b = (positions[0][step] - positions[2][step]).norm();
    let c = (positions[1][step] - positions[2][step]).norm();
    let mut sides = [a, b, c];
    sides.sort_by(f64::total_cmp);
    let perimeter = (a + b + c).max(1e-12);
    [sides[0] / perimeter, sides[1] / perimeter, sides[2] / perimeter, perimeter.ln() * 0.05]
}

/// Distance between two triangle poses.
pub(crate) fn pose_distance(positions: &[Vec<Vector3<f64>>], step_a: usize, step_b: usize) -> f64 {
    let pa = shape_pose(positions, step_a);
    let pb = shape_pose(positions, step_b);
    pa.iter().zip(pb.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// One chosen match cut: (outgoing seconds, incoming seconds, distance).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatchCut {
    pub(crate) out_seconds: f64,
    pub(crate) in_seconds: f64,
    pub(crate) distance: f64,
}

/// Grid-search the (outgoing, incoming) second pair minimizing pose
/// distance; either side may be a fixed single point.
pub(crate) fn best_match_cut(
    positions: &[Vec<Vector3<f64>>],
    out_range: (f64, f64),
    out_step_at: &dyn Fn(f64) -> usize,
    in_range: (f64, f64),
    in_step_at: &dyn Fn(f64) -> usize,
) -> MatchCut {
    let samples = |range: (f64, f64)| -> Vec<f64> {
        let (lo, hi) = range;
        if hi <= lo + 1e-9 {
            return vec![lo];
        }
        let count = ((hi - lo) / SEARCH_STEP).round() as usize + 1;
        (0..count).map(|index| lo + index as f64 * SEARCH_STEP).collect()
    };
    let mut best =
        MatchCut { out_seconds: out_range.0, in_seconds: in_range.0, distance: f64::MAX };
    for &out_s in &samples(out_range) {
        let out_step = out_step_at(out_s);
        for &in_s in &samples(in_range) {
            let distance = pose_distance(positions, out_step, in_step_at(in_s));
            if distance < best.distance {
                best = MatchCut { out_seconds: out_s, in_seconds: in_s, distance };
            }
        }
    }
    best
}

/// Per-window RMS level in dB around a boundary; used to verify the score
/// is continuous across every cut (< 1.5 dB jumps per the checklist).
pub(crate) fn max_boundary_jump_db(
    left: &[f64],
    right: &[f64],
    boundaries: &[f64],
    sample_rate: f64,
) -> f64 {
    let window = (sample_rate * 0.25) as usize;
    let rms = |start: usize| -> f64 {
        let end = (start + window).min(left.len());
        if end <= start {
            return 0.0;
        }
        let sum: f64 = (start..end).map(|i| left[i] * left[i] + right[i] * right[i]).sum();
        (sum / (2 * (end - start)) as f64).sqrt().max(1e-9)
    };
    let mut worst = 0.0f64;
    for &boundary in boundaries {
        let at = (boundary * sample_rate) as usize;
        if at < window || at + window >= left.len() {
            continue;
        }
        let before = rms(at - window);
        let after = rms(at);
        worst = worst.max((20.0 * (after / before).log10()).abs());
    }
    worst
}

/// The act plan produced by the match-cut pass (pure data for the EDL).
struct ActPlan {
    label: &'static str,
    source: String,
    trim_start: f64,
    duration: f64,
    transition: Transition,
    numeral: Option<&'static str>,
}

impl VizMode for Broadcast {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("broadcast").expect("broadcast is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("broadcast skipped: trajectory too short");
            return Ok(());
        }

        // --- Source artifacts (the planner ran the acts first: lower ids).
        let viz_dir = format!("{}/viz", ctx.seed_dir);
        let sources = [
            ("mission-control/mission_control.mp4", "Act I (mission control)"),
            ("editorial-retime/retimed.mp4", "Act II (retime)"),
            ("corotating/reveal.mp4", "Act III (reveal)"),
            ("bullet-time/bullet_time.mp4", "Act IV (bullet time)"),
            ("epilogue/epilogue.mp4", "Act V (epilogue)"),
        ];
        let main_video = format!("{}/videos/web/main.mp4", ctx.seed_dir);
        let mut missing: Vec<&str> = Vec::new();
        for (relative, label) in &sources {
            if !std::path::Path::new(&format!("{viz_dir}/{relative}")).exists() {
                missing.push(label);
            }
        }
        if !std::path::Path::new(&main_video).exists() {
            missing.push("main video");
        }
        if !missing.is_empty() {
            warn!("broadcast skipped: missing sources: {}", missing.join(", "));
            return Ok(());
        }
        let path_of = |relative: &str| format!("{viz_dir}/{relative}");

        // --- Source durations from each mode's own schedule constants.
        let mission_dur = ctx.quality.scale_count(1800) as f64 / 60.0;
        let retimed_frames = ctx.quality.scale_count(900);
        let retimed_dur = retimed_frames as f64 / 30.0;
        let reveal_dur = ctx.quality.scale_count(1800) as f64 / 60.0;
        let bullet_frames = ctx.quality.scale_count(1440);
        let bullet_dur = bullet_frames as f64 / 60.0;
        let epilogue_frames = ctx.quality.scale_count(1350);
        let epilogue_dur = epilogue_frames as f64 / 30.0;
        let recap_dur = (epilogue_frames as f64 * super::epilogue::RECAP_FRACTION).max(1.0) / 30.0;
        // Draft sources are quartered; scale nominal takes to match.
        let take_scale = ctx.quality.scale_count(1000) as f64 / 1000.0;

        // --- Frame -> step maps (each act's own schedule, reproduced).
        let drama = ctx.events().drama(ctx.kinematics());
        let retime_schedule = super::editorial_retime::build_schedule(&drama, retimed_frames);
        let (bullet_open_step, bullet_close_step) =
            super::bullet_time::boundary_steps(ctx, bullet_frames);
        let linear = |duration: f64| {
            move |t: f64| -> usize { ((t / duration.max(1e-9)) * steps as f64) as usize }
        };
        let mission_step = linear(mission_dur);
        let main_step = linear(MAIN_SECONDS);
        let reveal_step = linear(reveal_dur);
        let retime_step = |t: f64| -> usize {
            let frame = ((t * 30.0) as usize).min(retime_schedule.len().saturating_sub(1));
            retime_schedule.get(frame).copied().unwrap_or(steps).min(steps - 1)
        };
        let recap_step = linear(recap_dur);

        // --- Act II opening: center the deepest drama in the main trim.
        let main_take = (25.0 * take_scale).min(MAIN_SECONDS);
        let peak_step = drama
            .iter()
            .enumerate()
            .step_by(500)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(step, _)| step);
        let peak_t = peak_step as f64 / steps as f64 * MAIN_SECONDS;
        let main_nominal = (peak_t - main_take * 0.5).clamp(0.0, MAIN_SECONDS - main_take);

        // --- Match cuts at every act boundary.
        let mission_take = (25.0 * take_scale).min(mission_dur);
        let cut1 = best_match_cut(
            ctx.positions,
            ((mission_take - 1.5).max(1.0), mission_take.min(mission_dur - 0.05)),
            &mission_step,
            ((main_nominal - 1.5).max(0.0), (main_nominal + 1.5).min(MAIN_SECONDS - main_take)),
            &main_step,
        );
        let cut2 = best_match_cut(
            ctx.positions,
            (
                cut1.in_seconds + main_take - 1.5,
                (cut1.in_seconds + main_take + 1.5).min(MAIN_SECONDS - 0.05),
            ),
            &main_step,
            (0.0, 0.0),
            &retime_step,
        );
        let cut3 = best_match_cut(
            ctx.positions,
            (retimed_dur, retimed_dur),
            &retime_step,
            (0.0, (4.0 * take_scale).min(reveal_dur * 0.3)),
            &reveal_step,
        );
        let cut4 = best_match_cut(
            ctx.positions,
            ((reveal_dur * 0.85).max(1.0), reveal_dur - 0.05),
            &reveal_step,
            (0.0, 0.0),
            &|_| bullet_open_step,
        );
        let cut5 = best_match_cut(
            ctx.positions,
            (0.0, 0.0),
            &|_| bullet_close_step,
            ((0.5 * take_scale).max(0.05), (recap_dur - 2.0).max(0.2)),
            &recap_step,
        );

        // --- Cards: five roman numerals + the seed title.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(64);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(64);
        let numeral_card = |numeral: &str, path: &str| -> Result<()> {
            let cw = video_w as usize;
            let ch = video_h as usize;
            let paper = Paper::DeepBlack;
            let mut card = vec![paper_color(paper); cw * ch];
            let ink = ink_color(paper);
            let style = TextStyle {
                align: Align::Center,
                opacity: 0.92,
                tracking: 0.24,
                ..TextStyle::new(Face::Sans, type_px(ch, 6), ink)
            };
            draw_text(&mut card, cw, ch, cw as f64 / 2.0, ch as f64 * 0.55, &style, numeral);
            let mut bytes = Vec::new();
            encode_linear_rec2020_to_u16(&card, &mut bytes);
            let image = crate::render::ImageBuffer::<crate::render::Rgb<u16>, Vec<u16>>::from_raw(
                cw as u32, ch as u32, bytes,
            )
            .expect("card sized");
            crate::render::save_image_as_png_16bit(&image, path)?;
            Ok(())
        };
        let mut temp_files: Vec<String> = Vec::new();
        let numerals = ["I", "II", "III", "IV", "V"];
        let mut card_paths = Vec::new();
        for numeral in numerals {
            let path = sink.path(&format!("card_{}.png", numeral.to_lowercase()));
            numeral_card(numeral, &path)?;
            card_paths.push(path.clone());
            temp_files.push(path);
        }
        let title_path = sink.path("card_title.png");
        {
            let cw = video_w as usize;
            let ch = video_h as usize;
            let paper = Paper::DeepBlack;
            let mut card = vec![paper_color(paper); cw * ch];
            let ink = ink_color(paper);
            let seed_style = TextStyle {
                align: Align::Center,
                tabular: true,
                ..TextStyle::new(Face::Sans, type_px(ch, 5), ink)
            };
            draw_text(
                &mut card,
                cw,
                ch,
                cw as f64 / 2.0,
                ch as f64 * 0.52,
                &seed_style,
                &format!("0x{}", ctx.seed_hex.to_uppercase()),
            );
            let sub_style = TextStyle {
                align: Align::Center,
                opacity: 0.75,
                ..TextStyle::caption(Face::Mono, type_px(ch, 0), ink)
            };
            draw_text(
                &mut card,
                cw,
                ch,
                cw as f64 / 2.0,
                ch as f64 * 0.62,
                &sub_style,
                "the five-act broadcast",
            );
            let mut bytes = Vec::new();
            encode_linear_rec2020_to_u16(&card, &mut bytes);
            let image = crate::render::ImageBuffer::<crate::render::Rgb<u16>, Vec<u16>>::from_raw(
                cw as u32, ch as u32, bytes,
            )
            .expect("card sized");
            crate::render::save_image_as_png_16bit(&image, &title_path)?;
            temp_files.push(title_path.clone());
        }

        // --- The act plan.
        let acts = [
            ActPlan {
                label: "act I: discovery",
                source: path_of(sources[0].0),
                trim_start: 0.0,
                duration: cut1.out_seconds,
                transition: Transition::Cut,
                numeral: Some("I"),
            },
            ActPlan {
                label: "act II: the dance (main)",
                source: main_video.clone(),
                trim_start: cut1.in_seconds,
                duration: cut2.out_seconds - cut1.in_seconds,
                transition: Transition::Cut,
                numeral: Some("II"),
            },
            ActPlan {
                label: "act II: the dance (retimed)",
                source: path_of(sources[1].0),
                trim_start: cut2.in_seconds,
                duration: retimed_dur - cut2.in_seconds,
                transition: Transition::Xfade(1.0),
                numeral: None,
            },
            ActPlan {
                label: "act III: revelation",
                source: path_of(sources[2].0),
                trim_start: cut3.in_seconds,
                duration: cut4.out_seconds - cut3.in_seconds,
                transition: Transition::Cut,
                numeral: Some("III"),
            },
            ActPlan {
                label: "act IV: the held breath",
                source: path_of(sources[3].0),
                trim_start: 0.0,
                duration: bullet_dur,
                transition: Transition::Cut,
                numeral: Some("IV"),
            },
            ActPlan {
                label: "act V: the end",
                source: path_of(sources[4].0),
                trim_start: cut5.in_seconds,
                duration: epilogue_dur - cut5.in_seconds,
                transition: Transition::Cut,
                numeral: Some("V"),
            },
        ];

        let mut segments: Vec<Segment> = Vec::new();
        for act in &acts {
            if let Some(numeral) = act.numeral {
                let index = numerals.iter().position(|&n| n == numeral).expect("numeral");
                segments.push(Segment {
                    source: card_paths[index].clone(),
                    is_card: true,
                    trim_start: 0.0,
                    duration: ACT_CARD_SECONDS,
                    label: format!("card {numeral}"),
                    transition_in: if segments.is_empty() {
                        Transition::Cut
                    } else {
                        Transition::Xfade(0.4)
                    },
                });
            }
            segments.push(Segment {
                source: act.source.clone(),
                is_card: false,
                trim_start: act.trim_start,
                duration: act.duration.max(0.5),
                label: act.label.to_string(),
                transition_in: act.transition,
            });
        }
        // Master still + seed title close the film; the still hold fills to
        // the 180 s target at final quality.
        let master_still = format!("{}/images/source/master.png", ctx.seed_dir);
        let title_seconds = (3.0 * take_scale).max(1.0);
        let so_far = timeline_seconds(&segments);
        let master_hold = if ctx.quality == crate::viz::context::VizQuality::Final {
            (TARGET_SECONDS - so_far - title_seconds + 0.8 + 0.6).clamp(2.0, 12.0)
        } else {
            2.0
        };
        if std::path::Path::new(&master_still).exists() {
            segments.push(Segment {
                source: master_still,
                is_card: true,
                trim_start: 0.0,
                duration: master_hold,
                label: "master hold".into(),
                transition_in: Transition::Xfade(0.8),
            });
        }
        segments.push(Segment {
            source: title_path.clone(),
            is_card: true,
            trim_start: 0.0,
            duration: title_seconds,
            label: "seed title".into(),
            transition_in: Transition::Xfade(0.6),
        });
        let total = timeline_seconds(&segments);
        info!(
            "   broadcast: {} segments, {total:.1}s timeline, match cuts {:.4}/{:.4}/{:.4}/{:.4}/{:.4}",
            segments.len(),
            cut1.distance,
            cut2.distance,
            cut3.distance,
            cut4.distance,
            cut5.distance
        );

        // --- Segment clocks for the score (film time of each boundary).
        let mut boundaries: Vec<f64> = Vec::new();
        let mut card_windows: Vec<(f64, f64)> = Vec::new();
        let mut act_windows: Vec<(f64, f64, &str)> = Vec::new();
        {
            let mut clock = 0.0;
            for (index, segment) in segments.iter().enumerate() {
                if index > 0 {
                    if let Transition::Xfade(fade) = segment.transition_in {
                        clock -= fade;
                    }
                    boundaries.push(clock);
                }
                if segment.is_card {
                    card_windows.push((clock, clock + segment.duration));
                } else {
                    act_windows.push((clock, clock + segment.duration, &segment.label));
                }
                clock += segment.duration;
            }
        }
        let act4_start = act_windows
            .iter()
            .find(|(_, _, label)| label.contains("act IV"))
            .map_or(total * 0.65, |&(start, _, _)| start);
        let act5_start = act_windows
            .iter()
            .find(|(_, _, label)| label.contains("act V"))
            .map_or(total * 0.75, |&(start, _, _)| start);
        let title_start = total - title_seconds;

        // --- One continuous score.
        let score = quantized_score(ctx);
        let sample_rate = f64::from(SAMPLE_RATE);
        let total_samples = (total * sample_rate) as usize;
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];

        // Strain sub-bass entering at Act IV (resampled over IV..end).
        let (h_plus, _) = strain_series(ctx.positions, &ctx.kinematics().masses);
        let strain_span = ((total - act4_start) * sample_rate) as usize;
        let mut strain = resample_linear(&h_plus, strain_span.max(1));
        {
            let peak = strain.iter().fold(1e-12f64, |acc, &v| acc.max(v.abs()));
            let mut low = Biquad::default();
            low.set_lowpass(110.0, 0.7, sample_rate);
            for value in &mut strain {
                *value = low.process(*value / peak);
            }
        }

        // Bed level per act, interpolated smoothly (no jumps at cuts).
        let bed_target = |t: f64| -> f64 {
            if t >= title_start {
                0.04
            } else if t >= act5_start {
                0.14
            } else if t >= act4_start {
                0.05
            } else {
                let progress = (t / act4_start.max(1e-9)).clamp(0.0, 1.0);
                0.20 + 0.06 * (progress * std::f64::consts::PI).sin()
            }
        };
        let root = score.root_hz;
        let triad: Vec<f64> = (0..3)
            .map(|voice| {
                score.voices[voice].first().map_or(root, |note| note_frequency(root, note))
            })
            .collect();
        let chord_adsr = Adsr { attack: 0.4, decay: 2.8, sustain: 0.35, release: 2.0 };

        let mut phase_a = 0.0f64;
        let mut phase_b = 0.0f64;
        let mut bed_filter = Biquad::default();
        let mut bed_level = 0.0f64;
        let smoothing = 1.0 - (-1.0 / (sample_rate * 0.8)).exp();
        for sample in 0..total_samples {
            let t = sample as f64 / sample_rate;
            // Bed: two detuned saws at the root and fifth, slow lowpass.
            let f_a = root * 0.5;
            let f_b = root * 0.749; // a hair under 3/2: slow beating
            phase_a = (phase_a + f_a / sample_rate) % 1.0;
            phase_b = (phase_b + f_b / sample_rate) % 1.0;
            if sample.is_multiple_of(64) {
                let brightness =
                    if t < act4_start { 320.0 + 700.0 * (t / act4_start) } else { 260.0 };
                bed_filter.set_lowpass(brightness, 0.8, sample_rate);
            }
            bed_level += (bed_target(t) - bed_level) * smoothing;
            let mut value = bed_filter.process(
                saw_bandlimited(phase_a, f_a / sample_rate) * 0.6
                    + saw_bandlimited(phase_b, f_b / sample_rate) * 0.4,
            ) * bed_level;

            // Bells land the match cuts musically.
            for &cut in &boundaries {
                if t >= cut && t < cut + 2.2 {
                    value += fm_bell(t - cut, root * 2.0, 1.6) * 0.16;
                }
            }
            // Silence beat before Act V.
            let beat = if t >= act5_start - 1.0 && t < act5_start - 0.15 { 0.06 } else { 1.0 };
            // Duck under cards.
            let duck = card_windows
                .iter()
                .map(|&(start, end)| if t >= start && t <= end { 0.4 } else { 1.0 })
                .fold(1.0f64, f64::min);
            let mut frame = (value * beat * duck, value * beat * duck);

            // Strain sub-bass from Act IV (fades in over 2 s).
            if t >= act4_start {
                let index = ((t - act4_start) * sample_rate) as usize;
                if index < strain.len() {
                    let fade_in = ((t - act4_start) / 2.0).clamp(0.0, 1.0);
                    let sub = strain[index] * 0.30 * fade_in * beat;
                    frame.0 += sub;
                    frame.1 += sub;
                }
            }
            // Final chord: the V16 triad on the title card.
            if t >= title_start {
                let held = t - title_start;
                let envelope = chord_adsr.amplitude(held, title_seconds * 0.7);
                for (voice, &frequency) in triad.iter().enumerate() {
                    let tone = (std::f64::consts::TAU * frequency * held).sin()
                        + 0.25 * (std::f64::consts::TAU * frequency * 2.0 * held).sin()
                        + 0.12 * (std::f64::consts::TAU * frequency * 3.0 * held).sin();
                    let (l, r) = crate::viz::common::audio::equal_power_pan(
                        tone * envelope * 0.09,
                        (voice as f64 - 1.0) * 0.5,
                    );
                    frame.0 += l;
                    frame.1 += r;
                }
            }
            left[sample] = frame.0;
            right[sample] = frame.1;
        }
        soft_limit(&mut left, &mut right, 1.2);
        normalize_to_lufs(&mut left, &mut right, -16.0, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.4) as usize);
        fade_ends(&mut right, (sample_rate * 0.4) as usize);
        let jump_db = max_boundary_jump_db(&left, &right, &boundaries, sample_rate);
        info!("   broadcast: worst boundary level jump {jump_db:.2} dB");
        let mix_path = sink.path("broadcast_mix.wav");
        write_wav_stereo_24bit(&mix_path, &left, &right)?;
        temp_files.push(mix_path.clone());

        // --- Assemble.
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("broadcast.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("broadcast_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        assemble(&segments, Some(&mix_path), video_w, video_h, FPS, &outputs)?;
        sink.record("broadcast.mp4", "video");
        sink.record("broadcast_hq.mp4", "video");
        for path in &temp_files {
            let _ = std::fs::remove_file(path);
        }

        // --- EDL sidecar.
        let cuts = [cut1, cut2, cut3, cut4, cut5];
        let edl: Vec<serde_json::Value> = segments
            .iter()
            .map(|segment| {
                serde_json::json!({
                    "label": segment.label,
                    "source": segment.source,
                    "trim_start": segment.trim_start,
                    "duration": segment.duration,
                    "transition": match segment.transition_in {
                        Transition::Cut => "cut".to_string(),
                        Transition::Xfade(fade) => format!("xfade {fade:.2}"),
                    },
                })
            })
            .collect();
        let meta = serde_json::json!({
            "seconds": total,
            "target_seconds": TARGET_SECONDS,
            "fps": FPS,
            "match_cuts": cuts
                .iter()
                .map(|cut| {
                    serde_json::json!({
                        "out_seconds": cut.out_seconds,
                        "in_seconds": cut.in_seconds,
                        "pose_distance": cut.distance,
                    })
                })
                .collect::<Vec<_>>(),
            "worst_boundary_jump_db": jump_db,
            "segments": edl,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("broadcast_edl.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic *deforming* trajectories: bodies orbit at 2/4/6
    /// revolutions per run, so the triangle's shape pattern repeats exactly
    /// once per half-run (relative phases realign at multiples of 2 pi).
    fn deforming_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let revolutions = 2.0 * (f64::from(body) + 1.0);
                let radius = 1.0 + 0.3 * f64::from(body);
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64 * std::f64::consts::TAU * revolutions;
                        Vector3::new(t.cos() * radius, t.sin() * radius, 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pose_distance_is_zero_for_identical_steps_and_grows_with_shear() {
        let positions = deforming_positions(3_000);
        assert!(pose_distance(&positions, 500, 500) < 1e-12);
        // Half a run later the relative phases realign: the shape repeats.
        assert!(pose_distance(&positions, 500, 2_000) < 1e-9);
        // A quarter run in between the shape differs.
        assert!(pose_distance(&positions, 500, 1_250) > 1e-4);
    }

    #[test]
    fn match_cut_search_finds_a_congruent_pose() {
        let positions = deforming_positions(3_000);
        // Outgoing shows step 500 (5 s at 100 steps/s). Inside the 10..25 s
        // window the triangle is congruent to it exactly twice: mirrored at
        // 10 s and identically at 20 s (sorted side lengths accept both --
        // a mirrored triangle reads as a pose match at a cut). Anywhere
        // else the shape differs; the search must land on one of the two.
        let step_of = |t: f64| -> usize { (t * 100.0) as usize };
        let cut = best_match_cut(&positions, (5.0, 5.0), &step_of, (10.0, 25.0), &step_of);
        assert!(
            (cut.in_seconds - 10.0).abs() <= 0.2 || (cut.in_seconds - 20.0).abs() <= 0.2,
            "expected a congruence at ~10 s or ~20 s, chose {} (distance {})",
            cut.in_seconds,
            cut.distance
        );
        assert!(cut.distance < 1e-3);
        // A window that excludes both congruences cannot do as well.
        let off = best_match_cut(&positions, (5.0, 5.0), &step_of, (11.0, 18.0), &step_of);
        assert!(off.distance > cut.distance, "off-window match must be worse");
    }

    #[test]
    fn boundary_jump_detector_flags_steps_and_passes_ramps() {
        let sample_rate = 1_000.0;
        let length = 4_000usize;
        // A hard 12 dB step at t = 2 s.
        let stepped: Vec<f64> = (0..length)
            .map(|i| if i < 2_000 { 0.1 } else { 0.4 } * ((i as f64 * 0.7).sin()))
            .collect();
        let jump = max_boundary_jump_db(&stepped, &stepped, &[2.0], sample_rate);
        assert!(jump > 6.0, "step must be flagged, got {jump:.2} dB");
        // A smooth ramp across the same boundary stays quiet.
        let ramped: Vec<f64> = (0..length)
            .map(|i| {
                let ramp = 0.1 + 0.3 * ((i as f64 - 1_500.0) / 1_000.0).clamp(0.0, 1.0);
                ramp * ((i as f64 * 0.7).sin())
            })
            .collect();
        let quiet = max_boundary_jump_db(&ramped, &ramped, &[2.0], sample_rate);
        assert!(quiet < 3.0, "ramp should pass, got {quiet:.2} dB");
    }
}
