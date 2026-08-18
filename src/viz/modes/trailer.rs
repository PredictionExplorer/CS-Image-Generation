//! V47 `trailer` -- Sixty Seconds, Auto-Edited.
//!
//! A deterministic 60 s trailer per seed: cold open on the three most
//! violent close approaches (macro push-ins from the frame archive), a
//! typeset title card, a spectral-sweep interlude, two drama-cut montage
//! trims of the main video, and the master still to close. Cuts quantize
//! to the sonification's bell onsets when its sidecar exists; the score
//! bed is re-synthesized to the EDL with the same voice primitives (V10
//! stems are not persisted). Assembled by the compositor.

use crate::error::Result;
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio::{
    Biquad, SAMPLE_RATE, fade_ends, fm_bell, normalize_to_lufs, saw_bandlimited, soft_limit,
    write_wav_stereo_24bit,
};
use crate::viz::common::compositor::{Segment, Transition, assemble, timeline_seconds};
use crate::viz::common::display::encode_linear_rec2020_to_u16;
use crate::viz::common::style::{Paper, ink_color, paper_color, type_px};
use crate::viz::common::text::{Align, Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Trailer length in seconds.
const TRAILER_SECONDS: f64 = 60.0;
/// Macro push-in zoom.
const MACRO_ZOOM: f64 = 2.4;
/// Trailer frame rate.
const FPS: u32 = 30;

/// The trailer mode.
pub struct Trailer;

impl VizMode for Trailer {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("trailer").expect("trailer is in the catalog")
    }

    fn needs_frame_archive(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(archive) = &ctx.frame_archive else {
            warn!("trailer skipped: no frame archive (image-only run)");
            return Ok(());
        };
        let main_video = format!("{}/videos/web/main.mp4", ctx.seed_dir);
        let sweep_video = format!("{}/videos/web/spectral_sweep.mp4", ctx.seed_dir);
        if !std::path::Path::new(&main_video).exists() {
            warn!("trailer skipped: main video missing");
            return Ok(());
        }
        let mut rng = ctx.fork_rng("trailer");
        let steps = ctx.step_count();
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(64);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(64);

        // --- Shot 1-3: macro push-ins from the archive event windows.
        // The archive stores the event windows first (three runs of ~37
        // frames), then the 24 drama samples; group by contiguity.
        let mut windows: Vec<Vec<usize>> = Vec::new();
        for (position, &(frame, _)) in archive.frames.iter().enumerate() {
            match windows.last_mut() {
                Some(window) if frame <= archive.frames[position.saturating_sub(1)].0 + 4 => {
                    window.push(position);
                }
                _ => windows.push(vec![position]),
            }
        }
        windows.retain(|window| window.len() >= 6);
        windows.truncate(3);
        if windows.is_empty() {
            // No contiguous event windows (short or placid runs): build the
            // cold open from thirds of whatever the archive holds.
            let count = archive.frames.len();
            for shot in 0..3.min(count) {
                let start = shot * count / 3;
                let end = ((shot + 1) * count / 3).max(start + 1);
                windows.push((start..end).collect());
            }
            warn!("trailer: no event windows; cold open falls back to archive thirds");
        }
        // Seed-varying shot order inside the cold-open slot.
        if windows.len() > 1 && rng.next_f64() > 0.5 {
            windows.swap(0, 1);
        }
        if windows.len() > 2 && rng.next_f64() > 0.5 {
            windows.swap(1, 2);
        }

        let aw = archive.width as usize;
        let ah = archive.height as usize;
        let mut push_in_paths: Vec<(String, f64)> = Vec::new();
        for (shot, window) in windows.iter().enumerate() {
            let file = sink.path(&format!("shot_pushin_{shot}.mp4"));
            let shot_frames = window.len() * 2;
            let outputs = [VideoOutputSpec {
                output_file: file.clone(),
                options: VideoEncodingOptions::web_compatible(),
            }];
            create_videos_from_frames_singlepass(
                video_w,
                video_h,
                FPS,
                |out| {
                    let mut frame_rgb = vec![(0.0, 0.0, 0.0); video_w as usize * video_h as usize];
                    let mut bytes: Vec<u16> = Vec::new();
                    for output_frame in 0..shot_frames {
                        let archive_pos = window[output_frame / 2];
                        let rgb8 = &archive.frames[archive_pos].1;
                        let t = output_frame as f64 / (shot_frames - 1).max(1) as f64;
                        let zoom = 1.4 + (MACRO_ZOOM - 1.4) * t;
                        for y in 0..video_h as usize {
                            for x in 0..video_w as usize {
                                // Zoom about the archive frame center.
                                let sx = (x as f64 / f64::from(video_w) - 0.5) / zoom + 0.5;
                                let sy = (y as f64 / f64::from(video_h) - 0.5) / zoom + 0.5;
                                let ax = ((sx * aw as f64) as usize).min(aw - 1);
                                let ay = ((sy * ah as f64) as usize).min(ah - 1);
                                let base = (ay * aw + ax) * 3;
                                let decode = |v: u8| (f64::from(v) / 255.0).powf(2.2);
                                frame_rgb[y * video_w as usize + x] = (
                                    decode(rgb8[base]),
                                    decode(rgb8[base + 1]),
                                    decode(rgb8[base + 2]),
                                );
                            }
                        }
                        encode_linear_rec2020_to_u16(&frame_rgb, &mut bytes);
                        out.write_all(bytemuck::cast_slice(&bytes))
                            .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    }
                    Ok(())
                },
                &outputs,
            )?;
            push_in_paths.push((file, shot_frames as f64 / f64::from(FPS)));
        }

        // --- Title card.
        let card_path = sink.path("title_card.png");
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
                "a three-body light sculpture",
            );
            let mut bytes = Vec::new();
            encode_linear_rec2020_to_u16(&card, &mut bytes);
            let image = crate::render::ImageBuffer::<crate::render::Rgb<u16>, Vec<u16>>::from_raw(
                cw as u32, ch as u32, bytes,
            )
            .expect("card sized");
            crate::render::save_image_as_png_16bit(&image, &card_path)?;
        }

        // --- Drama montage trims of the main video.
        let drama = ctx.events().drama(ctx.kinematics());
        let main_seconds = 30.0;
        let mut peaks: Vec<(f64, usize)> =
            drama.iter().enumerate().step_by(500).map(|(step, &d)| (d, step)).collect();
        peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut montage_starts: Vec<f64> = Vec::new();
        for &(_, step) in &peaks {
            let start = (step as f64 / steps as f64 * main_seconds - 3.0).clamp(0.0, 24.0);
            if montage_starts.iter().all(|&s| (s - start).abs() > 7.0) {
                montage_starts.push(start);
            }
            if montage_starts.len() == 2 {
                break;
            }
        }
        while montage_starts.len() < 2 {
            montage_starts.push(6.0 * montage_starts.len() as f64);
        }

        // --- Bell onsets from the sonification sidecar (cut quantization).
        let bells: Vec<f64> =
            std::fs::read_to_string(format!("{}/viz/sonification/score.json", ctx.seed_dir))
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|json| {
                    json.get("bells").and_then(|list| {
                        list.as_array().map(|bells| {
                            bells
                                .iter()
                                .filter_map(|bell| {
                                    bell.get("time_s").and_then(serde_json::Value::as_f64)
                                })
                                .collect()
                        })
                    })
                })
                .unwrap_or_default();
        let quantize = |seconds: f64| -> f64 {
            let tolerance = 4.0 / f64::from(FPS);
            // Bells live on the 30 s main-video clock; map into trailer time
            // 1:1 (the montage plays at native speed).
            bells
                .iter()
                .copied()
                .find(|bell| (bell - seconds).abs() <= tolerance)
                .unwrap_or(seconds)
        };

        // --- EDL: tension-curve shot lengths (long -> short -> long).
        let master_still = format!("{}/images/source/master.png", ctx.seed_dir);
        let mut segments: Vec<Segment> = Vec::new();
        let push_lengths: [f64; 3] = [3.2, 2.2, 4.2];
        for (index, (path, available)) in push_in_paths.iter().enumerate() {
            segments.push(Segment {
                source: path.clone(),
                is_card: false,
                trim_start: 0.0,
                duration: push_lengths[index % 3].min(*available),
                label: format!("cold-open push-in {index}"),
                transition_in: Transition::Cut,
            });
        }
        segments.push(Segment {
            source: card_path.clone(),
            is_card: true,
            trim_start: 0.0,
            duration: 4.0,
            label: "title card".into(),
            transition_in: Transition::Xfade(0.6),
        });
        if std::path::Path::new(&sweep_video).exists() {
            segments.push(Segment {
                source: sweep_video.clone(),
                is_card: false,
                trim_start: 1.0,
                duration: 7.0,
                label: "sweep interlude".into(),
                transition_in: Transition::Xfade(0.6),
            });
        }
        for (index, &start) in montage_starts.iter().enumerate() {
            segments.push(Segment {
                source: main_video.clone(),
                is_card: false,
                trim_start: quantize(start),
                duration: if index == 0 { 12.0 } else { 9.0 },
                label: format!("drama montage {index}"),
                transition_in: Transition::Cut,
            });
        }
        // Close on the master still, filling to exactly 60 s.
        let so_far = timeline_seconds(&segments) + 0.8;
        let hold = (TRAILER_SECONDS - so_far).max(3.0);
        if std::path::Path::new(&master_still).exists() {
            segments.push(Segment {
                source: master_still,
                is_card: true,
                trim_start: 0.0,
                duration: hold,
                label: "master hold".into(),
                transition_in: Transition::Xfade(0.8),
            });
        }
        let total = timeline_seconds(&segments);
        info!("   trailer: {} segments, {total:.2}s timeline", segments.len());

        // --- Score bed re-synthesized to the EDL.
        let sample_rate = f64::from(SAMPLE_RATE);
        let total_samples = (total * sample_rate) as usize;
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];
        let mut phase = 0.0f64;
        let mut filter = Biquad::default();
        // Cut times for bell hits and card ducking windows.
        let mut boundaries: Vec<f64> = Vec::new();
        let mut cards: Vec<(f64, f64)> = Vec::new();
        {
            let mut clock = 0.0;
            for (index, segment) in segments.iter().enumerate() {
                if index > 0 {
                    boundaries.push(clock);
                }
                if segment.is_card {
                    cards.push((clock, clock + segment.duration));
                }
                clock += segment.duration;
                if let Transition::Xfade(fade) = segment.transition_in {
                    clock -= fade;
                }
            }
        }
        for sample in 0..total_samples {
            let t = sample as f64 / sample_rate;
            let progress = t / total;
            let frequency = 55.0 * 2.0_f64.powf(0.6 + 0.5 * (progress * 3.1).sin());
            phase = (phase + frequency / sample_rate) % 1.0;
            if sample.is_multiple_of(64) {
                filter.set_lowpass(240.0 + 900.0 * progress, 0.9, sample_rate);
            }
            let mut value = filter.process(saw_bandlimited(phase, frequency / sample_rate)) * 0.28;
            for &cut in &boundaries {
                if t >= cut && t < cut + 2.5 {
                    value += fm_bell(t - cut, 440.0, 1.8) * 0.22;
                }
            }
            // Side-chain duck under cards.
            let duck = cards
                .iter()
                .map(|&(start, end)| if t >= start && t <= end { 0.35 } else { 1.0 })
                .fold(1.0f64, f64::min);
            left[sample] = value * duck;
            right[sample] = value * duck;
        }
        soft_limit(&mut left, &mut right, 1.2);
        normalize_to_lufs(&mut left, &mut right, -16.0, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.4) as usize);
        fade_ends(&mut right, (sample_rate * 0.4) as usize);
        let mix_path = sink.path("trailer_mix.wav");
        write_wav_stereo_24bit(&mix_path, &left, &right)?;

        // --- Assemble.
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("trailer.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("trailer_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        assemble(&segments, Some(&mix_path), video_w, video_h, FPS, &outputs)?;
        sink.record("trailer.mp4", "video");
        sink.record("trailer_hq.mp4", "video");
        let _ = std::fs::remove_file(&mix_path);
        for (path, _) in &push_in_paths {
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_file(&card_path);

        // --- EDL sidecar (reproducible by construction).
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
            "fps": FPS,
            "macro_zoom": MACRO_ZOOM,
            "bell_quantized": !bells.is_empty(),
            "segments": edl,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("edl.json", &json, "data")?;
        Ok(())
    }
}
